# app/services/rag_service.py

import logging
import uuid
from typing import List, Tuple, Dict
from pathlib import Path
import time

import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
from google.cloud import documentai, storage, aiplatform, firestore

from app.core.config import settings
from app.core.prompts import RAG_PROMPT_TEMPLATE

# --- Initialize logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Text Processing Configuration ---
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 100


# --- Client Initializations ---
vertexai.init(project=settings.GCP_PROJECT_ID, location=settings.VERTEX_AI_REGION)
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
llm_model = GenerativeModel("gemini-2.0-flash-lite-001") # Using gemini-pro as flash-lite is not a valid model

firestore_client = firestore.Client(project=settings.GCP_PROJECT_ID)
GROUP_INDEX_COLLECTION = "sara_group_indexes"
CHUNK_COLLECTION = "sara_document_chunks"


# ==============================================================================
# --- NEW: Index and Endpoint Management ---
# ==============================================================================

def get_or_create_group_resources(group_id: str) -> Dict[str, str]:
    """
    Checks if a vector index and endpoint exist for a group. If not, creates them.
    This is the core of the "Index-per-Group" architecture.

    Args:
        group_id (str): The unique ID for the WhatsApp group.

    Returns:
        Dict[str, str]: A dictionary containing the 'index_id' and 'endpoint_id'.
    """
    # Sanitize group_id to be a valid Firestore document ID and for resource names
    safe_group_id = group_id.replace(":", "_").replace("+", "")
    group_doc_ref = firestore_client.collection(GROUP_INDEX_COLLECTION).document(safe_group_id)
    
    group_doc = group_doc_ref.get()
    if group_doc.exists:
        logger.info(f"Found existing resources for group {group_id}.")
        return group_doc.to_dict()

    logger.warning(f"No resources found for group {group_id}. Creating new index and endpoint...")
    
    # --- Create a new Vector Search Index ---
    # Use underscores instead of hyphens for compatibility with deployed_index_id
    unique_suffix = uuid.uuid4().hex[:6]
    index_display_name = f"sara_index_{safe_group_id}_{unique_suffix}"
    try:
        index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
            display_name=index_display_name,
            dimensions=768,
            approximate_neighbors_count=10,
            distance_measure_type="DOT_PRODUCT_DISTANCE",
            index_update_method="STREAM_UPDATE" # Allows for near real-time updates
        )
        logger.info(f"Successfully created new index: {index.resource_name}")
        
        # --- Create a new Index Endpoint ---
        endpoint_display_name = f"sara_endpoint_{safe_group_id}_{unique_suffix}"
        endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name=endpoint_display_name,
            public_endpoint_enabled=True # Make it accessible to our Cloud Run service
        )
        logger.info(f"Successfully created new endpoint: {endpoint.resource_name}")

        # --- Deploy the Index to the Endpoint ---
        # THIS IS THE FIX: The ID must not contain hyphens.
        deployed_index_id = f"deployed_{index_display_name}"
        logger.info(f"Deploying index to endpoint with ID: {deployed_index_id}... This may take up to 20 minutes.")
        endpoint.deploy_index(
            index=index,
            deployed_index_id=deployed_index_id
        )
        logger.info(f"Successfully deployed index to endpoint.")

        # --- Store the new resource IDs in Firestore for future use ---
        new_resources = {
            "index_id": index.resource_name,
            "endpoint_id": endpoint.resource_name,
            "deployed_index_id": deployed_index_id,
            "group_id": group_id
        }
        group_doc_ref.set(new_resources)
        logger.info(f"Saved new resource mapping to Firestore for group {group_id}.")

        return new_resources
        
    except Exception as e:
        logger.error(f"Failed to create and deploy resources for group {group_id}: {e}", exc_info=True)
        return None


# ==============================================================================
# --- Core RAG Logic (Updated for New Architecture) ---
# ==============================================================================

def ingest_document(file_path: Path, mime_type: str, group_id: str, doc_id: str):
    logger.info(f"--- Starting ingestion for doc_id: {doc_id} in group: {group_id} ---")
    
    # 1. Get the dedicated resources for this group (create if they don't exist)
    resources = get_or_create_group_resources(group_id)
    if not resources:
        logger.error(f"Could not retrieve or create resources for group {group_id}. Aborting ingestion.")
        return

    # 2. Process the document to get text
    full_text = process_document(file_path, mime_type)
    if not full_text:
        logger.error("Ingestion failed: Could not extract text."); return

    # 3. Chunk, embed, and prepare data
    text_chunks = chunk_text(full_text)
    # The chunk_id no longer needs the group_id prefix, as the index is dedicated.
    chunk_ids = [f"{doc_id}::{i}" for i, _ in enumerate(text_chunks)]
    embeddings = get_embeddings(text_chunks)
    if not embeddings:
        logger.error("Ingestion failed: Could not generate embeddings."); return
        
    # Prepare data for upserting into the specific index
    datapoints = [{"datapoint_id": id, "feature_vector": vec} for id, vec in zip(chunk_ids, embeddings)]
    
    # Also prepare the text chunks for storage in Firestore
    firestore_data = [{"id": id, "text": chunk} for id, chunk in zip(chunk_ids, text_chunks)]

    try:
        # 4. Upsert vectors to the group's dedicated index
        index = aiplatform.MatchingEngineIndex(index_name=resources["index_id"])
        index.upsert_datapoints(datapoints=datapoints)
        logger.info(f"Successfully upserted {len(datapoints)} vectors to index {resources['index_id']}.")

        # 5. Store the text chunks in Firestore for later retrieval
        firestore_batch = firestore_client.batch()
        for data in firestore_data:
            doc_ref = firestore_client.collection(CHUNK_COLLECTION).document(data["id"])
            firestore_batch.set(doc_ref, {"text": data["text"]})
        firestore_batch.commit()
        logger.info(f"Successfully stored {len(firestore_data)} text chunks in Firestore.")

        logger.info(f"--- ✅ ✅ ✅ Ingestion pipeline for doc_id: {doc_id} COMPLETED ---")

    except Exception as e:
        logger.error(f"--- ❌ ❌ ❌ Ingestion pipeline for doc_id: {doc_id} FAILED during upsert. ---", exc_info=True)


def answer_question(question: str, group_id: str) -> str:
    logger.info(f"--- Answering question for group: {group_id} ---")
    
    # 1. Get the dedicated resources for this group
    resources = get_or_create_group_resources(group_id)
    if not resources:
        # This can happen if the group has never uploaded a document.
        return "Disculpa, necesito que compartas al menos un documento en este grupo antes de que pueda responder preguntas."

    # 2. Create an embedding for the user's question
    logger.info(f"Original question: '{question}'")
    question_embedding = get_embeddings([question])[0]
    if not question_embedding:
        return "Disculpa, no pude procesar la pregunta."
        
    try:
        # 3. Query the group's dedicated endpoint
        endpoint = aiplatform.MatchingEngineIndexEndpoint(resources["endpoint_id"])
        neighbor_results = endpoint.find_neighbors(
            queries=[question_embedding],
            deployed_index_id=resources["deployed_index_id"],
            num_neighbors=3
        )
        logger.info("Successfully found neighbors in Vector Search.")
        
        # 4. Retrieve the text chunks from Firestore using the neighbor IDs
        neighbor_ids = []
        if neighbor_results and neighbor_results[0]:
            for neighbor in neighbor_results[0]:
                neighbor_ids.append(neighbor.id)
        
        if not neighbor_ids:
            logger.warning("Vector search returned no neighbors.");
            return "Disculpa, la información que tengo no es suficiente para responder tu pregunta."

        doc_refs = [firestore_client.collection(CHUNK_COLLECTION).document(id) for id in neighbor_ids]
        docs = firestore_client.get_all(doc_refs)
        
        context_chunks = [doc.to_dict().get("text", "") for doc in docs if doc.exists]
        context = "\n---\n".join(context_chunks)
        logger.info(f"Retrieved {len(context_chunks)} text chunks from Firestore.")
        
        # 5. Ask the LLM to generate an answer
        final_answer = ask_llm(question, context)
        
        logger.info(f"Final answer: '{final_answer}'")
        return final_answer
        
    except Exception as e:
        logger.error(f"Error during query or generation for group {group_id}: {e}", exc_info=True)
        return "Disculpa, ocurrió un error al buscar la respuesta."


# ==============================================================================
# --- Helper Functions (Unchanged) ---
# ==============================================================================

def process_document(file_path: Path, mime_type: str) -> str:
    # This function is correct and unchanged.
    try:
        logger.info(f"Processing document '{file_path.name}' with Document AI...")
        docai_client = documentai.DocumentProcessorServiceClient(client_options={"api_endpoint": f"{settings.DOCUMENT_AI_LOCATION}-documentai.googleapis.com"})
        processor_name = docai_client.processor_path(settings.GCP_PROJECT_ID, settings.DOCUMENT_AI_LOCATION, settings.DOCUMENT_AI_PROCESSOR_ID)
        with open(file_path, "rb") as f:
            image_content = f.read()
        raw_document = documentai.RawDocument(content=image_content, mime_type=mime_type)
        request = documentai.ProcessRequest(name=processor_name, raw_document=raw_document)
        result = docai_client.process_document(request=request)
        logger.info("Successfully extracted text from document.")
        return result.document.text
    except Exception as e:
        logger.error(f"Error processing document with Document AI: {e}", exc_info=True)
        return ""

def chunk_text(text: str) -> List[str]:
    # This function is correct and unchanged.
    logger.info(f"Chunking text of length {len(text)}...")
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    logger.info(f"Created {len(chunks)} chunks.")
    return chunks

def get_embeddings(texts: List[str]) -> List[List[float]]:
    # This function is correct and unchanged.
    logger.info(f"Generating embeddings for {len(texts)} chunks...")
    try:
        embeddings = embedding_model.get_embeddings(texts)
        vectors = [embedding.values for embedding in embeddings]
        logger.info("Successfully generated embeddings.")
        return vectors
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}", exc_info=True)
        return []

def ask_llm(question: str, context: str) -> str:
    # This function is correct and unchanged.
    logger.info("Sending prompt to LLM...")
    try:
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
        response = llm_model.generate_content(prompt)
        logger.info("Successfully received answer from LLM.")
        return response.text
    except Exception as e:
        logger.error(f"Error generating content with LLM: {e}", exc_info=True)
        return "Disculpa, ocurrió un error al generar la respuesta."
