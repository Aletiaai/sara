# app/services/rag_service.py

import logging
import uuid
import re
from typing import List, Tuple, Dict
from pathlib import Path
import time
from datetime import datetime

import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
from google.cloud import documentai, storage, aiplatform, firestore

from app.core.config import settings
from app.core.prompts import RAG_PROMPT_TEMPLATE

# --- Initialize logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Client Initializations ---
vertexai.init(project=settings.GCP_PROJECT_ID, location=settings.VERTEX_AI_REGION)
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
llm_model = GenerativeModel("gemini-2.0-flash-lite-001") # Using gemini-pro as flash-lite is not a valid model

firestore_client = firestore.Client(project=settings.GCP_PROJECT_ID)
GROUP_INDEX_COLLECTION = "sara_group_indexes"
CHUNK_COLLECTION = "sara_document_chunks"

# ==============================================================================
# --- NEW: Metadata Extraction and Smart Chunking ---
# ==============================================================================

def extract_metadata_and_chunk(full_text: str, file_name: str) -> Tuple[Dict, List[str]]:
    """
    Analyzes the full text of a document to extract key metadata and splits the content into logical paragraph-based chunks.
    Args:
        full_text (str): The complete text content from Document AI.
        file_name (str): The original name of the uploaded file.
    Returns:
        A tuple containing:
        - A dictionary of extracted metadata.
        - A list of text chunks (paragraphs).
    """
    metadata = {"original_file_name": file_name}
    lines = full_text.split('\n')
    
    # Simple classification logic: check for keywords. We can make this much more sophisticated later.
    if "secretario de acuerdos" in full_text.lower():
        metadata["document_type"] = "Judge Response"
        # Extract metadata for Judge's document. Case Number (top right)
        match = re.search(r"EXPEDIENTE NÚMERO: (\S+)", full_text, re.IGNORECASE)
        if match: metadata["case_number"] = match.group(1)
        # Date (usually near the top)
        match = re.search(r"(\w+,\s*\d+\s+de\s+\w+\s+de\s+\d{4})", full_text)
        if match: metadata["document_date"] = match.group(1)
        # Judge Name (near the signature)
        match = re.search(r"JUEZ\s+([A-Z\sÁÉÍÓÚÑ]+)", full_text, re.IGNORECASE)
        if match: metadata["judge_name"] = match.group(1).strip()

    else:
        metadata["document_type"] = "Lawyer Submission"
        # Extract metadata for Lawyer's document
        if lines:
            metadata["case_number"] = lines[0] # Assuming first line is case number
            metadata["judge_info"] = lines[1] # Assuming second line is judge info
        # Client Name (near the end)
        # This is a simple heuristic; can be improved
        last_lines = lines[-5:]
        for line in reversed(last_lines):
            if "C." in line or "c." in line:
                metadata["client_name"] = line.strip()
                break
    
    # --- Smart Chunking ---
    # Split the document into paragraphs based on double newlines.
    # We also filter out very short "chunks" that are likely just whitespace or headings.
    paragraphs = full_text.split('\n\n')
    chunks = [p.strip() for p in paragraphs if len(p.strip()) > 100] # Only keep substantial paragraphs
    
    logger.info(f"Identified document as '{metadata.get('document_type', 'Unknown')}'. Extracted metadata: {metadata}")
    logger.info(f"Split document into {len(chunks)} paragraph-based chunks.")
    
    return metadata, chunks


# ==============================================================================
# --- Index and Endpoint Management ---
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
    logger.info(f"--- Starting ADVANCED ingestion for doc_id: {doc_id} in group: {group_id} ---")
    
    resources = get_or_create_group_resources(group_id)
    if not resources:
        logger.error("Could not retrieve or create resources. Aborting."); return

    full_text = process_document(file_path, mime_type)
    if not full_text:
        logger.error("Ingestion failed: Could not extract text."); return

    # 1. Extract metadata and smart chunks
    metadata, text_chunks = extract_metadata_and_chunk(full_text, file_path.name)
    
    if not text_chunks:
        logger.warning("No substantial text chunks found after smart chunking. Aborting ingestion.")
        return

    # 2. Prepare data for storage
    chunk_ids = [f"{doc_id}::{i}" for i, _ in enumerate(text_chunks)]
    embeddings = get_embeddings(text_chunks)
    if not embeddings:
        logger.error("Ingestion failed: Could not generate embeddings."); return
        
    vector_datapoints = [{"datapoint_id": id, "feature_vector": vec} for id, vec in zip(chunk_ids, embeddings)]
    
    try:
        # 3. Upsert vectors to the group's dedicated index
        index = aiplatform.MatchingEngineIndex(index_name=resources["index_id"])
        index.upsert_datapoints(datapoints=vector_datapoints)
        logger.info(f"Successfully upserted {len(vector_datapoints)} vectors.")

        # 4. Store the text chunks AND their metadata in Firestore
        firestore_batch = firestore_client.batch()
        for i, chunk_text in enumerate(text_chunks):
            chunk_id = chunk_ids[i]
            doc_ref = firestore_client.collection(CHUNK_COLLECTION).document(chunk_id)
            # Create a rich document in Firestore
            firestore_doc = {
                "text": chunk_text,
                "document_id": doc_id,
                "group_id": group_id,
                "chunk_number": i + 1,
                **metadata  # Add all extracted metadata to each chunk document
            }
            firestore_batch.set(doc_ref, firestore_doc)
        firestore_batch.commit()
        logger.info(f"Successfully stored {len(text_chunks)} rich chunks in Firestore.")

        logger.info(f"--- ✅ ✅ ✅ Advanced ingestion for doc_id: {doc_id} COMPLETED ---")

    except Exception as e:
        logger.error(f"--- ❌ ❌ ❌ Ingestion pipeline FAILED during upsert. ---", exc_info=True)

def answer_question(question: str, group_id: str) -> str:
    logger.info(f"--- Answering question for group: {group_id} ---")
    
    resources = get_or_create_group_resources(group_id)
    if not resources:
        return "Disculpa, necesito que compartas al menos un documento en este grupo antes de que pueda responder preguntas."

    logger.info(f"Original question: '{question}'")
    question_embedding = get_embeddings([question])[0]
    if not question_embedding:
        return "Disculpa, no pude procesar la pregunta."
        
    try:
        endpoint = aiplatform.MatchingEngineIndexEndpoint(resources["endpoint_id"])
        
        neighbor_results = endpoint.find_neighbors(
            queries=[question_embedding],
            deployed_index_id=resources["deployed_index_id"],
            num_neighbors=5 # Let's retrieve a few more chunks to have more context
        )
        logger.info("Successfully found neighbors in Vector Search.")
        
        neighbor_ids = [neighbor.id for r in neighbor_results for neighbor in r]
        
        if not neighbor_ids:
            logger.warning("Vector search returned no neighbors.");
            return "Disculpa, la información que tengo no es suficiente para responder tu pregunta."

        # Retrieve the rich documents from Firestore
        doc_refs = [firestore_client.collection(CHUNK_COLLECTION).document(id) for id in neighbor_ids]
        docs = firestore_client.get_all(doc_refs)
        
        # Build a richer context string that includes metadata
        context_parts = []
        for doc in docs:
            if doc.exists:
                doc_data = doc.to_dict()
                context_str = (
                    f"Fragmento del documento '{doc_data.get('original_file_name', 'N/A')}' "
                    f"(Expediente: {doc_data.get('case_number', 'N/A')}, "
                    f"Fecha: {doc_data.get('document_date', 'N/A')}):\n"
                    f"'{doc_data.get('text', '')}'"
                )
                context_parts.append(context_str)

        context = "\n\n---\n\n".join(context_parts)
        
        logger.info(f"Retrieved {len(context_parts)} rich chunks. Passing the following CONTEXT to the LLM:")
        logger.info(f"\n--- BEGIN CONTEXT ---\n{context}\n--- END CONTEXT ---")
        
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
