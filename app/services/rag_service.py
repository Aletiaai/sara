# app/services/rag_service.py

import logging
import uuid
from typing import List, Tuple
from pathlib import Path

import vertexai
from vertexai.language_models import TextEmbeddingModel
from google.api_core.client_options import ClientOptions
from google.cloud import documentai, storage, aiplatform

from app.core.config import settings

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Text Processing Configuration ---
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 100


# --- Vertex AI Initialization ---
vertexai.init(
    project=settings.GCP_PROJECT_ID,
    location=settings.VERTEX_AI_REGION,
)

embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")

# --- CORRECTED CLIENT INITIALIZATION FOR UPSERT ---
# We need the IndexServiceClient to manage data on the index itself.
client_options = ClientOptions(
    api_endpoint=f"{settings.VERTEX_AI_REGION}-aiplatform.googleapis.com"
)
vector_search_index_client = aiplatform.gapic.IndexServiceClient(client_options=client_options)


# --- Path Definitions ---
# This is the full resource name for the Index itself, required for upserting.
vector_search_index_path = vector_search_index_client.index_path(
    project=settings.GCP_PROJECT_ID,
    location=settings.VERTEX_AI_REGION,
    index=settings.VECTOR_SEARCH_INDEX_ID,
)


def process_document(file_path: Path, mime_type: str) -> str:
    # This function is correct and unchanged.
    try:
        logger.info(f"Processing document '{file_path.name}' with Document AI...")
        docai_client = documentai.DocumentProcessorServiceClient(
            client_options=ClientOptions(
                api_endpoint=f"{settings.DOCUMENT_AI_LOCATION}-documentai.googleapis.com"
            )
        )
        processor_name = docai_client.processor_path(
            settings.GCP_PROJECT_ID,
            settings.DOCUMENT_AI_LOCATION,
            settings.DOCUMENT_AI_PROCESSOR_ID,
        )
        with open(file_path, "rb") as f:
            image_content = f.read()
        raw_document = documentai.RawDocument(
            content=image_content, mime_type=mime_type
        )
        request = documentai.ProcessRequest(
            name=processor_name, raw_document=raw_document
        )
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


def upsert_to_vector_search(datapoints: List[Tuple[str, List[float]]]) -> bool:
    """
    Upserts (inserts or updates) data points into our Vector Search index.
    This function now returns True on success and False on failure.
    """
    logger.info(f"Upserting {len(datapoints)} vectors to index...")
    try:
        index_datapoints = [
            aiplatform.gapic.IndexDatapoint(datapoint_id=dp_id, feature_vector=vector)
            for dp_id, vector in datapoints
        ]

        # Construct the request using the correct client and the INDEX path.
        upsert_request = aiplatform.gapic.UpsertDatapointsRequest(
            index=vector_search_index_path, # <-- CORRECTED FIELD NAME
            datapoints=index_datapoints,
        )
        
        vector_search_index_client.upsert_datapoints(request=upsert_request)
        
        logger.info(f"Successfully upserted {len(datapoints)} datapoints to Vector Search.")
        return True
    except Exception as e:
        logger.error(f"Error upserting datapoints to Vector Search: {e}", exc_info=True)
        return False


def ingest_document(file_path: Path, mime_type: str, group_id: str, doc_id: str):
    # This is the main orchestration function.
    logger.info(f"--- Starting ingestion for doc_id: {doc_id} in group: {group_id} ---")
    
    full_text = process_document(file_path, mime_type)
    if not full_text:
        logger.error("Ingestion failed: Could not extract text.")
        return

    text_chunks = chunk_text(full_text)
    chunk_ids = [f"{group_id}::{doc_id}::{i}" for i, _ in enumerate(text_chunks)]
    embeddings = get_embeddings(text_chunks)
    if not embeddings:
        logger.error("Ingestion failed: Could not generate embeddings.")
        return
        
    datapoints_to_upsert = list(zip(chunk_ids, embeddings))
    
    # --- CORRECTED LOGGING LOGIC ---
    # We now check if the upsert was successful before logging a success message.
    success = upsert_to_vector_search(datapoints_to_upsert)
    
    if success:
        logger.info(f"--- ✅ ✅ ✅ Ingestion pipeline for doc_id: {doc_id} COMPLETED SUCCESSFULLY ---")
    else:
        logger.error(f"--- ❌ ❌ ❌ Ingestion pipeline for doc_id: {doc_id} FAILED during vector upsert. ---")
