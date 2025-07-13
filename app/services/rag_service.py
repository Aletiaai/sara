# app/services/rag_service.py

import logging
import uuid
import re
import json
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from datetime import datetime

import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
from google.cloud import documentai, storage, aiplatform, firestore

from app.core.config import settings
from app.core.prompts import RAG_PROMPT_TEMPLATE, METADATA_EXTRACTION_PROMPT

# --- Initialize logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Client Initializations ---
vertexai.init(project=settings.GCP_PROJECT_ID, location=settings.VERTEX_AI_REGION)
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
llm_model = GenerativeModel("gemini-2.0-flash-lite-001")

firestore_client = firestore.Client(project=settings.GCP_PROJECT_ID)
GROUP_INDEX_COLLECTION = "sara_group_indexes"
CHUNK_COLLECTION = "sara_document_chunks"
# New: A dedicated document to store the ID of our one shared endpoint
SHARED_RESOURCES_DOC = "sara_shared_resources"


# ==============================================================================
# --- NEW: Shared Endpoint Management ---
# ==============================================================================

def get_or_create_shared_endpoint() -> aiplatform.MatchingEngineIndexEndpoint:
    """
    Gets the one shared Index Endpoint for the entire application.
    If it doesn't exist, it creates it.
    """
    shared_ref = firestore_client.collection(GROUP_INDEX_COLLECTION).document(SHARED_RESOURCES_DOC)
    shared_doc = shared_ref.get()

    if shared_doc.exists:
        endpoint_id = shared_doc.to_dict().get("endpoint_id")
        logger.info(f"Found shared endpoint: {endpoint_id}")
        return aiplatform.MatchingEngineIndexEndpoint(endpoint_id)

    logger.warning("No shared endpoint found. Creating a new one...")
    try:
        endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name="sara-shared-endpoint",
            public_endpoint_enabled=True
        )
        logger.info(f"Successfully created shared endpoint: {endpoint.resource_name}")
        shared_ref.set({"endpoint_id": endpoint.resource_name})
        return endpoint
    except Exception as e:
        logger.error(f"Failed to create shared endpoint: {e}", exc_info=True)
        return None


def get_or_create_group_index(group_id: str, shared_endpoint: aiplatform.MatchingEngineIndexEndpoint) -> Dict[str, str]:
    """
    Gets or creates a dedicated Index for a specific group and deploys it
    to the shared endpoint.
    """
    safe_group_id = group_id.replace(":", "_").replace("+", "")
    group_doc_ref = firestore_client.collection(GROUP_INDEX_COLLECTION).document(safe_group_id)
    
    group_doc = group_doc_ref.get()
    if group_doc.exists:
        logger.info(f"Found existing index for group {group_id}.")
        return group_doc.to_dict()

    logger.warning(f"No index found for group {group_id}. Creating and deploying new index...")
    
    unique_suffix = uuid.uuid4().hex[:6]
    index_display_name = f"sara_index_{safe_group_id}_{unique_suffix}"
    try:
        index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
            display_name=index_display_name,
            dimensions=768,
            approximate_neighbors_count=10,
            distance_measure_type="DOT_PRODUCT_DISTANCE",
            index_update_method="STREAM_UPDATE"
        )
        logger.info(f"Successfully created new index: {index.resource_name}")
        
        deployed_index_id = f"deployed_{index_display_name}"
        logger.info(f"Deploying index to shared endpoint with ID: {deployed_index_id}... This may take 20+ minutes.")
        
        shared_endpoint.deploy_index(
            index=index,
            deployed_index_id=deployed_index_id
        )
        logger.info(f"Successfully deployed index to shared endpoint.")

        new_resources = {
            "index_id": index.resource_name,
            "deployed_index_id": deployed_index_id,
            "group_id": group_id
        }
        group_doc_ref.set(new_resources)
        logger.info(f"Saved new resource mapping to Firestore for group {group_id}.")
        return new_resources
        
    except Exception as e:
        logger.error(f"Failed to create and deploy index for group {group_id}: {e}", exc_info=True)
        return None


# ==============================================================================
# --- Core RAG Logic (Updated for LLM-based Metadata Extraction) ---
# ==============================================================================

def ingest_document(file_path: Path, mime_type: str, group_id: str, doc_id: str, original_filename: str = None):
    logger.info(f"--- Starting ingestion for doc_id: {doc_id} in group: {group_id} ---")

    # Use original filename if provided, otherwise fall back to file_path.name
    actual_filename = original_filename if original_filename else file_path.name
    logger.info(f"Processing document: {actual_filename} (temp file: {file_path.name})")
    
    # 1. Get the one shared endpoint for the app
    shared_endpoint = get_or_create_shared_endpoint()
    if not shared_endpoint:
        logger.error("Could not get shared endpoint. Aborting."); return

    # 2. Get (or create and deploy) the dedicated index for this group
    group_resources = get_or_create_group_index(group_id, shared_endpoint)
    if not group_resources:
        logger.error(f"Could not get group resources for {group_id}. Aborting."); return

    # 3. Process document to extract full text
    full_text = process_document(file_path, mime_type)
    if not full_text:
        logger.error("Ingestion failed: Could not extract text."); return

    # 4. Extract metadata using LLM
    metadata = extract_metadata_with_llm(full_text, actual_filename)
    if not metadata:
        logger.error("Ingestion failed: Could not extract metadata."); return

    # 5. Create text chunks
    text_chunks = create_smart_chunks(full_text)
    if not text_chunks:
        logger.warning("No substantial chunks found. Aborting."); return

    # 6. Generate embeddings and store in vector index
    chunk_ids = [f"{doc_id}::{i}" for i, _ in enumerate(text_chunks)]
    embeddings = get_embeddings(text_chunks)
    if not embeddings:
        logger.error("Ingestion failed: Could not generate embeddings."); return
        
    vector_datapoints = [{"datapoint_id": id, "feature_vector": vec} for id, vec in zip(chunk_ids, embeddings)]
    
    try:
        index = aiplatform.MatchingEngineIndex(index_name=group_resources["index_id"])
        index.upsert_datapoints(datapoints=vector_datapoints)
        logger.info(f"Successfully upserted {len(vector_datapoints)} vectors.")

        # 7. Store chunks with rich metadata in Firestore
        firestore_batch = firestore_client.batch()
        for i, chunk_text in enumerate(text_chunks):
            chunk_id = chunk_ids[i]
            doc_ref = firestore_client.collection(CHUNK_COLLECTION).document(chunk_id)
            firestore_doc = {
                "text": chunk_text,
                "document_id": doc_id,
                "group_id": group_id,
                **metadata  # This now includes all LLM-extracted metadata
            }
            firestore_batch.set(doc_ref, firestore_doc)
        firestore_batch.commit()
        logger.info(f"Successfully stored {len(text_chunks)} rich chunks in Firestore.")
        logger.info(f"--- ✅ ✅ ✅ Ingestion for doc_id: {doc_id} COMPLETED ---")
    except Exception as e:
        logger.error(f"--- ❌ ❌ ❌ Ingestion FAILED during upsert. ---", exc_info=True)


def answer_question(question: str, group_id: str) -> str:
    logger.info(f"--- Answering question for group: {group_id} ---")
    
    shared_endpoint = get_or_create_shared_endpoint()
    if not shared_endpoint:
        return "Disculpa, hay un problema con la configuración del servicio principal."

    safe_group_id = group_id.replace(":", "_").replace("+", "")
    group_doc_ref = firestore_client.collection(GROUP_INDEX_COLLECTION).document(safe_group_id)
    group_resources = group_doc_ref.get()
    
    if not group_resources.exists:
        return "Disculpa, necesito que compartas al menos un documento en este grupo antes de que pueda responder preguntas."

    resources = group_resources.to_dict()
    logger.info(f"Original question: '{question}'")
    
    # Step 1: Get ALL chunks for this group first (for metadata filtering)
    try:
        # Query all chunks for this group
        all_chunks_query = firestore_client.collection(CHUNK_COLLECTION).where("group_id", "==", group_id).limit(200)
        all_chunks_docs = all_chunks_query.stream()
        
        all_chunks_data = []
        for doc in all_chunks_docs:
            if doc.exists:
                chunk_data = doc.to_dict()
                chunk_data['chunk_id'] = doc.id
                all_chunks_data.append(chunk_data)
        
        logger.info(f"Retrieved {len(all_chunks_data)} total chunks for metadata filtering")
        
        # Step 2: Apply metadata filtering
        filtered_chunks = filter_chunks_by_metadata(question, all_chunks_data, group_id)
        
        if not filtered_chunks:
            logger.info("No chunks passed metadata filtering, falling back to semantic search only")
            # Fall back to original semantic search
            question_embedding = get_embeddings([question])[0]
            if not question_embedding:
                return "Disculpa, no pude procesar la pregunta."
            
            neighbor_results = shared_endpoint.find_neighbors(
                queries=[question_embedding],
                deployed_index_id=resources["deployed_index_id"],
                num_neighbors=5
            )
            neighbor_ids = [neighbor.id for r in neighbor_results for neighbor in r]
            
            if not neighbor_ids:
                return "Disculpa, la información que tengo no es suficiente para responder tu pregunta."
            
            # Get the actual chunk data
            doc_refs = [firestore_client.collection(CHUNK_COLLECTION).document(id) for id in neighbor_ids]
            docs = firestore_client.get_all(doc_refs)
            filtered_chunks = [doc.to_dict() for doc in docs if doc.exists]
        
        # Step 3: Perform semantic search on filtered chunks
        logger.info(f"Performing semantic search on {len(filtered_chunks)} filtered chunks")
        
        # Extract chunk IDs from filtered chunks
        filtered_chunk_ids = [chunk['chunk_id'] for chunk in filtered_chunks]
        
        # Generate embedding for question
        question_embedding = get_embeddings([question])[0]
        if not question_embedding:
            return "Disculpa, no pude procesar la pregunta."
        
        # Perform vector search on all chunks but we'll filter results
        neighbor_results = shared_endpoint.find_neighbors(
            queries=[question_embedding],
            deployed_index_id=resources["deployed_index_id"],
            num_neighbors=min(20, len(filtered_chunks))  # Search more but we'll filter
        )
        
        # Get neighbor IDs and filter them to only include our metadata-filtered chunks
        all_neighbor_ids = [neighbor.id for r in neighbor_results for neighbor in r]
        relevant_neighbor_ids = [nid for nid in all_neighbor_ids if nid in filtered_chunk_ids]
        
        # If no semantic matches in filtered set, take top metadata-filtered chunks
        if not relevant_neighbor_ids:
            logger.info("No semantic matches in filtered chunks, using top metadata matches")
            relevant_neighbor_ids = filtered_chunk_ids[:5]  # Take top 5 by metadata score
        
        # Get the final chunks for context
        doc_refs = [firestore_client.collection(CHUNK_COLLECTION).document(id) for id in relevant_neighbor_ids[:5]]
        docs = firestore_client.get_all(doc_refs)
        
        context_parts = []
        for doc in docs:
            if doc.exists:
                doc_data = doc.to_dict()
                # Enhanced context with metadata
                context_str = (
                    f"Fragmento del documento '{doc_data.get('original_file_name', 'Documento')}' "
                    f"(Expediente: {doc_data.get('case_number', 'N/A')}"
                )
                
                # Add related case numbers if available
                related_cases = doc_data.get('related_case_numbers', [])
                if related_cases:
                    context_str += f", Casos relacionados: {', '.join(map(str, related_cases))}"
                
                context_str += (
                    f", Tipo: {doc_data.get('document_type', 'N/A')}, "
                    f"Acción legal: {doc_data.get('legal_action_type', 'N/A')}, "
                    f"Juzgado: {doc_data.get('court_info', 'N/A')}, "
                    f"Fecha: {doc_data.get('document_date', 'N/A')}):\n"
                    f"'{doc_data.get('text', '')}'"
                )
                context_parts.append(context_str)

        if not context_parts:
            return "Disculpa, no encontré información relevante para responder tu pregunta."

        context = "\n\n---\n\n".join(context_parts)
        logger.info(f"Retrieved {len(context_parts)} final chunks for answer generation")
        
        final_answer = ask_llm(question, context)
        logger.info(f"Final answer generated successfully")
        return final_answer
        
    except Exception as e:
        logger.error(f"Error during hybrid retrieval: {e}", exc_info=True)
        return "Disculpa, ocurrió un error al buscar la respuesta."


# ==============================================================================
# --- Helper Functions ---
# ==============================================================================

def process_document(file_path: Path, mime_type: str) -> str:
    """
    Processes a document using Document AI and returns the full text.
    """
    try:
        logger.info(f"Processing document '{file_path.name}' with Document AI...")
        docai_client = documentai.DocumentProcessorServiceClient(
            client_options={"api_endpoint": f"{settings.DOCUMENT_AI_LOCATION}-documentai.googleapis.com"}
        )
        processor_name = docai_client.processor_path(
            settings.GCP_PROJECT_ID, 
            settings.DOCUMENT_AI_LOCATION, 
            settings.DOCUMENT_AI_PROCESSOR_ID
        )
        
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


def extract_metadata_with_llm(full_text: str, file_name: str) -> Optional[Dict]:
    """Uses LLM to extract structured metadata from the document text."""
    logger.info("Extracting metadata using LLM for file: {file_name}")
    
    # For very long documents, we'll use first and last portions
    # This ensures we get header info and signature/date info
    text_for_analysis = get_text_for_metadata_extraction(full_text)
    
    try:
        prompt = METADATA_EXTRACTION_PROMPT.format(document_text=text_for_analysis)
        response = llm_model.generate_content(prompt)
        
        # Parse the JSON response
        metadata_json = response.text.strip()
        
        # Remove any markdown formatting if present
        if metadata_json.startswith("```json"):
            metadata_json = metadata_json[7:-3]
        elif metadata_json.startswith("```"):
            metadata_json = metadata_json[3:-3]
        
        metadata = json.loads(metadata_json)
        
        # Add the actual file name as fallback
        if not metadata.get("original_file_name"):
            metadata["original_file_name"] = file_name
        
        logger.info(f"Successfully extracted metadata: {metadata}")
        return metadata
        
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON from LLM response: {e}")
        logger.error(f"LLM response was: {response.text}")
        # Return basic metadata as fallback
        return {
            "document_type": "Unknown",
            "case_number": None,
            "related_case_numbers": [],  # NEW
            "judge_name": None,
            "court_info": None,
            "document_date": None,
            "case_subject": None,
            "parties": None,
            "original_file_name": file_name,
            "document_priority": "medium",  # NEW
            "legal_action_type": None,  # NEW
            "document_status": "unknown"  # NEW
        }
    except Exception as e:
        logger.error(f"Error extracting metadata with LLM: {e}", exc_info=True)
        return None


def get_text_for_metadata_extraction(full_text: str, max_chars: int = 4000) -> str:
    """
    Extracts the most relevant parts of the document for metadata extraction.
    Takes from the beginning and end of the document where metadata typically appears.
    """
    if len(full_text) <= max_chars:
        return full_text
    
    # Take first portion (header info)
    first_portion = full_text[:max_chars // 2]
    
    # Take last portion (signature, date info)
    last_portion = full_text[-(max_chars // 2):]
    
    # Combine with separator
    return first_portion + "\n\n[... DOCUMENTO CONTINÚA ...]\n\n" + last_portion


def create_smart_chunks(text: str, max_chunk_size: int = 1000, min_chunk_size: int = 100, overlap: int = 50) -> List[str]:
    """
    Creates chunks using multiple strategies to handle scanned documents better.
    """
    chunks = []
    
    # Strategy 1: Try paragraph-based chunking first
    paragraph_chunks = try_paragraph_chunking(text, max_chunk_size, min_chunk_size)
    if len(paragraph_chunks) > 1:
        logger.info("Using paragraph-based chunking strategy")
        return paragraph_chunks
    
    # Strategy 2: Try sentence-based chunking
    sentence_chunks = try_sentence_chunking(text, max_chunk_size, min_chunk_size, overlap)
    if len(sentence_chunks) > 1:
        logger.info("Using sentence-based chunking strategy")
        return sentence_chunks
    
    # Strategy 3: Fall back to fixed-size chunking with overlap
    logger.info("Using fixed-size chunking strategy")
    return create_fixed_size_chunks(text, max_chunk_size, overlap)


def try_paragraph_chunking(text: str, max_chunk_size: int, min_chunk_size: int) -> List[str]:
    """
    Attempts to chunk by paragraphs, combining small ones and splitting large ones.
    """
    # Try different paragraph separators
    separators = ['\n\n\n', '\n\n', '\n']
    
    for separator in separators:
        paragraphs = text.split(separator)
        if len(paragraphs) > 1:
            chunks = []
            current_chunk = ""
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                
                # If adding this paragraph would exceed max size, save current chunk
                if current_chunk and len(current_chunk) + len(paragraph) > max_chunk_size:
                    if len(current_chunk) >= min_chunk_size:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    # Add to current chunk
                    current_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            
            # Add the last chunk
            if current_chunk and len(current_chunk) >= min_chunk_size:
                chunks.append(current_chunk.strip())
            
            # If we got multiple chunks, return them
            if len(chunks) > 1:
                return chunks
    
    return []


def try_sentence_chunking(text: str, max_chunk_size: int, min_chunk_size: int, overlap: int) -> List[str]:
    """
    Attempts to chunk by sentences, respecting size limits.
    """
    # Simple sentence splitting (can be improved with more sophisticated NLP)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    if len(sentences) <= 1:
        return []
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # If adding this sentence would exceed max size, save current chunk
        if current_chunk and len(current_chunk) + len(sentence) > max_chunk_size:
            if len(current_chunk) >= min_chunk_size:
                chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap from previous chunk
            if overlap > 0 and chunks:
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk = sentence
        else:
            # Add to current chunk
            current_chunk = current_chunk + " " + sentence if current_chunk else sentence
    
    # Add the last chunk
    if current_chunk and len(current_chunk) >= min_chunk_size:
        chunks.append(current_chunk.strip())
    
    return chunks if len(chunks) > 1 else []


def create_fixed_size_chunks(text: str, max_chunk_size: int, overlap: int) -> List[str]:
    """
    Creates fixed-size chunks with overlap as a fallback strategy.
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_chunk_size
        
        # If this isn't the last chunk, try to end at a word boundary
        if end < len(text):
            # Find the last space before the max_chunk_size limit
            last_space = text.rfind(' ', start, end)
            if last_space > start:
                end = last_space
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap if end - overlap > start else end
        
        # Prevent infinite loop
        if start >= len(text):
            break
    
    return chunks


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Converts a list of text chunks into a list of embeddings.
    """
    logger.info(f"Generating embeddings for {len(texts)} chunks...")
    try:
        embeddings = embedding_model.get_embeddings(texts)
        return [embedding.values for embedding in embeddings]
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}", exc_info=True)
        return []


def ask_llm(question: str, context: str) -> str:
    """
    Constructs a prompt and sends it to the Gemini LLM to get an answer.
    """
    logger.info("Sending prompt to LLM...")
    try:
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
        response = llm_model.generate_content(prompt)
        logger.info("Successfully received answer from LLM.")
        return response.text
    except Exception as e:
        logger.error(f"Error generating content with LLM: {e}", exc_info=True)
        return "Disculpa, ocurrió un error al generar la respuesta."
    
def filter_chunks_by_metadata(question: str, chunks_data: List[Dict], group_id: str) -> List[Dict]:
    """Intelligently filters chunks based on metadata before semantic search."""
    logger.info(f"Filtering {len(chunks_data)} chunks by metadata...")
    
    # Extract potential case numbers, legal terms, and entities from question
    question_lower = question.lower()
    
    # Find case numbers in question (pattern: digits, sometimes with letters)
    case_number_pattern = r'\b\d+[/-]?\w*\b'
    potential_case_numbers = re.findall(case_number_pattern, question)
    
    # Legal action keywords
    legal_action_keywords = {
        'amparo': ['amparo', 'amparo directo', 'amparo indirecto'],
        'civil': ['civil', 'demanda civil', 'juicio civil'],
        'penal': ['penal', 'proceso penal', 'delito'],
        'laboral': ['laboral', 'trabajo', 'empleado'],
        'mercantil': ['mercantil', 'comercial', 'empresa'],
        'familiar': ['familiar', 'divorcio', 'alimentos', 'custodia'],
        'administrativo': ['administrativo', 'gobierno', 'autoridad']
    }
    
    # Document type importance for different question types
    resolution_keywords = ['resolución', 'sentencia', 'acuerdo', 'decreto', 'fallo']
    notification_keywords = ['notificación', 'citatorio', 'emplazamiento']
    filing_keywords = ['demanda', 'escrito', 'promoción', 'alegatos']
    
    filtered_chunks = []
    
    for chunk in chunks_data:
        score = 0
        
        # 1. Case number matching (highest priority)
        chunk_case_number = chunk.get('case_number', '')
        related_cases = chunk.get('related_case_numbers', [])
        
        if chunk_case_number:
            for potential_case in potential_case_numbers:
                if potential_case in chunk_case_number:
                    score += 100
                    break
        
        # Check related case numbers
        for related_case in related_cases:
            for potential_case in potential_case_numbers:
                if potential_case in str(related_case):
                    score += 80
                    break
        
        # 2. Legal action type matching
        legal_action_type = chunk.get('legal_action_type', '').lower()
        for action_type, keywords in legal_action_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                if action_type in legal_action_type or any(keyword in legal_action_type for keyword in keywords):
                    score += 50
                    break
        
        # 3. Document type relevance
        doc_type = chunk.get('document_type', '').lower()
        
        # If question asks for resolution-type documents
        if any(keyword in question_lower for keyword in resolution_keywords):
            if any(keyword in doc_type for keyword in ['resolución', 'sentencia', 'acuerdo', 'decreto']):
                score += 40
        
        # If question asks for notifications
        elif any(keyword in question_lower for keyword in notification_keywords):
            if any(keyword in doc_type for keyword in ['notificación', 'citatorio', 'emplazamiento']):
                score += 40
        
        # If question asks for filings
        elif any(keyword in question_lower for keyword in filing_keywords):
            if any(keyword in doc_type for keyword in ['demanda', 'escrito', 'promoción']):
                score += 40
        
        # 4. Document priority boost
        priority = chunk.get('document_priority', 'medium')
        if priority == 'high':
            score += 20
        elif priority == 'medium':
            score += 10
        
        # 5. Recent document boost (if we have dates)
        doc_date = chunk.get('document_date')
        if doc_date:
            try:
                # Simple date boost - more recent documents get slight preference
                # This is a simplified implementation
                score += 5
            except:
                pass
        
        # 6. Judge/court matching
        if 'juez' in question_lower or 'juzgado' in question_lower:
            judge_name = chunk.get('judge_name', '').lower()
            court_info = chunk.get('court_info', '').lower()
            # Extract judge names from question (simplified)
            if judge_name and any(word in judge_name for word in question_lower.split()):
                score += 30
            if court_info and any(word in court_info for word in question_lower.split()):
                score += 25
        
        # Only include chunks with some relevance score
        if score > 0:
            chunk['relevance_score'] = score
            filtered_chunks.append(chunk)
    
    # Sort by relevance score (highest first)
    filtered_chunks.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
    
    # Return top chunks (limit to avoid too much noise)
    max_chunks = min(20, len(filtered_chunks))  # Increased from typical 5 for better coverage
    result = filtered_chunks[:max_chunks]
    
    logger.info(f"Filtered to {len(result)} relevant chunks based on metadata")
    return result