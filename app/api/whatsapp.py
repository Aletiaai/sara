import logging
import json
import uuid
import tempfile
from pathlib import Path

from fastapi import APIRouter, Request, Response, BackgroundTasks, HTTPException, Header
from twilio.request_validator import RequestValidator

from app.core.config import settings
from app.services import rag_service

# --- Dynamic Service Loading ---
if settings.WHATSAPP_PROVIDER.lower() == "twilio":
    from app.services import whatsapp_service_t as whatsapp_service
    from app.services.whatsapp_service_t import extract_filename_from_message, generate_smart_filename
    PROVIDER = "twilio"
else:
    from app.services import whatsapp_service_vn as whatsapp_service
    PROVIDER = "meta"

# --- Router & Logger Initialization ---
router = APIRouter()
logger = logging.getLogger(__name__)

# --- Twilio Specific Initializer ---
if PROVIDER == "twilio":
    twilio_validator = RequestValidator(settings.TWILIO_AUTH_TOKEN)


# --- Background Task Handlers ---
async def handle_document_message(media_identifier: str, mime_type: str, group_id: str, message_body: str = ""):
    """
    Handles document ingestion.
    """
    logger.info(f"Handling document in background for group: {group_id}")
    
    doc_id = str(uuid.uuid4())
    
    with tempfile.NamedTemporaryFile(delete=True, suffix=".tmp") as temp_file:
        local_path = Path(temp_file.name)
        
        success, suggested_filename = whatsapp_service.download_media(media_identifier, local_path, mime_type)
        
        if success:
            final_filename = ""
            if PROVIDER == 'twilio':
                original_filename = extract_filename_from_message(message_body)
                final_filename = original_filename or suggested_filename or generate_smart_filename(group_id, mime_type)
            else: # Meta
                final_filename = f"document_{doc_id}.pdf"

            logger.info(f"Using filename: '{final_filename}' for ingestion.")

            rag_service.ingest_document(
                file_path=local_path,
                mime_type=mime_type,
                group_id=group_id,
                doc_id=doc_id,
                original_filename=final_filename
            )
        else:
            logger.error(f"Failed to download media from URL for group {group_id}")


async def handle_text_message(question: str, group_id: str):
    """Handles question answering."""
    logger.info(f"Handling question in background for group: {group_id}")
    answer = rag_service.answer_question(question, group_id)
    whatsapp_service.send_whatsapp_message(to=group_id, message_text=answer)


# --- Webhook Endpoints ---
@router.post("/")
async def receive_message(request: Request, background_tasks: BackgroundTasks, x_twilio_signature: str = Header(None)):
    """Handles all incoming WhatsApp messages and media."""
    logger.info(f"Webhook received. Application running with PROVIDER: '{PROVIDER}'")
    
    if PROVIDER == "twilio":
        logger.info("Processing as Twilio request.")
        try:
            form_data = await request.form()
            if not twilio_validator.validate(str(request.url), form_data, x_twilio_signature):
                logger.warning("Twilio request validation failed.")
                raise HTTPException(status_code=403, detail="Invalid Twilio signature.")
            
            group_id = form_data.get("From")
            num_media = int(form_data.get("NumMedia", 0))
            message_body = form_data.get("Body", "")

            if num_media > 0:
                media_url = form_data.get("MediaUrl0")
                mime_type = form_data.get("MediaContentType0")
                background_tasks.add_task(handle_document_message, media_url, mime_type, group_id, message_body)
            else:
                text_content = message_body.strip()
                if text_content.lower().startswith("@sara"):
                    question = text_content[5:].strip()
                    background_tasks.add_task(handle_text_message, question, group_id)
                    
            return Response(content="", media_type="text/xml")
        except Exception as e:
            logger.error(f"Error processing Twilio form data: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail="Could not process form data.")
    
    else: # Meta Provider Logic
        logger.info("Processing as Meta request.")
        try:
            data = await request.json()
            entry = data.get("entry", [])
            if not entry: return Response(status_code=200)

            changes = entry[0].get("changes", [])
            if not changes: return Response(status_code=200)
            
            value = changes[0].get("value", {})
            messages = value.get("messages", [])
            if not messages: return Response(status_code=200)

            message_data = messages[0]
            group_id = message_data["from"]
            message_type = message_data["type"]

            if message_type == "document":
                document_info = message_data["document"]
                media_id = document_info["id"]
                mime_type = document_info["mime_type"]
                message_body = document_info.get("caption", "")
                
                if mime_type in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                    background_tasks.add_task(handle_document_message, media_id, mime_type, group_id, message_body)
            
            elif message_type == "text":
                text_content = message_data["text"]["body"].strip()
                if text_content.lower().startswith("@sara"):
                    question = text_content[5:].strip()
                    background_tasks.add_task(handle_text_message, question, group_id)

            return Response(status_code=200)
        except json.JSONDecodeError:
            logger.error("JSONDecodeError: A request was not in the expected Meta JSON format.", exc_info=True)
            raise HTTPException(status_code=400, detail="Invalid JSON format for Meta provider.")


@router.get("/")
async def verify_webhook(request: Request):
    """Handles webhook verification for the Meta provider."""
    if PROVIDER == "meta":
        verification_result = whatsapp_service.verify_webhook(request.query_params)
        if isinstance(verification_result, str):
            return Response(content=verification_result, status_code=200, media_type="text/plain")
        else:
            raise HTTPException(status_code=403, detail="Webhook verification failed.")
    return Response(content="Endpoint only used for Meta verification.", status_code=200)