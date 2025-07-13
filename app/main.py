# app/main.py

# --- ENVIRONMENT LOADING (MUST BE FIRST) ---
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

print("--- SARA Legal Assistant: Starting Application ---")

current_path = Path.cwd()
ENV_PATH = current_path / ".env"

if not ENV_PATH.exists():
    print(f"FATAL ERROR: Could not find the .env file at '{ENV_PATH}'.")
    sys.exit(1) 

# Your fix is the correct and robust way to handle this.
load_dotenv(dotenv_path=ENV_PATH, override=True)

key_file_name = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not key_file_name:
    print("FATAL ERROR: GOOGLE_APPLICATION_CREDENTIALS not set in .env file.")
    sys.exit(1)

key_file_path = current_path / key_file_name
if not key_file_path.exists():
    print(f"FATAL ERROR: Service account key file not found at '{key_file_path}'.")
    sys.exit(1)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_file_path)
print(f"SUCCESS: .env loaded. Using credentials from '{key_file_path}'.")


# --- REGULAR APPLICATION IMPORTS ---
import logging
import json
import uuid
import tempfile
import uvicorn

from fastapi import FastAPI, Request, Response, BackgroundTasks, HTTPException, Header
from twilio.request_validator import RequestValidator

from app.core.config import settings
from app.services import rag_service

# --- Dynamic Service Loading ---
if settings.WHATSAPP_PROVIDER.lower() == "twilio":
    # Import the new functions in addition to the service module itself
    from app.services import whatsapp_service_t as whatsapp_service
    from app.services.whatsapp_service_t import extract_filename_from_message, generate_smart_filename
    PROVIDER = "twilio"
else:
    from app.services import whatsapp_service_vn as whatsapp_service
    PROVIDER = "meta"

# --- Logging Setup ---
log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Sara - Legal Assistant",
    description=f"A RAG assistant integrated with WhatsApp via {PROVIDER.upper()}.",
    version="1.0.0",
)

# --- Twilio Specific Initializer ---
if PROVIDER == "twilio":
    twilio_validator = RequestValidator(settings.TWILIO_AUTH_TOKEN)


# --- Webhook Endpoints ---
@app.post("/webhook")
async def receive_message(request: Request, background_tasks: BackgroundTasks, x_twilio_signature: str = Header(None)):
    """ Single webhook endpoint that routes all incoming WhatsApp events."""
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
            message_body = form_data.get("Body", "") # Get the text caption

            if num_media > 0:
                # This is a document message
                media_url = form_data.get("MediaUrl0")
                mime_type = form_data.get("MediaContentType0")
                # Pass the message_body to the background task
                background_tasks.add_task(handle_document_message, media_url, mime_type, group_id, message_body)
                
            else:
                # This is a text message
                text_content = message_body.strip()
                if text_content.lower().startswith("@sara"):
                    question = text_content[5:].strip() # Remove "@sara"
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
                # Meta API doesn't reliably provide a filename in the payload, but we can pass the caption if it exists.
                message_body = document_info.get("caption", "")
                
                if mime_type in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                    background_tasks.add_task(handle_document_message, media_id, mime_type, group_id, message_body)
            
            elif message_type == "text":
                text_content = message_data["text"]["body"].strip()
                if text_content.lower().startswith("@sara"):
                    question = text_content[5:].strip() # Remove "@sara"
                    background_tasks.add_task(handle_text_message, question, group_id)

            return Response(status_code=200)
        except json.JSONDecodeError:
            logger.error("JSONDecodeError: This is a Twilio request but the app is running in 'meta' mode. Check .env file.")
            raise HTTPException(status_code=400, detail="Invalid JSON format for Meta provider.")


@app.get("/webhook")
async def verify_webhook(request: Request):
    """ Handles the one-time webhook verification for the Meta provider. """
    if PROVIDER == "meta":
        verification_result = whatsapp_service.verify_webhook(request.query_params)
        if isinstance(verification_result, str):
            return Response(content=verification_result, status_code=200, media_type="text/plain")
        else:
            raise HTTPException(status_code=403, detail="Webhook verification failed.")
    return Response(content="Endpoint only used for Meta verification.", status_code=200)


# --- Background Task Handlers ---

async def handle_document_message(media_identifier: str, mime_type: str, group_id: str, message_body: str = ""):
    """
    Handles document ingestion, now with logic to determine the original filename.
    """
    logger.info(f"Handling document in background for group: {group_id}")
    
    doc_id = str(uuid.uuid4())
    
    with tempfile.NamedTemporaryFile(delete=True, suffix=".tmp") as temp_file:
        local_path = Path(temp_file.name)
        
        # 1. Download the media file
        # The aliased whatsapp_service will call the correct download function
        # for either Twilio or Meta.
        success, suggested_filename = whatsapp_service.download_media(media_identifier, local_path, mime_type)
        
        if success:
            # 2. Determine the best filename to use
            # Priority 1: Filename explicitly sent in the message body/caption
            original_filename = extract_filename_from_message(message_body) if PROVIDER == "twilio" else None
            
            if original_filename:
                final_filename = original_filename
            # Priority 2: Filename found in the download headers
            elif suggested_filename:
                final_filename = suggested_filename
            # Priority 3: Generate a smart fallback name
            else:
                final_filename = generate_smart_filename(group_id, mime_type) if PROVIDER == "twilio" else f"documento_{doc_id}.pdf"

            logger.info(f"Using filename: '{final_filename}' for ingestion.")

            # 3. Start the ingestion process, passing the final filename
            rag_service.ingest_document(
                file_path=local_path,
                mime_type=mime_type,
                group_id=group_id,
                doc_id=doc_id,
                original_filename=final_filename # Pass the determined filename
            )
        else:
            logger.error(f"Failed to download media from URL for group {group_id}")


async def handle_text_message(question: str, group_id: str):
    """Handles question answering."""
    logger.info(f"Handling question in background for group: {group_id}")
    answer = rag_service.answer_question(question, group_id)
    whatsapp_service.send_whatsapp_message(to=group_id, message_text=answer)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"--- Uvicorn starting up for provider: '{PROVIDER}' ---")
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
