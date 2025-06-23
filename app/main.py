# app/main.py

# --- ENVIRONMENT LOADING (MUST BE FIRST) ---
# This block is the definitive fix. It runs before anything else.
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

print("--- SARA Legal Assistant: Starting Application ---")

# We start from the current working directory, where you run `python -m app.main`.
# This is the most reliable way to find the .env file in your project root.
current_path = Path.cwd()
ENV_PATH = current_path / ".env"

if not ENV_PATH.exists():
    print(f"FATAL ERROR: Could not find the .env file at '{ENV_PATH}'.")
    print(f"Please ensure you are running the command from the root 'sara/' directory.")
    sys.exit(1) # Exit the program with an error code.

load_dotenv(dotenv_path=ENV_PATH)

# --- DEBUGGING: Prove that the variable is loaded correctly ---
provider_from_env = os.getenv("WHATSAPP_PROVIDER")
print(f"SUCCESS: .env file loaded from '{ENV_PATH}'")
print(f"VALUE CHECK: WHATSAPP_PROVIDER is set to '{provider_from_env}'")
if not provider_from_env:
    print("FATAL ERROR: WHATSAPP_PROVIDER is not set in your .env file!")
    sys.exit(1)

# --- Now, and only now, we import the rest of the application ---
import logging
import json
import uuid
import tempfile
import uvicorn

from fastapi import FastAPI, Request, Response, BackgroundTasks, HTTPException, Header
from twilio.request_validator import RequestValidator

from app.core.config import settings
from app.services import rag_service

# --- Dynamic Service Loading (will now work correctly) ---
if settings.WHATSAPP_PROVIDER.lower() == "twilio":
    from app.services import whatsapp_service_t as whatsapp_service
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
            if num_media > 0:
                media_url = form_data.get("MediaUrl0")
                mime_type = form_data.get("MediaContentType0")
                background_tasks.add_task(handle_document_message, media_url, mime_type, group_id)
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
                if mime_type in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                    background_tasks.add_task(handle_document_message, media_id, mime_type, group_id)
            return Response(status_code=200)
        except json.JSONDecodeError:
            logger.error("JSONDecodeError: This is a Twilio request but the app is running in 'meta' mode. Check .env file.")
            raise HTTPException(status_code=400, detail="Invalid JSON format for Meta provider.")

@app.get("/webhook")
async def verify_webhook(request: Request):
    if PROVIDER == "meta":
        verification_result = whatsapp_service.verify_webhook(request.query_params)
        if isinstance(verification_result, str):
            return Response(content=verification_result, status_code=200, media_type="text/plain")
        else:
            raise HTTPException(status_code=403, detail="Webhook verification failed.")
    return Response(content="Endpoint only used for Meta verification.", status_code=200)

async def handle_document_message(media_identifier: str, mime_type: str, group_id: str):
    logger.info(f"Handling document in background for group: {group_id}")
    if media_identifier.startswith("http"):
        media_url = media_identifier
    else:
        media_url = whatsapp_service.get_media_url(media_identifier)
        if not media_url:
            logger.error(f"Could not get media URL for Meta media_id: {media_identifier}")
            return
    doc_id = str(uuid.uuid4())
    file_extension = ".pdf" if "pdf" in mime_type else ".docx"
    with tempfile.NamedTemporaryFile(delete=True, suffix=file_extension) as temp_file:
        local_path = Path(temp_file.name)
        download_successful = whatsapp_service.download_media(media_url, local_path)
        if download_successful:
            logger.info(f"Starting ingestion for document {doc_id} from group {group_id}")
            rag_service.ingest_document(local_path, mime_type, group_id, doc_id)
        else:
            logger.error(f"Failed to download media from URL for group {group_id}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"--- Uvicorn starting up for provider: '{PROVIDER}' ---")
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)

