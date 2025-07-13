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
import uvicorn
from fastapi import FastAPI
from app.core.config import settings

# Import the new, separated routers
from app.api import whatsapp


# --- Logging Setup ---
log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Sara - Legal Assistant",
    description=f"A RAG assistant integrated with WhatsApp via {settings.WHATSAPP_PROVIDER.upper()} and Web.",
    version="1.0.0",
)


# --- Include API Routers ---
# We add each router with a specific prefix and tags for organization.
app.include_router(whatsapp.router, prefix="/webhook", tags=["WhatsApp"])


# --- Main Application Runner ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    provider = settings.WHATSAPP_PROVIDER.upper()
    logger.info(f"--- Uvicorn starting up for provider: '{provider}' ---")
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)