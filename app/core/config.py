# app/core/config.py

import os
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    This class now assumes that the .env file has already been loaded
    into the process's environment by main.py.
    """
    # --- Google Cloud ---
    GCP_PROJECT_ID: str
    GOOGLE_APPLICATION_CREDENTIALS: str

    # --- Document AI ---
    DOCUMENT_AI_PROCESSOR_ID: str
    DOCUMENT_AI_LOCATION: str = "us"

    # --- Vertex AI ---
    VERTEX_AI_REGION: str = "us-central1"
    VECTOR_SEARCH_INDEX_ID: str
    VECTOR_SEARCH_ENDPOINT_ID: str

    # --- Twilio ---
    TWILIO_ACCOUNT_SID: str = ""
    TWILIO_AUTH_TOKEN: str = ""
    TWILIO_PHONE_NUMBER: str = ""

    # --- Provider Selection ---
    WHATSAPP_PROVIDER: str = "meta"

    class Config:
        # Pydantic will read from the environment variables, case-insensitively.
        case_sensitive = False

# This line reads the environment variables and creates the settings object.
settings = Settings()

# This part is still necessary to tell the Google libraries where to find the key file.
BASE_DIR = Path(__file__).resolve().parent.parent.parent
credentials_path = BASE_DIR / settings.GOOGLE_APPLICATION_CREDENTIALS
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)