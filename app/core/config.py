# app/core/config.py

import os
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

# This allows me to reliably locate files like 'service-account-key.json'
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    """Application settings loaded from environment variables. Pydantic's BaseSettings automatically reads from a .env file."""
    # --- Google Cloud ---
    GCP_PROJECT_ID: str = Field(..., env="GCP_PROJECT_ID")
    GOOGLE_APPLICATION_CREDENTIALS: str = Field(..., env="GOOGLE_APPLICATION_CREDENTIALS")

    # --- Document AI ---
    DOCUMENT_AI_PROCESSOR_ID: str = Field(..., env="DOCUMENT_AI_PROCESSOR_ID")
    DOCUMENT_AI_LOCATION: str = Field("us", env="DOCUMENT_AI_LOCATION")

    # --- Vertex AI ---
    VERTEX_AI_REGION: str = Field("us-central1", env="VERTEX_AI_REGION")
    VECTOR_SEARCH_INDEX_ID: str = Field(..., env="VECTOR_SEARCH_INDEX_ID")
    VECTOR_SEARCH_ENDPOINT_ID: str = Field(..., env="VECTOR_SEARCH_ENDPOINT_ID")

    # --- WhatsApp ---
    WHATSAPP_API_TOKEN: str = Field(..., env="WHATSAPP_API_TOKEN")
    WHATSAPP_PHONE_NUMBER_ID: str = Field(..., env="WHATSAPP_PHONE_NUMBER_ID")
    WHATSAPP_VERIFY_TOKEN: str = Field(..., env="WHATSAPP_VERIFY_TOKEN")


    class Config:
        """Pydantic settings configuration. Specifies the location of the .env file."""
        env_file = os.path.join(BASE_DIR, ".env")
        env_file_encoding = 'utf-8'


# Create a single, importable instance of the settings
settings = Settings()

# Google Cloud client libraries automatically use this variable.
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(BASE_DIR, settings.GOOGLE_APPLICATION_CREDENTIALS)
