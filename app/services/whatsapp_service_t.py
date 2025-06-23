# app/services/whatsapp_service_t.py

import logging
import requests
from pathlib import Path
from twilio.rest import Client

from app.core.config import settings

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Twilio Client Initialization ---
try:
    twilio_client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
    logger.info("Twilio client initialized successfully.")
except Exception as e:
    logger.error("Failed to initialize Twilio client. Check credentials.", exc_info=True)
    twilio_client = None


def send_whatsapp_message(to: str, message_text: str):
    """
    Sends a text message to a specified WhatsApp number via Twilio.
    """
    if not twilio_client:
        logger.error("Cannot send message, Twilio client is not available.")
        return

    logger.info(f"Sending Twilio message to {to}: '{message_text}'")
    try:
        message = twilio_client.messages.create(
            from_=settings.TWILIO_PHONE_NUMBER,
            body=message_text,
            to=to
        )
        logger.info(f"Message sent successfully. SID: {message.sid}")
    except Exception as e:
        logger.error(f"Failed to send Twilio message to {to}: {e}", exc_info=True)


def download_media(media_url: str, local_path: Path) -> bool:
    """
    Downloads a media file from a Twilio URL and saves it locally.
    This function now includes HTTP Basic Authentication.
    """
    logger.info(f"Downloading Twilio media from URL to '{local_path}'")
    try:
        # --- THIS IS THE FIX ---
        # We now provide the Account SID and Auth Token to authenticate the download request.
        auth_tuple = (settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        with requests.get(media_url, stream=True, auth=auth_tuple) as r:
            r.raise_for_status() # This will now succeed instead of raising a 401 error
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info("Media downloaded successfully.")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download media from Twilio URL: {e}", exc_info=True)
        return False

