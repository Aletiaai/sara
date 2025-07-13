# app/services/whatsapp_service_vn.py

import logging
import requests
from pathlib import Path

from app.core.config import settings

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- WhatsApp API Configuration ---
WHATSAPP_API_URL = f"https://graph.facebook.com/v22.0/{settings.WHATSAPP_PHONE_NUMBER_ID}/messages"
HEADERS = {
    "Authorization": f"Bearer {settings.WHATSAPP_API_TOKEN}",
    "Content-Type": "application/json",
}


def send_whatsapp_message(to: str, message_text: str):
    """
    Sends a text message to a specified WhatsApp number or group.

    Args:
        to (str): The recipient's WhatsApp ID (can be a group_id).
        message_text (str): The text content of the message to send.
    """
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "text": {"body": message_text},
    }
    logger.info(f"Sending message to {to}: '{message_text}'")
    try:
        response = requests.post(WHATSAPP_API_URL, json=payload, headers=HEADERS)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        logger.info(f"Message sent successfully. Response: {response.json()}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send WhatsApp message to {to}: {e}", exc_info=True)


def get_media_url(media_id: str) -> str | None:
    """
    Retrieves the direct download URL for a media file from its ID.

    Args:
        media_id (str): The ID of the media object provided by WhatsApp.

    Returns:
        str | None: The direct URL to download the media, or None if an error occurs.
    """
    media_api_url = f"https://graph.facebook.com/v19.0/{media_id}/"
    headers = {"Authorization": f"Bearer {settings.WHATSAPP_API_TOKEN}"}
    logger.info(f"Fetching media URL for media_id: {media_id}")
    try:
        response = requests.get(media_api_url, headers=headers)
        response.raise_for_status()
        data = response.json()
        if "url" in data:
            logger.info("Successfully fetched media URL.")
            return data["url"]
        else:
            logger.error(f"Media URL not found in response: {data}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get media URL for {media_id}: {e}", exc_info=True)
        return None


def download_media(media_url: str, local_path: Path) -> bool:
    """
    Downloads a file from a given URL and saves it to a local path.

    Args:
        media_url (str): The direct download URL for the media.
        local_path (Path): The local file path to save the downloaded content.

    Returns:
        bool: True if download was successful, False otherwise.
    """
    headers = {"Authorization": f"Bearer {settings.WHATSAPP_API_TOKEN}"}
    logger.info(f"Downloading media from URL to '{local_path}'")
    try:
        with requests.get(media_url, headers=headers, stream=True) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info("Media downloaded successfully.")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download media: {e}", exc_info=True)
        return False


def verify_webhook(query_params: dict) -> str | int:
    """
    Verifies the webhook subscription with Meta/WhatsApp.

    Args:
        query_params (dict): A dictionary representing the query parameters
                             from the incoming GET request.

    Returns:
        str | int: Returns the challenge string if verification is successful,
                   otherwise returns an HTTP status code for failure.
    """
    mode = query_params.get("hub.mode")
    token = query_params.get("hub.verify_token")
    challenge = query_params.get("hub.challenge")

    if mode == "subscribe" and token == settings.WHATSAPP_VERIFY_TOKEN:
        logger.info("Webhook verification successful!")
        return challenge
    else:
        logger.warning("Webhook verification failed. Check tokens.")
        return 403 # Forbidden
