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


def download_media(media_url: str, local_path: Path, content_type: str = None) -> tuple[bool, str]:
    """
    Downloads a media file from a Twilio URL and saves it locally.
    Returns (success, suggested_filename)
    """
    logger.info(f"Downloading Twilio media from URL to '{local_path}'")
    suggested_filename = None
    
    try:
        auth_tuple = (settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        with requests.get(media_url, stream=True, auth=auth_tuple) as r:
            r.raise_for_status()
            
            # Try to get filename from Content-Disposition header
            content_disposition = r.headers.get('content-disposition')
            if content_disposition:
                import re
                filename_match = re.search(r'filename[^;=\n]*=(([\'"]).*?\2|[^;\n]*)', content_disposition)
                if filename_match:
                    suggested_filename = filename_match.group(1).strip('\'"')
                    logger.info(f"Found filename in headers: {suggested_filename}")
            
            # If no filename in headers, try to guess from content type
            if not suggested_filename and content_type:
                extension_map = {
                    'application/pdf': '.pdf',
                    'image/jpeg': '.jpg',
                    'image/png': '.png',
                    'application/msword': '.doc',
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
                    'text/plain': '.txt'
                }
                extension = extension_map.get(content_type, '')
                if extension:
                    suggested_filename = f"documento{extension}"
                    logger.info(f"Generated filename from content type: {suggested_filename}")
            
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        logger.info("Media downloaded successfully.")
        return True, suggested_filename
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download media from Twilio URL: {e}", exc_info=True)
        return False, None
    
def extract_filename_from_message(message_body: str) -> str:
    """
    Extracts filename from WhatsApp message body if user includes it.
    Users can send messages like:
    - "acuerdo-26-febrero.pdf" (just the filename)
    - "Enviando: acuerdo-26-febrero.pdf"
    - "Archivo: acuerdo-26-febrero.pdf"
    """
    if not message_body:
        return None
    
    import re
    
    # Remove common WhatsApp artifacts
    clean_body = message_body.strip()
    
    # Patterns to match filenames
    patterns = [
        # "Enviando: filename.pdf" or "Archivo: filename.pdf"
        r'(?:enviando|archivo|file|documento):\s*([^\s]+\.(?:pdf|docx?|txt|jpe?g|png))',
        # Just "filename.pdf" (entire message is filename)
        r'^([^\s]+\.(?:pdf|docx?|txt|jpe?g|png))$',
        # "filename.pdf" anywhere in the message
        r'([^\s]+\.(?:pdf|docx?|txt|jpe?g|png))',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, clean_body.lower())
        if match:
            filename = match.group(1)
            logger.info(f"Extracted filename from message: {filename}")
            return filename
    
    return None

def generate_smart_filename(sender_number: str, content_type: str, timestamp: str = None) -> str:
    """
    Generates a meaningful filename when original is not available.
    """
    from datetime import datetime
    
    # Get file extension from content type
    extension_map = {
        'application/pdf': '.pdf',
        'image/jpeg': '.jpg',
        'image/png': '.png',
        'application/msword': '.doc',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
        'text/plain': '.txt'
    }
    
    extension = extension_map.get(content_type, '.pdf')  # Default to PDF
    
    # Clean sender number (remove +52 country code, etc.)
    clean_number = sender_number.replace('+', '').replace('whatsapp:', '')[-10:]  # Last 10 digits
    
    # Use timestamp or current time
    if timestamp:
        time_str = timestamp
    else:
        time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    filename = f"doc_{clean_number}_{time_str}{extension}"
    logger.info(f"Generated smart filename: {filename}")
    return filename    