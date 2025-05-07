from flask import Blueprint
from app_utils import validate_payload, queue_task_wrapper
import logging
import os
from services.authentication import authenticate
from services.ffmpeg_toolkit import extract_audio
from services.file_management import download_file
from services.local_storage import store_file
import config

v1_media_transform_media_to_mp3_bp = Blueprint('v1_media_transform_media_to_mp3', __name__)
logger = logging.getLogger(__name__)

@v1_media_transform_media_to_mp3_bp.route('/v1/media/transform/media_to_mp3', methods=['POST'])
@authenticate
@validate_payload({
    "type": "object",
    "properties": {
        "media_url": {"type": "string", "format": "uri"},
        "bitrate": {"type": "string", "pattern": "^[0-9]+k$"},
        "webhook_url": {"type": "string", "format": "uri"},
        "id": {"type": "string"}
    },
    "required": ["media_url"],
    "additionalProperties": False
})
@queue_task_wrapper(bypass_queue=False)
def media_to_mp3(job_id, data):
    media_url = data['media_url']
    bitrate = data.get('bitrate', '192k')
    webhook_url = data.get('webhook_url')
    id = data.get('id')

    logger.info(f"Job {job_id}: Received media to MP3 request for {media_url}")

    try:
        # Descargar archivo de media
        media_path = download_file(media_url, config.TEMP_DIR)
        logger.info(f"Job {job_id}: Downloaded media to {media_path}")

        # Extraer audio en formato MP3
        output_file = extract_audio(media_path, format='mp3', bitrate=bitrate)
        logger.info(f"Job {job_id}: Audio extraction completed successfully")

        # Almacenar archivo resultante
        file_url = store_file(output_file)
        logger.info(f"Job {job_id}: Output audio stored locally: {file_url}")

        # Limpiar archivos temporales
        os.remove(media_path)
        os.remove(output_file)

        return file_url, "/v1/media/transform/media_to_mp3", 200

    except Exception as e:
        logger.error(f"Job {job_id}: Error during media to MP3 conversion - {str(e)}")
        # Limpiar archivos temporales en caso de error
        if 'media_path' in locals() and os.path.exists(media_path):
            os.remove(media_path)
        
        if 'output_file' in locals() and os.path.exists(output_file):
            os.remove(output_file)
            
        return str(e), "/v1/media/transform/media_to_mp3", 500
