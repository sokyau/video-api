from flask import Blueprint, jsonify
from app_utils import validate_payload, queue_task_wrapper
import logging
import os
from services.authentication import authenticate
from services.transcription import transcribe_media
from services.file_management import download_file
import config

v1_media_media_transcribe_bp = Blueprint('v1_media_media_transcribe', __name__)
logger = logging.getLogger(__name__)

@v1_media_media_transcribe_bp.route('/v1/media/media_transcribe', methods=['POST'])
@authenticate
@validate_payload({
    "type": "object",
    "properties": {
        "media_url": {"type": "string", "format": "uri"},
        "language": {"type": "string"},
        "output_format": {"type": "string", "enum": ["txt", "srt", "vtt", "json"]},
        "webhook_url": {"type": "string", "format": "uri"},
        "id": {"type": "string"}
    },
    "required": ["media_url"],
    "additionalProperties": False
})
@queue_task_wrapper(bypass_queue=False)
def media_transcribe(job_id, data):
    media_url = data['media_url']
    language = data.get('language', 'auto')
    output_format = data.get('output_format', 'txt')
    webhook_url = data.get('webhook_url')
    id = data.get('id')

    logger.info(f"Job {job_id}: Received media transcription request for {media_url}")

    try:
        # Descargar archivo de media
        media_path = download_file(media_url, config.TEMP_DIR)
        logger.info(f"Job {job_id}: Downloaded media to {media_path}")

        # Extraer audio y transcribir
        transcription_result = transcribe_media(media_path, language, output_format, job_id)
        logger.info(f"Job {job_id}: Transcription completed successfully")

        # Limpiar archivo temporal
        os.remove(media_path)

        return transcription_result, "/v1/media/media_transcribe", 200

    except Exception as e:
        logger.error(f"Job {job_id}: Error during media transcription - {str(e)}")
        # Limpiar archivo temporal en caso de error
        if 'media_path' in locals() and os.path.exists(media_path):
            os.remove(media_path)
            
        return str(e), "/v1/media/media_transcribe", 500
