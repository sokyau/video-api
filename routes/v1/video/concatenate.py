from flask import Blueprint
from app_utils import validate_payload, queue_task_wrapper
import logging
import os
from services.authentication import authenticate
from services.file_management import download_file
from services.ffmpeg_toolkit import concatenate_videos
from services.local_storage import store_file
import config

v1_video_concatenate_bp = Blueprint('v1_video_concatenate', __name__)
logger = logging.getLogger(__name__)

@v1_video_concatenate_bp.route('/v1/video/concatenate', methods=['POST'])
@authenticate
@validate_payload({
    "type": "object",
    "properties": {
        "video_urls": {
            "type": "array",
            "items": {"type": "string", "format": "uri"},
            "minItems": 2
        },
        "webhook_url": {"type": "string", "format": "uri"},
        "id": {"type": "string"}
    },
    "required": ["video_urls"],
    "additionalProperties": False
})
@queue_task_wrapper(bypass_queue=False)
def concatenate_videos_endpoint(job_id, data):
    video_urls = data['video_urls']
    webhook_url = data.get('webhook_url')
    id = data.get('id')

    logger.info(f"Job {job_id}: Received concatenate videos request for {len(video_urls)} videos")

    try:
        # Descargar todos los videos
        video_paths = []
        for i, url in enumerate(video_urls):
            try:
                video_path = download_file(url, config.TEMP_DIR)
                video_paths.append(video_path)
                logger.info(f"Job {job_id}: Downloaded video {i+1}/{len(video_urls)}")
            except Exception as e:
                logger.error(f"Job {job_id}: Error downloading video {i+1}: {str(e)}")
                # Limpiar archivos ya descargados
                for path in video_paths:
                    if os.path.exists(path):
                        os.remove(path)
                raise

        # Concatenar videos
        output_file = concatenate_videos(video_paths)
        logger.info(f"Job {job_id}: Videos concatenated successfully")

        # Almacenar archivo resultante
        file_url = store_file(output_file)
        logger.info(f"Job {job_id}: Output video stored locally: {file_url}")

        # Limpiar archivos temporales
        for path in video_paths:
            if os.path.exists(path):
                os.remove(path)
        
        if os.path.exists(output_file):
            os.remove(output_file)

        return file_url, "/v1/video/concatenate", 200

    except Exception as e:
        logger.error(f"Job {job_id}: Error during video concatenation - {str(e)}")
        # Asegurar limpieza de archivos temporales en caso de error
        if 'video_paths' in locals():
            for path in video_paths:
                if os.path.exists(path):
                    os.remove(path)
        
        if 'output_file' in locals() and os.path.exists(output_file):
            os.remove(output_file)
            
        return str(e), "/v1/video/concatenate", 500
