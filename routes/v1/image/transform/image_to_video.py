from flask import Blueprint
from app_utils import validate_payload, queue_task_wrapper
import logging
import os
from services.authentication import authenticate
from services.file_management import download_file
from services.ffmpeg_toolkit import image_to_video
from services.local_storage import store_file
import config

v1_image_transform_image_to_video_bp = Blueprint('v1_image_transform_image_to_video', __name__)
logger = logging.getLogger(__name__)

@v1_image_transform_image_to_video_bp.route('/v1/image/transform/image_to_video', methods=['POST'])
@authenticate
@validate_payload({
    "type": "object",
    "properties": {
        "image_url": {"type": "string", "format": "uri"},
        "duration": {"type": "number", "minimum": 1, "maximum": 60},
        "width": {"type": "integer", "minimum": 100, "maximum": 3840},
        "height": {"type": "integer", "minimum": 100, "maximum": 2160},
        "webhook_url": {"type": "string", "format": "uri"},
        "id": {"type": "string"}
    },
    "required": ["image_url", "duration"],
    "additionalProperties": False
})
@queue_task_wrapper(bypass_queue=False)
def image_to_video_endpoint(job_id, data):
    image_url = data['image_url']
    duration = data['duration']
    width = data.get('width')
    height = data.get('height')
    webhook_url = data.get('webhook_url')
    id = data.get('id')

    logger.info(f"Job {job_id}: Received image to video request for {image_url}")

    try:
        # Descargar imagen
        image_path = download_file(image_url, config.TEMP_DIR)
        logger.info(f"Job {job_id}: Downloaded image to {image_path}")

        # Convertir imagen a video
        output_file = image_to_video(image_path, duration, width=width, height=height)
        logger.info(f"Job {job_id}: Image to video conversion completed successfully")

        # Almacenar archivo resultante
        file_url = store_file(output_file)
        logger.info(f"Job {job_id}: Output video stored locally: {file_url}")

        # Limpiar archivos temporales
        os.remove(image_path)
        os.remove(output_file)

        return file_url, "/v1/image/transform/image_to_video", 200

    except Exception as e:
        logger.error(f"Job {job_id}: Error during image to video conversion - {str(e)}")
        # Limpiar archivos temporales en caso de error
        if 'image_path' in locals() and os.path.exists(image_path):
            os.remove(image_path)
        
        if 'output_file' in locals() and os.path.exists(output_file):
            os.remove(output_file)
            
        return str(e), "/v1/image/transform/image_to_video", 500
