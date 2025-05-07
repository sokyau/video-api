from flask import Blueprint
from app_utils import validate_payload, queue_task_wrapper
import logging
import os
from services.authentication import authenticate
from services.animated_text import process_animated_text
from services.local_storage import store_file

v1_video_animated_text_bp = Blueprint('v1_video_animated_text', __name__)
logger = logging.getLogger(__name__)

@v1_video_animated_text_bp.route('/v1/video/animated_text', methods=['POST'])
@authenticate
@validate_payload({
    "type": "object",
    "properties": {
        "video_url": {"type": "string", "format": "uri"},
        "text": {"type": "string"},
        "animation": {
            "type": "string",
            "enum": ["fade", "slide", "zoom", "typewriter", "bounce"]
        },
        "position": {
            "type": "string",
            "enum": ["top", "bottom", "center"]
        },
        "font": {"type": "string"},
        "font_size": {"type": "integer", "minimum": 12, "maximum": 120},
        "color": {"type": "string"},
        "duration": {"type": "number", "minimum": 1, "maximum": 20},
        "webhook_url": {"type": "string", "format": "uri"},
        "id": {"type": "string"}
    },
    "required": ["video_url", "text"],
    "additionalProperties": False
})
@queue_task_wrapper(bypass_queue=False)
def animated_text(job_id, data):
    video_url = data['video_url']
    text = data['text']
    animation = data.get('animation', 'fade')
    position = data.get('position', 'bottom')
    font = data.get('font', 'Arial')
    font_size = data.get('font_size', 36)
    color = data.get('color', 'white')
    duration = data.get('duration', 3.0)
    webhook_url = data.get('webhook_url')
    id = data.get('id')

    logger.info(f"Job {job_id}: Received animated text request for video: {video_url}")

    try:
        # Procesar texto animado
        output_file = process_animated_text(
            video_url, text, animation, position, font, font_size, color, duration, job_id
        )
        logger.info(f"Job {job_id}: Animated text process completed successfully")

        # Almacenar archivo localmente
        file_url = store_file(output_file)
        logger.info(f"Job {job_id}: Output video stored locally: {file_url}")

        # Limpiar archivo temporal después de almacenamiento
        os.remove(output_file)

        return file_url, "/v1/video/animated_text", 200

    except Exception as e:
        logger.error(f"Job {job_id}: Error during animated text process - {str(e)}")
        return str(e), "/v1/video/animated_text", 500
