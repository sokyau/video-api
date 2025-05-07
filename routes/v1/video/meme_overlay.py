from flask import Blueprint
from app_utils import validate_payload, queue_task_wrapper
import logging
import os
from services.authentication import authenticate
from services.meme_overlay import process_meme_overlay
from services.local_storage import store_file

v1_video_meme_overlay_bp = Blueprint('v1_video_meme_overlay', __name__)
logger = logging.getLogger(__name__)

@v1_video_meme_overlay_bp.route('/v1/video/meme_overlay', methods=['POST'])
@authenticate
@validate_payload({
    "type": "object",
    "properties": {
        "video_url": {"type": "string", "format": "uri"},
        "meme_url": {"type": "string", "format": "uri"},
        "position": {
            "type": "string",
            "enum": ["top", "bottom", "left", "right", "top_left", "top_right", "bottom_left", "bottom_right", "center"]
        },
        "scale": {"type": "number", "minimum": 0.1, "maximum": 1.0},
        "webhook_url": {"type": "string", "format": "uri"},
        "id": {"type": "string"}
    },
    "required": ["video_url", "meme_url"],
    "additionalProperties": False
})
@queue_task_wrapper(bypass_queue=False)
def meme_overlay(job_id, data):
    video_url = data['video_url']
    meme_url = data['meme_url']
    position = data.get('position', 'bottom')
    scale = data.get('scale', 0.3)
    webhook_url = data.get('webhook_url')
    id = data.get('id')

    logger.info(f"Job {job_id}: Received meme overlay request for {video_url}")

    try:
        # Procesar overlay
        output_file = process_meme_overlay(
            video_url, meme_url, position, scale, job_id, webhook_url
        )
        logger.info(f"Job {job_id}: Meme overlay process completed successfully")

        # Almacenar archivo localmente
        file_url = store_file(output_file)
        logger.info(f"Job {job_id}: Output video stored locally: {file_url}")

        # Limpiar archivo temporal después de almacenamiento
        os.remove(output_file)

        return file_url, "/v1/video/meme_overlay", 200

    except Exception as e:
        logger.error(f"Job {job_id}: Error during meme overlay process - {str(e)}")
        return str(e), "/v1/video/meme_overlay", 500
