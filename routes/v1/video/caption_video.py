from flask import Blueprint
from app_utils import validate_payload, queue_task_wrapper
import logging
import os
from services.authentication import authenticate
from services.caption_video import add_captions_to_video
from services.local_storage import store_file

v1_video_caption_video_bp = Blueprint('v1_video_caption_video', __name__)
logger = logging.getLogger(__name__)

@v1_video_caption_video_bp.route('/v1/video/caption_video', methods=['POST'])
@authenticate
@validate_payload({
    "type": "object",
    "properties": {
        "video_url": {"type": "string", "format": "uri"},
        "subtitles_url": {"type": "string", "format": "uri"},
        "font": {"type": "string"},
        "font_size": {"type": "integer", "minimum": 12, "maximum": 72},
        "font_color": {"type": "string"},
        "background": {"type": "boolean"},
        "position": {"type": "string", "enum": ["bottom", "top"]},
        "webhook_url": {"type": "string", "format": "uri"},
        "id": {"type": "string"}
    },
    "required": ["video_url", "subtitles_url"],
    "additionalProperties": False
})
@queue_task_wrapper(bypass_queue=False)
def caption_video(job_id, data):
    video_url = data['video_url']
    subtitles_url = data['subtitles_url']
    font = data.get('font', 'Arial')
    font_size = data.get('font_size', 24)
    font_color = data.get('font_color', 'white')
    background = data.get('background', True)
    position = data.get('position', 'bottom')
    webhook_url = data.get('webhook_url')
    id = data.get('id')

    logger.info(f"Job {job_id}: Received caption video request for {video_url}")

    try:
        # Añadir subtítulos al video
        output_file = add_captions_to_video(
            video_url, 
            subtitles_url, 
            font, 
            font_size, 
            font_color, 
            background, 
            position, 
            job_id
        )
        logger.info(f"Job {job_id}: Caption video process completed successfully")

        # Almacenar archivo localmente
        file_url = store_file(output_file)
        logger.info(f"Job {job_id}: Output video stored locally: {file_url}")

        # Limpiar archivo temporal después de almacenamiento
        os.remove(output_file)

        return file_url, "/v1/video/caption_video", 200

    except Exception as e:
        logger.error(f"Job {job_id}: Error during caption video process - {str(e)}")
        return str(e), "/v1/video/caption_video", 500
