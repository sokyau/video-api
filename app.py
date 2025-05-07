from flask import Flask, jsonify, request
import logging
import os
import threading
import time
from services.cleanup_service import CleanupService
from app_utils import start_queue_processors, cleanup_completed_tasks
from version import get_version_info
import config

# Configurar logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Crear aplicación Flask
app = Flask(__name__)

# Cargar y registrar blueprints
from routes.v1.video.meme_overlay import v1_video_meme_overlay_bp
from routes.v1.video.caption_video import v1_video_caption_video_bp
from routes.v1.video.concatenate import v1_video_concatenate_bp
from routes.v1.video.animated_text import v1_video_animated_text_bp
from routes.v1.media.transform.media_to_mp3 import v1_media_transform_media_to_mp3_bp
from routes.v1.media.media_transcribe import v1_media_media_transcribe_bp
from routes.v1.ffmpeg.ffmpeg_compose import v1_ffmpeg_compose_bp
from routes.v1.image.transform.image_to_video import v1_image_transform_image_to_video_bp

app.register_blueprint(v1_video_meme_overlay_bp)
app.register_blueprint(v1_video_caption_video_bp)
app.register_blueprint(v1_video_concatenate_bp)
app.register_blueprint(v1_video_animated_text_bp)
app.register_blueprint(v1_media_transform_media_to_mp3_bp)
app.register_blueprint(v1_media_media_transcribe_bp)
app.register_blueprint(v1_ffmpeg_compose_bp)
app.register_blueprint(v1_image_transform_image_to_video_bp)

# Iniciar servicio de limpieza
cleanup_service = CleanupService(interval_minutes=30)

@app.route('/', methods=['GET'])
def index():
    """Endpoint raíz con información de la API"""
    return jsonify({
        "service": "Video Processing API",
        "status": "operational",
        **get_version_info()
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para comprobación de salud del servicio"""
    # Comprobar acceso a almacenamiento
    storage_ok = os.access(config.STORAGE_PATH, os.W_OK)
    
    # Comprobar disponibilidad de ffmpeg
    try:
        import subprocess
        ffmpeg_result = subprocess.run(['ffmpeg', '-version'], 
                                      capture_output=True, 
                                      check=True)
        ffmpeg_ok = ffmpeg_result.returncode == 0
    except Exception:
        ffmpeg_ok = False
    
    status = "healthy" if storage_ok and ffmpeg_ok else "degraded"
    
    return jsonify({
        "status": status,
        "storage": "ok" if storage_ok else "error",
        "ffmpeg": "ok" if ffmpeg_ok else "error",
        "uptime": time.time() - startup_time
    }), 200 if status == "healthy" else 503

@app.route('/version', methods=['GET'])
def version():
    """Endpoint para información de versión"""
    return jsonify(get_version_info())

@app.route('/maintenance/cleanup', methods=['POST'])
def trigger_cleanup():
    """Endpoint para iniciar limpieza manual de archivos"""
    from services.authentication import authenticate
    
    @authenticate
    def _authenticated_cleanup():
        files_removed, bytes_freed = cleanup_service.run_now()
        return jsonify({
            "status": "success",
            "files_removed": files_removed,
            "space_freed_mb": bytes_freed / 1024 / 1024
        })
    
    return _authenticated_cleanup()

@app.errorhandler(404)
def not_found(error):
    """Manejador de errores 404"""
    return jsonify({
        "status": "error",
        "error": "Not Found",
        "message": "The requested endpoint does not exist."
    }), 404

@app.errorhandler(500)
def server_error(error):
    """Manejador de errores 500"""
    logger.error(f"Server error: {str(error)}")
    return jsonify({
        "status": "error",
        "error": "Internal Server Error",
        "message": "An unexpected error occurred."
    }), 500

def periodic_tasks():
    """Ejecuta tareas periódicas"""
    while True:
        try:
            # Limpiar información de tareas completadas
            cleanup_completed_tasks(max_age_hours=6)
        except Exception as e:
            logger.error(f"Error in periodic tasks: {str(e)}")
        
        # Esperar 15 minutos antes de la próxima ejecución
        time.sleep(15 * 60)

# Variables globales para la aplicación
startup_time = time.time()

# Inicializar servicios al arrancar la aplicación
@app.before_first_request
def initialize_services():
    """Inicializa servicios necesarios antes de la primera solicitud"""
    # Iniciar procesadores de cola
    worker_count = config.WORKER_PROCESSES
    start_queue_processors(num_workers=worker_count)
    logger.info(f"Started {worker_count} queue processor workers")
    
    # Iniciar servicio de limpieza
    cleanup_service.start()
    logger.info("Cleanup service started")
    
    # Iniciar thread para tareas periódicas
    periodic_thread = threading.Thread(target=periodic_tasks, daemon=True)
    periodic_thread.start()
    logger.info("Periodic tasks thread started")

# Punto de entrada para ejecución directa
if __name__ == '__main__':
    # En modo desarrollo, inicializar servicios inmediatamente
    initialize_services()
    
    # Ejecutar aplicación en modo desarrollo
    app.run(host='0.0.0.0', port=8080, debug=True)
