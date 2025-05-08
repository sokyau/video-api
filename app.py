from flask import Flask, jsonify, request, send_from_directory, Response
import logging
import os
import threading
import time
import shutil
import json
import platform
import psutil
from datetime import datetime
from logging.handlers import RotatingFileHandler
from werkzeug.middleware.proxy_fix import ProxyFix
from services.cleanup_service import CleanupService
from app_utils import start_queue_processors, cleanup_completed_tasks, get_queue_stats
from version import get_version_info
import config

# Configurar logging
log_directory = 'logs'
os.makedirs(log_directory, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Añadir handler para archivo con rotación
file_handler = RotatingFileHandler(
    os.path.join(log_directory, 'videoapi.log'),
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logging.getLogger().addHandler(file_handler)

logger = logging.getLogger(__name__)

# Crear aplicación Flask
app = Flask(__name__)

# Configurar para trabajar detrás de proxy
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# Variables globales para la aplicación
startup_time = time.time()
worker_count = config.WORKER_PROCESSES
cleanup_interval = getattr(config, 'CLEANUP_INTERVAL_MINUTES', 30)

# Inicializar servicios
cleanup_service = CleanupService(interval_minutes=cleanup_interval)

# Cargar y registrar blueprints
from routes.v1.video.meme_overlay import v1_video_meme_overlay_bp
from routes.v1.video.caption_video import v1_video_caption_video_bp
from routes.v1.video.concatenate import v1_video_concatenate_bp
from routes.v1.video.animated_text import v1_video_animated_text_bp
from routes.v1.media.transform.media_to_mp3 import v1_media_transform_media_to_mp3_bp
from routes.v1.media.media_transcribe import v1_media_media_transcribe_bp
from routes.v1.ffmpeg.ffmpeg_compose import v1_ffmpeg_compose_bp
from routes.v1.image.transform.image_to_video import v1_image_transform_image_to_video_bp
from services.authentication import authenticate

# Registrar todos los blueprints
app.register_blueprint(v1_video_meme_overlay_bp)
app.register_blueprint(v1_video_caption_video_bp)
app.register_blueprint(v1_video_concatenate_bp)
app.register_blueprint(v1_video_animated_text_bp)
app.register_blueprint(v1_media_transform_media_to_mp3_bp)
app.register_blueprint(v1_media_media_transcribe_bp)
app.register_blueprint(v1_ffmpeg_compose_bp)
app.register_blueprint(v1_image_transform_image_to_video_bp)

# Configurar CORS para permitir solicitudes de origen cruzado si es necesario
@app.after_request
def add_cors_headers(response):
    # Estas cabeceras permiten CORS para API
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-API-Key'
    return response

@app.route('/', methods=['GET'])
def index():
    """Endpoint raíz con información de la API"""
    return jsonify({
        "service": "Video Processing API",
        "status": "operational",
        "endpoints": {
            "video": [
                "/v1/video/meme_overlay",
                "/v1/video/caption_video",
                "/v1/video/concatenate",
                "/v1/video/animated_text"
            ],
            "media": [
                "/v1/media/transform/media_to_mp3",
                "/v1/media/media_transcribe"
            ],
            "ffmpeg": [
                "/v1/ffmpeg/ffmpeg_compose"
            ],
            "image": [
                "/v1/image/transform/image_to_video"
            ]
        },
        **get_version_info()
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para comprobación de salud del servicio"""
    health_status = {
        "status": "healthy",
        "checks": {},
        "uptime": time.time() - startup_time,
        "uptime_formatted": format_time_delta(time.time() - startup_time)
    }
    
    # Comprobar acceso a almacenamiento
    try:
        storage_ok = os.access(config.STORAGE_PATH, os.W_OK)
        test_file_path = os.path.join(config.STORAGE_PATH, '.health_check_test')
        with open(test_file_path, 'w') as f:
            f.write('test')
        os.remove(test_file_path)
        health_status["checks"]["storage"] = {"status": "ok"}
    except Exception as e:
        storage_ok = False
        health_status["checks"]["storage"] = {
            "status": "error",
            "message": str(e)
        }
    
    # Comprobar disponibilidad de ffmpeg
    try:
        import subprocess
        ffmpeg_result = subprocess.run(['ffmpeg', '-version'], 
                                      capture_output=True, 
                                      text=True,
                                      check=True)
        ffmpeg_ok = ffmpeg_result.returncode == 0
        ffmpeg_version = ffmpeg_result.stdout.split('\n')[0] if ffmpeg_result.stdout else "Unknown"
        health_status["checks"]["ffmpeg"] = {
            "status": "ok",
            "version": ffmpeg_version
        }
    except Exception as e:
        ffmpeg_ok = False
        health_status["checks"]["ffmpeg"] = {
            "status": "error",
            "message": str(e)
        }
    
    # Comprobar disponibilidad de ffprobe
    try:
        ffprobe_result = subprocess.run(['ffprobe', '-version'], 
                                      capture_output=True, 
                                      text=True,
                                      check=True)
        ffprobe_ok = ffprobe_result.returncode == 0
        ffprobe_version = ffprobe_result.stdout.split('\n')[0] if ffprobe_result.stdout else "Unknown"
        health_status["checks"]["ffprobe"] = {
            "status": "ok",
            "version": ffprobe_version
        }
    except Exception as e:
        ffprobe_ok = False
        health_status["checks"]["ffprobe"] = {
            "status": "error",
            "message": str(e)
        }
    
    # Comprobar espacio en disco
    try:
        disk_usage = shutil.disk_usage(config.STORAGE_PATH)
        disk_total_gb = disk_usage.total / (1024**3)
        disk_free_gb = disk_usage.free / (1024**3)
        disk_used_percent = (disk_usage.used / disk_usage.total) * 100
        
        disk_status = "ok"
        if disk_used_percent > 90:
            disk_status = "warning"
        if disk_used_percent > 95:
            disk_status = "error"
        
        health_status["checks"]["disk"] = {
            "status": disk_status,
            "total_gb": round(disk_total_gb, 2),
            "free_gb": round(disk_free_gb, 2),
            "used_percent": round(disk_used_percent, 2)
        }
        disk_ok = disk_status != "error"
    except Exception as e:
        disk_ok = False
        health_status["checks"]["disk"] = {
            "status": "error",
            "message": str(e)
        }
    
    # Verificar el sistema de cola
    try:
        queue_stats = get_queue_stats()
        health_status["checks"]["queue"] = {
            "status": "ok",
            "stats": queue_stats
        }
        queue_ok = True
    except Exception as e:
        queue_ok = False
        health_status["checks"]["queue"] = {
            "status": "error",
            "message": str(e)
        }
    
    # Determinar estado general
    if all([storage_ok, ffmpeg_ok, ffprobe_ok, disk_ok, queue_ok]):
        health_status["status"] = "healthy"
        status_code = 200
    else:
        if any([not disk_ok, not storage_ok]):
            # Problemas críticos
            health_status["status"] = "critical"
            status_code = 503  # Service Unavailable
        else:
            # Problemas no críticos
            health_status["status"] = "degraded"
            status_code = 200  # No bloqueante pero advertencia
    
    return jsonify(health_status), status_code

@app.route('/metrics', methods=['GET'])
@authenticate
def metrics():
    """Endpoint para métricas del sistema"""
    # Obtener información del sistema
    cpu_usage = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk_usage = shutil.disk_usage(config.STORAGE_PATH)
    disk_total_gb = disk_usage.total / (1024**3)
    disk_free_gb = disk_usage.free / (1024**3)
    disk_used_percent = (disk_usage.used / disk_usage.total) * 100
    
    # Obtener estadísticas de la cola
    queue_stats = get_queue_stats()
    
    # Obtener conteo de archivos en almacenamiento
    storage_file_count = 0
    storage_size_bytes = 0
    try:
        for root, _, files in os.walk(config.STORAGE_PATH):
            for file in files:
                if not file.endswith('.meta'):
                    file_path = os.path.join(root, file)
                    storage_file_count += 1
                    storage_size_bytes += os.path.getsize(file_path)
    except Exception as e:
        logger.error(f"Error contando archivos en almacenamiento: {str(e)}")
    
    # Compilar métricas
    metrics_data = {
        "system": {
            "cpu_usage_percent": cpu_usage,
            "memory_total_gb": memory.total / (1024**3),
            "memory_available_gb": memory.available / (1024**3),
            "memory_used_percent": memory.percent,
            "disk_total_gb": round(disk_total_gb, 2),
            "disk_free_gb": round(disk_free_gb, 2),
            "disk_used_percent": round(disk_used_percent, 2),
            "host": platform.node(),
            "platform": platform.platform(),
            "uptime_seconds": time.time() - startup_time,
            "uptime_formatted": format_time_delta(time.time() - startup_time)
        },
        "storage": {
            "file_count": storage_file_count,
            "size_mb": round(storage_size_bytes / (1024**2), 2)
        },
        "queue": queue_stats,
        "config": {
            "worker_processes": config.WORKER_PROCESSES,
            "max_queue_length": config.MAX_QUEUE_LENGTH,
            "max_file_age_hours": config.MAX_FILE_AGE_HOURS,
            "ffmpeg_threads": config.FFMPEG_THREADS
        },
        "version": get_version_info()
    }
    
    return jsonify(metrics_data)

@app.route('/version', methods=['GET'])
def version():
    """Endpoint para información de versión"""
    version_info = get_version_info()
    version_info["python_version"] = platform.python_version()
    version_info["platform"] = platform.platform()
    
    return jsonify(version_info)

@app.route('/storage/<path:filename>', methods=['GET'])
def serve_file(filename):
    """Servir archivos desde el directorio de almacenamiento"""
    # Esta función permite servir archivos directamente desde Flask 
    # en modo de desarrollo o cuando no se usa Nginx para servir archivos
    
    # Verificación de seguridad básica para evitar path traversal
    if '..' in filename or filename.startswith('/'):
        return jsonify({"error": "Acceso no autorizado"}), 403
    
    # Verificar si el archivo existe
    file_path = os.path.join(config.STORAGE_PATH, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "Archivo no encontrado"}), 404
    
    # Determinar tipo MIME basado en extensión
    content_type = None
    ext = os.path.splitext(filename)[1].lower()
    if ext in ['.mp4', '.webm']:
        content_type = f'video/{ext[1:]}'
    elif ext in ['.mp3', '.wav', '.ogg']:
        content_type = f'audio/{ext[1:]}'
    elif ext in ['.jpg', '.jpeg']:
        content_type = 'image/jpeg'
    elif ext in ['.png']:
        content_type = 'image/png'
    
    return send_from_directory(
        config.STORAGE_PATH, 
        filename, 
        as_attachment=False,
        mimetype=content_type
    )

@app.route('/maintenance/cleanup', methods=['POST'])
@authenticate
def trigger_cleanup():
    """Endpoint para iniciar limpieza manual de archivos"""
    try:
        files_removed, bytes_freed = cleanup_service.run_now()
        return jsonify({
            "status": "success",
            "files_removed": files_removed,
            "space_freed_mb": bytes_freed / 1024 / 1024
        })
    except Exception as e:
        logger.error(f"Error en limpieza manual: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/jobs/<job_id>', methods=['GET'])
@authenticate
def get_job_status(job_id):
    """Endpoint para consultar el estado de un trabajo"""
    from app_utils import get_job_status as get_status
    
    status = get_status(job_id)
    if status:
        return jsonify({
            "status": "success",
            "job_id": job_id,
            "job_status": status
        })
    else:
        return jsonify({
            "status": "error",
            "job_id": job_id,
            "error": "Trabajo no encontrado"
        }), 404

@app.errorhandler(404)
def not_found(error):
    """Manejador de errores 404"""
    return jsonify({
        "status": "error",
        "error": "Not Found",
        "message": "El endpoint solicitado no existe.",
        "available_endpoints": {
            "video": [
                "/v1/video/meme_overlay",
                "/v1/video/caption_video",
                "/v1/video/concatenate",
                "/v1/video/animated_text"
            ],
            "media": [
                "/v1/media/transform/media_to_mp3",
                "/v1/media/media_transcribe"
            ],
            "ffmpeg": [
                "/v1/ffmpeg/ffmpeg_compose"
            ],
            "image": [
                "/v1/image/transform/image_to_video"
            ]
        }
    }), 404

@app.errorhandler(400)
def bad_request(error):
    """Manejador de errores 400"""
    return jsonify({
        "status": "error",
        "error": "Bad Request",
        "message": str(error)
    }), 400

@app.errorhandler(401)
def unauthorized(error):
    """Manejador de errores 401"""
    return jsonify({
        "status": "error",
        "error": "Unauthorized",
        "message": "Se requiere autenticación mediante API key (X-API-Key header)"
    }), 401

@app.errorhandler(500)
def server_error(error):
    """Manejador de errores 500"""
    logger.error(f"Error del servidor: {str(error)}")
    return jsonify({
        "status": "error",
        "error": "Internal Server Error",
        "message": "Ocurrió un error inesperado en el servidor",
        "error_details": str(error) if app.debug else None
    }), 500

def format_time_delta(seconds):
    """Formatea segundos en un formato legible (días, horas, minutos, segundos)"""
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    parts = []
    if days > 0:
        parts.append(f"{int(days)} días")
    if hours > 0 or days > 0:
        parts.append(f"{int(hours)} horas")
    if minutes > 0 or hours > 0 or days > 0:
        parts.append(f"{int(minutes)} minutos")
    parts.append(f"{int(seconds)} segundos")
    
    return ", ".join(parts)

def periodic_tasks():
    """Ejecuta tareas periódicas"""
    while True:
        try:
            # Limpiar información de tareas completadas
            tasks_cleaned = cleanup_completed_tasks(max_age_hours=config.MAX_FILE_AGE_HOURS)
            if tasks_cleaned > 0:
                logger.info(f"Limpieza periódica: {tasks_cleaned} tareas eliminadas del registro")
            
            # Monitorear y registrar uso de recursos
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            if cpu_usage > 90 or memory.percent > 90:
                logger.warning(f"Uso alto de recursos - CPU: {cpu_usage}%, Memoria: {memory.percent}%")
            
        except Exception as e:
            logger.error(f"Error en tareas periódicas: {str(e)}")
        
        # Esperar antes de la próxima ejecución (15 minutos)
        time.sleep(15 * 60)

def initialize_services():
    """Inicializa servicios necesarios"""
    try:
        # Asegurar que existen directorios necesarios
        os.makedirs(config.STORAGE_PATH, exist_ok=True)
        os.makedirs(config.TEMP_DIR, exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Registrar información del sistema
        logger.info(f"Sistema: {platform.system()} {platform.release()}")
        logger.info(f"Python: {platform.python_version()}")
        cpu_count = os.cpu_count() or 0
        memory = psutil.virtual_memory()
        logger.info(f"Recursos: {cpu_count} CPUs, {memory.total / (1024**3):.1f} GB RAM")
        
        # Iniciar procesadores de cola
        start_queue_processors(num_workers=worker_count)
        logger.info(f"Iniciados {worker_count} procesadores de cola")
        
        # Iniciar servicio de limpieza
        cleanup_service.start()
        logger.info(f"Servicio de limpieza iniciado (intervalo: {cleanup_interval} minutos)")
        
        # Iniciar thread para tareas periódicas
        periodic_thread = threading.Thread(target=periodic_tasks, daemon=True)
        periodic_thread.start()
        logger.info("Tareas periódicas iniciadas")
        
        # Realizar limpieza inicial
        files_removed, bytes_freed = cleanup_service.run_now()
        if files_removed > 0:
            logger.info(f"Limpieza inicial: eliminados {files_removed} archivos ({bytes_freed/1024/1024:.2f} MB)")
        
        logger.info(f"VideoAPI v{get_version_info()['version']} inicializada correctamente")
        
    except Exception as e:
        logger.error(f"Error inicializando servicios: {str(e)}")
        raise

# Inicialización antes de primera solicitud
@app.before_first_request
def before_first_request():
    """Ejecutado antes de la primera solicitud"""
    initialize_services()

# Manejo de señales para shutdown limpio
def graceful_shutdown(signal_num, frame):
    """Realiza un shutdown limpio cuando se recibe una señal"""
    logger.info(f"Recibida señal {signal_num}, realizando shutdown limpio...")
    
    # Detener servicio de limpieza
    if 'cleanup_service' in globals():
        cleanup_service.stop()
        logger.info("Servicio de limpieza detenido")
    
    # Limpiar archivos temporales
    try:
        temp_files = [f for f in os.listdir(config.TEMP_DIR) if os.path.isfile(os.path.join(config.TEMP_DIR, f))]
        for temp_file in temp_files:
            os.remove(os.path.join(config.TEMP_DIR, temp_file))
        logger.info(f"Limpiados {len(temp_files)} archivos temporales")
    except Exception as e:
        logger.error(f"Error limpiando archivos temporales: {str(e)}")
    
    logger.info("Shutdown completado")
    os._exit(0)

# Registrar manejadores de señales
try:
    import signal
    signal.signal(signal.SIGTERM, graceful_shutdown)
    signal.signal(signal.SIGINT, graceful_shutdown)
except (ImportError, AttributeError):
    # Windows no tiene SIGTERM
    pass

# Punto de entrada para ejecución directa
if __name__ == '__main__':
    # En modo desarrollo, inicializar servicios inmediatamente
    initialize_services()
    
    # Ejecutar aplicación en modo desarrollo
    debug_mode = config.LOG_LEVEL == 'DEBUG'
    app.run(host='0.0.0.0', port=8080, debug=debug_mode)

# Agregar estas líneas a app.py después de importar Flask
from flask_swagger_ui import get_swaggerui_blueprint

# Después de crear la aplicación Flask, añadir estas líneas:
SWAGGER_URL = '/api/docs'
API_URL = '/static/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Video Processing API"
    }
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)
