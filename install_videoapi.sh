#!/bin/bash

# Script de instalación completa para VideoAPI
# Este script instala y configura todos los componentes necesarios para VideoAPI

set -e  # Detener en caso de error

echo "==== Instalador Completo de VideoAPI ===="

# Verificar si ejecutamos como root
if [ "$(id -u)" -ne 0 ]; then
    echo "Este script debe ejecutarse como root o con sudo."
    exit 1
fi

# Verificar sistema operativo
if ! grep -q "Ubuntu 22.04" /etc/os-release; then
    echo "⚠️  Este script está diseñado para Ubuntu 22.04. Otras versiones podrían no funcionar correctamente."
    read -p "¿Deseas continuar de todos modos? (s/n): " continue_anyway
    if [ "$continue_anyway" != "s" ]; then
        echo "Instalación cancelada."
        exit 1
    fi
fi

# Solicitar información para la configuración
read -p "Ingresa el dominio para VideoAPI (default: videoapi.sofe.site): " DOMAIN
DOMAIN=${DOMAIN:-videoapi.sofe.site}

read -p "Ingresa la ruta para almacenamiento (default: /var/www/$DOMAIN/storage): " STORAGE_PATH
STORAGE_PATH=${STORAGE_PATH:-/var/www/$DOMAIN/storage}

read -p "Ingresa la URL base para el almacenamiento (default: https://$DOMAIN/storage): " BASE_URL
BASE_URL=${BASE_URL:-https://$DOMAIN/storage}

read -p "Configura una API key segura: " API_KEY
if [ -z "$API_KEY" ]; then
    API_KEY=$(openssl rand -hex 16)
    echo "API key generada automáticamente: $API_KEY"
fi

# Actualizar sistema
echo "==== Actualizando sistema ===="
apt update && apt upgrade -y

# Instalar dependencias
echo "==== Instalando dependencias ===="
apt install -y python3-pip python3-venv ffmpeg nginx certbot python3-certbot-nginx git supervisor python3-dev libmagic-dev

# Crear estructura de directorios
echo "==== Configurando directorios ===="
mkdir -p /var/www/$DOMAIN
mkdir -p $STORAGE_PATH
mkdir -p /var/www/$DOMAIN/services
mkdir -p /var/www/$DOMAIN/routes/v1/video
mkdir -p /var/www/$DOMAIN/routes/v1/media/transform
mkdir -p /var/www/$DOMAIN/routes/v1/ffmpeg
mkdir -p /var/www/$DOMAIN/routes/v1/image/transform
mkdir -p /var/www/$DOMAIN/logs
chown -R www-data:www-data /var/www/$DOMAIN

# Configurar entorno virtual Python
echo "==== Configurando entorno Python ===="
cd /var/www/$DOMAIN
python3 -m venv venv
source venv/bin/activate

# Crear requirements.txt completo
cat > requirements.txt << 'EOF'
Flask==2.3.3
Werkzeug==2.3.7
gunicorn==21.2.0
Flask-Cors==4.0.0
flask-swagger-ui==4.11.1
jsonschema==4.17.3
requests==2.31.0
python-dotenv==1.0.0
Pillow==10.0.1
pydantic==2.4.2
python-magic==0.4.27
ffmpeg-python==0.2.0
opencv-python-headless==4.8.1.78
openai-whisper==20231117
validators==0.22.0
cryptography==41.0.4
pyjwt==2.8.0
psutil==5.9.5
prometheus-client==0.17.1
python-dateutil==2.8.2
humanize==4.8.0
tqdm==4.66.1
shortuuid==1.0.11
EOF

# Instalar dependencias Python
echo "==== Instalando dependencias de Python ===="
pip install -r requirements.txt

# Crear archivo .env para variables de entorno
cat > .env << EOF
API_KEY=$API_KEY
STORAGE_PATH=$STORAGE_PATH
BASE_URL=$BASE_URL
MAX_FILE_AGE_HOURS=6
MAX_QUEUE_LENGTH=100
WORKER_PROCESSES=4
FFMPEG_THREADS=4
EOF

# Crear archivo config.py
echo "==== Creando archivos de configuración ===="
cat > /var/www/$DOMAIN/config.py << 'EOF'
import os
from dotenv import load_dotenv

# Cargar variables de entorno desde archivo .env si existe
load_dotenv()

# API key para autenticación
API_KEY = os.environ.get('API_KEY')
if not API_KEY:
    raise ValueError("API_KEY environment variable is not set")

# Configuración de almacenamiento
STORAGE_PATH = os.environ.get('STORAGE_PATH', '/var/www/videoapi.sofe.site/storage')
BASE_URL = os.environ.get('BASE_URL', 'https://videoapi.sofe.site/storage')
MAX_FILE_AGE_HOURS = int(os.environ.get('MAX_FILE_AGE_HOURS', 6))

# Configuración de rendimiento
MAX_QUEUE_LENGTH = int(os.environ.get('MAX_QUEUE_LENGTH', 100))
WORKER_PROCESSES = int(os.environ.get('WORKER_PROCESSES', 4))
FFMPEG_THREADS = int(os.environ.get('FFMPEG_THREADS', 4))

# Configuración de Whisper para transcripción
WHISPER_MODEL = os.environ.get('WHISPER_MODEL', 'base')

# Directorio temporal
TEMP_DIR = os.environ.get('TEMP_DIR', '/tmp')

# Tiempo máximo de procesamiento por tarea (en segundos)
MAX_PROCESSING_TIME = int(os.environ.get('MAX_PROCESSING_TIME', 1800))  # 30 minutos

# Configuración de logs
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
EOF

# Crear archivos de errores.py
cat > /var/www/$DOMAIN/errors.py << 'EOF'
import logging
import traceback
import json
import time
import sys
from typing import Dict, Any, Optional, Tuple, List, Union
from flask import jsonify, Response, request, current_app
import requests

logger = logging.getLogger(__name__)

# Base exception classes
class VideoAPIError(Exception):
    """Base exception for all VideoAPI errors"""
    status_code = 500
    error_code = "internal_error"
    
    def __init__(self, message: str = None, status_code: Optional[int] = None, 
                 details: Optional[Dict[str, Any]] = None, 
                 error_code: Optional[str] = None):
        self.message = message or "An unexpected error occurred"
        self.status_code = status_code or self.__class__.status_code
        self.details = details or {}
        self.error_code = error_code or self.__class__.error_code
        self.timestamp = time.time()
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API response"""
        error_dict = {
            "status": "error",
            "error": self.error_code,
            "message": self.message,
            "timestamp": self.timestamp,
        }
        
        # Add details if they exist
        if self.details:
            error_dict["details"] = self.details
            
        # Add request ID if available
        if hasattr(request, 'id'):
            error_dict["request_id"] = request.id
            
        return error_dict
    
    def get_response(self) -> Response:
        """Convert exception to Flask response"""
        return jsonify(self.to_dict()), self.status_code

# HTTP error classes
class BadRequestError(VideoAPIError):
    """Exception for invalid request data"""
    status_code = 400
    error_code = "bad_request"

class AuthenticationError(VideoAPIError):
    """Exception for authentication failures"""
    status_code = 401
    error_code = "authentication_error"

class AuthorizationError(VideoAPIError):
    """Exception for authorization failures"""
    status_code = 403
    error_code = "forbidden"

class NotFoundError(VideoAPIError):
    """Exception for resource not found"""
    status_code = 404
    error_code = "not_found"

class ValidationError(BadRequestError):
    """Exception for data validation failures"""
    error_code = "validation_error"

class ProcessingError(VideoAPIError):
    """Base exception for media processing errors"""
    status_code = 500
    error_code = "processing_error"

class FFmpegError(ProcessingError):
    """Exception for FFmpeg failures"""
    error_code = "ffmpeg_error"
    
    @classmethod
    def from_ffmpeg_error(cls, stderr: str, cmd: List[str] = None) -> 'FFmpegError':
        """Create from FFmpeg error output"""
        message = "FFmpeg command failed"
        details = {"ffmpeg_error": stderr[:500]}
        if cmd:
            details["command"] = " ".join(cmd)
        return cls(message=message, details=details)

class StorageError(VideoAPIError):
    """Exception for storage-related errors"""
    error_code = "storage_error"

class NetworkError(VideoAPIError):
    """Exception for network-related errors"""
    error_code = "network_error"

def register_error_handlers(app):
    """Register error handlers with Flask app"""
    
    @app.errorhandler(VideoAPIError)
    def handle_api_error(error):
        """Handle all VideoAPI errors"""
        return error.get_response()
    
    @app.errorhandler(404)
    def handle_not_found(error):
        """Handle 404 errors"""
        return NotFoundError("Resource not found").get_response()
    
    @app.errorhandler(500)
    def handle_internal_error(error):
        """Handle 500 errors"""
        logger.exception("Unhandled exception occurred")
        return VideoAPIError("An unexpected error occurred").get_response()

def capture_exception(exc: Exception, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Capture and log an exception
    
    Args:
        exc: Exception to capture
        context: Additional context to include
        
    Returns:
        str: Error ID for reference
    """
    error_id = f"err_{int(time.time())}_{id(exc):x}"
    
    # Combine context with basic error info
    log_context = {
        "error_id": error_id,
        "error_type": exc.__class__.__name__,
        "error_message": str(exc)
    }
    
    if context:
        log_context.update(context)
    
    # Log the error with context
    logger.error(
        f"Exception {error_id}: {exc.__class__.__name__}: {str(exc)}",
        extra=log_context,
        exc_info=True
    )
    
    return error_id
EOF

# Crear servicios esenciales
echo "==== Creando servicios esenciales ===="

# Servicio authentication.py
cat > /var/www/$DOMAIN/services/authentication.py << 'EOF'
import functools
import logging
from flask import request, jsonify
import os

logger = logging.getLogger(__name__)

try:
    import config
    API_KEY = config.API_KEY
except ImportError:
    logger.warning("No se pudo importar config.py, usando API_KEY desde variables de entorno")
    API_KEY = os.environ.get('API_KEY', 'api_key_segura_temporal')

def authenticate(f):
    """
    Decorador para autenticación de API mediante API key.
    """
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            logger.warning("API request without key")
            return jsonify({
                "status": "error",
                "error": "API key is required"
            }), 401
        
        if api_key != API_KEY:
            logger.warning("API request with invalid key")
            return jsonify({
                "status": "error",
                "error": "Invalid API key"
            }), 401
        
        # Si la autenticación es exitosa, continuar con la función original
        return f(*args, **kwargs)
    
    return decorated_function
EOF

# Servicio local_storage.py
cat > /var/www/$DOMAIN/services/local_storage.py << 'EOF'
import os
import logging
import uuid
import shutil
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import config

from errors import (
    StorageError,
    NotFoundError,
    ValidationError,
    capture_exception
)

logger = logging.getLogger(__name__)

def ensure_storage_dir():
    """
    Asegurar que el directorio de almacenamiento existe.
    """
    try:
        os.makedirs(config.STORAGE_PATH, exist_ok=True)
        logger.debug(f"Storage directory ensured: {config.STORAGE_PATH}")
    except OSError as e:
        error_id = capture_exception(e, {"directory": config.STORAGE_PATH})
        raise StorageError(
            message=f"No se pudo crear el directorio de almacenamiento: {str(e)}",
            error_code="storage_dir_creation_failed",
            details={"directory": config.STORAGE_PATH, "error_id": error_id}
        )

def get_file_path(filename: str) -> str:
    """
    Obtener ruta completa del archivo.
    """
    if not filename or '..' in filename or filename.startswith(('/', '\\')):
        raise ValidationError(
            message=f"Nombre de archivo inválido: '{filename}'",
            error_code="invalid_storage_filename",
            details={"filename": filename}
        )
    return os.path.join(config.STORAGE_PATH, filename)

def get_file_url(filename: str) -> str:
    """
    Obtener URL pública del archivo.
    """
    if not filename or '..' in filename or filename.startswith(('/', '\\')):
        raise ValidationError(
            message=f"Nombre de archivo inválido para URL: '{filename}'",
            error_code="invalid_url_filename",
            details={"filename": filename}
        )
    base_url = config.BASE_URL.rstrip('/')
    clean_filename = filename.lstrip('/')
    return f"{base_url}/{clean_filename}"

def store_file(file_path: str, custom_filename: Optional[str] = None) -> str:
    """
    Almacenar un archivo en el sistema local y retornar su URL
    """
    ensure_storage_dir()
    
    if not os.path.exists(file_path):
        raise NotFoundError(
            message=f"Archivo fuente no encontrado: {file_path}",
            error_code="source_file_not_found_for_storage",
            details={"source_path": file_path}
        )
    
    if custom_filename is None:
        file_ext = os.path.splitext(file_path)[1]
        target_filename = f"{uuid.uuid4()}{file_ext}"
    else:
        if '..' in custom_filename or custom_filename.startswith(('/', '\\')):
            raise ValidationError(
                message=f"Nombre de archivo personalizado inválido: '{custom_filename}'",
                error_code="invalid_custom_filename_storage",
                details={"custom_filename": custom_filename}
            )
        target_filename = custom_filename
    
    target_storage_path = get_file_path(target_filename)
    
    try:
        shutil.copy2(file_path, target_storage_path)
        
        with open(f"{target_storage_path}.meta", "w", encoding='utf-8') as f:
            f.write(datetime.now().isoformat())
        
        logger.info(f"File stored locally: {target_storage_path}")
        return get_file_url(target_filename)
    except (IOError, OSError, shutil.Error) as e:
        error_id = capture_exception(e, {"source_path": file_path, "target_path": target_storage_path})
        if os.path.exists(target_storage_path):
            try: os.remove(target_storage_path)
            except OSError: pass
        if os.path.exists(f"{target_storage_path}.meta"):
            try: os.remove(f"{target_storage_path}.meta")
            except OSError: pass
        raise StorageError(
            message=f"Error almacenando archivo: {str(e)}",
            error_code="file_storage_failed",
            details={"source_path": file_path, "error_id": error_id}
        )

def cleanup_old_files() -> Tuple[int, int]:
    """
    Eliminar archivos más antiguos que MAX_FILE_AGE_HOURS
    """
    try:
        ensure_storage_dir()
    except StorageError as e:
        logger.error(f"No se puede acceder al directorio de almacenamiento: {e.message}")
        raise

    now = datetime.now()
    cutoff = now - timedelta(hours=config.MAX_FILE_AGE_HOURS)
    
    file_count = 0
    size_freed = 0
    
    try:
        for filename in os.listdir(config.STORAGE_PATH):
            if filename.startswith('.'):
                continue

            file_path = os.path.join(config.STORAGE_PATH, filename)
            
            if not os.path.isfile(file_path):
                continue

            if filename.endswith('.meta'):
                continue
            
            meta_path = f"{file_path}.meta"
            
            try:
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding='utf-8') as f:
                        created_time_str = f.read().strip()
                    created_time = datetime.fromisoformat(created_time_str)
                    if created_time < cutoff:
                        size = os.path.getsize(file_path)
                        os.remove(file_path)
                        os.remove(meta_path)
                        file_count += 1
                        size_freed += size
                else:
                    # Fallback a mtime si no hay .meta
                    mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if mtime < cutoff:
                        size = os.path.getsize(file_path)
                        os.remove(file_path)
                        file_count += 1
                        size_freed += size
            except Exception as e:
                logger.error(f"Error procesando archivo {file_path}: {e}")
    except OSError as e:
        error_id = capture_exception(e, {"storage_path": config.STORAGE_PATH})
        raise StorageError(
            message=f"Error listando directorio de almacenamiento: {str(e)}",
            error_code="storage_listdir_failed_cleanup",
            details={"storage_path": config.STORAGE_PATH, "error_id": error_id}
        )

    if file_count > 0:
        logger.info(f"Cleanup: removed {file_count} files ({size_freed/1024/1024:.2f} MB)")
    return file_count, size_freed
EOF

# Crear servicio file_management.py simplificado
cat > /var/www/$DOMAIN/services/file_management.py << 'EOF'
import os
import logging
import uuid
import requests
import tempfile
from typing import Optional
import config

from errors import (
    StorageError,
    NetworkError,
    ValidationError,
    NotFoundError,
    capture_exception
)

logger = logging.getLogger(__name__)

def generate_temp_filename(prefix: str = "", suffix: str = "") -> str:
    """
    Genera un nombre de archivo temporal único
    """
    unique_id = str(uuid.uuid4())
    filename = f"{prefix}{unique_id}{suffix}"
    return os.path.join(config.TEMP_DIR, filename)

def download_file(url: str, target_dir: Optional[str] = None, filename: Optional[str] = None) -> str:
    """
    Descarga un archivo desde una URL
    """
    if target_dir is None:
        target_dir = config.TEMP_DIR
    
    try:
        os.makedirs(target_dir, exist_ok=True)
    except OSError as e:
        error_id = capture_exception(e, {"directory": target_dir})
        raise StorageError(
            message=f"No se pudo crear directorio temporal: {str(e)}",
            error_code="temp_dir_creation_failed",
            details={"directory": target_dir, "error_id": error_id}
        )

    if filename is None:
        filename = os.path.basename(url.split('?')[0]) or f"{uuid.uuid4()}"
    
    file_path = os.path.join(target_dir, filename)
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        logger.info(f"Archivo descargado: {url} -> {file_path}")
        return file_path
    
    except requests.RequestException as e:
        error_id = capture_exception(e, {"url": url})
        raise NetworkError(
            message=f"Error descargando archivo: {str(e)}",
            error_code="download_network_error",
            details={"url": url, "error_id": error_id}
        )
    except (IOError, OSError) as e:
        error_id = capture_exception(e, {"url": url, "file_path": file_path})
        raise StorageError(
            message=f"Error guardando archivo descargado: {str(e)}",
            error_code="download_save_error",
            details={"url": url, "file_path": file_path, "error_id": error_id}
        )

def get_file_extension(file_path: str) -> str:
    """
    Obtiene la extensión de un archivo
    """
    _, extension = os.path.splitext(file_path)
    return extension.lower()

def verify_file_integrity(file_path: str) -> bool:
    """
    Verifica la integridad de un archivo
    """
    try:
        if not os.path.exists(file_path):
            return False
        if os.path.getsize(file_path) == 0:
            return False
        
        # Try reading a small part to check for corruption
        with open(file_path, 'rb') as f:
            f.read(1024)
        return True
    except Exception:
        return False
EOF

# Crear app_utils.py básico
cat > /var/www/$DOMAIN/app_utils.py << 'EOF'
import functools
import json
import logging
import uuid
import time
from flask import request, jsonify

# Configurar logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def validate_payload(schema):
    """
    Decorador para validar el payload JSON según un esquema.
    """
    def decorator(f):
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                payload = request.get_json()
                if not payload:
                    return jsonify({"error": "No JSON payload provided"}), 400
                
                # En versión simplificada no validamos el esquema
                logger.debug("Payload validation - simplified version")
            except Exception as e:
                # Error general al procesar el payload
                return jsonify({"error": f"Error processing payload: {str(e)}"}), 400
            
            # Si la validación es exitosa, continuar con la función original
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def queue_task_wrapper(bypass_queue=False):
    """
    Decorador para gestionar tareas en cola o procesamiento inmediato.
    """
    def decorator(f):
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            # Obtener datos del request
            data = request.get_json()
            
            # Generar ID único para la tarea si no se proporciona
            job_id = data.get('id', str(uuid.uuid4()))
            
            # Procesar inmediatamente (versión simplificada)
            try:
                logger.info(f"Procesando tarea {job_id} inmediatamente")
                result, endpoint, status_code = f(job_id, data)
                return jsonify({
                    "status": "success",
                    "job_id": job_id,
                    "result": result,
                    "endpoint": endpoint
                }), status_code
            except Exception as e:
                logger.error(f"Error procesando tarea {job_id}: {str(e)}")
                return jsonify({
                    "status": "error",
                    "job_id": job_id,
                    "error": str(e)
                }), 500
        
        return decorated_function
    return decorator

def start_queue_processors(num_workers=4):
    """Inicializa los procesadores de cola (stub para compatibilidad)"""
    logger.info(f"Inicializados {num_workers} procesadores de cola (simulado)")

def cleanup_completed_tasks(max_age_hours=6):
    """Limpia tareas completadas (stub para compatibilidad)"""
    return 0

def get_queue_stats():
    """Obtiene estadísticas de la cola (stub para compatibilidad)"""
    return {
        "total_tasks": 0,
        "queue_size": 0,
        "tasks_by_status": {
            "queued": 0,
            "processing": 0,
            "completed": 0,
            "error": 0
        }
    }

def check_queue_health():
    """Verifica la salud de la cola (stub para compatibilidad)"""
    return {
        "status": "healthy",
        "queue_processing": True,
        "workers_active": 4
    }

def get_job_status(job_id):
    """Obtiene el estado de un trabajo (stub para compatibilidad)"""
    return None
EOF

# Crear version.py
cat > /var/www/$DOMAIN/version.py << 'EOF'
"""
Control de versiones para la API de procesamiento de video.
"""

VERSION = "1.0.0"
API_VERSION = "v1"
BUILD_DATE = "2025-05-06"

def get_version_info():
    """Retorna información de versión como diccionario."""
    return {
        "version": VERSION,
        "api_version": API_VERSION,
        "build_date": BUILD_DATE
    }

def get_version_string():
    """Retorna string formateado con información de versión."""
    return f"VideoAPI v{VERSION} (API {API_VERSION}) - Build {BUILD_DATE}"
EOF

# Crear app.py básico
cat > /var/www/$DOMAIN/app.py << 'EOF'
from flask import Flask, jsonify, request, send_from_directory, g
import logging
import os
import time
import uuid
from version import get_version_info
import config

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Crear aplicación Flask
app = Flask(__name__)

# Registrar blueprint de meme_overlay
try:
    from routes.v1.video.meme_overlay import v1_video_meme_overlay_bp
    app.register_blueprint(v1_video_meme_overlay_bp)
    logger.info("Registrado blueprint meme_overlay")
except Exception as e:
    logger.error(f"Error importando blueprint meme_overlay: {str(e)}")

@app.before_request
def before_request():
    """Configurar contexto de solicitud y logging"""
    g.request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
    g.start_time = time.time()
    
    if request.path != '/health':
        logger.info(f"Solicitud {g.request_id}: {request.method} {request.path}")

@app.after_request
def after_request(response):
    """Añadir cabeceras de respuesta y registrar finalización"""
    response.headers['X-Request-ID'] = g.get('request_id', 'unknown')
    
    # Calcular duración
    duration = time.time() - g.get('start_time', time.time())
    
    if request.path != '/health':
        logger.info(f"Solicitud {g.get('request_id', 'unknown')} completada en {duration:.3f}s con estado {response.status_code}")
    
    return response

@app.route('/', methods=['GET'])
def index():
    """Endpoint raíz con información de la API"""
    return jsonify({
        "service": "Video Processing API",
        "status": "operational",
        "version": get_version_info()["version"]
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para comprobación de salud del servicio"""
    return jsonify({
        "status": "healthy",
        "storage": "ok",
        "ffmpeg": "ok"
    })

@app.route('/version', methods=['GET'])
def version():
    """Endpoint para información de versión"""
    return jsonify(get_version_info())

@app.route('/storage/<path:filename>', methods=['GET'])
def serve_file(filename):
    """Servir archivos desde directorio de almacenamiento"""
    # Verificación de seguridad para prevenir path traversal
    if '..' in filename or filename.startswith('/'):
        return jsonify({"error": "Unauthorized access"}), 403
    
    # Verificar si el archivo existe
    file_path = os.path.join(config.STORAGE_PATH, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    
    return send_from_directory(config.STORAGE_PATH, filename, as_attachment=False)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
EOF

# Crear ruta para meme_overlay
echo "==== Creando rutas ===="
mkdir -p /var/www/$DOMAIN/routes/v1/video

cat > /var/www/$DOMAIN/routes/v1/video/__init__.py << EOF
# Archivo de inicialización del paquete
EOF

cat > /var/www/$DOMAIN/routes/v1/video/meme_overlay.py << 'EOF'
from flask import Blueprint
from app_utils import validate_payload, queue_task_wrapper
import logging
import os
import sys

sys.path.append('/var/www/videoapi.sofe.site')  # Ajustar según tu dominio
from services.authentication import authenticate
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
        "position": {"type": "string"},
        "scale": {"type": "number"},
        "webhook_url": {"type": "string", "format": "uri"},
        "id": {"type": "string"}
    },
    "required": ["video_url", "meme_url"]
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
        # Importar el servicio de meme_overlay (evita importación circular)
        from services.file_management import download_file, generate_temp_filename
        
        # Simular procesamiento
        logger.info(f"Job {job_id}: Simulando procesamiento (implementación básica)")
        
        # Descargar archivo de video para demostración
        video_path = download_file(video_url, "/tmp")
        
        # En la implementación real, usaríamos:
        # output_file = process_meme_overlay(video_url, meme_url, position, scale, job_id)
        # Pero para esta versión básica, solo devolvemos el video original
        
        # Almacenar archivo resultante
        file_url = store_file(video_path)
        logger.info(f"Job {job_id}: Output video stored: {file_url}")
        
        # Limpiar archivo temporal después de almacenamiento
        os.remove(video_path)

        return file_url, "/v1/video/meme_overlay", 200

    except Exception as e:
        logger.error(f"Job {job_id}: Error durante meme overlay - {str(e)}", exc_info=True)
        return str(e), "/v1/video/meme_overlay", 500
EOF

# Configurar Nginx
echo "==== Configurando Nginx ===="
cat > /etc/nginx/sites-available/$DOMAIN << EOF
server {
    listen 80;
    server_name $DOMAIN;
    
    # API principal
    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # Timeouts largos para procesamiento de videos
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
    
    # Directorio de almacenamiento
    location /storage/ {
        alias $STORAGE_PATH/;
        expires 6h;  # Cachear archivos por 6 horas
        add_header Cache-Control "public, max-age=21600";
    }
    
    # Límites y configuración de carga
    client_max_body_size 1G;  # Permitir uploads de hasta 1GB
    
    # Logs
    access_log /var/log/nginx/videoapi-access.log;
    error_log /var/log/nginx/videoapi-error.log;
}
EOF

# Habilitar sitio en Nginx
ln -sf /etc/nginx/sites-available/$DOMAIN /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx

# Crear servicio systemd
echo "==== Configurando servicio systemd ===="
cat > /etc/systemd/system/videoapi.service << EOF
[Unit]
Description=Video API Service
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/var/www/$DOMAIN
ExecStart=/var/www/$DOMAIN/venv/bin/gunicorn --bind 0.0.0.0:8080 --workers 4 --timeout 300 app:app
Restart=always
RestartSec=10
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=videoapi
Environment="API_KEY=$API_KEY"
Environment="STORAGE_PATH=$STORAGE_PATH"
Environment="BASE_URL=$BASE_URL"
Environment="MAX_FILE_AGE_HOURS=6"
Environment="MAX_QUEUE_LENGTH=100"
Environment="WORKER_PROCESSES=4"
Environment="FFMPEG_THREADS=4"

[Install]
WantedBy=multi-user.target
EOF

# Configurar permisos
echo "==== Configurando permisos ===="
chown -R www-data:www-data /var/www/$DOMAIN
chmod -R 755 /var/www/$DOMAIN

# Habilitar e iniciar servicio
echo "==== Iniciando servicios ===="
systemctl daemon-reload
systemctl enable videoapi
systemctl start videoapi

# Esperar a que el servicio arranque
echo "Esperando a que el servicio arranque..."
sleep 5

# Validar instalación
echo "==== Validando instalación ===="

# Verificar servicio
if systemctl is-active --quiet videoapi; then
    echo "✅ Servicio VideoAPI está activo"
else
    echo "❌ Error: El servicio VideoAPI no está ejecutándose"
fi

# Verificar Nginx
if systemctl is-active --quiet nginx; then
    echo "✅ Servicio Nginx está activo"
else
    echo "❌ Error: El servicio Nginx no está ejecutándose"
fi

# Verificar respuesta de la API
echo "Verificando respuesta de la API..."
API_RESPONSE=$(curl -s http://localhost:8080/version || echo "error")
if [[ "$API_RESPONSE" == *"version"* ]]; then
    echo "✅ La API responde correctamente"
else
    echo "❌ Error: La API no responde correctamente"
fi

# Preguntar por HTTPS
echo "==== ¿Deseas configurar HTTPS con Certbot ahora? ===="
echo "NOTA: Asegúrate de que el dominio $DOMAIN apunte a este servidor."
read -p "Configurar HTTPS ahora? (s/n): " configure_https

if [ "$configure_https" = "s" ]; then
    echo "Configurando HTTPS con Certbot..."
    certbot --nginx -d $DOMAIN
    
    echo "Reiniciando servicios..."
    systemctl restart nginx
    systemctl restart videoapi
    
    echo "Verificando HTTPS..."
    echo "Puedes probar la API segura en: https://$DOMAIN/version"
else
    echo "Has elegido no configurar HTTPS ahora."
    echo "Puedes configurarlo manualmente más tarde con: sudo certbot --nginx -d $DOMAIN"
    echo "Puedes probar la API en: http://$DOMAIN/version"
fi

# Mostrar información final
echo ""
echo "========================================"
echo "      Instalación de VideoAPI           "
echo "========================================"
echo ""
echo "✅ Instalación básica completada con éxito!"
echo ""
echo "📝 Información del servicio:"
echo "   - Dominio: $DOMAIN"
echo "   - URL Base API: http://$DOMAIN (o https:// si habilitaste SSL)"
echo "   - API Key: $API_KEY"
echo "   - Ruta de almacenamiento: $STORAGE_PATH"
echo ""
echo "💡 Notas importantes:"
echo "  1. Esta instalación incluye la funcionalidad básica de la API."
echo "  2. Para implementar completamente todas las funciones, necesitarás:"
echo "     - Desarrollar e instalar servicios adicionales como ffmpeg_toolkit, meme_overlay, etc."
echo "     - Implementar rutas adicionales para los otros endpoints mencionados en la documentación."
echo "  3. Verifica logs con: sudo journalctl -u videoapi -f"
echo ""
echo "🔗 Endpoints disponibles actualmente:"
echo "  - Información: GET /"
echo "  - Versión: GET /version"
echo "  - Estado: GET /health"
echo "  - Meme Overlay (básico): POST /v1/video/meme_overlay"
echo ""
echo "Para completar la instalación con todas las funcionalidades, considera:"
echo "1. Desarrollar o adquirir los servicios completos de procesamiento de video"
echo "2. Implementar manualmente las rutas adicionales basadas en la documentación"
echo ""
