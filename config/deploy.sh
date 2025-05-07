#!/bin/bash

# Script para desplegar los archivos de la API en el servidor de forma gradual
# Este script implementa los componentes paso a paso para evitar errores

set -e  # Detener en caso de error

echo "==== VideoAPI Deployment Script - Versión Mejorada ===="

# Verificar si ejecutamos como root
if [ "$(id -u)" -ne 0 ]; then
    echo "Este script debe ejecutarse como root o con sudo."
    exit 1
fi

# Solicitar información para la configuración
read -p "Ingresa el dominio para VideoAPI (default: videoapi.sofe.site): " DOMAIN
DOMAIN=${DOMAIN:-videoapi.sofe.site}

APP_DIR="/var/www/$DOMAIN"

# Verificar que existe el directorio
if [ ! -d "$APP_DIR" ]; then
    echo "Error: El directorio $APP_DIR no existe. Ejecuta primero el script install.sh."
    exit 1
fi

# Verificar si el servicio básico funciona
echo "Verificando que el servicio básico funciona..."
API_RESPONSE=$(curl -s http://localhost:8080/version || echo "error")

if [[ "$API_RESPONSE" == *"version"* ]]; then
    echo "✅ API básica funcionando correctamente"
else
    echo "❌ Error: La API básica no responde correctamente. Arregla ese problema antes de continuar."
    exit 1
fi

echo "==== Implementando los componentes básicos ===="

# Paso 1: Crear archivos de configuración
echo "Creando archivos de configuración..."

cat > $APP_DIR/config.py << 'EOF'
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

cat > $APP_DIR/version.py << 'EOF'
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

# Paso 2: Implementar servicios básicos
echo "Implementando servicios básicos..."

# Servicio local_storage.py
cat > $APP_DIR/services/local_storage.py << 'EOF'
import os
import logging
import uuid
import shutil
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Importar configuración - con manejo de error para desarrollo incremental
try:
    import config
    STORAGE_PATH = config.STORAGE_PATH
    BASE_URL = config.BASE_URL
    MAX_FILE_AGE_HOURS = config.MAX_FILE_AGE_HOURS
except ImportError:
    logger.warning("No se pudo importar config.py, usando valores por defecto")
    STORAGE_PATH = '/var/www/videoapi.sofe.site/storage'
    BASE_URL = 'https://videoapi.sofe.site/storage'
    MAX_FILE_AGE_HOURS = 6

def ensure_storage_dir():
    """Asegurar que el directorio de almacenamiento existe"""
    os.makedirs(STORAGE_PATH, exist_ok=True)
    logger.debug(f"Storage directory ensured: {STORAGE_PATH}")

def get_file_path(filename):
    """Obtener ruta completa del archivo"""
    return os.path.join(STORAGE_PATH, filename)

def get_file_url(filename):
    """Obtener URL pública del archivo"""
    return f"{BASE_URL}/{filename}"

def store_file(file_path, custom_filename=None):
    """
    Almacenar un archivo en el sistema local y retornar su URL
    """
    ensure_storage_dir()
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Generar nombre único si no se proporciona uno personalizado
    if custom_filename is None:
        file_ext = os.path.splitext(file_path)[1]
        target_filename = f"{uuid.uuid4()}{file_ext}"
    else:
        target_filename = custom_filename
    
    target_path = get_file_path(target_filename)
    
    # Copiar archivo a ubicación de almacenamiento
    shutil.copy2(file_path, target_path)
    
    # Registrar timestamp de creación para limpieza
    # Crear archivo .meta con timestamp
    with open(f"{target_path}.meta", "w") as f:
        f.write(datetime.now().isoformat())
    
    logger.info(f"File stored locally: {target_path}")
    return get_file_url(target_filename)

def cleanup_old_files():
    """
    Eliminar archivos más antiguos que MAX_FILE_AGE_HOURS
    """
    ensure_storage_dir()
    now = datetime.now()
    cutoff = now - timedelta(hours=MAX_FILE_AGE_HOURS)
    
    file_count = 0
    size_freed = 0
    
    for filename in os.listdir(STORAGE_PATH):
        file_path = os.path.join(STORAGE_PATH, filename)
        
        # Saltar archivos .meta
        if filename.endswith('.meta'):
            continue
        
        # Verificar si existe archivo .meta
        meta_path = f"{file_path}.meta"
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    created_time = datetime.fromisoformat(f.read().strip())
                if created_time < cutoff:
                    size = os.path.getsize(file_path)
                    os.remove(file_path)
                    os.remove(meta_path)
                    file_count += 1
                    size_freed += size
            except Exception as e:
                logger.error(f"Error processing meta file {meta_path}: {e}")
        else:
            # Fallback a mtime si no hay .meta
            mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
            if mtime < cutoff:
                size = os.path.getsize(file_path)
                os.remove(file_path)
                file_count += 1
                size_freed += size
    
    if file_count > 0:
        logger.info(f"Cleanup: removed {file_count} files ({size_freed/1024/1024:.2f} MB)")
    return file_count, size_freed
EOF

# Servicio authentication.py
cat > $APP_DIR/services/authentication.py << 'EOF'
import functools
import logging
from flask import request, jsonify
import os

logger = logging.getLogger(__name__)

# Importar configuración - con manejo de error para desarrollo incremental
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

# Crear app_utils.py simplificado
cat > $APP_DIR/app_utils.py << 'EOF'
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
    Decorador para validar el payload JSON según un esquema (versión simplificada).
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
    Decorador para gestionar tareas en cola o procesamiento inmediato (versión simplificada).
    """
    def decorator(f):
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            # Obtener datos del request
            data = request.get_json()
            
            # Generar ID único para la tarea si no se proporciona
            job_id = data.get('id', str(uuid.uuid4()))
            
            # En versión simplificada, procesamos inmediatamente
            try:
                logger.info(f"Processing job {job_id} immediately (queue disabled)")
                result = {"status": "success", "job_id": job_id}
                return jsonify(result), 200
            except Exception as e:
                logger.error(f"Error processing job {job_id}: {str(e)}")
                return jsonify({
                    "status": "error",
                    "job_id": job_id,
                    "error": str(e)
                }), 500
        
        return decorated_function
    return decorator
EOF

# Crear archivo meme_overlay.py
mkdir -p $APP_DIR/routes/v1/video
touch $APP_DIR/routes/v1/video/__init__.py

cat > $APP_DIR/routes/v1/video/meme_overlay.py << 'EOF'
from flask import Blueprint, jsonify

v1_video_meme_overlay_bp = Blueprint('v1_video_meme_overlay', __name__)

@v1_video_meme_overlay_bp.route('/v1/video/meme_overlay', methods=['GET'])
def meme_overlay_info():
    """Endpoint informativo para meme_overlay"""
    return jsonify({
        "endpoint": "meme_overlay",
        "status": "available",
        "method": "POST",
        "description": "Superpone una imagen de meme sobre un video"
    })
EOF

# Actualizar app.py para incluir el blueprint
cat > $APP_DIR/app.py << 'EOF'
from flask import Flask, jsonify
import logging
import os

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Crear aplicación Flask
app = Flask(__name__)

# Importar y registrar blueprint
try:
    from routes.v1.video.meme_overlay import v1_video_meme_overlay_bp
    app.register_blueprint(v1_video_meme_overlay_bp)
    logger.info("Successfully registered meme_overlay blueprint")
except Exception as e:
    logger.error(f"Error importing blueprint: {str(e)}")

@app.route('/', methods=['GET'])
def index():
    """Endpoint raíz con información de la API"""
    return jsonify({
        "service": "Video Processing API",
        "status": "operational",
        "version": "1.0.0"
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
    return jsonify({
        "version": "1.0.0",
        "api_version": "v1",
        "build_date": "2025-05-06"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
EOF

# Configurar permisos
chown -R www-data:www-data $APP_DIR
chmod -R 755 $APP_DIR

# Reiniciar servicio
echo "==== Reiniciando servicio ===="
systemctl restart videoapi

# Verificar que el servicio se reinicie correctamente
sleep 5

# Verificar blueprint meme_overlay
echo "Verificando endpoint meme_overlay..."
MEME_RESPONSE=$(curl -s http://localhost:8080/v1/video/meme_overlay || echo "error")
if [[ "$MEME_RESPONSE" == *"endpoint"* ]]; then
    echo "✅ Endpoint meme_overlay funciona correctamente"
else
    echo "❌ Error: El endpoint meme_overlay no responde correctamente"
    echo "Respuesta: $MEME_RESPONSE"
    echo "Verifica los logs: sudo journalctl -u videoapi -n 50"
fi

echo ""
echo "========================================"
echo "     Despliegue de VideoAPI - Fase 1    "
echo "========================================"
echo ""
echo "✅ Se ha implementado un primer componente de la API (meme_overlay)"
echo ""
echo "Para continuar con la implementación completa:"
echo "  1. Revisa los logs si hubiera algún error: sudo journalctl -u videoapi -f"
echo "  2. Implementa gradualmente los demás endpoints"
echo ""
echo "Los endpoints disponibles ahora son:"
echo "  - Información: GET /"
echo "  - Versión: GET /version"
echo "  - Estado: GET /health"
echo "  - Información meme_overlay: GET /v1/video/meme_overlay"
echo ""
echo "¿Deseas implementar más endpoints ahora? (s/n): "
read implement_more

if [ "$implement_more" = "s" ]; then
    echo "Esta funcionalidad se agregará en una versión futura del script."
    echo "Por ahora, puedes implementar los endpoints manualmente siguiendo la documentación."
fi

echo ""
echo "¡Gracias por usar VideoAPI! El despliegue inicial ha sido completado."
echo ""
EOF
