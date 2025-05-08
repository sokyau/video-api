# --- START OF FILE config.py ---

import os
from dotenv import load_dotenv

# Cargar variables de entorno desde archivo .env si existe
load_dotenv()

# --------------------------------------------------------------------------
# --- Security Configuration ---
# --------------------------------------------------------------------------
# API key para autenticación (Required)
API_KEY = os.environ.get('API_KEY')
if not API_KEY:
    # This should ideally be logged if logging is configured,
    # but logging might not be set up when this module is imported.
    # Raising ValueError ensures the application doesn't start without API_KEY.
    print("CRITICAL ERROR: API_KEY environment variable is not set. Application cannot start.")
    raise ValueError("API_KEY environment variable is not set")

# --------------------------------------------------------------------------
# --- Storage Configuration ---
# --------------------------------------------------------------------------
# Ruta base donde se almacenan los archivos generados
STORAGE_PATH = os.environ.get('STORAGE_PATH', '/var/www/videoapi.sofe.site/storage')
# URL base pública para acceder a los archivos almacenados
BASE_URL = os.environ.get('BASE_URL', 'https://videoapi.sofe.site/storage')
# Tiempo máximo (en horas) que se conservan los archivos antes de la limpieza
MAX_FILE_AGE_HOURS = int(os.environ.get('MAX_FILE_AGE_HOURS', 6))
# Directorio temporal para archivos intermedios
TEMP_DIR = os.environ.get('TEMP_DIR', '/tmp')

# --------------------------------------------------------------------------
# --- Performance and Processing Configuration ---
# --------------------------------------------------------------------------
# Longitud máxima de la cola de procesamiento
MAX_QUEUE_LENGTH = int(os.environ.get('MAX_QUEUE_LENGTH', 100))
# Número de procesos worker para manejar tareas en paralelo
WORKER_PROCESSES = int(os.environ.get('WORKER_PROCESSES', 4))
# Número máximo de hilos que FFmpeg puede usar por tarea (0 = auto)
FFMPEG_THREADS = int(os.environ.get('FFMPEG_THREADS', 4))
# Tiempo máximo de procesamiento por tarea (en segundos)
MAX_PROCESSING_TIME = int(os.environ.get('MAX_PROCESSING_TIME', 1800))  # 30 minutos default
# Timeout específico para comandos FFmpeg (sobrescribe MAX_PROCESSING_TIME si se usa en ffmpeg_toolkit)
FFMPEG_TIMEOUT = int(os.environ.get('FFMPEG_TIMEOUT', MAX_PROCESSING_TIME))
# Usar aceleración por GPU si está disponible (la implementación específica depende del hardware/drivers)
USE_GPU_ACCELERATION = os.environ.get('USE_GPU_ACCELERATION', 'false').lower() == 'true'


# --------------------------------------------------------------------------
# --- Transcription Configuration ---
# --------------------------------------------------------------------------
# Modelo de Whisper a utilizar (e.g., tiny, base, small, medium, large)
WHISPER_MODEL = os.environ.get('WHISPER_MODEL', 'base')
# Ruta al ejecutable de Whisper CLI (si no está en el PATH)
WHISPER_CLI_PATH = os.environ.get('WHISPER_CLI_PATH', 'whisper')
# Timeout para la ejecución de Whisper (puede ser largo)
WHISPER_TIMEOUT = int(os.environ.get('WHISPER_TIMEOUT', 3600)) # 1 hora default

# --------------------------------------------------------------------------
# --- Error Handling and Monitoring Configuration ---
# --------------------------------------------------------------------------
# Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
# DSN de Sentry para monitoreo de errores (opcional)
SENTRY_DSN = os.environ.get('SENTRY_DSN', '')
# Nombre del entorno para monitoreo (e.g., production, staging, development)
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'production')
# Ruta al archivo de log específico para errores (usado por TimedRotatingFileHandler en app.py)
ERROR_LOG_FILE = os.environ.get('ERROR_LOG_FILE', 'logs/error.log') # Note: app.py currently hardcodes this path logic
# Días de retención para los archivos de log de errores rotados
ERROR_RETENTION_DAYS = int(os.environ.get('ERROR_RETENTION_DAYS', 30))
# URL de Webhook para notificaciones de error (opcional)
ERROR_WEBHOOK_URL = os.environ.get('ERROR_WEBHOOK_URL', '')

# --- Configuration Validation (Optional but Recommended) ---
# Example: Ensure LOG_LEVEL is valid
valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
if LOG_LEVEL not in valid_log_levels:
    print(f"WARNING: Invalid LOG_LEVEL '{LOG_LEVEL}' provided. Defaulting to INFO. Valid levels: {valid_log_levels}")
    LOG_LEVEL = 'INFO'

# Ensure paths are absolute if necessary, or handle relative paths carefully
STORAGE_PATH = os.path.abspath(STORAGE_PATH)
TEMP_DIR = os.path.abspath(TEMP_DIR)
# Consider validating worker counts, timeouts etc. are reasonable positive numbers

print(f"Config loaded: LOG_LEVEL={LOG_LEVEL}, WORKER_PROCESSES={WORKER_PROCESSES}, STORAGE_PATH={STORAGE_PATH}")
if SENTRY_DSN:
    print("Sentry DSN is configured.")

# --- END OF FILE config.py ---
