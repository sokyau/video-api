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
