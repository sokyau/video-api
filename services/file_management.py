import os
import uuid
import logging
import requests
import tempfile
from urllib.parse import urlparse
import config

logger = logging.getLogger(__name__)

def generate_temp_filename(prefix="", suffix=""):
    """
    Genera un nombre de archivo temporal único
    
    Args:
        prefix (str): Prefijo para el nombre del archivo
        suffix (str): Sufijo/extensión para el archivo
    
    Returns:
        str: Ruta completa al archivo temporal
    """
    unique_id = str(uuid.uuid4())
    filename = f"{prefix}{unique_id}{suffix}"
    return os.path.join(config.TEMP_DIR, filename)

def download_file(url, target_dir=None, filename=None):
    """
    Descarga un archivo desde una URL
    
    Args:
        url (str): URL del archivo a descargar
        target_dir (str, optional): Directorio donde guardar el archivo
        filename (str, optional): Nombre personalizado para el archivo
    
    Returns:
        str: Ruta al archivo descargado
    """
    if target_dir is None:
        target_dir = config.TEMP_DIR
    
    # Asegurarse que el directorio existe
    os.makedirs(target_dir, exist_ok=True)
    
    # Determinar nombre de archivo si no se proporciona
    if filename is None:
        parsed_url = urlparse(url)
        path = parsed_url.path
        filename = os.path.basename(path)
        
        # Si no se puede extraer un nombre de archivo válido de la URL
        if not filename or '.' not in filename:
            filename = f"{uuid.uuid4()}.tmp"
    
    file_path = os.path.join(target_dir, filename)
    
    try:
        # Descargar archivo en bloques
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        logger.info(f"File downloaded successfully: {url} -> {file_path}")
        return file_path
    
    except Exception as e:
        logger.error(f"Error downloading file from {url}: {str(e)}")
        # Limpiar archivo parcialmente descargado si hubo error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise

def ensure_directory(directory):
    """
    Asegura que un directorio existe, creándolo si es necesario
    
    Args:
        directory (str): Ruta del directorio a verificar/crear
    """
    os.makedirs(directory, exist_ok=True)
    logger.debug(f"Ensured directory exists: {directory}")

def get_file_extension(file_path):
    """
    Obtiene la extensión de un archivo
    
    Args:
        file_path (str): Ruta del archivo
    
    Returns:
        str: Extensión del archivo (con punto)
    """
    _, extension = os.path.splitext(file_path)
    return extension.lower()

def get_file_name_without_extension(file_path):
    """
    Obtiene el nombre del archivo sin extensión
    
    Args:
        file_path (str): Ruta del archivo
    
    Returns:
        str: Nombre del archivo sin extensión
    """
    basename = os.path.basename(file_path)
    return os.path.splitext(basename)[0]

def create_temp_file(content=None, suffix=None):
    """
    Crea un archivo temporal
    
    Args:
        content (bytes, optional): Contenido a escribir en el archivo
        suffix (str, optional): Sufijo/extensión para el archivo
    
    Returns:
        str: Ruta al archivo temporal creado
    """
    fd, temp_path = tempfile.mkstemp(suffix=suffix, dir=config.TEMP_DIR)
    
    try:
        if content is not None:
            with os.fdopen(fd, 'wb') as f:
                f.write(content)
        else:
            os.close(fd)
    except Exception as e:
        os.close(fd)
        logger.error(f"Error creating temp file: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    
    logger.debug(f"Created temporary file: {temp_path}")
    return temp_path

def is_video_file(file_path):
    """
    Verifica si un archivo es un video basado en su extensión
    
    Args:
        file_path (str): Ruta del archivo
    
    Returns:
        bool: True si es un archivo de video, False en caso contrario
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm']
    ext = get_file_extension(file_path)
    return ext in video_extensions

def is_audio_file(file_path):
    """
    Verifica si un archivo es un audio basado en su extensión
    
    Args:
        file_path (str): Ruta del archivo
    
    Returns:
        bool: True si es un archivo de audio, False en caso contrario
    """
    audio_extensions = ['.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a']
    ext = get_file_extension(file_path)
    return ext in audio_extensions

def is_image_file(file_path):
    """
    Verifica si un archivo es una imagen basado en su extensión
    
    Args:
        file_path (str): Ruta del archivo
    
    Returns:
        bool: True si es un archivo de imagen, False en caso contrario
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    ext = get_file_extension(file_path)
    return ext in image_extensions
