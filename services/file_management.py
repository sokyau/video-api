import os
import uuid
import logging
import requests
import tempfile
import hashlib
import socket
import ipaddress
import magic
import time
import shutil
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional, List, Tuple, Dict, Union, Any
import config

logger = logging.getLogger(__name__)

# Constantes para configuración
MAX_DOWNLOAD_SIZE = getattr(config, 'MAX_DOWNLOAD_SIZE', 1024 * 1024 * 1024)  # 1GB por defecto
DOWNLOAD_TIMEOUT = getattr(config, 'DOWNLOAD_TIMEOUT', 300)  # 5 minutos por defecto
MAX_RETRIES = getattr(config, 'DOWNLOAD_MAX_RETRIES', 3)  # Número máximo de reintentos
ALLOWED_MIME_TYPES = {
    'video': ['video/mp4', 'video/avi', 'video/x-msvideo', 'video/quicktime', 'video/x-matroska', 'video/webm'],
    'audio': ['audio/mpeg', 'audio/x-wav', 'audio/ogg', 'audio/flac', 'audio/aac', 'audio/mp4'],
    'image': ['image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/webp']
}

def validate_url(url: str) -> str:
    """
    Valida una URL para prevenir ataques SSRF
    
    Args:
        url (str): URL a validar
    
    Returns:
        str: URL validada
        
    Raises:
        ValueError: Si la URL no es válida o potencialmente peligrosa
    """
    try:
        # Validar formato básico
        parsed_url = urlparse(url)
        
        # Verificar esquema
        if parsed_url.scheme not in ['http', 'https']:
            raise ValueError(f"Esquema de URL no permitido: {parsed_url.scheme}. Solo se permiten http y https.")
            
        # Verificar que contiene hostname
        if not parsed_url.netloc:
            raise ValueError("URL inválida: falta hostname")
            
        # Verificar que no es una IP interna/privada
        hostname = parsed_url.netloc.split(':')[0]  # Eliminar puerto si existe
        
        # Verificar si es una dirección IP
        try:
            ip = ipaddress.ip_address(hostname)
            # Verificar si es una IP privada, loopback, link-local, etc.
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
                raise ValueError(f"No se permite acceder a redes privadas/internas: {hostname}")
        except ValueError:
            # No es una IP, intentar resolver el hostname
            try:
                ip_address = socket.gethostbyname(hostname)
                ip = ipaddress.ip_address(ip_address)
                if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
                    raise ValueError(f"El hostname {hostname} resuelve a una IP privada/interna: {ip_address}")
            except socket.gaierror:
                logger.warning(f"No se pudo resolver el hostname: {hostname}")
                # Continuamos, ya que puede ser un error temporal o un dominio que no existe
        
        # URL parece segura
        return url
        
    except Exception as e:
        logger.error(f"Error validando URL {url}: {str(e)}")
        raise ValueError(f"URL inválida o insegura: {str(e)}")

def generate_temp_filename(prefix: str = "", suffix: str = "") -> str:
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

def download_file(url: str, target_dir: Optional[str] = None, 
                  filename: Optional[str] = None, 
                  max_size: Optional[int] = None,
                  validate_mime: bool = True) -> str:
    """
    Descarga un archivo desde una URL con validación mejorada y control de tamaño
    
    Args:
        url (str): URL del archivo a descargar
        target_dir (str, optional): Directorio donde guardar el archivo
        filename (str, optional): Nombre personalizado para el archivo
        max_size (int, optional): Tamaño máximo permitido en bytes
        validate_mime (bool): Si se debe validar el tipo MIME del archivo
    
    Returns:
        str: Ruta al archivo descargado
    
    Raises:
        ValueError: Si el archivo excede el tamaño máximo o el tipo MIME no es válido
        requests.RequestException: Si ocurre un error durante la descarga
    """
    # Validar URL para prevenir SSRF
    url = validate_url(url)
    
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
            # Intentar detectar extensión basada en Content-Type
            try:
                head_response = requests.head(url, timeout=10)
                content_type = head_response.headers.get('Content-Type', '')
                extension = get_extension_from_content_type(content_type)
                if extension:
                    filename = f"{uuid.uuid4()}{extension}"
                else:
                    filename = f"{uuid.uuid4()}.tmp"
            except Exception:
                filename = f"{uuid.uuid4()}.tmp"
    
    file_path = os.path.join(target_dir, filename)
    temp_file_path = f"{file_path}.part"  # Archivo temporal durante la descarga
    
    # Establecer tamaño máximo
    if max_size is None:
        max_size = MAX_DOWNLOAD_SIZE
    
    # Implementar reintentos para errores temporales
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            # Descargar archivo en bloques con control de tamaño
            with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
                response.raise_for_status()
                
                # Verificar Content-Length si está disponible
                content_length = response.headers.get('Content-Length')
                if content_length and int(content_length) > max_size:
                    raise ValueError(f"El archivo es demasiado grande: {int(content_length)} bytes (máximo: {max_size} bytes)")
                
                # Verificar Content-Type si validate_mime es True
                if validate_mime:
                    content_type = response.headers.get('Content-Type', '')
                    if not is_valid_content_type(content_type):
                        raise ValueError(f"Tipo de contenido no permitido: {content_type}")
                
                downloaded_size = 0
                start_time = time.time()
                
                with open(temp_file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            downloaded_size += len(chunk)
                            if downloaded_size > max_size:
                                f.close()
                                os.remove(temp_file_path)
                                raise ValueError(f"Archivo demasiado grande (límite: {max_size} bytes)")
                            f.write(chunk)
                
                # Verificar el tipo MIME real del archivo
                if validate_mime and os.path.exists(temp_file_path):
                    detected_mime = detect_mime_type(temp_file_path)
                    if not is_valid_content_type(detected_mime):
                        os.remove(temp_file_path)
                        raise ValueError(f"Tipo de archivo no permitido: {detected_mime}")
                
                # Si todo está bien, mover el archivo temporal al destino final
                shutil.move(temp_file_path, file_path)
                
                download_time = time.time() - start_time
                download_speed = downloaded_size / (download_time * 1024)  # KB/s
                
                logger.info(f"Archivo descargado exitosamente: {url} -> {file_path} ({downloaded_size/1024:.2f} KB, {download_speed:.2f} KB/s)")
                
                return file_path
        
        except (requests.RequestException, ConnectionError, TimeoutError) as e:
            retry_count += 1
            logger.warning(f"Error temporal descargando archivo (intento {retry_count}/{MAX_RETRIES}): {str(e)}")
            
            # Limpiar archivo parcial si existe
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            
            if retry_count >= MAX_RETRIES:
                logger.error(f"Falló la descarga después de {MAX_RETRIES} intentos: {url}")
                raise
            
            # Esperar antes de reintentar con backoff exponencial
            time.sleep(2 ** retry_count)
        
        except Exception as e:
            logger.error(f"Error descargando archivo desde {url}: {str(e)}")
            
            # Limpiar archivos parciales
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if os.path.exists(file_path):
                os.remove(file_path)
            
            raise

def detect_mime_type(file_path: str) -> str:
    """
    Detecta el tipo MIME real de un archivo
    
    Args:
        file_path (str): Ruta al archivo
    
    Returns:
        str: Tipo MIME del archivo
    """
    try:
        mime = magic.Magic(mime=True)
        return mime.from_file(file_path)
    except Exception as e:
        logger.error(f"Error detectando tipo MIME: {str(e)}")
        # Fallback a detección por extensión
        ext = get_file_extension(file_path)
        if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            return 'video/mp4'
        elif ext in ['.mp3', '.wav', '.ogg', '.flac']:
            return 'audio/mpeg'
        elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
            return 'image/jpeg'
        return 'application/octet-stream'

def is_valid_content_type(content_type: str) -> bool:
    """
    Verifica si un tipo de contenido es permitido
    
    Args:
        content_type (str): Tipo MIME a verificar
    
    Returns:
        bool: True si es permitido, False en caso contrario
    """
    # Lista completa de tipos permitidos
    all_allowed_types = []
    for category in ALLOWED_MIME_TYPES.values():
        all_allowed_types.extend(category)
    
    # Si está vacío, permitir cualquier tipo (más inseguro)
    if not all_allowed_types:
        return True
    
    # Verificar contra la lista de tipos permitidos
    for allowed_type in all_allowed_types:
        if content_type.startswith(allowed_type):
            return True
    
    return False

def get_extension_from_content_type(content_type: str) -> str:
    """
    Obtiene la extensión de archivo apropiada para un tipo MIME
    
    Args:
        content_type (str): Tipo MIME
    
    Returns:
        str: Extensión de archivo con punto o cadena vacía si no se reconoce
    """
    mime_to_ext = {
        'video/mp4': '.mp4',
        'video/x-msvideo': '.avi',
        'video/quicktime': '.mov',
        'video/x-matroska': '.mkv',
        'video/webm': '.webm',
        'audio/mpeg': '.mp3',
        'audio/x-wav': '.wav',
        'audio/ogg': '.ogg',
        'audio/flac': '.flac',
        'audio/aac': '.aac',
        'audio/mp4': '.m4a',
        'image/jpeg': '.jpg',
        'image/png': '.png',
        'image/gif': '.gif',
        'image/bmp': '.bmp',
        'image/webp': '.webp'
    }
    
    # Buscar tipo exacto
    if content_type in mime_to_ext:
        return mime_to_ext[content_type]
    
    # Buscar tipo parcial
    for mime, ext in mime_to_ext.items():
        if content_type.startswith(mime):
            return ext
    
    return ''

def ensure_directory(directory: str) -> None:
    """
    Asegura que un directorio existe, creándolo si es necesario
    
    Args:
        directory (str): Ruta del directorio a verificar/crear
    """
    try:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Directorio asegurado: {directory}")
    except Exception as e:
        logger.error(f"Error creando directorio {directory}: {str(e)}")
        raise

def get_file_extension(file_path: str) -> str:
    """
    Obtiene la extensión de un archivo
    
    Args:
        file_path (str): Ruta del archivo
    
    Returns:
        str: Extensión del archivo (con punto)
    """
    _, extension = os.path.splitext(file_path)
    return extension.lower()

def get_file_name_without_extension(file_path: str) -> str:
    """
    Obtiene el nombre del archivo sin extensión
    
    Args:
        file_path (str): Ruta del archivo
    
    Returns:
        str: Nombre del archivo sin extensión
    """
    basename = os.path.basename(file_path)
    return os.path.splitext(basename)[0]

def create_temp_file(content: Optional[bytes] = None, suffix: Optional[str] = None) -> str:
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
        logger.error(f"Error creando archivo temporal: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    
    logger.debug(f"Archivo temporal creado: {temp_path}")
    return temp_path

def is_video_file(file_path: str) -> bool:
    """
    Verifica si un archivo es un video basado en su extensión y/o tipo MIME
    
    Args:
        file_path (str): Ruta del archivo
    
    Returns:
        bool: True si es un archivo de video, False en caso contrario
    """
    # Verificar por extensión primero (más rápido)
    video_extensions = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm']
    ext = get_file_extension(file_path)
    
    if ext in video_extensions:
        return True
    
    # Si hay ambigüedad, verificar por tipo MIME (más preciso pero más lento)
    try:
        mime_type = detect_mime_type(file_path)
        return mime_type.startswith('video/')
    except Exception:
        # Si falla la detección MIME, confiar solo en la extensión
        return False

def is_audio_file(file_path: str) -> bool:
    """
    Verifica si un archivo es un audio basado en su extensión y/o tipo MIME
    
    Args:
        file_path (str): Ruta del archivo
    
    Returns:
        bool: True si es un archivo de audio, False en caso contrario
    """
    audio_extensions = ['.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a']
    ext = get_file_extension(file_path)
    
    if ext in audio_extensions:
        return True
    
    # Verificar por tipo MIME si hay dudas
    try:
        mime_type = detect_mime_type(file_path)
        return mime_type.startswith('audio/')
    except Exception:
        return False

def is_image_file(file_path: str) -> bool:
    """
    Verifica si un archivo es una imagen basado en su extensión y/o tipo MIME
    
    Args:
        file_path (str): Ruta del archivo
    
    Returns:
        bool: True si es un archivo de imagen, False en caso contrario
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    ext = get_file_extension(file_path)
    
    if ext in image_extensions:
        return True
    
    # Verificar por tipo MIME si hay dudas
    try:
        mime_type = detect_mime_type(file_path)
        return mime_type.startswith('image/')
    except Exception:
        return False

def calculate_file_hash(file_path: str, algorithm: str = 'sha256') -> str:
    """
    Calcula el hash de un archivo
    
    Args:
        file_path (str): Ruta al archivo
        algorithm (str): Algoritmo de hash (md5, sha1, sha256)
    
    Returns:
        str: Hash del archivo en formato hexadecimal
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
    
    hash_algorithms = {
        'md5': hashlib.md5,
        'sha1': hashlib.sha1,
        'sha256': hashlib.sha256
    }
    
    if algorithm not in hash_algorithms:
        raise ValueError(f"Algoritmo de hash no soportado: {algorithm}")
    
    hash_func = hash_algorithms[algorithm]()
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()

def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Obtiene información detallada sobre un archivo
    
    Args:
        file_path (str): Ruta al archivo
    
    Returns:
        dict: Información del archivo (tamaño, tipo, hash, etc.)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
    
    file_stat = os.stat(file_path)
    
    # Detectar tipo MIME
    mime_type = detect_mime_type(file_path)
    
    # Categorizar archivo
    file_type = 'unknown'
    if mime_type.startswith('video/'):
        file_type = 'video'
    elif mime_type.startswith('audio/'):
        file_type = 'audio'
    elif mime_type.startswith('image/'):
        file_type = 'image'
    
    return {
        'path': file_path,
        'name': os.path.basename(file_path),
        'size': file_stat.st_size,
        'size_human': format_size(file_stat.st_size),
        'created': file_stat.st_ctime,
        'modified': file_stat.st_mtime,
        'mime_type': mime_type,
        'type': file_type,
        'extension': get_file_extension(file_path)
    }

def format_size(size_bytes: int) -> str:
    """
    Formatea un tamaño en bytes a formato legible
    
    Args:
        size_bytes (int): Tamaño en bytes
    
    Returns:
        str: Tamaño formateado (ej. '4.2 MB')
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024 or unit == 'TB':
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024

def safe_delete_file(file_path: str) -> bool:
    """
    Elimina un archivo de forma segura, manejando errores
    
    Args:
        file_path (str): Ruta al archivo a eliminar
    
    Returns:
        bool: True si se eliminó correctamente, False en caso contrario
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Archivo eliminado: {file_path}")
            return True
        return False
    except Exception as e:
        logger.error(f"Error eliminando archivo {file_path}: {str(e)}")
        return False

def cleanup_temp_files(max_age_hours: int = 24, directory: Optional[str] = None) -> Tuple[int, int]:
    """
    Limpia archivos temporales más antiguos que max_age_hours
    
    Args:
        max_age_hours (int): Edad máxima en horas
        directory (str, optional): Directorio a limpiar (por defecto config.TEMP_DIR)
    
    Returns:
        tuple: (Número de archivos eliminados, Bytes liberados)
    """
    if directory is None:
        directory = config.TEMP_DIR
    
    if not os.path.exists(directory):
        logger.warning(f"El directorio de limpieza no existe: {directory}")
        return 0, 0
    
    max_age_seconds = max_age_hours * 3600
    current_time = time.time()
    
    files_removed = 0
    bytes_freed = 0
    
    for root, _, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            
            # Ignorar directorios
            if os.path.isdir(file_path):
                continue
                
            # Verificar edad del archivo
            try:
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > max_age_seconds:
                    file_size = os.path.getsize(file_path)
                    if safe_delete_file(file_path):
                        files_removed += 1
                        bytes_freed += file_size
            except Exception as e:
                logger.error(f"Error procesando archivo para limpieza {file_path}: {str(e)}")
    
    if files_removed > 0:
        logger.info(f"Limpieza: eliminados {files_removed} archivos ({bytes_freed/1024/1024:.2f} MB)")
    
    return files_removed, bytes_freed

def download_files_batch(urls: List[str], target_dir: Optional[str] = None) -> List[str]:
    """
    Descarga múltiples archivos en paralelo
    
    Args:
        urls (list): Lista de URLs para descargar
        target_dir (str, optional): Directorio destino
    
    Returns:
        list: Lista de rutas a los archivos descargados
    """
    if target_dir is None:
        target_dir = config.TEMP_DIR
    
    ensure_directory(target_dir)
    
    downloaded_files = []
    failed_downloads = []
    
    for url in urls:
        try:
            file_path = download_file(url, target_dir)
            downloaded_files.append(file_path)
        except Exception as e:
            logger.error(f"Error descargando archivo batch {url}: {str(e)}")
            failed_downloads.append(url)
    
    if failed_downloads:
        logger.warning(f"Fallaron {len(failed_downloads)} descargas de {len(urls)}")
    
    return downloaded_files

def verify_file_integrity(file_path: str, expected_hash: Optional[str] = None, algorithm: str = 'sha256') -> bool:
    """
    Verifica la integridad de un archivo
    
    Args:
        file_path (str): Ruta al archivo
        expected_hash (str, optional): Hash esperado para comparar
        algorithm (str): Algoritmo de hash (md5, sha1, sha256)
    
    Returns:
        bool: True si la verificación fue exitosa
    """
    if not os.path.exists(file_path):
        logger.error(f"Archivo no encontrado para verificación: {file_path}")
        return False
    
    # Verificar que el archivo no está vacío
    if os.path.getsize(file_path) == 0:
        logger.error(f"Archivo vacío: {file_path}")
        return False
    
    # Si se proporcionó un hash esperado, comparar
    if expected_hash:
        actual_hash = calculate_file_hash(file_path, algorithm)
        if actual_hash.lower() != expected_hash.lower():
            logger.error(f"Verificación de hash fallida para {file_path}. Esperado: {expected_hash}, Actual: {actual_hash}")
            return False
    
    # Intentar abrir el archivo para verificar que no está corrupto
    try:
        with open(file_path, 'rb') as f:
            # Leer primeros bytes para verificar que el archivo se puede leer
            f.read(1024)
        return True
    except Exception as e:
        logger.error(f"Error verificando integridad del archivo {file_path}: {str(e)}")
        return False
