import os
import uuid
import logging
import requests
import tempfile
import hashlib
import socket
import ipaddress
import time
import shutil
import mimetypes
import re
from pathlib import Path
from urllib.parse import urlparse, unquote
from typing import Optional, List, Tuple, Dict, Union, Any
from concurrent.futures import ThreadPoolExecutor
import config

logger = logging.getLogger(__name__)

# Constantes para configuración
MAX_DOWNLOAD_SIZE = getattr(config, 'MAX_DOWNLOAD_SIZE', 1024 * 1024 * 1024)  # 1GB por defecto
DOWNLOAD_TIMEOUT = getattr(config, 'DOWNLOAD_TIMEOUT', 300)  # 5 minutos por defecto
MAX_RETRIES = getattr(config, 'DOWNLOAD_MAX_RETRIES', 3)  # Número máximo de reintentos
FILE_CACHE_DIR = getattr(config, 'FILE_CACHE_DIR', os.path.join(config.TEMP_DIR, 'cache'))
CACHE_MAX_AGE = getattr(config, 'CACHE_MAX_AGE', 3600)  # 1 hora en segundos
DOWNLOAD_CHUNK_SIZE = 8192  # 8KB por chunk, ajustable según la red

# Crear directorio de caché si no existe
os.makedirs(FILE_CACHE_DIR, exist_ok=True)

# Lista de tipos MIME permitidos
ALLOWED_MIME_TYPES = {
    'video': [
        'video/mp4', 'video/x-msvideo', 'video/mpeg', 'video/quicktime', 
        'video/x-matroska', 'video/webm', 'video/x-flv', 'video/3gpp'
    ],
    'audio': [
        'audio/mpeg', 'audio/x-wav', 'audio/ogg', 'audio/flac', 'audio/aac', 
        'audio/mp4', 'audio/x-m4a', 'audio/webm', 'audio/x-ms-wma'
    ],
    'image': [
        'image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/webp', 
        'image/tiff', 'image/svg+xml', 'image/x-icon'
    ],
    'document': [
        'application/pdf', 'text/plain', 'application/json', 'text/html',
        'text/xml', 'application/xml', 'application/x-subrip', 'text/vtt'
    ]
}

# Lista de dominios para redes privadas/locales
PRIVATE_NETWORK_PATTERNS = [
    r'^10\.\d+\.\d+\.\d+$',           # 10.0.0.0/8
    r'^172\.(1[6-9]|2[0-9]|3[0-1])\.\d+\.\d+$',  # 172.16.0.0/12
    r'^192\.168\.\d+\.\d+$',          # 192.168.0.0/16
    r'^127\.\d+\.\d+\.\d+$',          # 127.0.0.0/8
    r'^169\.254\.\d+\.\d+$',          # 169.254.0.0/16
    r'^fc00:',                        # fc00::/7
    r'^fe80:',                        # fe80::/10
    r'^::1$',                         # localhost
    r'^[fF][dD]',                     # fd00::/8
]

# Cache de archivos en memoria (hash -> ruta)
file_cache = {}
file_cache_lock = threading.RLock()

class FileValidationError(Exception):
    """Excepción para errores de validación de archivos"""
    pass

class DownloadError(Exception):
    """Excepción para errores de descarga"""
    pass

class URLValidationError(Exception):
    """Excepción para errores de validación de URL"""
    pass

def validate_url(url: str) -> str:
    """
    Valida una URL para prevenir ataques SSRF
    
    Args:
        url (str): URL a validar
    
    Returns:
        str: URL validada
        
    Raises:
        URLValidationError: Si la URL no es válida o potencialmente peligrosa
    """
    try:
        # Validar formato básico
        if not isinstance(url, str) or not url:
            raise URLValidationError("La URL no puede estar vacía o no ser una cadena de texto")
            
        # Decodificar URL para evitar evasión de filtros
        decoded_url = unquote(url)
        
        # Verificar esquema
        parsed_url = urlparse(decoded_url)
        
        if parsed_url.scheme not in ['http', 'https']:
            raise URLValidationError(f"Esquema de URL no permitido: {parsed_url.scheme}. Solo se permiten http y https.")
            
        # Verificar que contiene hostname
        if not parsed_url.netloc:
            raise URLValidationError("URL inválida: falta hostname")
        
        # Extraer hostname limpio (eliminar puerto si existe)
        hostname = parsed_url.netloc.split(':')[0]
        
        # Verificar formatos no permitidos
        disallowed_hostnames = [
            'localhost', '127.0.0.1', '0.0.0.0', '::1',
            '[::1]', '[0:0:0:0:0:0:0:1]'
        ]
        
        if hostname.lower() in disallowed_hostnames:
            raise URLValidationError(f"Hostname no permitido: {hostname}")
        
        # Verificar direcciones IP locales/privadas
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
                raise URLValidationError(f"No se permite acceder a redes privadas/internas: {hostname}")
        except ValueError:
            # No es una IP, intentar resolver el hostname
            try:
                # Verificar patrones de redes privadas conocidos
                for pattern in PRIVATE_NETWORK_PATTERNS:
                    if re.match(pattern, hostname):
                        raise URLValidationError(f"Hostname con patrón de red privada no permitido: {hostname}")
                
                # Resolver dominio a IP
                ip_addresses = socket.getaddrinfo(hostname, None)
                
                for family, _, _, _, sockaddr in ip_addresses:
                    # Extraer IP del sockaddr
                    if family == socket.AF_INET:
                        ip_str = sockaddr[0]
                    elif family == socket.AF_INET6:
                        ip_str = sockaddr[0]
                    else:
                        continue
                    
                    ip = ipaddress.ip_address(ip_str)
                    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
                        raise URLValidationError(f"El hostname {hostname} resuelve a una IP privada/interna: {ip_str}")
            
            except socket.gaierror:
                # No se pudo resolver, es posible que el dominio no exista
                # No bloqueamos, ya que la conexión fallará más tarde
                logger.warning(f"No se pudo resolver hostname: {hostname}")
        
        # Verificar caracteres sospechosos
        suspicious_chars = ['@', '..', '\\', '\r', '\n', '\t', '\0']
        if any(char in decoded_url for char in suspicious_chars):
            raise URLValidationError(f"URL contiene caracteres sospechosos: {decoded_url}")
        
        # URL parece segura
        return url
        
    except URLValidationError as e:
        # Reenviar excepción específica
        raise
    except Exception as e:
        logger.error(f"Error validando URL {url}: {str(e)}")
        raise URLValidationError(f"URL inválida o insegura: {str(e)}")

def generate_cache_key(url: str) -> str:
    """
    Genera una clave de caché única para una URL
    
    Args:
        url (str): URL para la que generar clave
    
    Returns:
        str: Clave de caché (hash SHA-256)
    """
    return hashlib.sha256(url.encode('utf-8')).hexdigest()

def get_cached_file(url: str) -> Optional[str]:
    """
    Intenta obtener un archivo de la caché
    
    Args:
        url (str): URL del archivo
    
    Returns:
        Optional[str]: Ruta al archivo en caché, o None si no está cacheado
    """
    cache_key = generate_cache_key(url)
    
    with file_cache_lock:
        # Verificar en memoria
        if cache_key in file_cache:
            file_path = file_cache[cache_key]
            
            # Verificar que el archivo aún existe
            if os.path.exists(file_path):
                # Verificar que no es demasiado viejo
                mtime = os.path.getmtime(file_path)
                if time.time() - mtime <= CACHE_MAX_AGE:
                    logger.debug(f"Archivo cacheado encontrado para {url}")
                    return file_path
                else:
                    # Archivo expirado, eliminar de caché
                    logger.debug(f"Archivo cacheado expirado para {url}")
                    del file_cache[cache_key]
                    try:
                        os.remove(file_path)
                    except:
                        pass
            else:
                # Archivo no encontrado, eliminar de caché
                del file_cache[cache_key]
    
    # No encontrado en caché o expirado
    return None

def cache_file(url: str, file_path: str) -> None:
    """
    Añade un archivo a la caché
    
    Args:
        url (str): URL del archivo
        file_path (str): Ruta al archivo local
    """
    cache_key = generate_cache_key(url)
    
    with file_cache_lock:
        # Generar nombre en caché
        cache_filename = f"{cache_key}{get_file_extension(file_path)}"
        cache_path = os.path.join(FILE_CACHE_DIR, cache_filename)
        
        # Copiar archivo a la caché
        try:
            shutil.copy2(file_path, cache_path)
            file_cache[cache_key] = cache_path
            logger.debug(f"Archivo añadido a caché: {url} -> {cache_path}")
        except Exception as e:
            logger.error(f"Error cacheando archivo {url}: {str(e)}")

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
                  validate_mime: bool = True,
                  use_cache: bool = True) -> str:
    """
    Descarga un archivo desde una URL con validación y caché
    
    Args:
        url (str): URL del archivo a descargar
        target_dir (str, optional): Directorio donde guardar el archivo
        filename (str, optional): Nombre personalizado para el archivo
        max_size (int, optional): Tamaño máximo permitido en bytes
        validate_mime (bool): Si se debe validar el tipo MIME del archivo
        use_cache (bool): Si se debe usar/actualizar la caché
    
    Returns:
        str: Ruta al archivo descargado
    
    Raises:
        DownloadError: Si ocurre un error durante la descarga
        FileValidationError: Si el archivo no pasa la validación
        URLValidationError: Si la URL no es válida
    """
    # Validar URL para prevenir SSRF
    try:
        url = validate_url(url)
    except URLValidationError as e:
        logger.error(f"URL inválida: {url} - {str(e)}")
        raise
        
    # Intentar obtener de caché si está habilitado
    if use_cache:
        cached_path = get_cached_file(url)
        if cached_path:
            logger.info(f"Usando archivo cacheado para {url}")
            
            # Si se especifica un directorio de destino distinto a la caché
            if target_dir and target_dir != FILE_CACHE_DIR:
                # Copiar a ubicación solicitada
                if filename is None:
                    filename = os.path.basename(cached_path)
                
                target_path = os.path.join(target_dir, filename)
                shutil.copy2(cached_path, target_path)
                return target_path
            
            return cached_path
    
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
                    raise FileValidationError(f"El archivo es demasiado grande: {int(content_length)} bytes (máximo: {max_size} bytes)")
                
                # Verificar Content-Type si validate_mime es True
                if validate_mime:
                    content_type = response.headers.get('Content-Type', '')
                    if not is_valid_content_type(content_type):
                        raise FileValidationError(f"Tipo de contenido no permitido: {content_type}")
                
                downloaded_size = 0
                start_time = time.time()
                
                with open(temp_file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                        if chunk:
                            downloaded_size += len(chunk)
                            if downloaded_size > max_size:
                                f.close()
                                os.remove(temp_file_path)
                                raise FileValidationError(f"Archivo demasiado grande (límite: {max_size} bytes)")
                            f.write(chunk)
                
                # Verificar el tipo MIME real del archivo
                if validate_mime and os.path.exists(temp_file_path):
                    detected_mime = detect_mime_type(temp_file_path)
                    if not is_valid_content_type(detected_mime):
                        os.remove(temp_file_path)
                        raise FileValidationError(f"Tipo de archivo no permitido: {detected_mime}")
                
                # Si todo está bien, mover el archivo temporal al destino final
                shutil.move(temp_file_path, file_path)
                
                download_time = time.time() - start_time
                download_speed = downloaded_size / (download_time * 1024)  # KB/s
                
                logger.info(f"Archivo descargado: {url} -> {file_path} ({downloaded_size/1024:.2f} KB, {download_speed:.2f} KB/s)")
                
                # Cachar archivo si la característica está habilitada
                if use_cache and downloaded_size > 0:
                    cache_file(url, file_path)
                
                return file_path
        
        except (requests.RequestException, ConnectionError, TimeoutError) as e:
            retry_count += 1
            logger.warning(f"Error temporal descargando archivo (intento {retry_count}/{MAX_RETRIES}): {str(e)}")
            
            # Limpiar archivo parcial si existe
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            
            if retry_count >= MAX_RETRIES:
                logger.error(f"Falló la descarga después de {MAX_RETRIES} intentos: {url}")
                raise DownloadError(f"Error descargando archivo después de {MAX_RETRIES} intentos: {str(e)}")
            
            # Esperar antes de reintentar con backoff exponencial
            time.sleep(2 ** retry_count)
        
        except FileValidationError as e:
            # Pasar la excepción de validación
            logger.error(f"Error de validación para {url}: {str(e)}")
            
            # Limpiar archivos parciales
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if os.path.exists(file_path):
                os.remove(file_path)
            
            raise
            
        except Exception as e:
            logger.error(f"Error descargando archivo desde {url}: {str(e)}")
            
            # Limpiar archivos parciales
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if os.path.exists(file_path):
                os.remove(file_path)
            
            raise DownloadError(f"Error descargando archivo: {str(e)}")

def download_files_batch(urls: List[str], target_dir: Optional[str] = None, 
                         validate_mime: bool = True, max_workers: int = 4) -> List[str]:
    """
    Descarga múltiples archivos en paralelo
    
    Args:
        urls (list): Lista de URLs para descargar
        target_dir (str, optional): Directorio destino
        validate_mime (bool): Si se debe validar el tipo MIME
        max_workers (int): Número máximo de workers para descargas paralelas
        
    Returns:
        list: Lista de rutas a los archivos descargados
    """
    if target_dir is None:
        target_dir = config.TEMP_DIR
    
    ensure_directory(target_dir)
    
    # Ajustar número de workers según la cantidad de archivos
    if max_workers > len(urls):
        max_workers = len(urls)
    
    downloaded_files = []
    failed_downloads = []
    
    # Función para descargar un archivo y manejar excepciones
    def download_single_file(url):
        try:
            return download_file(url, target_dir, validate_mime=validate_mime)
        except Exception as e:
            logger.error(f"Error descargando archivo batch {url}: {str(e)}")
            failed_downloads.append((url, str(e)))
            return None
    
    # Descargar archivos en paralelo
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(download_single_file, urls))
    
    # Filtrar resultados exitosos
    downloaded_files = [path for path in results if path is not None]
    
    # Registrar estadísticas
    success_count = len(downloaded_files)
    fail_count = len(failed_downloads)
    logger.info(f"Descarga batch completada: {success_count} éxitos, {fail_count} fallos")
    
    if failed_downloads:
        logger.warning(f"Fallaron {fail_count} descargas de {len(urls)}. Primer error: {failed_downloads[0][1]}")
    
    return downloaded_files

def detect_mime_type(file_path: str) -> str:
    """
    Detecta el tipo MIME real de un archivo
    
    Args:
        file_path (str): Ruta al archivo
    
    Returns:
        str: Tipo MIME del archivo
    """
    try:
        # Intentar detectar por contenido (necesita python-magic)
        try:
            import magic
            mime = magic.Magic(mime=True)
            return mime.from_file(file_path)
        except (ImportError, AttributeError):
            # Fallback a mimetypes si python-magic no está disponible
            content_type, _ = mimetypes.guess_type(file_path)
            if content_type:
                return content_type
            
            # Detección básica por extensión
            ext = get_file_extension(file_path).lower()
            mime_mapping = {
                '.mp4': 'video/mp4',
                '.avi': 'video/x-msvideo',
                '.mov': 'video/quicktime',
                '.mkv': 'video/x-matroska',
                '.webm': 'video/webm',
                '.mp3': 'audio/mpeg',
                '.wav': 'audio/x-wav',
                '.ogg': 'audio/ogg',
                '.flac': 'audio/flac',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp',
                '.pdf': 'application/pdf',
                '.txt': 'text/plain',
                '.srt': 'application/x-subrip',
                '.vtt': 'text/vtt',
            }
            return mime_mapping.get(ext, 'application/octet-stream')
    
    except Exception as e:
        logger.error(f"Error detectando tipo MIME: {str(e)}")
        return 'application/octet-stream'

def is_valid_content_type(content_type: str) -> bool:
    """
    Verifica si un tipo de contenido es permitido
    
    Args:
        content_type (str): Tipo MIME a verificar
    
    Returns:
        bool: True si es permitido, False en caso contrario
    """
    if not content_type:
        return False
        
    # Lista completa de tipos permitidos
    all_allowed_types = []
    for category in ALLOWED_MIME_TYPES.values():
        all_allowed_types.extend(category)
    
    # Si está vacío, permitir cualquier tipo (más inseguro)
    if not all_allowed_types:
        return True
    
    # Normalizar content_type (eliminar parámetros)
    if ';' in content_type:
        content_type = content_type.split(';')[0].strip()
    
    # Verificar contra la lista de tipos permitidos
    for allowed_type in all_allowed_types:
        if content_type.lower() == allowed_type.lower() or content_type.lower().startswith(allowed_type.lower() + '+'):
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
    if ';' in content_type:
        content_type = content_type.split(';')[0].strip()
        
    mime_to_ext = {
        'video/mp4': '.mp4',
        'video/x-msvideo': '.avi',
        'video/quicktime': '.mov',
        'video/x-matroska': '.mkv',
        'video/webm': '.webm',
        'video/3gpp': '.3gp',
        'video/x-flv': '.flv',
        'audio/mpeg': '.mp3',
        'audio/x-wav': '.wav',
        'audio/wav': '.wav',
        'audio/ogg': '.ogg',
        'audio/flac': '.flac',
        'audio/aac': '.aac',
        'audio/mp4': '.m4a',
        'audio/x-m4a': '.m4a',
        'audio/webm': '.weba',
        'image/jpeg': '.jpg',
        'image/png': '.png',
        'image/gif': '.gif',
        'image/bmp': '.bmp',
        'image/webp': '.webp',
        'image/svg+xml': '.svg',
        'application/pdf': '.pdf',
        'text/plain': '.txt',
        'application/x-subrip': '.srt',
        'text/vtt': '.vtt',
        'application/json': '.json',
        'text/html': '.html',
        'text/xml': '.xml',
        'application/xml': '.xml'
    }
    
    # Buscar tipo exacto
    if content_type.lower() in mime_to_ext:
        return mime_to_ext[content_type.lower()]
    
    # Buscar tipo parcial
    for mime, ext in mime_to_ext.items():
        if content_type.lower().startswith(mime.lower()):
            return ext
    
    # Usar mimetypes como fallback
    ext = mimetypes.guess_extension(content_type)
    if ext:
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
    video_extensions = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm', '.3gp']
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
    audio_extensions = ['.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a', '.wma']
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
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', '.ico']
    ext = get_file_extension(file_path)
    
    if ext in image_extensions:
        return True
    
    # Verificar por tipo MIME si hay dudas
    try:
        mime_type = detect_mime_type(file_path)
        return mime_type.startswith('image/')
    except Exception:
        return False

def calculate_file_hash(file_path: str, algorithm: str = 'sha256', block_size: int = 65536) -> str:
    """
    Calcula el hash de un archivo de manera eficiente
    
    Args:
        file_path (str): Ruta al archivo
        algorithm (str): Algoritmo de hash (md5, sha1, sha256)
        block_size (int): Tamaño de bloque para lectura
    
    Returns:
        str: Hash del archivo en formato hexadecimal
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
    
    hash_algorithms = {
        'md5': hashlib.md5,
        'sha1': hashlib.sha1,
        'sha256': hashlib.sha256,
        'sha512': hashlib.sha512
    }
    
    if algorithm not in hash_algorithms:
        raise ValueError(f"Algoritmo de hash no soportado: {algorithm}")
    
    hash_func = hash_algorithms[algorithm]()
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(block_size), b''):
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
    elif mime_type.startswith('text/') or mime_type == 'application/pdf':
        file_type = 'document'
    
    # Calcular hash para archivos pequeños (<100MB)
    file_hash = None
    if file_stat.st_size < 100 * 1024 * 1024:
        try:
            file_hash = calculate_file_hash(file_path, algorithm='sha256')
        except Exception as e:
            logger.warning(f"No se pudo calcular hash para {file_path}: {str(e)}")
    
    return {
        'path': file_path,
        'name': os.path.basename(file_path),
        'size': file_stat.st_size,
        'size_human': format_size(file_stat.st_size),
        'created': file_stat.st_ctime,
        'modified': file_stat.st_mtime,
        'mime_type': mime_type,
        'type': file_type,
        'extension': get_file_extension(file_path),
        'hash_sha256': file_hash
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
            
            # Ignorar archivos especiales
            if filename.startswith('.'):
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

def cleanup_cache(max_age_hours: int = 24) -> Tuple[int, int]:
    """
    Limpia archivos de caché antiguos
    
    Args:
        max_age_hours (int): Edad máxima en horas
    
    Returns:
        tuple: (Número de archivos eliminados, Bytes liberados)
    """
    # Llamar a cleanup_temp_files con el directorio de caché
    return cleanup_temp_files(max_age_hours, FILE_CACHE_DIR)

# Inicialización del módulo
def init_module():
    """Inicializa el módulo"""
    # Asegurar directorio de caché
    os.makedirs(FILE_CACHE_DIR, exist_ok=True)
    
    # Inicializar tipos MIME
    mimetypes.init()
    
    # Agregar tipos MIME adicionales
    mimetypes.add_type('application/x-subrip', '.srt')
    mimetypes.add_type('text/vtt', '.vtt')
    
    logger.info(f"Módulo file_management inicializado. Directorio caché: {FILE_CACHE_DIR}")

# Inicializar al importar
init_module()
