# --- START OF FILE file_management.py ---

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
import threading # Added import
from pathlib import Path
from urllib.parse import urlparse, unquote
from typing import Optional, List, Tuple, Dict, Union, Any
from concurrent.futures import ThreadPoolExecutor
import config

# Assuming errors.py is in the same directory or accessible via PYTHONPATH
from errors import (
    StorageError,
    NetworkError,
    ValidationError,
    NotFoundError,
    capture_exception
)

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

# Removed old custom exceptions: FileValidationError, DownloadError, URLValidationError

def validate_url(url: str) -> str:
    """
    Valida una URL para prevenir ataques SSRF

    Args:
        url (str): URL a validar

    Returns:
        str: URL validada

    Raises:
        ValidationError: Si la URL no es válida o potencialmente peligrosa
    """
    try:
        if not isinstance(url, str) or not url:
            raise ValidationError(message="La URL no puede estar vacía o no ser una cadena de texto",
                                  error_code="invalid_url_empty",
                                  details={"url": url})

        decoded_url = unquote(url)
        parsed_url = urlparse(decoded_url)

        if parsed_url.scheme not in ['http', 'https']:
            raise ValidationError(message=f"Esquema de URL no permitido: {parsed_url.scheme}. Solo se permiten http y https.",
                                  error_code="invalid_url_scheme",
                                  details={"url": url, "scheme": parsed_url.scheme})

        if not parsed_url.netloc:
            raise ValidationError(message="URL inválida: falta hostname",
                                  error_code="invalid_url_no_hostname",
                                  details={"url": url})

        hostname = parsed_url.netloc.split(':')[0]
        disallowed_hostnames = [
            'localhost', '127.0.0.1', '0.0.0.0', '::1',
            '[::1]', '[0:0:0:0:0:0:0:1]'
        ]

        if hostname.lower() in disallowed_hostnames:
            raise ValidationError(message=f"Hostname no permitido: {hostname}",
                                  error_code="disallowed_hostname",
                                  details={"url": url, "hostname": hostname})

        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
                raise ValidationError(message=f"No se permite acceder a redes privadas/internas: {hostname}",
                                      error_code="private_ip_access",
                                      details={"url": url, "hostname": hostname, "ip": str(ip)})
        except ValueError: # Not an IP address, try to resolve
            try:
                for pattern in PRIVATE_NETWORK_PATTERNS:
                    if re.match(pattern, hostname):
                        raise ValidationError(message=f"Hostname con patrón de red privada no permitido: {hostname}",
                                              error_code="private_hostname_pattern",
                                              details={"url": url, "hostname": hostname, "pattern": pattern})

                # Resolve domain to IP
                ip_addresses = socket.getaddrinfo(hostname, None)
                for family, _, _, _, sockaddr in ip_addresses:
                    ip_str = sockaddr[0]
                    ip = ipaddress.ip_address(ip_str)
                    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
                        raise ValidationError(message=f"El hostname {hostname} resuelve a una IP privada/interna: {ip_str}",
                                              error_code="private_resolved_ip",
                                              details={"url": url, "hostname": hostname, "resolved_ip": ip_str})
            except socket.gaierror:
                logger.warning(f"No se pudo resolver hostname: {hostname}")
            except ValidationError: # Re-raise if already a ValidationError from pattern matching
                raise

        suspicious_chars = ['@', '..', '\\', '\r', '\n', '\t', '\0']
        if any(char in decoded_url for char in suspicious_chars):
            raise ValidationError(message=f"URL contiene caracteres sospechosos: {decoded_url}",
                                  error_code="suspicious_characters_in_url",
                                  details={"url": url, "decoded_url": decoded_url})
        return url

    except ValidationError: # Re-raise ValidationError as is
        raise
    except Exception as e:
        error_id = capture_exception(e, {"url": url})
        logger.error(f"Error validando URL {url}: {str(e)} (ID: {error_id})")
        raise ValidationError(message=f"URL inválida o insegura: {str(e)}",
                              error_code="url_validation_failed_generic",
                              details={"url": url, "original_error": str(e), "error_id": error_id})

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
        if cache_key in file_cache:
            file_path = file_cache[cache_key]
            if os.path.exists(file_path):
                mtime = os.path.getmtime(file_path)
                if time.time() - mtime <= CACHE_MAX_AGE:
                    logger.debug(f"Archivo cacheado encontrado para {url}")
                    return file_path
                else:
                    logger.debug(f"Archivo cacheado expirado para {url}")
                    del file_cache[cache_key]
                    try:
                        os.remove(file_path)
                    except OSError as e:
                        capture_exception(e, {"file_path": file_path, "context": "cache_cleanup_expired"})
                        logger.warning(f"Error eliminando archivo de caché expirado {file_path}: {str(e)}")
            else:
                del file_cache[cache_key]
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
        cache_filename = f"{cache_key}{get_file_extension(file_path)}"
        cache_path = os.path.join(FILE_CACHE_DIR, cache_filename)
        try:
            shutil.copy2(file_path, cache_path)
            file_cache[cache_key] = cache_path
            logger.debug(f"Archivo añadido a caché: {url} -> {cache_path}")
        except (IOError, OSError) as e:
            error_id = capture_exception(e, {"url": url, "file_path": file_path, "cache_path": cache_path})
            logger.error(f"Error cacheando archivo {url}: {str(e)} (ID: {error_id})")
            # Not raising StorageError here as caching is best-effort, but logging the error_id.

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
        NetworkError: Si ocurre un error durante la descarga.
        ValidationError: Si el archivo no pasa la validación o la URL no es válida.
        StorageError: Si hay un error de almacenamiento.
    """
    validated_url = url # Keep original url for logging/caching if needed, use validated_url for requests
    try:
        validated_url = validate_url(url)
    except ValidationError as e:
        logger.error(f"URL inválida para descarga: {url} - {e.message} (Code: {e.error_code}, Details: {e.details})")
        raise # Re-raise the ValidationError as is

    if use_cache:
        cached_path = get_cached_file(validated_url)
        if cached_path:
            logger.info(f"Usando archivo cacheado para {validated_url}")
            if target_dir and target_dir != FILE_CACHE_DIR:
                if filename is None:
                    filename = os.path.basename(cached_path)
                try:
                    target_path = os.path.join(target_dir, filename)
                    ensure_directory(target_dir) # Ensure target_dir exists before copy
                    shutil.copy2(cached_path, target_path)
                    return target_path
                except (IOError, OSError) as e:
                    error_id = capture_exception(e, {"url": validated_url, "cached_path": cached_path, "target_path": target_path if 'target_path' in locals() else None})
                    raise StorageError(
                        message=f"No se pudo copiar el archivo cacheado: {str(e)}",
                        error_code="cache_copy_error",
                        details={"url": validated_url, "cached_path": cached_path, "error_id": error_id}
                    )
            return cached_path

    if target_dir is None:
        target_dir = config.TEMP_DIR
    
    ensure_directory(target_dir) # Raises StorageError if fails

    if filename is None:
        parsed_url_obj = urlparse(validated_url)
        path_from_url = parsed_url_obj.path
        filename_from_url = os.path.basename(path_from_url)
        if not filename_from_url or '.' not in filename_from_url:
            try:
                # Use validated_url for HEAD request
                head_response = requests.head(validated_url, timeout=10, allow_redirects=True)
                head_response.raise_for_status()
                content_type_header = head_response.headers.get('Content-Type', '')
                extension = get_extension_from_content_type(content_type_header)
                filename = f"{uuid.uuid4()}{extension or '.tmp'}"
            except requests.RequestException as e:
                capture_exception(e, {"url": validated_url, "context": "filename_detection_head_request"})
                logger.warning(f"No se pudo obtener Content-Type para generar nombre de archivo para {validated_url}: {str(e)}")
                filename = f"{uuid.uuid4()}.tmp"
        else:
            filename = filename_from_url


    file_path = os.path.join(target_dir, filename)
    temp_file_path = f"{file_path}.{uuid.uuid4()}.part" # More unique temp file

    current_max_size = max_size if max_size is not None else MAX_DOWNLOAD_SIZE

    retry_count = 0
    last_exception = None

    while retry_count < MAX_RETRIES:
        try:
            with requests.get(validated_url, stream=True, timeout=DOWNLOAD_TIMEOUT, allow_redirects=True) as response:
                response.raise_for_status()

                content_length_header = response.headers.get('Content-Length')
                if content_length_header and int(content_length_header) > current_max_size:
                    raise ValidationError(
                        message=f"El archivo es demasiado grande (Content-Length): {int(content_length_header)} bytes (máximo: {current_max_size} bytes)",
                        error_code="file_too_large_header",
                        details={"url": validated_url, "content_length": int(content_length_header), "max_size": current_max_size}
                    )

                if validate_mime:
                    content_type_header = response.headers.get('Content-Type', '')
                    if not is_valid_content_type(content_type_header):
                        raise ValidationError(
                            message=f"Tipo de contenido no permitido (header): {content_type_header}",
                            error_code="invalid_content_type_header",
                            details={"url": validated_url, "content_type_header": content_type_header}
                        )

                downloaded_size = 0
                start_time = time.time()

                with open(temp_file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                        if chunk:
                            downloaded_size += len(chunk)
                            if downloaded_size > current_max_size:
                                # No need to f.close() or os.remove here, handled in except ValidationError
                                raise ValidationError(
                                    message=f"Archivo demasiado grande durante descarga (límite: {current_max_size} bytes)",
                                    error_code="file_too_large_stream",
                                    details={"url": validated_url, "downloaded_size": downloaded_size, "max_size": current_max_size}
                                )
                            f.write(chunk) # Can raise IOError/OSError

                if validate_mime and os.path.exists(temp_file_path): # File must exist to detect mime
                    detected_mime = detect_mime_type(temp_file_path)
                    if not is_valid_content_type(detected_mime):
                        raise ValidationError(
                            message=f"Tipo de archivo no permitido (detectado): {detected_mime}",
                            error_code="invalid_content_type_detected",
                            details={"url": validated_url, "detected_mime": detected_mime}
                        )
                
                shutil.move(temp_file_path, file_path) # Can raise IOError/OSError

                download_time = time.time() - start_time
                download_speed = (downloaded_size / (download_time * 1024)) if download_time > 0 else float('inf')

                logger.info(f"Archivo descargado: {validated_url} -> {file_path} ({downloaded_size/1024:.2f} KB, {download_speed:.2f} KB/s)")

                if use_cache and downloaded_size > 0:
                    cache_file(validated_url, file_path)
                
                return file_path # Success

        except requests.RequestException as e:
            last_exception = e
            retry_count += 1
            error_id = capture_exception(e, {"url": validated_url, "attempt": retry_count})
            logger.warning(f"Error de red descargando archivo (intento {retry_count}/{MAX_RETRIES}): {str(e)} (ID: {error_id})")
            if os.path.exists(temp_file_path):
                try: os.remove(temp_file_path)
                except OSError as rm_err: 
                    capture_exception(rm_err, {"file_path": temp_file_path, "context": "cleanup_after_network_error"})
                    logger.error(f"Error eliminando archivo temporal {temp_file_path} tras error de red: {str(rm_err)}")
            if retry_count >= MAX_RETRIES:
                logger.error(f"Falló la descarga de {validated_url} después de {MAX_RETRIES} intentos.")
                raise NetworkError(
                    message=f"Error descargando archivo {validated_url} después de {MAX_RETRIES} intentos: {str(e)}",
                    error_code="download_max_retries_reached",
                    details={"url": validated_url, "attempts": MAX_RETRIES, "original_error": str(e), "error_id": error_id}
                )
            time.sleep(2 ** retry_count)

        except ValidationError as e: # Non-retriable validation error
            # error_id already part of e if created by capture_exception, or None
            logger.error(f"Error de validación para {validated_url}: {e.message} (Code: {e.error_code}, Details: {e.details})")
            if os.path.exists(temp_file_path):
                try: os.remove(temp_file_path)
                except OSError as rm_err:
                    capture_exception(rm_err, {"file_path": temp_file_path, "context": "cleanup_after_validation_error"})
                    logger.error(f"Error eliminando archivo temporal {temp_file_path} tras error de validación: {str(rm_err)}")
            # No need to remove file_path, it shouldn't exist yet or is incomplete.
            raise # Propagate ValidationError

        except (IOError, OSError) as e: # Storage errors, e.g. disk full, permission error
            error_id = capture_exception(e, {"url": validated_url})
            logger.error(f"Error de I/O durante descarga de {validated_url}: {str(e)} (ID: {error_id})")
            if os.path.exists(temp_file_path):
                try: os.remove(temp_file_path)
                except OSError as rm_err:
                    capture_exception(rm_err, {"file_path": temp_file_path, "context": "cleanup_after_io_error"})
                    logger.error(f"Error eliminando archivo temporal {temp_file_path} tras error de I/O: {str(rm_err)}")
            raise StorageError(
                message=f"Error de I/O durante descarga: {str(e)}",
                error_code="download_io_error",
                details={"url": validated_url, "original_error": str(e), "error_id": error_id}
            )
        
        except Exception as e: # Catch any other unexpected error within the loop
            error_id = capture_exception(e, {"url": validated_url})
            logger.error(f"Error inesperado descargando archivo {validated_url}: {str(e)} (ID: {error_id})")
            if os.path.exists(temp_file_path):
                try: os.remove(temp_file_path)
                except OSError as rm_err:
                    capture_exception(rm_err, {"file_path": temp_file_path, "context": "cleanup_after_unexpected_error"})
                    logger.error(f"Error eliminando archivo temporal {temp_file_path} tras error inesperado: {str(rm_err)}")
            # Assuming this is a non-retriable, unexpected issue. Could be Network or Storage.
            # For safety, categorize as a generic NetworkError or a more specific internal error if possible.
            raise NetworkError( # Or a more generic application error
                message=f"Error inesperado durante la descarga de {validated_url}: {str(e)}",
                error_code="download_unexpected_error",
                details={"url": validated_url, "original_error": str(e), "error_id": error_id}
            )
    
    # Should not be reached if loop logic is correct (either returns or raises)
    # But as a fallback if loop finishes without success:
    if last_exception: # Should have been re-raised as NetworkError if max_retries hit
         error_id = capture_exception(last_exception, {"url": validated_url, "context": "fallback_after_retries"})
         raise NetworkError(
                message=f"La descarga falló después de los reintentos para {validated_url}: {str(last_exception)}",
                error_code="download_failed_after_retries_fallback",
                details={"url": validated_url, "original_error": str(last_exception), "error_id": error_id}
            )
    # Fallback for unknown state if loop finishes without return or explicit raise
    error_id = capture_exception(Exception("Unknown download failure"), {"url": validated_url, "context": "unknown_download_failure_state"})
    raise NetworkError(
        message=f"La descarga de {validated_url} falló por una razón desconocida.",
        error_code="download_unknown_failure",
        details={"url": validated_url, "error_id": error_id}
    )


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
        list: Lista de rutas a los archivos descargados (successful ones)
    """
    if target_dir is None:
        target_dir = config.TEMP_DIR

    ensure_directory(target_dir) # Can raise StorageError

    actual_max_workers = min(max_workers, len(urls)) if urls else 1

    downloaded_files = []
    failed_downloads: List[Tuple[str, str, Optional[str]]] = [] # url, error_message, error_code

    def download_single_file_wrapper(url_to_download: str):
        try:
            # Pass use_cache=True by default, or make it a param of download_files_batch
            return download_file(url_to_download, target_dir, validate_mime=validate_mime, use_cache=True)
        except (NetworkError, ValidationError, StorageError) as e:
            logger.error(f"Error descargando archivo batch {url_to_download}: {e.message} (Code: {e.error_code})")
            failed_downloads.append((url_to_download, e.message, e.error_code))
            return None
        except Exception as e: # Catch any other unexpected error
            error_id = capture_exception(e, {"url": url_to_download, "context": "batch_download_unexpected"})
            logger.error(f"Error inesperado descargando archivo batch {url_to_download}: {str(e)} (ID: {error_id})")
            failed_downloads.append((url_to_download, str(e), "unexpected_batch_download_error"))
            return None

    if not urls:
        return []

    with ThreadPoolExecutor(max_workers=actual_max_workers) as executor:
        results = list(executor.map(download_single_file_wrapper, urls))

    downloaded_files = [path for path in results if path is not None]

    success_count = len(downloaded_files)
    fail_count = len(failed_downloads)
    logger.info(f"Descarga batch completada: {success_count} éxitos, {fail_count} fallos de {len(urls)} URLs.")

    if failed_downloads:
        logger.warning(f"Fallaron {fail_count} descargas. Primer error: URL={failed_downloads[0][0]}, Msg='{failed_downloads[0][1]}', Code='{failed_downloads[0][2]}'")

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
        if not os.path.exists(file_path): # Add check for existence
            logger.warning(f"Archivo no encontrado para detección MIME: {file_path}")
            return 'application/octet-stream' # Or raise NotFoundError
        
        # Intentar detectar por contenido (necesita python-magic)
        try:
            import magic
            mime = magic.Magic(mime=True)
            return mime.from_file(file_path)
        except (ImportError, AttributeError, magic.MagicException) as e: # magic.MagicException for libmagic errors
            if not isinstance(e, ImportError) and not isinstance(e, AttributeError):
                 capture_exception(e, {"file_path": file_path, "context": "magic_detection_error"})
                 logger.warning(f"python-magic falló para {file_path}: {str(e)}. Usando fallback.")
            
            # Fallback a mimetypes si python-magic no está disponible o falla
            content_type, _ = mimetypes.guess_type(file_path)
            if content_type:
                return content_type

            # Detección básica por extensión como último recurso
            ext = get_file_extension(file_path).lower()
            mime_mapping = {
                '.mp4': 'video/mp4', '.avi': 'video/x-msvideo', '.mov': 'video/quicktime',
                '.mkv': 'video/x-matroska', '.webm': 'video/webm', '.mp3': 'audio/mpeg',
                '.wav': 'audio/x-wav', '.ogg': 'audio/ogg', '.flac': 'audio/flac',
                '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
                '.gif': 'image/gif', '.webp': 'image/webp', '.pdf': 'application/pdf',
                '.txt': 'text/plain', '.srt': 'application/x-subrip', '.vtt': 'text/vtt',
            }
            return mime_mapping.get(ext, 'application/octet-stream')

    except Exception as e:
        error_id = capture_exception(e, {"file_path": file_path})
        logger.error(f"Error detectando tipo MIME para {file_path}: {str(e)} (ID: {error_id})")
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

    all_allowed_types = [mime for category_mimes in ALLOWED_MIME_TYPES.values() for mime in category_mimes]

    if not all_allowed_types: # If config is empty, allow all (less secure)
        logger.warning("ALLOWED_MIME_TYPES está vacío, permitiendo todos los tipos MIME.")
        return True

    normalized_content_type = content_type.split(';')[0].strip().lower()

    for allowed_type in all_allowed_types:
        if normalized_content_type == allowed_type.lower() or \
           normalized_content_type.startswith(allowed_type.lower() + '+'):
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
    if not content_type: return ''
    normalized_content_type = content_type.split(';')[0].strip().lower()

    # Prioritize our explicit mapping
    mime_to_ext = {
        'video/mp4': '.mp4', 'video/x-msvideo': '.avi', 'video/quicktime': '.mov',
        'video/x-matroska': '.mkv', 'video/webm': '.webm', 'video/3gpp': '.3gp',
        'video/x-flv': '.flv', 'audio/mpeg': '.mp3', 'audio/x-wav': '.wav',
        'audio/wav': '.wav', 'audio/ogg': '.ogg', 'audio/flac': '.flac',
        'audio/aac': '.aac', 'audio/mp4': '.m4a', 'audio/x-m4a': '.m4a',
        'audio/webm': '.weba', 'image/jpeg': '.jpg', 'image/png': '.png',
        'image/gif': '.gif', 'image/bmp': '.bmp', 'image/webp': '.webp',
        'image/svg+xml': '.svg', 'application/pdf': '.pdf', 'text/plain': '.txt',
        'application/x-subrip': '.srt', 'text/vtt': '.vtt', 'application/json': '.json',
        'text/html': '.html', 'text/xml': '.xml', 'application/xml': '.xml'
    }
    
    if normalized_content_type in mime_to_ext:
        return mime_to_ext[normalized_content_type]

    # Check for partial matches (e.g., application/vnd.ms-excel might be guessed as .xls by mimetypes)
    for mime_key, ext_val in mime_to_ext.items():
        if normalized_content_type.startswith(mime_key):
            return ext_val
            
    # Fallback to mimetypes standard library
    ext = mimetypes.guess_extension(normalized_content_type)
    return ext if ext else ''


def ensure_directory(directory: str) -> None:
    """
    Asegura que un directorio existe, creándolo si es necesario

    Args:
        directory (str): Ruta del directorio a verificar/crear
    Raises:
        StorageError: Si no se puede crear el directorio.
    """
    try:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Directorio asegurado: {directory}")
    except (IOError, OSError) as e:
        error_id = capture_exception(e, {"directory": directory})
        logger.error(f"Error creando directorio {directory}: {str(e)} (ID: {error_id})")
        raise StorageError(
            message=f"No se pudo crear el directorio {directory}: {str(e)}",
            error_code="directory_creation_failed",
            details={"directory": directory, "original_error": str(e), "error_id": error_id}
        )

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
    Raises:
        StorageError: Si falla la creación o escritura del archivo temporal.
    """
    fd = -1 # Initialize fd
    temp_path = "" # Initialize temp_path
    try:
        # Ensure TEMP_DIR exists before creating temp file in it
        ensure_directory(config.TEMP_DIR)
        
        fd, temp_path = tempfile.mkstemp(suffix=suffix, dir=config.TEMP_DIR)
        if content is not None:
            with os.fdopen(fd, 'wb') as f:
                f.write(content)
            # fd is closed by os.fdopen's context manager
        else:
            os.close(fd) # Close if not used by fdopen
        
        logger.debug(f"Archivo temporal creado: {temp_path}")
        return temp_path
    except (IOError, OSError, Exception) as e: # Exception for other potential issues
        if fd != -1: # fd was assigned
            try:
                # Check if fd is still open (might be closed by failed fdopen or already closed)
                # This is tricky; os.close on an already closed fd can raise OSError
                # Best effort to close if it wasn't passed to a successful fdopen that closed it.
                # If content is None, we explicitly closed it, or tried to.
                # If content is not None, fdopen context manager should handle it on success.
                # If fdopen itself failed, fd might still be open.
                if content is None or ('f' not in locals() or (locals().get('f') and not locals()['f'].closed)):
                     os.close(fd)
            except OSError:
                pass # Ignore errors on close during an error condition
        
        error_id = capture_exception(e, {"suffix": suffix, "temp_path_attempted": temp_path})
        logger.error(f"Error creando archivo temporal: {str(e)} (ID: {error_id})")
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError as rm_err:
                capture_exception(rm_err, {"file_path": temp_path, "context": "cleanup_after_temp_file_creation_error"})
                logger.error(f"Error eliminando archivo temporal {temp_path} tras fallo de creación: {str(rm_err)}")
        raise StorageError(
            message=f"Error creando archivo temporal: {str(e)}",
            error_code="temp_file_creation_failed",
            details={"suffix": suffix, "original_error": str(e), "error_id": error_id}
        )

def is_video_file(file_path: str) -> bool:
    video_extensions = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm', '.3gp']
    if get_file_extension(file_path) in video_extensions:
        return True
    try:
        return detect_mime_type(file_path).startswith('video/')
    except Exception: return False

def is_audio_file(file_path: str) -> bool:
    audio_extensions = ['.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a', '.wma']
    if get_file_extension(file_path) in audio_extensions:
        return True
    try:
        return detect_mime_type(file_path).startswith('audio/')
    except Exception: return False

def is_image_file(file_path: str) -> bool:
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', '.ico']
    if get_file_extension(file_path) in image_extensions:
        return True
    try:
        return detect_mime_type(file_path).startswith('image/')
    except Exception: return False

def calculate_file_hash(file_path: str, algorithm: str = 'sha256', block_size: int = 65536) -> str:
    """
    Calcula el hash de un archivo de manera eficiente

    Args:
        file_path (str): Ruta al archivo
        algorithm (str): Algoritmo de hash (md5, sha1, sha256, sha512)
        block_size (int): Tamaño de bloque para lectura

    Returns:
        str: Hash del archivo en formato hexadecimal
    Raises:
        NotFoundError: Si el archivo no existe.
        ValidationError: Si el algoritmo no es soportado.
        StorageError: Si hay un error de I/O al leer el archivo.
    """
    if not os.path.exists(file_path):
        raise NotFoundError(
            message=f"Archivo no encontrado para calcular hash: {file_path}",
            error_code="file_not_found_for_hash",
            details={"file_path": file_path}
        )

    hash_algorithms = {'md5': hashlib.md5, 'sha1': hashlib.sha1, 'sha256': hashlib.sha256, 'sha512': hashlib.sha512}
    if algorithm not in hash_algorithms:
        raise ValidationError(
            message=f"Algoritmo de hash no soportado: {algorithm}",
            error_code="unsupported_hash_algorithm",
            details={"algorithm": algorithm, "supported_algorithms": list(hash_algorithms.keys())}
        )

    hash_func = hash_algorithms[algorithm]()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(block_size), b''):
                hash_func.update(chunk)
    except (IOError, OSError) as e:
        error_id = capture_exception(e, {"file_path": file_path, "algorithm": algorithm})
        raise StorageError(
            message=f"Error de I/O calculando hash para {file_path}: {str(e)}",
            error_code="hash_calculation_io_error",
            details={"file_path": file_path, "algorithm": algorithm, "error_id": error_id}
        )
    return hash_func.hexdigest()

def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Obtiene información detallada sobre un archivo

    Args:
        file_path (str): Ruta al archivo

    Returns:
        dict: Información del archivo (tamaño, tipo, hash, etc.)
    Raises:
        NotFoundError: Si el archivo no existe.
        StorageError: Si hay un error de I/O al acceder al archivo (más allá de no existir).
    """
    if not os.path.exists(file_path):
        raise NotFoundError(
            message=f"Archivo no encontrado para obtener información: {file_path}",
            error_code="file_not_found_for_info",
            details={"file_path": file_path}
        )
    try:
        file_stat = os.stat(file_path)
        mime_type = detect_mime_type(file_path) # Could have its own errors if file vanishes

        file_type = 'unknown'
        if mime_type.startswith('video/'): file_type = 'video'
        elif mime_type.startswith('audio/'): file_type = 'audio'
        elif mime_type.startswith('image/'): file_type = 'image'
        elif mime_type.startswith('text/') or mime_type == 'application/pdf': file_type = 'document'

        file_hash = None
        if file_stat.st_size < 100 * 1024 * 1024: # Only hash smaller files for performance
            try:
                file_hash = calculate_file_hash(file_path, algorithm='sha256')
            except (NotFoundError, ValidationError, StorageError) as e: # Catch errors from calculate_file_hash
                 capture_exception(e, {"file_path": file_path, "context": "get_file_info_hash_calculation"})
                 logger.warning(f"No se pudo calcular hash para {file_path} en get_file_info: {e.message}")
            except Exception as e: # Catch any other unexpected error during hashing
                 capture_exception(e, {"file_path": file_path, "context": "get_file_info_hash_calculation_unexpected"})
                 logger.warning(f"Error inesperado calculando hash para {file_path} en get_file_info: {str(e)}")


        return {
            'path': file_path, 'name': os.path.basename(file_path),
            'size': file_stat.st_size, 'size_human': format_size(file_stat.st_size),
            'created': file_stat.st_ctime, 'modified': file_stat.st_mtime,
            'mime_type': mime_type, 'type': file_type,
            'extension': get_file_extension(file_path), 'hash_sha256': file_hash
        }
    except (IOError, OSError) as e: # For os.stat or other unexpected OS errors
        error_id = capture_exception(e, {"file_path": file_path})
        raise StorageError(
            message=f"Error de I/O obteniendo información del archivo {file_path}: {str(e)}",
            error_code="file_info_io_error",
            details={"file_path": file_path, "error_id": error_id}
        )


def format_size(size_bytes: int) -> str:
    if size_bytes < 0: return "0 B" # Handle negative sizes if they somehow occur
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0 or unit == 'TB':
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB" # Should be rare

def safe_delete_file(file_path: str) -> bool:
    """
    Elimina un archivo de forma segura, manejando errores

    Args:
        file_path (str): Ruta al archivo a eliminar

    Returns:
        bool: True si se eliminó correctamente o no existía, False en caso de error al eliminar.
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Archivo eliminado: {file_path}")
        return True # Return True also if file didn't exist
    except (IOError, OSError) as e:
        error_id = capture_exception(e, {"file_path": file_path})
        logger.error(f"Error eliminando archivo {file_path}: {str(e)} (ID: {error_id})")
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

    if not os.path.isdir(directory): # Check if it's a directory
        logger.warning(f"El directorio de limpieza no existe o no es un directorio: {directory}")
        return 0, 0

    max_age_seconds = max_age_hours * 3600
    current_time = time.time()
    files_removed = 0
    bytes_freed = 0

    for root, _, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                if os.path.isfile(file_path) and not os.path.islink(file_path): # Ensure it's a regular file
                    if filename.startswith('.'): # Skip hidden files by default
                        continue
                    
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > max_age_seconds:
                        file_size = os.path.getsize(file_path)
                        if safe_delete_file(file_path):
                            files_removed += 1
                            bytes_freed += file_size
            except FileNotFoundError: # File might be deleted by another process
                 logger.debug(f"Archivo {file_path} no encontrado durante la limpieza, posiblemente ya eliminado.")
                 continue
            except Exception as e: # Catch other errors like permission issues
                error_id = capture_exception(e, {"file_path": file_path, "context": "temp_cleanup"})
                logger.error(f"Error procesando archivo para limpieza {file_path}: {str(e)} (ID: {error_id})")

    if files_removed > 0:
        logger.info(f"Limpieza ({directory}): eliminados {files_removed} archivos ({format_size(bytes_freed)})")
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
    try:
        if not os.path.exists(file_path):
            logger.error(f"Archivo no encontrado para verificación de integridad: {file_path}")
            return False
        if os.path.getsize(file_path) == 0:
            logger.error(f"Archivo vacío, verificación de integridad fallida: {file_path}")
            return False

        if expected_hash:
            actual_hash = calculate_file_hash(file_path, algorithm) # Can raise various errors
            if actual_hash.lower() != expected_hash.lower():
                logger.error(f"Verificación de hash fallida para {file_path}. Esperado: {expected_hash}, Actual: {actual_hash}")
                return False
        
        # Try reading a small part to check for immediate corruption
        with open(file_path, 'rb') as f:
            f.read(1024)
        return True

    except (NotFoundError, ValidationError, StorageError) as e: # From calculate_file_hash or os access
        logger.error(f"Error durante verificación de integridad de {file_path}: {e.message} (Code: {e.error_code})")
        return False
    except (IOError, OSError) as e: # For open/read issues
        error_id = capture_exception(e, {"file_path": file_path})
        logger.error(f"Error de I/O verificando integridad del archivo {file_path}: {str(e)} (ID: {error_id})")
        return False
    except Exception as e: # Catch-all for other unexpected issues
        error_id = capture_exception(e, {"file_path": file_path})
        logger.error(f"Error inesperado verificando integridad del archivo {file_path}: {str(e)} (ID: {error_id})")
        return False


def cleanup_cache(max_age_hours: int = 24) -> Tuple[int, int]:
    """
    Limpia archivos de caché antiguos

    Args:
        max_age_hours (int): Edad máxima en horas para archivos en la caché.

    Returns:
        tuple: (Número de archivos eliminados, Bytes liberados)
    """
    logger.info(f"Iniciando limpieza de caché en {FILE_CACHE_DIR} para archivos más antiguos de {max_age_hours} horas.")
    
    # Limpiar también la caché en memoria de claves que apuntan a archivos no existentes o expirados.
    # This is a more robust cleanup than just relying on get_cached_file's expiry.
    keys_to_delete_from_mem_cache = []
    max_age_seconds = max_age_hours * 3600
    current_time = time.time()

    with file_cache_lock:
        for cache_key, file_path_in_cache in list(file_cache.items()): # Iterate over a copy
            if not os.path.exists(file_path_in_cache):
                keys_to_delete_from_mem_cache.append(cache_key)
                logger.debug(f"Eliminando de caché en memoria (archivo no existe): {cache_key} -> {file_path_in_cache}")
            else:
                try:
                    if current_time - os.path.getmtime(file_path_in_cache) > max_age_seconds:
                        keys_to_delete_from_mem_cache.append(cache_key)
                        logger.debug(f"Eliminando de caché en memoria (archivo expirado en disco): {cache_key} -> {file_path_in_cache}")
                except OSError as e: # e.g. file deleted between exists() and getmtime()
                    capture_exception(e, {"file_path": file_path_in_cache, "context": "cache_cleanup_mem_check"})
                    keys_to_delete_from_mem_cache.append(cache_key)
                    logger.warning(f"Error accediendo a metadatos de archivo en caché {file_path_in_cache}: {str(e)}. Marcando para eliminar de caché en memoria.")


        for key in keys_to_delete_from_mem_cache:
            if key in file_cache:
                del file_cache[key]
    
    # Limpiar archivos del disco
    return cleanup_temp_files(max_age_hours, FILE_CACHE_DIR)


def init_module():
    """Inicializa el módulo"""
    try:
        ensure_directory(config.TEMP_DIR) # Ensure base temp dir exists
        ensure_directory(FILE_CACHE_DIR)   # Ensure cache dir exists
    except StorageError as e:
        # Log critical failure if directories can't be created, module might be unusable.
        logger.critical(f"Fallo crítico inicializando directorios del módulo file_management: {e.message} (Code: {e.error_code})")
        # Depending on application, might re-raise or exit. For a library, logging is often sufficient.

    mimetypes.init()
    mimetypes.add_type('application/x-subrip', '.srt', strict=False) # strict=False to avoid warnings if already there
    mimetypes.add_type('text/vtt', '.vtt', strict=False)
    
    logger.info(f"Módulo file_management inicializado. Directorio temporal: {config.TEMP_DIR}, Directorio caché: {FILE_CACHE_DIR}")

init_module()
# --- END OF FILE file_management.py ---
