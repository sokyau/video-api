# --- START OF FILE local_storage.py ---

import os
import logging
import uuid
import shutil
from datetime import datetime, timedelta
from pathlib import Path # Not strictly used, but often good for path manipulation
import config

# Assuming 'errors.py' is at the root or accessible
from errors import (
    StorageError,
    NotFoundError,
    ValidationError, # For potential validation issues, e.g., bad filename
    capture_exception
)

logger = logging.getLogger(__name__)

def ensure_storage_dir():
    """
    Asegurar que el directorio de almacenamiento existe.
    Raises:
        StorageError: Si no se puede crear el directorio.
    """
    try:
        os.makedirs(config.STORAGE_PATH, exist_ok=True)
        logger.debug(f"Storage directory ensured: {config.STORAGE_PATH}")
    except OSError as e:
        error_id = capture_exception(e, {"directory": config.STORAGE_PATH})
        raise StorageError(
            message=f"No se pudo asegurar el directorio de almacenamiento '{config.STORAGE_PATH}': {str(e)}",
            error_code="storage_dir_creation_failed",
            details={"directory": config.STORAGE_PATH, "original_error": str(e), "error_id": error_id}
        )

def get_file_path(filename: str) -> str:
    """
    Obtener ruta completa del archivo.
    Validates filename to prevent path traversal.
    Args:
        filename (str): The name of the file.
    Returns:
        str: The full path to the file in the storage directory.
    Raises:
        ValidationError: If the filename is invalid (e.g., contains path traversal).
    """
    if not filename or '..' in filename or filename.startswith(('/', '\\')):
        raise ValidationError(
            message=f"Nombre de archivo inválido o intento de path traversal: '{filename}'",
            error_code="invalid_storage_filename",
            details={"filename": filename}
        )
    return os.path.join(config.STORAGE_PATH, filename)

def get_file_url(filename: str) -> str:
    """
    Obtener URL pública del archivo.
    Args:
        filename (str): The name of the file.
    Returns:
        str: The public URL for the file.
    Raises:
        ValidationError: If the filename is invalid.
    """
    # Validation happens in get_file_path if we were to use it,
    # but here we just construct the URL. Basic check for safety.
    if not filename or '..' in filename or filename.startswith(('/', '\\')):
        raise ValidationError(
            message=f"Nombre de archivo inválido para URL: '{filename}'",
            error_code="invalid_url_filename",
            details={"filename": filename}
        )
    # Ensure BASE_URL ends with a slash if it doesn't already, and filename doesn't start with one.
    base_url = config.BASE_URL.rstrip('/')
    clean_filename = filename.lstrip('/')
    return f"{base_url}/{clean_filename}"


def store_file(file_path: str, custom_filename: Optional[str] = None) -> str:
    """
    Almacenar un archivo en el sistema local y retornar su URL
    
    Args:
        file_path (str): Ruta del archivo a almacenar
        custom_filename (str, optional): Nombre personalizado para el archivo
        
    Returns:
        str: URL pública del archivo almacenado
    Raises:
        NotFoundError: If the source file_path does not exist.
        StorageError: If there's an issue creating directories, copying the file, or writing metadata.
        ValidationError: If custom_filename is invalid.
    """
    ensure_storage_dir() # Can raise StorageError
    
    if not os.path.exists(file_path):
        raise NotFoundError(
            message=f"Archivo fuente no encontrado para almacenar: {file_path}",
            error_code="source_file_not_found_for_storage",
            details={"source_path": file_path}
        )
    
    target_filename: str
    if custom_filename is None:
        file_ext = os.path.splitext(file_path)[1]
        target_filename = f"{uuid.uuid4()}{file_ext}"
    else:
        # Validate custom_filename before using it in get_file_path
        if '..' in custom_filename or custom_filename.startswith(('/', '\\')):
             raise ValidationError(
                message=f"Nombre de archivo personalizado inválido: '{custom_filename}'",
                error_code="invalid_custom_filename_storage",
                details={"custom_filename": custom_filename}
            )
        target_filename = custom_filename
    
    # get_file_path will also validate the combined path implicitly by its own validation
    target_storage_path = get_file_path(target_filename) # Can raise ValidationError
    
    try:
        # Copiar archivo a ubicación de almacenamiento
        shutil.copy2(file_path, target_storage_path)
        
        # Crear archivo .meta con timestamp
        # Ensure the directory for meta file exists (should be same as target_storage_path's dir)
        with open(f"{target_storage_path}.meta", "w", encoding='utf-8') as f:
            f.write(datetime.now().isoformat())
        
        logger.info(f"File stored locally: {target_storage_path}")
        return get_file_url(target_filename) # Can raise ValidationError
    except (IOError, OSError, shutil.Error) as e:
        error_id = capture_exception(e, {"source_path": file_path, "target_path": target_storage_path})
        # Clean up partially stored file or meta file if copy/meta write failed
        if os.path.exists(target_storage_path):
            try: os.remove(target_storage_path)
            except OSError: pass
        if os.path.exists(f"{target_storage_path}.meta"):
            try: os.remove(f"{target_storage_path}.meta")
            except OSError: pass
        raise StorageError(
            message=f"Error almacenando archivo '{os.path.basename(file_path)}': {str(e)}",
            error_code="file_storage_failed",
            details={"source_path": file_path, "target_path": target_storage_path, "original_error": str(e), "error_id": error_id}
        )

def cleanup_old_files() -> Tuple[int, int]:
    """
    Eliminar archivos más antiguos que MAX_FILE_AGE_HOURS
    
    Returns:
        tuple: (Número de archivos eliminados, Bytes liberados)
    Raises:
        StorageError: If the storage directory cannot be accessed.
    """
    try:
        ensure_storage_dir() # Ensure dir exists, can raise StorageError
    except StorageError as e:
        # If ensure_storage_dir itself fails, we can't proceed with cleanup.
        # Log and re-raise or return 0,0 depending on desired behavior.
        logger.error(f"No se puede acceder al directorio de almacenamiento para limpieza: {e.message}")
        raise # Or return 0,0 if cleanup failure is non-critical for the caller

    now = datetime.now()
    try:
        max_age_hours = int(config.MAX_FILE_AGE_HOURS)
        if max_age_hours <=0:
            logger.info("La limpieza de archivos antiguos está desactivada (MAX_FILE_AGE_HOURS <= 0).")
            return 0,0
        cutoff = now - timedelta(hours=max_age_hours)
    except (AttributeError, ValueError, TypeError) as e:
        error_id = capture_exception(e, {"config_value_MAX_FILE_AGE_HOURS": getattr(config, "MAX_FILE_AGE_HOURS", "Not Set")})
        logger.error(f"Configuración MAX_FILE_AGE_HOURS inválida ({str(e)}). La limpieza de archivos no se ejecutará. Error ID: {error_id}")
        return 0, 0 # Cannot proceed without valid cutoff
    
    file_count = 0
    size_freed = 0
    
    try:
        for filename in os.listdir(config.STORAGE_PATH):
            # Skip hidden files or special files
            if filename.startswith('.'):
                continue

            file_path = os.path.join(config.STORAGE_PATH, filename) # Use os.path.join for safety
            
            # Skip directories
            if not os.path.isfile(file_path):
                continue

            # Saltar archivos .meta directamente in the loop
            if filename.endswith('.meta'):
                continue
            
            meta_path = f"{file_path}.meta"
            file_removed_this_iteration = False

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
                        file_removed_this_iteration = True
                        logger.debug(f"Removed old file (meta): {file_path}")
                elif not file_removed_this_iteration: # Fallback only if not already removed via meta
                    # Fallback a mtime si no hay .meta (y el archivo no fue ya eliminado)
                    mtime_ts = os.path.getmtime(file_path)
                    mtime_dt = datetime.fromtimestamp(mtime_ts)
                    if mtime_dt < cutoff:
                        size = os.path.getsize(file_path)
                        os.remove(file_path)
                        # No meta file to remove in this case
                        file_count += 1
                        size_freed += size
                        logger.debug(f"Removed old file (mtime): {file_path}")
            except FileNotFoundError: # File might have been deleted by another process between listdir and access
                logger.debug(f"File {file_path} not found during cleanup, possibly already deleted.")
                continue
            except (ValueError, IOError, OSError) as e: # For fromisoformat, getsize, remove errors
                error_id = capture_exception(e, {"context": "cleanup_old_files_item", "file_path": file_path, "meta_path": meta_path})
                logger.error(f"Error procesando archivo '{file_path}' o su metafile '{meta_path}' para limpieza: {str(e)} (ID: {error_id})")
    except OSError as e: # For os.listdir error
        error_id = capture_exception(e, {"storage_path": config.STORAGE_PATH})
        raise StorageError(
            message=f"Error listando directorio de almacenamiento '{config.STORAGE_PATH}' para limpieza: {str(e)}",
            error_code="storage_listdir_failed_cleanup",
            details={"storage_path": config.STORAGE_PATH, "error_id": error_id}
        )

    if file_count > 0:
        logger.info(f"Cleanup: removed {file_count} files ({size_freed/1024/1024:.2f} MB)")
    return file_count, size_freed


def file_exists(filename: str) -> bool:
    """
    Verificar si un archivo existe en el almacenamiento
    
    Args:
        filename (str): Nombre del archivo a verificar
    
    Returns:
        bool: True si el archivo existe, False en caso contrario
    Raises:
        ValidationError: If filename is invalid.
    """
    try:
        file_storage_path = get_file_path(filename) # Validates filename
        return os.path.exists(file_storage_path)
    except ValidationError: # If filename itself is invalid, it can't exist in our managed storage
        return False
    except Exception as e: # Should not happen if get_file_path is robust
        capture_exception(e, {"filename": filename, "context": "file_exists_unexpected"})
        logger.error(f"Error inesperado en file_exists para '{filename}': {str(e)}")
        return False


def get_file_info(filename: str) -> Optional[Dict[str, Any]]:
    """
    Obtener información sobre un archivo almacenado
    
    Args:
        filename (str): Nombre del archivo
    
    Returns:
        dict: Información del archivo (tamaño, tipo, fecha de creación), or None if not found.
    Raises:
        NotFoundError: If the file does not exist (explicitly raised).
        ValidationError: If filename is invalid.
        StorageError: For unexpected OS/IO errors when accessing file info.
    """
    try:
        file_storage_path = get_file_path(filename) # Validates filename
    except ValidationError as ve:
        # If filename is invalid, directly raise NotFoundError as it won't be found
        raise NotFoundError(message=f"Archivo '{filename}' no encontrado debido a nombre inválido.",
                            error_code="file_info_invalid_filename",
                            details={"filename": filename, "validation_error": ve.message})

    if not os.path.exists(file_storage_path):
        raise NotFoundError(
            message=f"Archivo no encontrado en almacenamiento: {filename}",
            error_code="file_info_not_found",
            details={"filename": filename}
        )
    
    created_time_iso: Optional[str] = None
    meta_path = f"{file_storage_path}.meta"
    
    try:
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding='utf-8') as f:
                created_time_str = f.read().strip()
            created_time_iso = datetime.fromisoformat(created_time_str).isoformat()
        else: # Fallback a mtime si no hay .meta
            created_time_iso = datetime.fromtimestamp(os.path.getmtime(file_storage_path)).isoformat()
    except (IOError, OSError, ValueError) as e: # For open, read, fromisoformat, getmtime errors
        error_id = capture_exception(e, {"context": "get_file_info_timestamp", "file_path": file_storage_path})
        logger.warning(f"No se pudo determinar la fecha de creación para '{filename}' (ID: {error_id}). Usando None.")
        # Continue with None for created_time_iso

    _, extension = os.path.splitext(filename)
    media_types = {
        '.mp4': 'video/mp4', '.webm': 'video/webm', '.mkv': 'video/x-matroska',
        '.mov': 'video/quicktime', '.avi': 'video/x-msvideo',
        '.mp3': 'audio/mpeg', '.wav': 'audio/wav', '.ogg': 'audio/ogg', '.aac': 'audio/aac', '.flac': 'audio/flac',
        '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.gif': 'image/gif', '.webp': 'image/webp',
        '.txt': 'text/plain', '.json': 'application/json', '.pdf': 'application/pdf'
    }
    media_type = media_types.get(extension.lower(), 'application/octet-stream')
    
    try:
        file_size = os.path.getsize(file_storage_path)
    except OSError as e:
        error_id = capture_exception(e, {"context": "get_file_info_size", "file_path": file_storage_path})
        raise StorageError(
            message=f"No se pudo obtener el tamaño del archivo '{filename}': {str(e)}",
            error_code="file_info_get_size_failed",
            details={"filename": filename, "file_path": file_storage_path, "error_id": error_id}
        )

    return {
        'filename': filename,
        'size': file_size,
        'created': created_time_iso, # Can be None if determination failed
        'media_type': media_type,
        'url': get_file_url(filename) # Can raise ValidationError
    }

# --- END OF FILE local_storage.py ---
