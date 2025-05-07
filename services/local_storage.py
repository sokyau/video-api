import os
import logging
import uuid
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import config

logger = logging.getLogger(__name__)

def ensure_storage_dir():
    """Asegurar que el directorio de almacenamiento existe"""
    os.makedirs(config.STORAGE_PATH, exist_ok=True)
    logger.debug(f"Storage directory ensured: {config.STORAGE_PATH}")

def get_file_path(filename):
    """Obtener ruta completa del archivo"""
    return os.path.join(config.STORAGE_PATH, filename)

def get_file_url(filename):
    """Obtener URL pública del archivo"""
    return f"{config.BASE_URL}/{filename}"

def store_file(file_path, custom_filename=None):
    """
    Almacenar un archivo en el sistema local y retornar su URL
    
    Args:
        file_path (str): Ruta del archivo a almacenar
        custom_filename (str, optional): Nombre personalizado para el archivo
        
    Returns:
        str: URL pública del archivo almacenado
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
    
    Returns:
        tuple: (Número de archivos eliminados, Bytes liberados)
    """
    ensure_storage_dir()
    now = datetime.now()
    cutoff = now - timedelta(hours=config.MAX_FILE_AGE_HOURS)
    
    file_count = 0
    size_freed = 0
    
    for filename in os.listdir(config.STORAGE_PATH):
        file_path = os.path.join(config.STORAGE_PATH, filename)
        
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
                    logger.debug(f"Removed old file: {file_path}")
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
                logger.debug(f"Removed old file (using mtime): {file_path}")
    
    if file_count > 0:
        logger.info(f"Cleanup: removed {file_count} files ({size_freed/1024/1024:.2f} MB)")
    return file_count, size_freed

def file_exists(filename):
    """
    Verificar si un archivo existe en el almacenamiento
    
    Args:
        filename (str): Nombre del archivo a verificar
    
    Returns:
        bool: True si el archivo existe, False en caso contrario
    """
    file_path = get_file_path(filename)
    return os.path.exists(file_path)

def get_file_info(filename):
    """
    Obtener información sobre un archivo almacenado
    
    Args:
        filename (str): Nombre del archivo
    
    Returns:
        dict: Información del archivo (tamaño, tipo, fecha de creación)
    """
    file_path = get_file_path(filename)
    
    if not os.path.exists(file_path):
        return None
    
    # Intentar obtener fecha de creación desde .meta
    created_time = None
    meta_path = f"{file_path}.meta"
    
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                created_time = datetime.fromisoformat(f.read().strip())
        except Exception:
            pass
    
    # Fallback a mtime si no hay .meta
    if created_time is None:
        created_time = datetime.fromtimestamp(os.path.getmtime(file_path))
    
    # Determinar tipo de archivo basado en extensión
    _, extension = os.path.splitext(filename)
    
    # Mapeo simple de extensiones comunes
    media_types = {
        '.mp4': 'video/mp4',
        '.webm': 'video/webm',
        '.mp3': 'audio/mpeg',
        '.wav': 'audio/wav',
        '.ogg': 'audio/ogg',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif'
    }
    
    media_type = media_types.get(extension.lower(), 'application/octet-stream')
    
    return {
        'filename': filename,
        'size': os.path.getsize(file_path),
        'created': created_time.isoformat(),
        'media_type': media_type,
        'url': get_file_url(filename)
    }
