import threading
import time
import logging
import os
import shutil
import re
import json
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from pathlib import Path

from services.local_storage import cleanup_old_files
from services.file_management import safe_delete_file, format_size
import config

logger = logging.getLogger(__name__)

# Constantes para configuración
DEFAULT_CLEANUP_INTERVAL = 30  # minutos
DEFAULT_MAX_FILE_AGE = 6  # horas
DEFAULT_EMERGENCY_FILE_AGE = 1  # hora (para limpieza de emergencia)
DEFAULT_DISK_THRESHOLD = 90  # porcentaje
DEFAULT_EMERGENCY_THRESHOLD = 95  # porcentaje para limpieza de emergencia
FILE_PATTERNS = {
    'temp': [r'\.temp$', r'\.tmp$', r'\.part$', r'_temp_'],
    'logs': [r'\.log$', r'\.log\.\d+$'],
    'output': [r'output_.*\.mp4$', r'result_.*\.mp4$'],
    'cache': [r'\.cache$', r'cache_.*']
}

class StorageStatus:
    """Clase para representar el estado del almacenamiento"""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class CleanupError(Exception):
    """Excepción para errores en el proceso de limpieza"""
    pass

class CleanupResult:
    """Clase para representar el resultado de una operación de limpieza"""
    def __init__(self, 
                 files_removed: int = 0, 
                 bytes_freed: int = 0, 
                 categories: Optional[Dict[str, int]] = None,
                 errors: Optional[List[str]] = None,
                 status: str = StorageStatus.NORMAL):
        self.files_removed = files_removed
        self.bytes_freed = bytes_freed
        self.categories = categories or {}
        self.errors = errors or []
        self.status = status
        self.timestamp = datetime.now()
    
    def __str__(self) -> str:
        """Representación en cadena del resultado"""
        return (f"CleanupResult: {self.files_removed} archivos eliminados, "
                f"{format_size(self.bytes_freed)} liberados, "
                f"estado: {self.status}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el resultado a diccionario para JSON"""
        return {
            'files_removed': self.files_removed,
            'bytes_freed': self.bytes_freed,
            'bytes_freed_formatted': format_size(self.bytes_freed),
            'categories': self.categories,
            'errors': self.errors,
            'status': self.status,
            'timestamp': self.timestamp.isoformat()
        }

class CleanupService:
    """
    Servicio mejorado para limpieza periódica de archivos antiguos y monitoreo de almacenamiento
    """
    def __init__(self, interval_minutes=DEFAULT_CLEANUP_INTERVAL, 
                 max_file_age_hours=None,
                 disk_threshold_percent=DEFAULT_DISK_THRESHOLD,
                 emergency_threshold_percent=DEFAULT_EMERGENCY_THRESHOLD,
                 cleanup_paths=None):
        """
        Inicializar servicio con configuración
        
        Args:
            interval_minutes: Intervalo de ejecución en minutos
            max_file_age_hours: Edad máxima de archivos en horas (None = usar config)
            disk_threshold_percent: Umbral de uso de disco para alerta
            emergency_threshold_percent: Umbral para limpieza de emergencia
            cleanup_paths: Lista de directorios adicionales a limpiar
        """
        self.interval = interval_minutes * 60  # Convertir a segundos
        self.max_file_age = max_file_age_hours or getattr(config, 'MAX_FILE_AGE_HOURS', DEFAULT_MAX_FILE_AGE)
        self.emergency_file_age = DEFAULT_EMERGENCY_FILE_AGE
        self.disk_threshold = disk_threshold_percent
        self.emergency_threshold = emergency_threshold_percent
        
        # Configurar rutas de limpieza
        self.storage_path = getattr(config, 'STORAGE_PATH', '/var/www/videoapi.sofe.site/storage')
        self.temp_dir = getattr(config, 'TEMP_DIR', '/tmp')
        
        # Directorios adicionales a limpiar
        self.cleanup_paths = cleanup_paths or []
        if hasattr(config, 'ADDITIONAL_CLEANUP_PATHS') and isinstance(config.ADDITIONAL_CLEANUP_PATHS, list):
            self.cleanup_paths.extend(config.ADDITIONAL_CLEANUP_PATHS)
        
        # Lista de extensiones protegidas (nunca eliminar)
        self.protected_extensions = getattr(config, 'PROTECTED_EXTENSIONS', ['.gitkeep', '.gitignore'])
        
        # Lista de patrones protegidos (nunca eliminar)
        self.protected_patterns = getattr(config, 'PROTECTED_PATTERNS', [r'\.git', r'__pycache__', r'venv'])
        
        # Variables de control
        self.shutdown_flag = threading.Event()
        self.pause_flag = threading.Event()
        self.thread = None
        self.manual_trigger = threading.Event()
        
        # Métricas y estadísticas
        self.last_run_time = None
        self.last_result = None
        self.run_count = 0
        self.total_files_cleaned = 0
        self.total_bytes_freed = 0
        self.history = []  # Historial limitado de resultados
        self.history_limit = 20
    
    def start(self):
        """Iniciar servicio en un hilo separado"""
        if self.thread is not None:
            logger.warning("Servicio de limpieza ya en ejecución")
            return
        
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info(f"Servicio de limpieza iniciado (intervalo: {self.interval/60} minutos)")
    
    def stop(self):
        """Detener servicio"""
        if self.thread is None:
            logger.warning("Servicio de limpieza no está en ejecución")
            return
        
        logger.info("Deteniendo servicio de limpieza...")
        self.shutdown_flag.set()
        self.manual_trigger.set()  # Para desbloquear el hilo si está en espera
        self.thread.join(timeout=60)  # Esperar hasta 60s para terminar
        
        if self.thread.is_alive():
            logger.warning("El servicio de limpieza no terminó correctamente")
        else:
            logger.info("Servicio de limpieza detenido correctamente")
            self.thread = None
            self.shutdown_flag.clear()
            self.manual_trigger.clear()
    
    def pause(self):
        """Pausar temporalmente el servicio"""
        logger.info("Pausando servicio de limpieza")
        self.pause_flag.set()
    
    def resume(self):
        """Reanudar el servicio pausado"""
        logger.info("Reanudando servicio de limpieza")
        self.pause_flag.clear()
    
    def request_cleanup(self):
        """Solicitar limpieza inmediata"""
        logger.info("Limpieza manual solicitada")
        self.manual_trigger.set()
    
    def _run(self):
        """Ejecutar ciclo de limpieza"""
        logger.info("Ciclo de limpieza iniciado")
        
        while not self.shutdown_flag.is_set():
            try:
                # Si estamos pausados, esperar hasta reactivación
                if self.pause_flag.is_set():
                    wait_time = 30  # Comprobar cada 30s si debemos reanudar
                    logger.debug("Servicio de limpieza pausado, esperando")
                    self.manual_trigger.wait(timeout=wait_time)
                    self.manual_trigger.clear()
                    continue
                
                # Verificar espacio en disco
                status = self._check_storage_status()
                
                # Ejecutar limpieza según el estado
                if status == StorageStatus.EMERGENCY:
                    logger.warning("¡EMERGENCIA DE ALMACENAMIENTO! Ejecutando limpieza agresiva")
                    self.last_result = self._perform_emergency_cleanup()
                elif status == StorageStatus.CRITICAL:
                    logger.warning("Almacenamiento crítico. Ejecutando limpieza intensiva")
                    self.last_result = self._perform_intensive_cleanup()
                elif self.manual_trigger.is_set() or status == StorageStatus.WARNING:
                    logger.info("Ejecutando limpieza estándar")
                    self.last_result = self.run_now()
                    self.manual_trigger.clear()
                else:
                    logger.debug("Almacenamiento normal. Ejecutando limpieza regular")
                    self.last_result = self._perform_regular_cleanup()
                
                # Actualizar estadísticas
                self.last_run_time = datetime.now()
                self.run_count += 1
                self.total_files_cleaned += self.last_result.files_removed
                self.total_bytes_freed += self.last_result.bytes_freed
                
                # Guardar historial limitado
                self.history.append(self.last_result.to_dict())
                if len(self.history) > self.history_limit:
                    self.history.pop(0)
                
                # Registrar resultados
                if self.last_result.files_removed > 0:
                    logger.info(str(self.last_result))
                
                # Esperar hasta el próximo ciclo o hasta recibir señal de disparo manual
                logger.debug(f"Esperando {self.interval/60} minutos para el próximo ciclo")
                self.manual_trigger.wait(timeout=self.interval)
                self.manual_trigger.clear()
                
            except Exception as e:
                logger.error(f"Error en ciclo de limpieza: {e}", exc_info=True)
                # Esperar un tiempo antes de intentar de nuevo
                time.sleep(60)  # 1 minuto
    
    def _check_storage_status(self) -> str:
        """
        Verificar estado del almacenamiento
        
        Returns:
            str: Estado del almacenamiento (NORMAL, WARNING, CRITICAL, EMERGENCY)
        """
        try:
            # Verificar espacio en disco para el directorio de almacenamiento
            disk_usage = shutil.disk_usage(self.storage_path)
            used_percent = (disk_usage.used / disk_usage.total) * 100
            
            if used_percent >= self.emergency_threshold:
                return StorageStatus.EMERGENCY
            elif used_percent >= self.disk_threshold:
                return StorageStatus.CRITICAL
            elif used_percent >= self.disk_threshold * 0.9:  # 90% del umbral
                return StorageStatus.WARNING
            else:
                return StorageStatus.NORMAL
                
        except Exception as e:
            logger.error(f"Error verificando almacenamiento: {e}")
            # En caso de error, ser conservador
            return StorageStatus.WARNING
    
    def get_storage_info(self) -> Dict[str, Any]:
        """
        Obtener información detallada del almacenamiento
        
        Returns:
            Dict: Información del almacenamiento
        """
        result = {}
        
        try:
            # Espacio total del sistema
            system_usage = shutil.disk_usage(self.storage_path)
            result['system'] = {
                'total_bytes': system_usage.total,
                'used_bytes': system_usage.used,
                'free_bytes': system_usage.free,
                'used_percent': (system_usage.used / system_usage.total) * 100,
                'total_formatted': format_size(system_usage.total),
                'used_formatted': format_size(system_usage.used),
                'free_formatted': format_size(system_usage.free)
            }
            
            # Uso por directorios principales
            directories = {
                'storage': self.storage_path,
                'temp': self.temp_dir
            }
            
            result['directories'] = {}
            
            for name, path in directories.items():
                if os.path.exists(path):
                    dir_size = self._get_directory_size(path)
                    result['directories'][name] = {
                        'path': path,
                        'size_bytes': dir_size,
                        'size_formatted': format_size(dir_size),
                        'file_count': self._count_files(path)
                    }
            
            # Estadísticas de limpieza
            result['cleanup_stats'] = {
                'run_count': self.run_count,
                'total_files_cleaned': self.total_files_cleaned,
                'total_bytes_freed': self.total_bytes_freed,
                'total_freed_formatted': format_size(self.total_bytes_freed),
                'last_run': self.last_run_time.isoformat() if self.last_run_time else None,
                'status': self._check_storage_status()
            }
            
            # Información de configuración
            result['config'] = {
                'interval_minutes': self.interval / 60,
                'max_file_age_hours': self.max_file_age,
                'disk_threshold_percent': self.disk_threshold,
                'emergency_threshold_percent': self.emergency_threshold
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error obteniendo información de almacenamiento: {e}")
            return {'error': str(e)}
    
    def _get_directory_size(self, path: str) -> int:
        """
        Calcular tamaño total de un directorio
        
        Args:
            path: Ruta del directorio
            
        Returns:
            int: Tamaño en bytes
        """
        total_size = 0
        
        try:
            for dirpath, _, filenames in os.walk(path):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    if os.path.isfile(file_path) and not os.path.islink(file_path):
                        total_size += os.path.getsize(file_path)
        except Exception as e:
            logger.error(f"Error calculando tamaño de directorio {path}: {e}")
        
        return total_size
    
    def _count_files(self, path: str) -> int:
        """
        Contar archivos en un directorio
        
        Args:
            path: Ruta del directorio
            
        Returns:
            int: Número de archivos
        """
        count = 0
        
        try:
            for dirpath, _, filenames in os.walk(path):
                count += len(filenames)
        except Exception as e:
            logger.error(f"Error contando archivos en {path}: {e}")
        
        return count
    
    def run_now(self) -> CleanupResult:
        """
        Ejecutar limpieza inmediatamente
        
        Returns:
            CleanupResult: Resultado de la limpieza
        """
        logger.info(f"Ejecutando limpieza (max age: {self.max_file_age} horas)")
        result = CleanupResult(status=self._check_storage_status())
        
        try:
            # Limpiar directorio de almacenamiento principal
            storage_files, storage_bytes = cleanup_old_files(
                max_age_hours=self.max_file_age,
                directory=self.storage_path
            )
            
            result.files_removed += storage_files
            result.bytes_freed += storage_bytes
            result.categories['storage'] = storage_files
            
            # Limpiar directorio temporal
            temp_files, temp_bytes = self._cleanup_temp_directory(self.temp_dir, self.max_file_age)
            
            result.files_removed += temp_files
            result.bytes_freed += temp_bytes
            result.categories['temp'] = temp_files
            
            # Limpiar directorios adicionales
            for path in self.cleanup_paths:
                if os.path.isdir(path):
                    try:
                        path_files, path_bytes = cleanup_old_files(
                            max_age_hours=self.max_file_age,
                            directory=path
                        )
                        
                        result.files_removed += path_files
                        result.bytes_freed += path_bytes
                        result.categories[os.path.basename(path)] = path_files
                    except Exception as e:
                        error_msg = f"Error limpiando directorio adicional {path}: {e}"
                        logger.error(error_msg)
                        result.errors.append(error_msg)
            
            logger.info(f"Limpieza completada: {result.files_removed} archivos, {format_size(result.bytes_freed)} liberados")
            return result
            
        except Exception as e:
            error_msg = f"Error en limpieza manual: {e}"
            logger.error(error_msg, exc_info=True)
            result.errors.append(error_msg)
            return result
    
    def _perform_regular_cleanup(self) -> CleanupResult:
        """
        Realizar limpieza regular (programada)
        
        Returns:
            CleanupResult: Resultado de la limpieza
        """
        # Para limpieza regular, usar la función estándar
        return self.run_now()
    
    def _perform_intensive_cleanup(self) -> CleanupResult:
        """
        Realizar limpieza intensiva cuando el almacenamiento está crítico
        
        Returns:
            CleanupResult: Resultado de la limpieza
        """
        logger.warning("Ejecutando limpieza intensiva - espacio en disco crítico")
        
        # Usar la mitad del tiempo máximo para limpieza intensiva
        intensive_age = max(1, self.max_file_age // 2)
        
        result = CleanupResult(status=StorageStatus.CRITICAL)
        
        try:
            # Limpiar almacenamiento principal con edad reducida
            storage_files, storage_bytes = cleanup_old_files(
                max_age_hours=intensive_age,
                directory=self.storage_path
            )
            
            result.files_removed += storage_files
            result.bytes_freed += storage_bytes
            result.categories['storage'] = storage_files
            
            # Limpiar directorio temporal de forma más agresiva
            temp_files, temp_bytes = self._cleanup_temp_directory(
                self.temp_dir, 
                intensive_age,
                include_patterns=FILE_PATTERNS['temp']
            )
            
            result.files_removed += temp_files
            result.bytes_freed += temp_bytes
            result.categories['temp'] = temp_files
            
            # Limpiar caché de forma más agresiva
            cache_dir = getattr(config, 'FILE_CACHE_DIR', os.path.join(self.temp_dir, 'cache'))
            if os.path.isdir(cache_dir):
                cache_files, cache_bytes = cleanup_old_files(
                    max_age_hours=intensive_age,
                    directory=cache_dir
                )
                
                result.files_removed += cache_files
                result.bytes_freed += cache_bytes
                result.categories['cache'] = cache_files
            
            # Limpiar directorios adicionales
            for path in self.cleanup_paths:
                if os.path.isdir(path):
                    try:
                        path_files, path_bytes = cleanup_old_files(
                            max_age_hours=intensive_age,
                            directory=path
                        )
                        
                        result.files_removed += path_files
                        result.bytes_freed += path_bytes
                        result.categories[os.path.basename(path)] = path_files
                    except Exception as e:
                        error_msg = f"Error en limpieza intensiva para {path}: {e}"
                        logger.error(error_msg)
                        result.errors.append(error_msg)
            
            logger.warning(f"Limpieza intensiva completada: {result.files_removed} archivos, {format_size(result.bytes_freed)}")
            return result
            
        except Exception as e:
            error_msg = f"Error en limpieza intensiva: {e}"
            logger.error(error_msg, exc_info=True)
            result.errors.append(error_msg)
            return result
    
    def _perform_emergency_cleanup(self) -> CleanupResult:
        """
        Realizar limpieza de emergencia cuando el almacenamiento está crítico
        
        Returns:
            CleanupResult: Resultado de la limpieza
        """
        logger.critical("¡EMERGENCIA DE ALMACENAMIENTO! Ejecutando limpieza agresiva")
        
        # Usar tiempo mínimo para emergencia
        emergency_age = self.emergency_file_age
        
        result = CleanupResult(status=StorageStatus.EMERGENCY)
        
        try:
            # Paso 1: Limpieza agresiva del almacenamiento
            storage_files, storage_bytes = cleanup_old_files(
                max_age_hours=emergency_age,
                directory=self.storage_path
            )
            
            result.files_removed += storage_files
            result.bytes_freed += storage_bytes
            result.categories['storage'] = storage_files
            
            # Paso 2: Limpiar todos los archivos temporales sin importar edad
            temp_files, temp_bytes = self._cleanup_temp_directory(
                self.temp_dir, 
                0,  # 0 = no importa la edad
                respect_age=False,
                include_patterns=FILE_PATTERNS['temp'] + FILE_PATTERNS['logs']
            )
            
            result.files_removed += temp_files
            result.bytes_freed += temp_bytes
            result.categories['temp'] = temp_files
            
            # Paso 3: Limpiar caché completamente
            cache_dir = getattr(config, 'FILE_CACHE_DIR', os.path.join(self.temp_dir, 'cache'))
            if os.path.isdir(cache_dir):
                cache_files, cache_bytes = self._cleanup_temp_directory(
                    cache_dir,
                    0,
                    respect_age=False  # Ignorar edad
                )
                
                result.files_removed += cache_files
                result.bytes_freed += cache_bytes
                result.categories['cache'] = cache_files
            
            # Paso 4: Limpiar archivos grandes de forma agresiva
            big_files, big_bytes = self._cleanup_big_files()
            
            result.files_removed += big_files
            result.bytes_freed += big_bytes
            result.categories['big_files'] = big_files
            
            # Registrar acciones de emergencia
            logger.critical(f"Limpieza de emergencia completada: {result.files_removed} archivos, {format_size(result.bytes_freed)}")
            
            # Verificar si fue suficiente
            post_status = self._check_storage_status()
            if post_status in [StorageStatus.EMERGENCY, StorageStatus.CRITICAL]:
                logger.critical(f"¡ALERTA! La limpieza de emergencia no liberó suficiente espacio. Estado: {post_status}")
            
            return result
            
        except Exception as e:
            error_msg = f"Error en limpieza de emergencia: {e}"
            logger.error(error_msg, exc_info=True)
            result.errors.append(error_msg)
            return result
    
    def _cleanup_temp_directory(self, directory: str, max_age_hours: int,
                               respect_age: bool = True,
                               include_patterns: Optional[List[str]] = None) -> Tuple[int, int]:
        """
        Limpia un directorio temporal con opciones avanzadas
        
        Args:
            directory: Directorio a limpiar
            max_age_hours: Edad máxima en horas
            respect_age: Si False, ignora la edad y limpia todo
            include_patterns: Patrones de archivos a incluir
            
        Returns:
            Tuple[int, int]: (Archivos eliminados, Bytes liberados)
        """
        if not os.path.exists(directory) or not os.path.isdir(directory):
            logger.warning(f"Directorio no existe o no es accesible: {directory}")
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
                
                # Verificar si está protegido
                if self._is_protected_file(filename, file_path):
                    continue
                
                # Verificar patrones de inclusión si se especifican
                if include_patterns and not any(re.search(pattern, filename) for pattern in include_patterns):
                    continue
                
                try:
                    # Verificar edad si se requiere
                    if respect_age:
                        file_age = current_time - os.path.getmtime(file_path)
                        if file_age <= max_age_seconds:
                            continue
                    
                    # Eliminar archivo
                    file_size = os.path.getsize(file_path)
                    if safe_delete_file(file_path):
                        files_removed += 1
                        bytes_freed += file_size
                except Exception as e:
                    logger.error(f"Error procesando archivo {file_path}: {e}")
        
        return files_removed, bytes_freed
    
    def _is_protected_file(self, filename: str, file_path: str) -> bool:
        """
        Verifica si un archivo está protegido contra eliminación
        
        Args:
            filename: Nombre del archivo
            file_path: Ruta completa al archivo
            
        Returns:
            bool: True si está protegido, False si se puede eliminar
        """
        # Verificar extensiones protegidas
        ext = os.path.splitext(filename)[1]
        if ext in self.protected_extensions:
            return True
        
        # Verificar patrones protegidos
        for pattern in self.protected_patterns:
            if re.search(pattern, file_path):
                return True
        
        return False
    
    def _cleanup_big_files(self, min_size_mb: int = 100) -> Tuple[int, int]:
        """
        Elimina archivos grandes en directorios de almacenamiento
        
        Args:
            min_size_mb: Tamaño mínimo en MB para considerar archivo grande
            
        Returns:
            Tuple[int, int]: (Archivos eliminados, Bytes liberados)
        """
        min_size = min_size_mb * 1024 * 1024  # Convertir a bytes
        files_removed = 0
        bytes_freed = 0
        
        # Directorios a revisar
        directories = [self.storage_path, self.temp_dir] + self.cleanup_paths
        
        # Recopilar archivos grandes
        big_files = []
        
        for directory in directories:
            if not os.path.exists(directory) or not os.path.isdir(directory):
                continue
                
            for root, _, files in os.walk(directory):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    
                    # Ignorar archivos protegidos
                    if self._is_protected_file(filename, file_path):
                        continue
                    
                    try:
                        file_size = os.path.getsize(file_path)
                        if file_size >= min_size:
                            # Guardar info para ordenar por tamaño
                            big_files.append((file_path, file_size))
                    except Exception as e:
                        logger.error(f"Error verificando tamaño de {file_path}: {e}")
        
        # Ordenar por tamaño (mayor a menor)
        big_files.sort(key=lambda x: x[1], reverse=True)
        
        # Eliminar archivos grandes, empezando por los más grandes
        for file_path, file_size in big_files:
            if safe_delete_file(file_path):
                files_removed += 1
                bytes_freed += file_size
                logger.info(f"Eliminado archivo grande: {file_path} ({format_size(file_size)})")
        
        return files_removed, bytes_freed
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtener estado actual del servicio
        
        Returns:
            Dict: Estado del servicio
        """
        return {
            'active': self.thread is not None and self.thread.is_alive(),
            'paused': self.pause_flag.is_set(),
            'last_run': self.last_run_time.isoformat() if self.last_run_time else None,
            'run_count': self.run_count,
            'total_files_cleaned': self.total_files_cleaned,
            'total_space_freed': format_size(self.total_bytes_freed),
            'storage_status': self._check_storage_status(),
            'interval_minutes': self.interval / 60,
            'max_file_age_hours': self.max_file_age,
            'history': self.history[-5:],  # Últimos 5 resultados
            'next_scheduled_run': (self.last_run_time + timedelta(seconds=self.interval)).isoformat() if self.last_run_time else None
        }
