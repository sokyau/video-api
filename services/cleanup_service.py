import threading
import time
import logging
from services.local_storage import cleanup_old_files
import config

logger = logging.getLogger(__name__)

class CleanupService:
    """Servicio para limpieza periódica de archivos antiguos"""
    
    def __init__(self, interval_minutes=30):
        """
        Inicializar servicio con intervalo en minutos
        
        Args:
            interval_minutes (int): Intervalo de ejecución en minutos
        """
        self.interval = interval_minutes * 60  # Convertir a segundos
        self.shutdown_flag = threading.Event()
        self.thread = None
    
    def start(self):
        """Iniciar servicio en un hilo separado"""
        if self.thread is not None:
            logger.warning("Cleanup service already running")
            return
        
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info(f"Cleanup service started (interval: {self.interval/60} minutes)")
    
    def stop(self):
        """Detener servicio"""
        if self.thread is None:
            logger.warning("Cleanup service not running")
            return
        
        self.shutdown_flag.set()
        self.thread.join()
        self.thread = None
        logger.info("Cleanup service stopped")
    
    def _run(self):
        """Ejecutar ciclo de limpieza"""
        while not self.shutdown_flag.is_set():
            try:
                files_removed, bytes_freed = cleanup_old_files()
                if files_removed > 0:
                    logger.info(f"Cleanup: removed {files_removed} files ({bytes_freed/1024/1024:.2f} MB)")
            except Exception as e:
                logger.error(f"Error in cleanup cycle: {e}")
            
            # Esperar hasta el próximo ciclo o hasta recibir señal de apagado
            self.shutdown_flag.wait(timeout=self.interval)

    def run_now(self):
        """
        Ejecutar limpieza inmediatamente
        
        Returns:
            tuple: (Número de archivos eliminados, Bytes liberados)
        """
        try:
            return cleanup_old_files()
        except Exception as e:
            logger.error(f"Error in immediate cleanup: {e}")
            return 0, 0
