import functools
import json
import logging
import uuid
import time
import threading
import queue
import requests
import os
import psutil
import signal
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from flask import request, jsonify
from jsonschema import validate, ValidationError, FormatChecker
import config

# Configurar logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=getattr(logging, config.LOG_LEVEL, "INFO")
)
logger = logging.getLogger(__name__)

# Cola de tareas global
task_queue = queue.PriorityQueue()

# Diccionario para seguimiento de tareas en progreso
tasks_in_progress = {}

# Lock para acceso seguro a tasks_in_progress
tasks_lock = threading.Lock()

# Flag para indicar si los workers deben detenerse
shutdown_flag = threading.Event()

# Thread pool para procesamiento
executor = None

class TaskPriority:
    """Niveles de prioridad para tareas en la cola"""
    HIGH = 0
    NORMAL = 5
    LOW = 10
    RETRY = 8  # Prioridad para tareas que están siendo reintentadas

def validate_payload(schema):
    """
    Decorador para validar el payload JSON según un esquema.
    
    Args:
        schema (dict): Esquema JSON para validación.
    """
    def decorator(f):
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            # Intentar obtener y validar payload JSON
            try:
                payload = request.get_json()
                if not payload:
                    return jsonify({"status": "error", "error": "No se proporcionó payload JSON"}), 400
                
                # Validar contra el esquema
                validate(instance=payload, schema=schema, format_checker=FormatChecker())
                
                # Validación adicional de URLs si están presentes
                for key, value in payload.items():
                    if isinstance(value, str) and key.endswith('_url'):
                        if not is_valid_url(value):
                            return jsonify({
                                "status": "error", 
                                "error": f"URL inválida: {key}",
                                "detail": "La URL debe comenzar con http:// o https:// y ser válida"
                            }), 400
                    
                    # Validar arrays de URLs
                    if isinstance(value, list) and key.endswith('_urls'):
                        for i, url in enumerate(value):
                            if isinstance(url, str) and not is_valid_url(url):
                                return jsonify({
                                    "status": "error", 
                                    "error": f"URL inválida en {key}[{i}]",
                                    "detail": "La URL debe comenzar con http:// o https:// y ser válida"
                                }), 400
                
            except ValidationError as e:
                # Error de validación de esquema
                path = ".".join(str(p) for p in e.path) if e.path else "desconocido"
                return jsonify({
                    "status": "error", 
                    "error": f"Error de validación en {path}",
                    "detail": str(e)
                }), 400
            except Exception as e:
                # Error general al procesar el payload
                logger.error(f"Error al procesar payload: {str(e)}")
                return jsonify({
                    "status": "error", 
                    "error": f"Error al procesar payload: {str(e)}"
                }), 400
            
            # Si la validación es exitosa, continuar con la función original
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def is_valid_url(url):
    """
    Validación básica de URL
    
    Args:
        url (str): URL a validar
        
    Returns:
        bool: True si la URL es válida
    """
    if not isinstance(url, str):
        return False
    
    try:
        return url.startswith(('http://', 'https://'))
    except:
        return False

def queue_task_wrapper(bypass_queue=False, priority=TaskPriority.NORMAL, max_retries=3):
    """
    Decorador para gestionar tareas en cola o procesamiento inmediato.
    
    Args:
        bypass_queue (bool): Si True, procesa inmediatamente sin encolar.
        priority (int): Prioridad de la tarea (menor número = mayor prioridad)
        max_retries (int): Número máximo de reintentos automáticos
    """
    def decorator(f):
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            # Obtener datos del request
            data = request.get_json()
            
            # Generar ID único para la tarea si no se proporciona
            job_id = data.get('id', str(uuid.uuid4()))
            
            # Verificar si la cola está llena
            if not bypass_queue and task_queue.qsize() >= config.MAX_QUEUE_LENGTH:
                logger.warning(f"Cola llena, rechazando tarea {job_id}")
                return jsonify({
                    "status": "error",
                    "job_id": job_id,
                    "error": "Servidor ocupado. La cola de procesamiento está llena. Intente más tarde."
                }), 503  # Service Unavailable
            
            if bypass_queue:
                # Procesamiento inmediato
                try:
                    start_time = time.time()
                    
                    # Registrar tarea como en procesamiento
                    with tasks_lock:
                        tasks_in_progress[job_id] = {
                            "status": "processing",
                            "created_at": start_time,
                            "started_at": start_time,
                            "endpoint": request.path,
                            "bypass_queue": True
                        }
                    
                    # Ejecutar función de procesamiento
                    result, endpoint, status_code = f(job_id, data)
                    
                    # Actualizar estado a completado
                    completion_time = time.time()
                    processing_time = completion_time - start_time
                    
                    with tasks_lock:
                        if job_id in tasks_in_progress:
                            tasks_in_progress[job_id].update({
                                "status": "completed",
                                "completed_at": completion_time,
                                "processing_time": processing_time,
                                "result": result
                            })
                    
                    logger.info(f"Tarea {job_id} completada directamente en {processing_time:.2f}s")
                    
                    return jsonify({
                        "status": "success",
                        "job_id": job_id,
                        "result": result,
                        "endpoint": endpoint,
                        "processing_time": f"{processing_time:.2f}s"
                    }), status_code
                
                except Exception as e:
                    logger.error(f"Error procesando tarea {job_id} directamente: {str(e)}", exc_info=True)
                    
                    # Actualizar estado a error
                    with tasks_lock:
                        if job_id in tasks_in_progress:
                            tasks_in_progress[job_id].update({
                                "status": "error",
                                "error": str(e),
                                "completed_at": time.time()
                            })
                    
                    return jsonify({
                        "status": "error",
                        "job_id": job_id,
                        "error": str(e)
                    }), 500
            else:
                # Añadir a la cola de procesamiento
                webhook_url = data.get('webhook_url')
                
                # Registrar tarea como pendiente
                with tasks_lock:
                    tasks_in_progress[job_id] = {
                        "status": "queued",
                        "created_at": time.time(),
                        "webhook_url": webhook_url,
                        "endpoint": request.path,
                        "priority": priority,
                        "max_retries": max_retries,
                        "retry_count": 0,
                        "payload_size": len(json.dumps(data))
                    }
                
                # Añadir a la cola con prioridad
                task_queue.put((
                    priority,  # Prioridad
                    time.time(),  # Timestamp para desempate
                    {
                        "job_id": job_id,
                        "function": f,
                        "data": data,
                        "retry_count": 0,
                        "max_retries": max_retries,
                        "priority": priority,
                        "created_at": time.time()
                    }
                ))
                
                queue_size = task_queue.qsize()
                logger.info(f"Tarea {job_id} añadida a cola para {request.path} (prioridad: {priority}, tamaño cola: {queue_size})")
                
                # Estimar tiempo de espera (muy aproximado)
                wait_estimate = "< 1 minuto"
                if queue_size > 10:
                    wait_estimate = "1-5 minutos"
                if queue_size > 30:
                    wait_estimate = "5-15 minutos"
                if queue_size > 50:
                    wait_estimate = "> 15 minutos"
                
                # Retornar respuesta inmediata con job_id
                return jsonify({
                    "status": "queued",
                    "job_id": job_id,
                    "message": "Tarea añadida a la cola de procesamiento",
                    "queue_position": queue_size,
                    "estimated_wait": wait_estimate,
                    "queue_status_url": f"/jobs/{job_id}"
                }), 202
        
        return decorated_function
    return decorator

def is_recoverable_error(error):
    """
    Determina si un error es recuperable y debería reintentarse.
    
    Args:
        error (Exception): El error a evaluar
        
    Returns:
        bool: True si el error es recuperable, False en caso contrario
    """
    # Errores de red/conexión
    if isinstance(error, (requests.RequestException, ConnectionError, TimeoutError, OSError)):
        if isinstance(error, FileNotFoundError):
            return False  # Los archivos que no existen no se van a recuperar
        return True
    
    # Errores de recursos temporales
    if hasattr(error, 'errno') and error.errno in (35, 36, 37, 11):  # EAGAIN, EWOULDBLOCK, etc.
        return True
    
    # Errores específicos de FFmpeg que podrían ser temporales
    if hasattr(error, 'stderr') and isinstance(error.stderr, str):
        recoverable_patterns = [
            'resource temporarily unavailable',
            'temporary failure',
            'retry',
            'connection reset',
            'network is unreachable',
            'input/output error',
            'disk quota exceeded',
            'no space left'
        ]
        if any(pattern in error.stderr.lower() for pattern in recoverable_patterns):
            return True
    
    # Verificar mensaje de error para patrones comunes de errores recuperables
    error_str = str(error).lower()
    recoverable_patterns = [
        'timeout', 
        'temporary', 
        'retriable', 
        'retry', 
        'connection',
        'network',
        'unavailable',
        'busy',
        'overload',
        'capacity',
        'throttled',
        'quota'
    ]
    
    return any(pattern in error_str for pattern in recoverable_patterns)

def process_task(task):
    """
    Procesa una tarea de la cola.
    
    Args:
        task (dict): Información de la tarea a procesar
        
    Returns:
        tuple: (éxito, resultado, mensaje)
    """
    job_id = task["job_id"]
    function = task["function"]
    data = task["data"]
    retry_count = task.get("retry_count", 0)
    
    # Actualizar estado de la tarea
    with tasks_lock:
        if job_id in tasks_in_progress:
            tasks_in_progress[job_id].update({
                "status": "processing",
                "started_at": time.time(),
                "retry_count": retry_count
            })
    
    logger.info(f"Procesando tarea {job_id} (intento: {retry_count+1})")
    
    try:
        # Establecer un timeout máximo para la tarea
        max_time = getattr(config, 'MAX_PROCESSING_TIME', 1800)  # 30 minutos por defecto
        
        # Ejecutar la función de procesamiento
        start_time = time.time()
        
        result, endpoint, status_code = function(job_id, data)
        processing_time = time.time() - start_time
        
        # Actualizar estado de finalización exitosa
        with tasks_lock:
            if job_id in tasks_in_progress:
                tasks_in_progress[job_id].update({
                    "status": "completed",
                    "result": result,
                    "completed_at": time.time(),
                    "processing_time": processing_time
                })
        
        # Enviar webhook si se especificó
        webhook_url = data.get('webhook_url')
        if webhook_url:
            send_webhook(webhook_url, {
                "status": "completed",
                "job_id": job_id,
                "result": result,
                "endpoint": endpoint,
                "processing_time": f"{processing_time:.2f}s"
            })
        
        logger.info(f"Tarea {job_id} completada exitosamente en {processing_time:.2f}s")
        return True, result, None
        
    except Exception as e:
        logger.error(f"Error procesando tarea {job_id}: {str(e)}", exc_info=True)
        
        # Determinar si el error es recuperable
        recoverable = is_recoverable_error(e)
        max_retries = task.get("max_retries", 3)
        
        # Implementar lógica de reintentos para errores recuperables
        if recoverable and retry_count < max_retries:
            # Calcular delay para el reintento con backoff exponencial
            retry_delay = min(60, 2 ** retry_count)  # Máximo 60 segundos
            
            # Actualizar estado para reintento
            with tasks_lock:
                if job_id in tasks_in_progress:
                    tasks_in_progress[job_id].update({
                        "status": "retrying",
                        "retry_count": retry_count + 1,
                        "last_error": str(e),
                        "retry_after": time.time() + retry_delay
                    })
            
            logger.info(f"Tarea {job_id} programada para reintento {retry_count + 1}/{max_retries} en {retry_delay}s")
            
            # La tarea será reencolada por el worker principal
            return False, None, {
                "error": str(e),
                "recoverable": True,
                "retry_count": retry_count + 1,
                "retry_delay": retry_delay
            }
        
        # Error terminal
        with tasks_lock:
            if job_id in tasks_in_progress:
                tasks_in_progress[job_id].update({
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "completed_at": time.time(),
                    "recoverable": recoverable
                })
        
        # Enviar webhook de error si se especificó
        webhook_url = data.get('webhook_url')
        if webhook_url:
            send_webhook(webhook_url, {
                "status": "error",
                "job_id": job_id,
                "error": str(e),
                "endpoint": request.path
            })
        
        return False, None, {
            "error": str(e),
            "recoverable": False
        }

def worker_process():
    """
    Procesador principal de cola de tareas con manejo mejorado de errores y reintentos.
    """
    logger.info("Procesador de cola iniciado")
    
    while not shutdown_flag.is_set():
        try:
            # Esperar hasta 1 segundo por una nueva tarea
            try:
                priority, _, task = task_queue.get(timeout=1)
                job_id = task["job_id"]
            except queue.Empty:
                continue
            
            # Procesar la tarea
            success, result, error_info = process_task(task)
            
            if not success and error_info and error_info.get("recoverable", False):
                # Reencolar la tarea con delay y prioridad ajustada
                retry_count = error_info.get("retry_count", 1)
                retry_delay = error_info.get("retry_delay", 5)
                
                # Esperar el delay antes de reencolar
                time.sleep(retry_delay)
                
                # Actualizar y reencolar la tarea
                task["retry_count"] = retry_count
                task_queue.put((
                    TaskPriority.RETRY,  # Usar prioridad de reintento
                    time.time(),          # Timestamp para desempate
                    task
                ))
                
                logger.info(f"Tarea {job_id} reencolada para reintento {retry_count}")
            
            # Marcar tarea como completada en la cola
            task_queue.task_done()
            
        except Exception as e:
            logger.error(f"Error crítico en procesador de cola: {str(e)}", exc_info=True)
            # Pequeña pausa antes de intentar nuevamente
            time.sleep(1)

def send_webhook(webhook_url, data, max_retries=3, retry_delay=2):
    """
    Envía una notificación webhook con reintentos.
    
    Args:
        webhook_url (str): URL del webhook
        data (dict): Datos a enviar
        max_retries (int): Número máximo de reintentos
        retry_delay (int): Retraso inicial entre reintentos (segundos)
    
    Returns:
        bool: True si se envió correctamente
    """
    def _send_webhook_async():
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'VideoAPI-Webhook/1.0'
        }
        
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    webhook_url,
                    headers=headers,
                    data=json.dumps(data),
                    timeout=10
                )
                
                if 200 <= response.status_code < 300:
                    logger.info(f"Webhook enviado exitosamente a {webhook_url}")
                    return True
                else:
                    logger.warning(f"Webhook falló con estado {response.status_code}: {response.text}")
                    
                    if attempt < max_retries:
                        # Calcular retraso con backoff exponencial
                        current_delay = retry_delay * (2 ** attempt)
                        logger.info(f"Reintentando webhook en {current_delay}s (intento {attempt+1}/{max_retries})")
                        time.sleep(current_delay)
                    else:
                        logger.error(f"Webhook falló después de {max_retries} intentos")
                        return False
            
            except Exception as e:
                logger.error(f"Error enviando webhook: {str(e)}")
                
                if attempt < max_retries:
                    # Calcular retraso con backoff exponencial
                    current_delay = retry_delay * (2 ** attempt)
                    logger.info(f"Reintentando webhook en {current_delay}s (intento {attempt+1}/{max_retries})")
                    time.sleep(current_delay)
                else:
                    logger.error(f"Webhook falló después de {max_retries} intentos debido a excepción: {str(e)}")
                    return False
        
        return False
    
    # Ejecutar webhook en un thread separado para no bloquear
    threading.Thread(target=_send_webhook_async, daemon=True).start()
    return True

def start_queue_processors(num_workers=4):
    """
    Inicia múltiples procesadores de cola en threads separados.
    
    Args:
        num_workers (int): Número de workers para procesar la cola.
    """
    global executor
    
    if executor is not None:
        logger.warning("Los procesadores de cola ya están en ejecución.")
        return
    
    # Determinar número óptimo de workers si no se especifica
    if num_workers <= 0:
        cpu_count = os.cpu_count() or 4
        available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
        
        # Ajustar según recursos disponibles
        if available_memory < 2:  # Menos de 2GB disponibles
            num_workers = max(1, cpu_count // 2)
        elif available_memory < 4:  # Menos de 4GB disponibles
            num_workers = max(2, cpu_count - 1)
        else:
            num_workers = cpu_count
        
        # Limitar por configuración
        max_workers = getattr(config, 'WORKER_PROCESSES', 4)
        num_workers = min(num_workers, max_workers)
    
    # Crear thread pool
    executor = ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="QueueWorker")
    
    # Iniciar workers
    for _ in range(num_workers):
        executor.submit(worker_process)
    
    logger.info(f"Iniciados {num_workers} procesadores de cola")
    
    # Registrar manejador de señales para shutdown limpio
    try:
        signal.signal(signal.SIGTERM, handle_shutdown)
        signal.signal(signal.SIGINT, handle_shutdown)
    except (ImportError, AttributeError):
        # Windows no tiene SIGTERM
        pass

def handle_shutdown(sig, frame):
    """Manejador de señales para shutdown limpio"""
    logger.info(f"Señal de apagado recibida ({sig}), deteniendo workers...")
    shutdown_flag.set()
    
    # Esperar a que los workers terminen (con timeout)
    if executor:
        executor.shutdown(wait=True, cancel_futures=True)
    
    logger.info("Todos los workers detenidos correctamente")

def stop_queue_processors():
    """
    Detiene los procesadores de cola de manera segura.
    """
    global executor
    
    if executor is None:
        logger.warning("No hay procesadores de cola en ejecución.")
        return
    
    # Señalizar a los workers que deben detenerse
    shutdown_flag.set()
    
    # Esperar a que los workers terminen (con timeout)
    executor.shutdown(wait=True, cancel_futures=True)
    executor = None
    
    # Restablecer flag para futuros usos
    shutdown_flag.clear()
    
    logger.info("Procesadores de cola detenidos correctamente")

def get_job_status(job_id):
    """
    Obtiene el estado actual de un trabajo.
    
    Args:
        job_id (str): ID del trabajo a consultar.
    
    Returns:
        dict: Información del estado del trabajo o None si no existe.
    """
    with tasks_lock:
        if job_id in tasks_in_progress:
            # Crear una copia para evitar problemas de concurrencia
            status_copy = tasks_in_progress[job_id].copy()
            
            # Calcular tiempos y progreso si está disponible
            if 'created_at' in status_copy:
                status_copy['age_seconds'] = time.time() - status_copy['created_at']
                
                if status_copy['status'] == 'completed' and 'completed_at' in status_copy:
                    status_copy['processing_time_seconds'] = status_copy['completed_at'] - status_copy.get('started_at', status_copy['created_at'])
                
                # Formato legible para tiempos
                for key in ['age_seconds', 'processing_time_seconds']:
                    if key in status_copy:
                        status_copy[f"{key}_formatted"] = format_time_seconds(status_copy[key])
            
            # Estimar tiempo de espera para tareas en cola
            if status_copy['status'] == 'queued':
                # Estimar basado en posición en cola y tiempo promedio
                queue_pos = 0
                with task_queue.mutex:
                    for i, (_, _, task) in enumerate(task_queue.queue):
                        if task["job_id"] == job_id:
                            queue_pos = i + 1
                            break
                
                status_copy['queue_position'] = queue_pos
                
                # Cálculo básico del tiempo estimado de espera
                avg_processing_time = 30  # 30 segundos por defecto
                status_copy['estimated_wait_seconds'] = queue_pos * avg_processing_time
                status_copy['estimated_wait'] = format_time_seconds(status_copy['estimated_wait_seconds'])
            
            return status_copy
        else:
            return None

def cleanup_completed_tasks(max_age_hours=6):
    """
    Limpia tareas completadas o con error más antiguas que max_age_hours.
    
    Args:
        max_age_hours (int): Edad máxima en horas antes de limpiar.
        
    Returns:
        int: Número de tareas limpiadas
    """
    max_age_seconds = max_age_hours * 3600
    current_time = time.time()
    
    with tasks_lock:
        to_remove = []
        
        for job_id, task_info in tasks_in_progress.items():
            status = task_info.get("status", "")
            
            if status in ["completed", "error"]:
                completed_at = task_info.get("completed_at", 0)
                if current_time - completed_at > max_age_seconds:
                    to_remove.append(job_id)
            
            # También limpiar tareas muy antiguas en cualquier estado (posibles tareas huérfanas)
            created_at = task_info.get("created_at", 0)
            if current_time - created_at > max_age_seconds * 2:  # El doble de tiempo para tareas huérfanas
                if job_id not in to_remove:
                    to_remove.append(job_id)
                    logger.warning(f"Limpiando tarea huérfana {job_id} en estado {status}")
        
        for job_id in to_remove:
            del tasks_in_progress[job_id]
    
    if to_remove:
        logger.info(f"Limpiadas {len(to_remove)} tareas completadas/huérfanas")
        return len(to_remove)
    
    return 0

def get_queue_stats():
    """
    Obtiene estadísticas actuales de la cola de tareas.
    
    Returns:
        dict: Estadísticas de tareas y cola
    """
    with tasks_lock:
        total = len(tasks_in_progress)
        queued = sum(1 for t in tasks_in_progress.values() if t.get('status') == 'queued')
        processing = sum(1 for t in tasks_in_progress.values() if t.get('status') == 'processing')
        completed = sum(1 for t in tasks_in_progress.values() if t.get('status') == 'completed')
        errors = sum(1 for t in tasks_in_progress.values() if t.get('status') == 'error')
        retrying = sum(1 for t in tasks_in_progress.values() if t.get('status') == 'retrying')
    
    # Obtener carga del sistema
    try:
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
    except:
        cpu_usage = 0
        memory_usage = 0
    
    # Estadísticas de colas por prioridad
    queue_by_priority = {}
    with task_queue.mutex:
        queue_items = list(task_queue.queue)
        for priority, _, _ in queue_items:
            queue_by_priority[priority] = queue_by_priority.get(priority, 0) + 1
    
    return {
        'total_tasks': total,
        'queue_size': task_queue.qsize(),
        'tasks_by_status': {
            'queued': queued,
            'processing': processing,
            'completed': completed,
            'error': errors,
            'retrying': retrying
        },
        'queue_by_priority': queue_by_priority,
        'system_load': {
            'cpu_percent': cpu_usage,
            'memory_percent': memory_usage
        },
        'timestamp': datetime.now().isoformat()
    }

def format_time_seconds(seconds):
    """
    Formatea un tiempo en segundos a un formato legible.
    
    Args:
        seconds (float): Tiempo en segundos
        
    Returns:
        str: Tiempo formateado
    """
    if seconds < 60:
        return f"{seconds:.1f} segundos"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutos"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} horas"

def cancel_job(job_id):
    """
    Cancela un trabajo pendiente.
    
    Args:
        job_id (str): ID del trabajo a cancelar
        
    Returns:
        bool: True si se canceló correctamente, False si no existe o no se puede cancelar
    """
    with tasks_lock:
        if job_id not in tasks_in_progress:
            return False
        
        status = tasks_in_progress[job_id].get('status')
        if status != 'queued':
            return False
        
        # Marcar como cancelado
        tasks_in_progress[job_id]['status'] = 'cancelled'
        tasks_in_progress[job_id]['cancelled_at'] = time.time()
    
    # Nota: La tarea seguirá en la cola, pero será ignorada cuando se procesaría
    logger.info(f"Tarea {job_id} marcada como cancelada")
    return True

def pause_processing():
    """
    Pausa temporalmente el procesamiento de nuevas tareas (útil para mantenimiento)
    """
    global executor
    
    if executor is None:
        logger.warning("No hay procesadores de cola en ejecución.")
        return False
    
    # Señalizar a los workers que deben pausar
    shutdown_flag.set()
    
    # Esperar a que terminen los workers actuales
    executor.shutdown(wait=True, cancel_futures=True)
    executor = None
    
    logger.info("Procesamiento de cola pausado")
    return True

def resume_processing(num_workers=None):
    """
    Reanuda el procesamiento de tareas después de una pausa
    """
    global executor
    
    if executor is not None:
        logger.warning("Los procesadores de cola ya están en ejecución.")
        return False
    
    # Restablecer flag
    shutdown_flag.clear()
    
    # Usar número de workers anterior o el especificado
    if num_workers is None:
        num_workers = getattr(config, 'WORKER_PROCESSES', 4)
    
    # Iniciar nuevos workers
    start_queue_processors(num_workers)
    
    logger.info(f"Procesamiento de cola reanudado con {num_workers} workers")
    return True

def get_task_history(limit=100, status=None, endpoint=None):
    """
    Obtiene el historial de tareas
    
    Args:
        limit (int): Número máximo de tareas a retornar
        status (str): Filtrar por estado
        endpoint (str): Filtrar por endpoint
        
    Returns:
        list: Lista de tareas
    """
    with tasks_lock:
        # Copiar tareas para evitar problemas de concurrencia
        all_tasks = []
        for job_id, task in tasks_in_progress.items():
            task_copy = task.copy()
            task_copy['job_id'] = job_id
            all_tasks.append(task_copy)
    
    # Aplicar filtros
    if status:
        all_tasks = [t for t in all_tasks if t.get('status') == status]
    
    if endpoint:
        all_tasks = [t for t in all_tasks if t.get('endpoint') == endpoint]
    
    # Ordenar por tiempo de creación (más recientes primero)
    all_tasks.sort(key=lambda x: x.get('created_at', 0), reverse=True)
    
    # Limitar número de resultados
    return all_tasks[:limit]
