import functools
import json
import logging
import uuid
import time
import threading
import queue
import requests
import os
from flask import request, jsonify
from jsonschema import validate, ValidationError
import config

# Configurar logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=getattr(logging, config.LOG_LEVEL, "INFO")
)
logger = logging.getLogger(__name__)

# Cola de tareas global
task_queue = queue.Queue()

# Diccionario para seguimiento de tareas en progreso
tasks_in_progress = {}

# Lock para acceso seguro a tasks_in_progress
tasks_lock = threading.Lock()

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
                    return jsonify({"error": "No JSON payload provided"}), 400
                
                validate(instance=payload, schema=schema)
            except ValidationError as e:
                # Error de validación de esquema
                return jsonify({"error": str(e)}), 400
            except Exception as e:
                # Error general al procesar el payload
                return jsonify({"error": f"Error processing payload: {str(e)}"}), 400
            
            # Si la validación es exitosa, continuar con la función original
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def queue_task_wrapper(bypass_queue=False):
    """
    Decorador para gestionar tareas en cola o procesamiento inmediato.
    
    Args:
        bypass_queue (bool): Si True, procesa inmediatamente sin encolar.
    """
    def decorator(f):
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            # Obtener datos del request
            data = request.get_json()
            
            # Generar ID único para la tarea si no se proporciona
            job_id = data.get('id', str(uuid.uuid4()))
            
            if bypass_queue:
                # Procesamiento inmediato
                try:
                    result, endpoint, status_code = f(job_id, data)
                    return jsonify({
                        "status": "success",
                        "job_id": job_id,
                        "result": result,
                        "endpoint": endpoint
                    }), status_code
                except Exception as e:
                    logger.error(f"Error processing job {job_id}: {str(e)}")
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
                        "endpoint": request.path
                    }
                
                # Añadir a la cola
                task_queue.put({
                    "job_id": job_id,
                    "function": f,
                    "data": data,
                    "retry_count": 0
                })
                
                logger.info(f"Job {job_id} added to queue for endpoint {request.path}")
                
                # Retornar respuesta inmediata con job_id
                return jsonify({
                    "status": "queued",
                    "job_id": job_id,
                    "message": "Task has been queued for processing"
                }), 202
        
        return decorated_function
    return decorator

def process_queue():
    """
    Procesador de cola de tareas mejorado con reintentos y mejor manejo de errores.
    """
    logger.info("Queue processor started")
    
    while True:
        try:
            # Obtener tarea de la cola (bloquea hasta que haya una disponible)
            task = task_queue.get()
            
            job_id = task["job_id"]
            function = task["function"]
            data = task["data"]
            retry_count = task.get("retry_count", 0)
            
            # Actualizar estado de la tarea
            with tasks_lock:
                if job_id in tasks_in_progress:
                    tasks_in_progress[job_id]["status"] = "processing"
                    tasks_in_progress[job_id]["started_at"] = time.time()
            
            logger.info(f"Processing job {job_id} (retry: {retry_count})")
            
            try:
                # Ejecutar la función de procesamiento con límite de tiempo
                # Aquí podría implementarse un timeout pero requeriría threading adicional
                result, endpoint, status_code = function(job_id, data)
                
                # Actualizar estado de finalización exitosa
                with tasks_lock:
                    if job_id in tasks_in_progress:
                        tasks_in_progress[job_id]["status"] = "completed"
                        tasks_in_progress[job_id]["result"] = result
                        tasks_in_progress[job_id]["completed_at"] = time.time()
                
                # Enviar webhook si se especificó
                webhook_url = data.get('webhook_url')
                if webhook_url:
                    from services.webhook import send_webhook_notification
                    send_webhook_notification(webhook_url, {
                        "status": "completed",
                        "job_id": job_id,
                        "result": result,
                        "endpoint": endpoint
                    })
                
                logger.info(f"Job {job_id} completed successfully")
                
            except Exception as e:
                logger.error(f"Error processing job {job_id}: {str(e)}")
                
                # Implementar lógica de reintentos para errores recuperables
                if retry_count < 3 and is_recoverable_error(e):
                    # Reencolar con contador incrementado
                    task["retry_count"] = retry_count + 1
                    task_queue.put(task)
                    logger.info(f"Job {job_id} requeued for retry {retry_count + 1}/3")
                    
                    # Actualizar estado para reintento
                    with tasks_lock:
                        if job_id in tasks_in_progress:
                            tasks_in_progress[job_id]["status"] = "retrying"
                            tasks_in_progress[job_id]["retry_count"] = retry_count + 1
                    
                    # No marcar como completado en la cola ya que lo estamos reintentando
                    continue
                
                # Actualizar estado de error
                with tasks_lock:
                    if job_id in tasks_in_progress:
                        tasks_in_progress[job_id]["status"] = "error"
                        tasks_in_progress[job_id]["error"] = str(e)
                        tasks_in_progress[job_id]["completed_at"] = time.time()
                
                # Enviar webhook de error si se especificó
                webhook_url = data.get('webhook_url')
                if webhook_url:
                    from services.webhook import send_webhook_notification
                    send_webhook_notification(webhook_url, {
                        "status": "error",
                        "job_id": job_id,
                        "error": str(e),
                        "endpoint": endpoint if 'endpoint' in locals() else request.path
                    })
            
            # Marcar tarea como completada en la cola
            task_queue.task_done()
            
        except Exception as e:
            logger.error(f"Critical error in queue processor: {str(e)}")
            # Pequeña pausa antes de intentar nuevamente
            time.sleep(1)

def is_recoverable_error(error):
    """
    Determina si un error es recuperable y debería reintentarse.
    
    Args:
        error (Exception): El error a evaluar
        
    Returns:
        bool: True si el error es recuperable, False en caso contrario
    """
    # Errores de red, temporales del sistema de archivos, etc.
    if isinstance(error, (requests.RequestException, IOError, ConnectionError, TimeoutError)):
        return True
    
    # Errores de procesamiento FFmpeg que podrían ser temporales
    if hasattr(error, 'stderr') and isinstance(error.stderr, str):
        if any(msg in error.stderr for msg in ['resource temporarily unavailable', 'temporary failure', 'retry']):
            return True
    
    return False

def start_queue_processors(num_workers=4):
    """
    Inicia múltiples procesadores de cola en threads separados.
    
    Args:
        num_workers (int): Número de workers para procesar la cola.
    """
    for i in range(num_workers):
        thread = threading.Thread(target=process_queue, daemon=True)
        thread.start()
        logger.info(f"Started queue processor thread {i+1}")

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
                    status_copy['processing_time'] = status_copy['completed_at'] - status_copy['created_at']
                
            return status_copy
        else:
            return None

def cleanup_completed_tasks(max_age_hours=6):
    """
    Limpia tareas completadas o con error más antiguas que max_age_hours.
    
    Args:
        max_age_hours (int): Edad máxima en horas antes de limpiar.
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
                    logger.warning(f"Removing orphaned task {job_id} in state {status}")
        
        for job_id in to_remove:
            del tasks_in_progress[job_id]
    
    if to_remove:
        logger.info(f"Cleaned up {len(to_remove)} completed/orphaned tasks")
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
    
    return {
        'total_tasks': total,
        'queue_size': task_queue.qsize(),
        'tasks_by_status': {
            'queued': queued,
            'processing': processing,
            'completed': completed,
            'error': errors,
            'retrying': retrying
        }
    }
