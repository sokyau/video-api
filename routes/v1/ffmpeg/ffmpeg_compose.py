from flask import Blueprint, jsonify
from app_utils import validate_payload, queue_task_wrapper
import logging
import os
import subprocess
import json
import time
from services.authentication import authenticate
from services.file_management import download_file, generate_temp_filename
from services.local_storage import store_file
import config

v1_ffmpeg_compose_bp = Blueprint('v1_ffmpeg_compose', __name__)
logger = logging.getLogger(__name__)

def get_optimal_thread_count():
    """Determinar el número óptimo de hilos basado en la carga del sistema"""
    try:
        cpu_count = os.cpu_count() or 4
        # Verificar carga del sistema
        load = os.getloadavg()[0]
        # Ajustar número de hilos basado en carga
        if load > cpu_count * 0.8:
            return max(2, cpu_count // 2)  # Carga alta, reducir hilos
        return min(cpu_count, config.FFMPEG_THREADS)
    except:
        return config.FFMPEG_THREADS  # Valor de respaldo de configuración

def validate_ffmpeg_parameters(params, more_params=None):
    """
    Valida los parámetros de FFmpeg para prevenir inyección de comandos
    
    Args:
        params: Parámetro individual o lista de parámetros
        more_params: Parámetros adicionales (opcional)
    
    Raises:
        ValueError: Si se detecta un parámetro potencialmente peligroso
    """
    # Lista de opciones potencialmente peligrosas
    dangerous_options = [
        '-f lavfi',  # Podría usarse para generar contenido ilimitado
        'system', 'exec',  # Ejecución de comandos del sistema
        '`', '$(', '${',  # Expansión de shell
        '&&', '||', ';',  # Operadores de concatenación
        '>', '>>', '<',  # Redirección
        'http://', 'https://', 'ftp://',  # URLs no controladas en parámetros
    ]
    
    # Lista de opciones seguras conocidas
    safe_options = [
        '-c:v', '-c:a', '-r', '-b:v', '-b:a', '-vf', '-af', '-t', '-to',
        '-crf', '-preset', '-profile:v', '-pix_fmt', '-movflags'
    ]
    
    # Convertir a lista si es un string
    if isinstance(params, str):
        params = [params]
    
    # Añadir more_params si se proporcionan
    all_params = params.copy() if params else []
    if more_params:
        if isinstance(more_params, str):
            all_params.append(more_params)
        else:
            all_params.extend(more_params)
    
    # Verificar cada parámetro
    for param in all_params:
        param_str = str(param).lower()
        
        # Verificar si es una opción segura conocida
        is_safe_option = False
        for safe_opt in safe_options:
            if param_str.startswith(safe_opt):
                is_safe_option = True
                break
        
        if is_safe_option:
            continue
            
        # Verificar opciones peligrosas
        for dangerous in dangerous_options:
            if dangerous in param_str:
                logger.warning(f"Parámetro FFmpeg peligroso detectado: {param}")
                raise ValueError(f"Parámetro FFmpeg potencialmente inseguro: {param}")
    
    # Verificación adicional para filter_complex
    if '-filter_complex' in all_params or '-vf' in all_params:
        idx = -1
        if '-filter_complex' in all_params:
            idx = all_params.index('-filter_complex') + 1
        elif '-vf' in all_params:
            idx = all_params.index('-vf') + 1
        
        if idx >= 0 and idx < len(all_params):
            filter_str = all_params[idx]
            if isinstance(filter_str, str):
                # Verificar comandos peligrosos en filtros
                danger_in_filter = any(d in filter_str for d in ['system', 'exec', '`', '$(', '${', ';'])
                if danger_in_filter:
                    logger.warning(f"Filtro FFmpeg peligroso detectado: {filter_str}")
                    raise ValueError(f"Filtro FFmpeg potencialmente inseguro: {filter_str}")

def verify_media_file(file_path):
    """
    Verifica que un archivo multimedia sea válido mediante FFprobe
    
    Args:
        file_path (str): Ruta al archivo multimedia
        
    Returns:
        bool: True si el archivo es válido, False si no
        
    Raises:
        ValueError: Si el archivo no es un archivo multimedia válido
    """
    try:
        cmd = [
            'ffprobe', 
            '-v', 'error', 
            '-select_streams', 'v:0', 
            '-show_entries', 'stream=codec_type', 
            '-of', 'json', 
            file_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        # Si el comando falló o no devolvió JSON válido
        if result.returncode != 0:
            raise ValueError(f"No es un archivo multimedia válido: {os.path.basename(file_path)}")
            
        # Intentar decodificar el JSON
        try:
            info = json.loads(result.stdout)
            # Verificar si tiene streams
            if 'streams' not in info or len(info['streams']) == 0:
                raise ValueError(f"No se encontraron streams multimedia en el archivo: {os.path.basename(file_path)}")
                
            return True
        except json.JSONDecodeError:
            raise ValueError(f"El archivo no es un archivo multimedia válido: {os.path.basename(file_path)}")
            
    except Exception as e:
        logger.error(f"Error verificando archivo multimedia {file_path}: {str(e)}")
        raise ValueError(f"Error validando archivo multimedia: {str(e)}")

@v1_ffmpeg_compose_bp.route('/v1/ffmpeg/ffmpeg_compose', methods=['POST'])
@authenticate
@validate_payload({
    "type": "object",
    "properties": {
        "inputs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "format": "uri"},
                    "options": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["url"]
            },
            "minItems": 1
        },
        "filter_complex": {"type": "string"},
        "output_options": {
            "type": "array",
            "items": {"type": "string"}
        },
        "webhook_url": {"type": "string", "format": "uri"},
        "id": {"type": "string"}
    },
    "required": ["inputs", "filter_complex"],
    "additionalProperties": False
})
@queue_task_wrapper(bypass_queue=False)
def ffmpeg_compose(job_id, data):
    inputs = data['inputs']
    filter_complex = data['filter_complex']
    output_options = data.get('output_options', [])
    webhook_url = data.get('webhook_url')
    id = data.get('id')

    logger.info(f"Job {job_id}: Received FFmpeg compose request with {len(inputs)} inputs")

    # Validar comando para evitar inyección
    validate_ffmpeg_parameters(filter_complex)
    if output_options:
        validate_ffmpeg_parameters(output_options)
    
    for input_data in inputs:
        if "options" in input_data and input_data["options"]:
            validate_ffmpeg_parameters(input_data["options"])

    try:
        # Descargar todos los archivos de entrada
        input_paths = []
        for i, input_data in enumerate(inputs):
            url = input_data["url"]
            try:
                file_path = download_file(url, config.TEMP_DIR)
                
                # Verificar que el archivo es válido
                verify_media_file(file_path)
                
                input_paths.append(file_path)
                logger.info(f"Job {job_id}: Downloaded input {i+1}/{len(inputs)}")
            except Exception as e:
                logger.error(f"Job {job_id}: Error downloading input {i+1}: {str(e)}")
                # Limpiar archivos ya descargados
                for path in input_paths:
                    if os.path.exists(path):
                        os.remove(path)
                raise RuntimeError(f"Error downloading input {i+1}: {str(e)}")

        # Generar archivo de salida
        output_path = generate_temp_filename(suffix=".mp4")

        # Construir comando FFmpeg
        cmd = ['ffmpeg']

        # Añadir entradas con sus opciones
        for i, (input_path, input_data) in enumerate(zip(input_paths, inputs)):
            if "options" in input_data and input_data["options"]:
                for option in input_data["options"]:
                    cmd.append(option)
            cmd.extend(['-i', input_path])

        # Añadir filtro complejo
        cmd.extend(['-filter_complex', filter_complex])

        # Añadir opciones de salida
        for option in output_options:
            cmd.append(option)

        # Añadir configuración de threads
        threads = get_optimal_thread_count()
        cmd.extend(['-threads', str(threads)])

        # Añadir salida y forzar sobreescritura
        cmd.extend(['-y', output_path])

        logger.info(f"Job {job_id}: Running FFmpeg command")
        logger.debug(f"Command: {' '.join(cmd)}")

        # Ejecutar FFmpeg con timeout
        max_execution_time = config.MAX_PROCESSING_TIME or 1800  # 30 minutos por defecto
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=max_execution_time
            )
            if result.returncode != 0:
                logger.error(f"FFmpeg command failed. Error: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
        except subprocess.TimeoutExpired:
            logger.error(f"FFmpeg command timed out after {max_execution_time} seconds")
            # Intentar terminar el proceso de FFmpeg que se quedó colgado
            for path in input_paths:
                if os.path.exists(path):
                    os.remove(path)
            if os.path.exists(output_path):
                os.remove(output_path)
            raise TimeoutError(f"FFmpeg processing timed out after {max_execution_time} seconds")

        logger.info(f"Job {job_id}: FFmpeg compose completed successfully in {time.time() - start_time:.2f} seconds")

        # Verificar que el archivo de salida existe y tiene tamaño
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("FFmpeg command completed but output file is missing or empty")

        # Almacenar archivo resultante
        file_url = store_file(output_path)
        logger.info(f"Job {job_id}: Output video stored locally: {file_url}")

        # Limpiar archivos temporales
        for path in input_paths:
            if os.path.exists(path):
                os.remove(path)
        
        os.remove(output_path)

        return file_url, "/v1/ffmpeg/ffmpeg_compose", 200

    except Exception as e:
        logger.error(f"Job {job_id}: Error during FFmpeg compose - {str(e)}")
        # Limpiar archivos temporales en caso de error
        if 'input_paths' in locals():
            for path in input_paths:
                if os.path.exists(path):
                    os.remove(path)
        
        if 'output_path' in locals() and os.path.exists(output_path):
            os.remove(output_path)
            
        # Determinar el código de estado HTTP adecuado según el error
        if isinstance(e, ValueError):
            return str(e), "/v1/ffmpeg/ffmpeg_compose", 400  # Bad Request para errores de validación
        elif isinstance(e, TimeoutError):
            return str(e), "/v1/ffmpeg/ffmpeg_compose", 408  # Request Timeout
        elif isinstance(e, subprocess.CalledProcessError):
            return f"Error de FFmpeg: {e.stderr}", "/v1/ffmpeg/ffmpeg_compose", 500
        else:
            return str(e), "/v1/ffmpeg/ffmpeg_compose", 500

@v1_ffmpeg_compose_bp.route('/v1/ffmpeg/status/<job_id>', methods=['GET'])
@authenticate
def get_job_status_endpoint(job_id):
    """Endpoint para consultar el estado de un trabajo"""
    from app_utils import get_job_status
    
    status = get_job_status(job_id)
    if status:
        return jsonify({
            "status": "success",
            "job_id": job_id,
            "job_status": status
        }), 200
    else:
        return jsonify({
            "status": "error",
            "job_id": job_id,
            "error": "Trabajo no encontrado"
        }), 404
