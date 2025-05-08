from flask import Blueprint, jsonify, request
from app_utils import validate_payload, queue_task_wrapper, TaskPriority
import logging
import os
import subprocess
import json
import time
import tempfile
import re
import shlex
import psutil
from typing import List, Dict, Any, Optional, Tuple, Union
from services.authentication import authenticate
from services.file_management import download_file, generate_temp_filename, verify_file_integrity
from services.local_storage import store_file
import config

v1_ffmpeg_compose_bp = Blueprint('v1_ffmpeg_compose', __name__)
logger = logging.getLogger(__name__)

# Lista blanca de opciones FFmpeg seguras
SAFE_FFMPEG_OPTIONS = [
    # Entrada/salida general
    '-i', '-y', '-n', '-to', '-t', '-ss', '-sseof', '-fs',
    
    # Opciones de video
    '-vcodec', '-c:v', '-b:v', '-crf', '-preset', '-profile:v', '-level', 
    '-r', '-s', '-vf', '-pix_fmt', '-frames:v', '-pass', '-aspect',
    
    # Opciones de audio
    '-acodec', '-c:a', '-b:a', '-ar', '-ac', '-af', '-sample_fmt', '-frames:a',
    
    # Contenedor y mapeo
    '-map', '-map_metadata', '-f', '-movflags', '-metadata',
    
    # Filtros
    '-filter_complex', '-filter:v', '-filter:a',
    
    # Otros
    '-threads', '-nostats', '-v'
]

# Lista negra de patrones peligrosos
DANGEROUS_PATTERNS = [
    # Ejecución de comandos
    r'`.*`',              # Backticks
    r'\$\(.*\)',          # $(command)
    r'\$\{.*\}',          # ${var}
    r';.*',               # Separador de comandos
    r'\|.*',              # Pipe
    r'&&.*',              # Logical AND
    r'>.*',               # Redirección
    r'<.*',               # Redirección
    r'\\',                # Escape de shell
    
    # Filtros potencialmente peligrosos
    r'movie=',            # Acceso a archivos externos en filtros
    r'lavfi',             # Filtro lavfi puede ser peligroso
    r"system\s*\(",       # Llamadas al sistema
    r"exec\s*\(",         # Ejecución
    r"popen\s*\(",        # Pipes
    r"subprocess",        # Subprocesos
    
    # Acceso a archivos no controlados
    r"file:.*/",          # Protocolos de acceso a archivos
    r"concat:.*",         # Concatenación insegura
]

# Lista blanca de filtros FFmpeg seguros
SAFE_FILTERS = [
    # Filtros de video básicos
    'scale', 'crop', 'pad', 'rotate', 'transpose', 'trim', 'setpts', 'fps', 
    'format', 'deinterlace', 'interlace', 'fieldorder', 'yadif', 'hflip', 'vflip',
    'drawtext', 'drawbox', 'overlay', 'split', 'zoompan', 'fade', 'setdar', 'setsar',
    
    # Filtros de audio básicos
    'aresample', 'atempo', 'volume', 'aformat', 'afade', 'adelay', 'aecho',
    'amix', 'amerge', 'asplit', 'atrim', 'asetpts', 'aphaser', 'flanger',
    
    # Filtros de composición
    'concat', 'hstack', 'vstack', 'xstack', 'overlay'
]

# Opciones para limitar recursos
DEFAULT_MAX_THREADS = config.FFMPEG_THREADS if hasattr(config, 'FFMPEG_THREADS') else 4
DEFAULT_TIMEOUT = getattr(config, 'MAX_PROCESSING_TIME', 1800)  # 30 minutos por defecto

def get_optimal_thread_count():
    """
    Determina el número óptimo de hilos basado en la carga actual del sistema
    
    Returns:
        int: Número óptimo de hilos para FFmpeg
    """
    try:
        cpu_count = os.cpu_count() or 4
        
        # Verificar carga del sistema
        if hasattr(os, 'getloadavg'):
            load = os.getloadavg()[0]
            # Si la carga está por encima del 80% de la capacidad, reducir hilos
            if load > cpu_count * 0.8:
                return max(2, min(cpu_count // 2, DEFAULT_MAX_THREADS))
        
        # Verificar uso de memoria
        mem = psutil.virtual_memory()
        if mem.percent > 85:  # Uso de memoria por encima del 85%
            return max(2, min(cpu_count // 2, DEFAULT_MAX_THREADS))
            
        # Uso normal
        return min(cpu_count, DEFAULT_MAX_THREADS)
    
    except Exception as e:
        logger.warning(f"Error determinando hilos óptimos: {str(e)}. Usando valor por defecto.")
        return DEFAULT_MAX_THREADS

def validate_ffmpeg_parameters(params: Union[str, List[str]], more_params: Optional[Union[str, List[str]]] = None) -> None:
    """
    Valida los parámetros de FFmpeg para prevenir inyección de comandos de manera robusta
    
    Args:
        params: Parámetro individual o lista de parámetros
        more_params: Parámetros adicionales (opcional)
    
    Raises:
        ValueError: Si se detecta un parámetro potencialmente peligroso
    """
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
    for i, param in enumerate(all_params):
        if not param:
            continue
            
        param_str = str(param).lower()
        
        # 1. Verificar si es una opción que comienza con '-'
        if param_str.startswith('-'):
            # Verificar si está en la lista blanca
            if not any(param_str.startswith(safe_opt.lower()) for safe_opt in SAFE_FFMPEG_OPTIONS):
                logger.warning(f"Parámetro FFmpeg potencialmente peligroso: {param}")
                raise ValueError(f"Parámetro FFmpeg no permitido: {param}")
                
        # 2. Verificar si contiene patrones peligrosos
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, param_str):
                logger.warning(f"Patrón peligroso detectado en parámetro FFmpeg: {param}")
                raise ValueError(f"Parámetro FFmpeg potencialmente inseguro: {param}")
    
    # Verificación especial para filter_complex o -vf
    for i, param in enumerate(all_params):
        if not param:
            continue
            
        param_str = str(param).lower()
        
        # Buscar parámetros de filtro
        if param_str in ['-filter_complex', '-vf', '-af', '-filter:v', '-filter:a']:
            # Asegurarse de que hay un parámetro siguiente
            if i + 1 < len(all_params):
                filter_str = str(all_params[i + 1])
                
                # Validar la cadena de filtros
                validate_filter_string(filter_str)
                
def validate_filter_string(filter_str: str) -> None:
    """
    Valida una cadena de filtro FFmpeg para asegurar que solo contiene filtros seguros
    
    Args:
        filter_str: Cadena de filtro a validar
        
    Raises:
        ValueError: Si se detecta un filtro potencialmente peligroso
    """
    # Detección de patrones peligrosos en filtros
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, filter_str):
            logger.warning(f"Patrón peligroso detectado en filtro: {filter_str}")
            raise ValueError(f"Filtro FFmpeg potencialmente inseguro. Detectado patrón no permitido.")
    
    # Separar filtros individuales
    # Esto es complejo porque los filtros pueden contener comas dentro de paréntesis
    # Por ejemplo: "scale=640:480,overlay=10:10:enable='between(t,1,5)'"
    
    # Análisis básico: dividir por comas fuera de paréntesis
    filter_parts = []
    current_part = ""
    paren_level = 0
    bracket_level = 0
    single_quote = False
    double_quote = False
    
    for char in filter_str:
        if char == '(' and not single_quote and not double_quote:
            paren_level += 1
            current_part += char
        elif char == ')' and not single_quote and not double_quote:
            paren_level -= 1
            current_part += char
        elif char == '[' and not single_quote and not double_quote:
            bracket_level += 1
            current_part += char
        elif char == ']' and not single_quote and not double_quote:
            bracket_level -= 1
            current_part += char
        elif char == "'" and not double_quote:
            single_quote = not single_quote
            current_part += char
        elif char == '"' and not single_quote:
            double_quote = not double_quote
            current_part += char
        elif char == ',' and paren_level == 0 and bracket_level == 0 and not single_quote and not double_quote:
            filter_parts.append(current_part)
            current_part = ""
        else:
            current_part += char
    
    if current_part:
        filter_parts.append(current_part)
    
    # Verificar cada filtro contra lista blanca
    for part in filter_parts:
        # Extraer nombre del filtro
        filter_name = part.split('=')[0].strip()
        
        # Limpiar referencias a pads como [in], [out], [0:v], etc.
        if filter_name.startswith('[') and ']' in filter_name:
            filter_name = filter_name.split(']', 1)[1].strip()
        
        if not any(filter_name == safe_filter for safe_filter in SAFE_FILTERS):
            logger.warning(f"Filtro no permitido detectado: {filter_name}")
            raise ValueError(f"Filtro FFmpeg '{filter_name}' no está en la lista de filtros permitidos")

def verify_media_file(file_path: str) -> bool:
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
            '-select_streams', 'v:0,a:0', 
            '-show_entries', 'stream=codec_type,codec_name,width,height,duration,bit_rate', 
            '-of', 'json', 
            file_path
        ]
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=False,
            timeout=30  # Timeout de 30 segundos
        )
        
        # Si el comando falló
        if result.returncode != 0:
            raise ValueError(f"No es un archivo multimedia válido: {os.path.basename(file_path)} - {result.stderr}")
            
        # Analizar JSON
        try:
            info = json.loads(result.stdout)
            
            # Verificar si tiene streams
            if 'streams' not in info or len(info['streams']) == 0:
                raise ValueError(f"No se encontraron streams multimedia en el archivo: {os.path.basename(file_path)}")
                
            # Verificar tipos de streams
            stream_types = [s.get('codec_type') for s in info['streams'] if 'codec_type' in s]
            if not stream_types:
                raise ValueError(f"No se pudo determinar tipos de codecs en el archivo: {os.path.basename(file_path)}")
            
            logger.debug(f"Archivo multimedia verificado: {os.path.basename(file_path)} - Streams: {stream_types}")
            return True
            
        except json.JSONDecodeError:
            raise ValueError(f"El archivo no es un archivo multimedia válido: {os.path.basename(file_path)}")
            
    except subprocess.TimeoutExpired:
        raise ValueError(f"Timeout verificando archivo multimedia: {os.path.basename(file_path)}")
    except Exception as e:
        logger.error(f"Error verificando archivo multimedia {file_path}: {str(e)}")
        raise ValueError(f"Error validando archivo multimedia: {str(e)}")

def generate_ffmpeg_command(
    input_paths: List[str], 
    input_options: List[List[str]], 
    filter_complex: str, 
    output_options: List[str], 
    output_path: str
) -> List[str]:
    """
    Genera un comando FFmpeg seguro con las opciones especificadas
    
    Args:
        input_paths: Lista de rutas de archivos de entrada
        input_options: Lista de opciones para cada entrada
        filter_complex: Filtro complejo
        output_options: Opciones para la salida
        output_path: Ruta del archivo de salida
        
    Returns:
        List[str]: Comando FFmpeg como lista de argumentos
    """
    cmd = ['ffmpeg', '-y']  # Siempre comenzar con ffmpeg y -y para sobreescribir
    
    # Añadir entradas con sus opciones
    for i, (input_path, options) in enumerate(zip(input_paths, input_options)):
        if options:
            cmd.extend(options)
        cmd.extend(['-i', input_path])
    
    # Añadir filtro complejo si se proporciona
    if filter_complex:
        cmd.extend(['-filter_complex', filter_complex])
    
    # Añadir opciones de salida
    if output_options:
        cmd.extend(output_options)
    
    # Añadir configuración de threads
    threads = get_optimal_thread_count()
    cmd.extend(['-threads', str(threads)])
    
    # Añadir salida
    cmd.append(output_path)
    
    return cmd

def log_execution_metrics(start_time: float, job_id: str, input_count: int, output_path: str) -> None:
    """
    Registra métricas de rendimiento de la ejecución
    
    Args:
        start_time: Tiempo de inicio
        job_id: ID del trabajo
        input_count: Número de archivos de entrada
        output_path: Ruta del archivo de salida
    """
    try:
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Obtener tamaño del archivo de salida
        output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        
        logger.info(
            f"Job {job_id}: FFmpeg completado en {execution_time:.2f}s - "
            f"{input_count} entradas -> {output_size_mb:.2f}MB salida"
        )
        
        # Registrar uso de memoria actual (como referencia)
        mem = psutil.virtual_memory()
        logger.debug(f"Uso de memoria después del procesamiento: {mem.percent}%")
    
    except Exception as e:
        logger.error(f"Error registrando métricas: {str(e)}")

def monitor_ffmpeg_process(process, timeout):
    """
    Monitorea un proceso FFmpeg en ejecución
    
    Args:
        process: Objeto de proceso
        timeout: Timeout en segundos
        
    Returns:
        tuple: (stdout, stderr)
        
    Raises:
        TimeoutError: Si el proceso excede el timeout
    """
    start_time = time.time()
    stdout_chunks = []
    stderr_chunks = []
    
    # Monitorear salida mientras se ejecuta
    while process.poll() is None:
        # Verificar timeout
        if time.time() - start_time > timeout:
            try:
                # Intentar terminar el proceso
                process.kill()
            except:
                pass
            raise TimeoutError(f"FFmpeg process timed out after {timeout} seconds")
        
        # Leer salida sin bloquear
        if process.stdout:
            chunk = process.stdout.read1(4096)
            if chunk:
                stdout_chunks.append(chunk)
        
        if process.stderr:
            chunk = process.stderr.read1(4096)
            if chunk:
                stderr_chunks.append(chunk)
        
        # Breve pausa para evitar 100% CPU
        time.sleep(0.1)
    
    # Leer restos de salida
    if process.stdout:
        chunk = process.stdout.read()
        if chunk:
            stdout_chunks.append(chunk)
    
    if process.stderr:
        chunk = process.stderr.read()
        if chunk:
            stderr_chunks.append(chunk)
    
    # Compilar salida completa
    stdout = b''.join(stdout_chunks).decode('utf-8', errors='replace')
    stderr = b''.join(stderr_chunks).decode('utf-8', errors='replace')
    
    return stdout, stderr

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
@queue_task_wrapper(bypass_queue=False, priority=TaskPriority.NORMAL)
def ffmpeg_compose(job_id, data):
    """
    Endpoint para componer videos usando comandos FFmpeg personalizados
    
    Args:
        job_id: ID del trabajo
        data: Datos de la solicitud
        
    Returns:
        tuple: (URL del resultado, endpoint, código de estado)
    """
    inputs = data['inputs']
    filter_complex = data['filter_complex']
    output_options = data.get('output_options', [])
    webhook_url = data.get('webhook_url')
    
    logger.info(f"Job {job_id}: Recibida solicitud FFmpeg con {len(inputs)} entradas")

    # Validar comando para evitar inyección
    try:
        validate_filter_string(filter_complex)
        
        if output_options:
            validate_ffmpeg_parameters(output_options)
        
        # Validar opciones de entrada
        for input_data in inputs:
            if "options" in input_data and input_data["options"]:
                validate_ffmpeg_parameters(input_data["options"])
    
    except ValueError as e:
        logger.error(f"Job {job_id}: Error de validación: {str(e)}")
        return str(e), "/v1/ffmpeg/ffmpeg_compose", 400

    # Crear directorio de trabajo temporal
    work_dir = tempfile.mkdtemp(prefix=f"ffmpeg_{job_id}_")
    
    try:
        # Descargar todos los archivos de entrada
        input_paths = []
        input_options_list = []
        
        for i, input_data in enumerate(inputs):
            url = input_data["url"]
            try:
                file_path = download_file(url, work_dir)
                
                # Verificar que el archivo es válido
                verify_media_file(file_path)
                
                # Almacenar la ruta y opciones
                input_paths.append(file_path)
                input_options_list.append(input_data.get("options", []))
                
                logger.info(f"Job {job_id}: Descargada entrada {i+1}/{len(inputs)}: {os.path.basename(file_path)}")
            
            except Exception as e:
                logger.error(f"Job {job_id}: Error descargando entrada {i+1}: {str(e)}")
                # Limpiar archivos descargados
                cleanup_temp_files(work_dir, input_paths)
                raise ValueError(f"Error descargando entrada {i+1}: {str(e)}")

        # Generar nombre de archivo de salida
        output_path = os.path.join(work_dir, f"output_{job_id}.mp4")

        # Generar comando FFmpeg
        cmd = generate_ffmpeg_command(
            input_paths, 
            input_options_list, 
            filter_complex, 
            output_options, 
            output_path
        )

        logger.info(f"Job {job_id}: Ejecutando comando FFmpeg")
        logger.debug(f"Comando: {' '.join(cmd)}")

        # Ejecutar FFmpeg con timeout y monitoreo
        start_time = time.time()
        max_execution_time = DEFAULT_TIMEOUT
        
        try:
            # Abrir proceso con pipes para todas las entradas/salidas
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                bufsize=10*1024*1024,  # 10MB buffer
                text=False  # Binario para evitar problemas de codificación
            )
            
            # Monitorear proceso
            stdout, stderr = monitor_ffmpeg_process(process, max_execution_time)
            
            # Verificar código de salida
            if process.returncode != 0:
                logger.error(f"Comando FFmpeg falló. Error: {stderr}")
                raise subprocess.CalledProcessError(process.returncode, cmd, stdout, stderr)
            
        except TimeoutError:
            logger.error(f"Job {job_id}: Timeout de FFmpeg después de {max_execution_time} segundos")
            cleanup_temp_files(work_dir, input_paths)
            raise TimeoutError(f"El procesamiento FFmpeg superó el tiempo límite de {max_execution_time} segundos")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Job {job_id}: Error de FFmpeg: {e.stderr}")
            cleanup_temp_files(work_dir, input_paths)
            raise ValueError(f"Error de FFmpeg: {e.stderr}")

        # Registrar métricas de ejecución
        log_execution_metrics(start_time, job_id, len(input_paths), output_path)

        # Verificar que el archivo de salida existe y tiene tamaño
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise ValueError("FFmpeg completó correctamente pero el archivo de salida está vacío o no existe")

        # Verificar integridad del archivo resultante
        if not verify_file_integrity(output_path):
            raise ValueError("El archivo generado no pasó la verificación de integridad")

        # Almacenar archivo resultante
        file_url = store_file(output_path)
        logger.info(f"Job {job_id}: Video almacenado en: {file_url}")

        # Limpiar archivos temporales
        cleanup_temp_files(work_dir, input_paths, output_path)

        return file_url, "/v1/ffmpeg/ffmpeg_compose", 200

    except Exception as e:
        logger.error(f"Job {job_id}: Error durante composición FFmpeg - {str(e)}", exc_info=True)
        
        # Limpiar archivos temporales incluso en caso de error
        cleanup_temp_files(work_dir, input_paths if 'input_paths' in locals() else [])
        
        # Determinar el código de estado HTTP adecuado según el error
        if isinstance(e, ValueError):
            return str(e), "/v1/ffmpeg/ffmpeg_compose", 400  # Bad Request para errores de validación
        elif isinstance(e, TimeoutError):
            return str(e), "/v1/ffmpeg/ffmpeg_compose", 408  # Request Timeout
        else:
            return str(e), "/v1/ffmpeg/ffmpeg_compose", 500

def cleanup_temp_files(work_dir, input_paths=None, output_path=None):
    """
    Limpia archivos temporales después del procesamiento
    
    Args:
        work_dir: Directorio de trabajo
        input_paths: Lista de rutas de archivos de entrada
        output_path: Ruta del archivo de salida
    """
    try:
        # Eliminar archivos de entrada si existen
        if input_paths:
            for path in input_paths:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
        
        # Eliminar archivo de salida si existe
        if output_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        
        # Eliminar directorio de trabajo
        if os.path.exists(work_dir):
            try:
                # Eliminar cualquier archivo restante en el directorio
                for filename in os.listdir(work_dir):
                    file_path = os.path.join(work_dir, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except:
                        pass
                
                # Eliminar el directorio
                os.rmdir(work_dir)
            except:
                pass
    
    except Exception as e:
        logger.error(f"Error limpiando archivos temporales: {str(e)}")

@v1_ffmpeg_compose_bp.route('/v1/ffmpeg/status/<job_id>', methods=['GET'])
@authenticate
def get_job_status_endpoint(job_id):
    """
    Endpoint para consultar el estado de un trabajo
    
    Args:
        job_id: ID del trabajo
        
    Returns:
        Response: Respuesta JSON con el estado del trabajo
    """
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

@v1_ffmpeg_compose_bp.route('/v1/ffmpeg/compose/examples', methods=['GET'])
def get_compose_examples():
    """
    Endpoint para obtener ejemplos de uso de ffmpeg_compose
    
    Returns:
        Response: Respuesta JSON con ejemplos de uso
    """
    examples = [
        {
            "description": "Superponer una imagen sobre un video",
            "request": {
                "inputs": [
                    {"url": "https://ejemplo.com/video.mp4"},
                    {"url": "https://ejemplo.com/logo.png", "options": ["-loop", "1"]}
                ],
                "filter_complex": "[0:v][1:v]overlay=10:10:enable='between(t,1,5)'[out]",
                "output_options": ["-map", "[out]", "-map", "0:a", "-c:a", "copy"]
            }
        },
        {
            "description": "Concatenar dos videos",
            "request": {
                "inputs": [
                    {"url": "https://ejemplo.com/video1.mp4"},
                    {"url": "https://ejemplo.com/video2.mp4"}
                ],
                "filter_complex": "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[outv][outa]",
                "output_options": ["-map", "[outv]", "-map", "[outa]"]
            }
        },
        {
            "description": "Aplicar texto con efecto de disolución",
            "request": {
                "inputs": [
                    {"url": "https://ejemplo.com/video.mp4"}
                ],
                "filter_complex": "[0:v]drawtext=text='Texto de ejemplo':fontsize=24:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2:enable='between(t,1,5)':alpha='if(lt(t,2),t-1,if(lt(t,4),1,5-t))'[out]",
                "output_options": ["-map", "[out]", "-map", "0:a", "-c:a", "copy"]
            }
        }
    ]
    
    return jsonify({
        "status": "success",
        "examples": examples,
        "note": "Estos ejemplos ilustran usos comunes del endpoint ffmpeg_compose. Adapta los parámetros a tus necesidades."
    }), 200
