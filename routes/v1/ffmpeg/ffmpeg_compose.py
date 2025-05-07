from flask import Blueprint
from app_utils import validate_payload, queue_task_wrapper
import logging
import os
import subprocess
import json
from services.authentication import authenticate
from services.file_management import download_file, generate_temp_filename
from services.local_storage import store_file
import config

v1_ffmpeg_compose_bp = Blueprint('v1_ffmpeg_compose', __name__)
logger = logging.getLogger(__name__)

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
    validate_ffmpeg_parameters(filter_complex, output_options)
    for input_data in inputs:
        if "options" in input_data:
            validate_ffmpeg_parameters(input_data["options"])

    try:
        # Descargar todos los archivos de entrada
        input_paths = []
        for i, input_data in enumerate(inputs):
            url = input_data["url"]
            try:
                file_path = download_file(url, config.TEMP_DIR)
                input_paths.append(file_path)
                logger.info(f"Job {job_id}: Downloaded input {i+1}/{len(inputs)}")
            except Exception as e:
                logger.error(f"Job {job_id}: Error downloading input {i+1}: {str(e)}")
                # Limpiar archivos ya descargados
                for path in input_paths:
                    if os.path.exists(path):
                        os.remove(path)
                raise

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

        # Añadir salida y forzar sobreescritura
        cmd.extend(['-y', output_path])

        logger.info(f"Job {job_id}: Running FFmpeg command")
        logger.debug(f"Command: {' '.join(cmd)}")

        # Ejecutar FFmpeg
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg command failed. Error: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

        logger.info(f"Job {job_id}: FFmpeg compose completed successfully")

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
            
        return str(e), "/v1/ffmpeg/ffmpeg_compose", 500

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
        '-f', 'lavfi',  # Podría usarse para generar contenido ilimitado
        'system', 'exec',  # Ejecución de comandos del sistema
        '`', '$(', '${',  # Expansión de shell
        '&&', '||', ';',  # Operadores de concatenación
        '>', '>>', '<',  # Redirección
        'http://', 'https://', 'ftp://'  # URLs no controladas
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
        for dangerous in dangerous_options:
            if dangerous in param_str:
                logger.warning(f"Dangerous FFmpeg parameter detected: {param}")
                raise ValueError(f"Potentially unsafe FFmpeg parameter: {param}")
