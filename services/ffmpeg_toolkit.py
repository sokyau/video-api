# --- START OF FILE ffmpeg_toolkit.py ---

import os
import subprocess
import logging
import json
import shutil
import time
import re
import platform
import psutil
from typing import Dict, List, Union, Optional, Tuple, Any

# Corrected import based on your app.py structure
# Assuming 'errors.py' is at the root of your project or accessible in PYTHONPATH
from errors import (
    ValidationError,
    NotFoundError,
    ProcessingError,
    StorageError,
    FFmpegError,  # Using your specific FFmpegError
    capture_exception
)

# Assuming 'file_management.py' is in 'services' package
from services.file_management import generate_temp_filename, get_file_extension, verify_file_integrity
import config

logger = logging.getLogger(__name__)

# Constantes de configuración
DEFAULT_TIMEOUT = getattr(config, 'FFMPEG_TIMEOUT', 1800)  # 30 minutos por defecto
MAX_THREADS = getattr(config, 'FFMPEG_THREADS', 4)
# MAX_RETRIES = getattr(config, 'FFMPEG_MAX_RETRIES', 2) # Not currently used

def get_optimal_thread_count() -> int:
    """
    Determina el número óptimo de hilos basado en la carga del sistema
    
    Returns:
        int: Número óptimo de hilos para FFmpeg
    """
    try:
        cpu_count = os.cpu_count() or 4
        if platform.system() != "Windows":
            # Check system load (Unix-like systems)
            load1, _, _ = os.getloadavg()
            if load1 > cpu_count * 0.8: # If 1-minute load average is high
                return max(1, cpu_count // 2) # Ensure at least 1 thread
        
        mem = psutil.virtual_memory()
        if mem.percent > 85: # If memory usage is high
            return max(1, cpu_count // 2) # Ensure at least 1 thread
            
        # Use configured MAX_THREADS if positive, otherwise cpu_count
        return min(cpu_count, MAX_THREADS) if MAX_THREADS > 0 else cpu_count
    except Exception as e:
        # Capture non-critical exception
        capture_exception(e, {"context": "get_optimal_thread_count_failed"})
        logger.warning(f"Error determinando hilos óptimos: {str(e)}. Usando valor por defecto: {MAX_THREADS if MAX_THREADS > 0 else 'cpu_count'}.")
        return MAX_THREADS if MAX_THREADS > 0 else (os.cpu_count() or 4)


def validate_ffmpeg_parameters(params: Union[str, List[str]]) -> None:
    """
    Valida los parámetros de FFmpeg para prevenir inyección de comandos
    
    Args:
        params: Parámetro individual o lista de parámetros
    
    Raises:
        ValidationError: Si se detecta un parámetro potencialmente peligroso
    """
    dangerous_patterns = [
        r'`.*`', r'\$\(.*\)', r'\$\{.*\}', r'[;&|]',
        # r'^-f\s+lavfi', # Re-evaluating: lavfi itself isn't always dangerous, depends on the filter.
                         # Consider more specific dangerous lavfi filters if needed.
        r'system\s*\(', r'exec\s*\(', r'subprocess',
        r'-filter_script', # Disallow external filter scripts for now
        # Consider if direct output to something other than a file path needs restriction
        # r'>[^,\s]*', # Potentially too broad, many valid uses of > in filters
        # r'<[^,\s]*',
        r'-vhook', # Deprecated and potentially risky
        r'-codec:v\s+copy.*\s+-bsf:v\s+mpeg4_unpack_bframes', # Known vulnerability pattern if source is untrusted
        r'file:', # Be careful with 'file:' protocol in inputs if paths are not strictly controlled
    ]
    
    params_list = [params] if isinstance(params, str) else params
    
    for param in params_list:
        param_str = str(param).lower() # Convert to string for safety
        for pattern in dangerous_patterns:
            if re.search(pattern, param_str):
                error_msg = f"Parámetro FFmpeg potencialmente inseguro detectado: {param_str}"
                logger.warning(error_msg + f" (Pattern: {pattern})")
                raise ValidationError(message=error_msg,
                                      error_code="unsafe_ffmpeg_parameter",
                                      details={"parameter": param_str, "matched_pattern": pattern})

def get_video_info(file_path: str) -> Dict[str, Any]:
    """
    Obtiene información detallada sobre un archivo de video usando ffprobe
    
    Args:
        file_path (str): Ruta al archivo de video
    
    Returns:
        dict: Información del video (duración, resolución, codec, etc.)
    Raises:
        NotFoundError: If the file_path does not exist.
        FFmpegError: If ffprobe fails (delegated from FFmpegError.from_ffmpeg_error).
        ProcessingError: If ffprobe output cannot be parsed or other unexpected error.
        ValidationError: If ffprobe output indicates a malformed input file.
    """
    if not os.path.exists(file_path):
        raise NotFoundError(message=f"Archivo de video no encontrado para ffprobe: {file_path}",
                            error_code="ffprobe_input_not_found",
                            details={"file_path": file_path})
    
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', file_path]
    cmd_str = ' '.join(cmd)
    logger.debug(f"Ejecutando comando ffprobe: {cmd_str}")
    
    try:
        # Increased timeout for ffprobe, as large files might take time to analyze
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=getattr(config, 'FFPROBE_TIMEOUT', 120))
        info = json.loads(result.stdout)
        
        video_info: Dict[str, Any] = {
            'format': info.get('format', {}).get('format_name', 'unknown'),
            'duration': float(info.get('format', {}).get('duration', 0.0)),
            'size': int(info.get('format', {}).get('size', 0)),
            'bit_rate': int(info.get('format', {}).get('bit_rate', 0)),
            'streams': []
        }
        
        for stream_data in info.get('streams', []):
            stream_type = stream_data.get('codec_type')
            parsed_stream: Dict[str, Any] = {'type': stream_type, 'index': stream_data.get('index')}
            
            if stream_type == 'video':
                parsed_stream.update({
                    'codec': stream_data.get('codec_name', 'unknown'),
                    'width': int(stream_data.get('width', 0)),
                    'height': int(stream_data.get('height', 0)),
                    'fps': eval(stream_data.get('r_frame_rate', '0/1.0')) if stream_data.get('r_frame_rate') else 0.0,
                    'bit_rate': int(stream_data.get('bit_rate', 0)) if stream_data.get('bit_rate') else 0,
                    'duration': float(stream_data.get('duration', 0.0)) if stream_data.get('duration') else video_info['duration'],
                    'pix_fmt': stream_data.get('pix_fmt', 'unknown')
                })
            elif stream_type == 'audio':
                parsed_stream.update({
                    'codec': stream_data.get('codec_name', 'unknown'),
                    'channels': int(stream_data.get('channels', 0)),
                    'sample_rate': int(stream_data.get('sample_rate', 0)),
                    'bit_rate': int(stream_data.get('bit_rate', 0)) if stream_data.get('bit_rate') else 0,
                    'duration': float(stream_data.get('duration', 0.0)) if stream_data.get('duration') else video_info['duration']
                })
            video_info['streams'].append(parsed_stream)
        
        video_streams = [s for s in video_info['streams'] if s['type'] == 'video']
        audio_streams = [s for s in video_info['streams'] if s['type'] == 'audio']
        
        if video_streams:
            vs = video_streams[0]
            video_info.update({'width': vs.get('width',0), 'height': vs.get('height',0), 'fps': vs.get('fps',0.0), 'video_codec': vs.get('codec')})
            if vs.get('width',0) > 0 and vs.get('height',0) > 0:
                video_info['aspect_ratio'] = vs['width'] / vs['height']
        
        if audio_streams:
            audio_s = audio_streams[0]
            video_info.update({'audio_codec': audio_s.get('codec'), 'audio_channels': audio_s.get('channels')})
        
        if video_info['duration'] > 0:
            s = video_info['duration']
            video_info['duration_formatted'] = f"{int(s // 3600):02d}:{int((s % 3600) // 60):02d}:{int(s % 60):02d}"
        
        logger.debug(f"Información del video extraída para {file_path}")
        return video_info
        
    except subprocess.CalledProcessError as e:
        error_id = capture_exception(e, {"command": cmd_str, "file_path": file_path, "stderr": e.stderr})
        # Use FFmpegError.from_ffmpeg_error to create the error
        ffmpeg_err = FFmpegError.from_ffmpeg_error(stderr=e.stderr, cmd=cmd)
        ffmpeg_err.details["error_id"] = error_id # Attach our generated error_id
        ffmpeg_err.details["file_path"] = file_path
        # Check for specific ffprobe issues that indicate input validation problems
        if "invalid data found" in e.stderr.lower() or "moov atom not found" in e.stderr.lower():
            raise ValidationError(message=f"FFprobe: Input file '{file_path}' may be malformed or unsupported. {ffmpeg_err.message}",
                                  error_code="ffprobe_malformed_input",
                                  details=ffmpeg_err.details)
        raise ffmpeg_err
    except json.JSONDecodeError as e:
        error_id = capture_exception(e, {"command": cmd_str, "file_path": file_path, "stdout_sample": result.stdout[:200] if 'result' in locals() else 'N/A'})
        raise ProcessingError(message=f"Error parseando salida JSON de ffprobe para '{file_path}': {str(e)}",
                              error_code="ffprobe_json_decode_error",
                              details={"file_path": file_path, "error_id": error_id})
    except subprocess.TimeoutExpired as e:
        error_id = capture_exception(e, {"command": cmd_str, "file_path": file_path})
        raise ProcessingError(message=f"Timeout ejecutando ffprobe para '{file_path}'",
                              error_code="ffprobe_timeout",
                              details={"file_path": file_path, "error_id": error_id})
    except Exception as e:
        error_id = capture_exception(e, {"command": cmd_str, "file_path": file_path})
        raise ProcessingError(message=f"Error inesperado obteniendo información del video '{file_path}': {str(e)}",
                              error_code="ffprobe_unexpected_error",
                              details={"file_path": file_path, "error_id": error_id})


def _execute_ffmpeg_process(cmd: List[str], output_path: str, operation_name: str, timeout: int) -> None:
    """Internal helper to run ffmpeg, handle errors, and check output."""
    cmd_str = ' '.join(cmd)
    logger.debug(f"Ejecutando comando FFmpeg ({operation_name}): {cmd_str}")
    start_time_exec = time.time()
    process = None
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0)
        stdout, stderr = process.communicate(timeout=timeout)
        
        processing_time = time.time() - start_time_exec
        
        if process.returncode != 0:
            error_id = capture_exception(Exception(f"FFmpeg {operation_name} failed"), # Base exception for Sentry
                                         {"command": cmd_str, "stderr": stderr, "stdout": stdout, "return_code": process.returncode})
            # Use the specific FFmpegError from errors.py
            ffmpeg_error = FFmpegError.from_ffmpeg_error(stderr=stderr, cmd=cmd)
            ffmpeg_error.details["error_id"] = error_id
            ffmpeg_error.details["operation"] = operation_name
            logger.error(f"Error en {operation_name} FFmpeg ({processing_time:.2f}s). Code: {ffmpeg_error.error_code}, Message: {ffmpeg_error.message}, Details: {ffmpeg_error.details}")
            raise ffmpeg_error
        
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            # This is a critical processing failure if output is not as expected
            error_id = capture_exception(Exception("FFmpeg output file missing or empty"),
                                         {"command": cmd_str, "output_path": output_path, "stderr": stderr, "operation": operation_name})
            raise ProcessingError(message=f"{operation_name} falló: archivo de salida '{output_path}' vacío o inexistente. Stderr: {stderr}",
                                  error_code="ffmpeg_output_file_missing",
                                  details={"operation": operation_name, "output_path": output_path, "stderr": stderr, "error_id": error_id})

        logger.info(f"{operation_name} FFmpeg exitosa: -> {output_path} en {processing_time:.2f} segundos")

    except subprocess.TimeoutExpired as e:
        if process: process.kill()
        error_id = capture_exception(e, {"command": cmd_str, "operation": operation_name, "timeout_seconds": timeout})
        # Raise ProcessingError for timeout, as it's an operational failure
        raise ProcessingError(message=f"{operation_name} excedió el tiempo límite de {timeout} segundos.",
                              error_code="ffmpeg_timeout",
                              details={"operation": operation_name, "command": cmd_str, "timeout_seconds": timeout, "error_id": error_id})
    except (FFmpegError, ProcessingError, ValidationError, NotFoundError, StorageError): # Re-raise our custom errors
        raise
    except Exception as e: 
        if process: process.kill()
        error_id = capture_exception(e, {"command": cmd_str, "operation": operation_name})
        raise ProcessingError(message=f"Error inesperado durante {operation_name} FFmpeg: {str(e)}",
                              error_code="ffmpeg_unexpected_runtime_error",
                              details={"operation": operation_name, "command": cmd_str, "error_id": error_id})


def convert_video(input_path: str, output_path: Optional[str] = None, format: Optional[str] = None, 
                 video_codec: Optional[str] = None, audio_codec: Optional[str] = None, 
                 width: Optional[int] = None, height: Optional[int] = None, 
                 bitrate: Optional[str] = None, framerate: Optional[int] = None, 
                 extra_args: Optional[List[str]] = None, timeout: Optional[int] = None) -> str:
    if not os.path.exists(input_path):
        raise NotFoundError(message=f"Archivo de entrada no encontrado para conversión: {input_path}", details={"file_path": input_path})
    
    final_output_path = output_path or generate_temp_filename(suffix=f".{format or 'mp4'}")
    
    cmd = ['ffmpeg', '-y', '-i', input_path]
    threads = get_optimal_thread_count()
    
    if video_codec: cmd.extend(['-c:v', video_codec])
    if audio_codec: cmd.extend(['-c:a', audio_codec])
    
    vf_parts = []
    if width or height:
        input_info = get_video_info(input_path) # Can raise various errors
        aspect_ratio = input_info.get('aspect_ratio')
        target_w, target_h = width, height
        
        if target_w and not target_h: # Width given, calculate height
            target_h = int(target_w / aspect_ratio) if aspect_ratio and aspect_ratio > 0 else -2 # -2 preserves aspect
        elif target_h and not target_w: # Height given, calculate width
            target_w = int(target_h * aspect_ratio) if aspect_ratio and aspect_ratio > 0 else -2
        
        if target_w and target_h: # Ensure width and height are even for many codecs
            target_w = target_w if target_w % 2 == 0 else target_w + 1
            target_h = target_h if target_h % 2 == 0 else target_h + 1
            vf_parts.append(f'scale={target_w}:{target_h}')

    if vf_parts:
        cmd.extend(['-vf', ",".join(vf_parts)])

    if bitrate: cmd.extend(['-b:v', bitrate])
    if framerate: cmd.extend(['-r', str(framerate)])
    cmd.extend(['-threads', str(threads)])
    
    if extra_args:
        validate_ffmpeg_parameters(extra_args) # Can raise ValidationError
        cmd.extend(extra_args)
    
    cmd.append(final_output_path)
    
    effective_timeout = timeout if timeout is not None else DEFAULT_TIMEOUT
    _execute_ffmpeg_process(cmd, final_output_path, "conversión de video", effective_timeout)
    return final_output_path


def extract_audio(video_path: str, output_path: Optional[str] = None, format: str = 'mp3', 
                  bitrate: Optional[str] = None, timeout: Optional[int] = None) -> str:
    if not os.path.exists(video_path):
        raise NotFoundError(message=f"Archivo de video no encontrado para extracción de audio: {video_path}", details={"file_path": video_path})

    final_output_path = output_path or generate_temp_filename(suffix=f".{format}")
    threads = get_optimal_thread_count()
    cmd = ['ffmpeg', '-y', '-i', video_path, '-vn', '-threads', str(threads)]
    
    # Recommended bitrates if not specified
    default_bitrates = {'mp3': '192k', 'aac': '128k', 'ogg': '160k'}
    effective_bitrate = bitrate or default_bitrates.get(format)

    if effective_bitrate: cmd.extend(['-b:a', effective_bitrate])
    
    codec_map = {
        'mp3': ['-codec:a', 'libmp3lame', '-q:a', '2'], # VBR quality
        'aac': ['-codec:a', 'aac'], # Let ffmpeg choose best AAC encoder if not specified by -strict experimental
        'wav': ['-codec:a', 'pcm_s16le'],
        'flac': ['-codec:a', 'flac'],
        'ogg': ['-codec:a', 'libvorbis', '-q:a', '5'] # VBR quality
    }
    if format in codec_map:
        cmd.extend(codec_map[format])
    elif format: # If format is specified but not in map, let ffmpeg infer codec from output extension
        logger.info(f"Formato de audio '{format}' no tiene mapeo de codec explícito, ffmpeg inferirá de la extensión.")
    else:
        raise ValidationError("Formato de audio no especificado para la extracción.", error_code="audio_format_missing")

    cmd.append(final_output_path)
    effective_timeout = timeout if timeout is not None else DEFAULT_TIMEOUT
    _execute_ffmpeg_process(cmd, final_output_path, "extracción de audio", effective_timeout)
    return final_output_path


def concatenate_videos(video_paths: List[str], output_path: Optional[str] = None, 
                       transcode: bool = False, timeout: Optional[int] = None) -> str:
    if not video_paths:
        raise ValidationError(message="No se proporcionaron rutas de video para concatenación", error_code="empty_video_list_concat")
    
    for i, path in enumerate(video_paths):
        if not os.path.exists(path):
            raise NotFoundError(message=f"Archivo de video no encontrado para concatenación (índice {i}): {path}", 
                                details={"file_path": path, "index": i})

    final_output_path = output_path or generate_temp_filename(suffix=get_file_extension(video_paths[0]))
    # Estimate timeout: base + per video, more if transcoding
    base_timeout_concat = getattr(config, "FFMPEG_CONCAT_BASE_TIMEOUT", 60)
    per_video_timeout_concat = getattr(config, "FFMPEG_CONCAT_PER_VIDEO_TIMEOUT", 120)
    transcode_multiplier = 2 if transcode else 1
    effective_timeout = timeout if timeout is not None else \
                        (base_timeout_concat + len(video_paths) * per_video_timeout_concat) * transcode_multiplier


    if not transcode and len(video_paths) > 1:
        try:
            first_info = get_video_info(video_paths[0])
            common_params = {
                'video_codec': first_info.get('video_codec'),
                'audio_codec': first_info.get('audio_codec'),
                'width': first_info.get('width'),
                'height': first_info.get('height'),
                'pix_fmt': next((s.get('pix_fmt') for s in first_info.get('streams',[]) if s.get('type') == 'video'), None),
                'sample_rate': next((s.get('sample_rate') for s in first_info.get('streams',[]) if s.get('type') == 'audio'), None),
                'channels': next((s.get('channels') for s in first_info.get('streams',[]) if s.get('type') == 'audio'), None),
            }
            for path_idx, path_val in enumerate(video_paths[1:], 1):
                info = get_video_info(path_val)
                current_pix_fmt = next((s.get('pix_fmt') for s in info.get('streams',[]) if s.get('type') == 'video'), None)
                current_sample_rate = next((s.get('sample_rate') for s in info.get('streams',[]) if s.get('type') == 'audio'), None)
                current_channels = next((s.get('channels') for s in info.get('streams',[]) if s.get('type') == 'audio'), None)

                if not all([info.get('video_codec') == common_params['video_codec'],
                            info.get('audio_codec') == common_params['audio_codec'],
                            info.get('width') == common_params['width'],
                            info.get('height') == common_params['height'],
                            current_pix_fmt == common_params['pix_fmt'],
                            current_sample_rate == common_params['sample_rate'],
                            current_channels == common_params['channels']
                            ]):
                    logger.info(f"Video {path_idx} ('{os.path.basename(path_val)}') no es compatible para concatenación directa. Forzando transcode.")
                    transcode = True
                    break
        except (NotFoundError, FFmpegError, ProcessingError, ValidationError) as e: # Catch our specific errors
            capture_exception(e, {"context": "concat_compatibility_check_failed", "videos": video_paths})
            logger.warning(f"Error verificando compatibilidad de videos para concatenación ({e.message}), forzando transcode.")
            transcode = True

    if not transcode:
        list_file_path = generate_temp_filename(suffix=".txt")
        try:
            with open(list_file_path, 'w', encoding='utf-8') as f: # Specify encoding
                for path_val in video_paths:
                    # Paths in concat demuxer list file should be relative to the list file's directory
                    # or absolute. Absolute is safer.
                    # Special characters in paths need careful handling.
                    # Simplest for ffmpeg concat is often to ensure paths are "clean".
                    # Using os.path.abspath is good.
                    # The single quotes are for the concat demuxer format, not shell escaping.
                    f.write(f"file '{os.path.abspath(path_val)}'\n")
            
            cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_file_path, '-c', 'copy', final_output_path]
            _execute_ffmpeg_process(cmd, final_output_path, "concatenación directa", effective_timeout)
            return final_output_path
        except (FFmpegError, ProcessingError, TimeoutError) as e: # If direct concat fails
            logger.warning(f"Concatenación directa falló ({e.message if hasattr(e, 'message') else str(e)}), intentando con transcoding.")
            transcode = True # Force transcode on retry
        finally:
            if os.path.exists(list_file_path):
                try: os.remove(list_file_path)
                except OSError as e_rm:
                    capture_exception(e_rm, {"file_path": list_file_path, "context": "concat_list_cleanup_error"})
                    logger.error(f"Error eliminando archivo de lista temporal '{list_file_path}': {str(e_rm)}")

    if transcode:
        inputs_cmd_part = []
        filter_complex_str_parts = []
        map_video_str = []
        map_audio_str = []

        # Determine common resolution (e.g., from first video or a target default)
        # For simplicity, let's try to use the first video's resolution if available
        try:
            ref_info = get_video_info(video_paths[0])
            target_w = ref_info.get('width', 1280)
            target_h = ref_info.get('height', 720)
            target_sar = ref_info.get('aspect_ratio', 16/9.0) # Sample Aspect Ratio for setsar
        except Exception:
            target_w, target_h, target_sar = 1280, 720, 16/9.0
            logger.warning("No se pudo obtener información del primer video para transcode, usando 1280x720.")
        
        # Ensure even dimensions
        target_w = target_w if target_w % 2 == 0 else target_w + 1
        target_h = target_h if target_h % 2 == 0 else target_h + 1


        for i, path_val in enumerate(video_paths):
            inputs_cmd_part.extend(['-i', path_val])
            # Scale, pad to target, set SAR, then prepare for concat
            filter_complex_str_parts.append(f"[{i}:v]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2,setsar={target_sar}[v{i}];")
            # Resample audio to a common format
            filter_complex_str_parts.append(f"[{i}:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[a{i}];")
            map_video_str.append(f"[v{i}]")
            map_audio_str.append(f"[a{i}]")
            
        filter_complex_str_parts.append("".join(map_video_str) + "".join(map_audio_str) + f"concat=n={len(video_paths)}:v=1:a=1[outv][outa]")
        
        cmd = ['ffmpeg', '-y'] + inputs_cmd_part + [
            '-filter_complex', "".join(filter_complex_str_parts),
            '-map', '[outv]', '-map', '[outa]',
            '-c:v', 'libx264', '-crf', '23', '-preset', 'medium', '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-b:a', '128k',
            '-threads', str(get_optimal_thread_count()),
            final_output_path
        ]
        _execute_ffmpeg_process(cmd, final_output_path, "concatenación con transcoding", effective_timeout)
        return final_output_path
    
    # This line should ideally not be reached if logic is correct
    raise ProcessingError(message="Concatenación de videos falló después de todos los intentos.", error_code="concat_failed_all_attempts")

# ... (Rest of the functions: add_subtitles, image_to_video, trim_video, add_watermark, overlay_video, add_text_overlay)
# ... will need similar refactoring using _execute_ffmpeg_process and the new error classes.

# For brevity, I will show the refactoring for execute_ffmpeg_command only,
# as the pattern is established. Apply it to the remaining functions.

def execute_ffmpeg_command(command: List[str], output_path_check: Optional[str] = None, timeout: Optional[int] = None) -> str:
    """
    Ejecuta un comando FFmpeg personalizado con validación de seguridad.
    
    Args:
        command (list): Lista de argumentos. Primer elemento debe ser 'ffmpeg' or will be added.
        output_path_check (str, optional): Path to check for existence and size after execution.
        timeout (int, optional): Timeout en segundos.
    
    Returns:
        str: Salida de stderr de FFmpeg (as it often contains progress/info).
    
    Raises:
        ValidationError: If the command is malformed or contains dangerous parameters.
        FFmpegError: If the FFmpeg command itself fails during execution.
        ProcessingError: For issues like timeout, missing output, or unexpected runtime errors.
    """
    if not command or not isinstance(command, list) or not all(isinstance(arg, str) for arg in command):
        raise ValidationError(message="El comando FFmpeg debe ser una lista de strings.", error_code="ffmpeg_cmd_malformed")

    cmd_to_run = list(command) # Make a copy

    # Ensure 'ffmpeg' is the executable
    if not os.path.basename(cmd_to_run[0]).lower().startswith('ffmpeg'):
         raise ValidationError(message=f"El primer elemento del comando debe ser 'ffmpeg' o una ruta a ffmpeg, no '{cmd_to_run[0]}'",
                               error_code="ffmpeg_cmd_invalid_executable")
    
    # Ensure -y for overwriting output, unless -n (no overwrite) is present
    # This logic is a bit simplistic, advanced ffmpeg use might place -y differently.
    # A more robust way is for the caller to ensure -y is correctly placed if needed.
    if '-y' not in cmd_to_run and '-n' not in cmd_to_run:
        # Insert -y after the ffmpeg executable path
        executable_path = cmd_to_run.pop(0)
        cmd_to_run.insert(0, '-y') # Add -y
        cmd_to_run.insert(0, executable_path) # Add executable back

    # Validate all parameters (excluding the 'ffmpeg' executable itself if it's just 'ffmpeg')
    # If cmd_to_run[0] is a full path, it's fine.
    validate_ffmpeg_parameters(cmd_to_run[1:]) # Can raise ValidationError
    
    cmd_str = ' '.join(cmd_to_run)
    logger.debug(f"Ejecutando comando FFmpeg personalizado: {cmd_str}")
    
    effective_timeout = timeout if timeout is not None else DEFAULT_TIMEOUT
    process = None
    try:
        start_time_proc = time.time()
        process = subprocess.Popen(cmd_to_run, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0)
        stdout, stderr = process.communicate(timeout=effective_timeout)
        processing_time = time.time() - start_time_proc

        if process.returncode != 0:
            error_id = capture_exception(Exception("FFmpeg custom command failed"),
                                         {"command": cmd_str, "stderr": stderr, "stdout": stdout, "return_code": process.returncode})
            ffmpeg_error = FFmpegError.from_ffmpeg_error(stderr=stderr, cmd=cmd_to_run)
            ffmpeg_error.details["error_id"] = error_id
            ffmpeg_error.details["operation"] = "custom_command_execution"
            logger.error(f"Error en comando FFmpeg personalizado ({processing_time:.2f}s). Code: {ffmpeg_error.error_code}, Message: {ffmpeg_error.message}")
            raise ffmpeg_error

        if output_path_check:
            if not os.path.exists(output_path_check) or os.path.getsize(output_path_check) == 0:
                error_id = capture_exception(Exception("FFmpeg custom command output file missing or empty"),
                                             {"command": cmd_str, "output_path_check": output_path_check, "stderr": stderr})
                raise ProcessingError(message=f"Comando FFmpeg personalizado falló: archivo de salida '{output_path_check}' vacío o inexistente. Stderr: {stderr}",
                                      error_code="ffmpeg_custom_cmd_output_missing",
                                      details={"command": cmd_str, "output_path_check": output_path_check, "stderr": stderr, "error_id": error_id})
        
        logger.info(f"Comando FFmpeg personalizado ejecutado exitosamente en {processing_time:.2f} segundos. Stderr: {stderr[:500]}...") # Log a sample of stderr
        return stderr 
    
    except subprocess.TimeoutExpired as e:
        if process: process.kill()
        error_id = capture_exception(e, {"command": cmd_str})
        raise ProcessingError(message=f"Comando FFmpeg personalizado excedió el tiempo límite de {effective_timeout} segundos.",
                              error_code="ffmpeg_custom_cmd_timeout",
                              details={"command": cmd_str, "timeout_seconds": effective_timeout, "error_id": error_id})
    except (FFmpegError, ValidationError, ProcessingError, NotFoundError, StorageError): # Re-raise our specific errors
        raise
    except Exception as e: # Catch-all for other unexpected Popen/communicate errors
        if process: process.kill()
        error_id = capture_exception(e, {"command": cmd_str})
        raise ProcessingError(message=f"Error inesperado ejecutando comando FFmpeg personalizado: {str(e)}",
                              error_code="ffmpeg_custom_cmd_unexpected_error",
                              details={"command": cmd_str, "error_id": error_id})

# --- END OF FILE ffmpeg_toolkit.py ---
