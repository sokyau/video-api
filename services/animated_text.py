# --- START OF FILE animated_text.py ---

import os
import subprocess # Still needed for direct ffprobe if not fully delegated
import logging
import json # For parsing ffprobe output directly if needed
from typing import Tuple # For type hinting

# Custom error handling
from errors import (
    ValidationError,
    NotFoundError,
    ProcessingError,
    FFmpegError, # If ffmpeg_toolkit raises it
    capture_exception
)

# File management and FFmpeg toolkit
from services.file_management import download_file, generate_temp_filename
# Assuming ffmpeg_toolkit.py is at the root or accessible like errors.py
# Adjust import if it's in 'services' or elsewhere.
try:
    import ffmpeg_toolkit
except ImportError:
    # Fallback if ffmpeg_toolkit is not directly importable at this level
    # This might indicate a need to adjust PYTHONPATH or project structure
    # For now, we'll assume it will be available or define a mock/stub if needed for isolated testing
    logging.warning("ffmpeg_toolkit.py not found directly, some FFmpeg operations might rely on direct subprocess calls.")
    ffmpeg_toolkit = None


import config

logger = logging.getLogger(__name__)

def _get_video_dimensions_duration(video_path: str) -> Tuple[int, int, float]:
    """
    Helper to get video dimensions and duration.
    Uses ffmpeg_toolkit if available, otherwise direct ffprobe.
    """
    if ffmpeg_toolkit:
        try:
            video_info = ffmpeg_toolkit.get_video_info(video_path)
            width = video_info.get('width', 0)
            height = video_info.get('height', 0)
            duration = video_info.get('duration', 0.0)
            if not width or not height or duration <= 0:
                raise ProcessingError(message="Información de video inválida obtenida de ffprobe.",
                                      error_code="invalid_video_metadata",
                                      details={"video_path": video_path, "info": video_info})
            return width, height, duration
        except (NotFoundError, FFmpegError, ProcessingError, ValidationError) as e:
            # Re-raise with context if needed, or let it propagate
            # Adding context specific to this operation
            e.details = e.details or {}
            e.details["operation_context"] = "get_video_dimensions_for_animation"
            raise
    else: # Fallback to direct ffprobe call
        logger.warning("Usando ffprobe directo porque ffmpeg_toolkit no está disponible.")
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,duration:format=duration', # Get stream duration too
            '-of', 'json', video_path
        ]
        cmd_str = ' '.join(probe_cmd)
        try:
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True, timeout=60)
            video_info_json = json.loads(result.stdout)

            stream_info = video_info_json.get('streams', [{}])[0]
            format_info = video_info_json.get('format', {})

            width = int(stream_info.get('width', 0))
            height = int(stream_info.get('height', 0))
            # Prefer stream duration if available, fallback to format duration
            duration_str = stream_info.get('duration') or format_info.get('duration')
            
            if not duration_str:
                 raise ProcessingError("No se pudo obtener la duración del video desde ffprobe.",
                                       error_code="ffprobe_missing_duration",
                                       details={"video_path": video_path, "output": result.stdout[:500]})
            
            duration = float(duration_str)

            if not width or not height or duration <= 0:
                raise ProcessingError(message="Información de video inválida obtenida de ffprobe (directo).",
                                      error_code="invalid_video_metadata_direct",
                                      details={"video_path": video_path, "output": result.stdout[:500]})
            return width, height, duration
        except subprocess.CalledProcessError as e:
            error_id = capture_exception(e, {"command": cmd_str, "stderr": e.stderr})
            raise FFmpegError.from_ffmpeg_error(stderr=e.stderr, cmd=probe_cmd)
        except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
            error_id = capture_exception(e, {"command": cmd_str, "stdout": result.stdout if 'result' in locals() else 'N/A'})
            raise ProcessingError(message=f"Error parseando salida de ffprobe (directo): {str(e)}",
                                  error_code="ffprobe_parse_error_direct",
                                  details={"error_id": error_id, "stdout_sample": result.stdout[:200] if 'result' in locals() else 'N/A'})


def process_animated_text(video_url: str, text: str, animation: str, position: str,
                          font: str, font_size: int, color: str, duration: float, job_id: str) -> str:
    """
    Procesa un video añadiendo texto animado.

    Args:
        video_url (str): URL del video base
        text (str): Texto a animar
        animation (str): Tipo de animación (fade, slide, zoom, typewriter, bounce)
        position (str): Posición del texto (top, bottom, center)
        font (str): Ruta al archivo de fuente del texto
        font_size (int): Tamaño de fuente
        color (str): Color del texto
        duration (float): Duración de la animación en segundos
        job_id (str): ID del trabajo

    Returns:
        str: Ruta al video procesado
    Raises:
        NotFoundError: If video_url is invalid or file not found after download.
        ValidationError: For invalid parameters.
        ProcessingError: For general processing issues.
        FFmpegError: For specific FFmpeg command failures.
    """
    video_path_local = None
    output_path_local = None

    try:
        # Validate parameters
        if not all([video_url, text, animation, position, font, job_id]):
            raise ValidationError("Parámetros requeridos faltantes para animación de texto.",
                                  error_code="animated_text_missing_params")
        if not isinstance(font_size, int) or font_size <= 0:
            raise ValidationError("font_size debe ser un entero positivo.", error_code="invalid_font_size")
        if not isinstance(duration, (int, float)) or duration <= 0:
            raise ValidationError("duration debe ser un número positivo.", error_code="invalid_animation_duration")
        
        # Check if font file exists, if it's a path.
        # FFmpeg's drawtext fontfile expects a path. If 'font' is just a name,
        # it depends on fontconfig setup, which might not be reliable in all environments.
        # For robustness, 'font' should ideally be a path to a .ttf or .otf file.
        if not os.path.exists(font):
            # Try to find font in common system paths or a configured font directory if applicable
            # For simplicity, we'll raise an error if not directly found.
            # In a real app, you might have a font resolver or default fallback font.
            raise NotFoundError(message=f"Archivo de fuente no encontrado: {font}. Por favor, provea una ruta válida.",
                                error_code="font_file_not_found",
                                details={"font_path_provided": font})


        # Descargar video
        # download_file can raise NetworkError, ValidationError, StorageError
        logger.info(f"Job {job_id}: Descargando video desde {video_url}")
        video_path_local = download_file(video_url, config.TEMP_DIR)
        logger.info(f"Job {job_id}: Video descargado a {video_path_local}")

        # Preparar ruta de salida
        output_path_local = generate_temp_filename(prefix=f"{job_id}_anim_", suffix=".mp4")

        # Obtener información del video (dimensiones y duración)
        video_width, video_height, video_duration_actual = _get_video_dimensions_duration(video_path_local)
        logger.info(f"Job {job_id}: Dimensiones de video: {video_width}x{video_height}, duración: {video_duration_actual}s")

        # Limitar duración de animación a la duración del video
        animation_duration = min(duration, video_duration_actual)
        if duration > video_duration_actual:
            logger.warning(f"Job {job_id}: Duración de animación ({duration}s) truncada a duración de video ({video_duration_actual}s).")


        # Generar filtros según el tipo de animación
        filter_complex_str = generate_text_animation_filter(
            text, animation, position, font, font_size, color,
            animation_duration, video_width, video_height
        )
        
        # Comando FFmpeg
        # Use ffmpeg_toolkit's execute_ffmpeg_command if available for consistent error handling and security
        cmd = [
            'ffmpeg', '-y', # -y added by execute_ffmpeg_command if not present
            '-i', video_path_local,
            '-filter_complex', filter_complex_str,
            '-c:a', 'copy', # Copy audio stream without re-encoding
            # '-b:v', '5M', # Example: Set video bitrate if needed
            # '-preset', 'medium', # Example: Encoding preset
            output_path_local
        ]
        
        logger.info(f"Job {job_id}: Ejecutando comando FFmpeg para texto animado.")
        
        if ffmpeg_toolkit:
            # output_path_check is used by execute_ffmpeg_command to verify output
            ffmpeg_toolkit.execute_ffmpeg_command(cmd, output_path_check=output_path_local, timeout=config.FFMPEG_TIMEOUT)
        else: # Fallback to direct subprocess call
            logger.warning("Ejecutando FFmpeg directamente (ffmpeg_toolkit no disponible).")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=config.FFMPEG_TIMEOUT, check=False) # check=False to handle error manually
            if result.returncode != 0:
                # This will be an FFmpegError from errors.py
                raise FFmpegError.from_ffmpeg_error(stderr=result.stderr, cmd=cmd)
            if not os.path.exists(output_path_local) or os.path.getsize(output_path_local) == 0:
                raise ProcessingError("FFmpeg (directo) no generó archivo de salida o está vacío.",
                                      error_code="ffmpeg_direct_output_missing",
                                      details={"command": ' '.join(cmd), "stderr": result.stderr[:500]})


        logger.info(f"Job {job_id}: Video con texto animado creado exitosamente: {output_path_local}")
        
        return output_path_local # Return before cleanup, caller should handle it or another mechanism

    except (NotFoundError, ValidationError, ProcessingError, FFmpegError) as e: # Catch our specific errors
        # error_id should already be part of e if generated by capture_exception or set in FFmpegError
        error_id = e.details.get("error_id") if hasattr(e, 'details') and isinstance(e.details, dict) else None
        if not error_id: # If not already captured
            error_id = capture_exception(e, {"job_id": job_id, "video_url": video_url, "animation_type": animation})
            if hasattr(e, 'details') and isinstance(e.details, dict):
                e.details["error_id"] = error_id
            elif not hasattr(e, 'details'):
                 e.details = {"error_id": error_id}


        logger.error(f"Job {job_id}: Error en process_animated_text ({e.error_code}): {e.message}. Details: {e.details}. Error ID: {error_id}")
        raise # Re-raise the original custom error
    except Exception as e: # Catch any other unexpected errors
        error_id = capture_exception(e, {"job_id": job_id, "video_url": video_url, "animation_type": animation})
        logger.error(f"Job {job_id}: Error inesperado en process_animated_text: {str(e)}. Error ID: {error_id}", exc_info=True)
        # Wrap in a generic ProcessingError
        raise ProcessingError(message=f"Error inesperado procesando texto animado: {str(e)}",
                              error_code="animated_text_unexpected_error",
                              details={"original_error": str(type(e).__name__), "error_id": error_id})
    finally:
        # Cleanup input file if it was downloaded and still exists
        if video_path_local and os.path.exists(video_path_local):
            try:
                os.remove(video_path_local)
                logger.debug(f"Job {job_id}: Archivo de video de entrada limpiado: {video_path_local}")
            except OSError as e_clean:
                capture_exception(e_clean, {"job_id": job_id, "file_path": video_path_local, "context": "cleanup_input_video_animated_text"})
                logger.warning(f"Job {job_id}: No se pudo limpiar el archivo de video de entrada '{video_path_local}': {str(e_clean)}")
        # Note: output_path_local is returned on success. If an error occurs before return,
        # it might be left behind. Consider a more robust temporary file management strategy
        # for intermediate files if this becomes an issue (e.g., using tempfile.NamedTemporaryFile context managers
        # or registering files for cleanup in a broader scope).


def generate_text_animation_filter(text: str, animation: str, position: str, font: str,
                                   font_size: int, color: str, duration: float,
                                   video_width: int, video_height: int) -> str:
    """
    Genera el filtro ffmpeg para animación de texto.
    Args:
        font (str): Ruta al archivo de fuente.
    """
    # Escape special characters for FFmpeg drawtext filter
    # : \ ' " %
    # Escaping single quotes for the text content itself.
    # Colon needs escaping if it's part of the text and not a filter option separator.
    # Percent needs escaping.
    # This is a simplified escaper. A more robust one would handle more cases or use library functions if available.
    text_escaped = text.replace("'", r"\'").replace(":", r"\:").replace("%", r"%%")
    
    # Font path escaping for FFmpeg (especially important on Windows)
    # FFmpeg generally prefers forward slashes.
    font_path_escaped = font.replace('\\', '/')
    if platform.system() == "Windows":
        # On Windows, colons in paths (e.g., C:\...) need to be escaped for some filters.
        font_path_escaped = font_path_escaped.replace(':', r'\\:')


    # Relative positioning
    y_positions = {
        "top": f"main_h*0.1",  # 10% from top
        "bottom": f"main_h*0.9 - text_h", # 10% from bottom, considering text height
        "center": f"(main_h-text_h)/2"
    }
    y_pos = y_positions.get(position.lower(), y_positions["center"]) # Default to center
    x_pos = f"(main_w-text_w)/2" # Always horizontally centered

    # Base drawtext options
    # Using main_w, main_h which are available in drawtext context
    base_options = f"text='{text_escaped}':fontfile='{font_path_escaped}':fontsize={font_size}:fontcolor={color}:x={x_pos}:y={y_pos}"

    filter_str = ""
    fade_d = min(duration / 4, 1.0) # Common fade duration, 1/4 of animation or 1s max

    if animation == "fade":
        # Fade in for fade_d, hold, fade out for fade_d
        # enable='between(t,0,duration)' ensures it's only active during the animation duration
        filter_str = f"drawtext={base_options}:alpha='if(lt(t,{fade_d}),t/{fade_d},if(lt(t,{duration-fade_d}),1,if(lt(t,{duration}),({duration}-t)/{fade_d},0)))':enable='between(t,0,{duration})'"
    
    elif animation == "slide":
        # Slide in from left, hold, (optional: slide out)
        # x position changes from -text_w to target x_pos
        slide_in_x = f"(-text_w + ({x_pos}+text_w)*(t/{fade_d}))" # Slides in over fade_d seconds
        hold_x = x_pos
        # For simplicity, this slide-in stops at the target position.
        # A slide-out would require more complex time conditioning.
        filter_str = f"drawtext={base_options}:x='if(lt(t,{fade_d}),{slide_in_x},{hold_x})':enable='between(t,0,{duration})'"

    elif animation == "zoom":
        # Zoom from 0 to font_size over fade_d seconds, hold
        current_fs = f"({font_size}*(t/{fade_d}))"
        filter_str = f"drawtext={base_options}:fontsize='if(lt(t,{fade_d}),{current_fs},{font_size})':enable='between(t,0,{duration})'"
        
    elif animation == "typewriter":
        # Ensure duration is not zero to avoid division by zero
        safe_duration = max(duration, 0.001)
        chars_per_second = len(text) / safe_duration
        # `n` is the number of characters to draw. `t*chars_per_second` gives chars over time.
        # `trunc` ensures integer. `min` caps at text length.
        num_chars_to_draw = f"trunc(min(t*{chars_per_second}, {len(text)}))"
        filter_str = f"drawtext={base_options}:text='{text_escaped}':expansion=none:alpha=1:fix_bounds=1:enable='between(t,0,{duration})',drawtext=text_shaping=0:text='{text_escaped}':fontfile='{font_path_escaped}':fontsize={font_size}:fontcolor={color}:x={x_pos}:y={y_pos}:fix_bounds=1:alpha='if(gt(n,{num_chars_to_draw}),0,1)':enable='between(t,0,{duration})'"
        # The typewriter effect is tricky. FFmpeg's native drawtext isn't ideal.
        # A more robust way often involves pre-rendering text or using complex expressions.
        # This is a simplified attempt. It might need a character-by-character substring approach.
        # For example: text=expr_str_eval('substring(string\, 0\, N)') where N is calculated.
        # This is highly complex to get right with escaping in one filter string.
        # A simpler "reveal" might be better:
        reveal_width = f"(w*({len(text)}/max(1,floor(t*{chars_per_second})))/ {len(text)})" # This logic is flawed.
        # Simpler typewriter (character count based enable of drawtext filter)
        # This is a placeholder, a true typewriter is hard with a single drawtext expression for variable text length
        logger.warning("Typewriter animation is a simplified implementation and might not be perfect.")
        filter_str = f"drawtext={base_options}:alpha=1:enable='between(t,0,{duration})'" # Fallback for typewriter for now
        # A better approach for typewriter is often to use a script that generates many drawtext commands or a subtitle file.


    elif animation == "bounce":
        # Bounce effect on y-axis
        # ensure duration is not zero
        safe_duration = max(duration, 0.001)
        bounce_freq_hz = 2.0 / safe_duration # e.g., 2 bounces over the duration
        bounce_amplitude_px = font_size * 0.3 # Bounce 30% of font size
        # Damped sine wave for y: y_base + amp * exp(-decay*t) * sin(2*PI*freq*t)
        decay_factor = 2.0 / safe_duration # decay over duration
        bounce_y = f"({y_pos} + {bounce_amplitude_px} * exp(-{decay_factor}*t) * sin(2*PI*{bounce_freq_hz}*t))"
        filter_str = f"drawtext={base_options}:y={bounce_y}:enable='between(t,0,{duration})'"

    else: # Default: simple fade in/out
        logger.warning(f"Animación desconocida '{animation}', usando 'fade' por defecto.")
        filter_str = f"drawtext={base_options}:alpha='if(lt(t,{fade_d}),t/{fade_d},if(lt(t,{duration-fade_d}),1,if(lt(t,{duration}),({duration}-t)/{fade_d},0)))':enable='between(t,0,{duration})'"
    
    logger.debug(f"Generated filter_complex: {filter_str}")
    return filter_str

# --- END OF FILE animated_text.py ---
