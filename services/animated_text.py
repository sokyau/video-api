import os
import subprocess
import logging
import tempfile
from services.file_management import download_file, generate_temp_filename
import config

logger = logging.getLogger(__name__)

def process_animated_text(video_url, text, animation, position, font, font_size, color, duration, job_id):
    """
    Procesa un video añadiendo texto animado
    
    Args:
        video_url (str): URL del video base
        text (str): Texto a animar
        animation (str): Tipo de animación (fade, slide, zoom, typewriter, bounce)
        position (str): Posición del texto (top, bottom, center)
        font (str): Fuente del texto
        font_size (int): Tamaño de fuente
        color (str): Color del texto
        duration (float): Duración de la animación en segundos
        job_id (str): ID del trabajo
    
    Returns:
        str: Ruta al video procesado
    """
    try:
        # Descargar video
        video_path = download_file(video_url, config.TEMP_DIR)
        logger.info(f"Downloaded video to {video_path}")
        
        # Preparar ruta de salida
        output_path = generate_temp_filename(prefix=f"{job_id}_", suffix=".mp4")
        
        # Obtener información del video (dimensiones y duración)
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height:format=duration',
            '-of', 'json',
            video_path
        ]
        
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        import json
        video_info = json.loads(result.stdout)
        
        video_width = video_info['streams'][0]['width']
        video_height = video_info['streams'][0]['height']
        video_duration = float(video_info['format']['duration'])
        
        logger.info(f"Video dimensions: {video_width}x{video_height}, duration: {video_duration}s")
        
        # Limitar duración de animación a la duración del video
        if duration > video_duration:
            duration = video_duration
        
        # Generar filtros según el tipo de animación
        filter_complex = generate_text_animation_filter(
            text, animation, position, font, font_size, color, 
            duration, video_width, video_height
        )
        
        # Comando FFmpeg
        cmd = [
            'ffmpeg', '-i', video_path,
            '-filter_complex', filter_complex,
            '-c:a', 'copy',
            '-y', output_path
        ]
        
        logger.info(f"Running FFmpeg command for animated text")
        logger.debug(f"Command: {' '.join(cmd)}")
        
        # Ejecutar FFmpeg
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg command failed. Error: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
        
        logger.info(f"Video with animated text created successfully: {output_path}")
        
        # Limpiar archivo de entrada
        os.remove(video_path)
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error in process_animated_text: {str(e)}", exc_info=True)
        # Limpiar archivos temporales si existen
        if 'video_path' in locals() and os.path.exists(video_path):
            os.remove(video_path)
        raise

def generate_text_animation_filter(text, animation, position, font, font_size, color, duration, video_width, video_height):
    """
    Genera el filtro ffmpeg para animación de texto
    
    Args:
        text (str): Texto a animar
        animation (str): Tipo de animación
        position (str): Posición del texto
        font (str): Fuente del texto
        font_size (int): Tamaño de fuente
        color (str): Color del texto
        duration (float): Duración de la animación
        video_width (int): Ancho del video
        video_height (int): Alto del video
    
    Returns:
        str: Filtro complejo para ffmpeg
    """
    # Escapar comillas en el texto
    text = text.replace('"', '\\"')
    
    # Calcular posición vertical según la opción elegida
    if position == "top":
        y_pos = f"h*0.1"
    elif position == "bottom":
        y_pos = f"h*0.9"
    else:  # center
        y_pos = f"h*0.5"
    
    # X siempre centrado horizontalmente
    x_pos = f"(w-text_w)/2"
    
    # Base de filtros de texto (común a todas las animaciones)
    text_base = f"drawtext=text='{text}':fontfile={font}:fontsize={font_size}:fontcolor={color}:x={x_pos}:y={y_pos}"
    
    # Filtros específicos según el tipo de animación
    if animation == "fade":
        # Fade in y fade out
        fade_duration = min(duration / 3, 1.0)  # Máximo 1 segundo para fade
        text_filter = f"{text_base}:alpha='if(lt(t,{fade_duration}),t/{fade_duration},if(lt(t,{duration-fade_duration}),1,1-(t-{duration-fade_duration})/{fade_duration}))'"
    
    elif animation == "slide":
        # Deslizar desde fuera de la pantalla
        text_filter = f"{text_base}:x='if(lt(t,{duration}),(w-text_w)*(t/{duration}),{x_pos})'"
    
    elif animation == "zoom":
        # Zoom desde tamaño pequeño a normal
        text_filter = f"drawtext=text='{text}':fontfile={font}:fontcolor={color}:x={x_pos}:y={y_pos}:fontsize='if(lt(t,{duration}),{font_size}*t/{duration},{font_size})'"
    
    elif animation == "typewriter":
        # Efecto máquina de escribir (aparece caracter por caracter)
        text_length = len(text)
        chars_per_second = text_length / duration
        text_filter = f"drawtext=text='{text}':fontfile={font}:fontsize={font_size}:fontcolor={color}:x={x_pos}:y={y_pos}:draw='lt(t*{chars_per_second},plen)':expansion=normal"
    
    elif animation == "bounce":
        # Efecto rebote
        bounce_freq = 2.0  # Frecuencia del rebote
        bounce_amp = 20.0  # Amplitud del rebote (pixels)
        text_filter = f"{text_base}:y='{y_pos}+{bounce_amp}*sin({bounce_freq}*PI*t)'"
    
    else:
        # Fallback a animación simple de fade por defecto
        text_filter = f"{text_base}:alpha='if(lt(t,1),t,if(lt(t,{duration-1}),1,{duration}-t))'"
    
    return text_filter
