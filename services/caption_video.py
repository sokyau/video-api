import os
import subprocess
import logging
from services.file_management import download_file, generate_temp_filename
import config

logger = logging.getLogger(__name__)

def add_captions_to_video(video_url, subtitles_url, font, font_size, font_color, background, position, job_id):
    """
    Añade subtítulos a un video
    
    Args:
        video_url (str): URL del video
        subtitles_url (str): URL del archivo de subtítulos (SRT, VTT)
        font (str): Nombre de la fuente para los subtítulos
        font_size (int): Tamaño de la fuente
        font_color (str): Color de la fuente (nombre o código hex)
        background (bool): Si debe tener fondo oscuro detrás del texto
        position (str): Posición de los subtítulos (bottom, top)
        job_id (str): ID del trabajo
    
    Returns:
        str: Ruta al video con subtítulos
    """
    try:
        # Descargar archivos
        video_path = download_file(video_url, config.TEMP_DIR)
        subtitles_path = download_file(subtitles_url, config.TEMP_DIR)
        
        logger.info(f"Downloaded video to {video_path}")
        logger.info(f"Downloaded subtitles to {subtitles_path}")

        # Verificar extensión de subtítulos
        subtitle_ext = os.path.splitext(subtitles_path)[1].lower()
        if subtitle_ext not in ['.srt', '.vtt', '.ass', '.ssa']:
            raise ValueError(f"Unsupported subtitle format: {subtitle_ext}. Supported formats: .srt, .vtt, .ass, .ssa")
        
        # Preparar ruta de salida
        output_path = generate_temp_filename(prefix=f"{job_id}_", suffix=".mp4")
        
        # Construir filtro de subtítulos
        subtitle_filter = build_subtitle_filter(subtitles_path, font, font_size, font_color, background, position)
        
        # Comando FFmpeg para añadir subtítulos
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', subtitle_filter,
            '-c:a', 'copy',
            '-y', output_path
        ]
        
        logger.info(f"Running FFmpeg command for adding subtitles")
        logger.debug(f"Command: {' '.join(cmd)}")
        
        # Ejecutar FFmpeg
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg command failed. Error: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
        
        logger.info(f"Video with subtitles created successfully: {output_path}")
        
        # Limpiar archivos de entrada
        os.remove(video_path)
        os.remove(subtitles_path)
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error in add_captions_to_video: {str(e)}", exc_info=True)
        # Limpiar archivos temporales si existen
        if 'video_path' in locals() and os.path.exists(video_path):
            os.remove(video_path)
        if 'subtitles_path' in locals() and os.path.exists(subtitles_path):
            os.remove(subtitles_path)
        raise

def build_subtitle_filter(subtitles_path, font, font_size, font_color, background, position):
    """
    Construye el filtro de subtítulos para FFmpeg
    
    Args:
        subtitles_path (str): Ruta al archivo de subtítulos
        font (str): Nombre de la fuente
        font_size (int): Tamaño de la fuente
        font_color (str): Color de la fuente
        background (bool): Si debe tener fondo
        position (str): Posición de los subtítulos (bottom, top)
    
    Returns:
        str: Filtro FFmpeg para subtítulos
    """
    # Escapar ruta para FFmpeg
    subtitles_path_escaped = subtitles_path.replace(':', '\\:').replace('\\', '\\\\')
    
    # Determinar si los subtítulos son externos (SRT, VTT) o internos (SSA, ASS)
    subtitle_ext = os.path.splitext(subtitles_path)[1].lower()
    
    if subtitle_ext in ['.srt', '.vtt']:
        # Configuración para SRT/VTT
        filter_options = []
        
        # Fuente
        if font:
            filter_options.append(f"fontname={font}")
        
        # Tamaño
        filter_options.append(f"fontsize={font_size}")
        
        # Color
        filter_options.append(f"fontcolor={font_color}")
        
        # Fondo
        if background:
            filter_options.append("force_style='BackColour=&H80000000,Outline=0'")
        
        # Posición
        if position == "top":
            filter_options.append("marginv=20")
        else:  # bottom (default)
            filter_options.append("marginv=30")
        
        # Construir filtro
        return f"subtitles={subtitles_path_escaped}:{':'.join(filter_options)}"
    
    else:  # .ass, .ssa
        # Para formatos ASS/SSA, simplemente usamos el archivo directamente
        return f"ass={subtitles_path_escaped}"
