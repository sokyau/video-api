import os
import subprocess
import logging
import json
from services.file_management import generate_temp_filename, get_file_extension
import config

logger = logging.getLogger(__name__)

def get_video_info(file_path):
    """
    Obtiene información detallada sobre un archivo de video usando ffprobe
    
    Args:
        file_path (str): Ruta al archivo de video
    
    Returns:
        dict: Información del video (duración, resolución, codec, etc.)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Video file not found: {file_path}")
    
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        file_path
    ]
    
    logger.debug(f"Running ffprobe command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        
        # Extraer información relevante
        video_info = {
            'format': info.get('format', {}).get('format_name', 'unknown'),
            'duration': float(info.get('format', {}).get('duration', 0)),
            'size': int(info.get('format', {}).get('size', 0)),
            'bit_rate': int(info.get('format', {}).get('bit_rate', 0)),
            'streams': []
        }
        
        # Procesar información de streams
        for stream in info.get('streams', []):
            stream_type = stream.get('codec_type')
            
            if stream_type == 'video':
                video_info['streams'].append({
                    'type': 'video',
                    'codec': stream.get('codec_name', 'unknown'),
                    'width': stream.get('width', 0),
                    'height': stream.get('height', 0),
                    'fps': eval(stream.get('r_frame_rate', '0/1')),
                    'bit_rate': int(stream.get('bit_rate', 0)),
                    'index': stream.get('index', 0)
                })
            elif stream_type == 'audio':
                video_info['streams'].append({
                    'type': 'audio',
                    'codec': stream.get('codec_name', 'unknown'),
                    'channels': stream.get('channels', 0),
                    'sample_rate': stream.get('sample_rate', 0),
                    'bit_rate': int(stream.get('bit_rate', 0)),
                    'index': stream.get('index', 0)
                })
                
        logger.debug(f"Video info extracted for {file_path}")
        return video_info
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running ffprobe: {e.stderr}")
        raise RuntimeError(f"Error getting video info: {e.stderr}")
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing ffprobe output: {e}")
        raise RuntimeError(f"Error parsing video info: {e}")

def convert_video(input_path, output_path=None, format=None, video_codec=None, 
                 audio_codec=None, width=None, height=None, bitrate=None, 
                 framerate=None, extra_args=None):
    """
    Convierte un video a otro formato o configuración usando ffmpeg
    
    Args:
        input_path (str): Ruta al archivo de entrada
        output_path (str, optional): Ruta para el archivo de salida
        format (str, optional): Formato de salida (mp4, webm, etc.)
        video_codec (str, optional): Codec de video (h264, vp9, etc.)
        audio_codec (str, optional): Codec de audio (aac, mp3, etc.)
        width (int, optional): Ancho del video de salida
        height (int, optional): Alto del video de salida
        bitrate (str, optional): Bitrate de salida (ej. "2M")
        framerate (int, optional): Framerate de salida
        extra_args (list, optional): Argumentos adicionales para ffmpeg
    
    Returns:
        str: Ruta al archivo de salida
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Generar ruta de salida si no se proporciona
    if output_path is None:
        ext = '.mp4'
        if format:
            ext = f".{format}"
        output_path = generate_temp_filename(suffix=ext)
    
    # Construir comando básico
    cmd = ['ffmpeg', '-i', input_path]
    
    # Agregar opciones de conversión
    if video_codec:
        cmd.extend(['-c:v', video_codec])
    
    if audio_codec:
        cmd.extend(['-c:a', audio_codec])
    
    if width and height:
        cmd.extend(['-vf', f'scale={width}:{height}'])
    
    if bitrate:
        cmd.extend(['-b:v', bitrate])
    
    if framerate:
        cmd.extend(['-r', str(framerate)])
    
    # Agregar argumentos extra si se proporcionan
    if extra_args:
        cmd.extend(extra_args)
    
    # Forzar sobreescritura si el archivo existe
    cmd.extend(['-y', output_path])
    
    logger.debug(f"Running ffmpeg command: {' '.join(cmd)}")
    
    try:
        # Ejecutar ffmpeg
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Video conversion successful: {input_path} -> {output_path}")
        return output_path
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in ffmpeg conversion: {e.stderr}")
        raise RuntimeError(f"Video conversion failed: {e.stderr}")

def extract_audio(video_path, output_path=None, format='mp3', bitrate=None):
    """
    Extrae el audio de un video
    
    Args:
        video_path (str): Ruta al archivo de video
        output_path (str, optional): Ruta para el archivo de audio
        format (str, optional): Formato de audio (mp3, wav, etc.)
        bitrate (str, optional): Bitrate del audio (ej. "192k")
    
    Returns:
        str: Ruta al archivo de audio
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Generar ruta de salida si no se proporciona
    if output_path is None:
        output_path = generate_temp_filename(suffix=f".{format}")
    
    # Construir comando
    cmd = ['ffmpeg', '-i', video_path, '-vn']  # -vn para descartar video
    
    if bitrate:
        cmd.extend(['-b:a', bitrate])
    
    cmd.extend(['-y', output_path])
    
    logger.debug(f"Running ffmpeg audio extraction: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Audio extraction successful: {video_path} -> {output_path}")
        return output_path
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in audio extraction: {e.stderr}")
        raise RuntimeError(f"Audio extraction failed: {e.stderr}")

def concatenate_videos(video_paths, output_path=None):
    """
    Concatena múltiples videos en uno solo
    
    Args:
        video_paths (list): Lista de rutas a los videos a concatenar
        output_path (str, optional): Ruta para el video resultante
    
    Returns:
        str: Ruta al video concatenado
    """
    if not video_paths:
        raise ValueError("No video paths provided for concatenation")
    
    for path in video_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")
    
    # Generar ruta de salida si no se proporciona
    if output_path is None:
        ext = get_file_extension(video_paths[0])
        output_path = generate_temp_filename(suffix=ext)
    
    # Crear archivo temporal de lista
    list_file = generate_temp_filename(suffix=".txt")
    
    try:
        with open(list_file, 'w') as f:
            for path in video_paths:
                f.write(f"file '{path}'\n")
        
        # Construir comando
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', list_file,
            '-c', 'copy',  # Copiar streams sin recodificar
            '-y', output_path
        ]
        
        logger.debug(f"Running ffmpeg concatenation: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Video concatenation successful: {len(video_paths)} videos -> {output_path}")
        
        return output_path
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in video concatenation: {e.stderr}")
        raise RuntimeError(f"Video concatenation failed: {e.stderr}")
    
    finally:
        # Limpiar archivo temporal de lista
        if os.path.exists(list_file):
            os.remove(list_file)

def add_subtitles(video_path, subtitles_path, output_path=None, font=None, fontsize=None):
    """
    Añade subtítulos a un video
    
    Args:
        video_path (str): Ruta al archivo de video
        subtitles_path (str): Ruta al archivo de subtítulos (srt, vtt, etc.)
        output_path (str, optional): Ruta para el video con subtítulos
        font (str, optional): Fuente para los subtítulos
        fontsize (int, optional): Tamaño de fuente para los subtítulos
    
    Returns:
        str: Ruta al video con subtítulos
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if not os.path.exists(subtitles_path):
        raise FileNotFoundError(f"Subtitles file not found: {subtitles_path}")
    
    # Generar ruta de salida si no se proporciona
    if output_path is None:
        ext = get_file_extension(video_path)
        output_path = generate_temp_filename(suffix=ext)
    
    # Construir comando
    cmd = ['ffmpeg', '-i', video_path]
    
    subtitle_options = []
    if font:
        subtitle_options.append(f"fontname={font}")
    if fontsize:
        subtitle_options.append(f"fontsize={fontsize}")
    
    # Determinar cómo aplicar los subtítulos según el formato
    subtitle_ext = get_file_extension(subtitles_path).lower()
    
    if subtitle_ext in ['.srt', '.vtt']:
        # Subtítulos como filtro
        filter_options = ':'.join(subtitle_options) if subtitle_options else ''
        filter_expr = f"subtitles={subtitles_path.replace(':', '\\:')}:{filter_options}"
        cmd.extend(['-vf', filter_expr])
        cmd.extend(['-c:a', 'copy'])
    else:
        # Subtítulos como stream separado
        cmd.extend(['-i', subtitles_path])
        cmd.extend(['-c:v', 'copy', '-c:a', 'copy', '-c:s', 'mov_text'])
    
    cmd.extend(['-y', output_path])
    
    logger.debug(f"Running ffmpeg add subtitles: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Added subtitles to video: {video_path} + {subtitles_path} -> {output_path}")
        return output_path
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error adding subtitles: {e.stderr}")
        raise RuntimeError(f"Failed to add subtitles: {e.stderr}")

def image_to_video(image_path, duration, output_path=None, width=None, height=None):
    """
    Convierte una imagen a video con duración especificada
    
    Args:
        image_path (str): Ruta a la imagen
        duration (float): Duración del video en segundos
        output_path (str, optional): Ruta para el video resultante
        width (int, optional): Ancho del video
        height (int, optional): Alto del video
    
    Returns:
        str: Ruta al video creado
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Generar ruta de salida si no se proporciona
    if output_path is None:
        output_path = generate_temp_filename(suffix=".mp4")
    
    # Construir comando
    cmd = ['ffmpeg', '-loop', '1', '-i', image_path, '-c:v', 'libx264']
    
    # Configurar resolución si se especifica
    if width and height:
        cmd.extend(['-vf', f'scale={width}:{height}'])
    
    # Configurar duración
    cmd.extend(['-t', str(duration)])
    
    # Ajustes adicionales para mejor calidad
    cmd.extend(['-pix_fmt', 'yuv420p', '-y', output_path])
    
    logger.debug(f"Running ffmpeg image to video: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Converted image to video: {image_path} -> {output_path} ({duration}s)")
        return output_path
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting image to video: {e.stderr}")
        raise RuntimeError(f"Failed to convert image to video: {e.stderr}")

def trim_video(video_path, start_time, duration=None, end_time=None, output_path=None):
    """
    Recorta un video según tiempo de inicio y duración o tiempo final
    
    Args:
        video_path (str): Ruta al video
        start_time (float): Tiempo de inicio en segundos
        duration (float, optional): Duración del recorte en segundos
        end_time (float, optional): Tiempo final en segundos (alternativa a duration)
        output_path (str, optional): Ruta para el video recortado
    
    Returns:
        str: Ruta al video recortado
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Verificar parámetros de tiempo
    if duration is None and end_time is None:
        raise ValueError("Either duration or end_time must be provided")
    
    # Calcular duración si se proporcionó end_time
    if duration is None:
        duration = end_time - start_time
    
    # Generar ruta de salida si no se proporciona
    if output_path is None:
        ext = get_file_extension(video_path)
        output_path = generate_temp_filename(suffix=ext)
    
    # Construir comando
    cmd = [
        'ffmpeg',
        '-ss', str(start_time),
        '-i', video_path,
        '-t', str(duration),
        '-c:v', 'copy',
        '-c:a', 'copy',
        '-avoid_negative_ts', '1',
        '-y', output_path
    ]
    
    logger.debug(f"Running ffmpeg trim video: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Trimmed video: {video_path} -> {output_path} (start={start_time}, duration={duration})")
        return output_path
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error trimming video: {e.stderr}")
        raise RuntimeError(f"Failed to trim video: {e.stderr}")

def add_watermark(video_path, watermark_path, position, scale=0.1, output_path=None):
    """
    Añade una marca de agua (imagen) a un video
    
    Args:
        video_path (str): Ruta al video
        watermark_path (str): Ruta a la imagen de marca de agua
        position (str): Posición (top, bottom, left, right, top_left, top_right, bottom_left, bottom_right, center)
        scale (float, optional): Escala relativa a la resolución del video (0.0-1.0)
        output_path (str, optional): Ruta para el video resultante
    
    Returns:
        str: Ruta al video con marca de agua
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if not os.path.exists(watermark_path):
        raise FileNotFoundError(f"Watermark image not found: {watermark_path}")
    
    # Generar ruta de salida si no se proporciona
    if output_path is None:
        ext = get_file_extension(video_path)
        output_path = generate_temp_filename(suffix=ext)
    
    # Mapeo de posiciones a expresiones filter_complex
    positions = {
        "top": f"x=(main_w-overlay_w)/2:y=main_h*0.05",
        "bottom": f"x=(main_w-overlay_w)/2:y=main_h*0.95-overlay_h",
        "left": f"x=main_w*0.05:y=(main_h-overlay_h)/2",
        "right": f"x=main_w*0.95-overlay_w:y=(main_h-overlay_h)/2",
        "top_left": f"x=main_w*0.05:y=main_h*0.05",
        "top_right": f"x=main_w*0.95-overlay_w:y=main_h*0.05",
        "bottom_left": f"x=main_w*0.05:y=main_h*0.95-overlay_h",
        "bottom_right": f"x=main_w*0.95-overlay_w:y=main_h*0.95-overlay_h",
        "center": f"x=(main_w-overlay_w)/2:y=(main_h-overlay_h)/2"
    }
    
    # Verificar que la posición sea válida
    if position not in positions:
        raise ValueError(f"Invalid position: {position}. Valid positions: {', '.join(positions.keys())}")
    
    # Construir comando
    filter_complex = f"[0:v][1:v]scale2ref=oh*mdar:ih*{scale}[main][overlay];[main][overlay]overlay={positions[position]}:format=auto,format=yuv420p"
    
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-i', watermark_path,
        '-filter_complex', filter_complex,
        '-c:a', 'copy',
        '-y', output_path
    ]
    
    logger.debug(f"Running ffmpeg add watermark: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Added watermark to video: {video_path} + {watermark_path} -> {output_path}")
        return output_path
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error adding watermark: {e.stderr}")
        raise RuntimeError(f"Failed to add watermark: {e.stderr}")
