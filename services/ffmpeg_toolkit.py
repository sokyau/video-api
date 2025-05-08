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
from services.file_management import generate_temp_filename, get_file_extension, verify_file_integrity
import config

logger = logging.getLogger(__name__)

# Constantes de configuración
DEFAULT_TIMEOUT = getattr(config, 'FFMPEG_TIMEOUT', 1800)  # 30 minutos por defecto
MAX_THREADS = getattr(config, 'FFMPEG_THREADS', 4)
MAX_RETRIES = getattr(config, 'FFMPEG_MAX_RETRIES', 2)

def get_optimal_thread_count() -> int:
    """
    Determina el número óptimo de hilos basado en la carga del sistema
    
    Returns:
        int: Número óptimo de hilos para FFmpeg
    """
    try:
        cpu_count = os.cpu_count() or 4
        # Verificar carga del sistema
        if platform.system() != "Windows":
            load = os.getloadavg()[0]
            # Si la carga es alta, reducir hilos
            if load > cpu_count * 0.8:
                return max(2, cpu_count // 2)
        
        # Verificar memoria disponible
        mem = psutil.virtual_memory()
        if mem.percent > 85:  # Si memoria está muy usada, reducir hilos
            return max(2, cpu_count // 2)
            
        return min(cpu_count, MAX_THREADS)
    except Exception as e:
        logger.warning(f"Error determinando hilos óptimos: {str(e)}. Usando valor por defecto.")
        return MAX_THREADS

def validate_ffmpeg_parameters(params: Union[str, List[str]]) -> None:
    """
    Valida los parámetros de FFmpeg para prevenir inyección de comandos
    
    Args:
        params: Parámetro individual o lista de parámetros
    
    Raises:
        ValueError: Si se detecta un parámetro potencialmente peligroso
    """
    dangerous_patterns = [
        r'`.*`',                    # Comandos backtick
        r'\$\(.*\)',                # Sustitución de comandos
        r'\$\{.*\}',                # Expansión de variables
        r'[;&|]',                   # Operadores shell
        r'^-f\s+lavfi',             # Filtro de entrada lavfi (puede ser peligroso)
        r'system\s*\(',             # Llamada a system()
        r'exec\s*\(',               # Llamada a exec()
        r'subprocess',              # Referencias a subprocess
        r'>[^,\s]*',                # Redirección de salida
        r'<[^,\s]*',                # Redirección de entrada
        r'-filter_complex.*\bsh\b', # Filtro sh
    ]
    
    # Convertir a lista si es un string
    if isinstance(params, str):
        params_list = [params]
    else:
        params_list = params
    
    # Verificar cada parámetro
    for param in params_list:
        param_str = str(param).lower()
        
        # Verificar patrones peligrosos
        for pattern in dangerous_patterns:
            if re.search(pattern, param_str):
                logger.warning(f"Parámetro FFmpeg peligroso detectado: {param}")
                raise ValueError(f"Parámetro FFmpeg potencialmente inseguro: {param}")

def get_video_info(file_path: str) -> Dict[str, Any]:
    """
    Obtiene información detallada sobre un archivo de video usando ffprobe
    
    Args:
        file_path (str): Ruta al archivo de video
    
    Returns:
        dict: Información del video (duración, resolución, codec, etc.)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Archivo de video no encontrado: {file_path}")
    
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        file_path
    ]
    
    logger.debug(f"Ejecutando comando ffprobe: {' '.join(cmd)}")
    
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
                    'bit_rate': int(stream.get('bit_rate', 0)) if stream.get('bit_rate') else 0,
                    'index': stream.get('index', 0),
                    'duration': float(stream.get('duration', 0)) if stream.get('duration') else video_info['duration'],
                    'pix_fmt': stream.get('pix_fmt', 'unknown')
                })
            elif stream_type == 'audio':
                video_info['streams'].append({
                    'type': 'audio',
                    'codec': stream.get('codec_name', 'unknown'),
                    'channels': stream.get('channels', 0),
                    'sample_rate': stream.get('sample_rate', 0),
                    'bit_rate': int(stream.get('bit_rate', 0)) if stream.get('bit_rate') else 0,
                    'index': stream.get('index', 0),
                    'duration': float(stream.get('duration', 0)) if stream.get('duration') else video_info['duration']
                })
        
        # Extraer información resumida para fácil acceso
        video_streams = [s for s in video_info['streams'] if s['type'] == 'video']
        audio_streams = [s for s in video_info['streams'] if s['type'] == 'audio']
        
        if video_streams:
            video_info['width'] = video_streams[0]['width']
            video_info['height'] = video_streams[0]['height']
            video_info['fps'] = video_streams[0]['fps']
            video_info['video_codec'] = video_streams[0]['codec']
        
        if audio_streams:
            video_info['audio_codec'] = audio_streams[0]['codec']
            video_info['audio_channels'] = audio_streams[0]['channels']
        
        # Calcular relación de aspecto
        if video_streams and video_streams[0]['width'] > 0 and video_streams[0]['height'] > 0:
            video_info['aspect_ratio'] = video_streams[0]['width'] / video_streams[0]['height']
        
        # Formato de duración más legible
        if video_info['duration'] > 0:
            mins, secs = divmod(video_info['duration'], 60)
            hours, mins = divmod(mins, 60)
            video_info['duration_formatted'] = f"{int(hours)}:{int(mins):02d}:{int(secs):02d}"
        
        logger.debug(f"Información del video extraída para {file_path}")
        return video_info
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error ejecutando ffprobe: {e.stderr}")
        raise RuntimeError(f"Error obteniendo información del video: {e.stderr}")
    except json.JSONDecodeError as e:
        logger.error(f"Error parseando salida de ffprobe: {e}")
        raise RuntimeError(f"Error parseando información del video: {e}")

def convert_video(input_path: str, output_path: Optional[str] = None, format: Optional[str] = None, 
                 video_codec: Optional[str] = None, audio_codec: Optional[str] = None, 
                 width: Optional[int] = None, height: Optional[int] = None, 
                 bitrate: Optional[str] = None, framerate: Optional[int] = None, 
                 extra_args: Optional[List[str]] = None, timeout: Optional[int] = None) -> str:
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
        timeout (int, optional): Timeout en segundos
    
    Returns:
        str: Ruta al archivo de salida
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Archivo de entrada no encontrado: {input_path}")
    
    # Generar ruta de salida si no se proporciona
    if output_path is None:
        ext = '.mp4'
        if format:
            ext = f".{format}"
        output_path = generate_temp_filename(suffix=ext)
    
    # Obtener información del video de entrada
    try:
        input_info = get_video_info(input_path)
        logger.debug(f"Información del video de entrada: {input_info}")
    except Exception as e:
        logger.warning(f"No se pudo obtener información del video de entrada: {str(e)}")
        input_info = {}
    
    # Construir comando básico
    cmd = ['ffmpeg', '-y', '-i', input_path]
    
    # Determinar número óptimo de hilos
    threads = get_optimal_thread_count()
    
    # Agregar opciones de conversión
    if video_codec:
        cmd.extend(['-c:v', video_codec])
    
    if audio_codec:
        cmd.extend(['-c:a', audio_codec])
    
    # Escalar video si se especifica width o height
    if width or height:
        # Si solo se especifica una dimensión, mantener la relación de aspecto
        if width and not height and 'aspect_ratio' in input_info:
            height = int(width / input_info['aspect_ratio'])
        elif height and not width and 'aspect_ratio' in input_info:
            width = int(height * input_info['aspect_ratio'])
        
        # Si ambos están especificados, usar esos valores exactos
        if width and height:
            cmd.extend(['-vf', f'scale={width}:{height}'])
    
    if bitrate:
        cmd.extend(['-b:v', bitrate])
    
    if framerate:
        cmd.extend(['-r', str(framerate)])
    
    # Configurar hilos
    cmd.extend(['-threads', str(threads)])
    
    # Agregar argumentos extra si se proporcionan
    if extra_args:
        # Validar argumentos extra para seguridad
        validate_ffmpeg_parameters(extra_args)
        cmd.extend(extra_args)
    
    # Configurar salida
    cmd.append(output_path)
    
    logger.debug(f"Ejecutando comando ffmpeg: {' '.join(cmd)}")
    
    # Ejecutar ffmpeg con timeout
    if timeout is None:
        timeout = DEFAULT_TIMEOUT
    
    try:
        start_time = time.time()
        
        # Ejecutar ffmpeg
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        
        # Esperar a que termine con timeout
        stdout, stderr = process.communicate(timeout=timeout)
        
        if process.returncode != 0:
            logger.error(f"Error en conversión ffmpeg: {stderr}")
            raise RuntimeError(f"Error en conversión de video: {stderr}")
        
        # Verificar que el archivo se generó correctamente
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("La conversión de video falló: archivo de salida vacío o inexistente")
        
        processing_time = time.time() - start_time
        logger.info(f"Conversión de video exitosa: {input_path} -> {output_path} en {processing_time:.2f} segundos")
        
        return output_path
    
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout en conversión de video después de {timeout} segundos")
        # Intentar terminar el proceso
        if 'process' in locals():
            process.kill()
        raise TimeoutError(f"La conversión de video excedió el tiempo límite de {timeout} segundos")
    
    except Exception as e:
        logger.error(f"Error en conversión de video: {str(e)}")
        raise RuntimeError(f"Error en conversión de video: {str(e)}")

def extract_audio(video_path: str, output_path: Optional[str] = None, format: str = 'mp3', 
                  bitrate: Optional[str] = None, timeout: Optional[int] = None) -> str:
    """
    Extrae el audio de un video
    
    Args:
        video_path (str): Ruta al archivo de video
        output_path (str, optional): Ruta para el archivo de audio
        format (str, optional): Formato de audio (mp3, wav, etc.)
        bitrate (str, optional): Bitrate del audio (ej. "192k")
        timeout (int, optional): Timeout en segundos
    
    Returns:
        str: Ruta al archivo de audio
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Archivo de video no encontrado: {video_path}")
    
    # Generar ruta de salida si no se proporciona
    if output_path is None:
        output_path = generate_temp_filename(suffix=f".{format}")
    
    # Determinar número óptimo de hilos
    threads = get_optimal_thread_count()
    
    # Construir comando
    cmd = [
        'ffmpeg', 
        '-y',
        '-i', video_path, 
        '-vn',  # descartar video
        '-threads', str(threads)
    ]
    
    if bitrate:
        cmd.extend(['-b:a', bitrate])
    
    # Codecs específicos según formato
    if format == 'mp3':
        cmd.extend(['-codec:a', 'libmp3lame', '-q:a', '2'])
    elif format == 'aac':
        cmd.extend(['-codec:a', 'aac', '-strict', 'experimental', '-q:a', '1'])
    elif format == 'wav':
        cmd.extend(['-codec:a', 'pcm_s16le'])
    elif format == 'flac':
        cmd.extend(['-codec:a', 'flac'])
    elif format == 'ogg':
        cmd.extend(['-codec:a', 'libvorbis', '-q:a', '5'])
    
    cmd.append(output_path)
    
    logger.debug(f"Ejecutando comando de extracción de audio: {' '.join(cmd)}")
    
    # Configurar timeout
    if timeout is None:
        timeout = DEFAULT_TIMEOUT
    
    try:
        start_time = time.time()
        
        # Ejecutar ffmpeg
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        
        # Esperar a que termine con timeout
        stdout, stderr = process.communicate(timeout=timeout)
        
        if process.returncode != 0:
            logger.error(f"Error en extracción de audio: {stderr}")
            raise RuntimeError(f"Error en extracción de audio: {stderr}")
        
        # Verificar que el archivo se generó correctamente
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("La extracción de audio falló: archivo de salida vacío o inexistente")
        
        processing_time = time.time() - start_time
        logger.info(f"Extracción de audio exitosa: {video_path} -> {output_path} en {processing_time:.2f} segundos")
        
        return output_path
    
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout en extracción de audio después de {timeout} segundos")
        # Intentar terminar el proceso
        if 'process' in locals():
            process.kill()
        raise TimeoutError(f"La extracción de audio excedió el tiempo límite de {timeout} segundos")
    
    except Exception as e:
        logger.error(f"Error en extracción de audio: {str(e)}")
        raise RuntimeError(f"Error en extracción de audio: {str(e)}")

def concatenate_videos(video_paths: List[str], output_path: Optional[str] = None, 
                       transcode: bool = False, timeout: Optional[int] = None) -> str:
    """
    Concatena múltiples videos en uno solo
    
    Args:
        video_paths (list): Lista de rutas a los videos a concatenar
        output_path (str, optional): Ruta para el video resultante
        transcode (bool): Si se debe transcodificar para compatibilidad
        timeout (int, optional): Timeout en segundos
    
    Returns:
        str: Ruta al video concatenado
    """
    if not video_paths:
        raise ValueError("No se proporcionaron rutas de video para concatenación")
    
    for path in video_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Archivo de video no encontrado: {path}")
    
    # Generar ruta de salida si no se proporciona
    if output_path is None:
        ext = get_file_extension(video_paths[0])
        output_path = generate_temp_filename(suffix=ext)
    
    # Verificar si los videos son compatibles para concatenación directa
    if not transcode and len(video_paths) > 1:
        try:
            # Obtener información de todos los videos para verificar compatibilidad
            video_infos = [get_video_info(path) for path in video_paths]
            
            # Verificar si los codecs, resoluciones y formatos son compatibles
            base_codec = video_infos[0].get('video_codec', '')
            base_width = video_infos[0].get('width', 0)
            base_height = video_infos[0].get('height', 0)
            
            for info in video_infos[1:]:
                if (info.get('video_codec', '') != base_codec or
                    info.get('width', 0) != base_width or
                    info.get('height', 0) != base_height):
                    logger.info("Videos no son compatibles para concatenación directa. Usando transcoding.")
                    transcode = True
                    break
        except Exception as e:
            logger.warning(f"Error verificando compatibilidad de videos: {str(e)}. Usando transcoding.")
            transcode = True
    
    # Método 1: Concatenación directa con archivo de lista (más rápido pero requiere compatibilidad)
    if not transcode:
        # Crear archivo temporal de lista
        list_file = generate_temp_filename(suffix=".txt")
        
        try:
            with open(list_file, 'w') as f:
                for path in video_paths:
                    f.write(f"file '{path}'\n")
            
            # Construir comando
            cmd = [
                'ffmpeg',
                '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', list_file,
                '-c', 'copy',  # Copiar streams sin recodificar
                output_path
            ]
            
            logger.debug(f"Ejecutando comando de concatenación directa: {' '.join(cmd)}")
            
            if timeout is None:
                timeout = DEFAULT_TIMEOUT
            
            # Ejecutar ffmpeg
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True
            )
            
            # Esperar a que termine con timeout
            stdout, stderr = process.communicate(timeout=timeout)
            
            if process.returncode != 0:
                logger.warning(f"Concatenación directa falló, intentando con transcoding: {stderr}")
                transcode = True
            else:
                # Verificar que el archivo se generó correctamente
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    logger.info(f"Concatenación directa exitosa: {len(video_paths)} videos -> {output_path}")
                    return output_path
                else:
                    logger.warning("Concatenación directa generó archivo vacío, intentando con transcoding")
                    transcode = True
        
        except Exception as e:
            logger.warning(f"Error en concatenación directa: {str(e)}. Intentando con transcoding.")
            transcode = True
        
        finally:
            # Limpiar archivo temporal de lista
            if os.path.exists(list_file):
                os.remove(list_file)
    
    # Método 2: Transcoding (más lento pero más compatible)
    if transcode:
        # Crear filtro de concatenación
        filter_complex = ""
        for i in range(len(video_paths)):
            filter_complex += f"[{i}:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}];"
        
        for i in range(len(video_paths)):
            filter_complex += f"[v{i}][{i}:a]"
        
        filter_complex += f"concat=n={len(video_paths)}:v=1:a=1[outv][outa]"
        
        # Construir comando complejo
        cmd = ['ffmpeg', '-y']
        
        # Añadir todas las entradas
        for path in video_paths:
            cmd.extend(['-i', path])
        
        # Añadir filtro complejo
        cmd.extend([
            '-filter_complex', filter_complex,
            '-map', '[outv]',
            '-map', '[outa]',
            '-c:v', 'libx264',
            '-crf', '23',
            '-preset', 'medium',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-threads', str(get_optimal_thread_count()),
            output_path
        ])
        
        logger.debug(f"Ejecutando comando de concatenación con transcoding: {' '.join(cmd)}")
        
        if timeout is None:
            # Para transcoding, usar un timeout más largo basado en la duración total
            timeout = DEFAULT_TIMEOUT * len(video_paths)
        
        try:
            start_time = time.time()
            
            # Ejecutar ffmpeg
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True
            )
            
            # Esperar a que termine con timeout
            stdout, stderr = process.communicate(timeout=timeout)
            
            if process.returncode != 0:
                logger.error(f"Error en concatenación con transcoding: {stderr}")
                raise RuntimeError(f"Error en concatenación de videos: {stderr}")
            
            # Verificar que el archivo se generó correctamente
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("La concatenación falló: archivo de salida vacío o inexistente")
            
            processing_time = time.time() - start_time
            logger.info(f"Concatenación con transcoding exitosa: {len(video_paths)} videos -> {output_path} en {processing_time:.2f} segundos")
            
            return output_path
        
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout en concatenación después de {timeout} segundos")
            # Intentar terminar el proceso
            if 'process' in locals():
                process.kill()
            raise TimeoutError(f"La concatenación excedió el tiempo límite de {timeout} segundos")
        
        except Exception as e:
            logger.error(f"Error en concatenación con transcoding: {str(e)}")
            raise RuntimeError(f"Error en concatenación de videos: {str(e)}")

def add_subtitles(video_path: str, subtitles_path: str, output_path: Optional[str] = None, 
                  font: Optional[str] = None, fontsize: Optional[int] = None, 
                  fontcolor: str = 'white', background: bool = True, 
                  position: str = 'bottom', timeout: Optional[int] = None) -> str:
    """
    Añade subtítulos a un video
    
    Args:
        video_path (str): Ruta al archivo de video
        subtitles_path (str): Ruta al archivo de subtítulos (srt, vtt, etc.)
        output_path (str, optional): Ruta para el video con subtítulos
        font (str, optional): Fuente para los subtítulos
        fontsize (int, optional): Tamaño de fuente para los subtítulos
        fontcolor (str): Color de la fuente
        background (bool): Si debe tener fondo oscuro
        position (str): Posición del texto (bottom, top)
        timeout (int, optional): Timeout en segundos
    
    Returns:
        str: Ruta al video con subtítulos
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Archivo de video no encontrado: {video_path}")
    
    if not os.path.exists(subtitles_path):
        raise FileNotFoundError(f"Archivo de subtítulos no encontrado: {subtitles_path}")
    
    # Generar ruta de salida si no se proporciona
    if output_path is None:
        ext = get_file_extension(video_path)
        output_path = generate_temp_filename(suffix=ext)
    
    # Determinar tipo de subtítulos
    subtitle_ext = get_file_extension(subtitles_path).lower()
    
    # Construir comando base
    cmd = ['ffmpeg', '-y', '-i', video_path]
    
    # Determinar el método a usar según el tipo de subtítulos
    if subtitle_ext in ['.srt', '.vtt']:
        # Opción 1: Subtítulos como filtro
        subtitle_options = []
        
        if font:
            subtitle_options.append(f"fontname={font}")
        if fontsize:
            subtitle_options.append(f"fontsize={fontsize}")
        
        subtitle_options.append(f"fontcolor={fontcolor}")
        
        if background:
            # Añadir fondo oscuro semitransparente
            subtitle_options.append("force_style='BackColour=&H80000000,Outline=0,BorderStyle=4'")
        
        # Posicionamiento
        if position == "top":
            subtitle_options.append("marginv=20")
        else:  # bottom (default)
            subtitle_options.append("marginv=30")
        
        # Escapar la ruta del archivo
        subtitles_path_escaped = subtitles_path.replace(':', '\\:').replace('\\', '\\\\')
        
        # Construir filtro
        filter_expr = f"subtitles={subtitles_path_escaped}"
        if subtitle_options:
            filter_expr += ":" + ":".join(subtitle_options)
        
        cmd.extend(['-vf', filter_expr])
        cmd.extend(['-c:a', 'copy'])
    else:
        # Opción 2: Subtítulos como stream
        cmd.extend(['-i', subtitles_path])
        cmd.extend(['-c:v', 'copy', '-c:a', 'copy', '-c:s', 'mov_text'])
    
    # Añadir número de hilos
    cmd.extend(['-threads', str(get_optimal_thread_count())])
    
    # Añadir salida
    cmd.append(output_path)
    
    logger.debug(f"Ejecutando comando para añadir subtítulos: {' '.join(cmd)}")
    
    # Configurar timeout
    if timeout is None:
        timeout = DEFAULT_TIMEOUT
    
    try:
        start_time = time.time()
        
        # Ejecutar ffmpeg
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        
        # Esperar a que termine con timeout
        stdout, stderr = process.communicate(timeout=timeout)
        
        if process.returncode != 0:
            logger.error(f"Error añadiendo subtítulos: {stderr}")
            raise RuntimeError(f"Error añadiendo subtítulos: {stderr}")
        
        # Verificar que el archivo se generó correctamente
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("Error al añadir subtítulos: archivo de salida vacío o inexistente")
        
        processing_time = time.time() - start_time
        logger.info(f"Subtítulos añadidos exitosamente: {video_path} + {subtitles_path} -> {output_path} en {processing_time:.2f} segundos")
        
        return output_path
    
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout añadiendo subtítulos después de {timeout} segundos")
        # Intentar terminar el proceso
        if 'process' in locals():
            process.kill()
        raise TimeoutError(f"El proceso de añadir subtítulos excedió el tiempo límite de {timeout} segundos")
    
    except Exception as e:
        logger.error(f"Error añadiendo subtítulos: {str(e)}")
        raise RuntimeError(f"Error añadiendo subtítulos: {str(e)}")

def image_to_video(image_path: str, duration: float, output_path: Optional[str] = None, 
                   width: Optional[int] = None, height: Optional[int] = None, 
                   transition: Optional[str] = None, timeout: Optional[int] = None) -> str:
    """
    Convierte una imagen a video con duración especificada
    
    Args:
        image_path (str): Ruta a la imagen
        duration (float): Duración del video en segundos
        output_path (str, optional): Ruta para el video resultante
        width (int, optional): Ancho del video
        height (int, optional): Alto del video
        transition (str, optional): Efecto de transición (fade, zoom, etc.)
        timeout (int, optional): Timeout en segundos
    
    Returns:
        str: Ruta al video creado
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Imagen no encontrada: {image_path}")
    
    # Generar ruta de salida si no se proporciona
    if output_path is None:
        output_path = generate_temp_filename(suffix=".mp4")
    
    # Determinar número óptimo de hilos
    threads = get_optimal_thread_count()
    
    # Construir comando
    cmd = ['ffmpeg', '-y', '-loop', '1', '-i', image_path]
    
    # Configurar filtros para efectos
    filter_complex = ""
    
    # Filtro de escala si se especifica resolución
    if width and height:
        filter_complex += f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
    
    # Añadir efecto de transición si se especifica
    if transition:
        if transition == 'fade':
            fade_duration = min(duration / 4, 2.0)  # Limitar fade a 2 segundos o 1/4 de duración
            if filter_complex:
                filter_complex += ","
            filter_complex += f"fade=t=in:st=0:d={fade_duration},fade=t=out:st={duration-fade_duration}:d={fade_duration}"
        elif transition == 'zoom':
            if filter_complex:
                filter_complex += ","
            filter_complex += f"zoompan=z='min(zoom+0.0015,1.5)':d={int(duration*25)}"
        elif transition == 'pan':
            if filter_complex:
                filter_complex += ","
            filter_complex += f"zoompan=z=1.1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={int(duration*25)}"
    
    # Añadir filtro si se definió
    if filter_complex:
        cmd.extend(['-vf', filter_complex])
    
    # Configurar duración y otros parámetros
    cmd.extend([
        '-t', str(duration),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-threads', str(threads),
        output_path
    ])
    
    logger.debug(f"Ejecutando comando para convertir imagen a video: {' '.join(cmd)}")
    
    # Configurar timeout
    if timeout is None:
        timeout = DEFAULT_TIMEOUT
    
    try:
        start_time = time.time()
        
        # Ejecutar ffmpeg
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        
        # Esperar a que termine con timeout
        stdout, stderr = process.communicate(timeout=timeout)
        
        if process.returncode != 0:
            logger.error(f"Error convirtiendo imagen a video: {stderr}")
            raise RuntimeError(f"Error convirtiendo imagen a video: {stderr}")
        
        # Verificar que el archivo se generó correctamente
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("Error al convertir imagen a video: archivo de salida vacío o inexistente")
        
        processing_time = time.time() - start_time
        logger.info(f"Imagen convertida a video exitosamente: {image_path} -> {output_path} ({duration}s) en {processing_time:.2f} segundos")
        
        return output_path
    
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout convirtiendo imagen a video después de {timeout} segundos")
        # Intentar terminar el proceso
        if 'process' in locals():
            process.kill()
        raise TimeoutError(f"La conversión de imagen a video excedió el tiempo límite de {timeout} segundos")
    
    except Exception as e:
        logger.error(f"Error convirtiendo imagen a video: {str(e)}")
        raise RuntimeError(f"Error convirtiendo imagen a video: {str(e)}")

def trim_video(video_path: str, start_time: float, duration: Optional[float] = None, 
               end_time: Optional[float] = None, output_path: Optional[str] = None, 
               accurate: bool = True, timeout: Optional[int] = None) -> str:
    """
    Recorta un video según tiempo de inicio y duración o tiempo final
    
    Args:
        video_path (str): Ruta al video
        start_time (float): Tiempo de inicio en segundos
        duration (float, optional): Duración del recorte en segundos
        end_time (float, optional): Tiempo final en segundos (alternativa a duration)
        output_path (str, optional): Ruta para el video recortado
        accurate (bool): Si se debe usar recorte preciso (más lento pero exacto)
        timeout (int, optional): Timeout en segundos
    
    Returns:
        str: Ruta al video recortado
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Archivo de video no encontrado: {video_path}")
    
    # Verificar parámetros de tiempo
    if duration is None and end_time is None:
        raise ValueError("Se debe proporcionar duration o end_time")
    
    # Calcular duración si se proporcionó end_time
    if duration is None:
        if end_time is not None:
            duration = end_time - start_time
        else:
            # Intentar obtener duración total del video
            try:
                video_info = get_video_info(video_path)
                duration = video_info.get('duration', 0) - start_time
            except Exception as e:
                logger.error(f"Error obteniendo duración del video: {str(e)}")
                raise ValueError("No se pudo determinar la duración. Proporciona duration o end_time.")
    
    # Generar ruta de salida si no se proporciona
    if output_path is None:
        ext = get_file_extension(video_path)
        output_path = generate_temp_filename(suffix=ext)
    
    # Determinar número óptimo de hilos
    threads = get_optimal_thread_count()
    
    # Construir comando
    if accurate:
        # Método 1: Recorte preciso (más lento pero exacto)
        cmd = [
            'ffmpeg',
            '-y',
            '-i', video_path,
            '-ss', str(start_time),
            '-t', str(duration),
            '-c:v', 'libx264',  # Recodificar para precisión
            '-c:a', 'aac',      # Recodificar audio
            '-threads', str(threads),
            output_path
        ]
    else:
        # Método 2: Recorte rápido (menos preciso pero más rápido)
        cmd = [
            'ffmpeg',
            '-y',
            '-ss', str(start_time),  # Colocar -ss antes de -i para búsqueda rápida
            '-i', video_path,
            '-t', str(duration),
            '-c:v', 'copy',
            '-c:a', 'copy',
            '-avoid_negative_ts', '1',
            '-threads', str(threads),
            output_path
        ]
    
    logger.debug(f"Ejecutando comando para recortar video: {' '.join(cmd)}")
    
    # Configurar timeout
    if timeout is None:
        timeout = DEFAULT_TIMEOUT
    
    try:
        start_process_time = time.time()
        
        # Ejecutar ffmpeg
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        
        # Esperar a que termine con timeout
        stdout, stderr = process.communicate(timeout=timeout)
        
        if process.returncode != 0:
            logger.error(f"Error recortando video: {stderr}")
            raise RuntimeError(f"Error recortando video: {stderr}")
        
        # Verificar que el archivo se generó correctamente
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("Error al recortar video: archivo de salida vacío o inexistente")
        
        processing_time = time.time() - start_process_time
        logger.info(f"Video recortado exitosamente: {video_path} -> {output_path} (inicio={start_time}, duración={duration}) en {processing_time:.2f} segundos")
        
        return output_path
    
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout recortando video después de {timeout} segundos")
        # Intentar terminar el proceso
        if 'process' in locals():
            process.kill()
        raise TimeoutError(f"El recorte de video excedió el tiempo límite de {timeout} segundos")
    
    except Exception as e:
        logger.error(f"Error recortando video: {str(e)}")
        raise RuntimeError(f"Error recortando video: {str(e)}")

def add_watermark(video_path: str, watermark_path: str, position: str, scale: float = 0.1, 
                  opacity: float = 1.0, output_path: Optional[str] = None, 
                  timeout: Optional[int] = None) -> str:
    """
    Añade una marca de agua (imagen) a un video
    
    Args:
        video_path (str): Ruta al video
        watermark_path (str): Ruta a la imagen de marca de agua
        position (str): Posición (top, bottom, left, right, top_left, top_right, bottom_left, bottom_right, center)
        scale (float, optional): Escala relativa a la resolución del video (0.0-1.0)
        opacity (float, optional): Opacidad de la marca de agua (0.0-1.0)
        output_path (str, optional): Ruta para el video resultante
        timeout (int, optional): Timeout en segundos
    
    Returns:
        str: Ruta al video con marca de agua
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Archivo de video no encontrado: {video_path}")
    
    if not os.path.exists(watermark_path):
        raise FileNotFoundError(f"Imagen de marca de agua no encontrada: {watermark_path}")
    
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
        valid_positions = ', '.join(positions.keys())
        raise ValueError(f"Posición inválida: {position}. Posiciones válidas: {valid_positions}")
    
    # Determinar número óptimo de hilos
    threads = get_optimal_thread_count()
    
    # Construir filtro complejo con opacidad
    filter_complex = f"[0:v][1:v]scale2ref=oh*mdar:ih*{scale}[main][overlay];"
    
    # Añadir opacidad si es menor a 1.0
    if opacity < 1.0:
        filter_complex += f"[overlay]format=rgba,colorchannelmixer=a={opacity}[overlay_opacity];"
        filter_complex += f"[main][overlay_opacity]overlay={positions[position]}:format=auto,format=yuv420p"
    else:
        filter_complex += f"[main][overlay]overlay={positions[position]}:format=auto,format=yuv420p"
    
    # Construir comando
    cmd = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        '-i', watermark_path,
        '-filter_complex', filter_complex,
        '-c:a', 'copy',
        '-threads', str(threads),
        output_path
    ]
    
    logger.debug(f"Ejecutando comando para añadir marca de agua: {' '.join(cmd)}")
    
    # Configurar timeout
    if timeout is None:
        timeout = DEFAULT_TIMEOUT
    
    try:
        start_time = time.time()
        
        # Ejecutar ffmpeg
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        
        # Esperar a que termine con timeout
        stdout, stderr = process.communicate(timeout=timeout)
        
        if process.returncode != 0:
            logger.error(f"Error añadiendo marca de agua: {stderr}")
            raise RuntimeError(f"Error añadiendo marca de agua: {stderr}")
        
        # Verificar que el archivo se generó correctamente
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("Error al añadir marca de agua: archivo de salida vacío o inexistente")
        
        processing_time = time.time() - start_time
        logger.info(f"Marca de agua añadida exitosamente: {video_path} + {watermark_path} -> {output_path} en {processing_time:.2f} segundos")
        
        return output_path
    
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout añadiendo marca de agua después de {timeout} segundos")
        # Intentar terminar el proceso
        if 'process' in locals():
            process.kill()
        raise TimeoutError(f"El proceso de añadir marca de agua excedió el tiempo límite de {timeout} segundos")
    
    except Exception as e:
        logger.error(f"Error añadiendo marca de agua: {str(e)}")
        raise RuntimeError(f"Error añadiendo marca de agua: {str(e)}")

def overlay_video(background_path: str, overlay_path: str, position: str, 
                  start_time: float = 0, scale: float = 0.3, 
                  output_path: Optional[str] = None, timeout: Optional[int] = None) -> str:
    """
    Superpone un video sobre otro
    
    Args:
        background_path (str): Ruta al video de fondo
        overlay_path (str): Ruta al video a superponer
        position (str): Posición (top, bottom, left, right, top_left, top_right, bottom_left, bottom_right, center)
        start_time (float): Tiempo de inicio de la superposición en segundos
        scale (float): Escala del video superpuesto relativa al fondo
        output_path (str, optional): Ruta para el video resultante
        timeout (int, optional): Timeout en segundos
    
    Returns:
        str: Ruta al video compuesto
    """
    if not os.path.exists(background_path):
        raise FileNotFoundError(f"Video de fondo no encontrado: {background_path}")
    
    if not os.path.exists(overlay_path):
        raise FileNotFoundError(f"Video a superponer no encontrado: {overlay_path}")
    
    # Generar ruta de salida si no se proporciona
    if output_path is None:
        ext = get_file_extension(background_path)
        output_path = generate_temp_filename(suffix=ext)
    
    # Mapeo de posiciones a expresiones filter_complex
    positions = {
        "top": f"x=(W-w)/2:y=H*0.05",
        "bottom": f"x=(W-w)/2:y=H*0.95-h",
        "left": f"x=W*0.05:y=(H-h)/2",
        "right": f"x=W*0.95-w:y=(H-h)/2",
        "top_left": f"x=W*0.05:y=H*0.05",
        "top_right": f"x=W*0.95-w:y=H*0.05",
        "bottom_left": f"x=W*0.05:y=H*0.95-h",
        "bottom_right": f"x=W*0.95-w:y=H*0.95-h",
        "center": f"x=(W-w)/2:y=(H-h)/2"
    }
    
    # Verificar que la posición sea válida
    if position not in positions:
        valid_positions = ', '.join(positions.keys())
        raise ValueError(f"Posición inválida: {position}. Posiciones válidas: {valid_positions}")
    
    # Determinar número óptimo de hilos
    threads = get_optimal_thread_count()
    
    # Construir filtro complejo
    filter_complex = f"[1:v]scale=iw*{scale}:-1[overlay];"
    
    # Si hay un tiempo de inicio, añadir retraso
    if start_time > 0:
        filter_complex += f"[0:v][overlay]overlay={positions[position]}:enable='gte(t,{start_time})',format=yuv420p"
    else:
        filter_complex += f"[0:v][overlay]overlay={positions[position]},format=yuv420p"
    
    # Construir comando
    cmd = [
        'ffmpeg',
        '-y',
        '-i', background_path,
        '-i', overlay_path,
        '-filter_complex', filter_complex,
        '-c:a', 'copy',  # Mantener audio original
        '-threads', str(threads),
        output_path
    ]
    
    logger.debug(f"Ejecutando comando para superponer video: {' '.join(cmd)}")
    
    # Configurar timeout
    if timeout is None:
        # Estimar timeout basado en duración de los videos
        try:
            bg_info = get_video_info(background_path)
            bg_duration = bg_info.get('duration', 0)
            timeout = int(bg_duration * 1.5) + DEFAULT_TIMEOUT
        except Exception:
            timeout = DEFAULT_TIMEOUT
    
    try:
        start_time_process = time.time()
        
        # Ejecutar ffmpeg
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        
        # Esperar a que termine con timeout
        stdout, stderr = process.communicate(timeout=timeout)
        
        if process.returncode != 0:
            logger.error(f"Error superponiendo video: {stderr}")
            raise RuntimeError(f"Error superponiendo video: {stderr}")
        
        # Verificar que el archivo se generó correctamente
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("Error al superponer video: archivo de salida vacío o inexistente")
        
        processing_time = time.time() - start_time_process
        logger.info(f"Video superpuesto exitosamente: {background_path} + {overlay_path} -> {output_path} en {processing_time:.2f} segundos")
        
        return output_path
    
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout superponiendo video después de {timeout} segundos")
        # Intentar terminar el proceso
        if 'process' in locals():
            process.kill()
        raise TimeoutError(f"El proceso de superposición de video excedió el tiempo límite de {timeout} segundos")
    
    except Exception as e:
        logger.error(f"Error superponiendo video: {str(e)}")
        raise RuntimeError(f"Error superponiendo video: {str(e)}")

def add_text_overlay(video_path: str, text: str, position: str, font: str = 'Arial', 
                     font_size: int = 24, font_color: str = 'white', 
                     background: bool = True, start_time: float = 0, 
                     duration: Optional[float] = None, output_path: Optional[str] = None, 
                     timeout: Optional[int] = None) -> str:
    """
    Añade texto superpuesto a un video
    
    Args:
        video_path (str): Ruta al video
        text (str): Texto a superponer
        position (str): Posición (top, bottom, center)
        font (str): Nombre de la fuente
        font_size (int): Tamaño de la fuente
        font_color (str): Color de la fuente
        background (bool): Si debe tener fondo oscuro
        start_time (float): Tiempo de inicio del texto en segundos
        duration (float, optional): Duración del texto en pantalla
        output_path (str, optional): Ruta para el video resultante
        timeout (int, optional): Timeout en segundos
    
    Returns:
        str: Ruta al video con texto
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Archivo de video no encontrado: {video_path}")
    
    # Generar ruta de salida si no se proporciona
    if output_path is None:
        ext = get_file_extension(video_path)
        output_path = generate_temp_filename(suffix=ext)
    
    # Determinar posición vertical
    if position == 'top':
        y_pos = f"10"
    elif position == 'bottom':
        y_pos = f"h-th-10"
    else:  # center
        y_pos = f"(h-th)/2"
    
    # Construir texto con filtro drawtext
    text_escaped = text.replace("'", "\\'").replace(':', '\\:')
    
    filter_text = f"drawtext=text='{text_escaped}':fontfile={font}:fontsize={font_size}:fontcolor={font_color}:x=(w-tw)/2:y={y_pos}"
    
    # Añadir condición de tiempo si se especifica duración
    if duration is not None:
        end_time = start_time + duration
        filter_text += f":enable='between(t,{start_time},{end_time})'"
    elif start_time > 0:
        filter_text += f":enable='gte(t,{start_time})'"
    
    # Añadir fondo oscuro si se requiere
    if background:
        filter_text += ":box=1:boxcolor=black@0.5:boxborderw=5"
    
    # Determinar número óptimo de hilos
    threads = get_optimal_thread_count()
    
    # Construir comando
    cmd = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        '-vf', filter_text,
        '-c:a', 'copy',
        '-threads', str(threads),
        output_path
    ]
    
    logger.debug(f"Ejecutando comando para añadir texto: {' '.join(cmd)}")
    
    # Configurar timeout
    if timeout is None:
        timeout = DEFAULT_TIMEOUT
    
    try:
        start_time_process = time.time()
        
        # Ejecutar ffmpeg
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        
        # Esperar a que termine con timeout
        stdout, stderr = process.communicate(timeout=timeout)
        
        if process.returncode != 0:
            logger.error(f"Error añadiendo texto al video: {stderr}")
            raise RuntimeError(f"Error añadiendo texto al video: {stderr}")
        
        # Verificar que el archivo se generó correctamente
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("Error al añadir texto: archivo de salida vacío o inexistente")
        
        processing_time = time.time() - start_time_process
        logger.info(f"Texto añadido exitosamente: {video_path} -> {output_path} en {processing_time:.2f} segundos")
        
        return output_path
    
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout añadiendo texto después de {timeout} segundos")
        # Intentar terminar el proceso
        if 'process' in locals():
            process.kill()
        raise TimeoutError(f"El proceso de añadir texto excedió el tiempo límite de {timeout} segundos")
    
    except Exception as e:
        logger.error(f"Error añadiendo texto al video: {str(e)}")
        raise RuntimeError(f"Error añadiendo texto al video: {str(e)}")

def execute_ffmpeg_command(command: List[str], timeout: Optional[int] = None) -> str:
    """
    Ejecuta un comando FFmpeg personalizado con validación de seguridad
    
    Args:
        command (list): Lista de argumentos para el comando FFmpeg
        timeout (int, optional): Timeout en segundos
    
    Returns:
        str: Salida de stderr de FFmpeg
    
    Raises:
        ValueError: Si el comando contiene parámetros peligrosos
        RuntimeError: Si el comando falla
        TimeoutError: Si el comando excede el timeout
    """
    # Validación de seguridad
    validate_ffmpeg_parameters(command)
    
    # Asegurar que el primer argumento sea 'ffmpeg'
    if not command or command[0] != 'ffmpeg':
        command.insert(0, 'ffmpeg')
    
    # Asegurar que existe -y para sobreescribir archivos
    if '-y' not in command:
        command.insert(1, '-y')
    
    logger.debug(f"Ejecutando comando FFmpeg personalizado: {' '.join(command)}")
    
    # Configurar timeout
    if timeout is None:
        timeout = DEFAULT_TIMEOUT
    
    try:
        start_time = time.time()
        
        # Ejecutar ffmpeg
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        
        # Esperar a que termine con timeout
        stdout, stderr = process.communicate(timeout=timeout)
        
        if process.returncode != 0:
            logger.error(f"Error en comando FFmpeg personalizado: {stderr}")
            raise RuntimeError(f"Error en comando FFmpeg: {stderr}")
        
        processing_time = time.time() - start_time
        logger.info(f"Comando FFmpeg personalizado ejecutado exitosamente en {processing_time:.2f} segundos")
        
        return stderr  # FFmpeg escribe la información útil en stderr
    
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout en comando FFmpeg personalizado después de {timeout} segundos")
        # Intentar terminar el proceso
        if 'process' in locals():
            process.kill()
        raise TimeoutError(f"El comando FFmpeg excedió el tiempo límite de {timeout} segundos")
    
    except Exception as e:
        logger.error(f"Error en comando FFmpeg personalizado: {str(e)}")
        raise RuntimeError(f"Error en comando FFmpeg: {str(e)}")
