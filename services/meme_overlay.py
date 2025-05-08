import os
import subprocess
import logging
import tempfile
import json
import math
import hashlib
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np

from services.file_management import download_file, generate_temp_filename, verify_file_integrity, is_image_file
import config

logger = logging.getLogger(__name__)

# Constantes para configuración
DEFAULT_SCALE = 0.3
DEFAULT_POSITION = "bottom"
DEFAULT_OPACITY = 1.0
DEFAULT_DURATION = 0.0  # 0 = toda la duración del video
DEFAULT_BORDER_WIDTH = 0  # 0 = sin borde
DEFAULT_BORDER_COLOR = "white"
DEFAULT_BORDER_RADIUS = 0  # 0 = sin redondear
DEFAULT_ROTATION = 0  # 0 = sin rotación
DEFAULT_EFFECT = "none"
DEFAULT_TRANSITION = "none"

AVAILABLE_POSITIONS = ["top", "bottom", "left", "right", "top_left", "top_right", "bottom_left", "bottom_right", "center"]
AVAILABLE_EFFECTS = ["none", "grayscale", "sepia", "blur", "sharpen", "edge", "emboss", "pixelate", "negative", "posterize"]
AVAILABLE_TRANSITIONS = ["none", "fade_in", "fade_out", "fade_in_out", "slide_in", "slide_out", "zoom_in", "zoom_out", "rotate_in", "pulse"]

FFMPEG_TIMEOUT = getattr(config, 'MAX_PROCESSING_TIME', 600)  # 10 minutos por defecto
USE_GPU = getattr(config, 'USE_GPU_ACCELERATION', False)  # Usar aceleración GPU si está disponible
CACHE_DIR = os.path.join(config.TEMP_DIR, 'meme_cache')  # Caché para memes procesados

# Crear directorio de caché si no existe
os.makedirs(CACHE_DIR, exist_ok=True)

class MemeOverlayError(Exception):
    """Excepción para errores específicos de meme overlay"""
    pass

def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    Obtiene información detallada sobre un archivo de video
    
    Args:
        video_path: Ruta al archivo de video
        
    Returns:
        Dict: Información del video (duración, resolución, etc.)
        
    Raises:
        MemeOverlayError: Si hay un error obteniendo la información
    """
    try:
        cmd = [
            'ffprobe', 
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,duration,r_frame_rate:format=duration',
            '-of', 'json',
            video_path
        ]
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True,
            timeout=30  # 30 segundos máximo
        )
        
        info = json.loads(result.stdout)
        
        # Extraer información relevante
        video_info = {}
        
        # Dimensiones
        if 'streams' in info and len(info['streams']) > 0:
            stream = info['streams'][0]
            video_info['width'] = int(stream.get('width', 0))
            video_info['height'] = int(stream.get('height', 0))
            
            # Framerate
            if 'r_frame_rate' in stream:
                rate = stream['r_frame_rate'].split('/')
                if len(rate) == 2 and int(rate[1]) != 0:
                    video_info['fps'] = int(rate[0]) / int(rate[1])
                else:
                    video_info['fps'] = float(rate[0])
        
        # Duración
        if 'format' in info and 'duration' in info['format']:
            video_info['duration'] = float(info['format']['duration'])
        elif 'streams' in info and len(info['streams']) > 0 and 'duration' in info['streams'][0]:
            video_info['duration'] = float(info['streams'][0]['duration'])
        else:
            raise MemeOverlayError("No se pudo determinar la duración del video")
        
        # Verificar información mínima necesaria
        required_keys = ['width', 'height', 'duration']
        for key in required_keys:
            if key not in video_info or not video_info[key]:
                raise MemeOverlayError(f"No se pudo obtener {key} del video")
        
        return video_info
    
    except subprocess.SubprocessError as e:
        raise MemeOverlayError(f"Error al analizar el video: {str(e)}")
    except json.JSONDecodeError:
        raise MemeOverlayError("Error al procesar la información del video")
    except Exception as e:
        raise MemeOverlayError(f"Error obteniendo información del video: {str(e)}")

def preprocess_meme_image(
    image_path: str,
    output_path: Optional[str] = None,
    scale: float = 1.0,
    opacity: float = 1.0,
    border_width: int = 0,
    border_color: str = "white",
    border_radius: int = 0,
    rotation: float = 0.0,
    effect: str = "none"
) -> str:
    """
    Preprocesa una imagen de meme para aplicar efectos y ajustes
    
    Args:
        image_path: Ruta de la imagen original
        output_path: Ruta de salida (opcional)
        scale: Factor de escala para la imagen
        opacity: Opacidad (0.0-1.0)
        border_width: Ancho del borde en píxeles
        border_color: Color del borde
        border_radius: Radio de las esquinas redondeadas en píxeles
        rotation: Ángulo de rotación en grados
        effect: Efecto visual a aplicar
        
    Returns:
        str: Ruta a la imagen procesada
    
    Raises:
        MemeOverlayError: Si hay un error procesando la imagen
    """
    try:
        # Verificar que la imagen existe
        if not os.path.exists(image_path):
            raise MemeOverlayError(f"Imagen no encontrada: {image_path}")
            
        # Asegurar que es un archivo de imagen
        if not is_image_file(image_path):
            raise MemeOverlayError(f"El archivo no es una imagen válida: {image_path}")
        
        # Generar hash único para la combinación de parámetros (para caché)
        params_str = f"{image_path}_{scale}_{opacity}_{border_width}_{border_color}_{border_radius}_{rotation}_{effect}"
        params_hash = hashlib.md5(params_str.encode()).hexdigest()
        
        # Verificar si ya existe en caché
        cache_path = os.path.join(CACHE_DIR, f"{params_hash}.png")
        if os.path.exists(cache_path):
            logger.debug(f"Usando imagen preprocesada desde caché: {cache_path}")
            
            # Si se especificó una ruta de salida diferente, copiar desde caché
            if output_path and output_path != cache_path:
                import shutil
                shutil.copy2(cache_path, output_path)
                return output_path
            
            return cache_path
        
        # Abrir la imagen con PIL
        img = Image.open(image_path)
        
        # Convertir a RGBA para manejar transparencia
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Aplicar efectos según lo solicitado
        if effect != "none":
            img = apply_effect(img, effect)
        
        # Aplicar escala si es diferente de 1.0
        if scale != 1.0:
            new_width = int(img.width * scale)
            new_height = int(img.height * scale)
            img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Aplicar rotación si es diferente de 0
        if rotation != 0:
            img = img.rotate(rotation, expand=True, resample=Image.BICUBIC)
        
        # Aplicar borde si width > 0
        if border_width > 0:
            # Crear una nueva imagen con el tamaño aumentado por el borde
            border_img = Image.new(
                'RGBA', 
                (img.width + 2 * border_width, img.height + 2 * border_width), 
                border_color
            )
            # Pegar la imagen original en el centro
            border_img.paste(img, (border_width, border_width), img)
            img = border_img
        
        # Aplicar esquinas redondeadas si radius > 0
        if border_radius > 0:
            img = round_corners(img, border_radius)
        
        # Aplicar opacidad si es menor que 1.0
        if opacity < 1.0:
            # Crear una copia de la imagen con canal alfa modificado
            data = np.array(img)
            # Multiplicar canal alfa por el factor de opacidad
            data[:, :, 3] = data[:, :, 3] * opacity
            img = Image.fromarray(data)
        
        # Determinar ruta de salida
        if output_path is None:
            output_path = cache_path
        
        # Guardar imagen procesada
        img.save(output_path, "PNG")
        
        # Si el resultado no es el caché, también guardar en caché para futuro uso
        if output_path != cache_path:
            img.save(cache_path, "PNG")
        
        logger.debug(f"Imagen de meme preprocesada guardada en: {output_path}")
        return output_path
        
    except Exception as e:
        raise MemeOverlayError(f"Error procesando imagen de meme: {str(e)}")

def apply_effect(img: Image.Image, effect: str) -> Image.Image:
    """
    Aplica un efecto visual a una imagen
    
    Args:
        img: Imagen PIL
        effect: Nombre del efecto a aplicar
        
    Returns:
        Image: Imagen con efecto aplicado
    """
    if effect == "grayscale":
        # Convertir a escala de grises pero mantener canal alfa
        gray = img.convert('L')
        alpha = img.getchannel('A')
        result = Image.merge('LA', (gray, alpha))
        return result.convert('RGBA')
    
    elif effect == "sepia":
        # Efecto sepia (tono marrón antiguo)
        sepia = img.convert('RGB')
        sepia = np.array(sepia)
        
        # Matriz de transformación sepia
        sepia_matrix = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        
        # Aplicar transformación
        red = np.array(sepia[:,:,0], dtype=np.uint16)
        green = np.array(sepia[:,:,1], dtype=np.uint16)
        blue = np.array(sepia[:,:,2], dtype=np.uint16)
        
        new_red = np.clip(red * sepia_matrix[0,0] + green * sepia_matrix[0,1] + blue * sepia_matrix[0,2], 0, 255).astype(np.uint8)
        new_green = np.clip(red * sepia_matrix[1,0] + green * sepia_matrix[1,1] + blue * sepia_matrix[1,2], 0, 255).astype(np.uint8)
        new_blue = np.clip(red * sepia_matrix[2,0] + green * sepia_matrix[2,1] + blue * sepia_matrix[2,2], 0, 255).astype(np.uint8)
        
        sepia = np.stack([new_red, new_green, new_blue], axis=2)
        
        # Restaurar canal alfa
        alpha = np.array(img.getchannel('A'))
        sepia_rgba = np.dstack((sepia, alpha))
        
        return Image.fromarray(sepia_rgba, 'RGBA')
    
    elif effect == "blur":
        # Efecto de desenfoque
        return img.filter(ImageFilter.GaussianBlur(radius=3))
    
    elif effect == "sharpen":
        # Efecto de enfoque
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(2.0)
    
    elif effect == "edge":
        # Detección de bordes
        edges = img.filter(ImageFilter.FIND_EDGES)
        alpha = img.getchannel('A')
        return Image.merge('RGBA', (edges.getchannel('R'), edges.getchannel('G'), edges.getchannel('B'), alpha))
    
    elif effect == "emboss":
        # Efecto de relieve
        emboss = img.filter(ImageFilter.EMBOSS)
        alpha = img.getchannel('A')
        return Image.merge('RGBA', (emboss.getchannel('R'), emboss.getchannel('G'), emboss.getchannel('B'), alpha))
    
    elif effect == "pixelate":
        # Efecto de pixelado
        downscale_factor = 10
        small = img.resize(
            (img.width // downscale_factor, img.height // downscale_factor),
            resample=Image.NEAREST
        )
        return small.resize(img.size, Image.NEAREST)
    
    elif effect == "negative":
        # Efecto negativo (invertir colores)
        neg = ImageOps.invert(img.convert('RGB'))
        alpha = img.getchannel('A')
        return Image.merge('RGBA', (neg.getchannel('R'), neg.getchannel('G'), neg.getchannel('B'), alpha))
    
    elif effect == "posterize":
        # Efecto de posterizado (reduce número de colores)
        poster = ImageOps.posterize(img.convert('RGB'), 3)
        alpha = img.getchannel('A')
        return Image.merge('RGBA', (poster.getchannel('R'), poster.getchannel('G'), poster.getchannel('B'), alpha))
    
    else:
        # Sin efecto o efecto desconocido
        return img

def round_corners(img: Image.Image, radius: int) -> Image.Image:
    """
    Aplica esquinas redondeadas a una imagen
    
    Args:
        img: Imagen PIL
        radius: Radio de las esquinas en píxeles
        
    Returns:
        Image: Imagen con esquinas redondeadas
    """
    # Crear una máscara circular
    mask = Image.new('L', img.size, 0)
    draw = ImageDraw.Draw(mask)
    
    # Dibujar un rectángulo con esquinas redondeadas
    draw.rounded_rectangle([(0, 0), (img.width, img.height)], radius=radius, fill=255)
    
    # Aplicar máscara
    result = img.copy()
    result.putalpha(mask)
    
    return result

def generate_overlay_filter(
    position: str,
    scale: float,
    x_offset: int = 0,
    y_offset: int = 0,
    start_time: float = 0.0,
    duration: float = 0.0,
    transition: str = "none",
    rotation: float = 0.0,
    width_ratio: float = 1.0,
    height_ratio: float = 1.0
) -> str:
    """
    Genera el filtro FFmpeg para overlay con efectos avanzados
    
    Args:
        position: Posición del meme
        scale: Escala relativa
        x_offset: Desplazamiento horizontal adicional
        y_offset: Desplazamiento vertical adicional
        start_time: Tiempo de inicio en segundos
        duration: Duración en segundos (0 = toda la duración)
        transition: Efecto de transición
        rotation: Rotación dinámica (para transiciones)
        width_ratio: Relación de ancho entrada/salida
        height_ratio: Relación de alto entrada/salida
        
    Returns:
        str: Expresión de filtro para FFmpeg
    """
    # Mapeo de posiciones a expresiones
    positions = {
        "top": f"x=(W-w)/2+{x_offset}:y=H*0.05+{y_offset}",
        "bottom": f"x=(W-w)/2+{x_offset}:y=H*0.95-h+{y_offset}",
        "left": f"x=W*0.05+{x_offset}:y=(H-h)/2+{y_offset}",
        "right": f"x=W*0.95-w+{x_offset}:y=(H-h)/2+{y_offset}",
        "top_left": f"x=W*0.05+{x_offset}:y=H*0.05+{y_offset}",
        "top_right": f"x=W*0.95-w+{x_offset}:y=H*0.05+{y_offset}",
        "bottom_left": f"x=W*0.05+{x_offset}:y=H*0.95-h+{y_offset}",
        "bottom_right": f"x=W*0.95-w+{x_offset}:y=H*0.95-h+{y_offset}",
        "center": f"x=(W-w)/2+{x_offset}:y=(H-h)/2+{y_offset}"
    }
    
    position_expr = positions.get(position, positions["bottom"])
    
    # Añadir condición de tiempo si se especifica
    if start_time > 0 or duration > 0:
        if duration > 0:
            end_time = start_time + duration
            enable_expr = f":enable='between(t,{start_time},{end_time})'"
        else:
            enable_expr = f":enable='gte(t,{start_time})'"
    else:
        enable_expr = ""
    
    # Añadir transición si se especifica
    if transition != "none":
        if transition == "fade_in":
            # Fade in durante los primeros N segundos
            fade_duration = min(1.0, duration * 0.3) if duration > 0 else 1.0
            opacity_expr = f":eval=frame:opacity='if(lt(t,{start_time+fade_duration}),(t-{start_time})/{fade_duration},1)'"
        
        elif transition == "fade_out":
            # Fade out durante los últimos N segundos
            fade_duration = min(1.0, duration * 0.3) if duration > 0 else 1.0
            end_time = start_time + duration if duration > 0 else 999999
            opacity_expr = f":eval=frame:opacity='if(gt(t,{end_time-fade_duration}),({end_time}-t)/{fade_duration},1)'"
        
        elif transition == "fade_in_out":
            # Fade in y fade out
            fade_duration = min(1.0, duration * 0.2) if duration > 0 else 1.0
            end_time = start_time + duration if duration > 0 else 999999
            opacity_expr = (f":eval=frame:opacity='if(lt(t,{start_time+fade_duration}),"
                           f"(t-{start_time})/{fade_duration},if(gt(t,{end_time-fade_duration}),"
                           f"({end_time}-t)/{fade_duration},1))'")
        
        elif transition == "slide_in":
            # Slide in desde fuera de la pantalla
            slide_duration = min(1.5, duration * 0.3) if duration > 0 else 1.5
            # Extraer coordenadas originales
            x_part = position_expr.split(':')[0]
            y_part = position_expr.split(':')[1].split(enable_expr)[0]
            
            # Modificar según dirección de deslizamiento basada en posición
            if position in ["top", "center", "bottom"]:
                # Deslizar desde la izquierda
                x_part = f"x='if(lt(t,{start_time+slide_duration}),W*(1-(t-{start_time})/{slide_duration})-w,{x_part[2:]})"
            elif position in ["left", "top_left", "bottom_left"]:
                # Deslizar desde la izquierda
                x_part = f"x='if(lt(t,{start_time+slide_duration}),-w+(W*0.05+{x_offset})*(t-{start_time})/{slide_duration},{x_part[2:]})"
            else:
                # Deslizar desde la derecha
                x_part = f"x='if(lt(t,{start_time+slide_duration}),W+(W*0.05-w-{x_offset})*(t-{start_time})/{slide_duration},{x_part[2:]})"
            
            position_expr = f"{x_part}:{y_part}"
            opacity_expr = ""
        
        elif transition == "slide_out":
            # Slide out fuera de la pantalla
            slide_duration = min(1.5, duration * 0.3) if duration > 0 else 1.5
            end_time = start_time + duration if duration > 0 else 999999
            # No implementado en esta versión simplificada
            opacity_expr = ""
        
        elif transition == "zoom_in":
            # Efecto de zoom in
            zoom_duration = min(1.5, duration * 0.3) if duration > 0 else 1.5
            # Extraer coordenadas originales
            parts = position_expr.split(':')
            x_part = parts[0].split('=')[1]
            y_part = parts[1].split('=')[1].split(enable_expr)[0]
            
            # Calcular el centro para el zoom
            center_x = f"({x_part}+w/2)"
            center_y = f"({y_part}+h/2)"
            
            # Zoom desde tamaño pequeño (10%) hasta tamaño normal
            new_x = f"({center_x}-w*if(lt(t,{start_time+zoom_duration}),0.1+(t-{start_time})/{zoom_duration}*0.9,1)/2)"
            new_y = f"({center_y}-h*if(lt(t,{start_time+zoom_duration}),0.1+(t-{start_time})/{zoom_duration}*0.9,1)/2)"
            
            position_expr = f"x={new_x}:y={new_y}:eval=frame"
            opacity_expr = ""
        
        elif transition == "zoom_out":
            # Efecto de zoom out
            # No implementado en esta versión simplificada
            opacity_expr = ""
        
        elif transition == "rotate_in":
            # Efecto de rotación
            rotate_duration = min(1.5, duration * 0.3) if duration > 0 else 1.5
            opacity_expr = f":eval=frame:angle='if(lt(t,{start_time+rotate_duration}),360*(1-(t-{start_time})/{rotate_duration}),0)'"
        
        elif transition == "pulse":
            # Efecto de pulso (escala oscilante)
            pulse_freq = 3.0  # Frecuencia del pulso
            pulse_amp = 0.1   # Amplitud del pulso
            
            # Extraer coordenadas originales
            parts = position_expr.split(':')
            x_part = parts[0].split('=')[1]
            y_part = parts[1].split('=')[1].split(enable_expr)[0]
            
            # Calcular el centro para el pulso
            center_x = f"({x_part}+w/2)"
            center_y = f"({y_part}+h/2)"
            
            # Factor de escala pulsante
            scale_factor = f"(1+{pulse_amp}*sin({pulse_freq}*PI*(t-{start_time})))"
            
            # Nuevas coordenadas con escala oscilante
            new_x = f"({center_x}-w*{scale_factor}/2)"
            new_y = f"({center_y}-h*{scale_factor}/2)"
            
            position_expr = f"x={new_x}:y={new_y}:eval=frame"
            opacity_expr = f":eval=frame"
        
        else:
            # Transición desconocida o no implementada
            opacity_expr = ""
    else:
        opacity_expr = ""
    
    # Construir filtro completo
    return f"{position_expr}{enable_expr}{opacity_expr}"

def process_meme_overlay(
    video_url: str, 
    meme_url: str, 
    position: str = DEFAULT_POSITION, 
    scale: float = DEFAULT_SCALE, 
    job_id: str = "",
    webhook_url: Optional[str] = None,
    opacity: float = DEFAULT_OPACITY,
    border_width: int = DEFAULT_BORDER_WIDTH,
    border_color: str = DEFAULT_BORDER_COLOR,
    border_radius: int = DEFAULT_BORDER_RADIUS,
    rotation: float = DEFAULT_ROTATION,
    effect: str = DEFAULT_EFFECT,
    transition: str = DEFAULT_TRANSITION,
    start_time: float = 0.0,
    duration: float = DEFAULT_DURATION,
    x_offset: int = 0,
    y_offset: int = 0
) -> str:
    """
    Superpone una imagen de meme sobre un video con opciones avanzadas
    
    Args:
        video_url: URL del video
        meme_url: URL de la imagen de meme
        position: Posición del meme
        scale: Escala relativa del meme (0.1-1.0)
        job_id: ID del trabajo
        webhook_url: URL para notificación webhook
        opacity: Opacidad del meme (0.0-1.0)
        border_width: Ancho del borde en píxeles
        border_color: Color del borde
        border_radius: Radio de las esquinas redondeadas en píxeles
        rotation: Ángulo de rotación en grados
        effect: Efecto visual a aplicar
        transition: Efecto de transición
        start_time: Tiempo de inicio en segundos
        duration: Duración en segundos (0 = toda la duración)
        x_offset: Desplazamiento horizontal adicional en píxeles
        y_offset: Desplazamiento vertical adicional en píxeles
        
    Returns:
        str: Ruta al video procesado
        
    Raises:
        MemeOverlayError: Si hay un error en el procesamiento
    """
    # Crear identificador de trabajo único si no se proporciona
    if not job_id:
        job_id = f"meme_{int(time.time())}_{os.path.basename(meme_url).split('.')[0]}"
    
    try:
        # Validar parámetros
        if position not in AVAILABLE_POSITIONS:
            raise MemeOverlayError(f"Posición no válida: {position}. Posiciones disponibles: {', '.join(AVAILABLE_POSITIONS)}")
        
        if scale <= 0 or scale > 1.0:
            raise MemeOverlayError(f"Escala no válida: {scale}. Debe estar entre 0.1 y 1.0")
        
        if opacity < 0 or opacity > 1.0:
            raise MemeOverlayError(f"Opacidad no válida: {opacity}. Debe estar entre 0.0 y 1.0")
        
        if effect not in AVAILABLE_EFFECTS:
            raise MemeOverlayError(f"Efecto no válido: {effect}. Efectos disponibles: {', '.join(AVAILABLE_EFFECTS)}")
        
        if transition not in AVAILABLE_TRANSITIONS:
            raise MemeOverlayError(f"Transición no válida: {transition}. Transiciones disponibles: {', '.join(AVAILABLE_TRANSITIONS)}")
        
        if start_time < 0:
            raise MemeOverlayError(f"Tiempo de inicio no válido: {start_time}. Debe ser mayor o igual a 0")
        
        if duration < 0:
            raise MemeOverlayError(f"Duración no válida: {duration}. Debe ser mayor o igual a 0")
        
        # Descargar archivos
        video_path = download_file(video_url, config.TEMP_DIR)
        meme_path = download_file(meme_url, config.TEMP_DIR)
        
        logger.info(f"Job {job_id}: Descargado video en {video_path}")
        logger.info(f"Job {job_id}: Descargada imagen de meme en {meme_path}")
        
        # Obtener información del video
        video_info = get_video_info(video_path)
        video_width = video_info['width']
        video_height = video_info['height']
        video_duration = video_info['duration']
        
        logger.info(f"Job {job_id}: Dimensiones del video: {video_width}x{video_height}, duración: {video_duration}s")
        
        # Validar tiempos
        if start_time >= video_duration:
            raise MemeOverlayError(f"Tiempo de inicio ({start_time}s) mayor que la duración del video ({video_duration}s)")
        
        if duration > 0 and start_time + duration > video_duration:
            logger.warning(f"Job {job_id}: Ajustando duración para caber en el video")
            duration = video_duration - start_time
        
        # Si duración es 0, usar toda la duración del video desde el tiempo de inicio
        if duration <= 0:
            duration = video_duration - start_time
        
        # Preparar ruta de salida
        output_path = generate_temp_filename(prefix=f"{job_id}_", suffix=".mp4")
        
        # Preprocesar imagen de meme
        processed_meme_path = preprocess_meme_image(
            meme_path,
            scale=1.0,  # Escala se manejará en FFmpeg para mayor flexibilidad
            opacity=opacity if transition == "none" else 1.0,  # Si hay transición, opacidad se maneja en filtro
            border_width=border_width,
            border_color=border_color,
            border_radius=border_radius,
            rotation=rotation,
            effect=effect
        )
        
        # Generar filtro para overlay
        overlay_filter = generate_overlay_filter(
            position=position,
            scale=scale,
            x_offset=x_offset,
            y_offset=y_offset,
            start_time=start_time,
            duration=duration,
            transition=transition,
            rotation=rotation,
            width_ratio=1.0,
            height_ratio=1.0
        )
        
        # Configurar aceleración por hardware si está disponible
        hw_accel = []
        if USE_GPU:
            # Esta sección variará según la GPU disponible (NVIDIA, AMD, Intel)
            # Aquí se usa NVIDIA como ejemplo
            try:
                # Verificar si hay GPU NVIDIA disponible
                nvidia_check = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                if nvidia_check.returncode == 0:
                    hw_accel = ['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda']
                    logger.info(f"Job {job_id}: Usando aceleración GPU (NVIDIA)")
            except:
                logger.info(f"Job {job_id}: Aceleración GPU no disponible o no configurada")
        
        # Construir comando FFmpeg
        cmd = [
            'ffmpeg', 
            *hw_accel,
            '-i', video_path, 
            '-i', processed_meme_path,
            '-filter_complex', f"[1:v]scale=iw*{scale}:-1[overlay];[0:v][overlay]overlay={overlay_filter}",
            '-c:a', 'copy',  # Mantener audio original
            '-c:v', 'libx264',  # Usar codec H.264 para mejor compatibilidad
            '-preset', 'medium',  # Equilibrio entre calidad y velocidad
            '-crf', '23',  # Calidad constante (23 es un buen equilibrio)
            '-y', output_path
        ]
        
        logger.info(f"Job {job_id}: Ejecutando comando FFmpeg para meme overlay")
        logger.debug(f"Comando: {' '.join(cmd)}")
        
        # Ejecutar FFmpeg con manejo de timeout
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                timeout=FFMPEG_TIMEOUT
            )
            
            if result.returncode != 0:
                error_msg = result.stderr
                logger.error(f"Job {job_id}: Error de FFmpeg: {error_msg}")
                raise MemeOverlayError(f"Error aplicando overlay: {error_msg}")
            
        except subprocess.TimeoutExpired:
            logger.error(f"Job {job_id}: Timeout ejecutando FFmpeg después de {FFMPEG_TIMEOUT} segundos")
            if os.path.exists(video_path):
                os.remove(video_path)
            if os.path.exists(processed_meme_path):
                os.remove(processed_meme_path)
            if os.path.exists(output_path):
                os.remove(output_path)
            raise MemeOverlayError(f"Timeout procesando video. La operación tardó más de {FFMPEG_TIMEOUT} segundos.")
        
        # Verificar que el archivo se generó correctamente
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise MemeOverlayError("Error generando video: el archivo de salida está vacío o no existe")
        
        # Verificar integridad del archivo de salida
        if not verify_file_integrity(output_path):
            raise MemeOverlayError("El archivo generado está corrupto o incompleto")
        
        logger.info(f"Job {job_id}: Video con meme overlay creado exitosamente: {output_path}")
        
        # Limpiar archivos temporales
        os.remove(video_path)
        os.remove(meme_path)
        
        # No eliminamos processed_meme_path porque puede estar en caché para uso futuro
        
        return output_path
    
    except MemeOverlayError:
        # Reenviar excepciones específicas
        raise
    except Exception as e:
        logger.error(f"Job {job_id}: Error en process_meme_overlay: {str(e)}", exc_info=True)
        # Limpiar archivos temporales si existen
        if 'video_path' in locals() and os.path.exists(video_path):
            os.remove(video_path)
        if 'meme_path' in locals() and os.path.exists(meme_path):
            os.remove(meme_path)
        if 'output_path' in locals() and os.path.exists(output_path):
            os.remove(output_path)
        raise MemeOverlayError(f"Error procesando meme overlay: {str(e)}")

def process_multiple_memes(
    video_url: str,
    meme_items: List[Dict[str, Any]],
    job_id: str = ""
) -> str:
    """
    Superpone múltiples memes en un video en una sola operación para mayor eficiencia
    
    Args:
        video_url: URL del video
        meme_items: Lista de diccionarios con configuración para cada meme
                    Cada dict debe tener: meme_url, position, scale, etc.
        job_id: ID del trabajo
        
    Returns:
        str: Ruta al video procesado
        
    Raises:
        MemeOverlayError: Si hay un error en el procesamiento
    """
    # Validar entrada
    if not meme_items or not isinstance(meme_items, list):
        raise MemeOverlayError("Se debe proporcionar al menos un meme")
    
    # Crear identificador de trabajo único si no se proporciona
    if not job_id:
        job_id = f"multi_meme_{int(time.time())}"
    
    try:
        # Descargar video
        video_path = download_file(video_url, config.TEMP_DIR)
        logger.info(f"Job {job_id}: Video descargado para multi-meme: {video_path}")
        
        # Obtener información del video
        video_info = get_video_info(video_path)
        video_width = video_info['width']
        video_height = video_info['height']
        video_duration = video_info['duration']
        
        # Preparar ruta de salida
        output_path = generate_temp_filename(prefix=f"{job_id}_multi_", suffix=".mp4")
        
        # Construir cadena de filtro complejo combinada
        filter_parts = []
        input_count = 1  # El video principal es la entrada 0
        
        for i, item in enumerate(meme_items):
            # Aplicar valores por defecto para campos faltantes
            meme_url = item.get('meme_url')
            if not meme_url:
                logger.warning(f"Job {job_id}: Meme #{i+1} omitido: URL no proporcionada")
                continue
                
            position = item.get('position', DEFAULT_POSITION)
            scale = item.get('scale', DEFAULT_SCALE)
            opacity = item.get('opacity', DEFAULT_OPACITY)
            border_width = item.get('border_width', DEFAULT_BORDER_WIDTH)
            border_color = item.get('border_color', DEFAULT_BORDER_COLOR)
            border_radius = item.get('border_radius', DEFAULT_BORDER_RADIUS)
            rotation = item.get('rotation', DEFAULT_ROTATION)
            effect = item.get('effect', DEFAULT_EFFECT)
            transition = item.get('transition', DEFAULT_TRANSITION)
            start_time = item.get('start_time', 0.0)
            duration = item.get('duration', DEFAULT_DURATION)
            x_offset = item.get('x_offset', 0)
            y_offset = item.get('y_offset', 0)
            
            try:
                # Validar parámetros básicos
                if position not in AVAILABLE_POSITIONS:
                    logger.warning(f"Job {job_id}: Meme #{i+1} posición inválida: {position}. Usando default.")
                    position = DEFAULT_POSITION
                
                if effect not in AVAILABLE_EFFECTS:
                    logger.warning(f"Job {job_id}: Meme #{i+1} efecto inválido: {effect}. Usando default.")
                    effect = DEFAULT_EFFECT
                
                if transition not in AVAILABLE_TRANSITIONS:
                    logger.warning(f"Job {job_id}: Meme #{i+1} transición inválida: {transition}. Usando default.")
                    transition = DEFAULT_TRANSITION
                
                # Validar tiempo
                if start_time >= video_duration:
                    logger.warning(f"Job {job_id}: Meme #{i+1} tiempo inicio fuera de video. Omitiendo.")
                    continue
                    
                if duration > 0 and start_time + duration > video_duration:
                    duration = video_duration - start_time
                
                # Descargar meme
                meme_path = download_file(meme_url, config.TEMP_DIR)
                
                # Preprocesar imagen
                processed_meme_path = preprocess_meme_image(
                    meme_path,
                    scale=1.0,
                    opacity=opacity if transition == "none" else 1.0,
                    border_width=border_width,
                    border_color=border_color,
                    border_radius=border_radius,
                    rotation=rotation,
                    effect=effect
                )
                
                # Añadir entrada al comando
                meme_index = input_count
                input_count += 1
                
                # Generar filtro overlay para este meme
                overlay_filter = generate_overlay_filter(
                    position=position,
                    scale=scale,
                    x_offset=x_offset,
                    y_offset=y_offset,
                    start_time=start_time,
                    duration=duration,
                    transition=transition,
                    rotation=rotation,
                    width_ratio=1.0,
                    height_ratio=1.0
                )
                
                # Añadir parte de filtro para este meme
                if i == 0:
                    # Primer meme
                    filter_parts.append(f"[{meme_index}:v]scale=iw*{scale}:-1[overlay{i}]")
                    filter_parts.append(f"[0:v][overlay{i}]overlay={overlay_filter}[v{i}]")
                else:
                    # Memes subsiguientes
                    filter_parts.append(f"[{meme_index}:v]scale=iw*{scale}:-1[overlay{i}]")
                    filter_parts.append(f"[v{i-1}][overlay{i}]overlay={overlay_filter}[v{i}]")
            
            except Exception as e:
                logger.error(f"Job {job_id}: Error procesando meme #{i+1}: {str(e)}")
                # Continuar con el siguiente meme si hay error en uno
                continue
        
        # Verificar que al menos un meme se procesó
        if not filter_parts:
            raise MemeOverlayError("No hay memes válidos para procesar")
        
        # Añadir salida final
        last_v = f"v{len(meme_items) - 1}"
        filter_complex = ";".join(filter_parts) + f";[{last_v}]format=yuv420p[outv]"
        
        # Construir comando FFmpeg para multi-meme
        cmd = ['ffmpeg', '-y', '-i', video_path]
        
        # Añadir entradas de memes
        for i in range(1, input_count):
            cmd.extend(['-i', f"{config.TEMP_DIR}/processed_meme_{i-1}.png"])
        
        # Completar comando
        cmd.extend([
            '-filter_complex', filter_complex,
            '-map', '[outv]',
            '-map', '0:a',
            '-c:a', 'copy',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            output_path
        ])
        
        logger.info(f"Job {job_id}: Ejecutando comando FFmpeg para múltiples memes")
        
        # Ejecutar FFmpeg
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                timeout=FFMPEG_TIMEOUT
            )
            
            if result.returncode != 0:
                logger.error(f"Job {job_id}: Error ejecutando FFmpeg para multi-meme: {result.stderr}")
                raise MemeOverlayError(f"Error generando video con múltiples memes: {result.stderr}")
            
        except subprocess.TimeoutExpired:
            logger.error(f"Job {job_id}: Timeout en FFmpeg después de {FFMPEG_TIMEOUT} segundos")
            # Limpiar archivos temporales
            cleanup_temp_files(job_id)
            raise MemeOverlayError(f"Timeout procesando video con múltiples memes")
        
        # Verificar archivo de salida
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise MemeOverlayError("Error generando video: archivo de salida vacío o inexistente")
        
        # Verificar integridad
        if not verify_file_integrity(output_path):
            raise MemeOverlayError("El archivo generado está corrupto o incompleto")
        
        logger.info(f"Job {job_id}: Video con múltiples memes creado: {output_path}")
        
        # Limpiar archivos temporales
        cleanup_temp_files(job_id)
        
        return output_path
    
    except MemeOverlayError:
        raise
    except Exception as e:
        logger.error(f"Job {job_id}: Error en process_multiple_memes: {str(e)}", exc_info=True)
        # Limpiar archivos temporales
        cleanup_temp_files(job_id)
        raise MemeOverlayError(f"Error procesando múltiples memes: {str(e)}")

def cleanup_temp_files(job_id: str) -> None:
    """
    Limpia archivos temporales relacionados con un trabajo
    
    Args:
        job_id: ID del trabajo
    """
    try:
        # Eliminar archivos de video y memes descargados
        for filename in os.listdir(config.TEMP_DIR):
            if job_id in filename or filename.startswith('processed_meme_'):
                file_path = os.path.join(config.TEMP_DIR, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
    except Exception as e:
        logger.error(f"Error limpiando archivos temporales para job {job_id}: {str(e)}")

def get_overlay_presets() -> Dict[str, Dict[str, Any]]:
    """
    Devuelve presets predefinidos para overlays comunes
    
    Returns:
        Dict: Diccionario de presets con sus configuraciones
    """
    return {
        "watermark_corner": {
            "position": "bottom_right",
            "scale": 0.15,
            "opacity": 0.8,
            "border_width": 0,
            "effect": "none",
            "transition": "fade_in",
            "description": "Marca de agua pequeña en esquina inferior derecha"
        },
        "logo_intro": {
            "position": "center",
            "scale": 0.5,
            "opacity": 1.0,
            "border_width": 0,
            "effect": "none",
            "transition": "fade_in_out",
            "duration": 3.0,
            "description": "Logo centrado con fade in/out para intros"
        },
        "commentary_meme": {
            "position": "bottom",
            "scale": 0.3,
            "opacity": 1.0,
            "border_width": 2,
            "border_color": "white",
            "border_radius": 10,
            "effect": "none",
            "transition": "slide_in",
            "description": "Meme de comentario que se desliza desde abajo"
        },
        "floating_emoji": {
            "position": "center",
            "scale": 0.2,
            "opacity": 1.0,
            "effect": "none",
            "transition": "pulse",
            "duration": 2.0,
            "description": "Emoji flotante con efecto de pulso"
        },
        "dramatic_effect": {
            "position": "center",
            "scale": 0.8,
            "opacity": 0.9,
            "effect": "blur",
            "transition": "fade_in_out",
            "duration": 1.5,
            "description": "Efecto dramático en toda la pantalla"
        }
    }
