import os
import subprocess
import logging
from services.file_management import download_file, generate_temp_filename
from PIL import Image
import config

logger = logging.getLogger(__name__)

def process_meme_overlay(video_url, meme_url, position, scale, job_id, webhook_url=None):
    """
    Superponer una imagen de meme sobre un video
    
    Args:
        video_url (str): URL del video
        meme_url (str): URL de la imagen de meme
        position (str): Posición del meme (top, bottom, left, right, etc.)
        scale (float): Escala relativa del meme (0.1-1.0)
        job_id (str): ID del trabajo
        webhook_url (str, optional): URL para notificación webhook
    
    Returns:
        str: Ruta al video con meme superpuesto
    """
    try:
        # Descargar archivos
        video_path = download_file(video_url, config.TEMP_DIR)
        meme_path = download_file(meme_url, config.TEMP_DIR)
        
        logger.info(f"Downloaded video to {video_path}")
        logger.info(f"Downloaded meme image to {meme_path}")

        # Preparar ruta de salida
        output_path = generate_temp_filename(prefix=f"{job_id}_", suffix=".mp4")

        # Obtener dimensiones del video
        probe_cmd = [
            'ffprobe', '-v', 'error', 
            '-select_streams', 'v:0', 
            '-show_entries', 'stream=width,height', 
            '-of', 'csv=p=0', video_path
        ]
        
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        video_width, video_height = map(int, probe_result.stdout.strip().split(','))
        logger.info(f"Video dimensions: {video_width}x{video_height}")

        # Calcular posición del overlay
        overlay_positions = {
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
        
        position_expr = overlay_positions.get(position, overlay_positions["bottom"])
        
        # Comando FFmpeg para superponer el meme
        cmd = [
            'ffmpeg', '-i', video_path, '-i', meme_path,
            '-filter_complex', f"[1:v]scale=iw*{scale}:-1[overlay];[0:v][overlay]overlay={position_expr}",
            '-c:a', 'copy', '-y', output_path
        ]
        
        logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
        
        # Ejecutar FFmpeg
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg command failed. Error: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
        
        logger.info(f"Video with meme overlay created successfully: {output_path}")
        
        # Limpiar archivos de entrada
        os.remove(video_path)
        os.remove(meme_path)
        
        return output_path
    except Exception as e:
        logger.error(f"Error in process_meme_overlay: {str(e)}", exc_info=True)
        # Limpiar archivos si existen
        if 'video_path' in locals() and os.path.exists(video_path):
            os.remove(video_path)
        if 'meme_path' in locals() and os.path.exists(meme_path):
            os.remove(meme_path)
        raise
