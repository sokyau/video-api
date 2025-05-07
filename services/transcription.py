import os
import json
import logging
import subprocess
import tempfile
from services.file_management import extract_audio, generate_temp_filename
from services.local_storage import store_file
import config

logger = logging.getLogger(__name__)

def transcribe_media(media_path, language='auto', output_format='txt', job_id=None):
    """
    Transcribe un archivo de audio o video utilizando Whisper
    
    Args:
        media_path (str): Ruta al archivo de media
        language (str, optional): Código de idioma o 'auto' para autodetección
        output_format (str, optional): Formato de salida (txt, srt, vtt, json)
        job_id (str, optional): ID del trabajo
    
    Returns:
        dict: Resultado de la transcripción con URL del archivo o texto directo
    """
    try:
        # Extraer audio si es un video
        audio_path = extract_audio_for_transcription(media_path)
        logger.info(f"Audio extracted for transcription: {audio_path}")
        
        # Preparar rutas de salida
        prefix = f"{job_id}_" if job_id else ""
        txt_output = generate_temp_filename(prefix=prefix, suffix=".txt")
        
        # Usar whisper para transcribir
        cmd = ['whisper', audio_path, '--model', config.WHISPER_MODEL]
        
        if language != 'auto':
            cmd.extend(['--language', language])
        
        if output_format != 'txt':
            cmd.extend(['--output_format', output_format])
        
        cmd.extend(['--output_dir', os.path.dirname(txt_output)])
        
        logger.info(f"Running Whisper command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Whisper command failed: {result.stderr}")
            raise RuntimeError(f"Transcription failed: {result.stderr}")
        
        # Determinar ruta del archivo de salida y formato de respuesta
        output_path = os.path.splitext(audio_path)[0]
        
        if output_format == 'txt':
            output_path = f"{output_path}.txt"
            with open(output_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            # Para formato TXT devolver el texto directamente
            response = {
                "transcription": text_content,
                "language_detected": language if language != 'auto' else "auto-detected"
            }
            
            # Limpiar archivo temporal
            os.remove(output_path)
            
        else:
            # Para otros formatos (SRT, VTT, JSON)
            ext = f".{output_format}"
            output_path = f"{output_path}{ext}"
            
            if not os.path.exists(output_path):
                logger.error(f"Expected output file not found: {output_path}")
                raise FileNotFoundError(f"Transcription output file not found")
            
            # Almacenar archivo para descarga
            file_url = store_file(output_path)
            
            response = {
                "transcription_url": file_url,
                "format": output_format,
                "language_detected": language if language != 'auto' else "auto-detected"
            }
            
            # Limpiar archivo temporal
            os.remove(output_path)
        
        # Limpiar audio extraído
        os.remove(audio_path)
        
        logger.info(f"Transcription completed successfully")
        return response
    
    except Exception as e:
        logger.error(f"Error in transcribe_media: {str(e)}", exc_info=True)
        # Limpiar archivos temporales en caso de error
        if 'audio_path' in locals() and os.path.exists(audio_path):
            os.remove(audio_path)
        
        if 'output_path' in locals() and os.path.exists(output_path):
            os.remove(output_path)
        
        raise

def extract_audio_for_transcription(media_path):
    """
    Extrae audio de un archivo multimedia en formato WAV para transcripción
    
    Args:
        media_path (str): Ruta al archivo multimedia
        
    Returns:
        str: Ruta al archivo de audio extraído
    """
    # Verificar si ya es un archivo de audio
    audio_extensions = ['.mp3', '.wav', '.aac', '.flac', '.ogg']
    if os.path.splitext(media_path)[1].lower() in audio_extensions:
        # Si ya es un archivo de audio, convertir a WAV para mejor compatibilidad
        output_path = generate_temp_filename(suffix=".wav")
        
        cmd = [
            'ffmpeg',
            '-i', media_path,
            '-ar', '16000',  # Muestreo común para reconocimiento de voz
            '-ac', '1',      # Mono para mejor rendimiento
            '-c:a', 'pcm_s16le',  # Formato PCM
            '-y', output_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    else:
        # Si es un video, extraer solo el audio
        output_path = generate_temp_filename(suffix=".wav")
        
        cmd = [
            'ffmpeg',
            '-i', media_path,
            '-vn',           # Sin video
            '-ar', '16000',  # Muestreo
            '-ac', '1',      # Mono
            '-c:a', 'pcm_s16le',  # Formato PCM
            '-y', output_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
