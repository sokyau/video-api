# --- START OF FILE transcription.py ---

import os
import json
import logging
import subprocess # For Whisper CLI and direct FFmpeg fallback
import tempfile # Not directly used now, but good to keep in mind for true temp files
from typing import Dict, Any, Optional # Added for type hinting

# Custom error handling
from errors import (
    ValidationError,
    NotFoundError,
    ProcessingError,
    FFmpegError,    # If ffmpeg_toolkit or direct ffmpeg call fails
    StorageError,   # If store_file fails
    capture_exception
)

# File management (for generate_temp_filename) and local storage
# We will replace the direct call to services.file_management.extract_audio
# with a call to our internal _extract_audio_for_transcription_internal
# which will use ffmpeg_toolkit if available.
from services.file_management import generate_temp_filename
from services.local_storage import store_file # Assumes this is refactored

# FFmpeg toolkit
try:
    import ffmpeg_toolkit
except ImportError:
    logging.warning("ffmpeg_toolkit.py not found directly. Audio extraction will use direct subprocess calls.")
    ffmpeg_toolkit = None # Fallback

import config

logger = logging.getLogger(__name__)

# Define WHISPER_CLI_PATH for clarity and potential configuration
WHISPER_CLI_PATH = getattr(config, 'WHISPER_CLI_PATH', 'whisper') # Default to 'whisper' if not in config

def _extract_audio_for_transcription_internal(media_path: str, job_id: Optional[str] = None) -> str:
    """
    Extracts audio from media to WAV format, 16kHz mono, for Whisper.
    Uses ffmpeg_toolkit if available.

    Args:
        media_path (str): Path to the input media file.
        job_id (str, optional): Job ID for logging/temp filenames.

    Returns:
        str: Path to the extracted WAV audio file.

    Raises:
        NotFoundError: If media_path does not exist.
        FFmpegError/ProcessingError: If FFmpeg fails.
        StorageError: If temporary file creation fails.
    """
    if not os.path.exists(media_path):
        raise NotFoundError(f"Media file not found for audio extraction: {media_path}",
                            error_code="extract_audio_input_not_found",
                            details={"media_path": media_path})

    prefix = f"{job_id}_audio_" if job_id else "audio_ext_"
    # generate_temp_filename from file_management should handle StorageError if config.TEMP_DIR is bad
    output_wav_path = generate_temp_filename(prefix=prefix, suffix=".wav")

    ffmpeg_params = [
        '-vn',  # No video
        '-ar', '16000',  # 16kHz sample rate
        '-ac', '1',  # Mono channel
        '-c:a', 'pcm_s16le',  # WAV format (signed 16-bit little-endian PCM)
        '-y' # Overwrite output
    ]

    if ffmpeg_toolkit and hasattr(ffmpeg_toolkit, 'execute_ffmpeg_command'):
        # Construct full command for execute_ffmpeg_command
        cmd = ['ffmpeg', '-i', media_path] + ffmpeg_params + [output_wav_path]
        try:
            ffmpeg_toolkit.execute_ffmpeg_command(cmd, output_path_check=output_wav_path, timeout=config.FFMPEG_TIMEOUT)
        except (FFmpegError, ProcessingError, ValidationError, TimeoutError) as e: # Catch toolkit's errors
            # Add context if not already present
            e.details = e.details or {}
            e.details["operation_context"] = "audio_extraction_for_transcription"
            e.details["input_media"] = media_path
            raise
    else: # Fallback to direct subprocess
        logger.warning("Usando FFmpeg directo para extracción de audio (transcription - _extract_audio_for_transcription_internal).")
        cmd = ['ffmpeg', '-i', media_path] + ffmpeg_params + [output_wav_path]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=config.FFMPEG_TIMEOUT)
            if not os.path.exists(output_wav_path) or os.path.getsize(output_wav_path) == 0:
                raise ProcessingError("FFmpeg (directo) no generó archivo WAV o está vacío para transcripción.",
                                      error_code="ffmpeg_direct_wav_output_missing",
                                      details={"command": ' '.join(cmd), "stderr": result.stderr[:500] if result else ""})
        except subprocess.CalledProcessError as e:
            # Convert to FFmpegError
            raise FFmpegError.from_ffmpeg_error(stderr=e.stderr, cmd=cmd)
        except subprocess.TimeoutExpired as e:
            error_id = capture_exception(e, {"command": ' '.join(cmd)})
            raise ProcessingError(f"Timeout extrayendo audio para transcripción: {media_path}",
                                  error_code="ffmpeg_extract_audio_timeout",
                                  details={"error_id": error_id, "media_path": media_path})
    return output_wav_path


def transcribe_media(media_path: str, language: str = 'auto', output_format: str = 'txt',
                     job_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Transcribe un archivo de audio o video utilizando Whisper CLI.
    
    Args:
        media_path (str): Ruta al archivo de media (local path, not URL).
        language (str, optional): Código de idioma o 'auto'.
        output_format (str, optional): Formato de salida (txt, srt, vtt, json, tsv, all).
        job_id (str, optional): ID del trabajo.
    
    Returns:
        dict: Resultado de la transcripción.
    Raises:
        NotFoundError: If media_path does not exist.
        ValidationError: For invalid parameters (e.g., unsupported output_format by this logic).
        ProcessingError: For Whisper CLI failures or other processing issues.
        FFmpegError: If underlying audio extraction fails.
        StorageError: If storing the output file fails.
    """
    audio_path_local: Optional[str] = None
    # Whisper output path is determined by whisper CLI itself based on --output_dir and input filename.
    # We need to predict it or find it.
    whisper_generated_output_path: Optional[str] = None

    # Create job_id if not provided for logging and temp file prefixing
    effective_job_id = job_id or f"transcribe_{int(time.time())}"

    try:
        if not os.path.exists(media_path):
            raise NotFoundError(f"Archivo multimedia no encontrado para transcripción: {media_path}",
                                error_code="transcribe_input_media_not_found",
                                details={"media_path": media_path})

        # Validate output_format against common Whisper formats
        supported_formats = ['txt', 'vtt', 'srt', 'tsv', 'json'] # 'all' is also valid for CLI
        if output_format not in supported_formats and output_format != 'all':
            raise ValidationError(f"Formato de salida no soportado para transcripción: '{output_format}'. Soportados: {supported_formats}",
                                  error_code="transcribe_unsupported_output_format",
                                  details={"requested_format": output_format, "supported": supported_formats})

        logger.info(f"Job {effective_job_id}: Iniciando transcripción para {media_path}")
        # Extraer audio si es un video, o convertir a WAV si es otro formato de audio
        audio_path_local = _extract_audio_for_transcription_internal(media_path, job_id=effective_job_id)
        logger.info(f"Job {effective_job_id}: Audio preparado para transcripción en {audio_path_local}")
        
        # --- Whisper Transcription ---
        # Create a temporary directory for Whisper to output its files.
        # This makes cleanup easier and output path prediction more reliable.
        with tempfile.TemporaryDirectory(prefix=f"{effective_job_id}_whisper_out_", dir=config.TEMP_DIR) as whisper_temp_out_dir:
            cmd = [
                WHISPER_CLI_PATH,
                audio_path_local,
                '--model', config.WHISPER_MODEL,
                '--output_dir', whisper_temp_out_dir # Whisper writes here
            ]
            
            if language.lower() != 'auto':
                cmd.extend(['--language', language])
            
            # Whisper CLI handles multiple formats with 'all' or specific format.
            # If 'all', it creates files for each format. We'll handle 'txt' specially for direct content.
            cmd.extend(['--output_format', output_format if output_format != 'txt' else 'txt']) # For txt, just get txt.
                                                                                              # If user wants 'all', they'd get multiple files.
                                                                                              # This function's design implies one primary output.

            logger.info(f"Job {effective_job_id}: Ejecutando comando Whisper: {' '.join(cmd)}")
            
            try:
                # Timeout for Whisper can be quite long depending on audio length and model
                whisper_timeout = getattr(config, 'WHISPER_TIMEOUT', 3600) # Default 1 hour
                result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=whisper_timeout) # check=False to handle manually
            except subprocess.TimeoutExpired as e:
                error_id = capture_exception(e, {"command": ' '.join(cmd), "job_id": effective_job_id})
                raise ProcessingError(f"Whisper CLI timeout después de {whisper_timeout}s para {audio_path_local}",
                                      error_code="whisper_timeout",
                                      details={"error_id": error_id, "audio_path": audio_path_local, "timeout": whisper_timeout})

            if result.returncode != 0:
                error_id = capture_exception(RuntimeError(f"Whisper CLI failed with stderr: {result.stderr[:1000]}"), 
                                             {"command": ' '.join(cmd), "job_id": effective_job_id, "stderr": result.stderr})
                # Attempt to give a more specific error based on common Whisper issues
                stderr_lower = result.stderr.lower()
                if "no such file or directory" in stderr_lower and "ffmpeg" in stderr_lower:
                     raise ProcessingError(f"Whisper CLI no pudo encontrar FFmpeg. Asegúrese que FFmpeg está en el PATH.",
                                           error_code="whisper_ffmpeg_not_found",
                                           details={"error_id": error_id, "stderr": result.stderr[:1000]})
                if "torch" in stderr_lower and ("out of memory" in stderr_lower or "cuda" in stderr_lower):
                     raise ProcessingError(f"Whisper CLI falló por memoria insuficiente (GPU/CPU) o problema de CUDA.",
                                           error_code="whisper_memory_cuda_error",
                                           details={"error_id": error_id, "stderr": result.stderr[:1000]})

                raise ProcessingError(f"Whisper CLI falló. Stderr: {result.stderr[:1000]}", # Limit stderr length
                                      error_code="whisper_cli_failed",
                                      details={"error_id": error_id, "stderr": result.stderr[:1000]})

            # --- Process Whisper Output ---
            # Whisper names output files based on the input audio file name.
            base_audio_name = os.path.splitext(os.path.basename(audio_path_local))[0]
            
            # Determine the language detected by Whisper (it usually prints this)
            # This is a heuristic, might need refinement based on actual Whisper output.
            detected_lang_from_whisper = language # Assume specified lang unless auto
            if language.lower() == 'auto':
                lang_match = re.search(r"Detected language: (\w+)", result.stdout + result.stderr, re.IGNORECASE)
                if lang_match:
                    detected_lang_from_whisper = lang_match.group(1)
                else:
                    detected_lang_from_whisper = "auto-detected (info not found in output)"


            if output_format == 'txt':
                whisper_generated_output_path = os.path.join(whisper_temp_out_dir, f"{base_audio_name}.txt")
                if not os.path.exists(whisper_generated_output_path):
                    raise ProcessingError(f"Archivo de transcripción TXT esperado no encontrado: {whisper_generated_output_path}",
                                          error_code="whisper_txt_output_missing",
                                          details={"expected_path": whisper_generated_output_path, "whisper_stdout": result.stdout[:500]})
                with open(whisper_generated_output_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                response = {
                    "transcription": text_content,
                    "format": "txt",
                    "language_detected": detected_lang_from_whisper
                }
            else: # srt, vtt, json, tsv, etc.
                whisper_generated_output_path = os.path.join(whisper_temp_out_dir, f"{base_audio_name}.{output_format}")
                if not os.path.exists(whisper_generated_output_path):
                    # If 'all' was used, it might be just one of them.
                    # This logic assumes specific format was requested.
                    raise ProcessingError(f"Archivo de transcripción '{output_format}' esperado no encontrado: {whisper_generated_output_path}",
                                          error_code="whisper_formatted_output_missing",
                                          details={"expected_path": whisper_generated_output_path, "whisper_stdout": result.stdout[:500]})
                
                # Store the generated file (e.g., SRT, VTT) using local_storage
                # store_file can raise StorageError, NotFoundError (if whisper_generated_output_path vanishes)
                file_url = store_file(whisper_generated_output_path,
                                      custom_filename=f"{effective_job_id}_transcription.{output_format}")
                response = {
                    "transcription_url": file_url,
                    "format": output_format,
                    "language_detected": detected_lang_from_whisper
                }
        # TemporaryDirectory (whisper_temp_out_dir) and its contents are auto-cleaned up here.

        logger.info(f"Job {effective_job_id}: Transcripción completada exitosamente.")
        return response
    
    except (NotFoundError, ValidationError, ProcessingError, FFmpegError, StorageError) as e:
        # These are already our custom errors, ensure error_id is captured and attached if not already
        error_id = e.details.get("error_id") if hasattr(e, 'details') and isinstance(e.details, dict) else None
        if not error_id:
             error_id = capture_exception(e, {"job_id": effective_job_id, "media_path_original": media_path, "language": language, "output_format": output_format})
             if hasattr(e, 'details') and isinstance(e.details, dict): e.details["error_id"] = error_id
             elif not hasattr(e, 'details'): e.details = {"error_id": error_id}

        logger.error(f"Job {effective_job_id}: Error en transcribe_media ({e.error_code}): {e.message}. Details: {e.details}")
        raise # Re-raise the captured custom error
    except Exception as e: # Catch any other unexpected errors
        error_id = capture_exception(e, {"job_id": effective_job_id, "media_path_original": media_path, "language": language, "output_format": output_format})
        logger.error(f"Job {effective_job_id}: Error inesperado en transcribe_media: {str(e)}. Error ID: {error_id}", exc_info=True)
        raise ProcessingError(message=f"Error inesperado durante la transcripción: {str(e)}",
                              error_code="transcription_unexpected_error",
                              details={"original_error_type": str(type(e).__name__), "error_id": error_id})
    finally:
        # Cleanup extracted audio file
        if audio_path_local and os.path.exists(audio_path_local):
            try:
                os.remove(audio_path_local)
                logger.debug(f"Job {effective_job_id}: Archivo de audio extraído limpiado: {audio_path_local}")
            except OSError as e_clean:
                capture_exception(e_clean, {"job_id": effective_job_id, "file_path": audio_path_local, "context": "cleanup_transcription_audio"})
                logger.warning(f"Job {effective_job_id}: No se pudo limpiar el archivo de audio extraído '{audio_path_local}': {str(e_clean)}")
        # whisper_generated_output_path is inside a TemporaryDirectory, so it's cleaned automatically.
        # If not using TemporaryDirectory, manual cleanup of whisper_generated_output_path would be needed here in case of error before storage.

# --- END OF FILE transcription.py ---
