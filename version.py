"""
Control de versiones para la API de procesamiento de video.
"""

VERSION = "1.0.0"
API_VERSION = "v1"
BUILD_DATE = "2025-05-06"

def get_version_info():
    """Retorna información de versión como diccionario."""
    return {
        "version": VERSION,
        "api_version": API_VERSION,
        "build_date": BUILD_DATE
    }

def get_version_string():
    """Retorna string formateado con información de versión."""
    return f"VideoAPI v{VERSION} (API {API_VERSION}) - Build {BUILD_DATE}"
