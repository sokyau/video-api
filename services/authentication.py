import functools
import logging
from flask import request, jsonify
import os

logger = logging.getLogger(__name__)

# Importar configuración - con manejo de error para desarrollo incremental
try:
    import config
    API_KEY = config.API_KEY
except ImportError:
    logger.warning("No se pudo importar config.py, usando API_KEY desde variables de entorno")
    API_KEY = os.environ.get('API_KEY', 'api_key_segura_temporal')

def authenticate(f):
    """
    Decorador para autenticación de API mediante API key.
    """
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            logger.warning("API request without key")
            return jsonify({
                "status": "error",
                "error": "API key is required"
            }), 401
        
        if api_key != API_KEY:
            logger.warning("API request with invalid key")
            return jsonify({
                "status": "error",
                "error": "Invalid API key"
            }), 401
        
        # Si la autenticación es exitosa, continuar con la función original
        return f(*args, **kwargs)
    
    return decorated_function
