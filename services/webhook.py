import requests
import logging
import json
import time

logger = logging.getLogger(__name__)

def send_webhook_notification(webhook_url, data, max_retries=3, retry_delay=1):
    """
    Envía una notificación webhook con los datos proporcionados
    
    Args:
        webhook_url (str): URL del webhook
        data (dict): Datos a enviar en el webhook
        max_retries (int, optional): Número máximo de reintentos en caso de fallo
        retry_delay (int, optional): Tiempo de espera entre reintentos (segundos)
    
    Returns:
        bool: True si el webhook se envió correctamente, False en caso contrario
    """
    headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'VideoAPI-Webhook-Service/1.0'
    }
    
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Enviar solicitud HTTP POST
            response = requests.post(
                webhook_url,
                headers=headers,
                data=json.dumps(data),
                timeout=10  # Timeout de 10 segundos
            )
            
            # Verificar respuesta
            if response.status_code >= 200 and response.status_code < 300:
                logger.info(f"Webhook notification sent successfully to {webhook_url}")
                return True
            else:
                logger.warning(f"Webhook failed with status {response.status_code}: {response.text}")
                
                # Incrementar contador de reintentos
                retry_count += 1
                
                if retry_count < max_retries:
                    # Esperar antes de reintentar con retraso exponencial
                    wait_time = retry_delay * (2 ** (retry_count - 1))
                    logger.info(f"Retrying webhook in {wait_time} seconds (attempt {retry_count+1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Webhook failed after {max_retries} attempts")
                    return False
        
        except Exception as e:
            logger.error(f"Error sending webhook notification: {str(e)}")
            
            # Incrementar contador de reintentos
            retry_count += 1
            
            if retry_count < max_retries:
                # Esperar antes de reintentar
                wait_time = retry_delay * (2 ** (retry_count - 1))
                logger.info(f"Retrying webhook in {wait_time} seconds (attempt {retry_count+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                logger.error(f"Webhook failed after {max_retries} attempts due to exception: {str(e)}")
                return False
    
    return False

def create_webhook_notification_data(job_id, status, result=None, error=None, endpoint=None):
    """
    Crea un objeto de datos para notificación webhook
    
    Args:
        job_id (str): ID del trabajo
        status (str): Estado del trabajo (completed, error)
        result (str, optional): Resultado del trabajo (URL del archivo)
        error (str, optional): Mensaje de error si hubo problema
        endpoint (str, optional): Endpoint que procesó la solicitud
    
    Returns:
        dict: Datos formateados para el webhook
    """
    notification_data = {
        "job_id": job_id,
        "status": status,
        "timestamp": time.time()
    }
    
    if result:
        notification_data["result"] = result
    
    if error:
        notification_data["error"] = error
    
    if endpoint:
        notification_data["endpoint"] = endpoint
    
    return notification_data
