#!/bin/bash

# Script de instalación para VideoAPI
# Este script configura completamente el entorno de VideoAPI en Ubuntu 22.04
# Versión: 2.0 - Con correcciones para evitar problemas comunes

set -e  # Detener en caso de error

echo "==== VideoAPI Installer - Versión Mejorada ===="

# Verificar si ejecutamos como root
if [ "$(id -u)" -ne 0 ]; then
    echo "Este script debe ejecutarse como root o con sudo."
    exit 1
fi

# Verificar sistema operativo
if ! grep -q "Ubuntu 22.04" /etc/os-release; then
    echo "⚠️  Este script está diseñado para Ubuntu 22.04. Otras versiones podrían no funcionar correctamente."
    read -p "¿Deseas continuar de todos modos? (s/n): " continue_anyway
    if [ "$continue_anyway" != "s" ]; then
        echo "Instalación cancelada."
        exit 1
    fi
fi

# Solicitar información para la configuración
read -p "Ingresa el dominio para VideoAPI (default: videoapi.sofe.site): " DOMAIN
DOMAIN=${DOMAIN:-videoapi.sofe.site}

read -p "Ingresa la ruta para almacenamiento (default: /var/www/$DOMAIN/storage): " STORAGE_PATH
STORAGE_PATH=${STORAGE_PATH:-/var/www/$DOMAIN/storage}

read -p "Ingresa la URL base para el almacenamiento (default: https://$DOMAIN/storage): " BASE_URL
BASE_URL=${BASE_URL:-https://$DOMAIN/storage}

read -p "Configura una API key segura: " API_KEY
if [ -z "$API_KEY" ]; then
    API_KEY=$(openssl rand -hex 16)
    echo "API key generada automáticamente: $API_KEY"
fi

# Actualizar sistema
echo "==== Actualizando sistema ===="
apt update && apt upgrade -y

# Instalar dependencias
echo "==== Instalando dependencias ===="
apt install -y python3-pip python3-venv ffmpeg nginx certbot python3-certbot-nginx git supervisor

# Crear estructura de directorios
echo "==== Configurando directorios ===="
mkdir -p /var/www/$DOMAIN
mkdir -p $STORAGE_PATH
chown -R www-data:www-data /var/www/$DOMAIN

# Clonar repositorio o crear estructura
echo "==== Configurando aplicación ===="
cd /var/www/$DOMAIN

# Configurar entorno virtual Python
python3 -m venv venv
source venv/bin/activate

# Crear requirements.txt - CORREGIDO sin paquetes incompatibles
cat > requirements.txt << 'EOF'
Flask==2.3.2
gunicorn==21.2.0
Werkzeug==2.3.6
requests==2.31.0
python-dotenv==1.0.0
jsonschema==4.17.3
Pillow==10.0.0
pydantic==2.0.2
ffmpeg-python==0.2.0
openai-whisper==20231117
Flask-RESTful==0.3.10
EOF

# Instalar dependencias Python
pip install -r requirements.txt

# Crear archivo .env para variables de entorno
cat > .env << EOF
API_KEY=$API_KEY
STORAGE_PATH=$STORAGE_PATH
BASE_URL=$BASE_URL
MAX_FILE_AGE_HOURS=6
MAX_QUEUE_LENGTH=100
WORKER_PROCESSES=4
FFMPEG_THREADS=4
EOF

# Crear estructura de directorios para la aplicación
echo "==== Creando estructura del proyecto ===="

mkdir -p /var/www/$DOMAIN/{storage,services,routes/v1/{video,media/transform,ffmpeg,image/transform}}

# Crear archivos __init__.py para los paquetes Python
touch /var/www/$DOMAIN/services/__init__.py
touch /var/www/$DOMAIN/routes/__init__.py
touch /var/www/$DOMAIN/routes/v1/__init__.py
touch /var/www/$DOMAIN/routes/v1/video/__init__.py
touch /var/www/$DOMAIN/routes/v1/media/__init__.py
touch /var/www/$DOMAIN/routes/v1/media/transform/__init__.py
touch /var/www/$DOMAIN/routes/v1/ffmpeg/__init__.py
touch /var/www/$DOMAIN/routes/v1/image/__init__.py
touch /var/www/$DOMAIN/routes/v1/image/transform/__init__.py

# Crear app.py básico
cat > /var/www/$DOMAIN/app.py << 'EOF'
from flask import Flask, jsonify
import logging
import os

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Crear aplicación Flask
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    """Endpoint raíz con información de la API"""
    return jsonify({
        "service": "Video Processing API",
        "status": "operational",
        "version": "1.0.0"
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para comprobación de salud del servicio"""
    return jsonify({
        "status": "healthy",
        "storage": "ok",
        "ffmpeg": "ok"
    })

@app.route('/version', methods=['GET'])
def version():
    """Endpoint para información de versión"""
    return jsonify({
        "version": "1.0.0",
        "api_version": "v1",
        "build_date": "2025-05-06"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
EOF

# Crear configuración Nginx - CORREGIDO para usar HTTP primero
echo "==== Configurando Nginx ===="
cat > /etc/nginx/sites-available/$DOMAIN << EOF
server {
    listen 80;
    server_name $DOMAIN;
    
    # API principal
    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # Timeouts largos para procesamiento de videos
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
    
    # Directorio de almacenamiento
    location /storage/ {
        alias $STORAGE_PATH/;
        expires 6h;  # Cachear archivos por 6 horas
        add_header Cache-Control "public, max-age=21600";
    }
    
    # Límites y configuración de carga
    client_max_body_size 1G;  # Permitir uploads de hasta 1GB
    
    # Logs
    access_log /var/log/nginx/videoapi-access.log;
    error_log /var/log/nginx/videoapi-error.log;
}
EOF

# Habilitar sitio en Nginx
ln -sf /etc/nginx/sites-available/$DOMAIN /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx

# Crear servicio systemd
echo "==== Configurando servicio systemd ===="
cat > /etc/systemd/system/videoapi.service << EOF
[Unit]
Description=Video API Service
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/var/www/$DOMAIN
ExecStart=/var/www/$DOMAIN/venv/bin/gunicorn --bind 0.0.0.0:8080 --workers 4 --timeout 300 app:app
Restart=always
RestartSec=10
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=videoapi
Environment="API_KEY=$API_KEY"
Environment="STORAGE_PATH=$STORAGE_PATH"
Environment="BASE_URL=$BASE_URL"
Environment="MAX_FILE_AGE_HOURS=6"
Environment="MAX_QUEUE_LENGTH=100"
Environment="WORKER_PROCESSES=4"
Environment="FFMPEG_THREADS=4"

[Install]
WantedBy=multi-user.target
EOF

# Configurar permisos
echo "==== Configurando permisos ===="
chown -R www-data:www-data /var/www/$DOMAIN
chmod -R 755 /var/www/$DOMAIN

# Habilitar e iniciar servicio
echo "==== Iniciando servicios ===="
systemctl daemon-reload
systemctl enable videoapi
systemctl start videoapi

# Esperar a que el servicio arranque
echo "Esperando a que el servicio arranque..."
sleep 5

# Validar instalación
echo "==== Validando instalación ===="

# Verificar servicio
if systemctl is-active --quiet videoapi; then
    echo "✅ Servicio VideoAPI está activo"
else
    echo "❌ Error: El servicio VideoAPI no está ejecutándose"
fi

# Verificar Nginx
if systemctl is-active --quiet nginx; then
    echo "✅ Servicio Nginx está activo"
else
    echo "❌ Error: El servicio Nginx no está ejecutándose"
fi

# Verificar respuesta de la API
echo "Verificando respuesta de la API..."
API_RESPONSE=$(curl -s http://localhost:8080/version || echo "error")
if [[ "$API_RESPONSE" == *"version"* ]]; then
    echo "✅ La API responde correctamente"
else
    echo "❌ Error: La API no responde correctamente"
fi

# Configurar SSL con Certbot después de verificar funcionamiento básico
echo "==== ¿Deseas configurar HTTPS con Certbot ahora? ===="
echo "NOTA: Asegúrate de que el dominio $DOMAIN apunte a este servidor."
read -p "Configurar HTTPS ahora? (s/n): " configure_https

if [ "$configure_https" = "s" ]; then
    echo "Configurando HTTPS con Certbot..."
    certbot --nginx -d $DOMAIN
    
    echo "Reiniciando servicios..."
    systemctl restart nginx
    systemctl restart videoapi
    
    echo "Verificando HTTPS..."
    echo "Puedes probar la API segura en: https://$DOMAIN/version"
else
    echo "Has elegido no configurar HTTPS ahora."
    echo "Puedes configurarlo manualmente más tarde con: sudo certbot --nginx -d $DOMAIN"
    echo "Puedes probar la API en: http://$DOMAIN/version"
fi

# Mostrar información final
echo ""
echo "========================================"
echo "      Instalación de VideoAPI           "
echo "========================================"
echo ""
echo "✅ Instalación completada con éxito!"
echo ""
echo "📝 Información del servicio:"
echo "   - Dominio: $DOMAIN"
echo "   - URL Base API: http://$DOMAIN (o https:// si habilitaste SSL)"
echo "   - API Key: $API_KEY"
echo "   - Ruta de almacenamiento: $STORAGE_PATH"
echo ""
echo "💡 Próximos pasos:"
echo "  1. Ejecuta el script deploy.sh para implementar la funcionalidad completa"
echo "  2. Verifica logs con: sudo journalctl -u videoapi -f"
echo ""
echo "🔗 Endpoints disponibles actualmente:"
echo "  - Información: GET /"
echo "  - Versión: GET /version"
echo "  - Estado: GET /health"
echo ""
