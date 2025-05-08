"""
app.py - Main entry point for the Video Processing API

This application provides a Flask-based REST API for video processing
optimized for automated YouTube content creation through n8n.
"""

from flask import Flask, jsonify, request, send_from_directory, Response, g
import logging
import os
import threading
import time
import shutil
import json
import platform
import psutil
import uuid
import traceback
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.exceptions import HTTPException

# Internal imports
from services.cleanup_service import CleanupService
from app_utils import (
    start_queue_processors, 
    cleanup_completed_tasks, 
    get_queue_stats,
    check_queue_health
)
from version import get_version_info
import config

# Setup directories
def setup_directories():
    """Ensure required directories exist"""
    directories = [
        'logs',
        config.STORAGE_PATH,
        config.TEMP_DIR,
        os.path.join(config.TEMP_DIR, 'cache'),
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Configure logging
def configure_logging():
    """Configure application logging with rotation and formatting"""
    log_directory = 'logs'
    os.makedirs(log_directory, exist_ok=True)
    
    # Determine log level from config
    log_level = getattr(logging, config.LOG_LEVEL, logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add handler for main log file with size-based rotation
    main_log_path = os.path.join(log_directory, 'videoapi.log')
    file_handler = RotatingFileHandler(
        main_log_path,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    
    # Add handler for error log with daily rotation
    error_log_path = os.path.join(log_directory, 'error.log')
    error_handler = TimedRotatingFileHandler(
        error_log_path,
        when='midnight',
        interval=1,  # Daily rotation
        backupCount=30  # Keep 30 days
    )
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)
    
    # Add handlers to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    
    # Create API logger
    logger = logging.getLogger('videoapi')
    logger.setLevel(log_level)
    
    return logger

# Initialize logger
setup_directories()
logger = configure_logging()

# Optional Sentry integration for error monitoring
if hasattr(config, 'SENTRY_DSN') and config.SENTRY_DSN:
    try:
        import sentry_sdk
        from sentry_sdk.integrations.flask import FlaskIntegration
        
        sentry_sdk.init(
            dsn=config.SENTRY_DSN,
            environment=getattr(config, 'ENVIRONMENT', 'production'),
            integrations=[FlaskIntegration()],
            traces_sample_rate=0.2,
            # Set the error sample rate, adjust based on traffic
            # Lower for high-traffic applications
            sample_rate=0.5,
            
            # Configure contexts you want attached to errors
            with_locals=True,
            request_bodies='small',
            
            # Set log levels for events
            before_send=lambda event, hint: event if event.get('level') != 'debug' else None
        )
        logger.info("Sentry monitoring initialized")
    except ImportError:
        logger.warning("Sentry SDK not installed. Error monitoring disabled.")
        pass

# Import error handling module
try:
    from errors import register_error_handlers, VideoAPIError, capture_exception
    from error_middleware import ErrorHandlingMiddleware
    logger.info("Error handling modules loaded")
except ImportError:
    logger.warning("Error handling modules not found. Using standard error handling.")
    # Define fallback to maintain compatibility
    def register_error_handlers(app):
        pass
    
    class VideoAPIError(Exception):
        def get_response(self):
            return jsonify({"status": "error", "message": str(self)}), 500
    
    def capture_exception(exc, context=None):
        logger.exception(f"Error: {str(exc)}")
        return str(uuid.uuid4())
    
    class ErrorHandlingMiddleware:
        def __init__(self, app):
            pass

# Create Flask application
app = Flask(__name__)

# Configure for working behind proxy
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# Initialize error handling middleware if available
if 'ErrorHandlingMiddleware' in globals():
    ErrorHandlingMiddleware(app)

# Register before/after request handlers
@app.before_request
def before_request():
    """Setup request context and logging"""
    request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
    g.request_id = request_id
    g.start_time = time.time()
    
    # Skip logging for health checks to reduce noise
    if request.path != '/health':
        logger.info(f"Request {request_id}: {request.method} {request.path}")

@app.after_request
def after_request(response):
    """Add response headers and log completion"""
    # Add request ID to response
    response.headers['X-Request-ID'] = g.get('request_id', 'unknown')
    
    # Add security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    
    # Calculate duration
    duration = time.time() - g.get('start_time', time.time())
    
    # Skip logging for health checks
    if request.path != '/health':
        logger.info(f"Request {g.get('request_id', 'unknown')} completed in {duration:.3f}s with status {response.status_code}")
    
    return response

# Variables globales para la aplicación
startup_time = time.time()
worker_count = config.WORKER_PROCESSES
cleanup_interval = getattr(config, 'CLEANUP_INTERVAL_MINUTES', 30)

# Inicializar servicios
cleanup_service = CleanupService(interval_minutes=cleanup_interval)

# Load blueprints dynamically
def register_blueprints():
    """Register all blueprint modules from the routes package"""
    try:
        # Core routes
        from routes.v1.video.meme_overlay import v1_video_meme_overlay_bp
        from routes.v1.video.caption_video import v1_video_caption_video_bp
        from routes.v1.video.concatenate import v1_video_concatenate_bp
        from routes.v1.video.animated_text import v1_video_animated_text_bp
        from routes.v1.media.transform.media_to_mp3 import v1_media_transform_media_to_mp3_bp
        from routes.v1.media.media_transcribe import v1_media_media_transcribe_bp
        from routes.v1.ffmpeg.ffmpeg_compose import v1_ffmpeg_compose_bp
        from routes.v1.image.transform.image_to_video import v1_image_transform_image_to_video_bp
        
        # Register all loaded blueprints
        blueprints = [
            v1_video_meme_overlay_bp,
            v1_video_caption_video_bp,
            v1_video_concatenate_bp,
            v1_video_animated_text_bp,
            v1_media_transform_media_to_mp3_bp,
            v1_media_media_transcribe_bp,
            v1_ffmpeg_compose_bp,
            v1_image_transform_image_to_video_bp,
        ]
        
        for blueprint in blueprints:
            app.register_blueprint(blueprint)
            logger.info(f"Registered blueprint: {blueprint.name}")
        
        return True
    except ImportError as e:
        logger.error(f"Error registering blueprints: {str(e)}")
        return False

# Register blueprints
if not register_blueprints():
    logger.critical("Failed to register all blueprints. Application may not function correctly.")

# Register error handlers
register_error_handlers(app)

# CORS configuration
@app.after_request
def add_cors_headers(response):
    """Add CORS headers to allow cross-origin requests"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-API-Key'
    return response

# Primary route handlers
@app.route('/', methods=['GET'])
def index():
    """API information endpoint"""
    endpoints = {
        "video": [
            "/v1/video/meme_overlay",
            "/v1/video/caption_video",
            "/v1/video/concatenate",
            "/v1/video/animated_text"
        ],
        "media": [
            "/v1/media/transform/media_to_mp3",
            "/v1/media/media_transcribe"
        ],
        "ffmpeg": [
            "/v1/ffmpeg/ffmpeg_compose"
        ],
        "image": [
            "/v1/image/transform/image_to_video"
        ]
    }
    
    return jsonify({
        "service": "Video Processing API",
        "status": "operational",
        "endpoints": endpoints,
        **get_version_info()
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    health_status = {
        "status": "healthy",
        "checks": {},
        "uptime": time.time() - startup_time,
        "uptime_formatted": format_time_delta(time.time() - startup_time)
    }
    
    # Check storage access
    try:
        storage_ok = os.access(config.STORAGE_PATH, os.W_OK)
        test_file_path = os.path.join(config.STORAGE_PATH, '.health_check_test')
        with open(test_file_path, 'w') as f:
            f.write('test')
        os.remove(test_file_path)
        health_status["checks"]["storage"] = {"status": "ok"}
    except Exception as e:
        storage_ok = False
        health_status["checks"]["storage"] = {
            "status": "error",
            "message": str(e)
        }
    
    # Check FFmpeg availability
    try:
        import subprocess
        ffmpeg_result = subprocess.run(['ffmpeg', '-version'], 
                                      capture_output=True, 
                                      text=True,
                                      check=True)
        ffmpeg_ok = ffmpeg_result.returncode == 0
        ffmpeg_version = ffmpeg_result.stdout.split('\n')[0] if ffmpeg_result.stdout else "Unknown"
        health_status["checks"]["ffmpeg"] = {
            "status": "ok",
            "version": ffmpeg_version
        }
    except Exception as e:
        ffmpeg_ok = False
        health_status["checks"]["ffmpeg"] = {
            "status": "error",
            "message": str(e)
        }
    
    # Check FFprobe availability
    try:
        ffprobe_result = subprocess.run(['ffprobe', '-version'], 
                                      capture_output=True, 
                                      text=True,
                                      check=True)
        ffprobe_ok = ffprobe_result.returncode == 0
        ffprobe_version = ffprobe_result.stdout.split('\n')[0] if ffprobe_result.stdout else "Unknown"
        health_status["checks"]["ffprobe"] = {
            "status": "ok",
            "version": ffprobe_version
        }
    except Exception as e:
        ffprobe_ok = False
        health_status["checks"]["ffprobe"] = {
            "status": "error",
            "message": str(e)
        }
    
    # Check disk space
    try:
        disk_usage = shutil.disk_usage(config.STORAGE_PATH)
        disk_total_gb = disk_usage.total / (1024**3)
        disk_free_gb = disk_usage.free / (1024**3)
        disk_used_percent = (disk_usage.used / disk_usage.total) * 100
        
        disk_status = "ok"
        if disk_used_percent > 90:
            disk_status = "warning"
        if disk_used_percent > 95:
            disk_status = "error"
        
        health_status["checks"]["disk"] = {
            "status": disk_status,
            "total_gb": round(disk_total_gb, 2),
            "free_gb": round(disk_free_gb, 2),
            "used_percent": round(disk_used_percent, 2)
        }
        disk_ok = disk_status != "error"
    except Exception as e:
        disk_ok = False
        health_status["checks"]["disk"] = {
            "status": "error",
            "message": str(e)
        }
    
    # Check queue processing
    try:
        queue_health = check_queue_health()
        health_status["checks"]["queue"] = queue_health
        queue_ok = queue_health.get("status") == "healthy"
    except Exception as e:
        queue_ok = False
        health_status["checks"]["queue"] = {
            "status": "error",
            "message": str(e)
        }
    
    # Determine overall health status
    if all([storage_ok, ffmpeg_ok, ffprobe_ok, disk_ok, queue_ok]):
        health_status["status"] = "healthy"
        status_code = 200
    else:
        if any([not disk_ok, not storage_ok]):
            # Critical problems
            health_status["status"] = "critical"
            status_code = 503  # Service Unavailable
        else:
            # Non-critical problems
            health_status["status"] = "degraded"
            status_code = 200  # Still return 200 to avoid triggering alerts
    
    return jsonify(health_status), status_code

@app.route('/metrics', methods=['GET'])
def metrics():
    """System metrics endpoint for monitoring"""
    from services.authentication import authenticate
    
    # Only allow authenticated access to metrics
    auth_result = authenticate(lambda: None)()
    if isinstance(auth_result, tuple) and auth_result[1] != 200:
        return auth_result
    
    # Get system information
    cpu_usage = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk_usage = shutil.disk_usage(config.STORAGE_PATH)
    disk_total_gb = disk_usage.total / (1024**3)
    disk_free_gb = disk_usage.free / (1024**3)
    disk_used_percent = (disk_usage.used / disk_usage.total) * 100
    
    # Get queue statistics
    queue_stats = get_queue_stats()
    
    # Count files in storage
    storage_file_count = 0
    storage_size_bytes = 0
    try:
        for root, _, files in os.walk(config.STORAGE_PATH):
            for file in files:
                if not file.endswith('.meta'):
                    file_path = os.path.join(root, file)
                    storage_file_count += 1
                    storage_size_bytes += os.path.getsize(file_path)
    except Exception as e:
        logger.error(f"Error counting storage files: {str(e)}")
    
    # Compile metrics
    metrics_data = {
        "system": {
            "cpu_usage_percent": cpu_usage,
            "memory_total_gb": memory.total / (1024**3),
            "memory_available_gb": memory.available / (1024**3),
            "memory_used_percent": memory.percent,
            "disk_total_gb": round(disk_total_gb, 2),
            "disk_free_gb": round(disk_free_gb, 2),
            "disk_used_percent": round(disk_used_percent, 2),
            "host": platform.node(),
            "platform": platform.platform(),
            "uptime_seconds": time.time() - startup_time,
            "uptime_formatted": format_time_delta(time.time() - startup_time)
        },
        "storage": {
            "file_count": storage_file_count,
            "size_mb": round(storage_size_bytes / (1024**2), 2)
        },
        "queue": queue_stats,
        "config": {
            "worker_processes": config.WORKER_PROCESSES,
            "max_queue_length": config.MAX_QUEUE_LENGTH,
            "max_file_age_hours": config.MAX_FILE_AGE_HOURS,
            "ffmpeg_threads": config.FFMPEG_THREADS
        },
        "version": get_version_info(),
        "timestamp": datetime.now().isoformat()
    }
    
    return jsonify(metrics_data)

@app.route('/version', methods=['GET'])
def version():
    """Version information endpoint"""
    version_info = get_version_info()
    version_info["python_version"] = platform.python_version()
    version_info["platform"] = platform.platform()
    
    return jsonify(version_info)

@app.route('/storage/<path:filename>', methods=['GET'])
def serve_file(filename):
    """Serve files from storage directory"""
    # Security check to prevent path traversal
    if '..' in filename or filename.startswith('/'):
        return jsonify({"error": "Unauthorized access"}), 403
    
    # Check if file exists
    file_path = os.path.join(config.STORAGE_PATH, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    
    # Determine MIME type
    content_type = None
    ext = os.path.splitext(filename)[1].lower()
    if ext in ['.mp4', '.webm']:
        content_type = f'video/{ext[1:]}'
    elif ext in ['.mp3', '.wav', '.ogg']:
        content_type = f'audio/{ext[1:]}'
    elif ext in ['.jpg', '.jpeg']:
        content_type = 'image/jpeg'
    elif ext in ['.png']:
        content_type = 'image/png'
    
    return send_from_directory(
        config.STORAGE_PATH, 
        filename, 
        as_attachment=False,
        mimetype=content_type
    )

@app.route('/maintenance/cleanup', methods=['POST'])
def trigger_cleanup():
    """Endpoint to manually trigger file cleanup"""
    from services.authentication import authenticate
    
    # Only allow authenticated access to cleanup
    auth_result = authenticate(lambda: None)()
    if isinstance(auth_result, tuple) and auth_result[1] != 200:
        return auth_result
    
    try:
        files_removed, bytes_freed = cleanup_service.run_now()
        return jsonify({
            "status": "success",
            "files_removed": files_removed,
            "space_freed_mb": bytes_freed / 1024 / 1024
        })
    except Exception as e:
        error_id = capture_exception(e, {"endpoint": "/maintenance/cleanup"})
        return jsonify({
            "status": "error",
            "error": str(e),
            "error_id": error_id
        }), 500

@app.route('/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get status of a specific job"""
    from services.authentication import authenticate
    from app_utils import get_job_status as utils_get_job_status
    
    # Only allow authenticated access to job status
    auth_result = authenticate(lambda: None)()
    if isinstance(auth_result, tuple) and auth_result[1] != 200:
        return auth_result
    
    status = utils_get_job_status(job_id)
    if status:
        return jsonify({
            "status": "success",
            "job_id": job_id,
            "job_status": status
        })
    else:
        return jsonify({
            "status": "error",
            "job_id": job_id,
            "error": "Job not found"
        }), 404

# Error handlers
@app.errorhandler(404)
def not_found(error):
    """404 error handler"""
    return jsonify({
        "status": "error",
        "error": "Not Found",
        "message": "The requested endpoint does not exist.",
        "available_endpoints": {
            "video": [
                "/v1/video/meme_overlay",
                "/v1/video/caption_video",
                "/v1/video/concatenate",
                "/v1/video/animated_text"
            ],
            "media": [
                "/v1/media/transform/media_to_mp3",
                "/v1/media/media_transcribe"
            ],
            "ffmpeg": [
                "/v1/ffmpeg/ffmpeg_compose"
            ],
            "image": [
                "/v1/image/transform/image_to_video"
            ]
        }
    }), 404

@app.errorhandler(400)
def bad_request(error):
    """400 error handler"""
    return jsonify({
        "status": "error",
        "error": "Bad Request",
        "message": str(error),
        "request_id": getattr(g, 'request_id', None)
    }), 400

@app.errorhandler(401)
def unauthorized(error):
    """401 error handler"""
    return jsonify({
        "status": "error",
        "error": "Unauthorized",
        "message": "Authentication required. Please provide a valid API key (X-API-Key header).",
        "request_id": getattr(g, 'request_id', None)
    }), 401

@app.errorhandler(500)
def server_error(error):
    """500 error handler"""
    error_id = capture_exception(error, {
        "path": request.path, 
        "method": request.method
    })
    
    return jsonify({
        "status": "error",
        "error": "Internal Server Error",
        "message": "An unexpected error occurred on the server",
        "error_id": error_id,
        "request_id": getattr(g, 'request_id', None)
    }), 500

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all uncaught exceptions"""
    error_id = capture_exception(e, {
        "path": request.path, 
        "method": request.method
    })
    
    # Handle custom API errors
    if isinstance(e, VideoAPIError):
        return e.get_response()
    
    # Handle HTTP exceptions
    if isinstance(e, HTTPException):
        response = e.get_response()
        response.data = json.dumps({
            "status": "error",
            "error": e.name,
            "message": e.description,
            "error_id": error_id,
            "request_id": getattr(g, 'request_id', None)
        })
        response.content_type = "application/json"
        return response
    
    # Generic server error for other exceptions
    logger.error(f"Unhandled exception: {e}", exc_info=True)
    return jsonify({
        "status": "error",
        "error": "internal_server_error",
        "message": "An unexpected error occurred",
        "error_id": error_id,
        "request_id": getattr(g, 'request_id', None)
    }), 500

def format_time_delta(seconds):
    """Format seconds into readable time format"""
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    parts = []
    if days > 0:
        parts.append(f"{int(days)} days")
    if hours > 0 or days > 0:
        parts.append(f"{int(hours)} hours")
    if minutes > 0 or hours > 0 or days > 0:
        parts.append(f"{int(minutes)} minutes")
    parts.append(f"{int(seconds)} seconds")
    
    return ", ".join(parts)

def initialize_services():
    """Initialize required services"""
    try:
        # Ensure required directories exist
        os.makedirs(config.STORAGE_PATH, exist_ok=True)
        os.makedirs(config.TEMP_DIR, exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Log system information
        logger.info(f"System: {platform.system()} {platform.release()}")
        logger.info(f"Python: {platform.python_version()}")
        cpu_count = os.cpu_count() or 0
        memory = psutil.virtual_memory()
        logger.info(f"Resources: {cpu_count} CPUs, {memory.total / (1024**3):.1f} GB RAM")
        
        # Start queue processors
        start_queue_processors(num_workers=worker_count)
        logger.info(f"Started {worker_count} queue processors")
        
        # Start cleanup service
        cleanup_service.start()
        logger.info(f"Cleanup service started (interval: {cleanup_interval} minutes)")
        
        # Start periodic task thread
        threading.Thread(target=periodic_tasks, daemon=True).start()
        logger.info("Periodic tasks started")
        
        # Perform initial cleanup
        files_removed, bytes_freed = cleanup_service.run_now()
        if files_removed > 0:
            logger.info(f"Initial cleanup: removed {files_removed} files ({bytes_freed/1024/1024:.2f} MB)")
        
        logger.info(f"VideoAPI v{get_version_info()['version']} initialized successfully")
        return True
        
    except Exception as e:
        logger.critical(f"Error initializing services: {str(e)}", exc_info=True)
        return False

def periodic_tasks():
    """Execute periodic background tasks"""
    while True:
        try:
            # Clean up completed tasks info
            tasks_cleaned = cleanup_completed_tasks(max_age_hours=config.MAX_FILE_AGE_HOURS)
            if tasks_cleaned > 0:
                logger.info(f"Periodic cleanup: {tasks_cleaned} tasks removed from registry")
            
            # Monitor resource usage
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            if cpu_usage > 90 or memory.percent > 90:
                logger.warning(f"High resource usage - CPU: {cpu_usage}%, Memory: {memory.percent}%")
            
            # Cleanup logs
            cleanup_old_logs()
            
        except Exception as e:
            logger.error(f"Error in periodic tasks: {str(e)}", exc_info=True)
        
        # Wait 15 minutes before next execution
        time.sleep(15 * 60)

def cleanup_old_logs():
    """Clean up old log files"""
    try:
        # Find log files older than X days
        log_dir = 'logs'
        max_age_days = 14  # Keep logs for 14 days
        now = time.time()
        max_age_seconds = max_age_days * 24 * 3600
        
        for filename in os.listdir(log_dir):
            file_path = os.path.join(log_dir, filename)
            if os.path.isfile(file_path) and filename.endswith('.log'):
                # Skip current log files
                if filename in ['videoapi.log', 'error.log']:
                    continue
                    
                # Check file age
                file_age = now - os.path.getmtime(file_path)
                if file_age > max_age_seconds:
                    os.remove(file_path)
                    logger.debug(f"Removed old log file: {filename}")
    except Exception as e:
        logger.error(f"Error cleaning up logs: {str(e)}")

# Flask before first request handler
@app.before_first_request
def before_first_request():
    """Initialize services before first request"""
    initialize_services()

# Signal handlers for graceful shutdown
def graceful_shutdown(signal_num, frame):
    """Perform clean shutdown when signal received"""
    logger.info(f"Received signal {signal_num}, performing graceful shutdown...")
    
    # Stop cleanup service
    if 'cleanup_service' in globals():
        cleanup_service.stop()
        logger.info("Cleanup service stopped")
    
    # Stop queue processors
    from app_utils import stop_queue_processors
    stop_queue_processors()
    logger.info("Queue processors stopped")
    
    # Clean up temporary files
    try:
        temp_files = [f for f in os.listdir(config.TEMP_DIR) if os.path.isfile(os.path.join(config.TEMP_DIR, f))]
        for temp_file in temp_files:
            os.remove(os.path.join(config.TEMP_DIR, temp_file))
        logger.info(f"Cleaned up {len(temp_files)} temporary files")
    except Exception as e:
        logger.error(f"Error cleaning temporary files: {str(e)}")
    
    logger.info("Shutdown completed")
    os._exit(0)

# Register signal handlers
try:
    import signal
    signal.signal(signal.SIGTERM, graceful_shutdown)
    signal.signal(signal.SIGINT, graceful_shutdown)
except (ImportError, AttributeError):
    # Windows doesn't have SIGTERM
    pass

# Add Swagger UI
try:
    from flask_swagger_ui import get_swaggerui_blueprint
    
    SWAGGER_URL = '/api/docs'
    API_URL = '/static/swagger.json'
    
    swaggerui_blueprint = get_swaggerui_blueprint(
        SWAGGER_URL,
        API_URL,
        config={
            'app_name': "Video Processing API"
        }
    )
    app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)
    logger.info("Swagger UI initialized")
except ImportError:
    logger.warning("flask-swagger-ui not installed. API documentation UI disabled.")

# Run the application
if __name__ == '__main__':
    # Initialize services immediately in development mode
    if not initialize_services():
        logger.critical("Failed to initialize services. Exiting.")
        exit(1)
    
    # Run in development mode
    debug_mode = config.LOG_LEVEL == 'DEBUG'
    app.run(host='0.0.0.0', port=8080, debug=debug_mode)
