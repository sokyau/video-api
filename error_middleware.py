"""
error_middleware.py - Middleware for request handling and error management
"""

import uuid
import time
import logging
import traceback
from flask import request, g, Flask
from werkzeug.exceptions import HTTPException
import config

logger = logging.getLogger(__name__)

class ErrorHandlingMiddleware:
    """
    Middleware for handling errors and request logging
    """
    
    def __init__(self, app):
        self.app = app
        
        # Wrap the app's dispatch_request method
        app.wsgi_app = self._middleware(app.wsgi_app)
        
        # Register error handlers
        from errors import register_error_handlers
        register_error_handlers(app)
    
    def _middleware(self, wsgi_app):
        """WSGI middleware wrapper"""
        
        def _wrapped_app(environ, start_response):
            # Generate request ID and start time
            request_id = str(uuid.uuid4())
            start_time = time.time()
            
            # Add request ID to request object
            environ['HTTP_X_REQUEST_ID'] = request_id
            
            # Process the request and catch any uncaught exceptions
            try:
                return wsgi_app(environ, start_response)
            except Exception as e:
                # Log the exception
                logger.exception(f"Unhandled exception in request {request_id}")
                
                # Create a simple error response
                status = '500 Internal Server Error'
                headers = [('Content-Type', 'application/json')]
                
                error_response = json.dumps({
                    'status': 'error',
                    'message': 'An unexpected error occurred',
                    'request_id': request_id
                }).encode('utf-8')
                
                start_response(status, headers)
                return [error_response]
            finally:
                # Log request completion
                duration = time.time() - start_time
                
                # Skip health checks for reduced log noise
                path = environ.get('PATH_INFO', '')
                if not (path == '/health' and environ.get('REQUEST_METHOD') == 'GET'):
                    logger.info(
                        f"Request {request_id} completed in {duration:.3f}s: "
                        f"{environ.get('REQUEST_METHOD')} {path}"
                    )
        
        return _wrapped_app

def setup_request_handlers(app):
    """Set up before/after request handlers for logging and tracking"""
    
    @app.before_request
    def before_request():
        """Actions to take before each request"""
        # Store request start time and ID
        g.start_time = time.time()
        g.request_id = request.headers.get('X-Request-ID') or str(uuid.uuid4())
        
        # Attach request ID to the request object for logging
        request.id = g.request_id
        
        # Log request details
        logger.info(
            f"Request {g.request_id} started: {request.method} {request.path} "
            f"from {request.remote_addr}"
        )
    
    @app.after_request
    def after_request(response):
        """Actions to take after each request"""
        # Calculate request duration
        duration = time.time() - g.get('start_time', time.time())
        
        # Add request ID to response headers
        response.headers['X-Request-ID'] = g.get('request_id', 'unknown')
        
        # Log response details (skip for health checks to reduce noise)
        if not (request.path == '/health' and request.method == 'GET'):
            logger.info(
                f"Request {g.get('request_id', 'unknown')} completed in {duration:.3f}s "
                f"with status {response.status_code}"
            )
        
        return response
    
    @app.teardown_request
    def teardown_request(exception):
        """Actions to take after request is torn down (even if exception)"""
        if exception:
            # Log exception if one occurred
            logger.exception(
                f"Exception in request {g.get('request_id', 'unknown')}: {str(exception)}"
            )

def init_app(app):
    """Initialize error handling for the app"""
    # Apply middleware
    ErrorHandlingMiddleware(app)
    
    # Set up request handlers
    setup_request_handlers(app)
