 - Centralized error handling for VideoAPI

This module defines custom exceptions, error handlers, and utilities
for consistent error management across the application.
"""

import logging
import traceback
import json
import time
import sys
from typing import Dict, Any, Optional, Tuple, List, Union
from flask import jsonify, Response, request, current_app
import requests

logger = logging.getLogger(__name__)

# Base exception classes
class VideoAPIError(Exception):
    """Base exception for all VideoAPI errors"""
    status_code = 500
    error_code = "internal_error"
    
    def __init__(self, message: str = None, status_code: Optional[int] = None, 
                 details: Optional[Dict[str, Any]] = None, 
                 error_code: Optional[str] = None):
        self.message = message or "An unexpected error occurred"
        self.status_code = status_code or self.__class__.status_code
        self.details = details or {}
        self.error_code = error_code or self.__class__.error_code
        self.timestamp = time.time()
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API response"""
        error_dict = {
            "status": "error",
            "error": self.error_code,
            "message": self.message,
            "timestamp": self.timestamp,
        }
        
        # Add details if they exist
        if self.details:
            error_dict["details"] = self.details
            
        # Add request ID if available
        if hasattr(request, 'id'):
            error_dict["request_id"] = request.id
            
        return error_dict
    
    def get_response(self) -> Response:
        """Convert exception to Flask response"""
        return jsonify(self.to_dict()), self.status_code

# HTTP error classes
class BadRequestError(VideoAPIError):
    """Exception for invalid request data"""
    status_code = 400
    error_code = "bad_request"

class AuthenticationError(VideoAPIError):
    """Exception for authentication failures"""
    status_code = 401
    error_code = "authentication_error"

class AuthorizationError(VideoAPIError):
    """Exception for authorization failures"""
    status_code = 403
    error_code = "forbidden"

class NotFoundError(VideoAPIError):
    """Exception for resource not found"""
    status_code = 404
    error_code = "not_found"

class ConflictError(VideoAPIError):
    """Exception for resource conflicts"""
    status_code = 409
    error_code = "conflict"

class RateLimitError(VideoAPIError):
    """Exception for rate limiting"""
    status_code = 429
    error_code = "rate_limit_exceeded"

class ServiceUnavailableError(VideoAPIError):
    """Exception for service unavailability"""
    status_code = 503
    error_code = "service_unavailable"

# Validation errors
class ValidationError(BadRequestError):
    """Exception for data validation failures"""
    error_code = "validation_error"
    
    @classmethod
    def from_validation_errors(cls, errors: Dict[str, List[str]]) -> 'ValidationError':
        """Create from validation error dict"""
        return cls(
            message="Validation error",
            details={"validation_errors": errors}
        )

# Processing errors
class ProcessingError(VideoAPIError):
    """Base exception for media processing errors"""
    status_code = 500
    error_code = "processing_error"

class FFmpegError(ProcessingError):
    """Exception for FFmpeg failures"""
    error_code = "ffmpeg_error"
    
    @classmethod
    def from_ffmpeg_error(cls, stderr: str, cmd: List[str] = None) -> 'FFmpegError':
        """Create from FFmpeg error output"""
        # Extract meaningful message from FFmpeg error
        message = "FFmpeg command failed"
        
        # Common error patterns to look for
        patterns = [
            "No such file or directory",
            "Invalid data found when processing input",
            "Conversion failed",
            "Error while decoding",
            "Cannot find a matching stream",
            "Unknown encoder",
            "not found"
        ]
        
        for pattern in patterns:
            if pattern in stderr:
                message = f"FFmpeg error: {pattern}"
                break
        
        details = {"ffmpeg_error": stderr[:500]}  # Limit length
        if cmd:
            details["command"] = " ".join(cmd)
            
        return cls(message=message, details=details)

class StorageError(VideoAPIError):
    """Exception for storage-related errors"""
    error_code = "storage_error"

class NetworkError(VideoAPIError):
    """Exception for network-related errors"""
    error_code = "network_error"
    
    @classmethod
    def from_request_exception(cls, exception: requests.RequestException) -> 'NetworkError':
        """Create from requests exception"""
        if isinstance(exception, requests.Timeout):
            return cls(
                message="Network timeout occurred",
                details={"url": getattr(exception.request, 'url', None)},
                error_code="network_timeout"
            )
        elif isinstance(exception, requests.ConnectionError):
            return cls(
                message="Connection error occurred",
                details={"url": getattr(exception.request, 'url', None)},
                error_code="connection_error"
            )
        else:
            return cls(
                message="Network error occurred",
                details={"url": getattr(exception.request, 'url', None)},
                error_code="general_network_error"
            )

class QueueError(VideoAPIError):
    """Exception for queue-related errors"""
    error_code = "queue_error"

# Registration with Flask
def register_error_handlers(app):
    """Register error handlers with Flask app"""
    
    @app.errorhandler(VideoAPIError)
    def handle_api_error(error):
        """Handle all VideoAPI errors"""
        return error.get_response()
    
    @app.errorhandler(404)
    def handle_not_found(error):
        """Handle 404 errors"""
        return NotFoundError("Resource not found").get_response()
    
    @app.errorhandler(405)
    def handle_method_not_allowed(error):
        """Handle 405 errors"""
        return BadRequestError(
            message=f"Method {request.method} not allowed for this endpoint",
            error_code="method_not_allowed"
        ).get_response()
    
    @app.errorhandler(500)
    def handle_internal_error(error):
        """Handle 500 errors"""
        logger.exception("Unhandled exception occurred")
        return VideoAPIError("An unexpected error occurred").get_response()

# Utility functions
def log_exception(exc: Exception, context: Optional[Dict[str, Any]] = None):
    """Log an exception with additional context"""
    error_id = f"err_{int(time.time())}_{id(exc):x}"
    
    # Combine context with basic error info
    log_context = {
        "error_id": error_id,
        "error_type": exc.__class__.__name__,
        "error_message": str(exc)
    }
    
    if context:
        log_context.update(context)
    
    # Log the error with context
    logger.error(
        f"Exception {error_id}: {exc.__class__.__name__}: {str(exc)}",
        extra=log_context,
        exc_info=True
    )
    
    return error_id

def classify_exception(exc: Exception) -> Tuple[str, bool]:
    """
    Classify an exception as recoverable or not
    
    Args:
        exc: The exception to classify
        
    Returns:
        tuple: (error_type, is_recoverable)
    """
    if isinstance(exc, VideoAPIError):
        # Our custom exceptions already have classification
        return exc.error_code, exc.status_code < 500
    
    # Network and IO errors are often recoverable
    if isinstance(exc, (requests.Timeout, requests.ConnectionError, 
                       ConnectionError, TimeoutError, IOError)):
        return "network_error", True
    
    # Permission errors are not recoverable without intervention
    if isinstance(exc, (PermissionError, OSError)):
        return "system_error", False
    
    # JSON and data format errors are not recoverable
    if isinstance(exc, (json.JSONDecodeError, TypeError, ValueError)):
        return "data_error", False
    
    # Default to unclassified and not recoverable
    return "unknown_error", False

def capture_exception(exc: Exception, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Capture and log an exception, and potentially send to monitoring service
    
    Args:
        exc: Exception to capture
        context: Additional context to include
        
    Returns:
        str: Error ID for reference
    """
    error_id = log_exception(exc, context)
    
    # Here you would integrate with your error monitoring service
    # For example, Sentry integration would go here
    # if sentry_sdk is available:
    try:
        import sentry_sdk
        if sentry_sdk.Hub.current.client:
            with sentry_sdk.push_scope() as scope:
                if context:
                    for key, value in context.items():
                        scope.set_extra(key, value)
                scope.set_tag("error_id", error_id)
                sentry_sdk.capture_exception(exc)
    except ImportError:
        # Sentry not available, just log
        pass
    
    return error_id
