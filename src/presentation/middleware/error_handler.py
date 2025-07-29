"""
Error Handler Middleware

"""
import logging
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.shared.exceptions import (
    DomainException,
    DocumentError,
    ChatError,
    FileStorageError,
    CitationError,
    ConfigurationError
)

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Middleware for handling global errors in the application."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and handle any errors that occur.
        
        Args:
            request: The incoming HTTP request
            call_next: The next middleware or route handler
            
        Returns:
            Response: HTTP response, potentially an error response
        """
        try:
            response = await call_next(request)
            return response
            
        except DomainException as e:
            logger.warning(f"Domain exception: {str(e)}")
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Domain Error",
                    "message": str(e),
                    "type": type(e).__name__
                }
            )
            
        except DocumentError as e:
            logger.warning(f"Document error: {str(e)}")
            return JSONResponse(
                status_code=404,
                content={
                    "error": "Document Error",
                    "message": str(e),
                    "type": type(e).__name__
                }
            )
            
        except ChatError as e:
            logger.warning(f"Chat error: {str(e)}")
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Chat Error",
                    "message": str(e),
                    "type": type(e).__name__
                }
            )
            

            
        except FileStorageError as e:
            logger.warning(f"File storage error: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": "File Storage Error",
                    "message": str(e),
                    "type": type(e).__name__
                }
            )
            
        except CitationError as e:
            logger.warning(f"Citation error: {str(e)}")
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Citation Error",
                    "message": str(e),
                    "type": type(e).__name__
                }
            )
            
        except ConfigurationError as e:
            logger.error(f"Configuration error: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Configuration Error",
                    "message": "Internal server configuration error",
                    "type": type(e).__name__
                }
            )
            
        except ValueError as e:
            logger.warning(f"Value error: {str(e)}")
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Validation Error",
                    "message": str(e),
                    "type": "ValueError"
                }
            )
            
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred",
                    "type": "InternalServerError"
                }
            )
