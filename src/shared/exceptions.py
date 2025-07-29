"""
Shared Exceptions

"""
from typing import Optional, Dict, Any


class BaseApplicationError(Exception):
    """Base exception for all application errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class ValidationError(BaseApplicationError):
    """Raised when validation fails."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "VALIDATION_ERROR", details)
        self.field = field
        self.value = value


class ConfigurationError(BaseApplicationError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "CONFIGURATION_ERROR", details)
        self.config_key = config_key


class DocumentNotFoundError(BaseApplicationError):
    """Raised when a document is not found."""
    
    def __init__(
        self,
        message: str,
        document_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "DOCUMENT_NOT_FOUND", details)
        self.document_id = document_id


class DocumentProcessingError(BaseApplicationError):
    """Raised when document processing fails."""
    
    def __init__(
        self,
        message: str,
        document_id: Optional[str] = None,
        stage: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "DOCUMENT_PROCESSING_ERROR", details)
        self.document_id = document_id
        self.stage = stage


class FileStorageError(BaseApplicationError):
    """Raised when file storage operations fail."""
    
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "FILE_STORAGE_ERROR", details)
        self.file_path = file_path
        self.operation = operation


class DatabaseError(BaseApplicationError):
    """Raised when database operations fail."""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        table: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "DATABASE_ERROR", details)
        self.operation = operation
        self.table = table


class EmbeddingError(BaseApplicationError):
    """Raised when embedding generation fails."""
    
    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        text_length: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "EMBEDDING_ERROR", details)
        self.model = model
        self.text_length = text_length


class VectorRepositoryError(BaseApplicationError):
    """Raised when vector repository operations fail."""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        document_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "VECTOR_REPOSITORY_ERROR", details)
        self.operation = operation
        self.document_id = document_id


class SearchError(BaseApplicationError):
    """Raised when search operations fail."""
    
    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        search_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "SEARCH_ERROR", details)
        self.query = query
        self.search_type = search_type


class ChatSessionNotFoundError(BaseApplicationError):
    """Raised when a chat session is not found."""
    
    def __init__(
        self,
        message: str,
        session_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "CHAT_SESSION_NOT_FOUND", details)
        self.session_id = session_id


class MessageGenerationError(BaseApplicationError):
    """Raised when message generation fails."""
    
    def __init__(
        self,
        message: str,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "MESSAGE_GENERATION_ERROR", details)
        self.session_id = session_id
        self.model = model


class CitationError(BaseApplicationError):
    """Raised when citation processing fails."""
    
    def __init__(
        self,
        message: str,
        document_id: Optional[str] = None,
        page_number: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "CITATION_ERROR", details)
        self.document_id = document_id
        self.page_number = page_number


class AuthenticationError(BaseApplicationError):
    """Raised when authentication fails."""
    
    def __init__(
        self,
        message: str,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "AUTHENTICATION_ERROR", details)
        self.user_id = user_id


class AuthorizationError(BaseApplicationError):
    """Raised when authorization fails."""
    
    def __init__(
        self,
        message: str,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "AUTHORIZATION_ERROR", details)
        self.user_id = user_id
        self.resource = resource
        self.action = action


class RateLimitError(BaseApplicationError):
    """Raised when rate limits are exceeded."""
    
    def __init__(
        self,
        message: str,
        limit: Optional[int] = None,
        window: Optional[int] = None,
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "RATE_LIMIT_ERROR", details)
        self.limit = limit
        self.window = window
        self.retry_after = retry_after


class ServiceUnavailableError(BaseApplicationError):
    """Raised when external services are unavailable."""
    
    def __init__(
        self,
        message: str,
        service: Optional[str] = None,
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "SERVICE_UNAVAILABLE", details)
        self.service = service
        self.retry_after = retry_after


class TimeoutError(BaseApplicationError):
    """Raised when operations timeout."""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "TIMEOUT_ERROR", details)
        self.operation = operation
        self.timeout_seconds = timeout_seconds


class ConcurrencyError(BaseApplicationError):
    """Raised when concurrency limits are exceeded."""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        max_concurrent: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "CONCURRENCY_ERROR", details)
        self.operation = operation
        self.max_concurrent = max_concurrent


class ResourceExhaustedError(BaseApplicationError):
    """Raised when system resources are exhausted."""
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        current_usage: Optional[str] = None,
        limit: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "RESOURCE_EXHAUSTED", details)
        self.resource_type = resource_type
        self.current_usage = current_usage
        self.limit = limit


class DataIntegrityError(BaseApplicationError):
    """Raised when data integrity constraints are violated."""
    
    def __init__(
        self,
        message: str,
        constraint: Optional[str] = None,
        table: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "DATA_INTEGRITY_ERROR", details)
        self.constraint = constraint
        self.table = table


class SerializationError(BaseApplicationError):
    """Raised when serialization/deserialization fails."""
    
    def __init__(
        self,
        message: str,
        data_type: Optional[str] = None,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "SERIALIZATION_ERROR", details)
        self.data_type = data_type
        self.operation = operation


# Exception mapping for HTTP status codes
EXCEPTION_STATUS_MAP = {
    ValidationError: 400,
    ConfigurationError: 500,
    DocumentNotFoundError: 404,
    DocumentProcessingError: 422,
    FileStorageError: 500,
    DatabaseError: 500,
    EmbeddingError: 502,
    VectorRepositoryError: 500,
    SearchError: 422,
    ChatSessionNotFoundError: 404,
    MessageGenerationError: 502,
    CitationError: 422,
    AuthenticationError: 401,
    AuthorizationError: 403,
    RateLimitError: 429,
    ServiceUnavailableError: 503,
    TimeoutError: 504,
    ConcurrencyError: 429,
    ResourceExhaustedError: 507,
    DataIntegrityError: 409,
    SerializationError: 422,
    BaseApplicationError: 500
}


def get_http_status_code(exception: Exception) -> int:
    """
    Get HTTP status code for an exception.
    
    Args:
        exception: Exception instance
        
    Returns:
        HTTP status code
    """
    exception_type = type(exception)
    return EXCEPTION_STATUS_MAP.get(exception_type, 500)
