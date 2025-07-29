"""
Domain Exceptions

Follows S.O.L.I.D principles:
- Single Responsibility: Each exception has a specific purpose
- Open/Closed: Can be extended with new exception types
- Liskov Substitution: All exceptions can be used as base exceptions
- Interface Segregation: Specific exceptions for specific error types
- Dependency Inversion: No dependencies on concrete implementations
"""
from typing import Optional, Any, Dict


class DomainException(Exception):
    """Base exception for all domain-related errors."""
    
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


class ValidationError(DomainException):
    """Raised when domain validation fails."""
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Optional[Any] = None
    ):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details={"field_name": field_name, "field_value": field_value}
        )
        self.field_name = field_name
        self.field_value = field_value


class BusinessRuleViolationError(DomainException):
    """Raised when business rules are violated."""
    
    def __init__(self, message: str, rule_name: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="BUSINESS_RULE_VIOLATION",
            details={"rule_name": rule_name}
        )
        self.rule_name = rule_name


class DocumentError(DomainException):
    """Base exception for document-related errors."""
    pass


class DocumentNotFoundError(DocumentError):
    """Raised when a document is not found."""
    
    def __init__(self, document_id: str):
        super().__init__(
            message=f"Document not found: {document_id}",
            error_code="DOCUMENT_NOT_FOUND",
            details={"document_id": document_id}
        )
        self.document_id = document_id


class DocumentProcessingError(DocumentError):
    """Raised when document processing fails."""
    
    def __init__(
        self,
        message: str,
        document_id: Optional[str] = None,
        processing_stage: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code="DOCUMENT_PROCESSING_ERROR",
            details={
                "document_id": document_id,
                "processing_stage": processing_stage
            }
        )
        self.document_id = document_id
        self.processing_stage = processing_stage


class InvalidDocumentStateError(DocumentError):
    """Raised when document is in invalid state for operation."""
    
    def __init__(
        self,
        message: str,
        document_id: str,
        current_state: str,
        expected_state: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code="INVALID_DOCUMENT_STATE",
            details={
                "document_id": document_id,
                "current_state": current_state,
                "expected_state": expected_state
            }
        )
        self.document_id = document_id
        self.current_state = current_state
        self.expected_state = expected_state


class ChatError(DomainException):
    """Base exception for chat-related errors."""
    pass


class ChatSessionNotFoundError(ChatError):
    """Raised when a chat session is not found."""
    
    def __init__(self, session_id: str):
        super().__init__(
            message=f"Chat session not found: {session_id}",
            error_code="CHAT_SESSION_NOT_FOUND",
            details={"session_id": session_id}
        )
        self.session_id = session_id


class MessageNotFoundError(ChatError):
    """Raised when a message is not found."""
    
    def __init__(self, message_id: str):
        super().__init__(
            message=f"Message not found: {message_id}",
            error_code="MESSAGE_NOT_FOUND",
            details={"message_id": message_id}
        )
        self.message_id = message_id


class InvalidSessionStateError(ChatError):
    """Raised when chat session is in invalid state for operation."""
    
    def __init__(
        self,
        message: str,
        session_id: str,
        current_state: str,
        expected_state: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code="INVALID_SESSION_STATE",
            details={
                "session_id": session_id,
                "current_state": current_state,
                "expected_state": expected_state
            }
        )
        self.session_id = session_id
        self.current_state = current_state
        self.expected_state = expected_state


class RepositoryError(DomainException):
    """Base exception for repository-related errors."""
    pass


class VectorRepositoryError(RepositoryError):
    """Raised when vector repository operations fail."""
    
    def __init__(self, message: str, operation: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="VECTOR_REPOSITORY_ERROR",
            details={"operation": operation}
        )
        self.operation = operation


class DatabaseConnectionError(RepositoryError):
    """Raised when database connection fails."""
    
    def __init__(self, message: str):
        super().__init__(
            message=message,
            error_code="DATABASE_CONNECTION_ERROR"
        )


class ConcurrencyError(RepositoryError):
    """Raised when concurrent access conflicts occur."""
    
    def __init__(self, message: str = "Concurrent access conflict", resource: Optional[str] = None):
        super().__init__(message)
        self.resource = resource


class MessageGenerationError(DomainException):
    """Raised when message generation fails."""
    
    def __init__(
        self,
        message: str,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.session_id = session_id
        self.model = model
        self.details = details or {}


class EmbeddingError(DomainException):
    """Raised when embedding generation fails."""
    
    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        text_length: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.model = model
        self.text_length = text_length
        self.details = details or {}


class ConfigurationError(DomainException):
    """Raised when configuration is invalid or missing."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.config_key = config_key
        self.details = details or {}


class CitationError(DomainException):
    """Raised when citation processing fails."""
    
    def __init__(
        self,
        message: str,
        document_id: Optional[str] = None,
        page_number: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.document_id = document_id
        self.page_number = page_number
        self.details = details or {}


class ConversationError(DomainException):
    """Raised when conversation operations fail."""
    
    def __init__(
        self,
        message: str,
        session_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.session_id = session_id
        self.details = details or {}


class AdvancedSearchError(DomainException):
    """Raised when advanced search operations fail."""
    
    def __init__(
        self,
        message: str,
        search_query: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.search_query = search_query
        self.details = details or {}


class DuplicateDocumentError(DomainException):
    """Raised when attempting to create a duplicate document."""
    
    def __init__(
        self,
        message: str,
        document_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.document_id = document_id
        self.details = details or {}


class CitationExtractionError(DomainException):
    """Raised when citation extraction operations fail."""
    
    def __init__(
        self,
        message: str,
        document_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.document_id = document_id
        self.details = details or {}


class CitationValidationError(DomainException):
    """Raised when citation validation operations fail."""
    
    def __init__(
        self,
        message: str,
        citation_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.citation_id = citation_id
        self.details = details or {}
