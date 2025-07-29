"""
Shared Exceptions

Custom exceptions for domain and application errors.
"""
from .domain_exceptions import (
    DomainException,
    ValidationError,
    BusinessRuleViolationError,
    DocumentError,
    DocumentNotFoundError,
    DocumentProcessingError,
    InvalidDocumentStateError,
    DuplicateDocumentError,
    ChatError,
    ChatSessionNotFoundError,
    MessageNotFoundError,
    MessageGenerationError,
    EmbeddingError,
    ConfigurationError,
    CitationError,
    ConversationError,
    InvalidSessionStateError,
    RepositoryError,
    VectorRepositoryError,
    DatabaseConnectionError,
    ConcurrencyError,
    AdvancedSearchError,
    CitationExtractionError,
    CitationValidationError,
)
from .file_storage_exceptions import FileStorageError

__all__ = [
    "DomainException",
    "ValidationError",
    "BusinessRuleViolationError",
    "DocumentError",
    "DocumentNotFoundError",
    "DocumentProcessingError",
    "FileStorageError",
    "InvalidDocumentStateError",
    "ChatError",
    "ChatSessionNotFoundError",
    "MessageNotFoundError",
    "MessageGenerationError",
    "EmbeddingError",
    "ConfigurationError",
    "CitationError",
    "ConversationError",
    "InvalidSessionStateError",
    "RepositoryError",
    "VectorRepositoryError",
    "DatabaseConnectionError",
    "ConcurrencyError",
]