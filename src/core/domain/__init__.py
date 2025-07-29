"""
Core Domain Layer

The domain layer contains the core business logic and rules.
It is independent of external concerns and frameworks.
"""
from .entities import (
    Document,
    DocumentStatus,
    ProcessingStage,
    DocumentChunk,
    ChatSession,
    SessionStatus,
    Message,
    MessageRole,
    MessageStatus,
    Citation,
)
from .value_objects import (
    DocumentId,
    PageNumber,
    FileMetadata,
    SessionId,
)
from .repositories import (
    DocumentRepository,
    ChatRepository,
    VectorRepository,
    VectorSearchResult,
)
from .services import (
    DocumentProcessingService,
)

__all__ = [
    # Entities
    "Document",
    "DocumentStatus",
    "ProcessingStage",
    "DocumentChunk",
    "ChatSession",
    "SessionStatus",
    "Message",
    "MessageRole",
    "MessageStatus",
    "Citation",
    # Value Objects
    "DocumentId",
    "PageNumber",
    "FileMetadata",
    "SessionId",
    # Repositories
    "DocumentRepository",
    "ChatRepository",
    "VectorRepository",
    "VectorSearchResult",
    # Services
    "DocumentProcessingService",
]