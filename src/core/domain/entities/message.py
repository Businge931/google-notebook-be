"""
Message Entity
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
import uuid

from ..value_objects import SessionId


class MessageRole(Enum):
    """Message role enumeration."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageStatus(Enum):
    """Message status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DELETED = "deleted"


@dataclass
class Citation:
    """Represents a citation within a message."""
    citation_id: str
    page_number: int
    text_snippet: str
    start_position: int
    end_position: int
    confidence_score: float = 1.0
    
    def __post_init__(self) -> None:
        """Validate citation data."""
        if not self.citation_id:
            raise ValueError("Citation ID cannot be empty")
        if self.page_number < 1:
            raise ValueError("Page number must be positive")
        if not self.text_snippet.strip():
            raise ValueError("Text snippet cannot be empty")
        if self.start_position < 0:
            raise ValueError("Start position cannot be negative")
        if self.end_position <= self.start_position:
            raise ValueError("End position must be greater than start position")
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")


@dataclass
class Message:
    """
    Message entity representing a chat message in a conversation.
    
    Manages message content, citations, and processing state.
    """
    message_id: str
    session_id: SessionId
    role: MessageRole
    content: str
    status: MessageStatus = MessageStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    citations: List[Citation] = field(default_factory=list)
    token_count: Optional[int] = None
    processing_time_ms: Optional[int] = None
    error_message: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate message state."""
        self._validate_content()
        self._validate_timestamps()
        self._validate_token_count()
        self._validate_processing_time()
        self._validate_status_consistency()
    
    def _validate_content(self) -> None:
        """Validate message content."""
        if not self.content or not self.content.strip():
            raise ValueError("Message content cannot be empty")
        
        if len(self.content) > 10000:  # 10KB limit
            raise ValueError("Message content too long (max 10000 characters)")
    
    def _validate_timestamps(self) -> None:
        """Validate timestamp consistency."""
        if self.updated_at < self.created_at:
            raise ValueError("Updated timestamp cannot be before created timestamp")
    
    def _validate_token_count(self) -> None:
        """Validate token count if provided."""
        if self.token_count is not None:
            if not isinstance(self.token_count, int) or self.token_count < 0:
                raise ValueError("Token count must be a non-negative integer")
    
    def _validate_processing_time(self) -> None:
        """Validate processing time if provided."""
        if self.processing_time_ms is not None:
            if not isinstance(self.processing_time_ms, int) or self.processing_time_ms < 0:
                raise ValueError("Processing time must be a non-negative integer")
    
    def _validate_status_consistency(self) -> None:
        """Validate status consistency with other fields."""
        if self.status == MessageStatus.FAILED and not self.error_message:
            raise ValueError("Failed messages must have an error message")
        
        if self.status == MessageStatus.COMPLETED and self.role == MessageRole.ASSISTANT:
            if not self.citations and self.role == MessageRole.ASSISTANT:
                # Assistant messages should typically have citations, but not required
                pass
    
    @classmethod
    def create_user_message(
        cls,
        session_id: SessionId,
        content: str
    ) -> Message:
        """Create a new user message."""
        return cls(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role=MessageRole.USER,
            content=content,
            status=MessageStatus.COMPLETED  # User messages are immediately complete
        )
    
    @classmethod
    def create_assistant_message(
        cls,
        session_id: SessionId,
        content: str = ""
    ) -> Message:
        """Create a new assistant message."""
        return cls(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=content,
            status=MessageStatus.PENDING
        )
    
    def start_processing(self) -> None:
        """Mark message as processing."""
        if self.status != MessageStatus.PENDING:
            raise ValueError(f"Cannot start processing message in {self.status.value} status")
        
        self.status = MessageStatus.PROCESSING
        self.updated_at = datetime.now(timezone.utc)
    
    def complete_processing(
        self,
        content: str,
        citations: List[Citation],
        token_count: Optional[int] = None,
        processing_time_ms: Optional[int] = None
    ) -> None:
        """Complete message processing with results."""
        if self.status != MessageStatus.PROCESSING:
            raise ValueError(f"Cannot complete processing for message in {self.status.value} status")
        
        if not content.strip():
            raise ValueError("Completed message content cannot be empty")
        
        self.content = content
        self.citations = citations
        self.token_count = token_count
        self.processing_time_ms = processing_time_ms
        self.status = MessageStatus.COMPLETED
        self.updated_at = datetime.now(timezone.utc)
    
    def fail_processing(self, error_message: str) -> None:
        """Mark message processing as failed."""
        if not error_message:
            raise ValueError("Error message cannot be empty")
        
        self.status = MessageStatus.FAILED
        self.error_message = error_message
        self.updated_at = datetime.now(timezone.utc)
    
    def add_citation(self, citation: Citation) -> None:
        """Add a citation to the message."""
        if self.role != MessageRole.ASSISTANT:
            raise ValueError("Only assistant messages can have citations")
        
        self.citations.append(citation)
        self.updated_at = datetime.now(timezone.utc)
    
    def delete(self) -> None:
        """Mark message as deleted."""
        self.status = MessageStatus.DELETED
        self.updated_at = datetime.now(timezone.utc)
    
    @property
    def is_from_user(self) -> bool:
        """Check if message is from user."""
        return self.role == MessageRole.USER
    
    @property
    def is_from_assistant(self) -> bool:
        """Check if message is from assistant."""
        return self.role == MessageRole.ASSISTANT
    
    @property
    def is_completed(self) -> bool:
        """Check if message processing is completed."""
        return self.status == MessageStatus.COMPLETED
    
    @property
    def is_failed(self) -> bool:
        """Check if message processing failed."""
        return self.status == MessageStatus.FAILED
    
    @property
    def is_deleted(self) -> bool:
        """Check if message is deleted."""
        return self.status == MessageStatus.DELETED
    
    @property
    def has_citations(self) -> bool:
        """Check if message has citations."""
        return len(self.citations) > 0
    
    @property
    def citation_count(self) -> int:
        """Get number of citations."""
        return len(self.citations)
    
    def __str__(self) -> str:
        """String representation of the message."""
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Message({self.message_id[:8]}, {self.role.value}, {self.status.value}, '{content_preview}')"
