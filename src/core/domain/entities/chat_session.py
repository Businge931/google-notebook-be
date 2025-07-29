"""
Chat Session Entity

"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from ..value_objects import SessionId, DocumentId


class SessionStatus(Enum):
    """Chat session status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"
    DELETED = "deleted"


@dataclass
class ChatSession:
    """
    Chat session entity representing a conversation about a document.
    
    Manages session lifecycle and maintains conversation context.
    """
    session_id: SessionId
    document_id: Optional[DocumentId] = None
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message_count: int = 0
    title: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate session state."""
        self._validate_timestamps()
        self._validate_message_count()
    
    def _validate_timestamps(self) -> None:
        """Validate timestamp consistency."""
        if self.updated_at < self.created_at:
            raise ValueError("Updated timestamp cannot be before created timestamp")
        
        if self.last_activity_at < self.created_at:
            raise ValueError("Last activity timestamp cannot be before created timestamp")
    
    def _validate_message_count(self) -> None:
        """Validate message count."""
        if self.message_count < 0:
            raise ValueError("Message count cannot be negative")
    
    @classmethod
    def create(
        cls,
        document_id: DocumentId,
        title: Optional[str] = None
    ) -> ChatSession:
        """Create a new chat session."""
        return cls(
            session_id=SessionId.generate(),
            document_id=document_id,
            title=title
        )
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        now = datetime.now(timezone.utc)
        self.last_activity_at = now
        self.updated_at = now
    
    def increment_message_count(self) -> None:
        """Increment message count and update activity."""
        self.message_count += 1
        self.update_activity()
    
    def set_title(self, title: str) -> None:
        """Set or update session title."""
        if not title or not title.strip():
            raise ValueError("Session title cannot be empty")
        
        if len(title) > 200:
            raise ValueError("Session title too long (max 200 characters)")
        
        self.title = title.strip()
        self.updated_at = datetime.now(timezone.utc)
    
    def deactivate(self) -> None:
        """Deactivate the session."""
        if self.status == SessionStatus.DELETED:
            raise ValueError("Cannot deactivate deleted session")
        
        self.status = SessionStatus.INACTIVE
        self.updated_at = datetime.now(timezone.utc)
    
    def reactivate(self) -> None:
        """Reactivate the session."""
        if self.status == SessionStatus.DELETED:
            raise ValueError("Cannot reactivate deleted session")
        
        self.status = SessionStatus.ACTIVE
        self.update_activity()
    
    def archive(self) -> None:
        """Archive the session."""
        if self.status == SessionStatus.DELETED:
            raise ValueError("Cannot archive deleted session")
        
        self.status = SessionStatus.ARCHIVED
        self.updated_at = datetime.now(timezone.utc)
    
    def delete(self) -> None:
        """Mark session as deleted."""
        self.status = SessionStatus.DELETED
        self.updated_at = datetime.now(timezone.utc)
    
    @property
    def is_active(self) -> bool:
        """Check if session is active."""
        return self.status == SessionStatus.ACTIVE
    
    @property
    def is_deleted(self) -> bool:
        """Check if session is deleted."""
        return self.status == SessionStatus.DELETED
    
    @property
    def can_receive_messages(self) -> bool:
        """Check if session can receive new messages."""
        return self.status == SessionStatus.ACTIVE
    
    @property
    def display_title(self) -> str:
        """Get display title for the session."""
        if self.title:
            return self.title
        return f"Chat Session {self.session_id.value[:8]}"
    
    def __str__(self) -> str:
        """String representation of the session."""
        return f"ChatSession({self.session_id}, {self.document_id}, {self.status.value})"
