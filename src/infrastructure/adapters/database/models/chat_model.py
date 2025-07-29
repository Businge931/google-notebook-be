"""
Chat Database Models

"""
from datetime import datetime
from typing import List, Optional
from sqlalchemy import String, Integer, Text, Enum as SQLEnum, JSON, ForeignKey, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..base import BaseModel
from src.core.domain.entities import SessionStatus, MessageRole, MessageStatus


class ChatSessionModel(BaseModel):
    """
    SQLAlchemy model for ChatSession entity.
    
    Maps domain ChatSession entity to database representation.
    """
    __tablename__ = "chat_sessions"
    
    # Primary key
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    
    # Foreign key to document (nullable to support document-less sessions)
    document_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("documents.id"),
        nullable=True,
        index=True
    )
    
    # Session information
    status: Mapped[SessionStatus] = mapped_column(
        SQLEnum(SessionStatus),
        nullable=False,
        default=SessionStatus.ACTIVE
    )
    
    title: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    message_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    
    # Activity tracking
    last_activity_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False
    )
    
    # Relationships
    document: Mapped["DocumentModel"] = relationship(
        "DocumentModel",
        back_populates="chat_sessions"
    )
    
    messages: Mapped[List["MessageModel"]] = relationship(
        "MessageModel",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="MessageModel.created_at"
    )
    
    def __repr__(self) -> str:
        """String representation of the chat session model."""
        return f"<ChatSessionModel(id={self.id}, document_id={self.document_id}, status={self.status.value})>"


class MessageModel(BaseModel):
    """
    SQLAlchemy model for Message entity.
    
    Represents chat messages in conversations.
    """
    __tablename__ = "messages"
    
    # Primary key
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    
    # Foreign key to session
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("chat_sessions.id"),
        nullable=False,
        index=True
    )
    
    # Message information
    role: Mapped[MessageRole] = mapped_column(
        SQLEnum(MessageRole),
        nullable=False
    )
    
    content: Mapped[str] = mapped_column(Text, nullable=False)
    
    status: Mapped[MessageStatus] = mapped_column(
        SQLEnum(MessageStatus),
        nullable=False,
        default=MessageStatus.PENDING
    )
    
    # Processing information
    token_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    processing_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Relationships
    session: Mapped["ChatSessionModel"] = relationship(
        "ChatSessionModel",
        back_populates="messages"
    )
    
    citations: Mapped[List["CitationModel"]] = relationship(
        "CitationModel",
        back_populates="message",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        """String representation of the message model."""
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"<MessageModel(id={self.id}, role={self.role.value}, status={self.status.value})>"


class CitationModel(BaseModel):
    """
    SQLAlchemy model for Citation entity.
    
    Represents citations within messages.
    """
    __tablename__ = "citations"
    
    # Primary key
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    
    # Foreign key to message
    message_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("messages.id"),
        nullable=False,
        index=True
    )
    
    # Citation information
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    text_snippet: Mapped[str] = mapped_column(Text, nullable=False)
    start_position: Mapped[int] = mapped_column(Integer, nullable=False)
    end_position: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence_score: Mapped[float] = mapped_column(nullable=False, default=1.0)
    
    # Additional metadata
    extra_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    # Relationships
    message: Mapped["MessageModel"] = relationship(
        "MessageModel",
        back_populates="citations"
    )
    
    def __repr__(self) -> str:
        """String representation of the citation model."""
        snippet_preview = self.text_snippet[:30] + "..." if len(self.text_snippet) > 30 else self.text_snippet
        return f"<CitationModel(id={self.id}, page={self.page_number}, snippet='{snippet_preview}')>"
