"""
Document Database Model

"""
from datetime import datetime
from typing import List, Optional
from sqlalchemy import String, Integer, Text, Enum as SQLEnum, JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..base import BaseModel
from src.core.domain.entities import DocumentStatus, ProcessingStage


class DocumentModel(BaseModel):
    """
    SQLAlchemy model for Document entity.
    
    Maps domain Document entity to database representation while maintaining
    separation of concerns (Dependency Inversion Principle).
    """
    __tablename__ = "documents"
    
    # Primary key
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    
    # File metadata
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    file_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    original_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    
    # Processing information
    status: Mapped[DocumentStatus] = mapped_column(
        SQLEnum(DocumentStatus),
        nullable=False,
        default=DocumentStatus.UPLOADED
    )
    processing_stage: Mapped[ProcessingStage] = mapped_column(
        SQLEnum(ProcessingStage),
        nullable=False,
        default=ProcessingStage.UPLOAD_COMPLETE
    )
    
    # Document content information
    page_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_chunks: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Error handling
    processing_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Upload timestamp (from file metadata)
    upload_timestamp: Mapped[datetime] = mapped_column(nullable=False)
    
    # Relationships
    chunks: Mapped[List["DocumentChunkModel"]] = relationship(
        "DocumentChunkModel",
        back_populates="document",
        cascade="all, delete-orphan"
    )
    
    chat_sessions: Mapped[List["ChatSessionModel"]] = relationship(
        "ChatSessionModel",
        back_populates="document",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        """String representation of the document model."""
        return f"<DocumentModel(id={self.id}, filename={self.filename}, status={self.status.value})>"


class DocumentChunkModel(BaseModel):
    """
    SQLAlchemy model for DocumentChunk entity.
    
    Represents text chunks extracted from documents.
    """
    __tablename__ = "document_chunks"
    
    # Primary key
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    
    # Foreign key to document
    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.id"),
        nullable=False,
        index=True
    )
    
    # Chunk information
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    text_content: Mapped[str] = mapped_column(Text, nullable=False)
    start_position: Mapped[int] = mapped_column(Integer, nullable=False)
    end_position: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Additional metadata
    extra_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    # Relationships
    document: Mapped["DocumentModel"] = relationship(
        "DocumentModel",
        back_populates="chunks"
    )
    
    def __repr__(self) -> str:
        """String representation of the chunk model."""
        return f"<DocumentChunkModel(id={self.id}, document_id={self.document_id}, page={self.page_number})>"
