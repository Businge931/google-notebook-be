"""
Document Entity

"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List

from ..value_objects import DocumentId, FileMetadata, PageNumber


class DocumentStatus(Enum):
    """Document processing status enumeration."""
    UPLOADING = "UPLOADING"
    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"
    DELETED = "DELETED"


class ProcessingStage(Enum):
    """Document processing stage enumeration."""
    UPLOAD_STARTED = "UPLOAD_STARTED"
    UPLOAD_COMPLETE = "UPLOAD_COMPLETE"
    TEXT_EXTRACTION = "TEXT_EXTRACTION"
    PARSING_STARTED = "PARSING_STARTED"
    CHUNKING = "CHUNKING"
    VECTORIZATION = "VECTORIZATION"
    INDEXING = "INDEXING"
    COMPLETE = "COMPLETE"
    PARSING_FAILED = "PARSING_FAILED"


@dataclass
class DocumentChunk:
    """Represents a chunk of text from the document."""
    chunk_id: str
    page_number: PageNumber
    text_content: str
    start_position: int
    end_position: int
    
    def __post_init__(self) -> None:
        """Validate chunk data."""
        if not self.chunk_id:
            raise ValueError("Chunk ID cannot be empty")
        if not self.text_content.strip():
            raise ValueError("Chunk text content cannot be empty")
        if self.start_position < 0:
            raise ValueError("Start position cannot be negative")
        if self.end_position <= self.start_position:
            raise ValueError("End position must be greater than start position")


@dataclass
class Document:
    """
    Document entity representing a PDF document in the system.
    
    Encapsulates document state and business rules while maintaining
    data integrity through value objects.
    """
    document_id: DocumentId
    file_metadata: FileMetadata
    status: DocumentStatus = DocumentStatus.UPLOADED
    processing_stage: ProcessingStage = ProcessingStage.UPLOAD_COMPLETE
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    page_count: Optional[int] = None
    total_chunks: Optional[int] = None
    processing_error: Optional[str] = None
    file_path: Optional[str] = None
    chunks: List[DocumentChunk] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate document state."""
        self._validate_timestamps()
        self._validate_page_count()
        self._validate_processing_state()
    
    def _validate_timestamps(self) -> None:
        """Validate timestamp consistency."""
        # Handle timezone-aware vs naive datetime comparison
        try:
            if self.updated_at < self.created_at:
                raise ValueError("Updated timestamp cannot be before created timestamp")
        except TypeError:
            # Handle timezone-aware vs naive datetime comparison
            # Convert both to UTC for comparison if one is timezone-aware
            from datetime import timezone
            
            created_utc = self.created_at
            updated_utc = self.updated_at
            
            # If one is timezone-aware and the other is not, normalize them
            if self.created_at.tzinfo is not None and self.updated_at.tzinfo is None:
                updated_utc = self.updated_at.replace(tzinfo=timezone.utc)
            elif self.created_at.tzinfo is None and self.updated_at.tzinfo is not None:
                created_utc = self.created_at.replace(tzinfo=timezone.utc)
            
            if updated_utc < created_utc:
                raise ValueError("Updated timestamp cannot be before created timestamp")
    
    def _validate_page_count(self) -> None:
        """Validate page count if provided."""
        if self.page_count is not None:
            if not isinstance(self.page_count, int) or self.page_count < 1:
                raise ValueError("Page count must be a positive integer")
    
    def _validate_processing_state(self) -> None:
        """Validate processing state consistency."""
        if self.status == DocumentStatus.FAILED and not self.processing_error:
            raise ValueError("Failed documents must have a processing error")
        
        if self.status == DocumentStatus.PROCESSED and self.processing_stage != ProcessingStage.COMPLETE:
            raise ValueError("Processed documents must have complete processing stage")
    
    @classmethod
    def create(
        cls,
        file_metadata: FileMetadata,
        file_path: Optional[str] = None
    ) -> Document:
        """Create a new document instance."""
        return cls(
            document_id=DocumentId.generate(),
            file_metadata=file_metadata,
            file_path=file_path
        )
    
    def start_processing(self) -> None:
        """Mark document as processing."""
        if self.status not in [DocumentStatus.UPLOADED, DocumentStatus.FAILED, DocumentStatus.PROCESSING]:
            raise ValueError(f"Cannot start processing document in {self.status.value} status")
        
        # Only update status and stage if not already processing
        if self.status != DocumentStatus.PROCESSING:
            self.status = DocumentStatus.PROCESSING
            self.processing_stage = ProcessingStage.TEXT_EXTRACTION
        
        self.updated_at = datetime.utcnow()
    
    def advance_processing_stage(self, stage: ProcessingStage) -> None:
        """Advance to the next processing stage."""
        if self.status != DocumentStatus.PROCESSING:
            raise ValueError(f"Cannot advance processing stage for document in {self.status.value} status")
        
        # Validate stage progression - allow flexible progression for basic processing
        valid_progressions = {
            ProcessingStage.UPLOAD_COMPLETE: [ProcessingStage.TEXT_EXTRACTION, ProcessingStage.PARSING_STARTED],
            ProcessingStage.TEXT_EXTRACTION: [ProcessingStage.PARSING_STARTED, ProcessingStage.CHUNKING],
            ProcessingStage.PARSING_STARTED: [ProcessingStage.CHUNKING],
            ProcessingStage.CHUNKING: [ProcessingStage.VECTORIZATION, ProcessingStage.COMPLETE],  # Allow skip to COMPLETE
            ProcessingStage.VECTORIZATION: [ProcessingStage.INDEXING, ProcessingStage.COMPLETE],
            ProcessingStage.INDEXING: [ProcessingStage.COMPLETE],
        }
        
        allowed_stages = valid_progressions.get(self.processing_stage, [])
        if stage not in allowed_stages:
            raise ValueError(f"Invalid stage progression from {self.processing_stage.value} to {stage.value}. Allowed: {[s.value for s in allowed_stages]}")
        
        self.processing_stage = stage
        self.updated_at = datetime.utcnow()
        
        # Mark as processed if complete
        if stage == ProcessingStage.COMPLETE:
            self.status = DocumentStatus.PROCESSED
    
    def mark_processing_failed(self, error_message: str) -> None:
        """Mark document processing as failed."""
        if not error_message:
            raise ValueError("Error message cannot be empty")
        
        self.status = DocumentStatus.FAILED
        self.processing_error = error_message
        self.updated_at = datetime.utcnow()
    
    def set_page_count(self, page_count: int) -> None:
        """Set the total page count for the document."""
        if page_count < 1:
            raise ValueError("Page count must be positive")
        
        self.page_count = page_count
        self.updated_at = datetime.utcnow()
    
    def add_chunk(self, chunk: DocumentChunk) -> None:
        """Add a text chunk to the document."""
        if self.page_count and chunk.page_number.value > self.page_count:
            raise ValueError(f"Chunk page number {chunk.page_number.value} exceeds document page count {self.page_count}")
        
        self.chunks.append(chunk)
        self.total_chunks = len(self.chunks)
        self.updated_at = datetime.utcnow()
    
    def get_chunks_for_page(self, page_number: PageNumber) -> List[DocumentChunk]:
        """Get all chunks for a specific page."""
        return [chunk for chunk in self.chunks if chunk.page_number == page_number]
    
    def complete_parsing(self, page_count: int, chunks: List[DocumentChunk]) -> None:
        """Complete the parsing phase of document processing.
        
        Args:
            page_count: Total number of pages in the document
            chunks: List of document chunks created during parsing
        """
        # Set page count
        self.set_page_count(page_count)
        
        # Add all chunks to the document
        for chunk in chunks:
            self.add_chunk(chunk)
        
        # Advance to chunking stage (parsing is complete)
        self.advance_processing_stage(ProcessingStage.CHUNKING)
        
        self.updated_at = datetime.utcnow()
    
    def mark_as_deleted(self) -> None:
        """Mark document as deleted."""
        self.status = DocumentStatus.DELETED
        self.updated_at = datetime.utcnow()
    
    @property
    def is_processing_complete(self) -> bool:
        """Check if document processing is complete."""
        return self.status == DocumentStatus.PROCESSED and self.processing_stage == ProcessingStage.COMPLETE
    
    @property
    def is_failed(self) -> bool:
        """Check if document processing failed."""
        return self.status == DocumentStatus.FAILED
    
    @property
    def is_deleted(self) -> bool:
        """Check if document is deleted."""
        return self.status == DocumentStatus.DELETED
    
    @property
    def can_be_queried(self) -> bool:
        """Check if document can be queried (processed successfully)."""
        return self.is_processing_complete
    
    def __str__(self) -> str:
        """String representation of the document."""
        return f"Document({self.document_id}, {self.file_metadata.filename}, {self.status.value})"
