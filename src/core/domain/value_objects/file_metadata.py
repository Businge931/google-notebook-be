"""
File Metadata Value Object

"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Set
from pathlib import Path


@dataclass(frozen=True)
class FileMetadata:
    """
    Value object representing file metadata.
    
    Contains essential file information with validation.
    """
    filename: str
    file_size: int  # in bytes
    mime_type: str
    upload_timestamp: datetime
    original_path: Optional[str] = None
    
    # Allowed MIME types for PDF files
    ALLOWED_MIME_TYPES: Set[str] = frozenset({
        'application/pdf',
        'application/x-pdf',
        'application/acrobat',
        'applications/vnd.pdf',
        'text/pdf',
        'text/x-pdf'
    })
    
    # Maximum file size (100MB in bytes)
    MAX_FILE_SIZE: int = 100 * 1024 * 1024
    
    def __post_init__(self) -> None:
        """Validate file metadata."""
        self._validate_filename()
        self._validate_file_size()
        self._validate_mime_type()
        self._validate_timestamp()
    
    def _validate_filename(self) -> None:
        """Validate filename format and extension."""
        if not self.filename:
            raise ValueError("Filename cannot be empty")
        
        if len(self.filename) > 255:
            raise ValueError("Filename too long (max 255 characters)")
        
        # Check for invalid characters
        invalid_chars = {'<', '>', ':', '"', '|', '?', '*', '\0'}
        if any(char in self.filename for char in invalid_chars):
            raise ValueError(f"Filename contains invalid characters: {self.filename}")
        
        # Ensure PDF extension
        if not self.filename.lower().endswith('.pdf'):
            raise ValueError("File must have .pdf extension")
    
    def _validate_file_size(self) -> None:
        """Validate file size constraints."""
        if not isinstance(self.file_size, int):
            raise TypeError("File size must be an integer")
        
        if self.file_size <= 0:
            raise ValueError("File size must be positive")
        
        if self.file_size > self.MAX_FILE_SIZE:
            raise ValueError(f"File size exceeds maximum allowed size of {self.MAX_FILE_SIZE} bytes")
    
    def _validate_mime_type(self) -> None:
        """Validate MIME type for PDF files."""
        if not self.mime_type:
            raise ValueError("MIME type cannot be empty")
        
        if self.mime_type not in self.ALLOWED_MIME_TYPES:
            raise ValueError(f"Invalid MIME type: {self.mime_type}. Must be a PDF file.")
    
    def _validate_timestamp(self) -> None:
        """Validate upload timestamp."""
        if not isinstance(self.upload_timestamp, datetime):
            raise TypeError("Upload timestamp must be a datetime object")
        
        # Ensure timestamp is not in the future (with small tolerance for clock skew)
        now = datetime.utcnow()
        if self.upload_timestamp > now:
            raise ValueError("Upload timestamp cannot be in the future")
    
    @classmethod
    def create(
        cls,
        filename: str,
        file_size: int,
        mime_type: str,
        original_path: Optional[str] = None
    ) -> FileMetadata:
        """Create FileMetadata with current timestamp."""
        return cls(
            filename=filename,
            file_size=file_size,
            mime_type=mime_type,
            upload_timestamp=datetime.utcnow(),
            original_path=original_path
        )
    
    @property
    def file_extension(self) -> str:
        """Get file extension."""
        return Path(self.filename).suffix.lower()
    
    @property
    def file_size_mb(self) -> float:
        """Get file size in megabytes."""
        return self.file_size / (1024 * 1024)
    
    @property
    def is_pdf(self) -> bool:
        """Check if file is a PDF."""
        return self.file_extension == '.pdf' and self.mime_type in self.ALLOWED_MIME_TYPES
    
    def __str__(self) -> str:
        """String representation of file metadata."""
        return f"{self.filename} ({self.file_size_mb:.2f}MB, {self.mime_type})"
