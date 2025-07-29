"""
File Storage Exceptions

Custom exceptions for file storage operations.
"""
from typing import Optional
from .domain_exceptions import DomainException


class FileStorageError(DomainException):
    """Base exception for file storage operations."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, operation: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="FILE_STORAGE_ERROR",
            details={
                "file_path": file_path,
                "operation": operation
            }
        )
        self.file_path = file_path
        self.operation = operation
