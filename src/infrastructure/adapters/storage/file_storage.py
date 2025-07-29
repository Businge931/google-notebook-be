"""
File Storage Interface and Implementations
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO, Optional
import os
import shutil
from datetime import datetime

from ...config.settings import FileStorageSettings
from src.shared.exceptions import FileStorageError


class FileStorage(ABC):
    """
    Abstract interface for file storage operations.
    
    Defines the contract for file storage adapters following
    Interface Segregation Principle.
    """
    
    @abstractmethod
    async def save_file(
        self,
        file_data: BinaryIO,
        file_path: str,
        content_type: Optional[str] = None
    ) -> str:
        """
        Save a file to storage.
        
        Args:
            file_data: Binary file data
            file_path: Relative path where file should be stored
            content_type: MIME type of the file
            
        Returns:
            Full path to the saved file
            
        Raises:
            FileStorageError: If save operation fails
        """
        pass
    
    @abstractmethod
    async def get_file(self, file_path: str) -> BinaryIO:
        """
        Retrieve a file from storage.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Binary file data
            
        Raises:
            FileStorageError: If file not found or retrieval fails
        """
        pass
    
    @abstractmethod
    async def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from storage.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file was deleted, False if not found
            
        Raises:
            FileStorageError: If delete operation fails
        """
        pass
    
    @abstractmethod
    async def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists in storage.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file exists, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_file_size(self, file_path: str) -> int:
        """
        Get the size of a file in bytes.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File size in bytes
            
        Raises:
            FileStorageError: If file not found or operation fails
        """
        pass
    
    @abstractmethod
    async def get_file_url(self, file_path: str) -> str:
        """
        Get a URL to access the file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            URL to access the file
            
        Raises:
            FileStorageError: If file not found or operation fails
        """
        pass


class LocalFileStorage(FileStorage):
    """
    Local filesystem implementation of FileStorage.
    
    Stores files on the local filesystem following Single Responsibility Principle.
    """
    
    def __init__(self, settings: FileStorageSettings):
        """
        Initialize local file storage.
        
        Args:
            settings: File storage configuration
        """
        self._base_path = Path(settings.upload_dir)
        self._base_url = getattr(settings, 'base_url', None) or "http://localhost:8000"
        
        # Create base directory if it doesn't exist
        self._base_path.mkdir(parents=True, exist_ok=True)
    
    async def save_file(
        self,
        file_data: BinaryIO,
        file_path: str,
        content_type: Optional[str] = None
    ) -> str:
        """
        Save a file to local filesystem.
        
        Args:
            file_data: Binary file data
            file_path: Relative path where file should be stored
            content_type: MIME type of the file (not used in local storage)
            
        Returns:
            Full path to the saved file
            
        Raises:
            FileStorageError: If save operation fails
        """
        try:
            # Ensure file path is safe (no directory traversal)
            safe_path = self._sanitize_path(file_path)
            full_path = self._base_path / safe_path
            
            # Create parent directories if they don't exist
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save file
            with open(full_path, 'wb') as f:
                shutil.copyfileobj(file_data, f)
            
            # Return relative path (without base path) to avoid double prefix in get_file
            return safe_path
            
        except Exception as e:
            raise FileStorageError(f"Failed to save file {file_path}: {str(e)}")
    
    async def get_file(self, file_path: str) -> BinaryIO:
        """
        Retrieve a file from local filesystem.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Binary file data
            
        Raises:
            FileStorageError: If file not found or retrieval fails
        """
        try:
            safe_path = self._sanitize_path(file_path)
            full_path = self._base_path / safe_path
            
            if not full_path.exists():
                raise FileStorageError(f"File not found: {file_path}")
            
            return open(full_path, 'rb')
            
        except FileStorageError:
            raise
        except Exception as e:
            raise FileStorageError(f"Failed to retrieve file {file_path}: {str(e)}")
    
    async def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from local filesystem.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file was deleted, False if not found
            
        Raises:
            FileStorageError: If delete operation fails
        """
        try:
            safe_path = self._sanitize_path(file_path)
            full_path = self._base_path / safe_path
            
            if not full_path.exists():
                return False
            
            full_path.unlink()
            return True
            
        except Exception as e:
            raise FileStorageError(f"Failed to delete file {file_path}: {str(e)}")
    
    async def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists in local filesystem.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file exists, False otherwise
        """
        try:
            safe_path = self._sanitize_path(file_path)
            full_path = self._base_path / safe_path
            return full_path.exists() and full_path.is_file()
            
        except Exception:
            return False
    
    async def get_file_size(self, file_path: str) -> int:
        """
        Get the size of a file in bytes.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File size in bytes
            
        Raises:
            FileStorageError: If file not found or operation fails
        """
        try:
            safe_path = self._sanitize_path(file_path)
            full_path = self._base_path / safe_path
            
            if not full_path.exists():
                raise FileStorageError(f"File not found: {file_path}")
            
            return full_path.stat().st_size
            
        except FileStorageError:
            raise
        except Exception as e:
            raise FileStorageError(f"Failed to get file size {file_path}: {str(e)}")
    
    async def get_file_url(self, file_path: str) -> str:
        """
        Get a URL to access the file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            URL to access the file
            
        Raises:
            FileStorageError: If file not found or operation fails
        """
        try:
            if not await self.file_exists(file_path):
                raise FileStorageError(f"File not found: {file_path}")
            
            # Return URL for accessing the file
            safe_path = self._sanitize_path(file_path)
            return f"{self._base_url}/files/{safe_path}"
            
        except FileStorageError:
            raise
        except Exception as e:
            raise FileStorageError(f"Failed to get file URL {file_path}: {str(e)}")
    
    def _sanitize_path(self, file_path: str) -> str:
        """
        Sanitize file path to prevent directory traversal attacks.
        
        Args:
            file_path: Input file path
            
        Returns:
            Sanitized file path
            
        Raises:
            FileStorageError: If path is invalid
        """
        # Remove any leading slashes and resolve path
        clean_path = file_path.lstrip('/')
        
        # Check for directory traversal attempts
        if '..' in clean_path or clean_path.startswith('/'):
            raise FileStorageError(f"Invalid file path: {file_path}")
        
        return clean_path


class FileStorageFactory:
    """
    Factory for creating file storage instances.
    
    Follows Open/Closed Principle - can be extended with new storage types.
    """
    
    @staticmethod
    def create_storage(settings: FileStorageSettings) -> FileStorage:
        """
        Create appropriate file storage instance based on settings.
        
        Args:
            settings: File storage configuration
            
        Returns:
            File storage instance
            
        Raises:
            ValueError: If storage type is not supported
        """
        storage_type = settings.storage_type.lower()
        
        if storage_type == "local":
            return LocalFileStorage(settings)
        elif storage_type == "s3":
            # TODO: Implement S3FileStorage in future
            raise ValueError("S3 storage not yet implemented")
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")


# Utility functions for file operations
def generate_file_path(document_id: str, filename: str) -> str:
    """
    Generate a standardized file path for document storage.
    
    Args:
        document_id: Unique document identifier
        filename: Original filename
        
    Returns:
        Standardized file path
    """
    # Create path with date-based organization
    date_str = datetime.now().strftime("%Y/%m/%d")
    return f"documents/{date_str}/{document_id}/{filename}"


def get_file_extension(filename: str) -> str:
    """
    Extract file extension from filename.
    
    Args:
        filename: Name of the file
        
    Returns:
        File extension (without dot)
    """
    return Path(filename).suffix.lstrip('.').lower()


def is_allowed_file_type(filename: str, allowed_extensions: list) -> bool:
    """
    Check if file type is allowed.
    
    Args:
        filename: Name of the file
        allowed_extensions: List of allowed file extensions
        
    Returns:
        True if file type is allowed
    """
    extension = get_file_extension(filename)
    return extension in [ext.lower() for ext in allowed_extensions]
