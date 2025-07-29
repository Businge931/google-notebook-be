"""
Document Repository Interface

"""
from abc import ABC, abstractmethod
from typing import List, Optional

from ..entities import Document, DocumentStatus
from ..value_objects import DocumentId


class DocumentRepository(ABC):
    """
    Abstract repository interface for document persistence operations.
    
    Defines the contract for document storage without coupling to
    specific persistence technologies.
    """
    
    @abstractmethod
    async def save(self, document: Document) -> Document:
        """
        Save a document to the repository.
        
        Args:
            document: Document entity to save
            
        Returns:
            Saved document entity
            
        Raises:
            RepositoryError: If save operation fails
        """
        pass
    
    @abstractmethod
    async def find_by_id(self, document_id: DocumentId) -> Optional[Document]:
        """
        Find a document by its ID.
        
        Args:
            document_id: Unique document identifier
            
        Returns:
            Document entity if found, None otherwise
            
        Raises:
            RepositoryError: If query operation fails
        """
        pass
    
    @abstractmethod
    async def find_by_status(self, status: DocumentStatus) -> List[Document]:
        """
        Find documents by their status.
        
        Args:
            status: Document status to filter by
            
        Returns:
            List of documents with the specified status
            
        Raises:
            RepositoryError: If query operation fails
        """
        pass
    
    @abstractmethod
    async def find_all(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[Document]:
        """
        Find all documents with optional pagination.
        
        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            
        Returns:
            List of document entities
            
        Raises:
            RepositoryError: If query operation fails
        """
        pass
    
    @abstractmethod
    async def update(self, document: Document) -> Document:
        """
        Update an existing document.
        
        Args:
            document: Document entity with updated data
            
        Returns:
            Updated document entity
            
        Raises:
            RepositoryError: If update operation fails
            DocumentNotFoundError: If document doesn't exist
        """
        pass
    
    @abstractmethod
    async def delete(self, document_id: DocumentId) -> bool:
        """
        Delete a document by its ID.
        
        Args:
            document_id: Unique document identifier
            
        Returns:
            True if document was deleted, False if not found
            
        Raises:
            RepositoryError: If delete operation fails
        """
        pass
    
    @abstractmethod
    async def exists(self, document_id: DocumentId) -> bool:
        """
        Check if a document exists.
        
        Args:
            document_id: Unique document identifier
            
        Returns:
            True if document exists, False otherwise
            
        Raises:
            RepositoryError: If query operation fails
        """
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """
        Get total count of documents.
        
        Returns:
            Total number of documents
            
        Raises:
            RepositoryError: If count operation fails
        """
        pass
    
    @abstractmethod
    async def find_processing_documents(self) -> List[Document]:
        """
        Find documents that are currently being processed.
        
        Returns:
            List of documents in processing status
            
        Raises:
            RepositoryError: If query operation fails
        """
        pass
