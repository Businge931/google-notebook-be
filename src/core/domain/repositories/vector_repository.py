"""
Vector Repository Interface

"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from ..value_objects import DocumentId


class VectorSearchResult:
    """Represents a vector search result."""
    
    def __init__(
        self,
        chunk_id: str,
        document_id: DocumentId,
        page_number: int,
        text_content: str,
        similarity_score: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.page_number = page_number
        self.text_content = text_content
        self.similarity_score = similarity_score
        self.metadata = metadata or {}
        
        # Validate similarity score
        if not 0.0 <= similarity_score <= 1.0:
            raise ValueError("Similarity score must be between 0.0 and 1.0")


class VectorRepository(ABC):
    """
    Abstract repository interface for vector storage operations.
    
    Defines the contract for vector storage and similarity search
    without coupling to specific vector database technologies.
    """
    
    @abstractmethod
    async def store_document_vectors(
        self,
        document_id: DocumentId,
        chunks: List[Dict[str, Any]]
    ) -> bool:
        """
        Store document text chunks as vectors.
        
        Args:
            document_id: Document identifier
            chunks: List of text chunks with metadata
                Each chunk should contain:
                - chunk_id: Unique chunk identifier
                - text_content: Text content to vectorize
                - page_number: Page number in document
                - start_position: Start position in page
                - end_position: End position in page
                - metadata: Additional metadata
                
        Returns:
            True if vectors were stored successfully
            
        Raises:
            VectorRepositoryError: If storage operation fails
        """
        pass
    
    @abstractmethod
    async def search_similar(
        self,
        document_id: DocumentId,
        query_text: str,
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[VectorSearchResult]:
        """
        Search for similar text chunks within a document.
        
        Args:
            document_id: Document identifier to search within
            query_text: Text query to find similar content
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score threshold
            
        Returns:
            List of search results ordered by similarity score (descending)
            
        Raises:
            VectorRepositoryError: If search operation fails
        """
        pass
    
    @abstractmethod
    async def search_across_documents(
        self,
        query_text: str,
        document_ids: Optional[List[DocumentId]] = None,
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[VectorSearchResult]:
        """
        Search for similar text chunks across multiple documents.
        
        Args:
            query_text: Text query to find similar content
            document_ids: Optional list of document IDs to search within
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score threshold
            
        Returns:
            List of search results ordered by similarity score (descending)
            
        Raises:
            VectorRepositoryError: If search operation fails
        """
        pass
    
    @abstractmethod
    async def delete_document_vectors(self, document_id: DocumentId) -> bool:
        """
        Delete all vectors for a specific document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            True if vectors were deleted successfully
            
        Raises:
            VectorRepositoryError: If delete operation fails
        """
        pass
    
    @abstractmethod
    async def update_document_vectors(
        self,
        document_id: DocumentId,
        chunks: List[Dict[str, Any]]
    ) -> bool:
        """
        Update vectors for a document (delete old, store new).
        
        Args:
            document_id: Document identifier
            chunks: List of updated text chunks with metadata
            
        Returns:
            True if vectors were updated successfully
            
        Raises:
            VectorRepositoryError: If update operation fails
        """
        pass
    
    @abstractmethod
    async def get_document_chunk_count(self, document_id: DocumentId) -> int:
        """
        Get the number of chunks stored for a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Number of chunks stored for the document
            
        Raises:
            VectorRepositoryError: If count operation fails
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the vector database is healthy and accessible.
        
        Returns:
            True if vector database is healthy
            
        Raises:
            VectorRepositoryError: If health check fails
        """
        pass
