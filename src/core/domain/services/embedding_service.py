"""
Embedding Service Interface

"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from ..value_objects import DocumentId


@dataclass
class EmbeddingVector:
    """
    Represents an embedding vector following Single Responsibility Principle.
    """
    vector: List[float]
    dimension: int
    model: str
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate embedding vector after initialization."""
        if len(self.vector) != self.dimension:
            raise ValueError(f"Vector length {len(self.vector)} doesn't match dimension {self.dimension}")
        
        if not all(isinstance(x, (int, float)) for x in self.vector):
            raise ValueError("Vector must contain only numeric values")


@dataclass
class TextEmbedding:
    """
    Represents text with its embedding following Single Responsibility Principle.
    """
    text: str
    embedding: EmbeddingVector
    chunk_id: str
    document_id: DocumentId
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EmbeddingBatch:
    """
    Represents a batch of embeddings following Single Responsibility Principle.
    """
    embeddings: List[TextEmbedding]
    model: str
    total_tokens: Optional[int] = None
    processing_time_ms: Optional[int] = None


class EmbeddingService(ABC):
    """
    Abstract interface for text embedding services.
    
    Defines the contract for embedding generation following
    Interface Segregation Principle.
    """
    
    @abstractmethod
    async def generate_embedding(
        self,
        text: str,
        model: Optional[str] = None
    ) -> EmbeddingVector:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            model: Optional model name to use
            
        Returns:
            Embedding vector
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        pass
    
    @abstractmethod
    async def generate_embeddings_batch(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> List[EmbeddingVector]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            model: Optional model name to use
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        pass
    
    @abstractmethod
    def get_embedding_dimension(self, model: Optional[str] = None) -> int:
        """
        Get the dimension of embeddings for a model.
        
        Args:
            model: Optional model name
            
        Returns:
            Embedding dimension
        """
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Get list of available embedding models.
        
        Returns:
            List of model names
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the embedding service is healthy.
        
        Returns:
            True if service is healthy
        """
        pass


class DocumentEmbeddingService(ABC):
    """
    Abstract interface for document-level embedding operations.
    
    Handles document chunking and embedding following
    Interface Segregation Principle.
    """
    
    @abstractmethod
    async def embed_document_chunks(
        self,
        document_id: DocumentId,
        chunks: List[str],
        chunk_ids: List[str],
        model: Optional[str] = None
    ) -> EmbeddingBatch:
        """
        Generate embeddings for document chunks.
        
        Args:
            document_id: Document identifier
            chunks: List of text chunks
            chunk_ids: List of chunk identifiers
            model: Optional model name to use
            
        Returns:
            Batch of embeddings
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        pass
    
    @abstractmethod
    async def update_document_embeddings(
        self,
        document_id: DocumentId,
        chunks: List[str],
        chunk_ids: List[str],
        model: Optional[str] = None
    ) -> EmbeddingBatch:
        """
        Update embeddings for a document.
        
        Args:
            document_id: Document identifier
            chunks: Updated text chunks
            chunk_ids: List of chunk identifiers
            model: Optional model name to use
            
        Returns:
            Batch of updated embeddings
            
        Raises:
            EmbeddingError: If embedding update fails
        """
        pass
    
    @abstractmethod
    async def delete_document_embeddings(
        self,
        document_id: DocumentId
    ) -> bool:
        """
        Delete all embeddings for a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            True if deletion was successful
            
        Raises:
            EmbeddingError: If deletion fails
        """
        pass


class SimilaritySearchService(ABC):
    """
    Abstract interface for similarity search operations.
    
    Handles vector similarity search following
    Interface Segregation Principle.
    """
    
    @abstractmethod
    async def similarity_search(
        self,
        query_embedding: EmbeddingVector,
        limit: int = 10,
        threshold: Optional[float] = None,
        document_filter: Optional[DocumentId] = None
    ) -> List[TextEmbedding]:
        """
        Perform similarity search using embedding vector.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            document_filter: Optional document filter
            
        Returns:
            List of similar text embeddings
            
        Raises:
            SearchError: If search fails
        """
        pass
    
    @abstractmethod
    async def similarity_search_by_text(
        self,
        query_text: str,
        limit: int = 10,
        threshold: Optional[float] = None,
        document_filter: Optional[DocumentId] = None,
        model: Optional[str] = None
    ) -> List[TextEmbedding]:
        """
        Perform similarity search using text query.
        
        Args:
            query_text: Query text
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            document_filter: Optional document filter
            model: Optional model for query embedding
            
        Returns:
            List of similar text embeddings
            
        Raises:
            SearchError: If search fails
        """
        pass
    
    @abstractmethod
    async def get_similar_documents(
        self,
        query_text: str,
        limit: int = 5,
        model: Optional[str] = None
    ) -> List[DocumentId]:
        """
        Get documents similar to query text.
        
        Args:
            query_text: Query text
            limit: Maximum number of documents
            model: Optional model for query embedding
            
        Returns:
            List of similar document IDs
            
        Raises:
            SearchError: If search fails
        """
        pass


class EmbeddingServiceFactory(ABC):
    """
    Abstract factory for creating embedding service instances.
    
    Follows Open/Closed Principle for extensibility.
    """
    
    @abstractmethod
    def create_embedding_service(self, config: Dict[str, Any]) -> EmbeddingService:
        """
        Create embedding service instance.
        
        Args:
            config: Service configuration
            
        Returns:
            Configured embedding service
        """
        pass
    
    @abstractmethod
    def create_document_embedding_service(
        self,
        embedding_service: EmbeddingService,
        config: Dict[str, Any]
    ) -> DocumentEmbeddingService:
        """
        Create document embedding service instance.
        
        Args:
            embedding_service: Base embedding service
            config: Service configuration
            
        Returns:
            Configured document embedding service
        """
        pass
    
    @abstractmethod
    def create_similarity_search_service(
        self,
        config: Dict[str, Any]
    ) -> SimilaritySearchService:
        """
        Create similarity search service instance.
        
        Args:
            config: Service configuration
            
        Returns:
            Configured similarity search service
        """
        pass
