"""
Hugging Face Embedding Service Implementation

"""
import logging
from typing import List, Optional, Dict, Any
import asyncio
from functools import lru_cache

try:
    from sentence_transformers import SentenceTransformer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    torch = None
    TRANSFORMERS_AVAILABLE = False

from ....core.domain.services.embedding_service import (
    EmbeddingService,
    EmbeddingVector,
    DocumentEmbeddingService,
    TextEmbedding
)
from ....core.domain.value_objects import DocumentId
from ....shared.exceptions import EmbeddingError, ConfigurationError


class HuggingFaceEmbeddingService(EmbeddingService):
    """
    Hugging Face embedding service using sentence-transformers.
    
    Runs locally without requiring API keys or internet connection
    after initial model download.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        max_seq_length: int = 512
    ):
        """
        Initialize Hugging Face embedding service.
        
        Args:
            model_name: Name of the sentence-transformers model
            device: Device to run on ('cpu', 'cuda', or None for auto)
            max_seq_length: Maximum sequence length for the model
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ConfigurationError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        self._model_name = model_name
        self._max_seq_length = max_seq_length
        self._logger = logging.getLogger(__name__)
        
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        
        # Initialize model (lazy loading)
        self._model: Optional[SentenceTransformer] = None
        
        self._logger.info(f"Initialized HuggingFace embedding service with model: {model_name}")
    
    @property
    def model(self):
        """Get or initialize the sentence transformer model."""
        if self._model is None:
            try:
                self._logger.info(f"Loading model: {self._model_name}")
                self._model = SentenceTransformer(self._model_name, device=self._device)
                self._model.max_seq_length = self._max_seq_length
                self._logger.info(f"Model loaded successfully on device: {self._device}")
            except Exception as e:
                self._logger.error(f"Failed to load model {self._model_name}: {e}")
                raise ConfigurationError(f"Failed to load embedding model: {str(e)}")
        
        return self._model
    
    async def generate_embedding(
        self,
        text: str,
        model: Optional[str] = None
    ) -> EmbeddingVector:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            model: Ignored (model is set during initialization)
            
        Returns:
            Embedding vector
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, 
                lambda: self.model.encode([text], convert_to_tensor=False)[0]
            )
            
            # Convert to list of floats
            embedding_list = embedding.tolist()
            
            return EmbeddingVector(
                vector=embedding_list,
                dimension=len(embedding_list),
                model=self._model_name,
                metadata={"device": self._device}
            )
            
        except Exception as e:
            self._logger.error(f"Failed to generate embedding: {e}")
            raise EmbeddingError(f"Embedding generation failed: {str(e)}")
    
    async def generate_embeddings_batch(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> List[EmbeddingVector]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            model: Ignored (model is set during initialization)
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            if not texts:
                return []
            
            self._logger.info(f"Generating embeddings for {len(texts)} texts")
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.model.encode(texts, convert_to_tensor=False)
            )
            
            # Convert to list of EmbeddingVector objects
            result = []
            for i, embedding in enumerate(embeddings):
                embedding_list = embedding.tolist()
                result.append(EmbeddingVector(
                    vector=embedding_list,
                    dimension=len(embedding_list),
                    model=self._model_name,
                    metadata={
                        "device": self._device,
                        "batch_index": i
                    }
                ))
            
            self._logger.info(f"Generated {len(result)} embeddings successfully")
            return result
            
        except Exception as e:
            self._logger.error(f"Failed to generate batch embeddings: {e}")
            raise EmbeddingError(f"Batch embedding generation failed: {str(e)}")
    
    def get_embedding_dimension(self, model: Optional[str] = None) -> int:
        """
        Get the dimension of embeddings for the model.
        
        Args:
            model: Ignored (dimension is determined by the loaded model)
            
        Returns:
            Embedding dimension
        """
        # For sentence-transformers/all-MiniLM-L6-v2, dimension is 384
        # We'll return the actual dimension from the model if loaded
        if self._model is not None:
            return self.model.get_sentence_embedding_dimension()
        else:
            # Default dimension for all-MiniLM-L6-v2
            return 384
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available embedding models.
        
        Returns:
            List of popular sentence-transformer model names
        """
        return [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/all-distilroberta-v1",
            "sentence-transformers/paraphrase-MiniLM-L6-v2",
            "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
        ]
    
    async def health_check(self) -> bool:
        """
        Check if the embedding service is healthy.
        
        Returns:
            True if service is healthy
        """
        try:
            # Test with a simple embedding
            test_embedding = await self.generate_embedding("test")
            return True
            
        except Exception as e:
            self._logger.error(f"Health check failed: {e}")
            return False


class HuggingFaceDocumentEmbeddingService(DocumentEmbeddingService):
    """
    Document-level embedding service using Hugging Face transformers.
    """
    
    def __init__(
        self,
        embedding_service: HuggingFaceEmbeddingService,
        max_concurrent: int = 5
    ):
        """
        Initialize document embedding service.
        
        Args:
            embedding_service: Base embedding service
            max_concurrent: Maximum concurrent embedding operations
        """
        self._embedding_service = embedding_service
        self._max_concurrent = max_concurrent
        self._logger = logging.getLogger(__name__)
    
    async def generate_document_embeddings(
        self,
        texts: List[str],
        document_id: DocumentId,
        chunk_ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[TextEmbedding]:
        """
        Generate embeddings for document chunks.
        
        Args:
            texts: List of text chunks
            document_id: Document identifier
            chunk_ids: List of chunk identifiers
            metadata: Optional metadata for each chunk
            
        Returns:
            List of text embeddings
        """
        try:
            if len(texts) != len(chunk_ids):
                raise ValueError("Number of texts must match number of chunk IDs")
            
            # Generate embeddings in batch
            embedding_vectors = await self._embedding_service.generate_embeddings_batch(texts)
            
            # Create TextEmbedding objects
            result = []
            for i, (text, chunk_id, embedding_vector) in enumerate(zip(texts, chunk_ids, embedding_vectors)):
                chunk_metadata = metadata[i] if metadata and i < len(metadata) else None
                
                text_embedding = TextEmbedding(
                    text=text,
                    embedding=embedding_vector,
                    chunk_id=chunk_id,
                    document_id=document_id,
                    metadata=chunk_metadata
                )
                result.append(text_embedding)
            
            self._logger.info(f"Generated embeddings for {len(result)} document chunks")
            return result
            
        except Exception as e:
            self._logger.error(f"Failed to generate document embeddings: {e}")
            raise EmbeddingError(f"Document embedding generation failed: {str(e)}")


class HuggingFaceEmbeddingServiceFactory:
    """Factory for creating Hugging Face embedding services."""
    
    @staticmethod
    def create_embedding_service(
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        max_seq_length: int = 512
    ) -> HuggingFaceEmbeddingService:
        """
        Create a Hugging Face embedding service.
        
        Args:
            model_name: Name of the sentence-transformers model
            device: Device to run on
            max_seq_length: Maximum sequence length
            
        Returns:
            Configured embedding service
        """
        return HuggingFaceEmbeddingService(
            model_name=model_name,
            device=device,
            max_seq_length=max_seq_length
        )
    
    @staticmethod
    def create_document_embedding_service(
        embedding_service: HuggingFaceEmbeddingService,
        max_concurrent: int = 5
    ) -> HuggingFaceDocumentEmbeddingService:
        """
        Create a document embedding service.
        
        Args:
            embedding_service: Base embedding service
            max_concurrent: Maximum concurrent operations
            
        Returns:
            Configured document embedding service
        """
        return HuggingFaceDocumentEmbeddingService(
            embedding_service=embedding_service,
            max_concurrent=max_concurrent
        )
