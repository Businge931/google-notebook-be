"""
OpenAI Embedding Service Implementation

"""
from typing import List, Optional, Dict, Any
import asyncio
import logging
from datetime import datetime

try:
    import openai
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from src.core.domain.services.embedding_service import (
    EmbeddingService,
    DocumentEmbeddingService,
    EmbeddingVector,
    TextEmbedding,
    EmbeddingBatch
)
from src.core.domain.value_objects import DocumentId
from src.shared.exceptions import EmbeddingError, ConfigurationError
from src.shared.constants import AIConstants


class OpenAIEmbeddingService(EmbeddingService):
    """
    OpenAI implementation of EmbeddingService following Single Responsibility Principle.
    
    Handles text embedding using OpenAI's embedding models.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        max_retries: int = 3,
        timeout: float = 30.0
    ):
        """
        Initialize OpenAI embedding service.
        
        Args:
            api_key: OpenAI API key
            model: Default embedding model
            max_retries: Maximum number of retries
            timeout: Request timeout in seconds
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available")
        
        if not api_key:
            raise ConfigurationError("OpenAI API key is required")
        
        self._client = AsyncOpenAI(
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout
        )
        self._default_model = model
        self._logger = logging.getLogger(__name__)
        
        # Model configurations
        self._model_configs = {
            "text-embedding-3-small": {"dimension": 1536, "max_tokens": 8192},
            "text-embedding-3-large": {"dimension": 3072, "max_tokens": 8192},
            "text-embedding-ada-002": {"dimension": 1536, "max_tokens": 8192}
        }
    
    async def generate_embedding(
        self,
        text: str,
        model: Optional[str] = None
    ) -> EmbeddingVector:
        """
        Generate embedding for a single text using OpenAI.
        
        Args:
            text: Text to embed
            model: Optional model name to use
            
        Returns:
            Embedding vector
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            model_name = model or self._default_model
            
            # Validate model
            if model_name not in self._model_configs:
                raise EmbeddingError(f"Unsupported model: {model_name}")
            
            # Validate text length
            if len(text.strip()) == 0:
                raise EmbeddingError("Text cannot be empty")
            
            # Truncate text if too long
            max_tokens = self._model_configs[model_name]["max_tokens"]
            if len(text) > max_tokens * 4:  # Rough estimate: 1 token ≈ 4 characters
                text = text[:max_tokens * 4]
                self._logger.warning(f"Text truncated to {max_tokens * 4} characters")
            
            # Generate embedding
            response = await self._client.embeddings.create(
                model=model_name,
                input=text,
                encoding_format="float"
            )
            
            # Extract embedding data
            embedding_data = response.data[0].embedding
            dimension = self._model_configs[model_name]["dimension"]
            
            return EmbeddingVector(
                vector=embedding_data,
                dimension=dimension,
                model=model_name,
                metadata={
                    "usage": response.usage.model_dump() if response.usage else None,
                    "created_at": datetime.utcnow().isoformat()
                }
            )
            
        except openai.APIError as e:
            self._logger.error(f"OpenAI API error: {e}")
            raise EmbeddingError(f"OpenAI API error: {str(e)}")
        except Exception as e:
            self._logger.error(f"Embedding generation failed: {e}")
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
            model: Optional model name to use
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            if not texts:
                return []
            
            model_name = model or self._default_model
            
            # Validate model
            if model_name not in self._model_configs:
                raise EmbeddingError(f"Unsupported model: {model_name}")
            
            # Filter and validate texts
            valid_texts = []
            for text in texts:
                if text and text.strip():
                    # Truncate if too long
                    max_tokens = self._model_configs[model_name]["max_tokens"]
                    if len(text) > max_tokens * 4:
                        text = text[:max_tokens * 4]
                    valid_texts.append(text)
            
            if not valid_texts:
                raise EmbeddingError("No valid texts to embed")
            
            # Process in batches to avoid API limits
            batch_size = 100  # OpenAI batch limit
            all_embeddings = []
            
            for i in range(0, len(valid_texts), batch_size):
                batch_texts = valid_texts[i:i + batch_size]
                
                # Generate embeddings for batch
                response = await self._client.embeddings.create(
                    model=model_name,
                    input=batch_texts,
                    encoding_format="float"
                )
                
                # Extract embeddings
                dimension = self._model_configs[model_name]["dimension"]
                
                for embedding_data in response.data:
                    embedding_vector = EmbeddingVector(
                        vector=embedding_data.embedding,
                        dimension=dimension,
                        model=model_name,
                        metadata={
                            "usage": response.usage.model_dump() if response.usage else None,
                            "created_at": datetime.utcnow().isoformat(),
                            "batch_index": embedding_data.index
                        }
                    )
                    all_embeddings.append(embedding_vector)
            
            return all_embeddings
            
        except openai.APIError as e:
            self._logger.error(f"OpenAI API error in batch: {e}")
            raise EmbeddingError(f"OpenAI API error: {str(e)}")
        except Exception as e:
            self._logger.error(f"Batch embedding generation failed: {e}")
            raise EmbeddingError(f"Batch embedding generation failed: {str(e)}")
    
    def get_embedding_dimension(self, model: Optional[str] = None) -> int:
        """
        Get the dimension of embeddings for a model.
        
        Args:
            model: Optional model name
            
        Returns:
            Embedding dimension
        """
        model_name = model or self._default_model
        
        if model_name not in self._model_configs:
            raise EmbeddingError(f"Unsupported model: {model_name}")
        
        return self._model_configs[model_name]["dimension"]
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available embedding models.
        
        Returns:
            List of model names
        """
        return list(self._model_configs.keys())
    
    async def health_check(self) -> bool:
        """
        Check if the OpenAI embedding service is healthy.
        
        Returns:
            True if service is healthy
        """
        try:
            # Test with a simple embedding request
            test_embedding = await self.generate_embedding("health check")
            return len(test_embedding.vector) > 0
        except Exception as e:
            self._logger.error(f"Health check failed: {e}")
            return False


class OpenAIDocumentEmbeddingService(DocumentEmbeddingService):
    """
    OpenAI implementation of DocumentEmbeddingService.
    
    Handles document-level embedding operations using OpenAI.
    """
    
    def __init__(
        self,
        embedding_service: OpenAIEmbeddingService,
        max_concurrent: int = 10
    ):
        """
        Initialize document embedding service.
        
        Args:
            embedding_service: Base OpenAI embedding service
            max_concurrent: Maximum concurrent embedding requests
        """
        self._embedding_service = embedding_service
        self._max_concurrent = max_concurrent
        self._logger = logging.getLogger(__name__)
    
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
        try:
            if len(chunks) != len(chunk_ids):
                raise EmbeddingError("Number of chunks must match number of chunk IDs")
            
            if not chunks:
                return EmbeddingBatch(embeddings=[], model=model or "unknown")
            
            start_time = datetime.utcnow()
            
            # Generate embeddings for all chunks
            embedding_vectors = await self._embedding_service.generate_embeddings_batch(
                chunks, model
            )
            
            # Create TextEmbedding objects
            text_embeddings = []
            total_tokens = 0
            
            for i, (chunk, chunk_id, embedding_vector) in enumerate(zip(chunks, chunk_ids, embedding_vectors)):
                text_embedding = TextEmbedding(
                    text=chunk,
                    embedding=embedding_vector,
                    chunk_id=chunk_id,
                    document_id=document_id,
                    metadata={
                        "chunk_index": i,
                        "chunk_length": len(chunk),
                        "document_id": document_id.value
                    }
                )
                text_embeddings.append(text_embedding)
                
                # Accumulate token usage if available
                if embedding_vector.metadata and "usage" in embedding_vector.metadata:
                    usage = embedding_vector.metadata["usage"]
                    if usage and "total_tokens" in usage:
                        total_tokens += usage["total_tokens"]
            
            end_time = datetime.utcnow()
            processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            self._logger.info(
                f"Generated embeddings for {len(chunks)} chunks from document "
                f"{document_id.value} in {processing_time_ms}ms"
            )
            
            return EmbeddingBatch(
                embeddings=text_embeddings,
                model=model or self._embedding_service._default_model,
                total_tokens=total_tokens if total_tokens > 0 else None,
                processing_time_ms=processing_time_ms
            )
            
        except Exception as e:
            self._logger.error(f"Document embedding failed for {document_id.value}: {e}")
            raise EmbeddingError(f"Document embedding failed: {str(e)}")
    
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
        # For now, updating is the same as creating new embeddings
        # In a full implementation, this might involve deleting old embeddings first
        return await self.embed_document_chunks(document_id, chunks, chunk_ids, model)
    
    async def delete_document_embeddings(
        self,
        document_id: DocumentId
    ) -> bool:
        """
        Delete all embeddings for a document.
        
        Note: This is a placeholder implementation.
        In a full system, this would interact with the vector database.
        
        Args:
            document_id: Document identifier
            
        Returns:
            True if deletion was successful
        """
        try:
            self._logger.info(f"Deleting embeddings for document {document_id.value}")
            # Placeholder - would interact with vector database
            return True
        except Exception as e:
            self._logger.error(f"Failed to delete embeddings for {document_id.value}: {e}")
            return False


class OpenAIEmbeddingServiceFactory:
    """
    Factory for creating OpenAI embedding service instances.
    
    Follows Open/Closed Principle for extensibility.
    """
    
    @staticmethod
    def create_embedding_service(
        api_key: str,
        model: str = "text-embedding-3-small",
        max_retries: int = 3,
        timeout: float = 30.0
    ) -> OpenAIEmbeddingService:
        """
        Create OpenAI embedding service instance.
        
        Args:
            api_key: OpenAI API key
            model: Default embedding model
            max_retries: Maximum number of retries
            timeout: Request timeout in seconds
            
        Returns:
            Configured OpenAI embedding service
        """
        return OpenAIEmbeddingService(
            api_key=api_key,
            model=model,
            max_retries=max_retries,
            timeout=timeout
        )
    
    @staticmethod
    def create_document_embedding_service(
        embedding_service: OpenAIEmbeddingService,
        max_concurrent: int = 10
    ) -> OpenAIDocumentEmbeddingService:
        """
        Create OpenAI document embedding service instance.
        
        Args:
            embedding_service: Base OpenAI embedding service
            max_concurrent: Maximum concurrent embedding requests
            
        Returns:
            Configured document embedding service
        """
        return OpenAIDocumentEmbeddingService(
            embedding_service=embedding_service,
            max_concurrent=max_concurrent
        )
