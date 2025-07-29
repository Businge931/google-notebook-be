"""
Simple Local Embedding Service Implementation

Provides basic text embeddings using TF-IDF and simple text processing.
No external dependencies or API keys required - completely local and free.
"""
import logging
import hashlib
from typing import List, Optional, Dict, Any
import asyncio
import re
from collections import Counter
import math

from ....core.domain.services.embedding_service import (
    EmbeddingService,
    EmbeddingVector,
    DocumentEmbeddingService,
    TextEmbedding
)
from ....core.domain.value_objects import DocumentId
from ....shared.exceptions import EmbeddingError, ConfigurationError


class SimpleLocalEmbeddingService(EmbeddingService):
    """
    Simple local embedding service using TF-IDF-like approach.
    
    Completely local, no dependencies, no API keys required.
    Good for testing and development when external services are unavailable.
    """
    
    def __init__(
        self,
        embedding_dimension: int = 384,
        vocabulary_size: int = 10000
    ):
        """
        Initialize simple local embedding service.
        
        Args:
            embedding_dimension: Dimension of output embeddings
            vocabulary_size: Maximum vocabulary size for feature extraction
        """
        self._embedding_dimension = embedding_dimension
        self._vocabulary_size = vocabulary_size
        self._logger = logging.getLogger(__name__)
        
        # Simple vocabulary for consistent embeddings
        self._vocabulary: Dict[str, int] = {}
        self._idf_scores: Dict[str, float] = {}
        
        self._logger.info(f"Initialized Simple Local embedding service with dimension: {embedding_dimension}")
    
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Simple text preprocessing.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Convert to lowercase and extract words
        text = text.lower()
        # Remove special characters and split
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        # Filter out very short words
        words = [word for word in words if len(word) > 2]
        return words
    
    def _build_vocabulary(self, texts: List[str]) -> None:
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of texts to build vocabulary from
        """
        word_counts = Counter()
        doc_counts = Counter()
        
        # Count words and document frequencies
        for text in texts:
            words = self._preprocess_text(text)
            word_counts.update(words)
            doc_counts.update(set(words))  # Unique words per document
        
        # Select most common words for vocabulary
        most_common = word_counts.most_common(self._vocabulary_size)
        self._vocabulary = {word: idx for idx, (word, _) in enumerate(most_common)}
        
        # Calculate IDF scores
        total_docs = len(texts)
        for word in self._vocabulary:
            df = doc_counts.get(word, 1)
            self._idf_scores[word] = math.log(total_docs / df)
    
    def _text_to_vector(self, text: str) -> List[float]:
        """
        Convert text to embedding vector using TF-IDF-like approach.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        words = self._preprocess_text(text)
        word_counts = Counter(words)
        total_words = len(words)
        
        # Create TF-IDF vector
        vector = [0.0] * self._embedding_dimension
        
        for word, count in word_counts.items():
            if word in self._vocabulary:
                vocab_idx = self._vocabulary[word]
                if vocab_idx < self._embedding_dimension:
                    # TF-IDF score
                    tf = count / total_words
                    idf = self._idf_scores.get(word, 1.0)
                    tfidf = tf * idf
                    vector[vocab_idx] = tfidf
        
        # Add some basic text features to remaining dimensions
        if self._embedding_dimension > len(self._vocabulary):
            # Text length feature
            text_length_norm = min(len(text) / 1000.0, 1.0)
            vector[-4] = text_length_norm
            
            # Word count feature
            word_count_norm = min(len(words) / 100.0, 1.0)
            vector[-3] = word_count_norm
            
            # Average word length
            avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
            vector[-2] = min(avg_word_len / 10.0, 1.0)
            
            # Text hash for uniqueness
            text_hash = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            vector[-1] = (text_hash % 1000) / 1000.0
        
        return vector
    
    async def generate_embedding(
        self,
        text: str,
        model: Optional[str] = None
    ) -> EmbeddingVector:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            model: Ignored (only one model available)
            
        Returns:
            Embedding vector
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            # Build vocabulary if not exists (using single text)
            if not self._vocabulary:
                self._build_vocabulary([text])
            
            # Generate embedding
            vector = self._text_to_vector(text)
            
            return EmbeddingVector(
                vector=vector,
                dimension=self._embedding_dimension,
                model="simple-local-tfidf",
                metadata={"text_length": len(text)}
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
            model: Ignored (only one model available)
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            if not texts:
                return []
            
            self._logger.info(f"Generating embeddings for {len(texts)} texts")
            
            # Build vocabulary from all texts
            self._build_vocabulary(texts)
            
            # Generate embeddings for all texts
            result = []
            for i, text in enumerate(texts):
                vector = self._text_to_vector(text)
                result.append(EmbeddingVector(
                    vector=vector,
                    dimension=self._embedding_dimension,
                    model="simple-local-tfidf",
                    metadata={
                        "text_length": len(text),
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
        Get the dimension of embeddings.
        
        Args:
            model: Ignored
            
        Returns:
            Embedding dimension
        """
        return self._embedding_dimension
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available embedding models.
        
        Returns:
            List of model names
        """
        return ["simple-local-tfidf"]
    
    async def health_check(self) -> bool:
        """
        Check if the embedding service is healthy.
        
        Returns:
            True if service is healthy
        """
        try:
            # Test with a simple embedding
            test_embedding = await self.generate_embedding("test health check")
            return len(test_embedding.vector) == self._embedding_dimension
            
        except Exception as e:
            self._logger.error(f"Health check failed: {e}")
            return False


class SimpleLocalDocumentEmbeddingService(DocumentEmbeddingService):
    """
    Document-level embedding service using simple local embeddings.
    """
    
    def __init__(
        self,
        embedding_service: SimpleLocalEmbeddingService,
        max_concurrent: int = 10
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


class SimpleLocalEmbeddingServiceFactory:
    """Factory for creating simple local embedding services."""
    
    @staticmethod
    def create_embedding_service(
        embedding_dimension: int = 384,
        vocabulary_size: int = 10000
    ) -> SimpleLocalEmbeddingService:
        """
        Create a simple local embedding service.
        
        Args:
            embedding_dimension: Dimension of embeddings
            vocabulary_size: Maximum vocabulary size
            
        Returns:
            Configured embedding service
        """
        return SimpleLocalEmbeddingService(
            embedding_dimension=embedding_dimension,
            vocabulary_size=vocabulary_size
        )
    
    @staticmethod
    def create_document_embedding_service(
        embedding_service: SimpleLocalEmbeddingService,
        max_concurrent: int = 10
    ) -> SimpleLocalDocumentEmbeddingService:
        """
        Create a document embedding service.
        
        Args:
            embedding_service: Base embedding service
            max_concurrent: Maximum concurrent operations
            
        Returns:
            Configured document embedding service
        """
        return SimpleLocalDocumentEmbeddingService(
            embedding_service=embedding_service,
            max_concurrent=max_concurrent
        )
