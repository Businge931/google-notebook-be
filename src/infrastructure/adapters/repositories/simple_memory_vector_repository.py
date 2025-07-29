"""
Simple Memory Vector Repository Implementation

"""
import logging
import math
from typing import List, Optional, Dict, Any
import asyncio

from ....core.domain.repositories.vector_repository import (
    VectorRepository,
    VectorSearchResult
)
from ....core.domain.services.embedding_service import EmbeddingService
from ....core.domain.value_objects import DocumentId
from ....shared.exceptions import VectorRepositoryError


class SimpleMemoryVectorRepository(VectorRepository):
    """
    Simple in-memory vector repository using cosine similarity.
    
    Stores vectors in memory and performs brute-force similarity search.
    Perfect for development and testing when FAISS is overkill.
    """
    
    def __init__(self, embedding_service: EmbeddingService):
        """Initialize the memory vector repository.
        
        Args:
            embedding_service: Service for generating query embeddings
        """
        self._vectors: Dict[str, Dict[str, Any]] = {}  # document_id -> chunk_id -> data
        self._embedding_service = embedding_service
        self._logger = logging.getLogger(__name__)
        
        self._logger.info("Initialized Simple Memory Vector Repository with embedding service")
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        if len(vec1) != len(vec2):
            return 0.0
        
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        # Avoid division by zero
        if magnitude1 == 0.0 or magnitude2 == 0.0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = dot_product / (magnitude1 * magnitude2)
        
        # Normalize to 0-1 range (cosine similarity can be -1 to 1)
        return (similarity + 1.0) / 2.0
    
    async def store_document_vectors(
        self,
        document_id: DocumentId,
        chunks: List[Dict[str, Any]]
    ) -> bool:
        """
        Store document text chunks as vectors.
        
        Args:
            document_id: Document identifier
            chunks: List of text chunks with metadata and vectors
                
        Returns:
            True if vectors were stored successfully
            
        Raises:
            VectorRepositoryError: If storage operation fails
        """
        try:
            doc_id = document_id.value
            self._logger.error(f" VECTOR STORAGE: Starting storage for document {doc_id} with {len(chunks)} chunks")
            
            # Initialize document storage if not exists
            if doc_id not in self._vectors:
                self._vectors[doc_id] = {}
                self._logger.error(f" VECTOR STORAGE: Initialized storage for document {doc_id}")
                
            # Store each chunk
            for chunk in chunks:
                chunk_id = chunk.get('chunk_id')
                vector = chunk.get('vector')
                
                if not chunk_id or not vector:
                    raise VectorRepositoryError(f"Missing chunk_id or vector in chunk data")
                
                # Store chunk data with vector
                self._vectors[doc_id][chunk_id] = {
                    'chunk_id': chunk_id,
                    'text_content': chunk.get('text_content', ''),
                    'page_number': chunk.get('page_number', 0),
                    'start_position': chunk.get('start_position', 0),
                    'end_position': chunk.get('end_position', 0),
                    'vector': vector,
                    'metadata': chunk.get('metadata', {})
                }
            
            self._logger.error(f" VECTOR STORAGE: Successfully stored {len(chunks)} vectors for document {doc_id}")
            self._logger.error(f" VECTOR STORAGE: Repository now has {len(self._vectors)} documents with {sum(len(chunks) for chunks in self._vectors.values())} total chunks")
            self._logger.info(f"Stored {len(chunks)} vectors for document {doc_id}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to store vectors for document {document_id.value}: {e}")
            raise VectorRepositoryError(f"Failed to store vectors: {str(e)}")
    
    async def search_similar(
        self,
        query_text: str,
        document_ids: Optional[List[DocumentId]] = None,
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[VectorSearchResult]:
        """
        Search for similar vectors using embedding-based semantic similarity.
        
        Args:
            query_text: Text to search for
            document_ids: Optional list of document IDs to filter by
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            List of search results sorted by similarity score
        """
        try:
            # Debug: Check if we have any stored vectors
            total_docs = len(self._vectors)
            total_chunks = sum(len(chunks) for chunks in self._vectors.values())
            self._logger.info(
                f"Starting similarity search for query: '{query_text}' "
                f"(stored docs: {total_docs}, total chunks: {total_chunks})"
            )
            
            # Generate embedding for the query text
            self._logger.info(f"Generating embedding for query: '{query_text}'")
            query_embedding = await self._embedding_service.generate_embedding(query_text)
            
            if not query_embedding:
                self._logger.warning("Failed to generate query embedding, falling back to text search")
                return await self._fallback_text_search(
                    query_text, document_ids, limit, similarity_threshold
                )
            
            results = []
            
            for doc_id, chunks in self._vectors.items():
                # Filter by document IDs if specified
                if document_ids and DocumentId(doc_id) not in document_ids:
                    continue
                    
                for chunk_id, chunk_data in chunks.items():
                    try:
                        # Get stored embedding for this chunk (stored as 'vector' field)
                        chunk_embedding = chunk_data.get('vector') or chunk_data.get('embedding')
                        if not chunk_embedding:
                            self._logger.debug(f"No vector/embedding found for chunk {chunk_id}, skipping")
                            continue
                        
                        # Debug: Check the type of stored vector
                        self._logger.error(f"🔥 SIMILARITY SEARCH: chunk {chunk_id} vector type={type(chunk_embedding)}, has_len={hasattr(chunk_embedding, '__len__')}")
                        
                        # Ensure both embeddings are raw arrays (not EmbeddingVector objects)
                        if hasattr(query_embedding, 'vector'):
                            query_vector = query_embedding.vector
                        else:
                            query_vector = query_embedding
                            
                        if hasattr(chunk_embedding, 'vector'):
                            chunk_vector = chunk_embedding.vector
                        else:
                            chunk_vector = chunk_embedding
                        
                        # Debug: Check final vector types
                        self._logger.error(f"🔥 FINAL VECTORS: query type={type(query_vector)}, chunk type={type(chunk_vector)}")
                        
                        # Calculate cosine similarity between query and chunk embeddings
                        similarity_score = self._cosine_similarity(query_vector, chunk_vector)
                        
                        # Only include results above threshold
                        if similarity_score >= similarity_threshold:
                            metadata = chunk_data.get('metadata', {})
                            result = VectorSearchResult(
                                chunk_id=chunk_id,
                                document_id=DocumentId(doc_id),
                                page_number=metadata.get('page_number', 0),
                                text_content=chunk_data.get('text_content', ''),
                                similarity_score=similarity_score,
                                metadata={
                                    **metadata,
                                    'document_title': metadata.get('document_title', 'Unknown'),
                                    'document_filename': metadata.get('document_filename', 'Unknown'),
                                    'start_position': chunk_data.get('start_position', 0),
                                    'end_position': chunk_data.get('end_position', 0)
                                }
                            )
                            results.append(result)
                            
                    except Exception as e:
                        self._logger.warning(f"Error processing chunk {chunk_id}: {e}")
                        continue
            
            # Sort by similarity score (descending) and limit results
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            limited_results = results[:limit]
            
            self._logger.info(
                f"Found {len(limited_results)} similar chunks for query: '{query_text}' "
                f"using embedding-based search (threshold: {similarity_threshold})"
            )
            
            return limited_results
            
        except Exception as e:
            self._logger.error(f"Error in embedding-based similarity search: {e}")
            # Fall back to text-based search if embedding search fails
            self._logger.info("Falling back to text-based similarity search")
            return await self._fallback_text_search(
                query_text, document_ids, limit, similarity_threshold
            )
    
    async def _fallback_text_search(
        self,
        query_text: str,
        document_ids: Optional[List[DocumentId]] = None,
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[VectorSearchResult]:
        """
        Fallback text-based similarity search when embedding generation fails.
        
        Args:
            query_text: Text to search for
            document_ids: Optional list of document IDs to filter by
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            List of search results sorted by similarity score
        """
        results = []
        query_lower = query_text.lower().strip()
        query_words = set(query_lower.split())
        
        # Check if this is a title or page-related query
        is_title_query = any(word in query_lower for word in ['title', 'name', 'called', 'document'])
        is_page_query = any(word in query_lower for word in ['page', 'pages', 'how many'])
        
        for doc_id, chunks in self._vectors.items():
            # Filter by document IDs if specified
            if document_ids and DocumentId(doc_id) not in document_ids:
                continue
                
            for chunk_id, chunk_data in chunks.items():
                try:
                    # Get text content for similarity calculation
                    text_content = chunk_data.get('text_content', '').lower()
                    metadata = chunk_data.get('metadata', {})
                    
                    # Calculate base similarity score
                    similarity_score = 0.0
                    
                    # 1. Exact phrase matching (highest weight)
                    if query_lower in text_content:
                        similarity_score += 0.8
                    
                    # 2. Word overlap scoring
                    text_words = set(text_content.split())
                    word_overlap = len(query_words.intersection(text_words))
                    if len(query_words) > 0:
                        word_similarity = word_overlap / len(query_words)
                        similarity_score += word_similarity * 0.4
                    
                    # 3. Boost for title queries
                    if is_title_query:
                        # Check document title in metadata
                        doc_title = metadata.get('document_title', '').lower()
                        if any(word in doc_title for word in query_words):
                            similarity_score += 0.3
                        # Check if chunk contains title information
                        if any(word in text_content for word in ['title', 'heading', 'chapter']):
                            similarity_score += 0.2
                    
                    # 4. Boost for page queries
                    if is_page_query:
                        page_number = metadata.get('page_number')
                        if page_number is not None:
                            similarity_score += 0.2
                        if any(word in text_content for word in ['page', 'pages', 'total']):
                            similarity_score += 0.1
                    
                    # Only include results above threshold
                    if similarity_score >= similarity_threshold:
                        result = VectorSearchResult(
                            chunk_id=chunk_id,
                            document_id=doc_id,
                            document_title=metadata.get('document_title', 'Unknown'),
                            document_filename=metadata.get('document_filename', 'Unknown'),
                            page_number=metadata.get('page_number'),
                            text_content=chunk_data.get('text_content', ''),
                            start_position=chunk_data.get('start_position', 0),
                            end_position=chunk_data.get('end_position', 0),
                            similarity_score=min(similarity_score, 1.0),  # Cap at 1.0
                            metadata=metadata
                        )
                        results.append(result)
                        
                except Exception as e:
                    self._logger.warning(f"Error processing chunk {chunk_id} in fallback search: {e}")
                    continue
        
        # Sort by similarity score (descending) and limit results
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        limited_results = results[:limit]
        
        self._logger.info(
            f"Found {len(limited_results)} similar chunks for query: '{query_text}' "
            f"using fallback text search (threshold: {similarity_threshold})"
        )
        
        return limited_results
    
    async def delete_document_vectors(self, document_id: DocumentId) -> bool:
        """
        Delete all vectors for a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            True if vectors were deleted successfully
        """
        try:
            doc_id = document_id.value
            
            if doc_id in self._vectors:
                del self._vectors[doc_id]
                self._logger.info(f"Deleted vectors for document {doc_id}")
            
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to delete vectors for document {document_id.value}: {e}")
            raise VectorRepositoryError(f"Failed to delete vectors: {str(e)}")
    
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
        """
        try:
            all_results = []
            
            # Determine which documents to search
            search_doc_ids = [doc_id.value for doc_id in document_ids] if document_ids else list(self._vectors.keys())
            
            # Search each document
            for doc_id in search_doc_ids:
                if doc_id in self._vectors:
                    document_id = DocumentId(doc_id)
                    doc_results = await self.search_similar(
                        query_text=query_text,
                        document_ids=[document_id],
                        limit=limit,
                        similarity_threshold=similarity_threshold
                    )
                    all_results.extend(doc_results)
            
            # Sort all results by similarity score and limit
            all_results.sort(key=lambda x: x.similarity_score, reverse=True)
            return all_results[:limit]
            
        except Exception as e:
            self._logger.error(f"Failed to search across documents: {e}")
            raise VectorRepositoryError(f"Failed to search across documents: {str(e)}")
    
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
        """
        try:
            # Delete existing vectors
            await self.delete_document_vectors(document_id)
            
            # Store new vectors
            return await self.store_document_vectors(document_id, chunks)
            
        except Exception as e:
            self._logger.error(f"Failed to update vectors for document {document_id.value}: {e}")
            raise VectorRepositoryError(f"Failed to update vectors: {str(e)}")
    
    async def get_document_chunk_count(self, document_id: DocumentId) -> int:
        """
        Get the number of chunks stored for a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Number of chunks stored for the document
        """
        doc_id = document_id.value
        return len(self._vectors.get(doc_id, {}))
    
    async def health_check(self) -> bool:
        """
        Check if the vector database is healthy and accessible.
        
        Returns:
            True if vector database is healthy
        """
        try:
            # Simple health check - verify we can access the storage
            total_documents = len(self._vectors)
            self._logger.debug(f"Health check: {total_documents} documents in memory")
            return True
            
        except Exception as e:
            self._logger.error(f"Health check failed: {e}")
            return False


class SimpleMemoryVectorRepositoryFactory:
    """Factory for creating simple memory vector repositories."""
    
    @staticmethod
    def create_vector_repository(embedding_service: EmbeddingService) -> SimpleMemoryVectorRepository:
        """
        Create a simple memory vector repository with embedding service.
        
        Args:
            embedding_service: Service for generating query embeddings
        
        Returns:
            Configured vector repository
        """
        return SimpleMemoryVectorRepository(embedding_service)
