"""
FAISS Vector Repository Implementation

"""
import os
import json
import pickle
import asyncio
import logging
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from src.core.domain.repositories.vector_repository import (
    VectorRepository,
    VectorSearchResult
)
from src.core.domain.value_objects import DocumentId
from src.core.domain.services.embedding_service import EmbeddingService
from src.shared.exceptions import VectorRepositoryError, ConfigurationError


class FAISSVectorRepository(VectorRepository):
    """
    FAISS implementation of VectorRepository following Single Responsibility Principle.
    
    Handles vector storage and similarity search using Facebook AI Similarity Search (FAISS).
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        storage_path: str,
        index_type: str = "IVFFlat",
        embedding_dimension: int = 1536,
        nlist: int = 100,
        nprobe: int = 10
    ):
        """
        Initialize FAISS vector repository.
        
        Args:
            embedding_service: Service for generating embeddings
            storage_path: Path to store FAISS index and metadata
            index_type: Type of FAISS index (IVFFlat, Flat, HNSW)
            embedding_dimension: Dimension of embedding vectors
            nlist: Number of clusters for IVF index
            nprobe: Number of clusters to search
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS library not available")
        
        self._embedding_service = embedding_service
        self._storage_path = Path(storage_path)
        self._index_type = index_type
        self._embedding_dimension = embedding_dimension
        self._nlist = nlist
        self._nprobe = nprobe
        self._logger = logging.getLogger(__name__)
        
        # Create storage directory
        self._storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize FAISS index
        self._index = None
        self._metadata = {}  # Maps vector ID to chunk metadata
        self._document_chunks = {}  # Maps document_id to list of chunk IDs
        self._next_id = 0
        
        # Load existing index if available
        self._load_index()
    
    def _create_index(self) -> 'faiss.Index':
        """
        Create a new FAISS index based on configuration.
        
        Returns:
            Configured FAISS index
        """
        if not FAISS_AVAILABLE:
            raise ConfigurationError("FAISS is not available. Please install faiss-cpu or faiss-gpu.")
            
        if self._index_type == "Flat":
            # Exact search using L2 distance
            index = faiss.IndexFlatL2(self._embedding_dimension)
        elif self._index_type == "IVFFlat":
            # Approximate search using inverted file with flat quantizer
            quantizer = faiss.IndexFlatL2(self._embedding_dimension)
            index = faiss.IndexIVFFlat(quantizer, self._embedding_dimension, self._nlist)
            index.nprobe = self._nprobe
        elif self._index_type == "HNSW":
            # Hierarchical Navigable Small World graph
            index = faiss.IndexHNSWFlat(self._embedding_dimension, 32)
            index.hnsw.efConstruction = 40
            index.hnsw.efSearch = 16
        else:
            raise ConfigurationError(f"Unsupported index type: {self._index_type}")
        
        return index
    
    def _load_index(self):
        """Load existing FAISS index and metadata from storage."""
        try:
            index_path = self._storage_path / "faiss.index"
            metadata_path = self._storage_path / "metadata.json"
            
            self._logger.info(f"🔥 FAISS LOADING: Checking for existing index at {index_path}")
            self._logger.info(f"🔥 FAISS LOADING: Index exists: {index_path.exists()}, Metadata exists: {metadata_path.exists()}")
            print(f"🔥 FAISS PRINT: Checking for existing index at {index_path}")
            print(f"🔥 FAISS PRINT: Index exists: {index_path.exists()}, Metadata exists: {metadata_path.exists()}")
            
            if index_path.exists() and metadata_path.exists():
                # Load FAISS index
                self._index = faiss.read_index(str(index_path))
                
                # Load metadata
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                    self._metadata = data.get("metadata", {})
                    self._document_chunks = data.get("document_chunks", {})
                    self._next_id = data.get("next_id", 0)
                
                self._logger.info(f"🔥 FAISS LOADED: Successfully loaded FAISS index with {self._index.ntotal} vectors")
                self._logger.info(f"🔥 FAISS LOADED: Metadata entries: {len(self._metadata)}, Document chunks: {len(self._document_chunks)}")
                print(f"🔥 FAISS PRINT: Successfully loaded FAISS index with {self._index.ntotal} vectors")
                print(f"🔥 FAISS PRINT: Metadata entries: {len(self._metadata)}, Document chunks: {len(self._document_chunks)}")
            else:
                # Create new index
                self._index = self._create_index()
                self._logger.info(f"🔥 FAISS CREATED: Created new FAISS index: {self._index_type}")
                
        except Exception as e:
            self._logger.error(f"🔥 FAISS ERROR: Failed to load FAISS index: {e}")
            # Create new index as fallback
            self._index = self._create_index()
    
    def _save_index(self):
        """Save FAISS index and metadata to storage."""
        try:
            index_path = self._storage_path / "faiss.index"
            metadata_path = self._storage_path / "metadata.json"
            
            # Save FAISS index
            faiss.write_index(self._index, str(index_path))
            
            # Save metadata
            metadata_data = {
                "metadata": self._metadata,
                "document_chunks": self._document_chunks,
                "next_id": self._next_id,
                "last_updated": datetime.utcnow().isoformat()
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata_data, f, indent=2)
                
            self._logger.debug("Saved FAISS index and metadata")
            
        except Exception as e:
            self._logger.error(f"Failed to save FAISS index: {e}")
            raise VectorRepositoryError(f"Failed to save index: {str(e)}")
    
    async def store_document_vectors(
        self,
        document_id: DocumentId,
        chunks: List[Dict[str, Any]]
    ) -> bool:
        """
        Store document text chunks as vectors in FAISS index.
        
        Args:
            document_id: Document identifier
            chunks: List of text chunks with metadata
                
        Returns:
            True if vectors were stored successfully
            
        Raises:
            VectorRepositoryError: If storage operation fails
        """
        try:
            if not chunks:
                return True
            
            # Extract texts for embedding
            texts = [chunk["text_content"] for chunk in chunks]
            
            # Generate embeddings
            embedding_vectors = await self._embedding_service.generate_embeddings_batch(texts)
            
            # Prepare vectors for FAISS
            vectors = np.array([emb.vector for emb in embedding_vectors], dtype=np.float32)
            
            # Train index if needed (for IVF indices)
            if hasattr(self._index, 'is_trained') and not self._index.is_trained:
                if vectors.shape[0] >= self._nlist:
                    self._index.train(vectors)
                else:
                    # Not enough vectors to train, use a subset or create dummy vectors
                    dummy_vectors = np.random.random((self._nlist, self._embedding_dimension)).astype(np.float32)
                    self._index.train(dummy_vectors)
            
            # Add vectors to index
            start_id = self._next_id
            self._index.add(vectors)
            
            # Store metadata for each chunk
            chunk_ids = []
            doc_id_str = document_id.value
            
            for i, (chunk, embedding_vector) in enumerate(zip(chunks, embedding_vectors)):
                vector_id = start_id + i
                chunk_id = chunk["chunk_id"]
                chunk_ids.append(chunk_id)
                
                # Store chunk metadata (ensure JSON serializable)
                page_number = chunk.get("page_number", 1)
                # Convert PageNumber object to int if needed
                if hasattr(page_number, 'value'):
                    page_number = page_number.value
                elif not isinstance(page_number, (int, float)):
                    page_number = int(page_number) if page_number else 1
                
                self._metadata[str(vector_id)] = {
                    "chunk_id": chunk_id,
                    "document_id": doc_id_str,
                    "page_number": page_number,
                    "text_content": chunk["text_content"],
                    "start_position": chunk.get("start_position", 0),
                    "end_position": chunk.get("end_position", len(chunk["text_content"])),
                    "metadata": chunk.get("metadata", {}),
                    "embedding_model": embedding_vector.model,
                    "created_at": datetime.utcnow().isoformat()
                }
            
            # Update document chunks mapping
            if doc_id_str not in self._document_chunks:
                self._document_chunks[doc_id_str] = []
            self._document_chunks[doc_id_str].extend(chunk_ids)
            
            # Update next ID
            self._next_id += len(chunks)
            
            # Save index
            self._save_index()
            
            self._logger.info(
                f"Stored {len(chunks)} vectors for document {doc_id_str}"
            )
            
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to store vectors for document {document_id.value}: {e}")
            raise VectorRepositoryError(f"Failed to store vectors: {str(e)}")
    
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
        try:
            # Generate query embedding
            query_embedding = await self._embedding_service.generate_embedding(query_text)
            query_vector = np.array([query_embedding.vector], dtype=np.float32)
            
            # Search in FAISS index
            # Search more than needed to filter by document
            search_limit = min(limit * 10, self._index.ntotal)
            distances, indices = self._index.search(query_vector, search_limit)
            
            # Filter results by document and convert to similarity scores
            results = []
            doc_id_str = document_id.value
            
            for distance, idx in zip(distances[0], indices[0]):
                if idx == -1:  # No more results
                    break
                
                # Get metadata for this vector
                metadata = self._metadata.get(str(idx))
                if not metadata or metadata["document_id"] != doc_id_str:
                    continue
                
                # Convert L2 distance to similarity score (0-1)
                # Using exponential decay: similarity = exp(-distance)
                similarity_score = float(np.exp(-distance))
                
                if similarity_score < similarity_threshold:
                    continue
                
                # Create search result
                result = VectorSearchResult(
                    chunk_id=metadata["chunk_id"],
                    document_id=document_id,
                    page_number=metadata["page_number"],
                    text_content=metadata["text_content"],
                    similarity_score=similarity_score,
                    metadata={
                        "start_position": metadata["start_position"],
                        "end_position": metadata["end_position"],
                        "embedding_model": metadata["embedding_model"],
                        "distance": float(distance),
                        **metadata.get("metadata", {})
                    }
                )
                results.append(result)
                
                if len(results) >= limit:
                    break
            
            # Sort by similarity score (descending)
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            self._logger.debug(
                f"Found {len(results)} similar chunks for document {doc_id_str}"
            )
            
            return results
            
        except Exception as e:
            self._logger.error(f"Failed to search similar chunks: {e}")
            raise VectorRepositoryError(f"Failed to search similar chunks: {str(e)}")
    
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
        try:
            print(f"🔥 FAISS SEARCH: Starting search for query: '{query_text[:50]}...'")
            print(f"🔥 FAISS SEARCH: Index has {self._index.ntotal} vectors, threshold: {similarity_threshold}")
            
            # Generate query embedding
            query_embedding = await self._embedding_service.generate_embedding(query_text)
            query_vector = np.array([query_embedding.vector], dtype=np.float32)
            
            # Search in FAISS index
            search_limit = min(limit * 5, self._index.ntotal)
            print(f"🔥 FAISS SEARCH: Searching with limit: {search_limit}")
            distances, indices = self._index.search(query_vector, search_limit)
            print(f"🔥 FAISS SEARCH: Raw distances: {distances[0][:5]}, indices: {indices[0][:5]}")
            
            # Filter results by documents and convert to similarity scores
            results = []
            allowed_doc_ids = None
            if document_ids:
                allowed_doc_ids = {doc_id.value for doc_id in document_ids}
            
            for distance, idx in zip(distances[0], indices[0]):
                if idx == -1:  # No more results
                    break
                
                # Get metadata for this vector
                metadata = self._metadata.get(str(idx))
                if not metadata:
                    continue
                
                # Filter by document IDs if specified
                if allowed_doc_ids and metadata["document_id"] not in allowed_doc_ids:
                    continue
                
                # Convert L2 distance to similarity score
                similarity_score = float(np.exp(-distance))
                print(f"🔥 FAISS SEARCH: Vector {idx}, distance: {distance:.4f}, similarity: {similarity_score:.4f}, threshold: {similarity_threshold}")
                
                if similarity_score < similarity_threshold:
                    print(f"🔥 FAISS SEARCH: Filtered out vector {idx} (similarity {similarity_score:.4f} < threshold {similarity_threshold})")
                    continue
                
                # Create search result
                result = VectorSearchResult(
                    chunk_id=metadata["chunk_id"],
                    document_id=DocumentId(metadata["document_id"]),
                    page_number=metadata["page_number"],
                    text_content=metadata["text_content"],
                    similarity_score=similarity_score,
                    metadata={
                        "start_position": metadata["start_position"],
                        "end_position": metadata["end_position"],
                        "embedding_model": metadata["embedding_model"],
                        "distance": float(distance),
                        **metadata.get("metadata", {})
                    }
                )
                results.append(result)
                
                if len(results) >= limit:
                    break
            
            # Sort by similarity score (descending)
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            self._logger.debug(f"Found {len(results)} similar chunks across documents")
            
            return results
            
        except Exception as e:
            self._logger.error(f"Failed to search across documents: {e}")
            raise VectorRepositoryError(f"Failed to search across documents: {str(e)}")
    
    async def delete_document_vectors(self, document_id: DocumentId) -> bool:
        """
        Delete all vectors for a specific document.
        
        Note: FAISS doesn't support efficient deletion, so we rebuild the index.
        
        Args:
            document_id: Document identifier
            
        Returns:
            True if vectors were deleted successfully
            
        Raises:
            VectorRepositoryError: If delete operation fails
        """
        try:
            doc_id_str = document_id.value
            
            if doc_id_str not in self._document_chunks:
                return True  # No vectors to delete
            
            # Find vector IDs to delete
            chunk_ids_to_delete = set(self._document_chunks[doc_id_str])
            vector_ids_to_delete = []
            
            for vector_id, metadata in self._metadata.items():
                if metadata["chunk_id"] in chunk_ids_to_delete:
                    vector_ids_to_delete.append(int(vector_id))
            
            if not vector_ids_to_delete:
                return True
            
            # Rebuild index without deleted vectors
            remaining_vectors = []
            remaining_metadata = {}
            new_vector_id = 0
            
            for old_vector_id in range(self._index.ntotal):
                if old_vector_id not in vector_ids_to_delete:
                    # Get vector from index
                    vector = self._index.reconstruct(old_vector_id)
                    remaining_vectors.append(vector)
                    
                    # Update metadata with new ID
                    old_metadata = self._metadata.get(str(old_vector_id))
                    if old_metadata:
                        remaining_metadata[str(new_vector_id)] = old_metadata
                        new_vector_id += 1
            
            # Create new index
            new_index = self._create_index()
            
            if remaining_vectors:
                vectors_array = np.array(remaining_vectors, dtype=np.float32)
                
                # Train if needed
                if hasattr(new_index, 'is_trained') and not new_index.is_trained:
                    if vectors_array.shape[0] >= self._nlist:
                        new_index.train(vectors_array)
                    else:
                        dummy_vectors = np.random.random((self._nlist, self._embedding_dimension)).astype(np.float32)
                        new_index.train(dummy_vectors)
                
                new_index.add(vectors_array)
            
            # Update instance variables
            self._index = new_index
            self._metadata = remaining_metadata
            del self._document_chunks[doc_id_str]
            self._next_id = new_vector_id
            
            # Save updated index
            self._save_index()
            
            self._logger.info(f"Deleted vectors for document {doc_id_str}")
            
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to delete vectors for document {document_id.value}: {e}")
            raise VectorRepositoryError(f"Failed to delete vectors: {str(e)}")
    
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
            
        Raises:
            VectorRepositoryError: If count operation fails
        """
        try:
            doc_id_str = document_id.value
            return len(self._document_chunks.get(doc_id_str, []))
            
        except Exception as e:
            self._logger.error(f"Failed to get chunk count for document {document_id.value}: {e}")
            raise VectorRepositoryError(f"Failed to get chunk count: {str(e)}")
    
    async def health_check(self) -> bool:
        """
        Check if the FAISS vector database is healthy and accessible.
        
        Returns:
            True if vector database is healthy
            
        Raises:
            VectorRepositoryError: If health check fails
        """
        try:
            # Check if index is available
            if self._index is None:
                return False
            
            # Check if we can perform basic operations
            total_vectors = self._index.ntotal
            
            # Try a simple search if we have vectors
            if total_vectors > 0:
                dummy_vector = np.random.random((1, self._embedding_dimension)).astype(np.float32)
                _, _ = self._index.search(dummy_vector, min(1, total_vectors))
            
            return True
            
        except Exception as e:
            self._logger.error(f"FAISS health check failed: {e}")
            return False


class FAISSVectorRepositoryFactory:
    """
    Factory for creating FAISS vector repository instances.
    
    Follows Open/Closed Principle for extensibility.
    """
    
    @staticmethod
    def create_vector_repository(
        embedding_service: EmbeddingService,
        storage_path: str,
        index_type: str = "IVFFlat",
        embedding_dimension: int = 1536,
        nlist: int = 100,
        nprobe: int = 10
    ) -> FAISSVectorRepository:
        """
        Create FAISS vector repository instance.
        
        Args:
            embedding_service: Service for generating embeddings
            storage_path: Path to store FAISS index and metadata
            index_type: Type of FAISS index
            embedding_dimension: Dimension of embedding vectors
            nlist: Number of clusters for IVF index
            nprobe: Number of clusters to search
            
        Returns:
            Configured FAISS vector repository
        """
        return FAISSVectorRepository(
            embedding_service=embedding_service,
            storage_path=storage_path,
            index_type=index_type,
            embedding_dimension=embedding_dimension,
            nlist=nlist,
            nprobe=nprobe
        )
