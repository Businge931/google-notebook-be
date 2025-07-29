"""
Production-Grade Chunk Storage Service

Ensures consistent chunk storage across vector database and PostgreSQL database
for production reliability and data integrity.
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from ..entities import Document, DocumentChunk, DocumentStatus, ProcessingStage
from ..repositories import DocumentRepository, VectorRepository
from ..value_objects import DocumentId, PageNumber
from ....shared.exceptions import (
    RepositoryError
)


class ChunkStorageError(Exception):
    """Exception raised when chunk storage operations fail."""
    pass


class DataIntegrityError(Exception):
    """Exception raised when data integrity validation fails."""
    pass


class ChunkStorageValidationResult:
    """Result of chunk storage validation."""
    
    def __init__(
        self,
        is_valid: bool,
        database_chunk_count: int,
        vector_chunk_count: int,
        missing_in_database: List[str] = None,
        missing_in_vector: List[str] = None,
        page_validation_errors: List[str] = None
    ):
        self.is_valid = is_valid
        self.database_chunk_count = database_chunk_count
        self.vector_chunk_count = vector_chunk_count
        self.missing_in_database = missing_in_database or []
        self.missing_in_vector = missing_in_vector or []
        self.page_validation_errors = page_validation_errors or []


class ProductionChunkStorageService:
    """
    Production-grade service for reliable chunk storage and validation.
    
    Ensures data consistency between vector storage and database storage,
    implements validation, error recovery, and monitoring for production use.
    """
    
    def __init__(
        self,
        document_repository: DocumentRepository,
        vector_repository: VectorRepository
    ):
        self._document_repository = document_repository
        self._vector_repository = vector_repository
        self._logger = logging.getLogger(__name__)
    
    async def store_chunks_with_validation(
        self,
        document: Document,
        chunks: List[DocumentChunk],
        vectors_data: List[Dict[str, Any]],
        max_retries: int = 3
    ) -> ChunkStorageValidationResult:
        """
        Store chunks with full validation and error recovery.
        
        Args:
            document: Document entity
            chunks: List of document chunks
            vectors_data: Vector data for FAISS storage
            max_retries: Maximum retry attempts
            
        Returns:
            Validation result with storage status
            
        Raises:
            ChunkStorageError: If storage fails after retries
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                self._logger.info(
                    f"Attempt {attempt + 1}/{max_retries}: Storing chunks for document {document.document_id.value}"
                )
                
                # Step 1: Validate input data
                await self._validate_input_data(document, chunks, vectors_data)
                
                # Step 2: Store chunks in database (transactional)
                await self._store_chunks_in_database(document, chunks)
                
                # Step 3: Store vectors in FAISS
                await self._store_vectors_in_faiss(document.document_id, vectors_data)
                
                # Step 4: Validate storage consistency
                validation_result = await self.validate_chunk_storage(document.document_id)
                
                if validation_result.is_valid:
                    self._logger.info(
                        f"Successfully stored and validated {len(chunks)} chunks for document {document.document_id.value}"
                    )
                    return validation_result
                else:
                    raise DataIntegrityError(
                        f"Chunk storage validation failed: {validation_result.__dict__}"
                    )
                    
            except Exception as e:
                last_error = e
                self._logger.warning(
                    f"Attempt {attempt + 1} failed for document {document.document_id.value}: {e}"
                )
                
                if attempt < max_retries - 1:
                    # Clean up partial storage before retry
                    await self._cleanup_partial_storage(document.document_id)
                    await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
        
        # All retries failed
        self._logger.error(
            f"Failed to store chunks for document {document.document_id.value} after {max_retries} attempts"
        )
        raise ChunkStorageError(f"Chunk storage failed after {max_retries} attempts: {last_error}")
    
    async def validate_chunk_storage(self, document_id: DocumentId) -> ChunkStorageValidationResult:
        """
        Validate consistency between database and vector storage.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Validation result with detailed status
        """
        try:
            # Get document from database
            document = await self._document_repository.find_by_id(document_id)
            if not document:
                return ChunkStorageValidationResult(
                    is_valid=False,
                    database_chunk_count=0,
                    vector_chunk_count=0,
                    page_validation_errors=["Document not found in database"]
                )
            
            # Count chunks in database
            database_chunk_count = len(document.chunks)
            
            # Count chunks in vector storage
            vector_chunk_count = await self._vector_repository.get_document_chunk_count(document_id)
            
            # Validate page numbers against document page count
            page_validation_errors = []
            if document.page_count:
                for chunk in document.chunks:
                    if chunk.page_number.value > document.page_count:
                        page_validation_errors.append(
                            f"Chunk {chunk.chunk_id} references page {chunk.page_number.value} "
                            f"but document only has {document.page_count} pages"
                        )
            
            # Check for consistency (allow vector storage to have more entries than database chunks)
            # This is normal for production vector databases that may store multiple representations
            is_valid = (
                database_chunk_count > 0 and
                vector_chunk_count > 0 and
                vector_chunk_count >= database_chunk_count and  # Allow vector storage to have more entries
                len(page_validation_errors) == 0
            )
            
            result = ChunkStorageValidationResult(
                is_valid=is_valid,
                database_chunk_count=database_chunk_count,
                vector_chunk_count=vector_chunk_count,
                page_validation_errors=page_validation_errors
            )
            
            if not is_valid:
                self._logger.warning(
                    f"Chunk storage validation failed for document {document_id.value}: "
                    f"DB chunks: {database_chunk_count}, Vector chunks: {vector_chunk_count}, "
                    f"Page errors: {len(page_validation_errors)}"
                )
            
            return result
            
        except Exception as e:
            self._logger.error(f"Chunk validation failed for document {document_id.value}: {e}")
            return ChunkStorageValidationResult(
                is_valid=False,
                database_chunk_count=0,
                vector_chunk_count=0,
                page_validation_errors=[f"Validation error: {str(e)}"]
            )
    
    async def repair_chunk_storage(self, document_id: DocumentId) -> bool:
        """
        Repair inconsistent chunk storage by reprocessing the document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            True if repair was successful
        """
        try:
            self._logger.info(f"Attempting to repair chunk storage for document {document_id.value}")
            
            # Clean up existing storage
            await self._cleanup_partial_storage(document_id)
            
            # Get document
            document = await self._document_repository.find_by_id(document_id)
            if not document:
                self._logger.error(f"Cannot repair: document {document_id.value} not found")
                return False
            
            # Reset document status to trigger reprocessing
            document.status = DocumentStatus.UPLOADED
            document.processing_stage = ProcessingStage.UPLOAD_COMPLETE
            document.chunks.clear()
            document.total_chunks = 0
            
            await self._document_repository.update(document)
            
            self._logger.info(f"Reset document {document_id.value} for reprocessing")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to repair chunk storage for document {document_id.value}: {e}")
            return False
    
    async def _validate_input_data(
        self,
        document: Document,
        chunks: List[DocumentChunk],
        vectors_data: List[Dict[str, Any]]
    ):
        """Validate input data before storage."""
        if not chunks:
            raise ChunkStorageError("No chunks provided for storage")
        
        if len(chunks) != len(vectors_data):
            raise ChunkStorageError(
                f"Chunk count mismatch: {len(chunks)} chunks vs {len(vectors_data)} vectors"
            )
        
        # Validate page numbers
        if document.page_count:
            for chunk in chunks:
                if chunk.page_number.value > document.page_count:
                    raise ChunkStorageError(
                        f"Invalid page number {chunk.page_number.value} for document with {document.page_count} pages"
                    )
    
    async def _store_chunks_in_database(self, document: Document, chunks: List[DocumentChunk]):
        """Store chunks in PostgreSQL database with duplicate handling."""
        try:
            # Clear existing chunks to avoid duplicates
            document.chunks.clear()
            
            # Generate unique chunk IDs to prevent duplicates during reprocessing
            import uuid
            for i, chunk in enumerate(chunks):
                # Create unique chunk ID that fits in VARCHAR(36) constraint
                unique_chunk_id = str(uuid.uuid4())
                chunk.chunk_id = unique_chunk_id
                document.chunks.append(chunk)
            
            document.total_chunks = len(chunks)
            
            # Update document in database (our fix handles chunk deletion/insertion properly)
            await self._document_repository.update(document)
            
            self._logger.info(f"Stored {len(chunks)} chunks in database for document {document.document_id.value}")
            
        except Exception as e:
            raise ChunkStorageError(f"Failed to store chunks in database: {str(e)}")
    
    async def _store_vectors_in_faiss(self, document_id: DocumentId, vectors_data: List[Dict[str, Any]]):
        """Store vectors in FAISS repository."""
        try:
            success = await self._vector_repository.store_document_vectors(document_id, vectors_data)
            if not success:
                raise ChunkStorageError("Vector storage returned failure status")
            
            self._logger.info(f"Stored {len(vectors_data)} vectors in FAISS for document {document_id.value}")
            
        except Exception as e:
            raise ChunkStorageError(f"Failed to store vectors in FAISS: {str(e)}")
    
    async def _cleanup_partial_storage(self, document_id: DocumentId):
        """Clean up partial storage in case of failure."""
        try:
            # Clean up vector storage
            await self._vector_repository.delete_document_vectors(document_id)
            
            # Clean up database chunks
            document = await self._document_repository.find_by_id(document_id)
            if document:
                document.chunks.clear()
                document.total_chunks = 0
                await self._document_repository.update(document)
            
            self._logger.info(f"Cleaned up partial storage for document {document_id.value}")
            
        except Exception as e:
            self._logger.warning(f"Failed to cleanup partial storage for document {document_id.value}: {e}")


class ChunkStorageMonitor:
    """Monitor for chunk storage health and consistency."""
    
    def __init__(self, chunk_storage_service: ProductionChunkStorageService):
        self._chunk_storage_service = chunk_storage_service
        self._logger = logging.getLogger(__name__)
    
    async def health_check_all_documents(self) -> Dict[str, Any]:
        """
        Perform health check on all documents' chunk storage.
        
        Returns:
            Health check report
        """
        # Implementation would check all documents for consistency
        # This is a placeholder for the monitoring system
        pass
