"""
Document Vectorization Use Case

"""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging
from datetime import datetime

from ...domain.entities import Document, DocumentChunk
from ...domain.value_objects import DocumentId
from ...domain.repositories import DocumentRepository, VectorRepository
from ...domain.services.embedding_service import (
    DocumentEmbeddingService,
    EmbeddingBatch,
    TextEmbedding
)
from src.shared.exceptions import (
    DocumentNotFoundError,
    VectorRepositoryError,
    EmbeddingError,
    ValidationError
)


@dataclass
class VectorizationRequest:
    """Request object for document vectorization following Single Responsibility Principle."""
    document_id: DocumentId
    model: Optional[str] = None
    force_regenerate: bool = False
    chunk_size: Optional[int] = None
    overlap_size: Optional[int] = None


@dataclass
class VectorizationResponse:
    """Response object for document vectorization following Single Responsibility Principle."""
    document_id: DocumentId
    success: bool
    chunks_processed: int
    embeddings_generated: int
    processing_time_ms: int
    model_used: str
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BulkVectorizationRequest:
    """Request object for bulk document vectorization following Single Responsibility Principle."""
    document_ids: List[DocumentId]
    model: Optional[str] = None
    force_regenerate: bool = False
    max_concurrent: int = 5
    chunk_size: Optional[int] = None
    overlap_size: Optional[int] = None


@dataclass
class BulkVectorizationResponse:
    """Response object for bulk document vectorization following Single Responsibility Principle."""
    total_documents: int
    successful_documents: int
    failed_documents: int
    total_chunks_processed: int
    total_embeddings_generated: int
    total_processing_time_ms: int
    results: List[VectorizationResponse]
    error_summary: Optional[str] = None


class DocumentVectorizationUseCase:
    """
    Use case for document vectorization operations following Single Responsibility Principle.
    
    Orchestrates the process of generating and storing embeddings for document chunks.
    """
    
    def __init__(
        self,
        document_repository: DocumentRepository,
        vector_repository: VectorRepository,
        document_embedding_service: DocumentEmbeddingService
    ):
        """
        Initialize document vectorization use case.
        
        Args:
            document_repository: Repository for document operations
            vector_repository: Repository for vector operations
            document_embedding_service: Service for document embedding operations
        """
        self._document_repository = document_repository
        self._vector_repository = vector_repository
        self._document_embedding_service = document_embedding_service
        self._logger = logging.getLogger(__name__)
    
    async def vectorize_document(
        self,
        request: VectorizationRequest
    ) -> VectorizationResponse:
        """
        Vectorize a single document.
        
        Args:
            request: Vectorization request
            
        Returns:
            Vectorization response
            
        Raises:
            DocumentNotFoundError: If document doesn't exist
            VectorRepositoryError: If vector storage fails
            EmbeddingError: If embedding generation fails
        """
        start_time = datetime.utcnow()
        
        try:
            # Validate request
            if not request.document_id:
                raise ValidationError("Document ID is required")
            
            # Retrieve document
            document = await self._document_repository.get_by_id(request.document_id)
            if not document:
                raise DocumentNotFoundError(f"Document {request.document_id.value} not found")
            
            # Check if document is processed
            if not document.chunks:
                raise ValidationError(f"Document {request.document_id.value} has no chunks to vectorize")
            
            # Check if vectors already exist and force_regenerate is False
            if not request.force_regenerate:
                existing_count = await self._vector_repository.get_document_chunk_count(request.document_id)
                if existing_count > 0:
                    self._logger.info(
                        f"Document {request.document_id.value} already has {existing_count} vectors, skipping"
                    )
                    return VectorizationResponse(
                        document_id=request.document_id,
                        success=True,
                        chunks_processed=existing_count,
                        embeddings_generated=0,
                        processing_time_ms=0,
                        model_used="existing",
                        metadata={"skipped": True, "reason": "vectors_already_exist"}
                    )
            
            # Prepare chunks for vectorization
            chunk_data = []
            for chunk in document.chunks:
                chunk_dict = {
                    "chunk_id": chunk.id.value,
                    "text_content": chunk.content,
                    "page_number": chunk.page_number.value,
                    "start_position": chunk.start_position,
                    "end_position": chunk.end_position,
                    "metadata": {
                        "document_title": document.file_metadata.filename,
                        "document_filename": document.file_metadata.filename,
                        "chunk_order": chunk.order
                    }
                }
                chunk_data.append(chunk_dict)
            
            # Generate embeddings
            texts = [chunk["text_content"] for chunk in chunk_data]
            chunk_ids = [chunk["chunk_id"] for chunk in chunk_data]
            
            embedding_batch = await self._document_embedding_service.embed_document_chunks(
                document_id=request.document_id,
                chunks=texts,
                chunk_ids=chunk_ids,
                model=request.model
            )
            
            # Attach embeddings to chunk data before storing
            if len(embedding_batch.embeddings) != len(chunk_data):
                raise VectorRepositoryError(
                    f"Embedding count mismatch: {len(embedding_batch.embeddings)} embeddings "
                    f"for {len(chunk_data)} chunks"
                )
            
            for i, chunk in enumerate(chunk_data):
                chunk["embedding"] = embedding_batch.embeddings[i]
                chunk["vector"] = embedding_batch.embeddings[i]  # For backward compatibility
            
            # Store vectors in vector repository with embeddings attached
            await self._vector_repository.store_document_vectors(
                document_id=request.document_id,
                chunks=chunk_data
            )
            
            # Calculate processing time
            end_time = datetime.utcnow()
            processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            self._logger.info(
                f"Successfully vectorized document {request.document_id.value}: "
                f"{len(chunk_data)} chunks, {len(embedding_batch.embeddings)} embeddings"
            )
            
            return VectorizationResponse(
                document_id=request.document_id,
                success=True,
                chunks_processed=len(chunk_data),
                embeddings_generated=len(embedding_batch.embeddings),
                processing_time_ms=processing_time_ms,
                model_used=embedding_batch.model,
                metadata={
                    "total_tokens": embedding_batch.total_tokens,
                    "embedding_processing_time_ms": embedding_batch.processing_time_ms,
                    "document_title": document.file_metadata.filename,
                    "document_filename": document.file_metadata.filename
                }
            )
            
        except (DocumentNotFoundError, ValidationError) as e:
            # Re-raise domain exceptions
            raise e
        except Exception as e:
            self._logger.error(f"Vectorization failed for document {request.document_id.value}: {e}")
            
            # Calculate processing time for error case
            end_time = datetime.utcnow()
            processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            return VectorizationResponse(
                document_id=request.document_id,
                success=False,
                chunks_processed=0,
                embeddings_generated=0,
                processing_time_ms=processing_time_ms,
                model_used=request.model or "unknown",
                error_message=str(e)
            )
    
    async def update_document_vectors(
        self,
        request: VectorizationRequest
    ) -> VectorizationResponse:
        """
        Update vectors for a document (delete old, create new).
        
        Args:
            request: Vectorization request
            
        Returns:
            Vectorization response
        """
        # Force regeneration for updates
        request.force_regenerate = True
        return await self.vectorize_document(request)
    
    async def delete_document_vectors(
        self,
        document_id: DocumentId
    ) -> bool:
        """
        Delete all vectors for a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            True if deletion was successful
            
        Raises:
            VectorRepositoryError: If deletion fails
        """
        try:
            success = await self._vector_repository.delete_document_vectors(document_id)
            
            if success:
                self._logger.info(f"Deleted vectors for document {document_id.value}")
            else:
                self._logger.warning(f"Failed to delete vectors for document {document_id.value}")
            
            return success
            
        except Exception as e:
            self._logger.error(f"Error deleting vectors for document {document_id.value}: {e}")
            raise VectorRepositoryError(f"Failed to delete vectors: {str(e)}")
    
    async def get_vectorization_status(
        self,
        document_id: DocumentId
    ) -> Dict[str, Any]:
        """
        Get vectorization status for a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Vectorization status information
        """
        try:
            # Get document
            document = await self._document_repository.get_by_id(document_id)
            if not document:
                raise DocumentNotFoundError(f"Document {document_id.value} not found")
            
            # Get vector count
            vector_count = await self._vector_repository.get_document_chunk_count(document_id)
            chunk_count = len(document.chunks) if document.chunks else 0
            
            is_vectorized = vector_count > 0
            is_complete = vector_count == chunk_count and chunk_count > 0
            
            return {
                "document_id": document_id.value,
                "is_vectorized": is_vectorized,
                "is_complete": is_complete,
                "vector_count": vector_count,
                "chunk_count": chunk_count,
                "completion_percentage": (vector_count / chunk_count * 100) if chunk_count > 0 else 0,
                "document_status": document.status.value,
                "document_title": document.file_metadata.filename,
                "document_filename": document.file_metadata.filename
            }
            
        except DocumentNotFoundError:
            raise
        except Exception as e:
            self._logger.error(f"Error getting vectorization status for {document_id.value}: {e}")
            raise VectorRepositoryError(f"Failed to get vectorization status: {str(e)}")


class BulkDocumentVectorizationUseCase:
    """
    Use case for bulk document vectorization operations following Single Responsibility Principle.
    
    Handles vectorization of multiple documents with concurrency control.
    """
    
    def __init__(
        self,
        document_vectorization_use_case: DocumentVectorizationUseCase,
        max_concurrent: int = 5
    ):
        """
        Initialize bulk document vectorization use case.
        
        Args:
            document_vectorization_use_case: Single document vectorization use case
            max_concurrent: Maximum concurrent vectorization operations
        """
        self._document_vectorization_use_case = document_vectorization_use_case
        self._max_concurrent = max_concurrent
        self._logger = logging.getLogger(__name__)
    
    async def vectorize_documents(
        self,
        request: BulkVectorizationRequest
    ) -> BulkVectorizationResponse:
        """
        Vectorize multiple documents.
        
        Args:
            request: Bulk vectorization request
            
        Returns:
            Bulk vectorization response
        """
        start_time = datetime.utcnow()
        
        try:
            if not request.document_ids:
                return BulkVectorizationResponse(
                    total_documents=0,
                    successful_documents=0,
                    failed_documents=0,
                    total_chunks_processed=0,
                    total_embeddings_generated=0,
                    total_processing_time_ms=0,
                    results=[]
                )
            
            # Process documents with concurrency control
            import asyncio
            semaphore = asyncio.Semaphore(request.max_concurrent or self._max_concurrent)
            
            async def process_document(document_id: DocumentId) -> VectorizationResponse:
                async with semaphore:
                    vectorization_request = VectorizationRequest(
                        document_id=document_id,
                        model=request.model,
                        force_regenerate=request.force_regenerate,
                        chunk_size=request.chunk_size,
                        overlap_size=request.overlap_size
                    )
                    return await self._document_vectorization_use_case.vectorize_document(
                        vectorization_request
                    )
            
            # Execute all vectorization tasks
            tasks = [process_document(doc_id) for doc_id in request.document_ids]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            vectorization_results = []
            successful_count = 0
            failed_count = 0
            total_chunks = 0
            total_embeddings = 0
            errors = []
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Handle exceptions
                    error_result = VectorizationResponse(
                        document_id=request.document_ids[i],
                        success=False,
                        chunks_processed=0,
                        embeddings_generated=0,
                        processing_time_ms=0,
                        model_used=request.model or "unknown",
                        error_message=str(result)
                    )
                    vectorization_results.append(error_result)
                    failed_count += 1
                    errors.append(f"Document {request.document_ids[i].value}: {str(result)}")
                else:
                    # Handle successful results
                    vectorization_results.append(result)
                    if result.success:
                        successful_count += 1
                        total_chunks += result.chunks_processed
                        total_embeddings += result.embeddings_generated
                    else:
                        failed_count += 1
                        if result.error_message:
                            errors.append(f"Document {result.document_id.value}: {result.error_message}")
            
            # Calculate total processing time
            end_time = datetime.utcnow()
            total_processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Create error summary
            error_summary = None
            if errors:
                error_summary = f"Errors occurred in {len(errors)} documents: " + "; ".join(errors[:5])
                if len(errors) > 5:
                    error_summary += f" and {len(errors) - 5} more..."
            
            self._logger.info(
                f"Bulk vectorization completed: {successful_count}/{len(request.document_ids)} successful, "
                f"{total_chunks} chunks, {total_embeddings} embeddings"
            )
            
            return BulkVectorizationResponse(
                total_documents=len(request.document_ids),
                successful_documents=successful_count,
                failed_documents=failed_count,
                total_chunks_processed=total_chunks,
                total_embeddings_generated=total_embeddings,
                total_processing_time_ms=total_processing_time_ms,
                results=vectorization_results,
                error_summary=error_summary
            )
            
        except Exception as e:
            self._logger.error(f"Bulk vectorization failed: {e}")
            
            # Calculate processing time for error case
            end_time = datetime.utcnow()
            processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            return BulkVectorizationResponse(
                total_documents=len(request.document_ids),
                successful_documents=0,
                failed_documents=len(request.document_ids),
                total_chunks_processed=0,
                total_embeddings_generated=0,
                total_processing_time_ms=processing_time_ms,
                results=[],
                error_summary=f"Bulk vectorization failed: {str(e)}"
            )
