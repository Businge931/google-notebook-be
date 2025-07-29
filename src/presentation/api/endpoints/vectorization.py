"""
Vectorization API Endpoints

"""
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse

from ....core.application.use_cases.document_vectorization_use_case import (
    DocumentVectorizationUseCase,
    BulkDocumentVectorizationUseCase,
    VectorizationRequest,
    BulkVectorizationRequest
)
from ....core.application.use_cases.similarity_search_use_case import (
    SimilaritySearchUseCase,
    SimilaritySearchRequest,
    DocumentSimilarityRequest
)
from src.core.domain.value_objects import DocumentId
from ....infrastructure.di import (
    get_document_vectorization_use_case,
    get_bulk_document_vectorization_use_case,
    get_similarity_search_use_case
)
from src.shared.exceptions import (
    ValidationError,
    DocumentNotFoundError,
    VectorRepositoryError,
    EmbeddingError
)
from ..schemas.vectorization_schemas import (
    VectorizationRequestSchema,
    VectorizationResponseSchema,
    BulkVectorizationRequestSchema,
    BulkVectorizationResponseSchema,
    SimilaritySearchRequestSchema,
    SimilaritySearchResponseSchema,
    DocumentSimilarityRequestSchema,
    DocumentSimilarityResponseSchema,
    VectorizationStatusResponseSchema,
    SearchSuggestionsResponseSchema
)

# Create router
router = APIRouter(prefix="/vectorization", tags=["vectorization"])
logger = logging.getLogger(__name__)


@router.post(
    "/documents/{document_id}",
    response_model=VectorizationResponseSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Vectorize a document",
    description="Generate and store embeddings for a document's text chunks"
)
async def vectorize_document(
    document_id: str,
    request: VectorizationRequestSchema,
    background_tasks: BackgroundTasks,
    vectorization_use_case=Depends(get_document_vectorization_use_case)
):
    """
    Vectorize a single document following Single Responsibility Principle.
    
    Args:
        document_id: Document identifier
        request: Vectorization request parameters
        background_tasks: FastAPI background tasks
        vectorization_use_case: Document vectorization use case
        
    Returns:
        Vectorization response
        
    Raises:
        HTTPException: If vectorization fails
    """
    try:
        # Create use case request
        vectorization_request = VectorizationRequest(
            document_id=DocumentId(document_id),
            model=request.model,
            force_regenerate=request.force_regenerate,
            chunk_size=request.chunk_size,
            overlap_size=request.overlap_size
        )
        
        # Execute vectorization
        if request.async_processing:
            # Process in background
            background_tasks.add_task(
                vectorization_use_case.vectorize_document,
                vectorization_request
            )
            
            return VectorizationResponseSchema(
                document_id=document_id,
                success=True,
                chunks_processed=0,
                embeddings_generated=0,
                processing_time_ms=0,
                model_used=request.model or "default",
                metadata={"async_processing": True, "status": "queued"}
            )
        else:
            # Process synchronously
            response = await vectorization_use_case.vectorize_document(vectorization_request)
            
            return VectorizationResponseSchema(
                document_id=response.document_id.value,
                success=response.success,
                chunks_processed=response.chunks_processed,
                embeddings_generated=response.embeddings_generated,
                processing_time_ms=response.processing_time_ms,
                model_used=response.model_used,
                error_message=response.error_message,
                metadata=response.metadata
            )
            
    except DocumentNotFoundError as e:
        logger.warning(f"Document not found for vectorization: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {str(e)}"
        )
    except ValidationError as e:
        logger.warning(f"Validation error in vectorization: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(e)}"
        )
    except (VectorRepositoryError, EmbeddingError) as e:
        logger.error(f"Vectorization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vectorization failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in vectorization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.put(
    "/documents/{document_id}",
    response_model=VectorizationResponseSchema,
    summary="Update document vectors",
    description="Update embeddings for a document (delete old, create new)"
)
async def update_document_vectors(
    document_id: str,
    request: VectorizationRequestSchema,
    vectorization_use_case=Depends(get_document_vectorization_use_case)
):
    """
    Update vectors for a document following Single Responsibility Principle.
    
    Args:
        document_id: Document identifier
        request: Vectorization request parameters
        vectorization_use_case: Document vectorization use case
        
    Returns:
        Vectorization response
    """
    try:
        # Create use case request (force regenerate for updates)
        vectorization_request = VectorizationRequest(
            document_id=DocumentId(document_id),
            model=request.model,
            force_regenerate=True,  # Always force regenerate for updates
            chunk_size=request.chunk_size,
            overlap_size=request.overlap_size
        )
        
        # Execute vectorization update
        response = await vectorization_use_case.update_document_vectors(vectorization_request)
        
        return VectorizationResponseSchema(
            document_id=response.document_id.value,
            success=response.success,
            chunks_processed=response.chunks_processed,
            embeddings_generated=response.embeddings_generated,
            processing_time_ms=response.processing_time_ms,
            model_used=response.model_used,
            error_message=response.error_message,
            metadata=response.metadata
        )
        
    except DocumentNotFoundError as e:
        logger.warning(f"Document not found for vector update: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {str(e)}"
        )
    except ValidationError as e:
        logger.warning(f"Validation error in vector update: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Vector update failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector update failed: {str(e)}"
        )


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete document vectors",
    description="Delete all embeddings for a document"
)
async def delete_document_vectors(
    document_id: str,
    vectorization_use_case=Depends(get_document_vectorization_use_case)
):
    """
    Delete vectors for a document following Single Responsibility Principle.
    
    Args:
        document_id: Document identifier
        vectorization_use_case: Document vectorization use case
    """
    try:
        success = await vectorization_use_case.delete_document_vectors(
            DocumentId(document_id)
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete document vectors"
            )
            
    except VectorRepositoryError as e:
        logger.error(f"Vector deletion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector deletion failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in vector deletion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get(
    "/documents/{document_id}/status",
    response_model=VectorizationStatusResponseSchema,
    summary="Get vectorization status",
    description="Get vectorization status for a document"
)
async def get_vectorization_status(
    document_id: str,
    vectorization_use_case=Depends(get_document_vectorization_use_case)
):
    """
    Get vectorization status for a document following Single Responsibility Principle.
    
    Args:
        document_id: Document identifier
        vectorization_use_case: Document vectorization use case
        
    Returns:
        Vectorization status
    """
    try:
        status_info = await vectorization_use_case.get_vectorization_status(
            DocumentId(document_id)
        )
        
        return VectorizationStatusResponseSchema(**status_info)
        
    except DocumentNotFoundError as e:
        logger.warning(f"Document not found for status check: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Status check failed: {str(e)}"
        )


@router.post(
    "/bulk",
    response_model=BulkVectorizationResponseSchema,
    summary="Bulk vectorize documents",
    description="Vectorize multiple documents with concurrency control"
)
async def bulk_vectorize_documents(
    request: BulkVectorizationRequestSchema,
    bulk_vectorization_use_case=Depends(get_bulk_document_vectorization_use_case)
):
    """
    Bulk vectorize documents following Single Responsibility Principle.
    
    Args:
        request: Bulk vectorization request
        bulk_vectorization_use_case: Bulk document vectorization use case
        
    Returns:
        Bulk vectorization response
    """
    try:
        # Create use case request
        bulk_request = BulkVectorizationRequest(
            document_ids=[DocumentId(doc_id) for doc_id in request.document_ids],
            model=request.model,
            force_regenerate=request.force_regenerate,
            max_concurrent=request.max_concurrent,
            chunk_size=request.chunk_size,
            overlap_size=request.overlap_size
        )
        
        # Execute bulk vectorization
        response = await bulk_vectorization_use_case.vectorize_documents(bulk_request)
        
        return BulkVectorizationResponseSchema(
            total_documents=response.total_documents,
            successful_documents=response.successful_documents,
            failed_documents=response.failed_documents,
            total_chunks_processed=response.total_chunks_processed,
            total_embeddings_generated=response.total_embeddings_generated,
            total_processing_time_ms=response.total_processing_time_ms,
            results=[
                VectorizationResponseSchema(
                    document_id=result.document_id.value,
                    success=result.success,
                    chunks_processed=result.chunks_processed,
                    embeddings_generated=result.embeddings_generated,
                    processing_time_ms=result.processing_time_ms,
                    model_used=result.model_used,
                    error_message=result.error_message,
                    metadata=result.metadata
                )
                for result in response.results
            ],
            error_summary=response.error_summary
        )
        
    except ValidationError as e:
        logger.warning(f"Validation error in bulk vectorization: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Bulk vectorization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Bulk vectorization failed: {str(e)}"
        )


@router.post(
    "/search",
    response_model=SimilaritySearchResponseSchema,
    summary="Search similar chunks",
    description="Search for similar text chunks using vector similarity"
)
async def search_similar_chunks(
    request: SimilaritySearchRequestSchema,
    similarity_search_use_case=Depends(get_similarity_search_use_case)
):
    """
    Search for similar text chunks following Single Responsibility Principle.
    
    Args:
        request: Similarity search request
        similarity_search_use_case: Similarity search use case
        
    Returns:
        Similarity search response
    """
    try:
        # Create use case request
        search_request = SimilaritySearchRequest(
            query_text=request.query_text,
            document_id=DocumentId(request.document_id) if request.document_id else None,
            document_ids=[DocumentId(doc_id) for doc_id in request.document_ids] if request.document_ids else None,
            limit=request.limit,
            similarity_threshold=request.similarity_threshold,
            include_metadata=request.include_metadata
        )
        
        # Execute search
        response = await similarity_search_use_case.search_similar_chunks(search_request)
        
        return SimilaritySearchResponseSchema(
            query_text=response.query_text,
            results=[
                {
                    "chunk_id": result.chunk_id,
                    "document_id": result.document_id,
                    "document_title": result.document_title,
                    "document_filename": result.document_filename,
                    "page_number": result.page_number,
                    "text_content": result.text_content,
                    "similarity_score": result.similarity_score,
                    "start_position": result.start_position,
                    "end_position": result.end_position,
                    "metadata": result.metadata
                }
                for result in response.results
            ],
            total_results=response.total_results,
            search_time_ms=response.search_time_ms,
            search_scope=response.search_scope,
            similarity_threshold=response.similarity_threshold,
            metadata=response.metadata
        )
        
    except ValidationError as e:
        logger.warning(f"Validation error in similarity search: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Similarity search failed: {str(e)}"
        )


@router.post(
    "/search/documents",
    response_model=DocumentSimilarityResponseSchema,
    summary="Search similar documents",
    description="Search for documents similar to query text"
)
async def search_similar_documents(
    request: DocumentSimilarityRequestSchema,
    similarity_search_use_case=Depends(get_similarity_search_use_case)
):
    """
    Search for similar documents following Single Responsibility Principle.
    
    Args:
        request: Document similarity search request
        similarity_search_use_case: Similarity search use case
        
    Returns:
        Document similarity search response
    """
    try:
        # Create use case request
        search_request = DocumentSimilarityRequest(
            query_text=request.query_text,
            limit=request.limit,
            exclude_document_ids=[DocumentId(doc_id) for doc_id in request.exclude_document_ids] if request.exclude_document_ids else None,
            similarity_threshold=request.similarity_threshold
        )
        
        # Execute search
        response = await similarity_search_use_case.search_similar_documents(search_request)
        
        return DocumentSimilarityResponseSchema(
            query_text=response.query_text,
            similar_documents=response.similar_documents,
            total_documents=response.total_documents,
            search_time_ms=response.search_time_ms,
            similarity_threshold=response.similarity_threshold
        )
        
    except ValidationError as e:
        logger.warning(f"Validation error in document similarity search: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Document similarity search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document similarity search failed: {str(e)}"
        )


@router.get(
    "/search/suggestions",
    response_model=SearchSuggestionsResponseSchema,
    summary="Get search suggestions",
    description="Get search suggestions based on partial query"
)
async def get_search_suggestions(
    query: str,
    document_id: Optional[str] = None,
    limit: int = 5,
    similarity_search_use_case=Depends(get_similarity_search_use_case)
):
    """
    Get search suggestions following Single Responsibility Principle.
    
    Args:
        query: Partial query text
        document_id: Optional document to search within
        limit: Maximum number of suggestions
        similarity_search_use_case: Similarity search use case
        
    Returns:
        Search suggestions response
    """
    try:
        if len(query.strip()) < 3:
            return SearchSuggestionsResponseSchema(
                query=query,
                suggestions=[],
                total_suggestions=0
            )
        
        suggestions = await similarity_search_use_case.get_search_suggestions(
            partial_query=query,
            document_id=DocumentId(document_id) if document_id else None,
            limit=limit
        )
        
        return SearchSuggestionsResponseSchema(
            query=query,
            suggestions=suggestions,
            total_suggestions=len(suggestions)
        )
        
    except Exception as e:
        logger.error(f"Search suggestions failed: {e}")
        # Return empty suggestions on error rather than failing
        return SearchSuggestionsResponseSchema(
            query=query,
            suggestions=[],
            total_suggestions=0
        )
