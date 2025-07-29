"""
Document API Endpoints

"""
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, status
from fastapi.responses import JSONResponse, FileResponse
import logging

from ....core.application.use_cases import (
    DocumentUploadUseCase,
    DocumentUploadRequest,
    DocumentProcessingUseCase,
    DocumentProcessingRequest,
    DocumentManagementUseCase,
    DocumentRetrievalRequest,
    DocumentUpdateRequest,
    DocumentDeletionRequest,
    DocumentSearchUseCase,
    BulkDocumentProcessingUseCase
)
from src.core.domain.entities import DocumentStatus, ProcessingStage
from src.core.domain.value_objects import DocumentId
from ....infrastructure.di import (
    get_document_repository,
    get_file_storage_service,
    get_document_processing_service
)
from ....infrastructure.adapters.services.pdf_parsing_service import PDFParsingServiceFactory
from src.shared.exceptions import (
    ValidationError,
    DocumentNotFoundError,
    DocumentProcessingError,
    FileStorageError
)
from ..schemas.document_schemas import (
    DocumentUploadResponse,
    DocumentResponse,
    DocumentListResponse,
    DocumentProcessingResponse,
    DocumentUpdateRequest as DocumentUpdateSchema,
    DocumentSearchRequest,
    BulkProcessingRequest
)

# Create router
router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger(__name__)


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a PDF document",
    description="Upload a PDF document for processing and analysis"
)
async def upload_document(
    file: UploadFile = File(..., description="PDF file to upload"),
    document_repository=Depends(get_document_repository),
    file_storage=Depends(get_file_storage_service),
    document_processing_service=Depends(get_document_processing_service)
):
    """
    Upload a PDF document following Single Responsibility Principle.
    
    Args:
        file: Uploaded PDF file
        document_repository: Document repository dependency
        file_storage: File storage service dependency
        document_processing_service: Document processing service dependency
        
    Returns:
        Document upload response
        
    Raises:
        HTTPException: If upload fails
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('application/pdf'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are supported"
            )
        
        
        # Create upload request
        upload_request = DocumentUploadRequest(
            file_data=file.file,
            filename=file.filename,
            file_size=file.size,
            mime_type=file.content_type
        )
        
        # Create and execute upload use case with injected dependencies
        upload_use_case = DocumentUploadUseCase(
            document_repository=document_repository,
            file_storage=file_storage,
            document_processing_service=document_processing_service
        )
        
        response = await upload_use_case.execute(upload_request)
        
        logger.info(f"Document uploaded successfully: {response.document_id.value}")
        
        return DocumentUploadResponse(
            document_id=response.document_id.value,
            filename=response.filename,
            file_size=response.file_size,
            status=response.status.value,
            processing_stage=response.processing_stage.value,
            file_url=response.file_url,
            upload_timestamp=response.upload_timestamp
        )
        
    except ValidationError as e:
        logger.warning(f"Upload validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except FileStorageError as e:
        logger.error(f"File storage error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File storage failed"
        )
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Upload failed"
        )


@router.post(
    "/{document_id}/process",
    response_model=DocumentProcessingResponse,
    summary="Process a document",
    description="Start processing a document to extract text and create chunks"
)
async def process_document(
    document_id: str,
    force_reprocess: bool = Query(False, description="Force reprocessing even if already processed"),
    document_repository=Depends(get_document_repository),
    file_storage=Depends(get_file_storage_service),
    document_processing_service=Depends(get_document_processing_service)
):
    """
    Process a document following Single Responsibility Principle.
    
    Args:
        document_id: Document ID to process
        force_reprocess: Whether to force reprocessing
        document_repository: Document repository dependency
        file_storage: File storage service dependency
        document_processing_service: Document processing service dependency
        
    Returns:
        Document processing response
        
    Raises:
        HTTPException: If processing fails
    """
    try:
        # Create processing request
        processing_request = DocumentProcessingRequest(
            document_id=DocumentId(document_id),
            force_reprocess=force_reprocess
        )
        
        # Create PDF parsing service
        pdf_parsing_service = PDFParsingServiceFactory.create_default()
        
        # Create AI services for vectorization and indexing
        from ....infrastructure.di.ai_container import AIContainer, get_ai_container
        from ....infrastructure.config.settings import get_settings
        
        settings = get_settings()
        ai_container = get_ai_container(settings)
        embedding_service = ai_container.get_embedding_service()
        vector_repository = ai_container.get_vector_repository()
        
        # Create and execute processing use case with AI services
        processing_use_case = DocumentProcessingUseCase(
            document_repository=document_repository,
            file_storage=file_storage,
            pdf_parsing_service=pdf_parsing_service,
            document_processing_service=document_processing_service,
            embedding_service=embedding_service,
            vector_repository=vector_repository
        )
        
        response = await processing_use_case.execute(processing_request)
        
        logger.info(f"Document processed successfully: {document_id}")
        
        return DocumentProcessingResponse(
            document_id=response.document_id.value,
            status=response.status.value,
            processing_stage=response.processing_stage.value,
            page_count=response.page_count,
            chunk_count=response.chunk_count,
            processing_time_ms=response.processing_time_ms,
            error_message=response.error_message
        )
        
    except DocumentNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}"
        )
    except DocumentProcessingError as e:
        logger.error(f"Processing failed for document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Processing failed"
        )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get document details",
    description="Retrieve details of a specific document"
)
async def get_document(
    document_id: str,
    document_repository=Depends(get_document_repository),
    file_storage=Depends(get_file_storage_service),
    document_processing_service=Depends(get_document_processing_service)
):
    """
    Get document details following Single Responsibility Principle.
    
    Args:
        document_id: Document ID to retrieve
        document_repository: Document repository dependency
        file_storage: File storage service dependency
        document_processing_service: Document processing service dependency
        
    Returns:
        Document details
        
    Raises:
        HTTPException: If document not found
    """
    try:
        # Create management use case
        management_use_case = DocumentManagementUseCase(
            document_repository=document_repository,
            file_storage=file_storage,
            document_processing_service=document_processing_service
        )
        
        response = await management_use_case.get_document(DocumentId(document_id))
        
        if not response.success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=response.error_message
            )
        
        document = response.document
        
        return DocumentResponse(
            document_id=document.document_id.value,
            filename=document.file_metadata.filename,
            file_size=document.file_metadata.file_size,
            mime_type=document.file_metadata.mime_type,
            status=document.status.value,
            processing_stage=document.processing_stage.value,
            page_count=document.page_count,
            chunk_count=document.total_chunks,
            upload_timestamp=document.file_metadata.upload_timestamp,
            created_at=document.created_at,
            updated_at=document.updated_at,
            processing_error=document.processing_error
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document {document_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document"
        )


@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List documents",
    description="Retrieve a list of documents with optional filtering"
)
async def list_documents(
    status_filter: Optional[str] = Query(None, description="Filter by document status"),
    limit: Optional[int] = Query(50, ge=1, le=100, description="Maximum number of documents"),
    offset: Optional[int] = Query(0, ge=0, description="Number of documents to skip"),
    document_repository=Depends(get_document_repository),
    file_storage=Depends(get_file_storage_service),
    document_processing_service=Depends(get_document_processing_service)
):
    """
    List documents following Single Responsibility Principle.
    
    Args:
        status_filter: Optional status filter
        limit: Maximum number of documents
        offset: Number of documents to skip
        document_repository: Document repository dependency
        file_storage: File storage service dependency
        document_processing_service: Document processing service dependency
        
    Returns:
        List of documents
        
    Raises:
        HTTPException: If listing fails
    """
    try:
        # Parse status filter
        parsed_status = None
        if status_filter:
            try:
                parsed_status = DocumentStatus(status_filter.upper())
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid status: {status_filter}"
                )
        
        # Create retrieval request
        retrieval_request = DocumentRetrievalRequest(
            status_filter=parsed_status,
            limit=limit,
            offset=offset
        )
        
        # Create management use case
        management_use_case = DocumentManagementUseCase(
            document_repository=document_repository,
            file_storage=file_storage,
            document_processing_service=document_processing_service
        )
        
        response = await management_use_case.list_documents(retrieval_request)
        
        if not response.success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=response.error_message
            )
        
        # Convert documents to response format
        documents = [
            DocumentResponse(
                document_id=doc.document_id.value,
                filename=doc.file_metadata.filename,
                file_size=doc.file_metadata.file_size,
                mime_type=doc.file_metadata.mime_type,
                status=doc.status.value,
                processing_stage=doc.processing_stage.value,
                page_count=doc.page_count,
                chunk_count=doc.total_chunks,
                upload_timestamp=doc.file_metadata.upload_timestamp,
                created_at=doc.created_at,
                updated_at=doc.updated_at,
                processing_error=doc.processing_error
            )
            for doc in response.documents
        ]
        
        return DocumentListResponse(
            documents=documents,
            total=len(documents),
            limit=limit,
            offset=offset
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list documents"
        )


@router.put(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Update document",
    description="Update document status and metadata"
)
async def update_document(
    document_id: str,
    update_data: DocumentUpdateSchema,
    document_repository=Depends(get_document_repository),
    file_storage=Depends(get_file_storage_service),
    document_processing_service=Depends(get_document_processing_service)
):
    """
    Update document following Single Responsibility Principle.
    
    Args:
        document_id: Document ID to update
        update_data: Update data
        document_repository: Document repository dependency
        file_storage: File storage service dependency
        document_processing_service: Document processing service dependency
        
    Returns:
        Updated document details
        
    Raises:
        HTTPException: If update fails
    """
    try:
        # Parse status and processing stage
        parsed_status = None
        parsed_stage = None
        
        if update_data.status:
            try:
                parsed_status = DocumentStatus(update_data.status.upper())
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid status: {update_data.status}"
                )
        
        if update_data.processing_stage:
            try:
                parsed_stage = ProcessingStage(update_data.processing_stage.upper())
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid processing stage: {update_data.processing_stage}"
                )
        
        # Create update request
        update_request = DocumentUpdateRequest(
            document_id=DocumentId(document_id),
            status=parsed_status,
            processing_stage=parsed_stage,
            processing_error=update_data.processing_error
        )
        
        # Create management use case
        management_use_case = DocumentManagementUseCase(
            document_repository=document_repository,
            file_storage=file_storage,
            document_processing_service=document_processing_service
        )
        
        response = await management_use_case.update_document(update_request)
        
        if not response.success:
            if "not found" in response.error_message.lower():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=response.error_message
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=response.error_message
                )
        
        document = response.document
        
        return DocumentResponse(
            document_id=document.document_id.value,
            filename=document.file_metadata.filename,
            file_size=document.file_metadata.file_size,
            mime_type=document.file_metadata.mime_type,
            status=document.status.value,
            processing_stage=document.processing_stage.value,
            page_count=document.page_count,
            chunk_count=document.total_chunks,
            upload_timestamp=document.file_metadata.upload_timestamp,
            created_at=document.created_at,
            updated_at=document.updated_at,
            processing_error=document.processing_error
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update document {document_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update document"
        )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete document",
    description="Delete a document and optionally its associated file"
)
async def delete_document(
    document_id: str,
    delete_file: bool = Query(True, description="Whether to delete the associated file"),
    document_repository=Depends(get_document_repository),
    file_storage=Depends(get_file_storage_service),
    document_processing_service=Depends(get_document_processing_service)
):
    """
    Delete document following Single Responsibility Principle.
    
    Args:
        document_id: Document ID to delete
        delete_file: Whether to delete associated file
        document_repository: Document repository dependency
        file_storage: File storage service dependency
        document_processing_service: Document processing service dependency
        
    Raises:
        HTTPException: If deletion fails
    """
    try:
        # Create deletion request
        deletion_request = DocumentDeletionRequest(
            document_id=DocumentId(document_id),
            delete_file=delete_file
        )
        
        # Create management use case
        management_use_case = DocumentManagementUseCase(
            document_repository=document_repository,
            file_storage=file_storage,
            document_processing_service=document_processing_service
        )
        
        response = await management_use_case.delete_document(deletion_request)
        
        if not response.success:
            if "not found" in response.error_message.lower():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=response.error_message
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=response.error_message
                )
        
        logger.info(f"Document deleted successfully: {document_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document {document_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document"
        )


@router.get(
    "/{document_id}/download",
    summary="Download document file",
    description="Download the original PDF file for viewing"
)
async def download_document(
    document_id: str,
    document_repository=Depends(get_document_repository),
    file_storage=Depends(get_file_storage_service)
):
    """
    Download document file for viewing in PDF viewer.
    
    Args:
        document_id: Document ID to download
        document_repository: Document repository dependency
        file_storage: File storage service dependency
        
    Returns:
        FileResponse with PDF content
        
    Raises:
        HTTPException: If download fails
    """
    try:
        # Validate and create document ID
        try:
            doc_id = DocumentId(document_id)
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid document ID format: {document_id}, error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid document ID format: {document_id}"
            )
        
        # Create management use case
        management_use_case = DocumentManagementUseCase(
            document_repository=document_repository,
            file_storage=file_storage,
            document_processing_service=None  # Not needed for retrieval
        )
        
        # Get document directly (not using retrieval request)
        response = await management_use_case.get_document(doc_id)
        
        if not response.success:
            if "not found" in response.error_message.lower():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=response.error_message
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=response.error_message
                )
        
        # Check if file exists in storage
        if not await file_storage.file_exists(response.document.file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document file not found"
            )
        
        # For LocalFileStorage, construct the full file path
        if hasattr(file_storage, '_base_path'):
            # LocalFileStorage implementation
            safe_path = file_storage._sanitize_path(response.document.file_path)
            file_path = file_storage._base_path / safe_path
        else:
            # Fallback for other storage implementations
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Storage type not supported for direct file access"
            )
        
        logger.info(f"Document file downloaded: {document_id}")
        
        # Return file response with proper headers
        return FileResponse(
            path=str(file_path),
            media_type="application/pdf",
            filename=response.document.file_metadata.filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download document {document_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download document"
        )


@router.post(
    "/bulk/process",
    response_model=List[DocumentProcessingResponse],
    summary="Bulk process documents",
    description="Process multiple documents in bulk"
)
async def bulk_process_documents(
    request: BulkProcessingRequest,
    document_repository=Depends(get_document_repository),
    file_storage=Depends(get_file_storage_service),
    document_processing_service=Depends(get_document_processing_service)
):
    """
    Bulk process documents following Single Responsibility Principle.
    
    Args:
        request: Bulk processing request
        document_repository: Document repository dependency
        file_storage: File storage service dependency
        document_processing_service: Document processing service dependency
        
    Returns:
        List of processing responses
        
    Raises:
        HTTPException: If bulk processing fails
    """
    try:
        # Parse status filter
        parsed_status = None
        if request.status_filter:
            try:
                parsed_status = DocumentStatus(request.status_filter.upper())
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid status: {request.status_filter}"
                )
        
        # Create PDF parsing service
        pdf_parsing_service = PDFParsingServiceFactory.create_default()
        
        # Create processing use case
        processing_use_case = DocumentProcessingUseCase(
            document_repository=document_repository,
            file_storage=file_storage,
            pdf_parsing_service=pdf_parsing_service,
            document_processing_service=document_processing_service
        )
        
        # Create bulk processing use case
        bulk_use_case = BulkDocumentProcessingUseCase(
            document_repository=document_repository,
            document_processing_use_case=processing_use_case
        )
        
        responses = await bulk_use_case.execute(
            status_filter=parsed_status,
            limit=request.limit,
            force_reprocess=request.force_reprocess
        )
        
        # Convert to response format
        return [
            DocumentProcessingResponse(
                document_id=response.document_id.value,
                status=response.status.value,
                processing_stage=response.processing_stage.value,
                page_count=response.page_count,
                chunk_count=response.chunk_count,
                processing_time_ms=response.processing_time_ms,
                error_message=response.error_message
            )
            for response in responses
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Bulk processing failed"
        )
