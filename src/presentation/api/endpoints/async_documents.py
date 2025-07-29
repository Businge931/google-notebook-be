"""
Asynchronous Document Processing API Endpoints

Optimized for large PDF processing with:
- Non-blocking background processing
- Real-time progress tracking
- Job status monitoring
- WebSocket support for live updates
"""
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import logging
import json
import asyncio

from ....core.application.use_cases.async_document_processing_use_case import (
    AsyncDocumentProcessingUseCase,
    get_async_processing_use_case,
    ProcessingJob,
    JobStatus
)
from ....core.domain.value_objects import DocumentId
from ....core.domain.services.large_pdf_processing_service import (
    LargePDFProcessingService,
    LargePDFProcessingServiceFactory
)
from ....infrastructure.di import (
    get_document_repository,
    get_file_storage_service,
    get_document_processing_service
)
from ....infrastructure.di.ai_container import get_ai_container
from ....infrastructure.config.settings import get_settings
from ....shared.exceptions import DocumentNotFoundError, DocumentProcessingError
from ..schemas.document_schemas import DocumentProcessingResponse

# Create router
router = APIRouter(prefix="/documents", tags=["async-processing"])
logger = logging.getLogger(__name__)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        self.active_connections[job_id] = websocket
        logger.info(f"WebSocket connected for job {job_id}")

    def disconnect(self, job_id: str):
        if job_id in self.active_connections:
            del self.active_connections[job_id]
            logger.info(f"WebSocket disconnected for job {job_id}")

    async def send_progress(self, job_id: str, data: dict):
        websocket = self.active_connections.get(job_id)
        if websocket:
            try:
                await websocket.send_text(json.dumps(data))
            except Exception as e:
                logger.warning(f"Failed to send progress to {job_id}: {e}")
                self.disconnect(job_id)

manager = ConnectionManager()


def get_large_pdf_service(
    document_repository=Depends(get_document_repository),
    file_storage=Depends(get_file_storage_service)
) -> LargePDFProcessingService:
    """Get large PDF processing service with dependencies."""
    # Get AI services
    settings = get_settings()
    ai_container = get_ai_container(settings)
    embedding_service = ai_container.get_embedding_service()
    vector_repository = ai_container.get_vector_repository()
    
    # Create PDF parsing service
    from ....infrastructure.adapters.services.pdf_parsing_service import PDFParsingServiceFactory
    pdf_parsing_service = PDFParsingServiceFactory.create_default()
    
    # Create optimized large PDF service
    return LargePDFProcessingServiceFactory.create_optimized(
        document_repository=document_repository,
        file_storage=file_storage,
        pdf_parsing_service=pdf_parsing_service,
        embedding_service=embedding_service,
        vector_repository=vector_repository,
        max_memory_mb=512
    )


def get_async_processing_service(
    document_repository=Depends(get_document_repository),
    large_pdf_service: LargePDFProcessingService = Depends(get_large_pdf_service)
) -> AsyncDocumentProcessingUseCase:
    """Get async processing use case."""
    return get_async_processing_use_case(
        document_repository=document_repository,
        large_pdf_service=large_pdf_service
    )


@router.post(
    "/{document_id}/process-async",
    response_model=Dict[str, Any],
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start asynchronous document processing",
    description="Start background processing for large PDFs with immediate response"
)
async def start_async_processing(
    document_id: str,
    force_reprocess: bool = Query(False, description="Force reprocessing even if already processed"),
    async_service: AsyncDocumentProcessingUseCase = Depends(get_async_processing_service)
):
    """
    Start asynchronous document processing.
    
    Returns immediately with job_id for tracking progress.
    Use /documents/{document_id}/processing-status/{job_id} to check status.
    """
    try:
        # Start background processing
        job = await async_service.start_processing(
            document_id=DocumentId(document_id),
            force_reprocess=force_reprocess
        )
        
        logger.info(f"Started async processing job {job.job_id} for document {document_id}")
        
        return {
            "job_id": job.job_id,
            "document_id": job.document_id,
            "status": job.status.value,
            "created_at": job.created_at.isoformat(),
            "message": "Processing started in background",
            "status_url": f"/api/v1/documents/{document_id}/processing-status/{job.job_id}",
            "websocket_url": f"/api/v1/documents/progress/{job.job_id}"
        }
        
    except DocumentNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}"
        )
    except Exception as e:
        logger.error(f"Failed to start async processing: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start processing"
        )


@router.get(
    "/{document_id}/processing-status/{job_id}",
    response_model=Dict[str, Any],
    summary="Get processing job status",
    description="Get current status and progress of a processing job"
)
async def get_processing_status(
    document_id: str,
    job_id: str,
    async_service: AsyncDocumentProcessingUseCase = Depends(get_async_processing_service)
):
    """Get current processing status and progress."""
    try:
        job = await async_service.get_job_status(job_id)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {job_id}"
            )
        
        if job.document_id != document_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Job does not belong to this document"
            )
        
        response = {
            "job_id": job.job_id,
            "document_id": job.document_id,
            "status": job.status.value,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "error_message": job.error_message
        }
        
        # Add progress information if available
        if job.progress:
            response["progress"] = {
                "total_pages": job.progress.total_pages,
                "processed_pages": job.progress.processed_pages,
                "total_chunks": job.progress.total_chunks,
                "processed_chunks": job.progress.processed_chunks,
                "vectorized_chunks": job.progress.vectorized_chunks,
                "current_stage": job.progress.current_stage,
                "memory_usage_mb": job.progress.memory_usage_mb,
                "processing_time_ms": job.progress.processing_time_ms,
                "estimated_remaining_ms": job.progress.estimated_remaining_ms
            }
            
            # Calculate completion percentage
            if job.progress.total_chunks > 0:
                completion_percentage = (job.progress.vectorized_chunks / job.progress.total_chunks) * 100
                response["progress"]["completion_percentage"] = round(completion_percentage, 1)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get processing status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get processing status"
        )


@router.get(
    "/{document_id}/processing-status",
    response_model=Dict[str, Any],
    summary="Get current processing job for document",
    description="Get active processing job for a document"
)
async def get_document_processing_status(
    document_id: str,
    async_service: AsyncDocumentProcessingUseCase = Depends(get_async_processing_service)
):
    """Get active processing job for a document."""
    try:
        job = await async_service.get_document_job(document_id)
        
        if not job:
            return {
                "document_id": document_id,
                "status": "no_active_job",
                "message": "No active processing job for this document"
            }
        
        # Return same format as get_processing_status
        response = {
            "job_id": job.job_id,
            "document_id": job.document_id,
            "status": job.status.value,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "error_message": job.error_message
        }
        
        if job.progress:
            response["progress"] = {
                "total_pages": job.progress.total_pages,
                "processed_pages": job.progress.processed_pages,
                "total_chunks": job.progress.total_chunks,
                "processed_chunks": job.progress.processed_chunks,
                "vectorized_chunks": job.progress.vectorized_chunks,
                "current_stage": job.progress.current_stage,
                "memory_usage_mb": job.progress.memory_usage_mb,
                "processing_time_ms": job.progress.processing_time_ms,
                "estimated_remaining_ms": job.progress.estimated_remaining_ms
            }
            
            if job.progress.total_chunks > 0:
                completion_percentage = (job.progress.vectorized_chunks / job.progress.total_chunks) * 100
                response["progress"]["completion_percentage"] = round(completion_percentage, 1)
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get document processing status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get processing status"
        )


@router.delete(
    "/processing-jobs/{job_id}",
    response_model=Dict[str, Any],
    summary="Cancel processing job",
    description="Cancel a running processing job"
)
async def cancel_processing_job(
    job_id: str,
    async_service: AsyncDocumentProcessingUseCase = Depends(get_async_processing_service)
):
    """Cancel a processing job."""
    try:
        success = await async_service.cancel_job(job_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found or cannot be cancelled: {job_id}"
            )
        
        return {
            "job_id": job_id,
            "status": "cancelled",
            "message": "Processing job cancelled successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel job: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel job"
        )


@router.get(
    "/processing-jobs/summary",
    response_model=Dict[str, Any],
    summary="Get processing jobs summary",
    description="Get summary of all processing jobs"
)
async def get_jobs_summary(
    async_service: AsyncDocumentProcessingUseCase = Depends(get_async_processing_service)
):
    """Get summary of all processing jobs."""
    try:
        summary = async_service.get_job_summary()
        return summary
        
    except Exception as e:
        logger.error(f"Failed to get jobs summary: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get jobs summary"
        )


@router.websocket("/progress/{job_id}")
async def websocket_progress(
    websocket: WebSocket,
    job_id: str,
    async_service: AsyncDocumentProcessingUseCase = Depends(get_async_processing_service)
):
    """WebSocket endpoint for real-time progress updates."""
    await manager.connect(websocket, job_id)
    
    try:
        # Add progress callback for this job
        def send_progress_update(progress):
            asyncio.create_task(manager.send_progress(job_id, {
                "type": "progress_update",
                "job_id": job_id,
                "progress": {
                    "total_pages": progress.total_pages,
                    "processed_pages": progress.processed_pages,
                    "total_chunks": progress.total_chunks,
                    "processed_chunks": progress.processed_chunks,
                    "vectorized_chunks": progress.vectorized_chunks,
                    "current_stage": progress.current_stage,
                    "memory_usage_mb": progress.memory_usage_mb,
                    "processing_time_ms": progress.processing_time_ms,
                    "estimated_remaining_ms": progress.estimated_remaining_ms,
                    "completion_percentage": round((progress.vectorized_chunks / progress.total_chunks) * 100, 1) if progress.total_chunks > 0 else 0
                }
            }))
        
        async_service.add_progress_callback(job_id, send_progress_update)
        
        # Send initial status
        job = await async_service.get_job_status(job_id)
        if job:
            await manager.send_progress(job_id, {
                "type": "job_status",
                "job_id": job_id,
                "status": job.status.value,
                "message": f"Connected to job {job_id}"
            })
        
        # Keep connection alive and listen for client messages
        while True:
            try:
                # Wait for client message (ping/pong to keep alive)
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                elif message.get("type") == "get_status":
                    job = await async_service.get_job_status(job_id)
                    if job:
                        await manager.send_progress(job_id, {
                            "type": "job_status",
                            "job_id": job_id,
                            "status": job.status.value
                        })
                        
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.warning(f"WebSocket error for job {job_id}: {e}")
                break
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}", exc_info=True)
    finally:
        manager.disconnect(job_id)


# Background task for cleanup
@router.on_event("startup")
async def startup_cleanup_task():
    """Start background cleanup task."""
    async def cleanup_old_jobs():
        while True:
            try:
                # This would need to be injected properly in a real implementation
                # For now, we'll skip the cleanup in startup
                await asyncio.sleep(3600)  # Run every hour
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
                await asyncio.sleep(3600)
    
    # Start cleanup task
    asyncio.create_task(cleanup_old_jobs())
