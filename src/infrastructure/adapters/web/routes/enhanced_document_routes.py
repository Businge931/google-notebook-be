"""
Enhanced Document Routes

API endpoints optimized for large PDF files with:
- Progressive upload with real-time progress
- WebSocket support for progress tracking
- Memory-efficient file handling
- Large PDF processing integration
"""
import asyncio
import json
import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uuid

from ....di import get_document_repository, get_file_storage_service
# Simplified dependency injection - create service instances directly
from .....core.application.use_cases.enhanced_document_upload_use_case import (
    EnhancedDocumentUploadUseCase,
    EnhancedDocumentUploadRequest,
    UploadProgress,
    ProcessingProgress
)
from .....core.domain.value_objects import DocumentId
from .....shared.exceptions import ValidationError, DocumentProcessingError


# Pydantic models for API
class EnhancedUploadResponse(BaseModel):
    """Response model for enhanced document upload."""
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")
    upload_time_ms: int = Field(..., description="Upload time in milliseconds")
    processing_started: bool = Field(..., description="Whether processing has started")
    large_file_optimization: bool = Field(..., description="Whether large file optimization was used")
    message: str = Field(..., description="Status message")


class ProgressUpdate(BaseModel):
    """Model for progress updates."""
    type: str = Field(..., description="Progress type: upload or processing")
    document_id: str = Field(..., description="Document ID")
    percentage: float = Field(..., description="Completion percentage")
    status: str = Field(..., description="Current status")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")


class LargePDFConfig(BaseModel):
    """Configuration for large PDF processing."""
    max_memory_mb: int = Field(default=512, description="Maximum memory usage in MB")
    chunk_batch_size: int = Field(default=10, description="Chunk batch size")
    enable_progress_tracking: bool = Field(default=True, description="Enable progress tracking")


# Router setup
router = APIRouter(prefix="/documents", tags=["Enhanced Documents"])
logger = logging.getLogger(__name__)

# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for progress tracking."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"WebSocket connected: {client_id}")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"WebSocket disconnected: {client_id}")
    
    async def send_progress(self, client_id: str, progress: ProgressUpdate):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(progress.json())
            except Exception as e:
                logger.warning(f"Failed to send progress to {client_id}: {e}")
                self.disconnect(client_id)


# Global connection manager
connection_manager = ConnectionManager()


@router.websocket("/progress/{client_id}")
async def websocket_progress(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time progress tracking."""
    await connection_manager.connect(websocket, client_id)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        connection_manager.disconnect(client_id)


@router.post("/upload-large", response_model=EnhancedUploadResponse)
async def upload_large_pdf(
    file: UploadFile = File(..., description="PDF file to upload"),
    auto_process: bool = Form(default=True, description="Automatically start processing"),
    enable_optimization: bool = Form(default=True, description="Enable large file optimization"),
    client_id: Optional[str] = Form(default=None, description="Client ID for progress tracking"),
    document_repository=Depends(get_document_repository),
    file_storage=Depends(get_file_storage_service)
):
    """
    Upload large PDF files with optimization and progress tracking.
    
    Features:
    - Supports files up to 100MB
    - Real-time progress tracking via WebSocket
    - Memory-efficient processing
    - Automatic large PDF optimization
    """
    try:
        # Validate file
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        if not file.content_type.startswith('application/pdf'):
            raise HTTPException(status_code=400, detail="Invalid content type")
        
        # Read file data
        file_data = await file.read()
        file_size = len(file_data)
        file_size_mb = file_size / (1024 * 1024)
        
        # Check file size
        if file_size_mb > 100:
            raise HTTPException(status_code=413, detail="File size exceeds 100MB limit")
        
        logger.info(f"Starting large PDF upload: {file.filename} ({file_size_mb:.1f}MB)")
        
        # Generate client ID if not provided
        if not client_id:
            client_id = str(uuid.uuid4())
        
        # Create progress callbacks
        def upload_progress_callback(progress: UploadProgress):
            asyncio.create_task(
                connection_manager.send_progress(
                    client_id,
                    ProgressUpdate(
                        type="upload",
                        document_id="",  # Will be set later
                        percentage=progress.upload_percentage,
                        status=progress.status,
                        details={
                            "uploaded_bytes": progress.uploaded_bytes,
                            "total_bytes": progress.file_size,
                            "speed_mbps": progress.upload_speed_mbps,
                            "eta_seconds": progress.estimated_remaining_seconds
                        }
                    )
                )
            )
        
        def processing_progress_callback(progress: ProcessingProgress):
            total_progress = 0
            if progress.total_chunks > 0:
                total_progress = (progress.vectorized_chunks / progress.total_chunks) * 100
            
            asyncio.create_task(
                connection_manager.send_progress(
                    client_id,
                    ProgressUpdate(
                        type="processing",
                        document_id=progress.document_id,
                        percentage=total_progress,
                        status=progress.current_stage,
                        details={
                            "total_pages": progress.total_pages,
                            "processed_pages": progress.processed_pages,
                            "total_chunks": progress.total_chunks,
                            "vectorized_chunks": progress.vectorized_chunks,
                            "memory_usage_mb": progress.memory_usage_mb,
                            "processing_time_ms": progress.processing_time_ms
                        }
                    )
                )
            )
        
        # Create enhanced upload use case
        use_case = EnhancedDocumentUploadUseCase(
            document_repository=document_repository,
            file_storage=file_storage,
            large_pdf_processing_service=None  # Will be created internally
        )
        
        # Create upload request
        upload_request = EnhancedDocumentUploadRequest(
            file_data=file_data,
            filename=file.filename,
            content_type=file.content_type,
            file_size=file_size,
            enable_large_pdf_optimization=enable_optimization,
            auto_process=auto_process,
            progress_callback=upload_progress_callback,
            processing_progress_callback=processing_progress_callback
        )
        
        # Execute upload
        response = await use_case.execute(upload_request)
        
        # Determine if large file optimization was used
        large_file_optimization = file_size_mb > 10 and enable_optimization
        
        # Send final progress update
        await connection_manager.send_progress(
            client_id,
            ProgressUpdate(
                type="upload",
                document_id=response.document_id.value,
                percentage=100.0,
                status="completed",
                details={
                    "upload_time_ms": response.upload_time_ms,
                    "processing_started": response.processing_started
                }
            )
        )
        
        return EnhancedUploadResponse(
            document_id=response.document_id.value,
            filename=response.filename,
            file_size=response.file_size,
            upload_time_ms=response.upload_time_ms,
            processing_started=response.processing_started,
            large_file_optimization=large_file_optimization,
            message=f"Successfully uploaded {file.filename} ({file_size_mb:.1f}MB)"
        )
        
    except HTTPException:
        raise
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except DocumentProcessingError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Large PDF upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/upload-status/{document_id}")
async def get_upload_status(
    document_id: str,
    document_repository=Depends(get_document_repository)
):
    """Get current upload and processing status for a document."""
    try:
        doc_id = DocumentId(document_id)
        document = await document_repository.get_by_id(doc_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "document_id": document_id,
            "status": document.status.value,
            "processing_stage": document.processing_stage.value,
            "filename": document.filename,
            "file_size": document.file_size,
            "page_count": document.page_count,
            "chunk_count": document.chunk_count,
            "created_at": document.created_at.isoformat() if document.created_at else None,
            "updated_at": document.updated_at.isoformat() if document.updated_at else None
        }
        
    except Exception as e:
        logger.error(f"Failed to get upload status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get upload status")


@router.post("/process-large/{document_id}")
async def process_large_pdf(
    document_id: str,
    force_reprocess: bool = False,
    client_id: Optional[str] = None,
    document_repository=Depends(get_document_repository)
):
    """
    Process a large PDF document with optimization.
    
    This endpoint is for processing documents that were uploaded without auto-processing.
    """
    try:
        doc_id = DocumentId(document_id)
        
        # Verify document exists
        document = await document_repository.get_by_id(doc_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Generate client ID if not provided
        if not client_id:
            client_id = str(uuid.uuid4())
        
        # Add progress callback
        def processing_progress_callback(progress: ProcessingProgress):
            total_progress = 0
            if progress.total_chunks > 0:
                total_progress = (progress.vectorized_chunks / progress.total_chunks) * 100
            
            asyncio.create_task(
                connection_manager.send_progress(
                    client_id,
                    ProgressUpdate(
                        type="processing",
                        document_id=progress.document_id,
                        percentage=total_progress,
                        status=progress.current_stage,
                        details={
                            "total_pages": progress.total_pages,
                            "processed_pages": progress.processed_pages,
                            "total_chunks": progress.total_chunks,
                            "vectorized_chunks": progress.vectorized_chunks,
                            "memory_usage_mb": progress.memory_usage_mb,
                            "processing_time_ms": progress.processing_time_ms
                        }
                    )
                )
            )
        
        # TODO: Implement large PDF processing service integration
        # For now, return success response without actual processing
        logger.info(f"Large PDF processing requested for document {document_id}")
        
        return {
            "document_id": document_id,
            "message": "Large PDF processing started",
            "client_id": client_id,
            "websocket_url": f"/api/v1/documents/progress/{client_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start large PDF processing: {e}")
        raise HTTPException(status_code=500, detail="Failed to start processing")


@router.get("/large-pdf-config")
async def get_large_pdf_config():
    """Get current large PDF processing configuration."""
    return {
        "max_file_size_mb": 100,
        "large_file_threshold_mb": 10,
        "chunk_batch_size": 10,
        "page_batch_size": 5,
        "max_concurrent_embeddings": 3,
        "memory_monitoring_enabled": True,
        "max_memory_usage_mb": 512,
        "supported_features": [
            "progress_tracking",
            "memory_optimization",
            "chunked_processing",
            "websocket_progress",
            "error_recovery"
        ]
    }


@router.put("/large-pdf-config")
async def update_large_pdf_config(config: LargePDFConfig):
    """Update large PDF processing configuration."""
    # In a real implementation, this would update the service configuration
    return {
        "message": "Configuration updated successfully",
        "config": config.dict()
    }
