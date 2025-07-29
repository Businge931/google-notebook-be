"""
Document API Schemas

"""
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator


class DocumentUploadResponse(BaseModel):
    """
    Response schema for document upload following Single Responsibility Principle.
    """
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Name of the uploaded file")
    file_size: int = Field(..., description="Size of the file in bytes")
    status: str = Field(..., description="Current document status")
    processing_stage: str = Field(..., description="Current processing stage")
    file_url: Optional[str] = Field(None, description="URL to access the file")
    upload_timestamp: datetime = Field(..., description="When the file was uploaded")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DocumentResponse(BaseModel):
    """
    Response schema for document details following Single Responsibility Principle.
    """
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Name of the file")
    file_size: int = Field(..., description="Size of the file in bytes")
    mime_type: str = Field(..., description="MIME type of the file")
    status: str = Field(..., description="Current document status")
    processing_stage: str = Field(..., description="Current processing stage")
    page_count: Optional[int] = Field(None, description="Number of pages in the document")
    chunk_count: Optional[int] = Field(None, description="Number of text chunks created")
    upload_timestamp: datetime = Field(..., description="When the file was uploaded")
    created_at: datetime = Field(..., description="When the document was created")
    updated_at: datetime = Field(..., description="When the document was last updated")
    processing_error: Optional[str] = Field(None, description="Processing error message if any")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DocumentListResponse(BaseModel):
    """
    Response schema for document list following Single Responsibility Principle.
    """
    documents: List[DocumentResponse] = Field(..., description="List of documents")
    total: int = Field(..., description="Total number of documents returned")
    limit: int = Field(..., description="Maximum number of documents requested")
    offset: int = Field(..., description="Number of documents skipped")


class DocumentProcessingResponse(BaseModel):
    """
    Response schema for document processing following Single Responsibility Principle.
    """
    document_id: str = Field(..., description="Unique document identifier")
    status: str = Field(..., description="Current document status")
    processing_stage: str = Field(..., description="Current processing stage")
    page_count: Optional[int] = Field(None, description="Number of pages processed")
    chunk_count: Optional[int] = Field(None, description="Number of chunks created")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")


class DocumentUpdateRequest(BaseModel):
    """
    Request schema for document updates following Single Responsibility Principle.
    """
    status: Optional[str] = Field(None, description="New document status")
    processing_stage: Optional[str] = Field(None, description="New processing stage")
    processing_error: Optional[str] = Field(None, description="Processing error message")
    
    @validator('status')
    def validate_status(cls, v):
        """Validate document status."""
        if v is not None:
            valid_statuses = ['UPLOADED', 'PROCESSING', 'COMPLETED', 'FAILED']
            if v.upper() not in valid_statuses:
                raise ValueError(f"Invalid status. Must be one of: {valid_statuses}")
        return v
    
    @validator('processing_stage')
    def validate_processing_stage(cls, v):
        """Validate processing stage."""
        if v is not None:
            valid_stages = [
                'UPLOAD_COMPLETE', 'PARSING_STARTED', 'PARSING_COMPLETE',
                'CHUNKING_STARTED', 'CHUNKING_COMPLETE', 'PARSING_FAILED'
            ]
            if v.upper() not in valid_stages:
                raise ValueError(f"Invalid processing stage. Must be one of: {valid_stages}")
        return v


class DocumentSearchRequest(BaseModel):
    """
    Request schema for document search following Single Responsibility Principle.
    """
    filename_pattern: Optional[str] = Field(None, description="Pattern to search in filenames")
    start_date: Optional[datetime] = Field(None, description="Start date for date range search")
    end_date: Optional[datetime] = Field(None, description="End date for date range search")
    status_filter: Optional[str] = Field(None, description="Filter by document status")
    limit: Optional[int] = Field(50, ge=1, le=100, description="Maximum number of results")
    
    @validator('status_filter')
    def validate_status_filter(cls, v):
        """Validate status filter."""
        if v is not None:
            valid_statuses = ['UPLOADED', 'PROCESSING', 'COMPLETED', 'FAILED']
            if v.upper() not in valid_statuses:
                raise ValueError(f"Invalid status filter. Must be one of: {valid_statuses}")
        return v
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        """Validate that end_date is after start_date."""
        if v is not None and 'start_date' in values and values['start_date'] is not None:
            if v <= values['start_date']:
                raise ValueError("end_date must be after start_date")
        return v


class BulkProcessingRequest(BaseModel):
    """
    Request schema for bulk document processing following Single Responsibility Principle.
    """
    status_filter: Optional[str] = Field(None, description="Only process documents with this status")
    limit: Optional[int] = Field(10, ge=1, le=50, description="Maximum number of documents to process")
    force_reprocess: bool = Field(False, description="Force reprocessing even if already processed")
    
    @validator('status_filter')
    def validate_status_filter(cls, v):
        """Validate status filter."""
        if v is not None:
            valid_statuses = ['UPLOADED', 'PROCESSING', 'COMPLETED', 'FAILED']
            if v.upper() not in valid_statuses:
                raise ValueError(f"Invalid status filter. Must be one of: {valid_statuses}")
        return v


class DocumentStatisticsResponse(BaseModel):
    """
    Response schema for document statistics following Single Responsibility Principle.
    """
    total_documents: int = Field(..., description="Total number of documents")
    status_counts: dict = Field(..., description="Count of documents by status")
    processing_count: int = Field(..., description="Number of documents currently processing")
    average_file_size: Optional[float] = Field(None, description="Average file size in bytes")
    total_pages: Optional[int] = Field(None, description="Total number of pages across all documents")
    total_chunks: Optional[int] = Field(None, description="Total number of chunks across all documents")
    timestamp: datetime = Field(..., description="When statistics were generated")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DocumentChunkResponse(BaseModel):
    """
    Response schema for document chunks following Single Responsibility Principle.
    """
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Document this chunk belongs to")
    page_number: int = Field(..., description="Page number this chunk is from")
    text_content: str = Field(..., description="Text content of the chunk")
    start_position: int = Field(..., description="Start position in the page")
    end_position: int = Field(..., description="End position in the page")


class DocumentChunkListResponse(BaseModel):
    """
    Response schema for document chunk list following Single Responsibility Principle.
    """
    chunks: List[DocumentChunkResponse] = Field(..., description="List of document chunks")
    total: int = Field(..., description="Total number of chunks")
    document_id: str = Field(..., description="Document ID these chunks belong to")


class ErrorResponse(BaseModel):
    """
    Response schema for API errors following Single Responsibility Principle.
    """
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[dict] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the error occurred")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthCheckResponse(BaseModel):
    """
    Response schema for health check following Single Responsibility Principle.
    """
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    version: str = Field(..., description="API version")
    database_status: str = Field(..., description="Database connection status")
    file_storage_status: str = Field(..., description="File storage status")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
