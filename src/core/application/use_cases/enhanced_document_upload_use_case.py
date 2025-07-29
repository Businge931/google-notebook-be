"""
Enhanced Document Upload Use Case

Optimized for large PDF files with:
- Progressive upload with chunked streaming
- Real-time progress tracking
- Memory-efficient file handling
- Integration with large PDF processing service
"""
import asyncio
import logging
from typing import Optional, Dict, Any, Callable
from datetime import datetime
import aiofiles
import os

from ...domain.entities import Document, DocumentStatus, ProcessingStage
from ...domain.repositories import DocumentRepository
from ...domain.value_objects import DocumentId, FileMetadata
from ...domain.services.large_pdf_processing_service import (
    LargePDFProcessingService, 
    ProcessingProgress,
    LargePDFConfig
)
from ....infrastructure.adapters.storage.file_storage import FileStorage
from ....infrastructure.adapters.services.pdf_parsing_service import PDFParsingService
from ....shared.exceptions import (
    DocumentProcessingError,
    FileStorageError,
    ValidationError
)
from ....shared.constants import ProcessingConstants


class UploadProgress:
    """Progress tracking for file upload."""
    
    def __init__(self, file_size: int):
        self.file_size = file_size
        self.uploaded_bytes = 0
        self.upload_percentage = 0.0
        self.upload_speed_mbps = 0.0
        self.estimated_remaining_seconds = 0
        self.status = "uploading"
        self.error_message: Optional[str] = None
        self.start_time = datetime.now()
    
    def update(self, uploaded_bytes: int):
        """Update upload progress."""
        self.uploaded_bytes = uploaded_bytes
        self.upload_percentage = (uploaded_bytes / self.file_size) * 100
        
        # Calculate speed and ETA
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        if elapsed_time > 0:
            self.upload_speed_mbps = (uploaded_bytes / (1024 * 1024)) / elapsed_time
            remaining_bytes = self.file_size - uploaded_bytes
            if self.upload_speed_mbps > 0:
                self.estimated_remaining_seconds = remaining_bytes / (self.upload_speed_mbps * 1024 * 1024)


class EnhancedDocumentUploadRequest:
    """Enhanced request for document upload with large file support."""
    
    def __init__(
        self,
        file_data: bytes,
        filename: str,
        content_type: str,
        file_size: int,
        enable_large_pdf_optimization: bool = True,
        auto_process: bool = True,
        progress_callback: Optional[Callable[[UploadProgress], None]] = None,
        processing_progress_callback: Optional[Callable[[ProcessingProgress], None]] = None
    ):
        self.file_data = file_data
        self.filename = filename
        self.content_type = content_type
        self.file_size = file_size
        self.enable_large_pdf_optimization = enable_large_pdf_optimization
        self.auto_process = auto_process
        self.progress_callback = progress_callback
        self.processing_progress_callback = processing_progress_callback


class EnhancedDocumentUploadResponse:
    """Enhanced response for document upload."""
    
    def __init__(
        self,
        document_id: DocumentId,
        filename: str,
        file_size: int,
        upload_time_ms: int,
        processing_started: bool = False,
        processing_progress: Optional[ProcessingProgress] = None,
        error_message: Optional[str] = None
    ):
        self.document_id = document_id
        self.filename = filename
        self.file_size = file_size
        self.upload_time_ms = upload_time_ms
        self.processing_started = processing_started
        self.processing_progress = processing_progress
        self.error_message = error_message


class EnhancedDocumentUploadUseCase:
    """
    Enhanced use case for uploading documents with large file optimization.
    
    Features:
    - Chunked streaming upload for large files
    - Real-time progress tracking
    - Memory-efficient file handling
    - Automatic large PDF processing
    - Error recovery and validation
    """
    
    def __init__(
        self,
        document_repository: DocumentRepository,
        file_storage: FileStorage,
        large_pdf_processing_service: LargePDFProcessingService
    ):
        """Initialize enhanced upload use case."""
        self._document_repository = document_repository
        self._file_storage = file_storage
        self._large_pdf_processing_service = large_pdf_processing_service
        self._logger = logging.getLogger(__name__)
        
        # Configuration for large files
        self._large_file_threshold_mb = 10  # Files > 10MB use optimized processing
        self._max_file_size_mb = 100        # Maximum allowed file size
        self._chunk_size_bytes = 1024 * 1024  # 1MB chunks for streaming
    
    async def execute(self, request: EnhancedDocumentUploadRequest) -> EnhancedDocumentUploadResponse:
        """
        Execute enhanced document upload with large file optimization.
        
        Args:
            request: Enhanced upload request
            
        Returns:
            Enhanced upload response with progress tracking
            
        Raises:
            ValidationError: If file validation fails
            FileStorageError: If file storage fails
            DocumentProcessingError: If processing fails
        """
        start_time = datetime.now()
        
        try:
            # Validate file
            await self._validate_file(request)
            
            # Create upload progress tracker
            upload_progress = UploadProgress(request.file_size)
            
            # Determine if this is a large file
            file_size_mb = request.file_size / (1024 * 1024)
            is_large_file = file_size_mb > self._large_file_threshold_mb
            
            self._logger.info(
                f"Starting upload: {request.filename} ({file_size_mb:.1f}MB) "
                f"- Large file optimization: {is_large_file and request.enable_large_pdf_optimization}"
            )
            
            # Create document entity
            document_id = DocumentId.generate()
            file_metadata = FileMetadata(
                filename=request.filename,
                file_size=request.file_size,
                mime_type=request.content_type,
                upload_timestamp=datetime.utcnow()
            )
            document = Document(
                document_id=document_id,
                file_metadata=file_metadata,
                status=DocumentStatus.UPLOADING,
                processing_stage=ProcessingStage.UPLOAD_STARTED
            )
            
            # Save initial document
            await self._document_repository.save(document)
            
            # Upload file with progress tracking
            if is_large_file:
                file_path = await self._upload_large_file(request, upload_progress, document_id)
            else:
                file_path = await self._upload_standard_file(request, upload_progress, document_id)
            
            # Update document with file path
            document.file_path = file_path
            document.status = DocumentStatus.UPLOADED
            document.processing_stage = ProcessingStage.UPLOAD_COMPLETE
            await self._document_repository.update(document)
            
            upload_progress.status = "completed"
            if request.progress_callback:
                request.progress_callback(upload_progress)
            
            # Calculate upload time
            end_time = datetime.now()
            upload_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Start processing if requested
            processing_started = False
            processing_progress = None
            
            if request.auto_process:
                if is_large_file and request.enable_large_pdf_optimization:
                    # Use large PDF processing service
                    processing_started = True
                    
                    # Add processing progress callback if provided
                    if request.processing_progress_callback and self._large_pdf_processing_service:
                        self._large_pdf_processing_service.add_progress_callback(
                            request.processing_progress_callback
                        )
                    
                    # Start processing asynchronously if service is available
                    if self._large_pdf_processing_service:
                        asyncio.create_task(
                            self._process_large_pdf_async(document_id)
                        )
                else:
                    # Use standard processing (will be handled by existing pipeline)
                    processing_started = True
            
            self._logger.info(
                f"Upload completed: {request.filename} in {upload_time_ms}ms "
                f"- Processing started: {processing_started}"
            )
            
            return EnhancedDocumentUploadResponse(
                document_id=document_id,
                filename=request.filename,
                file_size=request.file_size,
                upload_time_ms=upload_time_ms,
                processing_started=processing_started,
                processing_progress=processing_progress
            )
            
        except Exception as e:
            self._logger.error(f"Enhanced upload failed: {e}", exc_info=True)
            
            # Update document status if it exists
            try:
                if 'document' in locals():
                    document.status = DocumentStatus.FAILED
                    document.processing_stage = ProcessingStage.UPLOAD_FAILED
                    await self._document_repository.update(document)
            except:
                pass
            
            raise DocumentProcessingError(f"Enhanced upload failed: {str(e)}")
    
    async def _validate_file(self, request: EnhancedDocumentUploadRequest):
        """Validate file before upload."""
        # Check file size
        file_size_mb = request.file_size / (1024 * 1024)
        if file_size_mb > self._max_file_size_mb:
            raise ValidationError(
                f"File size ({file_size_mb:.1f}MB) exceeds maximum allowed size ({self._max_file_size_mb}MB)"
            )
        
        # Check content type
        if not request.content_type.startswith('application/pdf'):
            raise ValidationError(f"Unsupported content type: {request.content_type}")
        
        # Check filename
        if not request.filename.lower().endswith('.pdf'):
            raise ValidationError("Only PDF files are supported")
        
        # Validate file data
        if len(request.file_data) != request.file_size:
            raise ValidationError("File size mismatch")
    
    async def _upload_large_file(
        self, 
        request: EnhancedDocumentUploadRequest, 
        progress: UploadProgress,
        document_id: DocumentId
    ) -> str:
        """Upload large file with chunked streaming."""
        try:
            # Create temporary file path
            temp_filename = f"temp_{datetime.now().timestamp()}_{request.filename}"
            
            # Stream upload in chunks
            uploaded_bytes = 0
            
            async with aiofiles.tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_path = temp_file.name
                
                # Write file in chunks
                for i in range(0, len(request.file_data), self._chunk_size_bytes):
                    chunk = request.file_data[i:i + self._chunk_size_bytes]
                    await temp_file.write(chunk)
                    
                    uploaded_bytes += len(chunk)
                    progress.update(uploaded_bytes)
                    
                    # Notify progress callback
                    if request.progress_callback:
                        request.progress_callback(progress)
                    
                    # Small delay to prevent overwhelming
                    await asyncio.sleep(0.01)
            
            # Move to final storage location
            with open(temp_path, 'rb') as temp_file:
                # Generate proper file path for storage
                from ....infrastructure.adapters.storage.file_storage import generate_file_path
                storage_path = generate_file_path(str(document_id.value), request.filename)
                file_path = await self._file_storage.save_file(
                    temp_file, 
                    storage_path,
                    request.content_type
                )
            
            # Cleanup temp file
            os.unlink(temp_path)
            
            return file_path
            
        except Exception as e:
            raise FileStorageError(f"Large file upload failed: {str(e)}")
    
    async def _upload_standard_file(
        self, 
        request: EnhancedDocumentUploadRequest, 
        progress: UploadProgress,
        document_id: DocumentId
    ) -> str:
        """Upload standard file."""
        try:
            # Simulate progress for small files
            progress.update(request.file_size)
            if request.progress_callback:
                request.progress_callback(progress)
            
            # Store file
            import io
            file_stream = io.BytesIO(request.file_data)
            # Generate proper file path for storage
            from ....infrastructure.adapters.storage.file_storage import generate_file_path
            storage_path = generate_file_path(str(document_id.value), request.filename)
            file_path = await self._file_storage.save_file(
                file_stream, 
                storage_path,
                request.content_type
            )
            
            return file_path
            
        except Exception as e:
            raise FileStorageError(f"Standard file upload failed: {str(e)}")
    
    async def _process_large_pdf_async(self, document_id: DocumentId):
        """Process large PDF asynchronously."""
        try:
            await self._large_pdf_processing_service.process_large_pdf(
                document_id=document_id,
                force_reprocess=False
            )
        except Exception as e:
            self._logger.error(f"Async large PDF processing failed: {e}")


class EnhancedDocumentUploadUseCaseFactory:
    """Factory for creating enhanced document upload use case instances."""
    
    @staticmethod
    def create(
        document_repository: DocumentRepository,
        file_storage: FileStorage,
        large_pdf_processing_service: LargePDFProcessingService
    ) -> EnhancedDocumentUploadUseCase:
        """Create enhanced document upload use case."""
        return EnhancedDocumentUploadUseCase(
            document_repository=document_repository,
            file_storage=file_storage,
            large_pdf_processing_service=large_pdf_processing_service
        )
