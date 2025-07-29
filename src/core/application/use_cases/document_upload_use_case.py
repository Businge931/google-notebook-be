"""
Document Upload Use Case

"""
from typing import BinaryIO, Optional
from datetime import datetime
import uuid

from ...domain.entities import Document, DocumentStatus, ProcessingStage
from ...domain.repositories import DocumentRepository
from ...domain.services import DocumentProcessingService
from ...domain.value_objects import DocumentId, FileMetadata
from ....infrastructure.adapters.storage.file_storage import FileStorage, generate_file_path
from src.shared.exceptions import (
    ValidationError,
    FileStorageError,
    RepositoryError,
    DocumentProcessingError
)
from src.shared.constants import FileConstants
from src.shared.utils import (
    validate_filename,
    validate_file_size,
    validate_mime_type,
    sanitize_filename
)


class DocumentUploadRequest:
    """
    Request object for document upload following Single Responsibility Principle.
    """
    
    def __init__(
        self,
        file_data: BinaryIO,
        filename: str,
        file_size: int,
        mime_type: str,
        original_path: Optional[str] = None
    ):
        """
        Initialize upload request.
        
        Args:
            file_data: Binary file data
            filename: Name of the file
            file_size: Size of file in bytes
            mime_type: MIME type of the file
            original_path: Original file path (optional)
        """
        self.file_data = file_data
        self.filename = filename
        self.file_size = file_size
        self.mime_type = mime_type
        self.original_path = original_path


class DocumentUploadResponse:
    """
    Response object for document upload following Single Responsibility Principle.
    """
    
    def __init__(
        self,
        document_id: DocumentId,
        filename: str,
        file_size: int,
        status: DocumentStatus,
        processing_stage: ProcessingStage,
        file_url: Optional[str] = None,
        upload_timestamp: Optional[datetime] = None
    ):
        """
        Initialize upload response.
        
        Args:
            document_id: Unique document identifier
            filename: Name of the uploaded file
            file_size: Size of file in bytes
            status: Current document status
            processing_stage: Current processing stage
            file_url: URL to access the file
            upload_timestamp: When the file was uploaded
        """
        self.document_id = document_id
        self.filename = filename
        self.file_size = file_size
        self.status = status
        self.processing_stage = processing_stage
        self.file_url = file_url
        self.upload_timestamp = upload_timestamp


class DocumentUploadUseCase:
    """
    Use case for uploading documents following Single Responsibility Principle.
    
    Orchestrates the document upload workflow including validation,
    file storage, and database persistence.
    """
    
    def __init__(
        self,
        document_repository: DocumentRepository,
        file_storage: FileStorage,
        document_processing_service: DocumentProcessingService
    ):
        """
        Initialize use case with dependencies.
        
        Args:
            document_repository: Repository for document persistence
            file_storage: Service for file storage operations
            document_processing_service: Service for document processing logic
        """
        self._document_repository = document_repository
        self._file_storage = file_storage
        self._document_processing_service = document_processing_service
    
    async def execute(self, request: DocumentUploadRequest) -> DocumentUploadResponse:
        """
        Execute document upload use case.
        
        Args:
            request: Document upload request
            
        Returns:
            Document upload response
            
        Raises:
            ValidationError: If request validation fails
            FileStorageError: If file storage operation fails
            RepositoryError: If database operation fails
            DocumentProcessingError: If document processing fails
        """
        try:
            # Step 1: Validate request
            await self._validate_request(request)
            
            # Step 2: Generate document ID and file path
            document_id = DocumentId(str(uuid.uuid4()))
            sanitized_filename = sanitize_filename(request.filename)
            file_path = generate_file_path(document_id.value, sanitized_filename)
            
            # Step 3: Create file metadata
            file_metadata = FileMetadata(
                filename=sanitized_filename,
                file_size=request.file_size,
                mime_type=request.mime_type,
                upload_timestamp=datetime.utcnow(),
                original_path=request.original_path
            )
            
            # Step 4: Store file
            stored_file_path = await self._file_storage.save_file(
                request.file_data,
                file_path,
                request.mime_type
            )
            
            # Step 6: Create document entity
            document = Document(
                document_id=document_id,
                file_metadata=file_metadata,
                status=DocumentStatus.UPLOADED,
                processing_stage=ProcessingStage.UPLOAD_COMPLETE,
                file_path=stored_file_path
            )
            
            # Step 7: Save document to repository
            saved_document = await self._document_repository.save(document)
            
            # Step 8: Generate file URL
            file_url = await self._file_storage.get_file_url(file_path)
            
            # Step 9: Create response
            return DocumentUploadResponse(
                document_id=saved_document.document_id,
                filename=saved_document.file_metadata.filename,
                file_size=saved_document.file_metadata.file_size,
                status=saved_document.status,
                processing_stage=saved_document.processing_stage,
                file_url=file_url,
                upload_timestamp=saved_document.file_metadata.upload_timestamp
            )
            
        except (ValidationError, FileStorageError, RepositoryError, DocumentProcessingError):
            # Re-raise domain exceptions
            raise
        except Exception as e:
            # Wrap unexpected exceptions
            raise DocumentProcessingError(f"Unexpected error during upload: {str(e)}")
    
    async def _validate_request(self, request: DocumentUploadRequest) -> None:
        """
        Validate upload request following Single Responsibility Principle.
        
        Args:
            request: Upload request to validate
            
        Raises:
            ValidationError: If validation fails
        """
        # Validate filename
        if not validate_filename(request.filename):
            raise ValidationError(f"Invalid filename: {request.filename}")
        
        # Validate file size
        if not validate_file_size(request.file_size):
            raise ValidationError(
                f"File size {request.file_size} exceeds maximum allowed size "
                f"{FileConstants.MAX_FILE_SIZE_BYTES}"
            )
        
        # Validate MIME type
        if not validate_mime_type(request.mime_type):
            raise ValidationError(f"Unsupported file type: {request.mime_type}")
        
        # Validate file data
        if not request.file_data:
            raise ValidationError("File data is required")
        
        # Check if file data is readable
        try:
            current_position = request.file_data.tell()
            request.file_data.seek(0, 2)  # Seek to end
            actual_size = request.file_data.tell()
            request.file_data.seek(current_position)  # Reset position
            
            if actual_size != request.file_size:
                raise ValidationError(
                    f"File size mismatch: expected {request.file_size}, "
                    f"actual {actual_size}"
                )
        except Exception as e:
            raise ValidationError(f"Invalid file data: {str(e)}")


class DocumentUploadUseCaseFactory:
    """
    Factory for creating DocumentUploadUseCase instances.
    
    Follows Open/Closed Principle - can be extended with new configurations.
    """
    
    @staticmethod
    def create(
        document_repository: DocumentRepository,
        file_storage: FileStorage,
        document_processing_service: DocumentProcessingService
    ) -> DocumentUploadUseCase:
        """
        Create document upload use case instance.
        
        Args:
            document_repository: Repository for document persistence
            file_storage: Service for file storage operations
            document_processing_service: Service for document processing logic
            
        Returns:
            Configured document upload use case
        """
        return DocumentUploadUseCase(
            document_repository=document_repository,
            file_storage=file_storage,
            document_processing_service=document_processing_service
        )
