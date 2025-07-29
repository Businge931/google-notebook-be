"""
Document Management Use Case
"""
from typing import List, Optional
from datetime import datetime

from ...domain.entities import Document, DocumentStatus, ProcessingStage
from ...domain.repositories import DocumentRepository
from ...domain.services import DocumentProcessingService
from ...domain.value_objects import DocumentId
from ....infrastructure.adapters.storage.file_storage import FileStorage
from src.shared.exceptions import (
    DocumentNotFoundError,
    DocumentProcessingError,
    RepositoryError,
    FileStorageError,
    ValidationError
)


class DocumentRetrievalRequest:
    """
    Request object for document retrieval following Single Responsibility Principle.
    """
    
    def __init__(
        self,
        document_id: Optional[DocumentId] = None,
        status_filter: Optional[DocumentStatus] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ):
        """
        Initialize retrieval request.
        
        Args:
            document_id: Specific document ID to retrieve
            status_filter: Filter documents by status
            limit: Maximum number of documents to return
            offset: Number of documents to skip
        """
        self.document_id = document_id
        self.status_filter = status_filter
        self.limit = limit
        self.offset = offset


class DocumentUpdateRequest:
    """
    Request object for document updates following Single Responsibility Principle.
    """
    
    def __init__(
        self,
        document_id: DocumentId,
        status: Optional[DocumentStatus] = None,
        processing_stage: Optional[ProcessingStage] = None,
        processing_error: Optional[str] = None
    ):
        """
        Initialize update request.
        
        Args:
            document_id: Document ID to update
            status: New document status
            processing_stage: New processing stage
            processing_error: Processing error message
        """
        self.document_id = document_id
        self.status = status
        self.processing_stage = processing_stage
        self.processing_error = processing_error


class DocumentDeletionRequest:
    """
    Request object for document deletion following Single Responsibility Principle.
    """
    
    def __init__(
        self,
        document_id: DocumentId,
        delete_file: bool = True
    ):
        """
        Initialize deletion request.
        
        Args:
            document_id: Document ID to delete
            delete_file: Whether to delete the associated file
        """
        self.document_id = document_id
        self.delete_file = delete_file


class DocumentManagementResponse:
    """
    Response object for document management operations.
    """
    
    def __init__(
        self,
        success: bool,
        document: Optional[Document] = None,
        documents: Optional[List[Document]] = None,
        message: Optional[str] = None,
        error_message: Optional[str] = None
    ):
        """
        Initialize management response.
        
        Args:
            success: Whether operation was successful
            document: Single document result
            documents: Multiple documents result
            message: Success message
            error_message: Error message if operation failed
        """
        self.success = success
        self.document = document
        self.documents = documents or []
        self.message = message
        self.error_message = error_message


class DocumentManagementUseCase:
    """
    Use case for managing documents following Single Responsibility Principle.
    
    Handles CRUD operations, status updates, and document lifecycle management.
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
    
    async def get_document(self, document_id: DocumentId) -> DocumentManagementResponse:
        """
        Retrieve a single document by ID.
        
        Args:
            document_id: Unique document identifier
            
        Returns:
            Document management response
        """
        try:
            document = await self._document_repository.find_by_id(document_id)
            
            if document is None:
                return DocumentManagementResponse(
                    success=False,
                    error_message=f"Document not found: {document_id.value}"
                )
            
            return DocumentManagementResponse(
                success=True,
                document=document,
                message="Document retrieved successfully"
            )
            
        except RepositoryError as e:
            return DocumentManagementResponse(
                success=False,
                error_message=f"Failed to retrieve document: {str(e)}"
            )
    
    async def list_documents(self, request: DocumentRetrievalRequest) -> DocumentManagementResponse:
        """
        List documents based on criteria.
        
        Args:
            request: Document retrieval request
            
        Returns:
            Document management response with list of documents
        """
        try:
            if request.status_filter:
                documents = await self._document_repository.find_by_status(request.status_filter)
            else:
                documents = await self._document_repository.find_all(
                    limit=request.limit,
                    offset=request.offset
                )
            
            return DocumentManagementResponse(
                success=True,
                documents=documents,
                message=f"Retrieved {len(documents)} documents"
            )
            
        except RepositoryError as e:
            return DocumentManagementResponse(
                success=False,
                error_message=f"Failed to list documents: {str(e)}"
            )
    
    async def update_document(self, request: DocumentUpdateRequest) -> DocumentManagementResponse:
        """
        Update document status and metadata.
        
        Args:
            request: Document update request
            
        Returns:
            Document management response
        """
        try:
            # Retrieve existing document
            document = await self._document_repository.find_by_id(request.document_id)
            if document is None:
                return DocumentManagementResponse(
                    success=False,
                    error_message=f"Document not found: {request.document_id.value}"
                )
            
            # Validate update request using domain service
            if request.status and request.processing_stage:
                self._document_processing_service.validate_status_transition(
                    document.status,
                    request.status,
                    document.processing_stage,
                    request.processing_stage
                )
            
            # Apply updates
            if request.status:
                document.status = request.status
            
            if request.processing_stage:
                document.processing_stage = request.processing_stage
            
            if request.processing_error:
                document.processing_error = request.processing_error
            
            # Update timestamp
            document.updated_at = datetime.utcnow()
            
            # Save updated document
            updated_document = await self._document_repository.update(document)
            
            return DocumentManagementResponse(
                success=True,
                document=updated_document,
                message="Document updated successfully"
            )
            
        except (DocumentNotFoundError, ValidationError, DocumentProcessingError) as e:
            return DocumentManagementResponse(
                success=False,
                error_message=str(e)
            )
        except RepositoryError as e:
            return DocumentManagementResponse(
                success=False,
                error_message=f"Failed to update document: {str(e)}"
            )
    
    async def delete_document(self, request: DocumentDeletionRequest) -> DocumentManagementResponse:
        """
        Delete document and optionally its associated file.
        
        Args:
            request: Document deletion request
            
        Returns:
            Document management response
        """
        try:
            # Retrieve document to get file path
            document = await self._document_repository.find_by_id(request.document_id)
            if document is None:
                return DocumentManagementResponse(
                    success=False,
                    error_message=f"Document not found: {request.document_id.value}"
                )
            
            # Delete file if requested
            file_deleted = False
            if request.delete_file and document.file_path:
                try:
                    file_deleted = await self._file_storage.delete_file(document.file_path)
                except FileStorageError as e:
                    # Log error but continue with database deletion
                    pass
            
            # Delete document from repository
            deleted = await self._document_repository.delete(request.document_id)
            
            if not deleted:
                return DocumentManagementResponse(
                    success=False,
                    error_message=f"Document not found: {request.document_id.value}"
                )
            
            message = "Document deleted successfully"
            if request.delete_file:
                if file_deleted:
                    message += " (file also deleted)"
                else:
                    message += " (file deletion failed)"
            
            return DocumentManagementResponse(
                success=True,
                message=message
            )
            
        except (ValidationError, DocumentProcessingError) as e:
            return DocumentManagementResponse(
                success=False,
                error_message=str(e)
            )
        except RepositoryError as e:
            return DocumentManagementResponse(
                success=False,
                error_message=f"Failed to delete document: {str(e)}"
            )
    
    async def get_document_statistics(self) -> DocumentManagementResponse:
        """
        Get document statistics and counts.
        
        Returns:
            Document management response with statistics
        """
        try:
            # Get total count
            total_count = await self._document_repository.count()
            
            # Get counts by status
            status_counts = {}
            for status in DocumentStatus:
                documents = await self._document_repository.find_by_status(status)
                status_counts[status.value] = len(documents)
            
            # Get processing documents
            processing_documents = await self._document_repository.find_processing_documents()
            
            statistics = {
                'total_documents': total_count,
                'status_counts': status_counts,
                'processing_count': len(processing_documents),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return DocumentManagementResponse(
                success=True,
                message="Statistics retrieved successfully",
                # Store statistics in a way that can be accessed
                documents=[]  # Could extend response to include statistics
            )
            
        except RepositoryError as e:
            return DocumentManagementResponse(
                success=False,
                error_message=f"Failed to get statistics: {str(e)}"
            )
    
    async def check_document_exists(self, document_id: DocumentId) -> DocumentManagementResponse:
        """
        Check if a document exists.
        
        Args:
            document_id: Document ID to check
            
        Returns:
            Document management response
        """
        try:
            exists = await self._document_repository.exists(document_id)
            
            return DocumentManagementResponse(
                success=True,
                message=f"Document {'exists' if exists else 'does not exist'}"
            )
            
        except RepositoryError as e:
            return DocumentManagementResponse(
                success=False,
                error_message=f"Failed to check document existence: {str(e)}"
            )


class DocumentSearchUseCase:
    """
    Use case for searching documents following Single Responsibility Principle.
    
    Handles document search and filtering operations.
    """
    
    def __init__(
        self,
        document_repository: DocumentRepository
    ):
        """
        Initialize search use case.
        
        Args:
            document_repository: Repository for document persistence
        """
        self._document_repository = document_repository
    
    async def search_by_filename(
        self,
        filename_pattern: str,
        limit: Optional[int] = None
    ) -> DocumentManagementResponse:
        """
        Search documents by filename pattern.
        
        Args:
            filename_pattern: Pattern to search for in filenames
            limit: Maximum number of results
            
        Returns:
            Document management response with matching documents
        """
        try:
            # Get all documents and filter by filename
            all_documents = await self._document_repository.find_all()
            
            matching_documents = [
                doc for doc in all_documents
                if filename_pattern.lower() in doc.file_metadata.filename.lower()
            ]
            
            if limit:
                matching_documents = matching_documents[:limit]
            
            return DocumentManagementResponse(
                success=True,
                documents=matching_documents,
                message=f"Found {len(matching_documents)} matching documents"
            )
            
        except RepositoryError as e:
            return DocumentManagementResponse(
                success=False,
                error_message=f"Search failed: {str(e)}"
            )
    
    async def search_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        limit: Optional[int] = None
    ) -> DocumentManagementResponse:
        """
        Search documents by upload date range.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            limit: Maximum number of results
            
        Returns:
            Document management response with matching documents
        """
        try:
            # Get all documents and filter by date
            all_documents = await self._document_repository.find_all()
            
            matching_documents = [
                doc for doc in all_documents
                if start_date <= doc.file_metadata.upload_timestamp <= end_date
            ]
            
            # Sort by upload timestamp (newest first)
            matching_documents.sort(
                key=lambda doc: doc.file_metadata.upload_timestamp,
                reverse=True
            )
            
            if limit:
                matching_documents = matching_documents[:limit]
            
            return DocumentManagementResponse(
                success=True,
                documents=matching_documents,
                message=f"Found {len(matching_documents)} documents in date range"
            )
            
        except RepositoryError as e:
            return DocumentManagementResponse(
                success=False,
                error_message=f"Date range search failed: {str(e)}"
            )


class DocumentManagementUseCaseFactory:
    """
    Factory for creating document management use case instances.
    
    Follows Open/Closed Principle for extensibility.
    """
    
    @staticmethod
    def create(
        document_repository: DocumentRepository,
        file_storage: FileStorage,
        document_processing_service: DocumentProcessingService
    ) -> DocumentManagementUseCase:
        """
        Create document management use case instance.
        
        Args:
            document_repository: Repository for document persistence
            file_storage: Service for file storage operations
            document_processing_service: Service for document processing logic
            
        Returns:
            Configured document management use case
        """
        return DocumentManagementUseCase(
            document_repository=document_repository,
            file_storage=file_storage,
            document_processing_service=document_processing_service
        )
    
    @staticmethod
    def create_search(
        document_repository: DocumentRepository
    ) -> DocumentSearchUseCase:
        """
        Create document search use case instance.
        
        Args:
            document_repository: Repository for document persistence
            
        Returns:
            Configured document search use case
        """
        return DocumentSearchUseCase(document_repository=document_repository)
