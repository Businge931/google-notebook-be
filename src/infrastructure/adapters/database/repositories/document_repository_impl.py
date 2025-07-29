"""
Document Repository Implementation
"""
from typing import List, Optional
from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.core.domain.entities import Document, DocumentStatus, DocumentChunk
from src.core.domain.repositories import DocumentRepository
from src.core.domain.value_objects import DocumentId, PageNumber, FileMetadata
from src.shared.exceptions import DocumentNotFoundError, RepositoryError
from ..models.document_model import DocumentModel, DocumentChunkModel


class DocumentRepositoryImpl(DocumentRepository):
    """
    SQLAlchemy implementation of DocumentRepository.
    
    Adapter that implements the domain repository interface using SQLAlchemy.
    Follows Dependency Inversion Principle by implementing the abstract interface.
    """
    
    def __init__(self, session: AsyncSession):
        """
        Initialize repository with database session.
        
        Args:
            session: Database session for operations
        """
        self._session = session
    
    async def save(self, document: Document) -> Document:
        """
        Save a document to the repository.
        
        Args:
            document: Document entity to save
            
        Returns:
            Saved document entity
            
        Raises:
            RepositoryError: If save operation fails
        """
        try:
            # Convert domain entity to database model
            document_model = self._domain_to_model(document)
            
            # Add to session and flush to get any generated values
            self._session.add(document_model)
            await self._session.flush()
            await self._session.refresh(document_model)
            
            # Convert back to domain entity
            return self._model_to_domain(document_model)
            
        except Exception as e:
            raise RepositoryError(f"Failed to save document: {str(e)}")
    
    async def find_by_id(self, document_id: DocumentId) -> Optional[Document]:
        """
        Find a document by its ID.
        
        Args:
            document_id: Unique document identifier
            
        Returns:
            Document entity if found, None otherwise
            
        Raises:
            RepositoryError: If query operation fails
        """
        try:
            stmt = (
                select(DocumentModel)
                .options(selectinload(DocumentModel.chunks))
                .where(DocumentModel.id == document_id.value)
            )
            
            result = await self._session.execute(stmt)
            document_model = result.scalar_one_or_none()
            
            if document_model is None:
                return None
            
            return self._model_to_domain(document_model)
            
        except Exception as e:
            raise RepositoryError(f"Failed to find document by ID: {str(e)}")
    
    async def get_by_id(self, document_id: DocumentId) -> Optional[Document]:
        """Alias for find_by_id to support advanced search use case."""
        return await self.find_by_id(document_id)
    
    async def find_by_status(self, status: DocumentStatus) -> List[Document]:
        """
        Find documents by their status.
        
        Args:
            status: Document status to filter by
            
        Returns:
            List of documents with the specified status
            
        Raises:
            RepositoryError: If query operation fails
        """
        try:
            stmt = (
                select(DocumentModel)
                .options(selectinload(DocumentModel.chunks))
                .where(DocumentModel.status == status)
                .order_by(DocumentModel.created_at.desc())
            )
            
            result = await self._session.execute(stmt)
            document_models = result.scalars().all()
            
            documents = []
            for model in document_models:
                domain_doc = self._model_to_domain(model)
                documents.append(domain_doc)
            
            return documents
            
        except Exception as e:
            raise RepositoryError(f"Failed to find documents by status: {str(e)}")
    
    async def find_all(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[Document]:
        """
        Find all documents with optional pagination.
        
        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            
        Returns:
            List of document entities
            
        Raises:
            RepositoryError: If query operation fails
        """
        try:
            stmt = (
                select(DocumentModel)
                .options(selectinload(DocumentModel.chunks))
                .order_by(DocumentModel.created_at.desc())
            )
            
            if offset:
                stmt = stmt.offset(offset)
            if limit:
                stmt = stmt.limit(limit)
            
            result = await self._session.execute(stmt)
            document_models = result.scalars().all()
            
            documents = []
            for model in document_models:
                domain_doc = self._model_to_domain(model)
                documents.append(domain_doc)
            
            return documents
            
        except Exception as e:
            raise RepositoryError(f"Failed to find all documents: {str(e)}")
    
    async def update(self, document: Document) -> Document:
        """
        Update an existing document.
        
        Args:
            document: Document entity with updated data
            
        Returns:
            Updated document entity
            
        Raises:
            RepositoryError: If update operation fails
            DocumentNotFoundError: If document doesn't exist
        """
        try:
            # Check if document exists
            existing = await self.find_by_id(document.document_id)
            if existing is None:
                raise DocumentNotFoundError(document.document_id.value)
            
            # PRODUCTION FIX: Handle chunks separately to avoid type casting issues
            # First, update document fields without chunks
            stmt = (
                update(DocumentModel)
                .where(DocumentModel.id == document.document_id.value)
                .values(
                    filename=document.file_metadata.filename,
                    file_size=document.file_metadata.file_size,
                    mime_type=document.file_metadata.mime_type,
                    file_path=document.file_path,
                    original_path=document.file_metadata.original_path,
                    status=document.status,
                    processing_stage=document.processing_stage,
                    page_count=document.page_count,
                    total_chunks=document.total_chunks,
                    processing_error=document.processing_error,
                    upload_timestamp=document.file_metadata.upload_timestamp,
                    updated_at=document.updated_at
                )
            )
            await self._session.execute(stmt)
            
            # Handle chunks separately to avoid merge conflicts
            if document.chunks:
                # Delete existing chunks for this document
                delete_chunks_stmt = delete(DocumentChunkModel).where(
                    DocumentChunkModel.document_id == document.document_id.value
                )
                await self._session.execute(delete_chunks_stmt)
                
                # Insert new chunks
                for chunk in document.chunks:
                    chunk_model = DocumentChunkModel(
                        id=chunk.chunk_id,
                        document_id=document.document_id.value,  # Ensure string type
                        page_number=chunk.page_number.value,
                        text_content=chunk.text_content,
                        start_position=chunk.start_position,
                        end_position=chunk.end_position,
                    )
                    self._session.add(chunk_model)
            
            await self._session.flush()
            
            # Return updated document
            updated_document = await self.find_by_id(document.document_id)
            return updated_document
            
        except DocumentNotFoundError:
            raise
        except Exception as e:
            # Rollback on error to prevent transaction issues
            await self._session.rollback()
            raise RepositoryError(f"Failed to update document: {str(e)}")
    
    async def delete(self, document_id: DocumentId) -> bool:
        """
        Delete a document by its ID.
        
        Args:
            document_id: Unique document identifier
            
        Returns:
            True if document was deleted, False if not found
            
        Raises:
            RepositoryError: If delete operation fails
        """
        try:
            # First delete all chunks associated with the document
            chunk_stmt = delete(DocumentChunkModel).where(
                DocumentChunkModel.document_id == document_id.value
            )
            await self._session.execute(chunk_stmt)
            
            # Then delete the document
            doc_stmt = delete(DocumentModel).where(DocumentModel.id == document_id.value)
            result = await self._session.execute(doc_stmt)
            
            # Commit the transaction
            await self._session.commit()
            
            return result.rowcount > 0
            
        except Exception as e:
            await self._session.rollback()
            raise RepositoryError(f"Failed to delete document: {str(e)}")
    
    async def exists(self, document_id: DocumentId) -> bool:
        """
        Check if a document exists.
        
        Args:
            document_id: Unique document identifier
            
        Returns:
            True if document exists, False otherwise
            
        Raises:
            RepositoryError: If query operation fails
        """
        try:
            stmt = select(func.count(DocumentModel.id)).where(
                DocumentModel.id == document_id.value
            )
            result = await self._session.execute(stmt)
            count = result.scalar()
            
            return count > 0
            
        except Exception as e:
            raise RepositoryError(f"Failed to check document existence: {str(e)}")
    
    async def count(self) -> int:
        """
        Get total count of documents.
        
        Returns:
            Total number of documents
            
        Raises:
            RepositoryError: If count operation fails
        """
        try:
            stmt = select(func.count(DocumentModel.id))
            result = await self._session.execute(stmt)
            
            return result.scalar()
            
        except Exception as e:
            raise RepositoryError(f"Failed to count documents: {str(e)}")
    
    async def find_processing_documents(self) -> List[Document]:
        """
        Find documents that are currently being processed.
        
        Returns:
            List of documents in processing status
            
        Raises:
            RepositoryError: If query operation fails
        """
        return await self.find_by_status(DocumentStatus.PROCESSING)
    
    def _domain_to_model(self, document: Document) -> DocumentModel:
        """
        Convert domain Document entity to database model.
        
        Args:
            document: Domain document entity
            
        Returns:
            Database document model
        """
        model = DocumentModel(
            id=document.document_id.value,
            filename=document.file_metadata.filename,
            file_size=document.file_metadata.file_size,
            mime_type=document.file_metadata.mime_type,
            file_path=document.file_path,
            original_path=document.file_metadata.original_path,
            status=document.status,
            processing_stage=document.processing_stage,
            page_count=document.page_count,
            total_chunks=document.total_chunks,
            processing_error=document.processing_error,
            upload_timestamp=document.file_metadata.upload_timestamp,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )
        
        # Convert chunks
        for chunk in document.chunks:
            chunk_model = DocumentChunkModel(
                id=chunk.chunk_id,
                document_id=document.document_id.value,
                page_number=chunk.page_number.value,
                text_content=chunk.text_content,
                start_position=chunk.start_position,
                end_position=chunk.end_position,
            )
            model.chunks.append(chunk_model)
        
        return model
    
    def _model_to_domain(self, model: DocumentModel) -> Document:
        """
        Convert database model to domain Document entity.
        
        Args:
            model: Database document model
            
        Returns:
            Domain document entity
        """
        # Create file metadata
        file_metadata = FileMetadata(
            filename=model.filename,
            file_size=model.file_size,
            mime_type=model.mime_type,
            upload_timestamp=model.upload_timestamp,
            original_path=model.original_path,
        )
        
        # Create document entity
        document = Document(
            document_id=DocumentId(model.id),
            file_metadata=file_metadata,
            status=model.status,
            processing_stage=model.processing_stage,
            created_at=model.created_at,
            updated_at=model.updated_at,
            page_count=model.page_count,
            total_chunks=model.total_chunks,
            processing_error=model.processing_error,
            file_path=model.file_path,
        )
        
        # Convert chunks only if they are already loaded (avoid lazy loading)
        # This prevents async context errors when chunks relationship is not eagerly loaded
        try:
            # Check if chunks are loaded without triggering lazy loading
            if hasattr(model, '__dict__') and 'chunks' in model.__dict__:
                for chunk_model in model.chunks:
                    chunk = DocumentChunk(
                        chunk_id=chunk_model.id,
                        page_number=PageNumber(chunk_model.page_number),
                        text_content=chunk_model.text_content,
                        start_position=chunk_model.start_position,
                        end_position=chunk_model.end_position,
                    )
                    document.chunks.append(chunk)
        except Exception:
            # If chunks loading fails, continue without chunks
            # This is acceptable for newly created documents
            pass
        
        return document
