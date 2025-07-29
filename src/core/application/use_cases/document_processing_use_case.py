"""
Document Processing Use Case
"""
from typing import List, Optional
from datetime import datetime
import logging

from ...domain.entities import Document, DocumentStatus, ProcessingStage, DocumentChunk
from ...domain.repositories import DocumentRepository
from ...domain.services import DocumentProcessingService
from ...domain.services.chunk_storage_service import ProductionChunkStorageService
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
import gc
import asyncio
from ...domain.value_objects import DocumentId
from ....infrastructure.adapters.services.pdf_parsing_service import PDFParsingService
from ....infrastructure.adapters.storage.file_storage import FileStorage

# Import AI services from core domain interfaces instead of specific implementations
from ...domain.services.embedding_service import EmbeddingService
from ...domain.repositories import VectorRepository
from ....shared.exceptions import (
    DocumentNotFoundError,
    DocumentProcessingError,
    RepositoryError,
    FileStorageError
)
from src.shared.constants import ProcessingConstants


class LargePDFConfig:
    """Configuration for large PDF processing optimization."""
    def __init__(self):
        self.max_memory_usage_mb = 512
        self.chunk_batch_size = 8
        self.page_batch_size = 3
        self.enable_gc_after_batch = True
        self.max_concurrent_embeddings = 2
        self.enable_memory_monitoring = True
        self.memory_cleanup_threshold_mb = 400


class DocumentProcessingRequest:
    """
    Request object for document processing following Single Responsibility Principle.
    """
    
    def __init__(
        self,
        document_id: DocumentId,
        force_reprocess: bool = False,
        chunk_size: Optional[int] = None
    ):
        """
        Initialize processing request.
        
        Args:
            document_id: Unique document identifier
            force_reprocess: Whether to force reprocessing even if already processed
            chunk_size: Custom chunk size for text splitting
        """
        self.document_id = document_id
        self.force_reprocess = force_reprocess
        self.chunk_size = chunk_size or ProcessingConstants.DEFAULT_CHUNK_SIZE


class DocumentProcessingResponse:
    """
    Response object for document processing following Single Responsibility Principle.
    """
    
    def __init__(
        self,
        document_id: DocumentId,
        status: DocumentStatus,
        processing_stage: ProcessingStage,
        page_count: Optional[int] = None,
        chunk_count: Optional[int] = None,
        processing_time_ms: Optional[int] = None,
        error_message: Optional[str] = None
    ):
        """
        Initialize processing response.
        
        Args:
            document_id: Unique document identifier
            status: Current document status
            processing_stage: Current processing stage
            page_count: Number of pages processed
            chunk_count: Number of chunks created
            processing_time_ms: Processing time in milliseconds
            error_message: Error message if processing failed
        """
        self.document_id = document_id
        self.status = status
        self.processing_stage = processing_stage
        self.page_count = page_count
        self.chunk_count = chunk_count
        self.processing_time_ms = processing_time_ms
        self.error_message = error_message


class DocumentProcessingUseCase:
    """
    Use case for processing documents following Single Responsibility Principle.
    
    Orchestrates the document processing workflow including PDF parsing,
    text extraction, chunking, and status updates.
    """
    
    def __init__(
        self,
        document_repository: DocumentRepository,
        file_storage: FileStorage,
        pdf_parsing_service: PDFParsingService,
        document_processing_service: DocumentProcessingService,
        embedding_service: EmbeddingService,
        vector_repository: VectorRepository
    ):
        """
        Initialize use case with dependencies.
        
        Args:
            document_repository: Repository for document persistence
            file_storage: Service for file storage operations
            pdf_parsing_service: Service for PDF parsing
            document_processing_service: Service for document processing logic
            embedding_service: Service for generating embeddings
            vector_repository: Repository for storing and searching vectors
        """
        self._document_repository = document_repository
        self._file_storage = file_storage
        self._pdf_parsing_service = pdf_parsing_service
        self._document_processing_service = document_processing_service
        self._embedding_service = embedding_service
        self._vector_repository = vector_repository
        self._logger = logging.getLogger(__name__)
        
        # Initialize production chunk storage service
        self._chunk_storage_service = ProductionChunkStorageService(
            document_repository=document_repository,
            vector_repository=vector_repository
        )
        
        # Large PDF optimization configuration
        self._large_file_threshold_mb = 10  # Files > 10MB use optimized processing
        self._large_pdf_config = LargePDFConfig()
    
    async def execute(self, request: DocumentProcessingRequest) -> DocumentProcessingResponse:
        """
        Execute document processing use case.
        
        Args:
            request: Document processing request
            
        Returns:
            Document processing response
            
        Raises:
            DocumentNotFoundError: If document doesn't exist
            DocumentProcessingError: If processing fails
            RepositoryError: If database operation fails
        """
        start_time = datetime.utcnow()
        
        try:
            # Step 1: Retrieve document
            self._logger.info(f"STEP 1: Retrieving document {request.document_id.value}")
            document = await self._document_repository.find_by_id(request.document_id)
            if document is None:
                raise DocumentNotFoundError(request.document_id.value)
            
            self._logger.info(f"Document found: status={document.status.value}, stage={document.processing_stage.value}")
            
            # Step 2: Check if processing is needed
            should_process = self._should_process_document(document, request.force_reprocess)
            self._logger.info(f"STEP 2: Should process document? {should_process} (force_reprocess={request.force_reprocess})")
            
            if not should_process:
                self._logger.info(f"EARLY RETURN: Document processing skipped")
                return DocumentProcessingResponse(
                    document_id=document.document_id,
                    status=document.status,
                    processing_stage=document.processing_stage,
                    page_count=document.page_count,
                    chunk_count=document.total_chunks
                )
            
            # Step 3: Update document status to processing
            self._logger.info(f"STEP 3: Starting document processing")
            
            # Handle force reprocessing by resetting status if needed
            if request.force_reprocess and document.status == DocumentStatus.PROCESSED:
                self._logger.info(f"Force reprocessing: resetting document status from PROCESSED to UPLOADED")
                document.status = DocumentStatus.UPLOADED
                document.processing_stage = ProcessingStage.UPLOAD_COMPLETE
            
            document.start_processing()
            await self._document_repository.update(document)
            
            # Step 4: Retrieve file from storage and detect large PDF
            file_data = await self._file_storage.get_file(document.file_path)
            file_size_mb = document.file_metadata.file_size / (1024 * 1024) if document.file_metadata.file_size else 0
            is_large_pdf = file_size_mb > self._large_file_threshold_mb
            
            self._logger.info(f"File size: {file_size_mb:.1f}MB - Large PDF optimization: {is_large_pdf}")
            
            # Step 5: Parse PDF document with optimization for large files
            self._logger.info(f"STEP 5: Starting PDF parsing for document {document.document_id.value}")
            
            if is_large_pdf:
                # Use memory-efficient parsing for large PDFs
                await self._ensure_memory_available()
            
            parsed_document = await self._pdf_parsing_service.parse_document(
                file_data,
                document.file_metadata.mime_type
            )
            
            # Step 6: Create document chunks with optimization for large PDFs
            self._logger.info(f"Creating chunks for document {document.document_id.value}")
            
            if is_large_pdf:
                # Use optimized chunking for large PDFs
                chunks = await self._create_chunks_optimized(parsed_document, request.chunk_size)
            else:
                # Use standard chunking for smaller PDFs
                chunks = await self._pdf_parsing_service.create_document_chunks(
                    parsed_document,
                    request.chunk_size
                )
            
            # Step 7: Validate processing results using domain service
            self._document_processing_service.validate_processing_results(
                document,
                parsed_document.total_pages,
                len(chunks)
            )
            
            # Step 8: Update document with processing results
            self._logger.info(f"Completing parsing for document {document.document_id.value}")
            document.complete_parsing(
                page_count=parsed_document.total_pages,
                chunks=chunks
            )
            self._logger.info(f"Document stage after complete_parsing: {document.processing_stage.value}")
            
            # Step 9: Advance to vectorization stage
            self._logger.info(f"Advancing document {document.document_id.value} to VECTORIZATION stage")
            document.advance_processing_stage(ProcessingStage.VECTORIZATION)
            self._logger.info(f"Document stage after vectorization: {document.processing_stage.value}")
            
            # Step 10: Perform vectorization (create embeddings for chunks)
            self._logger.error(f"🔥 VECTORIZATION STEP REACHED for document {document.document_id.value}")
            self._logger.info(f"Starting vectorization for document {document.document_id.value}")
            self._logger.info(f"Embedding service type: {type(self._embedding_service)}")
            self._logger.info(f"Vector repository type: {type(self._vector_repository)}")
            self._logger.info(f"Number of chunks to vectorize: {len(chunks)}")
            
            try:
                # Extract text from chunks for embedding
                chunk_texts = [chunk.text_content for chunk in chunks]
                chunk_ids = [f"{document.document_id.value}_{i}" for i in range(len(chunks))]
                
                self._logger.info(f"Extracted {len(chunk_texts)} chunk texts for embedding")
                for i, text in enumerate(chunk_texts[:2]):  # Log first 2 chunks
                    self._logger.info(f"Chunk {i} text preview: {text[:100]}...")
                
                # Generate embeddings for all chunks
                self._logger.info(f"Calling embedding service to generate embeddings...")
                embedding_vectors = await self._embedding_service.generate_embeddings_batch(chunk_texts)
                embeddings = [ev.vector for ev in embedding_vectors]
                self._logger.info(f"Generated {len(embeddings)} embeddings for document {document.document_id.value}")
                
                # Log embedding details
                if embeddings:
                    self._logger.info(f"First embedding vector length: {len(embeddings[0])}")
                    self._logger.info(f"First embedding vector preview: {embeddings[0][:5]}...")
                
            except Exception as e:
                self._logger.error(f"Vectorization failed for document {document.document_id.value}: {str(e)}")
                raise DocumentProcessingError(f"Vectorization failed: {str(e)}")
            
            # Step 11: Advance to indexing stage
            self._logger.info(f"Advancing document {document.document_id.value} to INDEXING stage")
            document.advance_processing_stage(ProcessingStage.INDEXING)
            self._logger.info(f"Document stage after indexing: {document.processing_stage.value}")
            
            # Step 12: Production-grade chunk storage with validation and error recovery
            self._logger.info(f"Starting production chunk storage for document {document.document_id.value}")
            self._logger.info(f"About to store {len(embeddings)} chunks with full validation")
            
            try:
                # Create chunks data for vector storage
                chunks_data = []
                for i, chunk in enumerate(chunks):
                    # Extract raw vector data from EmbeddingVector
                    raw_vector = embeddings[i].vector if hasattr(embeddings[i], 'vector') else embeddings[i]
                    
                    chunk_data = {
                        'chunk_id': chunk_ids[i],
                        'text_content': chunk.text_content,
                        'page_number': chunk.page_number,
                        'start_position': chunk.start_position,
                        'end_position': chunk.end_position,
                        'vector': raw_vector,
                        'metadata': {
                            'document_id': document.document_id.value,
                            'content_preview': chunk.text_content[:200]
                        }
                    }
                    chunks_data.append(chunk_data)
                
                self._logger.info(f"Created {len(chunks_data)} chunk data objects for production storage")
                
                # Use production-grade chunk storage service with validation and retry
                validation_result = await self._chunk_storage_service.store_chunks_with_validation(
                    document=document,
                    chunks=chunks,
                    vectors_data=chunks_data,
                    max_retries=3
                )
                
                if not validation_result.is_valid:
                    raise DocumentProcessingError(
                        f"Chunk storage validation failed: DB chunks: {validation_result.database_chunk_count}, "
                        f"Vector chunks: {validation_result.vector_chunk_count}, "
                        f"Page errors: {len(validation_result.page_validation_errors)}"
                    )
                
                self._logger.info(
                    f"✅ Successfully stored and validated {validation_result.database_chunk_count} chunks "
                    f"for document {document.document_id.value} (DB: {validation_result.database_chunk_count}, "
                    f"Vector: {validation_result.vector_chunk_count})"
                )
                
            except Exception as e:
                self._logger.error(f"Production chunk storage failed for document {document.document_id.value}: {str(e)}")
                raise DocumentProcessingError(f"Chunk storage failed: {str(e)}")
            
            # Step 13: Complete processing (advance to final stage)
            self._logger.info(f"Advancing document {document.document_id.value} to COMPLETE stage")
            document.advance_processing_stage(ProcessingStage.COMPLETE)
            self._logger.info(f"Document stage after advance_processing_stage: {document.processing_stage.value}")
            self._logger.info(f"Document status after advance_processing_stage: {document.status.value}")
            
            # Step 10: Save updated document
            self._logger.info(f"Saving updated document {document.document_id.value}")
            updated_document = await self._document_repository.update(document)
            self._logger.info(f"Document saved with status: {updated_document.status.value}, stage: {updated_document.processing_stage.value}")
            
            # Step 10: Calculate processing time
            end_time = datetime.utcnow()
            processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            self._logger.info(
                f"Successfully processed document {document.document_id.value} "
                f"in {processing_time_ms}ms"
            )
            
            return DocumentProcessingResponse(
                document_id=updated_document.document_id,
                status=updated_document.status,
                processing_stage=updated_document.processing_stage,
                page_count=updated_document.page_count,
                chunk_count=updated_document.total_chunks,
                processing_time_ms=processing_time_ms
            )
            
        except (DocumentNotFoundError, DocumentProcessingError, RepositoryError):
            # Re-raise domain exceptions
            raise
        except Exception as e:
            # Handle unexpected errors
            error_message = f"Unexpected error during processing: {str(e)}"
            self._logger.error(error_message, exc_info=True)
            
            # Try to update document status to failed
            try:
                if 'document' in locals():
                    document.mark_processing_failed(error_message)
                    await self._document_repository.update(document)
            except Exception as update_error:
                self._logger.error(f"Failed to update document status: {update_error}")
            
            # Calculate processing time
            end_time = datetime.utcnow()
            processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            return DocumentProcessingResponse(
                document_id=request.document_id,
                status=DocumentStatus.FAILED,
                processing_stage=ProcessingStage.PARSING_FAILED,
                processing_time_ms=processing_time_ms,
                error_message=error_message
            )
    
    def _should_process_document(self, document: Document, force_reprocess: bool) -> bool:
        """
        Determine if document should be processed.
        
        Args:
            document: Document entity
            force_reprocess: Whether to force reprocessing
            
        Returns:
            True if document should be processed
        """
        if force_reprocess:
            return True
            
        # Process if not yet processed
        if document.status != DocumentStatus.PROCESSED:
            return True
            
        # Process if processing stage is incomplete
        if document.processing_stage != ProcessingStage.COMPLETE:
            return True
            
        return False
    
    async def _ensure_memory_available(self):
        """Ensure sufficient memory is available for large PDF processing."""
        try:
            memory_mb = self._get_memory_usage()
            if memory_mb > self._large_pdf_config.memory_cleanup_threshold_mb:
                self._logger.info(f"Memory usage {memory_mb:.1f}MB exceeds threshold, performing cleanup")
                await self._force_memory_cleanup()
        except Exception as e:
            self._logger.warning(f"Memory check failed: {e}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if not PSUTIL_AVAILABLE:
            return 0.0
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    async def _force_memory_cleanup(self):
        """Force garbage collection and memory cleanup."""
        gc.collect()
        await asyncio.sleep(0.1)  # Allow cleanup to complete
    
    async def _create_chunks_optimized(self, parsed_document, chunk_size: int) -> List[DocumentChunk]:
        """Create chunks with optimization for large PDFs."""
        # Use the standard chunking method but with optimized settings
        optimized_chunk_size = min(chunk_size, 1000)  # Smaller chunks for large documents
        
        self._logger.info(f"Creating optimized chunks with size: {optimized_chunk_size}")
        
        # Use the existing create_document_chunks method
        chunks = await self._pdf_parsing_service.create_document_chunks(
            parsed_document, optimized_chunk_size
        )
        
        # Memory cleanup after chunking
        if self._large_pdf_config.enable_gc_after_batch:
            await self._force_memory_cleanup()
        
        self._logger.info(f"Created {len(chunks)} optimized chunks for large PDF")
        return chunks


class BulkDocumentProcessingUseCase:
    """
    Use case for processing multiple documents in bulk.
    
    Follows Single Responsibility Principle by focusing on bulk operations.
    """
    
    def __init__(
        self,
        document_repository: DocumentRepository,
        document_processing_use_case: DocumentProcessingUseCase
    ):
        """
        Initialize bulk processing use case.
        
        Args:
            document_repository: Repository for document persistence
            document_processing_use_case: Single document processing use case
        """
        self._document_repository = document_repository
        self._document_processing_use_case = document_processing_use_case
        self._logger = logging.getLogger(__name__)
    
    async def execute(
        self,
        status_filter: Optional[DocumentStatus] = None,
        limit: Optional[int] = None,
        force_reprocess: bool = False
    ) -> List[DocumentProcessingResponse]:
        """
        Execute bulk document processing.
        
        Args:
            status_filter: Only process documents with this status
            limit: Maximum number of documents to process
            force_reprocess: Whether to force reprocessing
            
        Returns:
            List of processing responses
        """
        try:
            # Find documents to process
            if status_filter:
                documents = await self._document_repository.find_by_status(status_filter)
            else:
                documents = await self._document_repository.find_all(limit=limit)
            
            if limit and len(documents) > limit:
                documents = documents[:limit]
            
            self._logger.info(f"Starting bulk processing of {len(documents)} documents")
            
            # Process each document
            results = []
            for document in documents:
                try:
                    request = DocumentProcessingRequest(
                        document_id=document.document_id,
                        force_reprocess=force_reprocess
                    )
                    
                    response = await self._document_processing_use_case.execute(request)
                    results.append(response)
                    
                except Exception as e:
                    self._logger.error(
                        f"Failed to process document {document.document_id.value}: {e}"
                    )
                    
                    # Add error response
                    results.append(DocumentProcessingResponse(
                        document_id=document.document_id,
                        status=DocumentStatus.FAILED,
                        processing_stage=ProcessingStage.PARSING_FAILED,
                        error_message=str(e)
                    ))
            
            self._logger.info(f"Completed bulk processing of {len(results)} documents")
            return results
            
        except Exception as e:
            self._logger.error(f"Bulk processing failed: {e}", exc_info=True)
            raise DocumentProcessingError(f"Bulk processing failed: {str(e)}")


class DocumentProcessingUseCaseFactory:
    """
    Factory for creating document processing use case instances.
    
    Follows Open/Closed Principle for extensibility.
    """
    
    @staticmethod
    def create(
        document_repository: DocumentRepository,
        file_storage: FileStorage,
        pdf_parsing_service: PDFParsingService,
        document_processing_service: DocumentProcessingService
    ) -> DocumentProcessingUseCase:
        """
        Create document processing use case instance.
        
        Args:
            document_repository: Repository for document persistence
            file_storage: Service for file storage operations
            pdf_parsing_service: Service for PDF parsing
            document_processing_service: Service for document processing logic
            
        Returns:
            Configured document processing use case
        """
        return DocumentProcessingUseCase(
            document_repository=document_repository,
            file_storage=file_storage,
            pdf_parsing_service=pdf_parsing_service,
            document_processing_service=document_processing_service
        )
    
    @staticmethod
    def create_bulk(
        document_repository: DocumentRepository,
        document_processing_use_case: DocumentProcessingUseCase
    ) -> BulkDocumentProcessingUseCase:
        """
        Create bulk document processing use case instance.
        
        Args:
            document_repository: Repository for document persistence
            document_processing_use_case: Single document processing use case
            
        Returns:
            Configured bulk processing use case
        """
        return BulkDocumentProcessingUseCase(
            document_repository=document_repository,
            document_processing_use_case=document_processing_use_case
        )
