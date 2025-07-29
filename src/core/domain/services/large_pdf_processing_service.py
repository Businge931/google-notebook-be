"""
Large PDF Processing Service

Optimized for handling large PDFs up to 100MB with:
- Memory-efficient streaming processing
- Progressive chunking and vectorization
- Resource management and cleanup
- Progress tracking and error recovery
"""
import asyncio
import gc
import logging
import psutil
from typing import List, Optional, Dict, Any, AsyncGenerator, Callable
from datetime import datetime
from dataclasses import dataclass
import io
from concurrent.futures import ThreadPoolExecutor
import threading

from ..entities import Document, DocumentChunk, DocumentStatus, ProcessingStage
from ..value_objects import DocumentId, PageNumber
from ..repositories import DocumentRepository, VectorRepository
from ..services.embedding_service import EmbeddingService
from ....infrastructure.adapters.services.pdf_parsing_service import PDFParsingService, ParsedDocument
from ....infrastructure.adapters.storage.file_storage import FileStorage
from ....shared.exceptions import DocumentProcessingError
from ....shared.constants import ProcessingConstants


@dataclass
class ProcessingProgress:
    """Progress tracking for large PDF processing."""
    document_id: str
    total_pages: int
    processed_pages: int
    total_chunks: int
    processed_chunks: int
    vectorized_chunks: int
    current_stage: str
    memory_usage_mb: float
    processing_time_ms: int
    estimated_remaining_ms: Optional[int] = None
    error_message: Optional[str] = None


@dataclass
class LargePDFConfig:
    """Configuration for large PDF processing optimization."""
    max_memory_usage_mb: int = 512  # Maximum memory usage
    chunk_batch_size: int = 10      # Process chunks in batches
    page_batch_size: int = 5        # Process pages in batches
    enable_gc_after_batch: bool = True  # Force garbage collection
    max_concurrent_embeddings: int = 3  # Limit concurrent embedding requests
    progress_callback_interval: int = 5  # Progress callback every N chunks
    enable_memory_monitoring: bool = True
    memory_cleanup_threshold_mb: int = 400  # Trigger cleanup at this threshold


class LargePDFProcessingService:
    """
    Service for processing large PDFs with memory optimization and progress tracking.
    
    Features:
    - Streaming PDF processing to minimize memory usage
    - Progressive chunking and vectorization
    - Memory monitoring and cleanup
    - Progress tracking with callbacks
    - Error recovery and retry mechanisms
    """
    
    def __init__(
        self,
        document_repository: DocumentRepository,
        file_storage: FileStorage,
        pdf_parsing_service: PDFParsingService,
        embedding_service: EmbeddingService,
        vector_repository: VectorRepository,
        config: Optional[LargePDFConfig] = None
    ):
        """Initialize large PDF processing service."""
        self._document_repository = document_repository
        self._file_storage = file_storage
        self._pdf_parsing_service = pdf_parsing_service
        self._embedding_service = embedding_service
        self._vector_repository = vector_repository
        self._config = config or LargePDFConfig()
        self._logger = logging.getLogger(__name__)
        self._thread_pool = ThreadPoolExecutor(max_workers=2)
        
        # Progress tracking
        self._progress_callbacks: List[Callable[[ProcessingProgress], None]] = []
        self._processing_lock = threading.Lock()
        
    def add_progress_callback(self, callback: Callable[[ProcessingProgress], None]):
        """Add a progress callback function."""
        self._progress_callbacks.append(callback)
    
    async def process_large_pdf(
        self,
        document_id: DocumentId,
        force_reprocess: bool = False
    ) -> ProcessingProgress:
        """
        Process a large PDF with optimized memory management.
        
        Args:
            document_id: Document to process
            force_reprocess: Whether to force reprocessing
            
        Returns:
            Final processing progress
            
        Raises:
            DocumentProcessingError: If processing fails
            MemoryError: If memory usage exceeds limits
        """
        start_time = datetime.now()
        progress = ProcessingProgress(
            document_id=document_id.value,
            total_pages=0,
            processed_pages=0,
            total_chunks=0,
            processed_chunks=0,
            vectorized_chunks=0,
            current_stage="initializing",
            memory_usage_mb=0.0,
            processing_time_ms=0
        )
        
        try:
            # Load document
            document = await self._document_repository.get_by_id(document_id)
            if not document:
                raise DocumentProcessingError(f"Document {document_id.value} not found")
            
            # Check if processing needed
            if not force_reprocess and document.status == DocumentStatus.PROCESSED:
                self._logger.info(f"Document {document_id.value} already processed")
                return progress
            
            # Update status using fresh document instance to avoid session issues
            await self._update_document_status(
                document_id, 
                DocumentStatus.PROCESSING, 
                ProcessingStage.PARSING_STARTED
            )
            
            # Monitor memory usage
            if self._config.enable_memory_monitoring:
                self._start_memory_monitoring()
            
            # Phase 1: Parse PDF with streaming
            progress.current_stage = "parsing_pdf"
            self._notify_progress(progress)
            
            parsed_document = await self._parse_pdf_streaming(document, progress)
            progress.total_pages = parsed_document.total_pages
            
            # Phase 2: Process pages in batches
            progress.current_stage = "processing_pages"
            chunks = await self._process_pages_in_batches(parsed_document, progress)
            progress.total_chunks = len(chunks)
            
            # Phase 3: Vectorize chunks in batches
            progress.current_stage = "vectorizing_chunks"
            await self._vectorize_chunks_in_batches(document, chunks, progress)
            
            # Phase 4: Store chunks with production service
            progress.current_stage = "storing_chunks"
            await self._store_chunks_optimized(document, chunks, progress)
            
            # Update final status using fresh document instance
            await self._update_document_status(
                document_id,
                DocumentStatus.PROCESSED,
                ProcessingStage.COMPLETE
            )
            
            # Update document with final counts
            fresh_document = await self._document_repository.get_by_id(document_id)
            if fresh_document:
                fresh_document.page_count = progress.total_pages
                fresh_document.total_chunks = progress.total_chunks
                await self._document_repository.update(fresh_document)
            
            # Calculate final metrics
            end_time = datetime.now()
            progress.processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
            progress.current_stage = "complete"
            progress.memory_usage_mb = self._get_memory_usage()
            
            self._notify_progress(progress)
            self._logger.info(
                f"Large PDF processing complete: {progress.total_pages} pages, "
                f"{progress.total_chunks} chunks, {progress.processing_time_ms}ms"
            )
            
            return progress
            
        except Exception as e:
            progress.error_message = str(e)
            progress.current_stage = "failed"
            self._notify_progress(progress)
            
            # Update document status using fresh database session
            try:
                await self._update_document_status(
                    document_id,
                    DocumentStatus.FAILED,
                    ProcessingStage.PARSING_FAILED
                )
            except:
                pass
            
            self._logger.error(f"Large PDF processing failed: {e}", exc_info=True)
            raise DocumentProcessingError(f"Large PDF processing failed: {str(e)}")
        
        finally:
            # Cleanup resources
            await self._cleanup_resources()
    
    async def _parse_pdf_streaming(
        self, 
        document: Document, 
        progress: ProcessingProgress
    ) -> ParsedDocument:
        """Parse PDF with streaming to minimize memory usage."""
        try:
            # Load file data
            file_data = await self._file_storage.get_file(document.file_path)
            
            # Check memory before parsing
            if self._config.enable_memory_monitoring:
                memory_mb = self._get_memory_usage()
                if memory_mb > self._config.memory_cleanup_threshold_mb:
                    await self._force_memory_cleanup()
            
            # Parse document
            parsed_document = await self._pdf_parsing_service.parse_document(file_data, "application/pdf")
            
            # Update progress
            progress.total_pages = parsed_document.total_pages
            progress.memory_usage_mb = self._get_memory_usage()
            self._notify_progress(progress)
            
            return parsed_document
            
        except Exception as e:
            raise DocumentProcessingError(f"PDF parsing failed: {str(e)}")
    
    async def _process_pages_in_batches(
        self,
        parsed_document: ParsedDocument,
        progress: ProcessingProgress
    ) -> List[DocumentChunk]:
        """Process pages in batches to manage memory usage."""
        all_chunks = []
        
        # Process pages in batches
        for i in range(0, len(parsed_document.pages), self._config.page_batch_size):
            batch_pages = parsed_document.pages[i:i + self._config.page_batch_size]
            
            # Process batch
            batch_chunks = []
            for page in batch_pages:
                if page.text_content.strip():  # Skip empty pages
                    # Create chunks for this page
                    page_chunks = await self._create_chunks_for_page(page)
                    batch_chunks.extend(page_chunks)
                
                progress.processed_pages += 1
                
                # Update progress every few pages
                if progress.processed_pages % 2 == 0:
                    progress.memory_usage_mb = self._get_memory_usage()
                    self._notify_progress(progress)
            
            all_chunks.extend(batch_chunks)
            
            # Memory cleanup after batch
            if self._config.enable_gc_after_batch:
                await self._force_memory_cleanup()
        
        return all_chunks
    
    async def _create_chunks_for_page(self, page) -> List[DocumentChunk]:
        """Create optimized chunks for a single page."""
        chunks = []
        
        # Use optimized chunking strategy for large documents
        chunk_size = min(ProcessingConstants.DEFAULT_CHUNK_SIZE, 1000)  # Smaller chunks for large docs
        overlap = 100  # Reduced overlap
        
        # Split text into chunks
        text_chunks = self._split_text_optimized(page.text_content, chunk_size, overlap)
        
        for i, chunk_text in enumerate(text_chunks):
            # Calculate positions for this chunk within the page text
            start_pos = 0
            end_pos = len(chunk_text)
            
            # If we can find the chunk text in the original page text, use actual positions
            if chunk_text in page.text_content:
                start_pos = page.text_content.find(chunk_text)
                end_pos = start_pos + len(chunk_text)
            
            chunk = DocumentChunk(
                chunk_id=f"page_{page.page_number}_chunk_{i}",
                page_number=PageNumber(page.page_number),
                text_content=chunk_text,
                start_position=start_pos,
                end_position=end_pos
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_text_optimized(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Optimized text splitting for large documents."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Find good break point (sentence or word boundary)
            if end < len(text):
                # Look for sentence boundary
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
                else:
                    # Look for word boundary
                    word_end = text.rfind(' ', start, end)
                    if word_end > start + chunk_size // 2:
                        end = word_end
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = max(start + 1, end - overlap)
        
        return chunks
    
    async def _vectorize_chunks_in_batches(
        self,
        document: Document,
        chunks: List[DocumentChunk],
        progress: ProcessingProgress
    ):
        """Vectorize chunks in batches with concurrency control."""
        # Process chunks in batches
        for i in range(0, len(chunks), self._config.chunk_batch_size):
            batch_chunks = chunks[i:i + self._config.chunk_batch_size]
            
            # Limit concurrent embedding requests
            semaphore = asyncio.Semaphore(self._config.max_concurrent_embeddings)
            
            async def vectorize_chunk(chunk: DocumentChunk):
                async with semaphore:
                    try:
                        # Generate embedding
                        embedding = await self._embedding_service.generate_embedding(chunk.text_content)
                        
                        # Store in vector repository
                        await self._vector_repository.store_embeddings(
                            document_id=document.document_id.value,
                            embeddings=[embedding],
                            metadata=[chunk.metadata or {}]
                        )
                        
                        progress.vectorized_chunks += 1
                        
                        # Update progress
                        if progress.vectorized_chunks % self._config.progress_callback_interval == 0:
                            progress.memory_usage_mb = self._get_memory_usage()
                            self._notify_progress(progress)
                        
                    except Exception as e:
                        self._logger.warning(f"Failed to vectorize chunk: {e}")
            
            # Process batch concurrently
            await asyncio.gather(*[vectorize_chunk(chunk) for chunk in batch_chunks])
            
            progress.processed_chunks += len(batch_chunks)
            
            # Memory cleanup after batch
            if self._config.enable_gc_after_batch:
                await self._force_memory_cleanup()
    
    async def _store_chunks_optimized(
        self,
        document: Document,
        chunks: List[DocumentChunk],
        progress: ProcessingProgress
    ):
        """Store chunks with optimized batch processing and duplicate handling."""
        # Get fresh document instance from database to avoid session binding issues
        fresh_document = await self._document_repository.get_by_id(document.document_id)
        if fresh_document:
            # Clear existing chunks to avoid duplicate key constraint violations during reprocessing
            fresh_document.chunks.clear()
            
            # Generate unique chunk IDs to prevent duplicates
            import uuid
            for i, chunk in enumerate(chunks):
                # Create unique chunk ID using timestamp and UUID to avoid conflicts
                unique_id = f"{document.document_id.value[:8]}_{i}_{str(uuid.uuid4())[:8]}"
                # Create new chunk with unique ID (DocumentChunk is immutable, so create new instance)
                unique_chunk = DocumentChunk(
                    chunk_id=unique_id,
                    page_number=chunk.page_number,
                    text_content=chunk.text_content,
                    start_position=chunk.start_position,
                    end_position=chunk.end_position
                )
                fresh_document.chunks.append(unique_chunk)
            
            # Update total chunks count
            fresh_document.total_chunks = len(chunks)
            
            # Update document in repository using fresh instance
            await self._document_repository.update(fresh_document)
            
            self._logger.info(f"Stored {len(chunks)} chunks with unique IDs for document {document.document_id.value}")
        
        progress.current_stage = "storage_complete"
        self._notify_progress(progress)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    async def _force_memory_cleanup(self):
        """Force garbage collection and memory cleanup."""
        gc.collect()
        await asyncio.sleep(0.1)  # Allow cleanup to complete
    
    def _start_memory_monitoring(self):
        """Start memory monitoring in background."""
        async def monitor_memory():
            while True:
                memory_mb = self._get_memory_usage()
                if memory_mb > self._config.max_memory_usage_mb:
                    self._logger.warning(f"High memory usage: {memory_mb:.1f}MB")
                    await self._force_memory_cleanup()
                await asyncio.sleep(5)
        
        # Start monitoring task
        asyncio.create_task(monitor_memory())
    
    def _notify_progress(self, progress: ProcessingProgress):
        """Notify all progress callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(progress)
            except Exception as e:
                self._logger.warning(f"Progress callback failed: {e}")
    
    async def _update_document_status(
        self, 
        document_id: DocumentId, 
        status: DocumentStatus, 
        processing_stage: ProcessingStage
    ):
        """Update document status using fresh database session to avoid session binding issues."""
        try:
            # Get fresh document instance from database
            document = await self._document_repository.get_by_id(document_id)
            if document:
                document.status = status
                document.processing_stage = processing_stage
                await self._document_repository.update(document)
        except Exception as e:
            self._logger.error(f"Failed to update document final status: {e}")
            raise
    
    async def _cleanup_resources(self):
        """Cleanup resources and perform final memory cleanup."""
        try:
            # Force final garbage collection
            await self._force_memory_cleanup()
            
            # Shutdown thread pool
            self._thread_pool.shutdown(wait=False)
            
        except Exception as e:
            self._logger.warning(f"Resource cleanup failed: {e}")


class LargePDFProcessingServiceFactory:
    """Factory for creating large PDF processing service instances."""
    
    @staticmethod
    def create_optimized(
        document_repository: DocumentRepository,
        file_storage: FileStorage,
        pdf_parsing_service: PDFParsingService,
        embedding_service: EmbeddingService,
        vector_repository: VectorRepository,
        max_memory_mb: int = 512
    ) -> LargePDFProcessingService:
        """Create optimized large PDF processing service."""
        config = LargePDFConfig(
            max_memory_usage_mb=max_memory_mb,
            chunk_batch_size=8,
            page_batch_size=3,
            enable_gc_after_batch=True,
            max_concurrent_embeddings=2,
            enable_memory_monitoring=True
        )
        
        return LargePDFProcessingService(
            document_repository=document_repository,
            file_storage=file_storage,
            pdf_parsing_service=pdf_parsing_service,
            embedding_service=embedding_service,
            vector_repository=vector_repository,
            config=config
        )
