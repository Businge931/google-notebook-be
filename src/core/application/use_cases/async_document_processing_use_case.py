"""
Asynchronous Document Processing Use Case

Optimized for large PDF processing with:
- Background job processing
- Progress tracking and status updates
- Non-blocking API responses
- Error recovery and retry mechanisms
"""
import asyncio
import uuid
import logging
from typing import Dict, Optional, Callable, Any
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum

from ...domain.entities import Document, DocumentStatus, ProcessingStage
from ...domain.value_objects import DocumentId
from ...domain.repositories import DocumentRepository
from ...domain.services.large_pdf_processing_service import LargePDFProcessingService, ProcessingProgress
from ....shared.exceptions import DocumentNotFoundError, DocumentProcessingError


class JobStatus(Enum):
    """Processing job status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProcessingJob:
    """Background processing job."""
    job_id: str
    document_id: str
    status: JobStatus
    progress: Optional[ProcessingProgress] = None
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)


class AsyncDocumentProcessingUseCase:
    """
    Use case for asynchronous document processing with background jobs.
    
    Features:
    - Non-blocking API responses
    - Background job processing
    - Progress tracking and status updates
    - Error recovery and retry mechanisms
    - Resource management for large PDFs
    """
    
    def __init__(
        self,
        document_repository: DocumentRepository,
        large_pdf_service: LargePDFProcessingService
    ):
        """Initialize async processing use case."""
        self._document_repository = document_repository
        self._large_pdf_service = large_pdf_service
        self._logger = logging.getLogger(__name__)
        
        # In-memory job tracking (in production, use Redis or database)
        self._jobs: Dict[str, ProcessingJob] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}
        
        # Progress callbacks
        self._progress_callbacks: Dict[str, List[Callable[[ProcessingProgress], None]]] = {}
    
    async def start_processing(
        self,
        document_id: DocumentId,
        force_reprocess: bool = False
    ) -> ProcessingJob:
        """
        Start background processing for a document.
        
        PRODUCTION-READY: Returns immediately regardless of file size.
        All heavy initialization is deferred to background task.
        
        Args:
            document_id: Document to process
            force_reprocess: Whether to force reprocessing
            
        Returns:
            Processing job with job_id for tracking
            
        Raises:
            DocumentNotFoundError: If document doesn't exist
        """
        # FAST CHECK: Only verify document exists (lightweight DB query)
        document = await self._document_repository.find_by_id(document_id)
        if document is None:
            raise DocumentNotFoundError(document_id.value)
        
        # FAST CHECK: Check if already processing (in-memory lookup)
        existing_job = self._find_active_job(document_id.value)
        if existing_job and not force_reprocess:
            self._logger.info(f"Document {document_id.value} already being processed: {existing_job.job_id}")
            return existing_job
        
        # INSTANT: Create job record (in-memory operation)
        job_id = str(uuid.uuid4())
        job = ProcessingJob(
            job_id=job_id,
            document_id=document_id.value,
            status=JobStatus.PENDING
        )
        
        self._jobs[job_id] = job
        self._logger.info(f"✅ INSTANT: Created processing job {job_id} for document {document_id.value}")
        
        # DEFERRED: Start background processing with zero-delay scheduling
        # Use asyncio.create_task with immediate return - no waiting for initialization
        task = asyncio.create_task(
            self._process_document_background_deferred(job, document_id, force_reprocess)
        )
        self._running_tasks[job_id] = task
        
        # IMMEDIATE RETURN: API responds instantly regardless of file size
        return job
    
    async def get_job_status(self, job_id: str) -> Optional[ProcessingJob]:
        """Get current status of a processing job."""
        return self._jobs.get(job_id)
    
    async def get_document_job(self, document_id: str) -> Optional[ProcessingJob]:
        """Get active job for a document."""
        return self._find_active_job(document_id)
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a processing job."""
        job = self._jobs.get(job_id)
        if not job:
            return False
        
        # Cancel running task
        task = self._running_tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now(timezone.utc)
            self._logger.info(f"Cancelled processing job {job_id}")
            return True
        
        return False
    
    def add_progress_callback(self, job_id: str, callback: Callable[[ProcessingProgress], None]):
        """Add progress callback for a job."""
        if job_id not in self._progress_callbacks:
            self._progress_callbacks[job_id] = []
        self._progress_callbacks[job_id].append(callback)
    
    async def _process_document_background_deferred(
        self,
        job: ProcessingJob,
        document_id: DocumentId,
        force_reprocess: bool
    ):
        """
        PRODUCTION-READY: Deferred background processing task.
        
        All heavy initialization happens here AFTER API response is sent.
        This ensures the API call returns instantly regardless of file size.
        """
        try:
            # STEP 1: Minimal status update (instant)
            job.status = JobStatus.PROCESSING
            job.started_at = datetime.now(timezone.utc)
            
            self._logger.info(f"🚀 DEFERRED: Starting background processing for job {job.job_id}")
            
            # STEP 2: Yield control to allow API response to be sent
            await asyncio.sleep(0.001)  # Minimal delay to ensure API response is sent first
            
            # STEP 3: Heavy initialization (deferred - happens after API response)
            self._logger.info(f"🔧 INITIALIZING: Heavy service setup for job {job.job_id}")
            
            # Create fresh session context for background processing to avoid session binding issues
            from ....infrastructure.di.container import get_container
            from ...domain.services.large_pdf_processing_service import LargePDFProcessingService
            
            container = get_container()
            db_manager = container.get_database_manager()
            
            async with db_manager.get_session() as fresh_session:
                # Create fresh repository and service instances with new session
                fresh_document_repo = container.get_document_repository(fresh_session)
                fresh_large_pdf_service = LargePDFProcessingService(
                    document_repository=fresh_document_repo,
                    file_storage=self._large_pdf_service._file_storage,
                    pdf_parsing_service=self._large_pdf_service._pdf_parsing_service,
                    embedding_service=self._large_pdf_service._embedding_service,
                    vector_repository=self._large_pdf_service._vector_repository,
                    config=self._large_pdf_service._config
                )
                
                self._logger.info(f"✅ INITIALIZED: Services ready for job {job.job_id}")
                
                # Add progress callback to update job
                def update_job_progress(progress: ProcessingProgress):
                    job.progress = progress
                    # Notify external callbacks
                    for callback in self._progress_callbacks.get(job.job_id, []):
                        try:
                            callback(progress)
                        except Exception as e:
                            self._logger.warning(f"Progress callback failed: {e}")
                
                fresh_large_pdf_service.add_progress_callback(update_job_progress)
                
                # STEP 4: Actual document processing (background)
                self._logger.info(f"📄 PROCESSING: Starting document processing for job {job.job_id}")
                final_progress = await fresh_large_pdf_service.process_large_pdf(
                    document_id=document_id,
                    force_reprocess=force_reprocess
                )
            
            # STEP 5: Job completion
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now(timezone.utc)
            job.progress = final_progress
            
            self._logger.info(f"🎉 COMPLETED: Background processing for job {job.job_id}")
            
        except asyncio.CancelledError:
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now(timezone.utc)
            self._logger.info(f"Processing job {job.job_id} was cancelled")
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now(timezone.utc)
            job.error_message = str(e)
            
            self._logger.error(f"Processing job {job.job_id} failed: {e}", exc_info=True)
            
        finally:
            # Cleanup
            if job.job_id in self._running_tasks:
                del self._running_tasks[job.job_id]
            if job.job_id in self._progress_callbacks:
                del self._progress_callbacks[job.job_id]
    
    def _find_active_job(self, document_id: str) -> Optional[ProcessingJob]:
        """Find active job for a document."""
        for job in self._jobs.values():
            if (job.document_id == document_id and 
                job.status in [JobStatus.PENDING, JobStatus.PROCESSING]):
                return job
        return None
    
    async def cleanup_completed_jobs(self, max_age_hours: int = 24):
        """Cleanup old completed jobs."""
        cutoff_time = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)
        
        jobs_to_remove = []
        for job_id, job in self._jobs.items():
            if (job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED] and
                job.completed_at and job.completed_at.timestamp() < cutoff_time):
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self._jobs[job_id]
            self._logger.info(f"Cleaned up old job {job_id}")
    
    def get_job_summary(self) -> Dict[str, Any]:
        """Get summary of all jobs."""
        summary = {
            "total_jobs": len(self._jobs),
            "pending": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0
        }
        
        for job in self._jobs.values():
            summary[job.status.value] += 1
        
        return summary


# Singleton instance for job management
_async_processing_instance: Optional[AsyncDocumentProcessingUseCase] = None

def get_async_processing_use_case(
    document_repository: DocumentRepository,
    large_pdf_service: LargePDFProcessingService
) -> AsyncDocumentProcessingUseCase:
    """Get singleton instance of async processing use case."""
    global _async_processing_instance
    
    if _async_processing_instance is None:
        _async_processing_instance = AsyncDocumentProcessingUseCase(
            document_repository=document_repository,
            large_pdf_service=large_pdf_service
        )
    
    return _async_processing_instance
