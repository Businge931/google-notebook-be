"""
Production Chunk Storage Health Check and Repair Use Case

Provides comprehensive health monitoring, validation, and automated repair
for chunk storage inconsistencies in production environments.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from ...domain.entities import Document, DocumentStatus, ProcessingStage
from ...domain.repositories import DocumentRepository, VectorRepository
from ...domain.services.chunk_storage_service import (
    ProductionChunkStorageService,
    ChunkStorageValidationResult
)
from ...domain.value_objects import DocumentId
from ....shared.exceptions import RepositoryError


@dataclass
class HealthCheckResult:
    """Result of chunk storage health check."""
    total_documents: int
    healthy_documents: int
    inconsistent_documents: int
    no_chunks_documents: int
    page_overflow_documents: int
    issues_found: List[Dict[str, Any]]
    repair_recommendations: List[str]


@dataclass
class RepairResult:
    """Result of chunk storage repair operation."""
    document_id: str
    repair_attempted: bool
    repair_successful: bool
    error_message: Optional[str] = None
    chunks_before: int = 0
    chunks_after: int = 0


class ChunkStorageHealthUseCase:
    """
    Production use case for chunk storage health monitoring and repair.
    
    Provides comprehensive health checks, automated repair, and monitoring
    for chunk storage consistency across vector database and PostgreSQL.
    """
    
    def __init__(
        self,
        document_repository: DocumentRepository,
        vector_repository: VectorRepository,
        chunk_storage_service: ProductionChunkStorageService
    ):
        self._document_repository = document_repository
        self._vector_repository = vector_repository
        self._chunk_storage_service = chunk_storage_service
        self._logger = logging.getLogger(__name__)
    
    async def perform_comprehensive_health_check(
        self,
        include_repair_recommendations: bool = True
    ) -> HealthCheckResult:
        """
        Perform comprehensive health check of all document chunk storage.
        
        Args:
            include_repair_recommendations: Whether to include repair suggestions
            
        Returns:
            Detailed health check result
        """
        try:
            self._logger.info("Starting comprehensive chunk storage health check")
            
            # Get all processed documents
            documents = await self._document_repository.find_by_status(DocumentStatus.PROCESSED)
            
            total_documents = len(documents)
            healthy_documents = 0
            inconsistent_documents = 0
            no_chunks_documents = 0
            page_overflow_documents = 0
            issues_found = []
            repair_recommendations = []
            
            for document in documents:
                try:
                    # Validate chunk storage for each document
                    validation_result = await self._chunk_storage_service.validate_chunk_storage(
                        document.document_id
                    )
                    
                    if validation_result.is_valid:
                        healthy_documents += 1
                    else:
                        # Categorize the issue
                        issue = {
                            "document_id": document.document_id.value,
                            "filename": document.file_metadata.filename,
                            "database_chunks": validation_result.database_chunk_count,
                            "vector_chunks": validation_result.vector_chunk_count,
                            "page_errors": validation_result.page_validation_errors,
                            "issue_type": self._categorize_issue(validation_result),
                            "severity": self._assess_severity(validation_result)
                        }
                        issues_found.append(issue)
                        
                        # Update counters
                        if validation_result.database_chunk_count == 0:
                            no_chunks_documents += 1
                        elif validation_result.database_chunk_count != validation_result.vector_chunk_count:
                            inconsistent_documents += 1
                        
                        if validation_result.page_validation_errors:
                            page_overflow_documents += 1
                        
                        # Generate repair recommendations
                        if include_repair_recommendations:
                            recommendations = self._generate_repair_recommendations(
                                document, validation_result
                            )
                            repair_recommendations.extend(recommendations)
                
                except Exception as e:
                    self._logger.error(
                        f"Health check failed for document {document.document_id.value}: {e}"
                    )
                    issues_found.append({
                        "document_id": document.document_id.value,
                        "filename": document.file_metadata.filename,
                        "error": str(e),
                        "issue_type": "HEALTH_CHECK_FAILED",
                        "severity": "HIGH"
                    })
            
            result = HealthCheckResult(
                total_documents=total_documents,
                healthy_documents=healthy_documents,
                inconsistent_documents=inconsistent_documents,
                no_chunks_documents=no_chunks_documents,
                page_overflow_documents=page_overflow_documents,
                issues_found=issues_found,
                repair_recommendations=repair_recommendations
            )
            
            self._logger.info(
                f"Health check complete: {healthy_documents}/{total_documents} healthy, "
                f"{len(issues_found)} issues found"
            )
            
            return result
            
        except Exception as e:
            self._logger.error(f"Comprehensive health check failed: {e}")
            raise RepositoryError(f"Health check failed: {str(e)}")
    
    async def repair_document_chunk_storage(
        self,
        document_id: DocumentId,
        force_reprocess: bool = False
    ) -> RepairResult:
        """
        Repair chunk storage for a specific document.
        
        Args:
            document_id: Document to repair
            force_reprocess: Whether to force complete reprocessing
            
        Returns:
            Repair operation result
        """
        try:
            self._logger.info(f"Starting chunk storage repair for document {document_id.value}")
            
            # Get current state
            validation_result = await self._chunk_storage_service.validate_chunk_storage(document_id)
            chunks_before = validation_result.database_chunk_count
            
            if validation_result.is_valid and not force_reprocess:
                return RepairResult(
                    document_id=document_id.value,
                    repair_attempted=False,
                    repair_successful=True,
                    error_message="Document already healthy",
                    chunks_before=chunks_before,
                    chunks_after=chunks_before
                )
            
            # Attempt repair
            repair_successful = await self._chunk_storage_service.repair_chunk_storage(document_id)
            
            if repair_successful:
                # Validate repair
                post_repair_validation = await self._chunk_storage_service.validate_chunk_storage(document_id)
                chunks_after = post_repair_validation.database_chunk_count
                
                return RepairResult(
                    document_id=document_id.value,
                    repair_attempted=True,
                    repair_successful=post_repair_validation.is_valid,
                    chunks_before=chunks_before,
                    chunks_after=chunks_after
                )
            else:
                return RepairResult(
                    document_id=document_id.value,
                    repair_attempted=True,
                    repair_successful=False,
                    error_message="Repair operation failed",
                    chunks_before=chunks_before,
                    chunks_after=0
                )
                
        except Exception as e:
            self._logger.error(f"Repair failed for document {document_id.value}: {e}")
            return RepairResult(
                document_id=document_id.value,
                repair_attempted=True,
                repair_successful=False,
                error_message=str(e),
                chunks_before=chunks_before if 'chunks_before' in locals() else 0,
                chunks_after=0
            )
    
    async def bulk_repair_inconsistent_documents(
        self,
        max_repairs: int = 10,
        dry_run: bool = False
    ) -> List[RepairResult]:
        """
        Repair multiple documents with chunk storage issues.
        
        Args:
            max_repairs: Maximum number of documents to repair
            dry_run: If True, only identify issues without repairing
            
        Returns:
            List of repair results
        """
        try:
            self._logger.info(f"Starting bulk repair (max: {max_repairs}, dry_run: {dry_run})")
            
            # Get health check to identify problematic documents
            health_check = await self.perform_comprehensive_health_check(
                include_repair_recommendations=False
            )
            
            # Sort issues by severity
            high_priority_issues = [
                issue for issue in health_check.issues_found
                if issue.get("severity") == "HIGH"
            ]
            
            repair_results = []
            repairs_performed = 0
            
            for issue in high_priority_issues[:max_repairs]:
                if repairs_performed >= max_repairs:
                    break
                
                document_id = DocumentId(issue["document_id"])
                
                if dry_run:
                    repair_results.append(RepairResult(
                        document_id=document_id.value,
                        repair_attempted=False,
                        repair_successful=False,
                        error_message="Dry run - repair not attempted"
                    ))
                else:
                    repair_result = await self.repair_document_chunk_storage(document_id)
                    repair_results.append(repair_result)
                    
                    if repair_result.repair_attempted:
                        repairs_performed += 1
            
            self._logger.info(f"Bulk repair complete: {repairs_performed} repairs attempted")
            return repair_results
            
        except Exception as e:
            self._logger.error(f"Bulk repair failed: {e}")
            raise RepositoryError(f"Bulk repair failed: {str(e)}")
    
    async def get_document_health_status(self, document_id: DocumentId) -> Dict[str, Any]:
        """
        Get detailed health status for a specific document.
        
        Args:
            document_id: Document to check
            
        Returns:
            Detailed health status
        """
        try:
            document = await self._document_repository.find_by_id(document_id)
            if not document:
                return {
                    "document_id": document_id.value,
                    "status": "NOT_FOUND",
                    "error": "Document not found"
                }
            
            validation_result = await self._chunk_storage_service.validate_chunk_storage(document_id)
            
            return {
                "document_id": document_id.value,
                "filename": document.file_metadata.filename,
                "document_status": document.status.value,
                "processing_stage": document.processing_stage.value,
                "page_count": document.page_count,
                "reported_chunks": document.total_chunks,
                "database_chunks": validation_result.database_chunk_count,
                "vector_chunks": validation_result.vector_chunk_count,
                "is_healthy": validation_result.is_valid,
                "page_validation_errors": validation_result.page_validation_errors,
                "issue_type": self._categorize_issue(validation_result) if not validation_result.is_valid else None,
                "severity": self._assess_severity(validation_result) if not validation_result.is_valid else None
            }
            
        except Exception as e:
            self._logger.error(f"Health status check failed for document {document_id.value}: {e}")
            return {
                "document_id": document_id.value,
                "status": "ERROR",
                "error": str(e)
            }
    
    def _categorize_issue(self, validation_result: ChunkStorageValidationResult) -> str:
        """Categorize the type of chunk storage issue."""
        if validation_result.database_chunk_count == 0:
            return "NO_CHUNKS_IN_DATABASE"
        elif validation_result.vector_chunk_count == 0:
            return "NO_CHUNKS_IN_VECTOR_STORE"
        elif validation_result.database_chunk_count != validation_result.vector_chunk_count:
            return "CHUNK_COUNT_MISMATCH"
        elif validation_result.page_validation_errors:
            return "PAGE_VALIDATION_ERRORS"
        else:
            return "UNKNOWN_ISSUE"
    
    def _assess_severity(self, validation_result: ChunkStorageValidationResult) -> str:
        """Assess the severity of chunk storage issues."""
        if validation_result.database_chunk_count == 0:
            return "HIGH"  # No chunks means search won't work
        elif validation_result.page_validation_errors:
            return "HIGH"  # Page errors cause citation failures
        elif abs(validation_result.database_chunk_count - validation_result.vector_chunk_count) > 5:
            return "MEDIUM"  # Large chunk count differences
        else:
            return "LOW"  # Minor inconsistencies
    
    def _generate_repair_recommendations(
        self,
        document: Document,
        validation_result: ChunkStorageValidationResult
    ) -> List[str]:
        """Generate repair recommendations for a document."""
        recommendations = []
        
        if validation_result.database_chunk_count == 0:
            recommendations.append(
                f"Document {document.document_id.value} ({document.file_metadata.filename}) "
                "has no chunks in database - requires complete reprocessing"
            )
        
        if validation_result.vector_chunk_count == 0:
            recommendations.append(
                f"Document {document.document_id.value} ({document.file_metadata.filename}) "
                "has no chunks in vector store - requires vectorization"
            )
        
        if validation_result.page_validation_errors:
            recommendations.append(
                f"Document {document.document_id.value} ({document.file_metadata.filename}) "
                f"has {len(validation_result.page_validation_errors)} page validation errors - "
                "requires chunk regeneration with proper page mapping"
            )
        
        return recommendations
