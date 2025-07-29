"""
Production Health Monitoring API Routes

Provides endpoints for chunk storage health monitoring, validation,
and automated repair in production environments.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Dict, Any, Optional
import logging

from .....core.application.use_cases.chunk_storage_health_use_case import (
    ChunkStorageHealthUseCase,
    HealthCheckResult,
    RepairResult
)
from .....core.domain.value_objects import DocumentId
from ....dependencies import get_chunk_storage_health_use_case
from .....shared.exceptions import RepositoryError

logger = logging.getLogger(__name__)

# Create router for health monitoring endpoints
health_router = APIRouter(prefix="/api/v1/health", tags=["health"])


@health_router.get("/chunk-storage/comprehensive")
async def get_comprehensive_health_check(
    include_recommendations: bool = Query(True, description="Include repair recommendations"),
    health_use_case: ChunkStorageHealthUseCase = Depends(get_chunk_storage_health_use_case)
) -> Dict[str, Any]:
    """
    Perform comprehensive health check of all document chunk storage.
    
    Returns detailed analysis of chunk storage consistency across
    vector database and PostgreSQL database.
    """
    try:
        logger.info("API: Starting comprehensive chunk storage health check")
        
        health_result = await health_use_case.perform_comprehensive_health_check(
            include_repair_recommendations=include_recommendations
        )
        
        return {
            "status": "success",
            "timestamp": "2025-07-28T13:48:01+03:00",
            "summary": {
                "total_documents": health_result.total_documents,
                "healthy_documents": health_result.healthy_documents,
                "inconsistent_documents": health_result.inconsistent_documents,
                "no_chunks_documents": health_result.no_chunks_documents,
                "page_overflow_documents": health_result.page_overflow_documents,
                "health_percentage": round(
                    (health_result.healthy_documents / max(health_result.total_documents, 1)) * 100, 2
                )
            },
            "issues": health_result.issues_found,
            "repair_recommendations": health_result.repair_recommendations if include_recommendations else []
        }
        
    except Exception as e:
        logger.error(f"Comprehensive health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )


@health_router.get("/chunk-storage/document/{document_id}")
async def get_document_health_status(
    document_id: str,
    health_use_case: ChunkStorageHealthUseCase = Depends(get_chunk_storage_health_use_case)
) -> Dict[str, Any]:
    """
    Get detailed health status for a specific document.
    
    Args:
        document_id: UUID of the document to check
    """
    try:
        logger.info(f"API: Checking health status for document {document_id}")
        
        doc_id = DocumentId(document_id)
        health_status = await health_use_case.get_document_health_status(doc_id)
        
        return {
            "status": "success",
            "timestamp": "2025-07-28T13:48:01+03:00",
            "document": health_status
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid document ID format: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Document health check failed for {document_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Document health check failed: {str(e)}"
        )


@health_router.post("/chunk-storage/repair/{document_id}")
async def repair_document_chunk_storage(
    document_id: str,
    force_reprocess: bool = Query(False, description="Force complete reprocessing"),
    health_use_case: ChunkStorageHealthUseCase = Depends(get_chunk_storage_health_use_case)
) -> Dict[str, Any]:
    """
    Repair chunk storage for a specific document.
    
    Args:
        document_id: UUID of the document to repair
        force_reprocess: Whether to force complete reprocessing
    """
    try:
        logger.info(f"API: Starting chunk storage repair for document {document_id}")
        
        doc_id = DocumentId(document_id)
        repair_result = await health_use_case.repair_document_chunk_storage(
            doc_id, force_reprocess=force_reprocess
        )
        
        return {
            "status": "success",
            "timestamp": "2025-07-28T13:48:01+03:00",
            "repair_result": {
                "document_id": repair_result.document_id,
                "repair_attempted": repair_result.repair_attempted,
                "repair_successful": repair_result.repair_successful,
                "error_message": repair_result.error_message,
                "chunks_before": repair_result.chunks_before,
                "chunks_after": repair_result.chunks_after,
                "improvement": repair_result.chunks_after - repair_result.chunks_before
            }
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid document ID format: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Document repair failed for {document_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Document repair failed: {str(e)}"
        )


@health_router.post("/chunk-storage/bulk-repair")
async def bulk_repair_inconsistent_documents(
    max_repairs: int = Query(10, ge=1, le=50, description="Maximum number of documents to repair"),
    dry_run: bool = Query(False, description="Only identify issues without repairing"),
    health_use_case: ChunkStorageHealthUseCase = Depends(get_chunk_storage_health_use_case)
) -> Dict[str, Any]:
    """
    Repair multiple documents with chunk storage issues.
    
    Args:
        max_repairs: Maximum number of documents to repair (1-50)
        dry_run: If true, only identify issues without repairing
    """
    try:
        logger.info(f"API: Starting bulk repair (max: {max_repairs}, dry_run: {dry_run})")
        
        repair_results = await health_use_case.bulk_repair_inconsistent_documents(
            max_repairs=max_repairs,
            dry_run=dry_run
        )
        
        successful_repairs = sum(1 for r in repair_results if r.repair_successful)
        attempted_repairs = sum(1 for r in repair_results if r.repair_attempted)
        
        return {
            "status": "success",
            "timestamp": "2025-07-28T13:48:01+03:00",
            "summary": {
                "total_processed": len(repair_results),
                "repairs_attempted": attempted_repairs,
                "repairs_successful": successful_repairs,
                "success_rate": round((successful_repairs / max(attempted_repairs, 1)) * 100, 2),
                "dry_run": dry_run
            },
            "repair_results": [
                {
                    "document_id": r.document_id,
                    "repair_attempted": r.repair_attempted,
                    "repair_successful": r.repair_successful,
                    "error_message": r.error_message,
                    "chunks_before": r.chunks_before,
                    "chunks_after": r.chunks_after,
                    "improvement": r.chunks_after - r.chunks_before
                }
                for r in repair_results
            ]
        }
        
    except Exception as e:
        logger.error(f"Bulk repair failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Bulk repair failed: {str(e)}"
        )


@health_router.get("/chunk-storage/monitoring/dashboard")
async def get_monitoring_dashboard(
    health_use_case: ChunkStorageHealthUseCase = Depends(get_chunk_storage_health_use_case)
) -> Dict[str, Any]:
    """
    Get comprehensive monitoring dashboard data for chunk storage health.
    
    Provides real-time metrics and alerts for production monitoring.
    """
    try:
        logger.info("API: Generating monitoring dashboard data")
        
        # Get comprehensive health check
        health_result = await health_use_case.perform_comprehensive_health_check(
            include_repair_recommendations=True
        )
        
        # Calculate metrics
        total_docs = health_result.total_documents
        healthy_docs = health_result.healthy_documents
        health_percentage = round((healthy_docs / max(total_docs, 1)) * 100, 2)
        
        # Categorize issues by severity
        high_severity_issues = [i for i in health_result.issues_found if i.get("severity") == "HIGH"]
        medium_severity_issues = [i for i in health_result.issues_found if i.get("severity") == "MEDIUM"]
        low_severity_issues = [i for i in health_result.issues_found if i.get("severity") == "LOW"]
        
        # Generate alerts
        alerts = []
        if health_percentage < 80:
            alerts.append({
                "level": "ERROR",
                "message": f"Chunk storage health is critical: {health_percentage}% healthy",
                "action_required": "Immediate attention required"
            })
        elif health_percentage < 95:
            alerts.append({
                "level": "WARNING", 
                "message": f"Chunk storage health is degraded: {health_percentage}% healthy",
                "action_required": "Investigation recommended"
            })
        
        if len(high_severity_issues) > 0:
            alerts.append({
                "level": "ERROR",
                "message": f"{len(high_severity_issues)} documents have high-severity chunk storage issues",
                "action_required": "Run bulk repair immediately"
            })
        
        return {
            "status": "success",
            "timestamp": "2025-07-28T13:48:01+03:00",
            "dashboard": {
                "overview": {
                    "total_documents": total_docs,
                    "healthy_documents": healthy_docs,
                    "health_percentage": health_percentage,
                    "total_issues": len(health_result.issues_found)
                },
                "issue_breakdown": {
                    "high_severity": len(high_severity_issues),
                    "medium_severity": len(medium_severity_issues),
                    "low_severity": len(low_severity_issues),
                    "no_chunks": health_result.no_chunks_documents,
                    "inconsistent": health_result.inconsistent_documents,
                    "page_overflow": health_result.page_overflow_documents
                },
                "alerts": alerts,
                "top_issues": health_result.issues_found[:10],  # Top 10 most critical
                "repair_recommendations": health_result.repair_recommendations[:5]  # Top 5 recommendations
            }
        }
        
    except Exception as e:
        logger.error(f"Monitoring dashboard generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Dashboard generation failed: {str(e)}"
        )
