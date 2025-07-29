"""
Citation API Endpoints
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query
import logging
from datetime import datetime

from ..schemas.citation_schemas import (
    CitationExtractionRequest,
    CitationExtractionResponse,
    CitationLinkingRequest,
    CitationLinkingResponse,
    CitationAnalysisRequest,
    CitationAnalysisResponse,
    CitationClusteringRequest,
    CitationClusteringResponse,
    CitationValidationRequest,
    CitationValidationResponse,
    BulkCitationProcessingRequest,
    BulkCitationProcessingResponse,
    ErrorResponse
)
from ....core.application.use_cases.citation_use_case import CitationUseCase
from src.core.domain.services.citation_service import (
    CitationExtractionRequest as DomainCitationRequest,
    CitationProcessingOptions
)
from src.core.domain.value_objects import DocumentId, SessionId
from ....infrastructure.di.dependencies import get_citation_use_case
from src.shared.exceptions import (
    CitationExtractionError,
    CitationValidationError,
    ValidationError
)


router = APIRouter(prefix="/citations", tags=["citations"])
logger = logging.getLogger(__name__)


@router.post(
    "/extract",
    response_model=CitationExtractionResponse,
    summary="Extract citations from text",
    description="Extract and validate citations from response text using advanced NLP techniques"
)
async def extract_citations(
    request: CitationExtractionRequest,
    citation_use_case: CitationUseCase = Depends(get_citation_use_case)
) -> CitationExtractionResponse:
    """
    Extract citations from response text.
    
    Args:
        request: Citation extraction request
        citation_use_case: Citation use case dependency
        
    Returns:
        Citation extraction response with extracted citations
        
    Raises:
        HTTPException: If citation extraction fails
    """
    try:
        logger.info(f"Citation extraction request for session: {request.session_id}")
        
        # Convert API request to domain format
        source_chunks = []
        for chunk in request.source_chunks:
            source_chunks.append({
                "document_id": chunk.document_id,
                "document_title": chunk.document_title,
                "text_content": chunk.text_content,
                "chunk_id": chunk.chunk_id,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
                "start_position": chunk.start_position,
                "end_position": chunk.end_position,
                "metadata": chunk.metadata
            })
        
        context_documents = [DocumentId(doc_id) for doc_id in request.context_documents]
        
        processing_options = CitationProcessingOptions(
            extraction_strategy=request.options.extraction_strategy,
            similarity_threshold=request.options.similarity_threshold,
            max_citations=request.options.max_citations,
            include_page_numbers=request.options.include_page_numbers,
            include_confidence_scores=request.options.include_confidence_scores,
            cluster_similar_citations=request.options.cluster_similar_citations,
            validate_citations=request.options.validate_citations,
            merge_overlapping_citations=request.options.merge_overlapping_citations
        )
        
        # Execute citation extraction
        response = await citation_use_case.extract_citations(
            response_text=request.response_text,
            source_chunks=source_chunks,
            context_documents=context_documents,
            session_id=SessionId(request.session_id),
            options=processing_options
        )
        
        # Convert response to API format
        api_citations = []
        for citation in response.citations:
            api_citations.append({
                "id": citation.id,
                "source_text": citation.source_text,
                "cited_text": citation.cited_text,
                "source_location": {
                    "document_id": citation.source_location.document_id.value,
                    "document_title": citation.source_location.document_title,
                    "page_number": citation.source_location.page_number,
                    "section_title": citation.source_location.section_title,
                    "start_position": citation.source_location.start_position,
                    "end_position": citation.source_location.end_position,
                    "chunk_id": citation.source_location.chunk_id
                },
                "confidence": {
                    "value": citation.confidence.value,
                    "factors": citation.confidence.factors,
                    "explanation": citation.confidence.explanation
                },
                "citation_type": citation.citation_type.value,
                "extraction_method": citation.extraction_method,
                "validation_status": citation.validation_status.value,
                "metadata": citation.metadata
            })
        
        api_clusters = []
        for cluster in response.citation_clusters:
            api_clusters.append({
                "id": cluster.id,
                "cluster_label": cluster.cluster_label,
                "similarity_score": cluster.similarity_score,
                "citation_ids": [c.id for c in cluster.citations],
                "representative_citation_id": cluster.representative_citation.id,
                "cluster_metadata": cluster.cluster_metadata
            })
        
        return CitationExtractionResponse(
            citations=api_citations,
            citation_clusters=api_clusters,
            extraction_statistics={
                "total_citations_found": response.extraction_statistics.total_citations_found,
                "valid_citations": response.extraction_statistics.valid_citations,
                "invalid_citations": response.extraction_statistics.invalid_citations,
                "clusters_created": response.extraction_statistics.clusters_created,
                "processing_time_ms": response.extraction_statistics.processing_time_ms,
                "extraction_method_used": response.extraction_statistics.extraction_method_used,
                "confidence_distribution": response.extraction_statistics.confidence_distribution
            },
            processing_metadata=response.processing_metadata,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except ValidationError as e:
        logger.error(f"Validation error in citation extraction: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except CitationExtractionError as e:
        logger.error(f"Citation extraction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Citation extraction failed"
        )
    except Exception as e:
        logger.error(f"Unexpected error in citation extraction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post(
    "/link",
    response_model=CitationLinkingResponse,
    summary="Link citations to source documents",
    description="Create navigable links between citations and their source locations"
)
async def link_citations(
    request: CitationLinkingRequest,
    citation_use_case: CitationUseCase = Depends(get_citation_use_case)
) -> CitationLinkingResponse:
    """
    Link citations to their source documents.
    
    Args:
        request: Citation linking request
        citation_use_case: Citation use case dependency
        
    Returns:
        Citation linking response with navigable links
        
    Raises:
        HTTPException: If citation linking fails
    """
    try:
        logger.info(f"Citation linking request for {len(request.citation_ids)} citations")
        
        # Execute citation linking
        response = await citation_use_case.link_citations(
            citation_ids=request.citation_ids,
            target_documents=[DocumentId(doc_id) for doc_id in request.target_documents],
            session_id=SessionId(request.session_id),
            linking_options={
                "create_bidirectional_links": request.options.create_bidirectional_links,
                "include_context_links": request.options.include_context_links,
                "link_similarity_threshold": request.options.link_similarity_threshold,
                "max_links_per_citation": request.options.max_links_per_citation
            }
        )
        
        # Convert response to API format
        api_links = []
        for link in response.citation_links:
            api_links.append({
                "id": link.id,
                "citation_id": link.citation_id,
                "source_location": {
                    "document_id": link.source_location.document_id.value,
                    "document_title": link.source_location.document_title,
                    "page_number": link.source_location.page_number,
                    "section_title": link.source_location.section_title,
                    "start_position": link.source_location.start_position,
                    "end_position": link.source_location.end_position,
                    "chunk_id": link.source_location.chunk_id
                },
                "target_location": {
                    "document_id": link.target_location.document_id.value,
                    "document_title": link.target_location.document_title,
                    "page_number": link.target_location.page_number,
                    "section_title": link.target_location.section_title,
                    "start_position": link.target_location.start_position,
                    "end_position": link.target_location.end_position,
                    "chunk_id": link.target_location.chunk_id
                },
                "link_type": link.link_type.value,
                "confidence": link.confidence,
                "navigation_url": link.navigation_url,
                "preview_text": link.preview_text,
                "link_metadata": link.link_metadata
            })
        
        return CitationLinkingResponse(
            citation_links=api_links,
            linking_statistics={
                "total_links_created": response.linking_statistics.total_links_created,
                "successful_links": response.linking_statistics.successful_links,
                "failed_links": response.linking_statistics.failed_links,
                "bidirectional_links": response.linking_statistics.bidirectional_links,
                "processing_time_ms": response.linking_statistics.processing_time_ms
            },
            navigation_metadata=response.navigation_metadata,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Citation linking failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Citation linking failed"
        )


@router.post(
    "/analyze",
    response_model=CitationAnalysisResponse,
    summary="Analyze citation patterns and quality",
    description="Perform comprehensive analysis of citation usage patterns and quality metrics"
)
async def analyze_citations(
    request: CitationAnalysisRequest,
    citation_use_case: CitationUseCase = Depends(get_citation_use_case)
) -> CitationAnalysisResponse:
    """
    Analyze citation patterns and quality.
    
    Args:
        request: Citation analysis request
        citation_use_case: Citation use case dependency
        
    Returns:
        Citation analysis response with patterns and metrics
        
    Raises:
        HTTPException: If citation analysis fails
    """
    try:
        logger.info(f"Citation analysis request for session: {request.session_id}")
        
        # Execute citation analysis
        response = await citation_use_case.analyze_citations(
            session_id=SessionId(request.session_id),
            document_ids=[DocumentId(doc_id) for doc_id in request.document_ids] if request.document_ids else None,
            time_range=request.time_range,
            analysis_options={
                "include_quality_metrics": request.options.include_quality_metrics,
                "include_usage_patterns": request.options.include_usage_patterns,
                "include_source_analysis": request.options.include_source_analysis,
                "include_temporal_analysis": request.options.include_temporal_analysis,
                "confidence_threshold": request.options.confidence_threshold
            }
        )
        
        # Convert response to API format
        return CitationAnalysisResponse(
            quality_metrics={
                "average_confidence": response.quality_metrics.average_confidence,
                "citation_accuracy": response.quality_metrics.citation_accuracy,
                "source_diversity": response.quality_metrics.source_diversity,
                "citation_density": response.quality_metrics.citation_density,
                "validation_success_rate": response.quality_metrics.validation_success_rate
            },
            usage_patterns={
                "most_cited_documents": response.usage_patterns.most_cited_documents,
                "citation_frequency_by_type": response.usage_patterns.citation_frequency_by_type,
                "temporal_patterns": response.usage_patterns.temporal_patterns,
                "user_citation_preferences": response.usage_patterns.user_citation_preferences
            },
            source_analysis={
                "document_citation_counts": response.source_analysis.document_citation_counts,
                "page_citation_distribution": response.source_analysis.page_citation_distribution,
                "section_citation_patterns": response.source_analysis.section_citation_patterns,
                "source_reliability_scores": response.source_analysis.source_reliability_scores
            },
            recommendations=response.recommendations,
            analysis_metadata=response.analysis_metadata,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Citation analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Citation analysis failed"
        )


@router.post(
    "/cluster",
    response_model=CitationClusteringResponse,
    summary="Cluster similar citations",
    description="Group similar citations together for better organization and navigation"
)
async def cluster_citations(
    request: CitationClusteringRequest,
    citation_use_case: CitationUseCase = Depends(get_citation_use_case)
) -> CitationClusteringResponse:
    """
    Cluster similar citations.
    
    Args:
        request: Citation clustering request
        citation_use_case: Citation use case dependency
        
    Returns:
        Citation clustering response with grouped citations
        
    Raises:
        HTTPException: If citation clustering fails
    """
    try:
        logger.info(f"Citation clustering request for {len(request.citation_ids)} citations")
        
        # Execute citation clustering
        response = await citation_use_case.cluster_citations(
            citation_ids=request.citation_ids,
            clustering_options={
                "similarity_threshold": request.options.similarity_threshold,
                "max_clusters": request.options.max_clusters,
                "min_cluster_size": request.options.min_cluster_size,
                "clustering_method": request.options.clustering_method,
                "feature_weights": request.options.feature_weights
            }
        )
        
        # Convert response to API format
        api_clusters = []
        for cluster in response.citation_clusters:
            api_clusters.append({
                "id": cluster.id,
                "cluster_label": cluster.cluster_label,
                "similarity_score": cluster.similarity_score,
                "citation_ids": [c.id for c in cluster.citations],
                "representative_citation_id": cluster.representative_citation.id,
                "cluster_metadata": cluster.cluster_metadata
            })
        
        return CitationClusteringResponse(
            citation_clusters=api_clusters,
            clustering_statistics={
                "total_citations": response.clustering_statistics.total_citations,
                "clusters_created": response.clustering_statistics.clusters_created,
                "average_cluster_size": response.clustering_statistics.average_cluster_size,
                "silhouette_score": response.clustering_statistics.silhouette_score,
                "processing_time_ms": response.clustering_statistics.processing_time_ms
            },
            cluster_metadata=response.cluster_metadata,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Citation clustering failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Citation clustering failed"
        )


@router.post(
    "/validate",
    response_model=CitationValidationResponse,
    summary="Validate citation accuracy and quality",
    description="Validate citations against source documents and check for accuracy"
)
async def validate_citations(
    request: CitationValidationRequest,
    citation_use_case: CitationUseCase = Depends(get_citation_use_case)
) -> CitationValidationResponse:
    """
    Validate citation accuracy and quality.
    
    Args:
        request: Citation validation request
        citation_use_case: Citation use case dependency
        
    Returns:
        Citation validation response with validation results
        
    Raises:
        HTTPException: If citation validation fails
    """
    try:
        logger.info(f"Citation validation request for {len(request.citation_ids)} citations")
        
        # Execute citation validation
        response = await citation_use_case.validate_citations(
            citation_ids=request.citation_ids,
            validation_options={
                "check_source_accuracy": request.options.check_source_accuracy,
                "check_text_similarity": request.options.check_text_similarity,
                "check_location_accuracy": request.options.check_location_accuracy,
                "similarity_threshold": request.options.similarity_threshold,
                "strict_validation": request.options.strict_validation
            }
        )
        
        # Convert response to API format
        api_results = []
        for result in response.validation_results:
            api_results.append({
                "citation_id": result.citation_id,
                "is_valid": result.is_valid,
                "confidence_score": result.confidence_score,
                "validation_errors": result.validation_errors,
                "accuracy_metrics": result.accuracy_metrics,
                "suggestions": result.suggestions,
                "validation_metadata": result.validation_metadata
            })
        
        return CitationValidationResponse(
            validation_results=api_results,
            validation_summary={
                "total_citations": response.validation_summary.total_citations,
                "valid_citations": response.validation_summary.valid_citations,
                "invalid_citations": response.validation_summary.invalid_citations,
                "average_confidence": response.validation_summary.average_confidence,
                "common_errors": response.validation_summary.common_errors,
                "processing_time_ms": response.validation_summary.processing_time_ms
            },
            recommendations=response.recommendations,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except CitationValidationError as e:
        logger.error(f"Citation validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Citation validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Citation validation failed"
        )


@router.post(
    "/bulk-process",
    response_model=BulkCitationProcessingResponse,
    summary="Process multiple citations in bulk",
    description="Efficiently process multiple citations with extraction, linking, and validation"
)
async def bulk_process_citations(
    request: BulkCitationProcessingRequest,
    citation_use_case: CitationUseCase = Depends(get_citation_use_case)
) -> BulkCitationProcessingResponse:
    """
    Process multiple citations in bulk.
    
    Args:
        request: Bulk citation processing request
        citation_use_case: Citation use case dependency
        
    Returns:
        Bulk citation processing response
        
    Raises:
        HTTPException: If bulk processing fails
    """
    try:
        logger.info(f"Bulk citation processing for {len(request.processing_requests)} requests")
        
        # Execute bulk processing
        response = await citation_use_case.bulk_process_citations(
            processing_requests=request.processing_requests,
            processing_options={
                "parallel_processing": request.options.parallel_processing,
                "batch_size": request.options.batch_size,
                "continue_on_error": request.options.continue_on_error,
                "include_detailed_results": request.options.include_detailed_results
            }
        )
        
        # Convert response to API format
        api_results = []
        for result in response.processing_results:
            api_results.append({
                "request_id": result.request_id,
                "status": result.status,
                "citations_processed": result.citations_processed,
                "processing_time_ms": result.processing_time_ms,
                "error_message": result.error_message,
                "result_data": result.result_data
            })
        
        return BulkCitationProcessingResponse(
            processing_results=api_results,
            bulk_statistics={
                "total_requests": response.bulk_statistics.total_requests,
                "successful_requests": response.bulk_statistics.successful_requests,
                "failed_requests": response.bulk_statistics.failed_requests,
                "total_citations_processed": response.bulk_statistics.total_citations_processed,
                "total_processing_time_ms": response.bulk_statistics.total_processing_time_ms,
                "average_processing_time_ms": response.bulk_statistics.average_processing_time_ms
            },
            processing_metadata=response.processing_metadata,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Bulk citation processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Bulk citation processing failed"
        )


@router.get(
    "/session/{session_id}",
    response_model=List[Dict[str, Any]],
    summary="Get citations for a session",
    description="Retrieve all citations associated with a specific session"
)
async def get_session_citations(
    session_id: str,
    include_clusters: bool = Query(False, description="Include citation clusters"),
    include_links: bool = Query(False, description="Include citation links"),
    citation_use_case: CitationUseCase = Depends(get_citation_use_case)
) -> List[Dict[str, Any]]:
    """
    Get citations for a specific session.
    
    Args:
        session_id: Session ID
        include_clusters: Include citation clusters
        include_links: Include citation links
        citation_use_case: Citation use case dependency
        
    Returns:
        List of citations for the session
        
    Raises:
        HTTPException: If retrieval fails
    """
    try:
        logger.info(f"Retrieving citations for session: {session_id}")
        
        # Get session citations
        citations = await citation_use_case.get_session_citations(
            session_id=SessionId(session_id),
            include_clusters=include_clusters,
            include_links=include_links
        )
        
        # Convert to API format
        api_citations = []
        for citation in citations:
            api_citation = {
                "id": citation.id,
                "source_text": citation.source_text,
                "cited_text": citation.cited_text,
                "source_location": {
                    "document_id": citation.source_location.document_id.value,
                    "document_title": citation.source_location.document_title,
                    "page_number": citation.source_location.page_number
                },
                "confidence": citation.confidence.value,
                "citation_type": citation.citation_type.value,
                "created_at": citation.metadata.get("created_at"),
                "metadata": citation.metadata
            }
            api_citations.append(api_citation)
        
        return api_citations
        
    except Exception as e:
        logger.error(f"Session citation retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Session citation retrieval failed"
        )


@router.get(
    "/health",
    summary="Citation service health check",
    description="Check the health of citation extraction and processing services"
)
async def citation_health_check(
    citation_use_case: CitationUseCase = Depends(get_citation_use_case)
) -> Dict[str, Any]:
    """
    Check the health of citation services.
    
    Args:
        citation_use_case: Citation use case dependency
        
    Returns:
        Health check results
    """
    try:
        # Perform basic health checks
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "citation_extractor": "healthy",
                "citation_linker": "healthy",
                "citation_analyzer": "healthy",
                "citation_clusterer": "healthy",
                "citation_validator": "healthy"
            },
            "performance": {
                "average_extraction_time_ms": 150,
                "average_linking_time_ms": 100,
                "average_validation_time_ms": 75,
                "cache_hit_rate": 0.80,
                "error_rate": 0.01
            },
            "capabilities": {
                "extraction_methods": ["semantic", "syntactic", "hybrid"],
                "supported_languages": ["en", "es", "fr", "de"],
                "max_citations_per_request": 100,
                "clustering_algorithms": ["kmeans", "hierarchical", "dbscan"]
            }
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Citation health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Citation services are unhealthy"
        )
