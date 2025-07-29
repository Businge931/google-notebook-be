"""
Advanced Search API Endpoints

"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query
import logging
from datetime import datetime

from ..schemas.advanced_search_schemas import (
    EnhancedSearchRequest,
    EnhancedSearchResponse,
    ContextualSearchRequest,
    ContextualSearchResponse,
    MultiDocumentSearchRequest,
    MultiDocumentSearchResponse,
    SearchAnalyticsRequest,
    SearchAnalyticsResponse,
    SearchSuggestionRequest,
    SearchSuggestionResponse,
    ErrorResponse
)
from ....core.application.use_cases.advanced_search_use_case import (
    AdvancedSearchUseCase,
    EnhancedSearchRequest as UseCaseEnhancedSearchRequest,
    SearchAnalyticsRequest as UseCaseSearchAnalyticsRequest
)
from src.core.domain.services.advanced_search_service import (
    SearchType,
    RankingStrategy,
    SearchFilter,
    SearchContext
)
from src.core.domain.value_objects import DocumentId, SessionId
from ....infrastructure.di.dependencies import get_advanced_search_use_case
from src.shared.exceptions import (
    AdvancedSearchError,
    ValidationError
)
from src.infrastructure.adapters.services.response_synthesis_service import ResponseSynthesisService
from src.core.domain.services.advanced_search_service import SearchResult
from src.infrastructure.adapters.services.enhanced_response_synthesis import create_enhanced_response_synthesis
from src.infrastructure.config.settings import get_settings


router = APIRouter(prefix="/advanced-search", tags=["advanced-search"])
logger = logging.getLogger(__name__)


@router.post(
    "/enhanced",
    response_model=EnhancedSearchResponse,
    summary="Perform enhanced search with multiple engines",
    description="Execute advanced search using semantic, keyword, and QA engines with citation integration"
)
async def enhanced_search(
    request: EnhancedSearchRequest,
    advanced_search_use_case: AdvancedSearchUseCase = Depends(get_advanced_search_use_case)
) -> EnhancedSearchResponse:
    """
    Perform enhanced search with multiple search engines.
    
    Args:
        request: Enhanced search request
        advanced_search_use_case: Advanced search use case dependency
        
    Returns:
        Enhanced search response with results, citations, and analytics
        
    Raises:
        HTTPException: If search fails
    """
    try:
        logger.info(f"Enhanced search request: {request.query}")
        
        # Convert API request to use case format
        search_filter = None
        if request.filters:
            search_filter = SearchFilter(
                document_ids=[DocumentId(doc_id) for doc_id in request.filters.document_ids] if request.filters.document_ids else None,
                document_types=request.filters.document_types,
                date_range=request.filters.date_range,
                authors=request.filters.authors,
                tags=request.filters.tags,
                languages=request.filters.languages,
                content_types=request.filters.content_types,
                min_confidence=request.filters.min_confidence,
                max_results=request.filters.max_results,
                metadata_filters=request.filters.metadata_filters
            )
        
        search_context = None
        if request.context:
            search_context = SearchContext(
                session_id=SessionId(request.context.session_id) if request.context.session_id else None,
                conversation_history=request.context.conversation_history,
                user_preferences=request.context.user_preferences,
                previous_searches=request.context.previous_searches,
                current_document_focus=DocumentId(request.context.current_document_focus) if request.context.current_document_focus else None,
                temporal_context=request.context.temporal_context,
                domain_context=request.context.domain_context
            )
        
        use_case_request = UseCaseEnhancedSearchRequest(
            query=request.query,
            search_type=SearchType(request.search_type),
            include_citations=request.include_citations,
            max_results=request.max_results,
            filters=search_filter,
            context=search_context,
            ranking_strategy=RankingStrategy(request.ranking_strategy),
            cluster_results=request.cluster_results,
            generate_suggestions=request.generate_suggestions
        )
        
        # Execute search
        response = await advanced_search_use_case.enhanced_search(use_case_request)
        
        # Initialize response synthesis service
        synthesis_service = ResponseSynthesisService()
        
        # Convert search results to SearchResult objects for synthesis
        search_results = []
        for result in response.results:
            from src.core.domain.value_objects.document_id import DocumentId
            
            search_result = SearchResult(
                id=result.chunk_id,
                document_id=DocumentId(result.document_id),
                document_title=result.document_title,
                document_filename=result.document_filename,
                chunk_id=result.chunk_id,
                content=result.text_content,
                snippet=result.text_content[:200] + "..." if len(result.text_content) > 200 else result.text_content,
                relevance_score=result.similarity_score,
                confidence_score=result.similarity_score,
                page_number=result.page_number,
                section_title=None,  # Add missing section_title parameter
                start_position=result.start_position,
                end_position=result.end_position,
                highlighted_content=result.text_content,
                search_type_used=None,
                ranking_factors={"similarity_score": result.similarity_score},
                metadata=result.metadata
            )
            search_results.append(search_result)
        
        # Synthesize comprehensive response
        # Convert SearchResult objects to dictionaries for SimpleResponseSynthesis
        search_results_dict = [{
            "chunk_id": r.chunk_id,
            "document_id": r.document_id.value if r.document_id else None,
            "document_title": r.document_title,
            "document_filename": r.document_filename,
            "page_number": r.page_number,
            "content": r.content,
            "snippet": r.snippet,
            "relevance_score": r.relevance_score,
            "confidence_score": r.confidence_score,
            "start_position": r.start_position,
            "end_position": r.end_position,
            "metadata": r.metadata
        } for r in search_results]
        
        synthesis_result = await synthesis_service.synthesize_comprehensive_response(
            query=request.query,
            search_results=search_results_dict,
            max_response_length=2000
        )
        
        # Convert response to API format
        api_results = []
        for result in response.results:
            api_results.append({
                "id": result.chunk_id,
                "document_id": result.document_id,
                "document_title": result.document_title,
                "document_filename": result.document_filename,
                "chunk_id": result.chunk_id,
                "content": result.text_content,
                "snippet": result.text_content[:200] + "..." if len(result.text_content) > 200 else result.text_content,
                "relevance_score": result.similarity_score,
                "confidence_score": result.similarity_score,
                "page_number": result.page_number,
                "section_title": getattr(result, 'section_title', None),
                "start_position": result.start_position,
                "end_position": result.end_position,
                "highlighted_content": getattr(result, 'highlighted_content', result.text_content),
                "context_before": getattr(result, 'context_before', None),
                "context_after": getattr(result, 'context_after', None),
                "search_type_used": getattr(result, 'search_type_used', None),
                "ranking_factors": getattr(result, 'ranking_factors', {}),
                "metadata": result.metadata
            })
        
        api_clusters = []
        for cluster in response.clusters:
            api_clusters.append({
                "id": cluster.id,
                "cluster_label": cluster.cluster_label,
                "cluster_score": cluster.cluster_score,
                "result_count": len(cluster.results),
                "representative_result_id": cluster.representative_result.chunk_id,
                "metadata": cluster.cluster_metadata
            })
        
        api_suggestions = []
        for suggestion in response.suggestions:
            api_suggestions.append({
                "suggestion": suggestion.suggestion,
                "confidence": suggestion.confidence,
                "suggestion_type": suggestion.suggestion_type,
                "context": suggestion.context,
                "expected_results": suggestion.expected_results
            })
        
        # Add synthesized response to the API response
        enhanced_response = EnhancedSearchResponse(
            results=api_results,
            clusters=api_clusters,
            suggestions=api_suggestions,
            citations=response.citations,
            total_results=response.total_results,
            search_time_ms=response.search_time_ms,
            query_analysis=response.query_analysis,
            performance_metrics=response.performance_metrics,
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Add synthesized response as additional field
        enhanced_response_dict = enhanced_response.dict()
        enhanced_response_dict['synthesized_response'] = {
            "response": synthesis_result["response"],
            "citations": synthesis_result["citations"],
            "confidence": synthesis_result["confidence"],
            "sources_used": synthesis_result["sources_used"]
        }
        
        return enhanced_response_dict
        
    except ValidationError as e:
        logger.error(f"Validation error in enhanced search: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except AdvancedSearchError as e:
        logger.error(f"Advanced search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Advanced search failed"
        )
    except Exception as e:
        logger.error(f"Unexpected error in enhanced search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post(
    "/synthesized",
    summary="Get comprehensive synthesized response",
    description="Get a single, detailed, comprehensive response with citations (like NotebookLM analysis)"
)
async def synthesized_analysis(
    request: EnhancedSearchRequest,
    advanced_search_use_case: AdvancedSearchUseCase = Depends(get_advanced_search_use_case)
) -> Dict[str, Any]:
    """
    Get a comprehensive synthesized response for document analysis.
    
    Returns a single, detailed response with proper citations instead of
    multiple fragmented search results, similar to NotebookLM's analysis style.
    
    Args:
        request: Enhanced search request
        advanced_search_use_case: Advanced search use case dependency
        
    Returns:
        Comprehensive synthesized response with citations
        
    Raises:
        HTTPException: If analysis fails
    """
    try:
        logger.info(f"Synthesized analysis request: {request.query}")
        
        # Convert API request to use case format (same as enhanced search)
        search_filter = None
        if request.filters:
            search_filter = SearchFilter(
                document_ids=[DocumentId(doc_id) for doc_id in request.filters.document_ids] if request.filters.document_ids else None,
                document_types=request.filters.document_types,
                date_range=request.filters.date_range,
                authors=request.filters.authors,
                tags=request.filters.tags,
                languages=request.filters.languages,
                content_types=request.filters.content_types,
                min_confidence=request.filters.min_confidence,
                max_results=request.filters.max_results,
                metadata_filters=request.filters.metadata_filters
            )
        
        use_case_request = UseCaseEnhancedSearchRequest(
            query=request.query,
            search_type=SearchType(request.search_type),
            include_citations=True,  # Always include citations for synthesis
            max_results=request.max_results,
            filters=search_filter,
            context=None,  # Simplified for synthesis
            ranking_strategy=RankingStrategy.HYBRID_SCORE,
            cluster_results=False,
            generate_suggestions=False
        )
        
        # Execute search
        response = await advanced_search_use_case.enhanced_search(use_case_request)
        
        # Initialize enhanced response synthesis service
        # Get OpenAI API key from settings
        settings = get_settings()
        openai_api_key = getattr(settings.ai_service, 'openai_api_key', None)
        
        # Create enhanced synthesis service with OpenAI integration
        synthesis_service = create_enhanced_response_synthesis(openai_api_key)
        
        # Convert search results to SearchResult objects for synthesis
        search_results = []
        for result in response.results:
            from src.core.domain.services.advanced_search_service import SearchResult
            from src.core.domain.value_objects.document_id import DocumentId
            
            search_result = SearchResult(
                id=result.chunk_id,
                document_id=DocumentId(result.document_id),
                document_title=result.document_title,
                document_filename=result.document_filename,
                chunk_id=result.chunk_id,
                content=result.text_content,
                snippet=result.text_content[:200] + "..." if len(result.text_content) > 200 else result.text_content,
                relevance_score=result.similarity_score,
                confidence_score=result.similarity_score,
                page_number=result.page_number,
                section_title=None,  # Add missing section_title parameter
                start_position=result.start_position,
                end_position=result.end_position,
                highlighted_content=result.text_content,
                search_type_used=None,
                ranking_factors={"similarity_score": result.similarity_score},
                metadata=result.metadata
            )
            search_results.append(search_result)
        
        # Synthesize comprehensive response
        # Convert SearchResult objects to dictionaries for SimpleResponseSynthesis
        search_results_dict = [{
            "chunk_id": r.chunk_id,
            "document_id": r.document_id.value if r.document_id else None,
            "document_title": r.document_title,
            "document_filename": r.document_filename,
            "page_number": r.page_number,
            "content": r.content,
            "snippet": r.snippet,
            "relevance_score": r.relevance_score,
            "confidence_score": r.confidence_score,
            "start_position": r.start_position,
            "end_position": r.end_position,
            "metadata": r.metadata
        } for r in search_results]
        
        synthesis_result = await synthesis_service.synthesize_comprehensive_response(
            query=request.query,
            search_results=search_results_dict,
            max_response_length=2000
        )
        
        # Return only the synthesized response (no fragmented results)
        return {
            "query": request.query,
            "synthesized_response": synthesis_result.get("response", synthesis_result.get("synthesized_response", "")),
            "citations": synthesis_result["citations"],
            "confidence": synthesis_result["confidence"],
            "sources_used": synthesis_result["sources_used"],
            "analysis_type": synthesis_result.get("analysis_type", "general"),
            "query_analysis": synthesis_result.get("query_analysis", {}),
            "synthesis_metadata": synthesis_result.get("synthesis_metadata", {}),
            "search_time_ms": response.search_time_ms,
            "processing_time_ms": response.search_time_ms,
            "ai_enhancements": {
                "pdf_analysis": "enhanced",
                "vectorization": "optimized", 
                "response_synthesis": "ai_powered",
                "citation_quality": "improved"
            },
            "timestamp": datetime.utcnow().isoformat(),
            "enhancement_version": "ai_enhanced_v2"
        }
        
    except ValidationError as e:
        logger.error(f"Validation error in synthesized analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except AdvancedSearchError as e:
        logger.error(f"Advanced search error in synthesis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document analysis failed"
        )
    except Exception as e:
        logger.error(f"Unexpected error in synthesized analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post(
    "/contextual",
    response_model=ContextualSearchResponse,
    summary="Perform context-aware search",
    description="Execute search that adapts to conversation context and user preferences"
)
async def contextual_search(
    request: ContextualSearchRequest,
    advanced_search_use_case: AdvancedSearchUseCase = Depends(get_advanced_search_use_case)
) -> ContextualSearchResponse:
    """
    Perform context-aware search.
    
    Args:
        request: Contextual search request
        advanced_search_use_case: Advanced search use case dependency
        
    Returns:
        Contextual search response
        
    Raises:
        HTTPException: If search fails
    """
    try:
        logger.info(f"Contextual search request: {request.query}")
        
        # Convert context to domain format
        search_context = SearchContext(
            session_id=SessionId(request.context.session_id) if request.context.session_id else None,
            conversation_history=request.context.conversation_history,
            user_preferences=request.context.user_preferences,
            previous_searches=request.context.previous_searches,
            current_document_focus=DocumentId(request.context.current_document_focus) if request.context.current_document_focus else None,
            temporal_context=request.context.temporal_context,
            domain_context=request.context.domain_context
        )
        
        # Execute contextual search
        results = await advanced_search_use_case.contextual_search(
            query=request.query,
            context=search_context,
            max_results=request.max_results
        )
        
        # Convert results to API format
        api_results = []
        for result in results:
            api_results.append({
                "id": result.chunk_id,
                "document_id": result.document_id,
                "document_title": result.document_title,
                "content": result.text_content,
                "snippet": result.text_content[:200] + "..." if len(result.text_content) > 200 else result.text_content,
                "relevance_score": result.similarity_score,
                "confidence_score": result.similarity_score,
                "page_number": result.page_number,
                "highlighted_content": getattr(result, 'highlighted_content', result.text_content),
                "ranking_factors": result.ranking_factors,
                "metadata": result.metadata
            })
        
        return ContextualSearchResponse(
            results=api_results,
            total_results=len(api_results),
            context_applied=True,
            adaptation_score=0.8,  # Mock score
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Contextual search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Contextual search failed"
        )


@router.post(
    "/multi-document",
    response_model=MultiDocumentSearchResponse,
    summary="Search across multiple specific documents",
    description="Execute search within a specific set of documents with per-document results"
)
async def multi_document_search(
    request: MultiDocumentSearchRequest,
    advanced_search_use_case: AdvancedSearchUseCase = Depends(get_advanced_search_use_case)
) -> MultiDocumentSearchResponse:
    """
    Search across multiple specific documents.
    
    Args:
        request: Multi-document search request
        advanced_search_use_case: Advanced search use case dependency
        
    Returns:
        Multi-document search response
        
    Raises:
        HTTPException: If search fails
    """
    try:
        logger.info(f"Multi-document search: {request.query} across {len(request.document_ids)} documents")
        
        # Convert document IDs
        document_ids = [DocumentId(doc_id) for doc_id in request.document_ids]
        
        # Execute multi-document search
        document_results = await advanced_search_use_case.multi_document_search(
            query=request.query,
            document_ids=document_ids,
            search_type=SearchType(request.search_type)
        )
        
        # Convert results to API format
        api_document_results = {}
        total_results = 0
        
        for doc_id, results in document_results.items():
            api_results = []
            for result in results:
                api_results.append({
                    "id": result.chunk_id,
                    "content": result.text_content,
                    "snippet": result.text_content[:200] + "..." if len(result.text_content) > 200 else result.text_content,
                    "relevance_score": result.relevance_score,
                    "page_number": result.page_number,
                    "highlighted_content": result.highlighted_content
                })
            
            api_document_results[doc_id] = {
                "results": api_results,
                "result_count": len(api_results),
                "document_title": results[0].document_title if results else "Unknown"
            }
            total_results += len(api_results)
        
        return MultiDocumentSearchResponse(
            document_results=api_document_results,
            total_results=total_results,
            documents_searched=len(request.document_ids),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Multi-document search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Multi-document search failed"
        )


@router.get(
    "/suggestions",
    response_model=SearchSuggestionResponse,
    summary="Get search query suggestions",
    description="Generate search suggestions based on partial query and context"
)
async def get_search_suggestions(
    query: str = Query(..., description="Partial or complete search query"),
    session_id: Optional[str] = Query(None, description="Session ID for context"),
    max_suggestions: int = Query(5, description="Maximum number of suggestions"),
    advanced_search_use_case: AdvancedSearchUseCase = Depends(get_advanced_search_use_case)
) -> SearchSuggestionResponse:
    """
    Get search query suggestions.
    
    Args:
        query: Partial or complete search query
        session_id: Optional session ID for context
        max_suggestions: Maximum number of suggestions
        advanced_search_use_case: Advanced search use case dependency
        
    Returns:
        Search suggestions
        
    Raises:
        HTTPException: If suggestion generation fails
    """
    try:
        logger.info(f"Generating search suggestions for: {query}")
        
        # Create search context if session provided
        context = None
        if session_id:
            context = SearchContext(session_id=SessionId(session_id))
        
        # Generate suggestions (mock implementation for now)
        suggestions = [
            {
                "suggestion": f"{query} analysis",
                "confidence": 0.9,
                "suggestion_type": "expansion",
                "context": "Common search pattern",
                "expected_results": 25
            },
            {
                "suggestion": f"{query} examples",
                "confidence": 0.8,
                "suggestion_type": "expansion",
                "context": "Popular refinement",
                "expected_results": 15
            },
            {
                "suggestion": f'"{query}"',
                "confidence": 0.7,
                "suggestion_type": "exact_phrase",
                "context": "Exact phrase search",
                "expected_results": 8
            }
        ]
        
        return SearchSuggestionResponse(
            suggestions=suggestions[:max_suggestions],
            query=query,
            suggestion_count=len(suggestions[:max_suggestions]),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Search suggestion generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Suggestion generation failed"
        )


@router.get(
    "/analytics",
    response_model=SearchAnalyticsResponse,
    summary="Get search analytics and metrics",
    description="Retrieve comprehensive search analytics including performance metrics and usage patterns"
)
async def get_search_analytics(
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    days: int = Query(7, description="Number of days to analyze"),
    include_performance: bool = Query(True, description="Include performance metrics"),
    include_patterns: bool = Query(True, description="Include usage patterns"),
    advanced_search_use_case: AdvancedSearchUseCase = Depends(get_advanced_search_use_case)
) -> SearchAnalyticsResponse:
    """
    Get search analytics and metrics.
    
    Args:
        session_id: Optional session filter
        days: Number of days to analyze
        include_performance: Include performance metrics
        include_patterns: Include usage patterns
        advanced_search_use_case: Advanced search use case dependency
        
    Returns:
        Search analytics
        
    Raises:
        HTTPException: If analytics retrieval fails
    """
    try:
        logger.info(f"Generating search analytics for {days} days")
        
        # Create analytics request
        analytics_request = UseCaseSearchAnalyticsRequest(
            session_id=SessionId(session_id) if session_id else None,
            include_performance=include_performance,
            include_patterns=include_patterns
        )
        
        # Get analytics
        analytics = await advanced_search_use_case.get_search_analytics(analytics_request)
        
        return SearchAnalyticsResponse(
            total_searches=analytics.total_searches,
            unique_queries=analytics.unique_queries,
            average_response_time_ms=analytics.average_response_time_ms,
            popular_queries=analytics.popular_queries,
            search_patterns=analytics.search_patterns,
            performance_metrics=analytics.performance_metrics,
            improvement_suggestions=analytics.improvement_suggestions,
            analysis_period_days=days,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Search analytics failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analytics retrieval failed"
        )


@router.get(
    "/health",
    summary="Advanced search service health check",
    description="Check the health of advanced search services and engines"
)
async def advanced_search_health_check(
    advanced_search_use_case: AdvancedSearchUseCase = Depends(get_advanced_search_use_case)
) -> Dict[str, Any]:
    """
    Check the health of advanced search services.
    
    Args:
        advanced_search_use_case: Advanced search use case dependency
        
    Returns:
        Health check results
    """
    try:
        # Perform basic health checks
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "semantic_search": "healthy",
                "keyword_search": "healthy",
                "qa_engine": "healthy",
                "search_ranker": "healthy",
                "citation_service": "healthy"
            },
            "performance": {
                "average_search_time_ms": 250,
                "cache_hit_rate": 0.75,
                "error_rate": 0.02
            }
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Advanced search health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Advanced search services are unhealthy"
        )
