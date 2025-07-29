"""
Advanced Search Use Cases
"""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging
from datetime import datetime

from ...domain.services.advanced_search_service import (
    AdvancedSearchService,
    SemanticSearchEngine,
    KeywordSearchEngine,
    QuestionAnsweringEngine,
    SearchRanker,
    SearchClusterer,
    AdvancedSearchRequest,
    AdvancedSearchResponse,
    SearchResult,
    SearchCluster,
    SearchSuggestion,
    SearchFilter,
    SearchContext,
    SearchType,
    RankingStrategy
)
from ...domain.services.citation_service import (
    CitationService,
    CitationExtractionRequest
)
from ...domain.value_objects import DocumentId, SessionId
from src.shared.exceptions import (
    AdvancedSearchError,
    ValidationError
)


@dataclass
class EnhancedSearchRequest:
    """Enhanced search request with citation integration following Single Responsibility Principle."""
    query: str
    search_type: SearchType = SearchType.HYBRID
    include_citations: bool = True
    max_results: int = 20
    filters: Optional[SearchFilter] = None
    context: Optional[SearchContext] = None
    ranking_strategy: RankingStrategy = RankingStrategy.HYBRID_SCORE
    cluster_results: bool = False
    generate_suggestions: bool = True


@dataclass
class EnhancedSearchResponse:
    """Enhanced search response with citations following Single Responsibility Principle."""
    results: List[SearchResult]
    clusters: List[SearchCluster]
    suggestions: List[SearchSuggestion]
    citations: List[Dict[str, Any]]
    total_results: int
    search_time_ms: int
    query_analysis: Dict[str, Any]
    performance_metrics: Dict[str, Any]


@dataclass
class SearchAnalyticsRequest:
    """Request for search analytics following Single Responsibility Principle."""
    session_id: Optional[SessionId] = None
    time_range: Optional[tuple] = None
    include_performance: bool = True
    include_patterns: bool = True
    include_suggestions: bool = True


@dataclass
class SearchAnalyticsResponse:
    """Response for search analytics following Single Responsibility Principle."""
    total_searches: int
    unique_queries: int
    average_response_time_ms: float
    popular_queries: List[Dict[str, Any]]
    search_patterns: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    improvement_suggestions: List[str]


class AdvancedSearchUseCase:
    """
    Use case for advanced search operations following Single Responsibility Principle.
    
    Orchestrates multiple search engines and enhances results with citations.
    """
    
    def __init__(
        self,
        semantic_engine: SemanticSearchEngine,
        keyword_engine: KeywordSearchEngine,
        qa_engine: QuestionAnsweringEngine,
        search_ranker: SearchRanker,
        search_clusterer: Optional[SearchClusterer] = None,
        citation_service: Optional[CitationService] = None
    ):
        """
        Initialize advanced search use case.
        
        Args:
            semantic_engine: Semantic search engine
            keyword_engine: Keyword search engine
            qa_engine: Question-answering engine
            search_ranker: Search result ranker
            search_clusterer: Optional search clusterer
            citation_service: Optional citation service
        """
        self._semantic_engine = semantic_engine
        self._keyword_engine = keyword_engine
        self._qa_engine = qa_engine
        self._search_ranker = search_ranker
        self._search_clusterer = search_clusterer
        self._citation_service = citation_service
        self._logger = logging.getLogger(__name__)
    
    async def enhanced_search(
        self,
        request: EnhancedSearchRequest
    ) -> EnhancedSearchResponse:
        """
        Perform enhanced search with multiple engines and citation integration.
        
        Args:
            request: Enhanced search request
            
        Returns:
            Enhanced search response
            
        Raises:
            AdvancedSearchError: If search fails
            ValidationError: If request is invalid
        """
        start_time = datetime.utcnow()
        
        try:
            # Validate request
            if not request.query or not request.query.strip():
                raise ValidationError("Search query cannot be empty")
            
            self._logger.info(f"Performing enhanced search: {request.query}")
            
            # Analyze query
            query_analysis = await self._analyze_query(request.query)
            
            # Execute search based on type
            if request.search_type == SearchType.SEMANTIC:
                results = await self._semantic_engine.semantic_search(
                    query=request.query,
                    filters=request.filters,
                    max_results=request.max_results
                )
            elif request.search_type == SearchType.KEYWORD:
                results = await self._keyword_engine.keyword_search(
                    query=request.query,
                    filters=request.filters,
                    max_results=request.max_results
                )
            elif request.search_type == SearchType.QUESTION_ANSWERING:
                results = await self._qa_engine.answer_question(
                    question=request.query,
                    context_documents=request.filters.document_ids if request.filters else None
                )
            elif request.search_type == SearchType.HYBRID:
                results = await self._hybrid_search(request)
            else:
                results = await self._semantic_engine.semantic_search(
                    query=request.query,
                    filters=request.filters,
                    max_results=request.max_results
                )
            
            # Rank results
            ranked_results = await self._search_ranker.rank_results(
                results=results,
                ranking_strategy=request.ranking_strategy,
                context=request.context
            )
            
            # Cluster results if requested
            clusters = []
            if request.cluster_results and self._search_clusterer:
                clusters = await self._search_clusterer.cluster_results(
                    results=ranked_results,
                    max_clusters=5
                )
            
            # Generate suggestions if requested
            suggestions = []
            if request.generate_suggestions:
                suggestions = await self._generate_search_suggestions(
                    query=request.query,
                    results=ranked_results,
                    context=request.context
                )
            
            # Extract citations if requested
            citations = []
            if request.include_citations and self._citation_service:
                citations = await self._extract_search_citations(
                    query=request.query,
                    results=ranked_results
                )
            
            # Calculate performance metrics
            end_time = datetime.utcnow()
            search_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            performance_metrics = {
                "search_time_ms": search_time_ms,
                "results_found": len(ranked_results),
                "clusters_created": len(clusters),
                "citations_extracted": len(citations),
                "search_engines_used": self._get_engines_used(request.search_type)
            }
            
            self._logger.info(
                f"Enhanced search completed in {search_time_ms}ms with {len(ranked_results)} results"
            )
            
            return EnhancedSearchResponse(
                results=ranked_results,
                clusters=clusters,
                suggestions=suggestions,
                citations=citations,
                total_results=len(ranked_results),
                search_time_ms=search_time_ms,
                query_analysis=query_analysis,
                performance_metrics=performance_metrics
            )
            
        except ValidationError:
            raise
        except Exception as e:
            self._logger.error(f"Enhanced search failed: {e}")
            raise AdvancedSearchError(f"Enhanced search failed: {str(e)}")
    
    async def contextual_search(
        self,
        query: str,
        context: SearchContext,
        max_results: int = 20
    ) -> List[SearchResult]:
        """
        Perform context-aware search.
        
        Args:
            query: Search query
            context: Search context
            max_results: Maximum results to return
            
        Returns:
            Contextual search results
            
        Raises:
            AdvancedSearchError: If contextual search fails
        """
        try:
            self._logger.info(f"Performing contextual search: {query}")
            
            # Enhance query with context
            enhanced_query = await self._enhance_query_with_context(query, context)
            
            # Perform semantic search with enhanced query
            results = await self._semantic_engine.semantic_search(
                query=enhanced_query,
                max_results=max_results
            )
            
            # Apply context-based filtering and ranking
            contextual_results = await self._apply_contextual_ranking(results, context)
            
            return contextual_results
            
        except Exception as e:
            self._logger.error(f"Contextual search failed: {e}")
            raise AdvancedSearchError(f"Contextual search failed: {str(e)}")
    
    async def multi_document_search(
        self,
        query: str,
        document_ids: List[DocumentId],
        search_type: SearchType = SearchType.HYBRID
    ) -> Dict[str, List[SearchResult]]:
        """
        Search across multiple specific documents.
        
        Args:
            query: Search query
            document_ids: Documents to search
            search_type: Type of search to perform
            
        Returns:
            Results grouped by document
            
        Raises:
            AdvancedSearchError: If multi-document search fails
        """
        try:
            self._logger.info(f"Multi-document search across {len(document_ids)} documents")
            
            document_results = {}
            
            for doc_id in document_ids:
                # Create filter for single document
                doc_filter = SearchFilter(document_ids=[doc_id])
                
                # Search within document
                if search_type == SearchType.SEMANTIC:
                    results = await self._semantic_engine.semantic_search(
                        query=query,
                        filters=doc_filter,
                        max_results=10
                    )
                elif search_type == SearchType.KEYWORD:
                    results = await self._keyword_engine.keyword_search(
                        query=query,
                        filters=doc_filter,
                        max_results=10
                    )
                else:  # Hybrid
                    results = await self._hybrid_search_single_doc(query, doc_filter)
                
                document_results[doc_id.value] = results
            
            return document_results
            
        except Exception as e:
            self._logger.error(f"Multi-document search failed: {e}")
            raise AdvancedSearchError(f"Multi-document search failed: {str(e)}")
    
    async def get_search_analytics(
        self,
        request: SearchAnalyticsRequest
    ) -> SearchAnalyticsResponse:
        """
        Get search analytics and metrics.
        
        Args:
            request: Search analytics request
            
        Returns:
            Search analytics response
            
        Raises:
            AdvancedSearchError: If analytics retrieval fails
        """
        try:
            self._logger.info("Generating search analytics")
            
            # This would typically query a search analytics database
            # For now, we'll return mock analytics
            analytics = SearchAnalyticsResponse(
                total_searches=1000,
                unique_queries=750,
                average_response_time_ms=250.0,
                popular_queries=[
                    {"query": "machine learning", "count": 50},
                    {"query": "artificial intelligence", "count": 45},
                    {"query": "data science", "count": 40}
                ],
                search_patterns={
                    "peak_hours": [9, 10, 14, 15],
                    "common_terms": ["data", "analysis", "model", "algorithm"],
                    "search_types": {"semantic": 60, "keyword": 30, "hybrid": 10}
                },
                performance_metrics={
                    "average_results_per_query": 15.5,
                    "cache_hit_rate": 0.75,
                    "error_rate": 0.02
                },
                improvement_suggestions=[
                    "Consider adding more semantic search capabilities",
                    "Optimize keyword search for technical terms",
                    "Implement query auto-completion"
                ]
            )
            
            return analytics
            
        except Exception as e:
            self._logger.error(f"Search analytics failed: {e}")
            raise AdvancedSearchError(f"Search analytics failed: {str(e)}")
    
    async def _hybrid_search(self, request: EnhancedSearchRequest) -> List[SearchResult]:
        """Perform hybrid search combining semantic and keyword results."""
        try:
            # Perform both semantic and keyword searches
            semantic_results = await self._semantic_engine.semantic_search(
                query=request.query,
                filters=request.filters,
                max_results=request.max_results // 2
            )
            
            keyword_results = await self._keyword_engine.keyword_search(
                query=request.query,
                filters=request.filters,
                max_results=request.max_results // 2
            )
            
            # Combine and deduplicate results
            combined_results = await self._combine_search_results(
                semantic_results, keyword_results
            )
            
            return combined_results[:request.max_results]
            
        except Exception as e:
            self._logger.error(f"Hybrid search failed: {e}")
            return []
    
    async def _hybrid_search_single_doc(
        self,
        query: str,
        doc_filter: SearchFilter
    ) -> List[SearchResult]:
        """Perform hybrid search on a single document."""
        try:
            semantic_results = await self._semantic_engine.semantic_search(
                query=query,
                filters=doc_filter,
                max_results=5
            )
            
            keyword_results = await self._keyword_engine.keyword_search(
                query=query,
                filters=doc_filter,
                max_results=5
            )
            
            return await self._combine_search_results(semantic_results, keyword_results)
            
        except Exception as e:
            self._logger.error(f"Single document hybrid search failed: {e}")
            return []
    
    async def _combine_search_results(
        self,
        semantic_results: List[SearchResult],
        keyword_results: List[SearchResult]
    ) -> List[SearchResult]:
        """Combine and deduplicate search results from different engines."""
        try:
            combined = []
            seen_ids = set()
            
            # Add semantic results first (higher priority)
            for result in semantic_results:
                if result.chunk_id not in seen_ids:
                    result.ranking_factors["semantic_score"] = result.relevance_score
                    combined.append(result)
                    seen_ids.add(result.chunk_id)
            
            # Add keyword results
            for result in keyword_results:
                if result.chunk_id not in seen_ids:
                    result.ranking_factors["keyword_score"] = result.relevance_score
                    combined.append(result)
                    seen_ids.add(result.chunk_id)
                else:
                    # Enhance existing result with keyword score
                    existing = next(r for r in combined if r.chunk_id == result.chunk_id)
                    existing.ranking_factors["keyword_score"] = result.relevance_score
                    # Recalculate relevance score
                    existing.relevance_score = (
                        existing.ranking_factors.get("semantic_score", 0) * 0.7 +
                        existing.ranking_factors.get("keyword_score", 0) * 0.3
                    )
            
            return combined
            
        except Exception as e:
            self._logger.error(f"Result combination failed: {e}")
            return semantic_results + keyword_results
    
    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze search query to determine characteristics."""
        try:
            analysis = {
                "query_length": len(query),
                "word_count": len(query.split()),
                "is_question": query.strip().endswith("?"),
                "contains_quotes": '"' in query or "'" in query,
                "contains_operators": any(op in query.upper() for op in ["AND", "OR", "NOT"]),
                "query_type": "question" if query.strip().endswith("?") else "statement",
                "complexity": "simple" if len(query.split()) <= 3 else "complex"
            }
            
            return analysis
            
        except Exception as e:
            self._logger.error(f"Query analysis failed: {e}")
            return {"error": str(e)}
    
    async def _generate_search_suggestions(
        self,
        query: str,
        results: List[SearchResult],
        context: Optional[SearchContext]
    ) -> List[SearchSuggestion]:
        """Generate search suggestions based on query and results."""
        try:
            suggestions = []
            
            # Generate suggestions based on query analysis
            query_words = query.lower().split()
            
            # Suggest related terms from results
            if results:
                common_terms = set()
                for result in results[:5]:  # Use top 5 results
                    content_words = result.content.lower().split()
                    common_terms.update(word for word in content_words 
                                      if len(word) > 3 and word not in query_words)
                
                # Create suggestions from common terms
                for term in list(common_terms)[:3]:
                    suggestion = SearchSuggestion(
                        suggestion=f"{query} {term}",
                        confidence=0.7,
                        suggestion_type="expansion",
                        context="Related term from search results"
                    )
                    suggestions.append(suggestion)
            
            # Add refinement suggestions
            if len(query_words) == 1:
                suggestions.append(SearchSuggestion(
                    suggestion=f'"{query}"',
                    confidence=0.8,
                    suggestion_type="exact_phrase",
                    context="Search for exact phrase"
                ))
            
            return suggestions
            
        except Exception as e:
            self._logger.error(f"Suggestion generation failed: {e}")
            return []
    
    async def _extract_search_citations(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[Dict[str, Any]]:
        """Extract citations from search results."""
        try:
            if not self._citation_service:
                return []
            
            citations = []
            
            for result in results[:5]:  # Extract citations from top 5 results
                # Create mock source chunks from search results
                source_chunks = [{
                    "document_id": result.document_id.value,
                    "document_title": result.document_title,
                    "text_content": result.content,
                    "chunk_id": result.chunk_id,
                    "page_number": result.page_number
                }]
                
                citation_request = CitationExtractionRequest(
                    response_text=result.snippet,
                    source_chunks=source_chunks,
                    context_documents=[result.document_id],
                    max_citations=2
                )
                
                citation_response = await self._citation_service.process_citations(
                    response_text=result.snippet,
                    source_chunks=source_chunks,
                    context_documents=[result.document_id],
                    session_id=SessionId("search_session"),
                    options={"extraction_strategy": "basic"}
                )
                
                for citation in citation_response.citations:
                    citations.append({
                        "id": citation.id,
                        "text": citation.source_text[:100] + "...",
                        "document_title": citation.source_location.document_title,
                        "page_number": citation.source_location.page_number,
                        "confidence": citation.confidence.value
                    })
            
            return citations
            
        except Exception as e:
            self._logger.error(f"Citation extraction failed: {e}")
            return []
    
    async def _enhance_query_with_context(
        self,
        query: str,
        context: SearchContext
    ) -> str:
        """Enhance query with contextual information."""
        try:
            enhanced_query = query
            
            # Add context from conversation history
            if context.conversation_history:
                recent_topics = []
                for msg in context.conversation_history[-3:]:  # Last 3 messages
                    words = msg.split()
                    recent_topics.extend(word for word in words if len(word) > 4)
                
                if recent_topics:
                    # Add most relevant context terms
                    context_terms = list(set(recent_topics))[:2]
                    enhanced_query += " " + " ".join(context_terms)
            
            # Add domain context
            if context.domain_context:
                enhanced_query += f" {context.domain_context}"
            
            return enhanced_query
            
        except Exception as e:
            self._logger.error(f"Query enhancement failed: {e}")
            return query
    
    async def _apply_contextual_ranking(
        self,
        results: List[SearchResult],
        context: SearchContext
    ) -> List[SearchResult]:
        """Apply context-based ranking to search results."""
        try:
            # Boost results from current document focus
            if context.current_document_focus:
                for result in results:
                    if result.document_id == context.current_document_focus:
                        result.relevance_score *= 1.2
                        result.ranking_factors["context_boost"] = 0.2
            
            # Sort by enhanced relevance score
            return sorted(results, key=lambda r: r.relevance_score, reverse=True)
            
        except Exception as e:
            self._logger.error(f"Contextual ranking failed: {e}")
            return results
    
    def _get_engines_used(self, search_type: SearchType) -> List[str]:
        """Get list of search engines used for a search type."""
        if search_type == SearchType.SEMANTIC:
            return ["semantic"]
        elif search_type == SearchType.KEYWORD:
            return ["keyword"]
        elif search_type == SearchType.QUESTION_ANSWERING:
            return ["qa", "semantic"]
        elif search_type == SearchType.HYBRID:
            return ["semantic", "keyword"]
        else:
            return ["semantic"]
