"""
Advanced Search API Schemas

"""
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class SearchTypeEnum(str, Enum):
    """Search type enumeration."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    QUESTION_ANSWERING = "question_answering"
    HYBRID = "hybrid"


class RankingStrategyEnum(str, Enum):
    """Ranking strategy enumeration."""
    RELEVANCE_SCORE = "relevance_score"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    KEYWORD_MATCH = "keyword_match"
    HYBRID_SCORE = "hybrid_score"
    TEMPORAL_RELEVANCE = "temporal_relevance"
    USER_PREFERENCE = "user_preference"


class SuggestionTypeEnum(str, Enum):
    """Search suggestion type enumeration."""
    EXPANSION = "expansion"
    REFINEMENT = "refinement"
    CORRECTION = "correction"
    EXACT_PHRASE = "exact_phrase"
    RELATED_TOPIC = "related_topic"


# Request/Response Models for Search Filters and Context

class SearchFilterRequest(BaseModel):
    """Search filter request schema."""
    document_ids: Optional[List[str]] = Field(None, description="Filter by specific document IDs")
    document_types: Optional[List[str]] = Field(None, description="Filter by document types")
    date_range: Optional[Dict[str, str]] = Field(None, description="Date range filter")
    authors: Optional[List[str]] = Field(None, description="Filter by authors")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    languages: Optional[List[str]] = Field(None, description="Filter by languages")
    content_types: Optional[List[str]] = Field(None, description="Filter by content types")
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum confidence threshold")
    max_results: Optional[int] = Field(None, ge=1, le=100, description="Maximum results per filter")
    metadata_filters: Optional[Dict[str, Any]] = Field(None, description="Additional metadata filters")


class SearchContextRequest(BaseModel):
    """Search context request schema."""
    session_id: Optional[str] = Field(None, description="Session ID for context")
    conversation_history: Optional[List[str]] = Field(None, description="Recent conversation messages")
    user_preferences: Optional[Dict[str, Any]] = Field(None, description="User search preferences")
    previous_searches: Optional[List[str]] = Field(None, description="Previous search queries")
    current_document_focus: Optional[str] = Field(None, description="Currently focused document ID")
    temporal_context: Optional[Dict[str, Any]] = Field(None, description="Temporal context information")
    domain_context: Optional[str] = Field(None, description="Domain or topic context")


# Enhanced Search Schemas

class EnhancedSearchRequest(BaseModel):
    """Enhanced search request schema."""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    search_type: SearchTypeEnum = Field(SearchTypeEnum.HYBRID, description="Type of search to perform")
    include_citations: bool = Field(True, description="Include citation extraction")
    max_results: int = Field(20, ge=1, le=100, description="Maximum number of results")
    filters: Optional[SearchFilterRequest] = Field(None, description="Search filters")
    context: Optional[SearchContextRequest] = Field(None, description="Search context")
    ranking_strategy: RankingStrategyEnum = Field(RankingStrategyEnum.HYBRID_SCORE, description="Result ranking strategy")
    cluster_results: bool = Field(False, description="Cluster similar results")
    generate_suggestions: bool = Field(True, description="Generate search suggestions")

    @validator('query')
    def validate_query(cls, v):
        """Validate search query."""
        if not v or not v.strip():
            raise ValueError("Search query cannot be empty")
        return v.strip()


class SearchResultResponse(BaseModel):
    """Search result response schema."""
    id: str = Field(..., description="Unique result ID")
    document_id: str = Field(..., description="Source document ID")
    document_title: str = Field(..., description="Source document title")
    document_filename: Optional[str] = Field(None, description="Source document filename")
    chunk_id: Optional[str] = Field(None, description="Chunk ID within document")
    content: str = Field(..., description="Full content of the result")
    snippet: str = Field(..., description="Highlighted snippet")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    page_number: Optional[int] = Field(None, description="Page number in document")
    section_title: Optional[str] = Field(None, description="Section title")
    start_position: Optional[int] = Field(None, description="Start position in document")
    end_position: Optional[int] = Field(None, description="End position in document")
    highlighted_content: Optional[str] = Field(None, description="Content with highlights")
    context_before: Optional[str] = Field(None, description="Context before match")
    context_after: Optional[str] = Field(None, description="Context after match")
    search_type_used: Optional[SearchTypeEnum] = Field(None, description="Search type that found this result")
    ranking_factors: Dict[str, Any] = Field(default_factory=dict, description="Factors used in ranking")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SearchClusterResponse(BaseModel):
    """Search cluster response schema."""
    id: str = Field(..., description="Cluster ID")
    cluster_label: str = Field(..., description="Human-readable cluster label")
    cluster_score: float = Field(..., ge=0.0, le=1.0, description="Cluster coherence score")
    result_count: int = Field(..., ge=1, description="Number of results in cluster")
    representative_result_id: str = Field(..., description="ID of representative result")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Cluster metadata")


class SearchSuggestionResponse(BaseModel):
    """Search suggestion response schema."""
    suggestion: str = Field(..., description="Suggested search query")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Suggestion confidence")
    suggestion_type: SuggestionTypeEnum = Field(..., description="Type of suggestion")
    context: str = Field(..., description="Context for suggestion")
    expected_results: Optional[int] = Field(None, description="Expected number of results")


class CitationResponse(BaseModel):
    """Citation response schema."""
    id: str = Field(..., description="Citation ID")
    text: str = Field(..., description="Citation text")
    document_title: str = Field(..., description="Source document title")
    page_number: Optional[int] = Field(None, description="Page number")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Citation confidence")


class EnhancedSearchResponse(BaseModel):
    """Enhanced search response schema."""
    results: List[SearchResultResponse] = Field(..., description="Search results")
    clusters: List[SearchClusterResponse] = Field(default_factory=list, description="Result clusters")
    suggestions: List[SearchSuggestionResponse] = Field(default_factory=list, description="Search suggestions")
    citations: List[CitationResponse] = Field(default_factory=list, description="Extracted citations")
    total_results: int = Field(..., ge=0, description="Total number of results")
    search_time_ms: int = Field(..., ge=0, description="Search execution time in milliseconds")
    query_analysis: Dict[str, Any] = Field(default_factory=dict, description="Query analysis results")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    timestamp: str = Field(..., description="Response timestamp")


# Contextual Search Schemas

class ContextualSearchRequest(BaseModel):
    """Contextual search request schema."""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    context: SearchContextRequest = Field(..., description="Search context")
    max_results: int = Field(20, ge=1, le=100, description="Maximum number of results")

    @validator('query')
    def validate_query(cls, v):
        """Validate search query."""
        if not v or not v.strip():
            raise ValueError("Search query cannot be empty")
        return v.strip()


class ContextualSearchResponse(BaseModel):
    """Contextual search response schema."""
    results: List[SearchResultResponse] = Field(..., description="Contextual search results")
    total_results: int = Field(..., ge=0, description="Total number of results")
    context_applied: bool = Field(..., description="Whether context was successfully applied")
    adaptation_score: float = Field(..., ge=0.0, le=1.0, description="Context adaptation score")
    timestamp: str = Field(..., description="Response timestamp")


# Multi-Document Search Schemas

class MultiDocumentSearchRequest(BaseModel):
    """Multi-document search request schema."""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    document_ids: List[str] = Field(..., min_items=1, max_items=50, description="Document IDs to search")
    search_type: SearchTypeEnum = Field(SearchTypeEnum.HYBRID, description="Type of search to perform")

    @validator('query')
    def validate_query(cls, v):
        """Validate search query."""
        if not v or not v.strip():
            raise ValueError("Search query cannot be empty")
        return v.strip()

    @validator('document_ids')
    def validate_document_ids(cls, v):
        """Validate document IDs."""
        if not v:
            raise ValueError("At least one document ID must be provided")
        return v


class DocumentSearchResult(BaseModel):
    """Document-specific search result schema."""
    results: List[SearchResultResponse] = Field(..., description="Results from this document")
    result_count: int = Field(..., ge=0, description="Number of results in this document")
    document_title: str = Field(..., description="Document title")


class MultiDocumentSearchResponse(BaseModel):
    """Multi-document search response schema."""
    document_results: Dict[str, DocumentSearchResult] = Field(..., description="Results grouped by document")
    total_results: int = Field(..., ge=0, description="Total results across all documents")
    documents_searched: int = Field(..., ge=0, description="Number of documents searched")
    timestamp: str = Field(..., description="Response timestamp")


# Search Suggestions Schemas

class SearchSuggestionRequest(BaseModel):
    """Search suggestion request schema."""
    query: str = Field(..., min_length=1, max_length=500, description="Partial or complete query")
    session_id: Optional[str] = Field(None, description="Session ID for context")
    max_suggestions: int = Field(5, ge=1, le=20, description="Maximum number of suggestions")

    @validator('query')
    def validate_query(cls, v):
        """Validate search query."""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class SearchSuggestionListResponse(BaseModel):
    """Search suggestion list response schema."""
    suggestions: List[SearchSuggestionResponse] = Field(..., description="Search suggestions")
    query: str = Field(..., description="Original query")
    suggestion_count: int = Field(..., ge=0, description="Number of suggestions returned")
    timestamp: str = Field(..., description="Response timestamp")


# Search Analytics Schemas

class SearchAnalyticsRequest(BaseModel):
    """Search analytics request schema."""
    session_id: Optional[str] = Field(None, description="Filter by session ID")
    time_range: Optional[Dict[str, str]] = Field(None, description="Time range for analysis")
    include_performance: bool = Field(True, description="Include performance metrics")
    include_patterns: bool = Field(True, description="Include usage patterns")
    include_suggestions: bool = Field(True, description="Include improvement suggestions")


class PopularQueryResponse(BaseModel):
    """Popular query response schema."""
    query: str = Field(..., description="Search query")
    count: int = Field(..., ge=1, description="Number of times searched")
    average_results: Optional[float] = Field(None, description="Average number of results")
    average_response_time_ms: Optional[float] = Field(None, description="Average response time")


class SearchAnalyticsResponse(BaseModel):
    """Search analytics response schema."""
    total_searches: int = Field(..., ge=0, description="Total number of searches")
    unique_queries: int = Field(..., ge=0, description="Number of unique queries")
    average_response_time_ms: float = Field(..., ge=0, description="Average response time")
    popular_queries: List[PopularQueryResponse] = Field(default_factory=list, description="Most popular queries")
    search_patterns: Dict[str, Any] = Field(default_factory=dict, description="Usage patterns")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    improvement_suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    analysis_period_days: int = Field(..., ge=1, description="Analysis period in days")
    timestamp: str = Field(..., description="Response timestamp")


# Error Response Schemas

class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


class ValidationErrorResponse(BaseModel):
    """Validation error response schema."""
    error: str = Field("validation_error", description="Error type")
    message: str = Field(..., description="Validation error message")
    field_errors: List[Dict[str, str]] = Field(default_factory=list, description="Field-specific errors")
    timestamp: str = Field(..., description="Error timestamp")


# Health Check Schemas

class AdvancedSearchHealthResponse(BaseModel):
    """Advanced search health check response schema."""
    status: str = Field(..., description="Overall health status")
    timestamp: str = Field(..., description="Health check timestamp")
    services: Dict[str, str] = Field(..., description="Individual service statuses")
    performance: Dict[str, Union[float, int]] = Field(..., description="Performance metrics")
    uptime_seconds: Optional[int] = Field(None, description="Service uptime in seconds")
    version: Optional[str] = Field(None, description="Service version")


# Export all schemas for easy importing
__all__ = [
    # Enums
    "SearchTypeEnum",
    "RankingStrategyEnum", 
    "SuggestionTypeEnum",
    
    # Request schemas
    "SearchFilterRequest",
    "SearchContextRequest",
    "EnhancedSearchRequest",
    "ContextualSearchRequest",
    "MultiDocumentSearchRequest",
    "SearchSuggestionRequest",
    "SearchAnalyticsRequest",
    
    # Response schemas
    "SearchResultResponse",
    "SearchClusterResponse",
    "SearchSuggestionResponse",
    "CitationResponse",
    "EnhancedSearchResponse",
    "ContextualSearchResponse",
    "DocumentSearchResult",
    "MultiDocumentSearchResponse",
    "SearchSuggestionListResponse",
    "PopularQueryResponse",
    "SearchAnalyticsResponse",
    "AdvancedSearchHealthResponse",
    
    # Error schemas
    "ErrorResponse",
    "ValidationErrorResponse"
]
