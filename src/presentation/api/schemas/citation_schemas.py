"""
Citation API Schemas

"""
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class CitationTypeEnum(str, Enum):
    """Citation type enumeration."""
    DIRECT_QUOTE = "direct_quote"
    PARAPHRASE = "paraphrase"
    REFERENCE = "reference"
    SUMMARY = "summary"
    INLINE = "inline"
    FOOTNOTE = "footnote"


class ExtractionStrategyEnum(str, Enum):
    """Citation extraction strategy enumeration."""
    SEMANTIC = "semantic"
    SYNTACTIC = "syntactic"
    HYBRID = "hybrid"
    RULE_BASED = "rule_based"
    ML_BASED = "ml_based"


class ValidationStatusEnum(str, Enum):
    """Citation validation status enumeration."""
    VALID = "valid"
    INVALID = "invalid"
    PENDING = "pending"
    UNCERTAIN = "uncertain"
    NEEDS_REVIEW = "needs_review"


class LinkTypeEnum(str, Enum):
    """Citation link type enumeration."""
    DIRECT = "direct"
    CONTEXTUAL = "contextual"
    RELATED = "related"
    BIDIRECTIONAL = "bidirectional"
    CROSS_REFERENCE = "cross_reference"


class ClusteringMethodEnum(str, Enum):
    """Citation clustering method enumeration."""
    KMEANS = "kmeans"
    HIERARCHICAL = "hierarchical"
    DBSCAN = "dbscan"
    SEMANTIC_SIMILARITY = "semantic_similarity"


# Source and Location Schemas

class SourceChunkRequest(BaseModel):
    """Source chunk request schema."""
    document_id: str = Field(..., description="Source document ID")
    document_title: str = Field(..., description="Source document title")
    text_content: str = Field(..., description="Text content of the chunk")
    chunk_id: Optional[str] = Field(None, description="Unique chunk identifier")
    page_number: Optional[int] = Field(None, ge=1, description="Page number in document")
    section_title: Optional[str] = Field(None, description="Section title")
    start_position: Optional[int] = Field(None, ge=0, description="Start position in document")
    end_position: Optional[int] = Field(None, ge=0, description="End position in document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @validator('end_position')
    def validate_positions(cls, v, values):
        """Validate that end position is after start position."""
        if v is not None and 'start_position' in values and values['start_position'] is not None:
            if v <= values['start_position']:
                raise ValueError("End position must be greater than start position")
        return v


class SourceLocationResponse(BaseModel):
    """Source location response schema."""
    document_id: str = Field(..., description="Source document ID")
    document_title: str = Field(..., description="Source document title")
    page_number: Optional[int] = Field(None, description="Page number")
    section_title: Optional[str] = Field(None, description="Section title")
    start_position: Optional[int] = Field(None, description="Start position")
    end_position: Optional[int] = Field(None, description="End position")
    chunk_id: Optional[str] = Field(None, description="Chunk identifier")


class CitationConfidenceResponse(BaseModel):
    """Citation confidence response schema."""
    value: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    factors: Dict[str, float] = Field(default_factory=dict, description="Confidence factors")
    explanation: str = Field(..., description="Confidence explanation")


# Citation Processing Options

class CitationProcessingOptionsRequest(BaseModel):
    """Citation processing options request schema."""
    extraction_strategy: ExtractionStrategyEnum = Field(ExtractionStrategyEnum.HYBRID, description="Extraction strategy")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Similarity threshold")
    max_citations: int = Field(10, ge=1, le=100, description="Maximum citations to extract")
    include_page_numbers: bool = Field(True, description="Include page numbers in citations")
    include_confidence_scores: bool = Field(True, description="Include confidence scores")
    cluster_similar_citations: bool = Field(True, description="Cluster similar citations")
    validate_citations: bool = Field(True, description="Validate extracted citations")
    merge_overlapping_citations: bool = Field(True, description="Merge overlapping citations")


class CitationLinkingOptionsRequest(BaseModel):
    """Citation linking options request schema."""
    create_bidirectional_links: bool = Field(True, description="Create bidirectional links")
    include_context_links: bool = Field(True, description="Include contextual links")
    link_similarity_threshold: float = Field(0.6, ge=0.0, le=1.0, description="Link similarity threshold")
    max_links_per_citation: int = Field(5, ge=1, le=20, description="Maximum links per citation")


class CitationAnalysisOptionsRequest(BaseModel):
    """Citation analysis options request schema."""
    include_quality_metrics: bool = Field(True, description="Include quality metrics")
    include_usage_patterns: bool = Field(True, description="Include usage patterns")
    include_source_analysis: bool = Field(True, description="Include source analysis")
    include_temporal_analysis: bool = Field(True, description="Include temporal analysis")
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Confidence threshold")


class CitationClusteringOptionsRequest(BaseModel):
    """Citation clustering options request schema."""
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Clustering similarity threshold")
    max_clusters: int = Field(10, ge=1, le=50, description="Maximum number of clusters")
    min_cluster_size: int = Field(2, ge=1, le=10, description="Minimum cluster size")
    clustering_method: ClusteringMethodEnum = Field(ClusteringMethodEnum.SEMANTIC_SIMILARITY, description="Clustering method")
    feature_weights: Dict[str, float] = Field(default_factory=dict, description="Feature weights for clustering")


class CitationValidationOptionsRequest(BaseModel):
    """Citation validation options request schema."""
    check_source_accuracy: bool = Field(True, description="Check source accuracy")
    check_text_similarity: bool = Field(True, description="Check text similarity")
    check_location_accuracy: bool = Field(True, description="Check location accuracy")
    similarity_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Validation similarity threshold")
    strict_validation: bool = Field(False, description="Use strict validation rules")


class BulkProcessingOptionsRequest(BaseModel):
    """Bulk processing options request schema."""
    parallel_processing: bool = Field(True, description="Enable parallel processing")
    batch_size: int = Field(10, ge=1, le=100, description="Batch size for processing")
    continue_on_error: bool = Field(True, description="Continue processing on errors")
    include_detailed_results: bool = Field(True, description="Include detailed results")


# Citation Extraction Schemas

class CitationExtractionRequest(BaseModel):
    """Citation extraction request schema."""
    response_text: str = Field(..., min_length=1, description="Response text to extract citations from")
    source_chunks: List[SourceChunkRequest] = Field(..., min_items=1, description="Source chunks for citation extraction")
    context_documents: List[str] = Field(..., description="Context document IDs")
    session_id: str = Field(..., description="Session ID")
    options: CitationProcessingOptionsRequest = Field(default_factory=CitationProcessingOptionsRequest, description="Processing options")

    @validator('response_text')
    def validate_response_text(cls, v):
        """Validate response text."""
        if not v or not v.strip():
            raise ValueError("Response text cannot be empty")
        return v.strip()


class CitationResponse(BaseModel):
    """Citation response schema."""
    id: str = Field(..., description="Citation ID")
    source_text: str = Field(..., description="Source text")
    cited_text: str = Field(..., description="Cited text")
    source_location: SourceLocationResponse = Field(..., description="Source location")
    confidence: CitationConfidenceResponse = Field(..., description="Citation confidence")
    citation_type: CitationTypeEnum = Field(..., description="Citation type")
    extraction_method: str = Field(..., description="Extraction method used")
    validation_status: ValidationStatusEnum = Field(..., description="Validation status")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class CitationClusterResponse(BaseModel):
    """Citation cluster response schema."""
    id: str = Field(..., description="Cluster ID")
    cluster_label: str = Field(..., description="Cluster label")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Cluster similarity score")
    citation_ids: List[str] = Field(..., description="Citation IDs in cluster")
    representative_citation_id: str = Field(..., description="Representative citation ID")
    cluster_metadata: Dict[str, Any] = Field(default_factory=dict, description="Cluster metadata")


class ExtractionStatisticsResponse(BaseModel):
    """Extraction statistics response schema."""
    total_citations_found: int = Field(..., ge=0, description="Total citations found")
    valid_citations: int = Field(..., ge=0, description="Valid citations")
    invalid_citations: int = Field(..., ge=0, description="Invalid citations")
    clusters_created: int = Field(..., ge=0, description="Clusters created")
    processing_time_ms: int = Field(..., ge=0, description="Processing time in milliseconds")
    extraction_method_used: str = Field(..., description="Extraction method used")
    confidence_distribution: Dict[str, int] = Field(default_factory=dict, description="Confidence distribution")


class CitationExtractionResponse(BaseModel):
    """Citation extraction response schema."""
    citations: List[CitationResponse] = Field(..., description="Extracted citations")
    citation_clusters: List[CitationClusterResponse] = Field(default_factory=list, description="Citation clusters")
    extraction_statistics: ExtractionStatisticsResponse = Field(..., description="Extraction statistics")
    processing_metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata")
    timestamp: str = Field(..., description="Response timestamp")


# Citation Linking Schemas

class CitationLinkingRequest(BaseModel):
    """Citation linking request schema."""
    citation_ids: List[str] = Field(..., min_items=1, description="Citation IDs to link")
    target_documents: List[str] = Field(..., description="Target document IDs")
    session_id: str = Field(..., description="Session ID")
    options: CitationLinkingOptionsRequest = Field(default_factory=CitationLinkingOptionsRequest, description="Linking options")


class CitationLinkResponse(BaseModel):
    """Citation link response schema."""
    id: str = Field(..., description="Link ID")
    citation_id: str = Field(..., description="Citation ID")
    source_location: SourceLocationResponse = Field(..., description="Source location")
    target_location: SourceLocationResponse = Field(..., description="Target location")
    link_type: LinkTypeEnum = Field(..., description="Link type")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Link confidence")
    navigation_url: Optional[str] = Field(None, description="Navigation URL")
    preview_text: Optional[str] = Field(None, description="Preview text")
    link_metadata: Dict[str, Any] = Field(default_factory=dict, description="Link metadata")


class LinkingStatisticsResponse(BaseModel):
    """Linking statistics response schema."""
    total_links_created: int = Field(..., ge=0, description="Total links created")
    successful_links: int = Field(..., ge=0, description="Successful links")
    failed_links: int = Field(..., ge=0, description="Failed links")
    bidirectional_links: int = Field(..., ge=0, description="Bidirectional links")
    processing_time_ms: int = Field(..., ge=0, description="Processing time in milliseconds")


class CitationLinkingResponse(BaseModel):
    """Citation linking response schema."""
    citation_links: List[CitationLinkResponse] = Field(..., description="Citation links")
    linking_statistics: LinkingStatisticsResponse = Field(..., description="Linking statistics")
    navigation_metadata: Dict[str, Any] = Field(default_factory=dict, description="Navigation metadata")
    timestamp: str = Field(..., description="Response timestamp")


# Citation Analysis Schemas

class CitationAnalysisRequest(BaseModel):
    """Citation analysis request schema."""
    session_id: str = Field(..., description="Session ID")
    document_ids: Optional[List[str]] = Field(None, description="Document IDs to analyze")
    time_range: Optional[Dict[str, str]] = Field(None, description="Time range for analysis")
    options: CitationAnalysisOptionsRequest = Field(default_factory=CitationAnalysisOptionsRequest, description="Analysis options")


class QualityMetricsResponse(BaseModel):
    """Quality metrics response schema."""
    average_confidence: float = Field(..., ge=0.0, le=1.0, description="Average confidence")
    citation_accuracy: float = Field(..., ge=0.0, le=1.0, description="Citation accuracy")
    source_diversity: float = Field(..., ge=0.0, le=1.0, description="Source diversity")
    citation_density: float = Field(..., ge=0.0, description="Citation density")
    validation_success_rate: float = Field(..., ge=0.0, le=1.0, description="Validation success rate")


class UsagePatternsResponse(BaseModel):
    """Usage patterns response schema."""
    most_cited_documents: List[Dict[str, Any]] = Field(default_factory=list, description="Most cited documents")
    citation_frequency_by_type: Dict[str, int] = Field(default_factory=dict, description="Citation frequency by type")
    temporal_patterns: Dict[str, Any] = Field(default_factory=dict, description="Temporal patterns")
    user_citation_preferences: Dict[str, Any] = Field(default_factory=dict, description="User citation preferences")


class SourceAnalysisResponse(BaseModel):
    """Source analysis response schema."""
    document_citation_counts: Dict[str, int] = Field(default_factory=dict, description="Document citation counts")
    page_citation_distribution: Dict[str, int] = Field(default_factory=dict, description="Page citation distribution")
    section_citation_patterns: Dict[str, Any] = Field(default_factory=dict, description="Section citation patterns")
    source_reliability_scores: Dict[str, float] = Field(default_factory=dict, description="Source reliability scores")


class CitationAnalysisResponse(BaseModel):
    """Citation analysis response schema."""
    quality_metrics: QualityMetricsResponse = Field(..., description="Quality metrics")
    usage_patterns: UsagePatternsResponse = Field(..., description="Usage patterns")
    source_analysis: SourceAnalysisResponse = Field(..., description="Source analysis")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    analysis_metadata: Dict[str, Any] = Field(default_factory=dict, description="Analysis metadata")
    timestamp: str = Field(..., description="Response timestamp")


# Citation Clustering Schemas

class CitationClusteringRequest(BaseModel):
    """Citation clustering request schema."""
    citation_ids: List[str] = Field(..., min_items=2, description="Citation IDs to cluster")
    options: CitationClusteringOptionsRequest = Field(default_factory=CitationClusteringOptionsRequest, description="Clustering options")


class ClusteringStatisticsResponse(BaseModel):
    """Clustering statistics response schema."""
    total_citations: int = Field(..., ge=0, description="Total citations")
    clusters_created: int = Field(..., ge=0, description="Clusters created")
    average_cluster_size: float = Field(..., ge=0.0, description="Average cluster size")
    silhouette_score: Optional[float] = Field(None, ge=-1.0, le=1.0, description="Silhouette score")
    processing_time_ms: int = Field(..., ge=0, description="Processing time in milliseconds")


class CitationClusteringResponse(BaseModel):
    """Citation clustering response schema."""
    citation_clusters: List[CitationClusterResponse] = Field(..., description="Citation clusters")
    clustering_statistics: ClusteringStatisticsResponse = Field(..., description="Clustering statistics")
    cluster_metadata: Dict[str, Any] = Field(default_factory=dict, description="Cluster metadata")
    timestamp: str = Field(..., description="Response timestamp")


# Citation Validation Schemas

class CitationValidationRequest(BaseModel):
    """Citation validation request schema."""
    citation_ids: List[str] = Field(..., min_items=1, description="Citation IDs to validate")
    options: CitationValidationOptionsRequest = Field(default_factory=CitationValidationOptionsRequest, description="Validation options")


class ValidationResultResponse(BaseModel):
    """Validation result response schema."""
    citation_id: str = Field(..., description="Citation ID")
    is_valid: bool = Field(..., description="Whether citation is valid")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Validation confidence")
    validation_errors: List[str] = Field(default_factory=list, description="Validation errors")
    accuracy_metrics: Dict[str, float] = Field(default_factory=dict, description="Accuracy metrics")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    validation_metadata: Dict[str, Any] = Field(default_factory=dict, description="Validation metadata")


class ValidationSummaryResponse(BaseModel):
    """Validation summary response schema."""
    total_citations: int = Field(..., ge=0, description="Total citations validated")
    valid_citations: int = Field(..., ge=0, description="Valid citations")
    invalid_citations: int = Field(..., ge=0, description="Invalid citations")
    average_confidence: float = Field(..., ge=0.0, le=1.0, description="Average confidence")
    common_errors: List[str] = Field(default_factory=list, description="Common validation errors")
    processing_time_ms: int = Field(..., ge=0, description="Processing time in milliseconds")


class CitationValidationResponse(BaseModel):
    """Citation validation response schema."""
    validation_results: List[ValidationResultResponse] = Field(..., description="Validation results")
    validation_summary: ValidationSummaryResponse = Field(..., description="Validation summary")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    timestamp: str = Field(..., description="Response timestamp")


# Bulk Processing Schemas

class BulkProcessingRequest(BaseModel):
    """Individual bulk processing request schema."""
    request_id: str = Field(..., description="Request ID")
    operation_type: str = Field(..., description="Operation type")
    request_data: Dict[str, Any] = Field(..., description="Request data")


class BulkCitationProcessingRequest(BaseModel):
    """Bulk citation processing request schema."""
    processing_requests: List[BulkProcessingRequest] = Field(..., min_items=1, description="Processing requests")
    options: BulkProcessingOptionsRequest = Field(default_factory=BulkProcessingOptionsRequest, description="Bulk processing options")


class ProcessingResultResponse(BaseModel):
    """Processing result response schema."""
    request_id: str = Field(..., description="Request ID")
    status: str = Field(..., description="Processing status")
    citations_processed: int = Field(..., ge=0, description="Citations processed")
    processing_time_ms: int = Field(..., ge=0, description="Processing time")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    result_data: Optional[Dict[str, Any]] = Field(None, description="Result data")


class BulkStatisticsResponse(BaseModel):
    """Bulk statistics response schema."""
    total_requests: int = Field(..., ge=0, description="Total requests")
    successful_requests: int = Field(..., ge=0, description="Successful requests")
    failed_requests: int = Field(..., ge=0, description="Failed requests")
    total_citations_processed: int = Field(..., ge=0, description="Total citations processed")
    total_processing_time_ms: int = Field(..., ge=0, description="Total processing time")
    average_processing_time_ms: float = Field(..., ge=0.0, description="Average processing time")


class BulkCitationProcessingResponse(BaseModel):
    """Bulk citation processing response schema."""
    processing_results: List[ProcessingResultResponse] = Field(..., description="Processing results")
    bulk_statistics: BulkStatisticsResponse = Field(..., description="Bulk statistics")
    processing_metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata")
    timestamp: str = Field(..., description="Response timestamp")


# Error Response Schemas

class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


class CitationValidationErrorResponse(BaseModel):
    """Citation validation error response schema."""
    error: str = Field("citation_validation_error", description="Error type")
    message: str = Field(..., description="Validation error message")
    citation_errors: List[Dict[str, str]] = Field(default_factory=list, description="Citation-specific errors")
    timestamp: str = Field(..., description="Error timestamp")


# Health Check Schemas

class CitationHealthResponse(BaseModel):
    """Citation service health check response schema."""
    status: str = Field(..., description="Overall health status")
    timestamp: str = Field(..., description="Health check timestamp")
    services: Dict[str, str] = Field(..., description="Individual service statuses")
    performance: Dict[str, Union[float, int]] = Field(..., description="Performance metrics")
    capabilities: Dict[str, Any] = Field(..., description="Service capabilities")
    uptime_seconds: Optional[int] = Field(None, description="Service uptime in seconds")
    version: Optional[str] = Field(None, description="Service version")


# Export all schemas for easy importing
__all__ = [
    # Enums
    "CitationTypeEnum",
    "ExtractionStrategyEnum",
    "ValidationStatusEnum",
    "LinkTypeEnum",
    "ClusteringMethodEnum",
    
    # Request schemas
    "SourceChunkRequest",
    "CitationProcessingOptionsRequest",
    "CitationLinkingOptionsRequest",
    "CitationAnalysisOptionsRequest",
    "CitationClusteringOptionsRequest",
    "CitationValidationOptionsRequest",
    "BulkProcessingOptionsRequest",
    "CitationExtractionRequest",
    "CitationLinkingRequest",
    "CitationAnalysisRequest",
    "CitationClusteringRequest",
    "CitationValidationRequest",
    "BulkProcessingRequest",
    "BulkCitationProcessingRequest",
    
    # Response schemas
    "SourceLocationResponse",
    "CitationConfidenceResponse",
    "CitationResponse",
    "CitationClusterResponse",
    "ExtractionStatisticsResponse",
    "CitationExtractionResponse",
    "CitationLinkResponse",
    "LinkingStatisticsResponse",
    "CitationLinkingResponse",
    "QualityMetricsResponse",
    "UsagePatternsResponse",
    "SourceAnalysisResponse",
    "CitationAnalysisResponse",
    "ClusteringStatisticsResponse",
    "CitationClusteringResponse",
    "ValidationResultResponse",
    "ValidationSummaryResponse",
    "CitationValidationResponse",
    "ProcessingResultResponse",
    "BulkStatisticsResponse",
    "BulkCitationProcessingResponse",
    "CitationHealthResponse",
    
    # Error schemas
    "ErrorResponse",
    "CitationValidationErrorResponse"
]
