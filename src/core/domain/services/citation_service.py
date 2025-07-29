"""
Advanced Citation Service Interface

"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Set, Tuple
from dataclasses import dataclass
from enum import Enum

from ..value_objects import DocumentId, SessionId
from ..entities import Message


class CitationType(Enum):
    """Types of citations following Single Responsibility Principle."""
    DIRECT_QUOTE = "direct_quote"
    PARAPHRASE = "paraphrase"
    REFERENCE = "reference"
    SUPPORTING_EVIDENCE = "supporting_evidence"
    CONTRADICTORY_EVIDENCE = "contradictory_evidence"


class CitationConfidence(Enum):
    """Citation confidence levels following Single Responsibility Principle."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class CitationProcessingOptions:
    """Options for citation processing following Single Responsibility Principle."""
    max_citations: int = 10
    min_confidence: CitationConfidence = CitationConfidence.MEDIUM
    include_context: bool = True
    context_length: int = 100
    enable_deduplication: bool = True
    similarity_threshold: float = 0.8


@dataclass
class CitationSpan:
    """
    Represents a span of text with citation following Single Responsibility Principle.
    """
    start_position: int
    end_position: int
    text: str
    citation_type: CitationType
    confidence: CitationConfidence
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SourceLocation:
    """
    Represents precise source location following Single Responsibility Principle.
    """
    document_id: DocumentId
    document_title: str
    page_number: Optional[int]
    paragraph_number: Optional[int]
    line_number: Optional[int]
    chunk_id: Optional[str]
    start_char: Optional[int]
    end_char: Optional[int]
    section_title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Citation:
    """
    Enhanced citation with precise location and context following Single Responsibility Principle.
    """
    id: str
    citation_spans: List[CitationSpan]
    source_location: SourceLocation
    source_text: str
    similarity_score: float
    relevance_score: float
    citation_type: CitationType
    confidence: CitationConfidence
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    generated_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CitationCluster:
    """
    Groups related citations following Single Responsibility Principle.
    """
    id: str
    citations: List[Citation]
    cluster_topic: str
    cluster_confidence: float
    representative_citation: Citation
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CitationExtractionRequest:
    """
    Request for citation extraction following Single Responsibility Principle.
    """
    response_text: str
    source_chunks: List[Dict[str, Any]]
    context_documents: List[DocumentId]
    extraction_strategy: str = "semantic"
    min_confidence: float = 0.7
    max_citations: int = 10
    include_context: bool = True
    cluster_citations: bool = True


@dataclass
class CitationExtractionResponse:
    """
    Response from citation extraction following Single Responsibility Principle.
    """
    citations: List[Citation]
    citation_clusters: List[CitationCluster]
    extraction_metadata: Dict[str, Any]
    processing_time_ms: int
    total_citations_found: int
    confidence_distribution: Dict[str, int]


class CitationExtractor(ABC):
    """
    Abstract interface for citation extraction.
    
    Defines the contract for extracting citations from text following
    Interface Segregation Principle.
    """
    
    @abstractmethod
    async def extract_citations(
        self,
        request: CitationExtractionRequest
    ) -> CitationExtractionResponse:
        """
        Extract citations from response text.
        
        Args:
            request: Citation extraction request
            
        Returns:
            Citation extraction response
            
        Raises:
            CitationExtractionError: If extraction fails
        """
        pass
    
    @abstractmethod
    async def validate_citations(
        self,
        citations: List[Citation],
        source_documents: List[DocumentId]
    ) -> List[Citation]:
        """
        Validate and filter citations.
        
        Args:
            citations: Citations to validate
            source_documents: Available source documents
            
        Returns:
            Validated citations
            
        Raises:
            CitationValidationError: If validation fails
        """
        pass
    
    @abstractmethod
    async def cluster_citations(
        self,
        citations: List[Citation],
        clustering_strategy: str = "semantic"
    ) -> List[CitationCluster]:
        """
        Group related citations into clusters.
        
        Args:
            citations: Citations to cluster
            clustering_strategy: Clustering approach
            
        Returns:
            Citation clusters
            
        Raises:
            CitationClusteringError: If clustering fails
        """
        pass


class CitationLinker(ABC):
    """
    Abstract interface for citation linking and navigation.
    
    Handles linking citations to source documents following
    Interface Segregation Principle.
    """
    
    @abstractmethod
    async def create_citation_links(
        self,
        citations: List[Citation],
        session_id: SessionId
    ) -> Dict[str, str]:
        """
        Create navigable links for citations.
        
        Args:
            citations: Citations to link
            session_id: Chat session identifier
            
        Returns:
            Citation ID to link URL mapping
            
        Raises:
            CitationLinkingError: If linking fails
        """
        pass
    
    @abstractmethod
    async def resolve_citation_location(
        self,
        citation_id: str,
        document_id: DocumentId
    ) -> Optional[SourceLocation]:
        """
        Resolve precise location of a citation.
        
        Args:
            citation_id: Citation identifier
            document_id: Document identifier
            
        Returns:
            Precise source location or None if not found
            
        Raises:
            CitationResolutionError: If resolution fails
        """
        pass
    
    @abstractmethod
    async def get_citation_context(
        self,
        citation: Citation,
        context_window: int = 500
    ) -> Dict[str, str]:
        """
        Get extended context around a citation.
        
        Args:
            citation: Citation to get context for
            context_window: Context window size in characters
            
        Returns:
            Context information (before, after, full_paragraph)
            
        Raises:
            CitationContextError: If context retrieval fails
        """
        pass


class CitationAnalyzer(ABC):
    """
    Abstract interface for citation analysis and quality assessment.
    
    Analyzes citation quality and relationships following
    Interface Segregation Principle.
    """
    
    @abstractmethod
    async def analyze_citation_quality(
        self,
        citations: List[Citation],
        response_text: str
    ) -> Dict[str, Any]:
        """
        Analyze quality and coverage of citations.
        
        Args:
            citations: Citations to analyze
            response_text: Generated response text
            
        Returns:
            Quality analysis results
            
        Raises:
            CitationAnalysisError: If analysis fails
        """
        pass
    
    @abstractmethod
    async def detect_citation_conflicts(
        self,
        citations: List[Citation]
    ) -> List[Dict[str, Any]]:
        """
        Detect conflicting or contradictory citations.
        
        Args:
            citations: Citations to analyze
            
        Returns:
            List of detected conflicts
            
        Raises:
            CitationConflictError: If conflict detection fails
        """
        pass
    
    @abstractmethod
    async def calculate_citation_coverage(
        self,
        citations: List[Citation],
        response_text: str
    ) -> Dict[str, float]:
        """
        Calculate how well citations cover the response.
        
        Args:
            citations: Citations to analyze
            response_text: Generated response text
            
        Returns:
            Coverage metrics
            
        Raises:
            CitationCoverageError: If coverage calculation fails
        """
        pass


class CitationService(ABC):
    """
    Main citation service interface combining all citation operations.
    
    Orchestrates citation extraction, linking, and analysis following
    Interface Segregation Principle.
    """
    
    @abstractmethod
    async def process_citations(
        self,
        response_text: str,
        source_chunks: List[Dict[str, Any]],
        context_documents: List[DocumentId],
        session_id: SessionId,
        options: Optional[Dict[str, Any]] = None
    ) -> CitationExtractionResponse:
        """
        Complete citation processing pipeline.
        
        Args:
            response_text: Generated response text
            source_chunks: Source chunks used for generation
            context_documents: Available documents
            session_id: Chat session identifier
            options: Processing options
            
        Returns:
            Complete citation processing results
            
        Raises:
            CitationProcessingError: If processing fails
        """
        pass
    
    @abstractmethod
    async def enhance_citations(
        self,
        citations: List[Citation],
        enhancement_options: Dict[str, Any]
    ) -> List[Citation]:
        """
        Enhance citations with additional metadata and context.
        
        Args:
            citations: Citations to enhance
            enhancement_options: Enhancement configuration
            
        Returns:
            Enhanced citations
            
        Raises:
            CitationEnhancementError: If enhancement fails
        """
        pass
    
    @abstractmethod
    async def get_citation_analytics(
        self,
        session_id: SessionId,
        time_range: Optional[Tuple[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Get citation analytics for a session.
        
        Args:
            session_id: Chat session identifier
            time_range: Optional time range filter
            
        Returns:
            Citation analytics and metrics
            
        Raises:
            CitationAnalyticsError: If analytics retrieval fails
        """
        pass


class CitationServiceFactory(ABC):
    """
    Abstract factory for creating citation service instances.
    
    Follows Open/Closed Principle for extensibility.
    """
    
    @abstractmethod
    def create_citation_extractor(
        self,
        config: Dict[str, Any]
    ) -> CitationExtractor:
        """
        Create citation extractor instance.
        
        Args:
            config: Extractor configuration
            
        Returns:
            Configured citation extractor
        """
        pass
    
    @abstractmethod
    def create_citation_linker(
        self,
        config: Dict[str, Any]
    ) -> CitationLinker:
        """
        Create citation linker instance.
        
        Args:
            config: Linker configuration
            
        Returns:
            Configured citation linker
        """
        pass
    
    @abstractmethod
    def create_citation_analyzer(
        self,
        config: Dict[str, Any]
    ) -> CitationAnalyzer:
        """
        Create citation analyzer instance.
        
        Args:
            config: Analyzer configuration
            
        Returns:
            Configured citation analyzer
        """
        pass
    
    @abstractmethod
    def create_citation_service(
        self,
        extractor: CitationExtractor,
        linker: CitationLinker,
        analyzer: CitationAnalyzer,
        config: Dict[str, Any]
    ) -> CitationService:
        """
        Create complete citation service instance.
        
        Args:
            extractor: Citation extractor
            linker: Citation linker
            analyzer: Citation analyzer
            config: Service configuration
            
        Returns:
            Configured citation service
        """
        pass
