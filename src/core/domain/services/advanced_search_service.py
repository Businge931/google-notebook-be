"""
Advanced Search Service Interface
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Set, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from ..value_objects import DocumentId, SessionId


class SearchType(Enum):
    """Types of search operations following Single Responsibility Principle."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    QUESTION_ANSWERING = "question_answering"
    CONCEPTUAL = "conceptual"
    TEMPORAL = "temporal"
    CROSS_REFERENCE = "cross_reference"


class SearchScope(Enum):
    """Search scope options following Single Responsibility Principle."""
    DOCUMENT = "document"
    COLLECTION = "collection"
    SESSION = "session"
    GLOBAL = "global"
    CONTEXTUAL = "contextual"


class RankingStrategy(Enum):
    """Ranking strategies for search results following Single Responsibility Principle."""
    RELEVANCE = "relevance"
    RECENCY = "recency"
    POPULARITY = "popularity"
    AUTHORITY = "authority"
    HYBRID_SCORE = "hybrid_score"
    PERSONALIZED = "personalized"


@dataclass
class SearchFilter:
    """
    Search filter criteria following Single Responsibility Principle.
    """
    document_ids: Optional[List[DocumentId]] = None
    document_types: Optional[List[str]] = None
    date_range: Optional[Tuple[datetime, datetime]] = None
    authors: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    languages: Optional[List[str]] = None
    content_types: Optional[List[str]] = None
    min_confidence: Optional[float] = None
    max_results: Optional[int] = None
    metadata_filters: Optional[Dict[str, Any]] = None


@dataclass
class SearchContext:
    """
    Search context information following Single Responsibility Principle.
    """
    session_id: Optional[SessionId] = None
    conversation_history: Optional[List[str]] = None
    user_preferences: Optional[Dict[str, Any]] = None
    previous_searches: Optional[List[str]] = None
    current_document_focus: Optional[DocumentId] = None
    temporal_context: Optional[datetime] = None
    domain_context: Optional[str] = None


@dataclass
class SearchResult:
    """
    Enhanced search result following Single Responsibility Principle.
    """
    id: str
    document_id: DocumentId
    document_title: str
    document_filename: str
    chunk_id: Optional[str]
    content: str
    snippet: str
    relevance_score: float
    confidence_score: float
    page_number: Optional[int]
    section_title: Optional[str]
    start_position: int
    end_position: int
    highlighted_content: str
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    search_type_used: Optional[SearchType] = None
    ranking_factors: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SearchCluster:
    """
    Grouped search results following Single Responsibility Principle.
    """
    id: str
    cluster_label: str
    results: List[SearchResult]
    cluster_score: float
    representative_result: SearchResult
    cluster_metadata: Optional[Dict[str, Any]] = None


@dataclass
class SearchSuggestion:
    """
    Search query suggestion following Single Responsibility Principle.
    """
    suggestion: str
    confidence: float
    suggestion_type: str
    context: Optional[str] = None
    expected_results: Optional[int] = None


@dataclass
class AdvancedSearchRequest:
    """
    Advanced search request following Single Responsibility Principle.
    """
    query: str
    search_type: SearchType = SearchType.HYBRID
    search_scope: SearchScope = SearchScope.GLOBAL
    ranking_strategy: RankingStrategy = RankingStrategy.HYBRID_SCORE
    filters: Optional[SearchFilter] = None
    context: Optional[SearchContext] = None
    max_results: int = 20
    include_snippets: bool = True
    include_highlights: bool = True
    cluster_results: bool = False
    generate_suggestions: bool = False
    explain_ranking: bool = False


@dataclass
class AdvancedSearchResponse:
    """
    Advanced search response following Single Responsibility Principle.
    """
    results: List[SearchResult]
    clusters: List[SearchCluster]
    suggestions: List[SearchSuggestion]
    total_results: int
    search_time_ms: int
    query_analysis: Dict[str, Any]
    ranking_explanation: Optional[Dict[str, Any]] = None
    search_metadata: Optional[Dict[str, Any]] = None


class SemanticSearchEngine(ABC):
    """
    Abstract interface for semantic search operations.
    
    Handles vector-based semantic search following
    Interface Segregation Principle.
    """
    
    @abstractmethod
    async def semantic_search(
        self,
        query: str,
        filters: Optional[SearchFilter] = None,
        max_results: int = 20
    ) -> List[SearchResult]:
        """
        Perform semantic search using vector similarity.
        
        Args:
            query: Search query
            filters: Search filters
            max_results: Maximum results to return
            
        Returns:
            Semantic search results
            
        Raises:
            SemanticSearchError: If search fails
        """
        pass
    
    @abstractmethod
    async def conceptual_search(
        self,
        concepts: List[str],
        filters: Optional[SearchFilter] = None,
        max_results: int = 20
    ) -> List[SearchResult]:
        """
        Search for conceptually related content.
        
        Args:
            concepts: List of concepts to search for
            filters: Search filters
            max_results: Maximum results to return
            
        Returns:
            Conceptual search results
            
        Raises:
            ConceptualSearchError: If search fails
        """
        pass
    
    @abstractmethod
    async def find_similar_content(
        self,
        reference_content: str,
        similarity_threshold: float = 0.7,
        max_results: int = 10
    ) -> List[SearchResult]:
        """
        Find content similar to reference text.
        
        Args:
            reference_content: Reference text
            similarity_threshold: Minimum similarity score
            max_results: Maximum results to return
            
        Returns:
            Similar content results
            
        Raises:
            SimilaritySearchError: If search fails
        """
        pass


class KeywordSearchEngine(ABC):
    """
    Abstract interface for keyword-based search operations.
    
    Handles traditional text search following
    Interface Segregation Principle.
    """
    
    @abstractmethod
    async def keyword_search(
        self,
        query: str,
        filters: Optional[SearchFilter] = None,
        max_results: int = 20
    ) -> List[SearchResult]:
        """
        Perform keyword-based search.
        
        Args:
            query: Search query
            filters: Search filters
            max_results: Maximum results to return
            
        Returns:
            Keyword search results
            
        Raises:
            KeywordSearchError: If search fails
        """
        pass
    
    @abstractmethod
    async def phrase_search(
        self,
        phrase: str,
        exact_match: bool = True,
        filters: Optional[SearchFilter] = None
    ) -> List[SearchResult]:
        """
        Search for exact phrases or near matches.
        
        Args:
            phrase: Phrase to search for
            exact_match: Whether to require exact matches
            filters: Search filters
            
        Returns:
            Phrase search results
            
        Raises:
            PhraseSearchError: If search fails
        """
        pass
    
    @abstractmethod
    async def boolean_search(
        self,
        boolean_query: str,
        filters: Optional[SearchFilter] = None
    ) -> List[SearchResult]:
        """
        Perform boolean search with AND, OR, NOT operators.
        
        Args:
            boolean_query: Boolean search expression
            filters: Search filters
            
        Returns:
            Boolean search results
            
        Raises:
            BooleanSearchError: If search fails
        """
        pass


class QuestionAnsweringEngine(ABC):
    """
    Abstract interface for question-answering search.
    
    Handles QA-specific search operations following
    Interface Segregation Principle.
    """
    
    @abstractmethod
    async def answer_question(
        self,
        question: str,
        context_documents: Optional[List[DocumentId]] = None,
        confidence_threshold: float = 0.5
    ) -> List[SearchResult]:
        """
        Find answers to specific questions.
        
        Args:
            question: Question to answer
            context_documents: Documents to search within
            confidence_threshold: Minimum confidence for answers
            
        Returns:
            Question-answering results
            
        Raises:
            QuestionAnsweringError: If QA search fails
        """
        pass
    
    @abstractmethod
    async def extract_facts(
        self,
        query: str,
        fact_types: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Extract factual information related to query.
        
        Args:
            query: Query for fact extraction
            fact_types: Types of facts to extract
            
        Returns:
            Factual search results
            
        Raises:
            FactExtractionError: If fact extraction fails
        """
        pass


class SearchRanker(ABC):
    """
    Abstract interface for search result ranking.
    
    Handles result ranking and scoring following
    Interface Segregation Principle.
    """
    
    @abstractmethod
    async def rank_results(
        self,
        results: List[SearchResult],
        ranking_strategy: RankingStrategy,
        context: Optional[SearchContext] = None
    ) -> List[SearchResult]:
        """
        Rank search results using specified strategy.
        
        Args:
            results: Results to rank
            ranking_strategy: Ranking approach
            context: Search context for personalization
            
        Returns:
            Ranked search results
            
        Raises:
            RankingError: If ranking fails
        """
        pass
    
    @abstractmethod
    async def calculate_relevance_score(
        self,
        result: SearchResult,
        query: str,
        context: Optional[SearchContext] = None
    ) -> float:
        """
        Calculate relevance score for a result.
        
        Args:
            result: Search result
            query: Original query
            context: Search context
            
        Returns:
            Relevance score (0.0 to 1.0)
            
        Raises:
            ScoringError: If scoring fails
        """
        pass
    
    @abstractmethod
    async def explain_ranking(
        self,
        result: SearchResult,
        query: str,
        ranking_factors: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Explain why a result was ranked at its position.
        
        Args:
            result: Search result
            query: Original query
            ranking_factors: Factors used in ranking
            
        Returns:
            Ranking explanation
            
        Raises:
            ExplanationError: If explanation generation fails
        """
        pass


class SearchClusterer(ABC):
    """
    Abstract interface for search result clustering.
    
    Groups related search results following
    Interface Segregation Principle.
    """
    
    @abstractmethod
    async def cluster_results(
        self,
        results: List[SearchResult],
        clustering_strategy: str = "semantic",
        max_clusters: int = 5
    ) -> List[SearchCluster]:
        """
        Group search results into clusters.
        
        Args:
            results: Results to cluster
            clustering_strategy: Clustering approach
            max_clusters: Maximum number of clusters
            
        Returns:
            Result clusters
            
        Raises:
            ClusteringError: If clustering fails
        """
        pass
    
    @abstractmethod
    async def generate_cluster_labels(
        self,
        cluster: SearchCluster
    ) -> str:
        """
        Generate descriptive label for a cluster.
        
        Args:
            cluster: Cluster to label
            
        Returns:
            Cluster label
            
        Raises:
            LabelGenerationError: If label generation fails
        """
        pass


class AdvancedSearchService(ABC):
    """
    Main advanced search service interface.
    
    Orchestrates all search operations following
    Interface Segregation Principle.
    """
    
    @abstractmethod
    async def advanced_search(
        self,
        request: AdvancedSearchRequest
    ) -> AdvancedSearchResponse:
        """
        Perform advanced search with multiple strategies.
        
        Args:
            request: Advanced search request
            
        Returns:
            Advanced search response
            
        Raises:
            AdvancedSearchError: If search fails
        """
        pass
    
    @abstractmethod
    async def hybrid_search(
        self,
        query: str,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        filters: Optional[SearchFilter] = None
    ) -> List[SearchResult]:
        """
        Combine semantic and keyword search results.
        
        Args:
            query: Search query
            semantic_weight: Weight for semantic results
            keyword_weight: Weight for keyword results
            filters: Search filters
            
        Returns:
            Hybrid search results
            
        Raises:
            HybridSearchError: If hybrid search fails
        """
        pass
    
    @abstractmethod
    async def contextual_search(
        self,
        query: str,
        context: SearchContext,
        adaptation_strength: float = 0.5
    ) -> List[SearchResult]:
        """
        Perform context-aware search.
        
        Args:
            query: Search query
            context: Search context
            adaptation_strength: How much to adapt to context
            
        Returns:
            Contextual search results
            
        Raises:
            ContextualSearchError: If contextual search fails
        """
        pass
    
    @abstractmethod
    async def generate_search_suggestions(
        self,
        partial_query: str,
        context: Optional[SearchContext] = None,
        max_suggestions: int = 5
    ) -> List[SearchSuggestion]:
        """
        Generate search query suggestions.
        
        Args:
            partial_query: Partial or complete query
            context: Search context
            max_suggestions: Maximum suggestions to return
            
        Returns:
            Search suggestions
            
        Raises:
            SuggestionError: If suggestion generation fails
        """
        pass
    
    @abstractmethod
    async def get_search_analytics(
        self,
        session_id: Optional[SessionId] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """
        Get search analytics and metrics.
        
        Args:
            session_id: Optional session filter
            time_range: Optional time range filter
            
        Returns:
            Search analytics
            
        Raises:
            AnalyticsError: If analytics retrieval fails
        """
        pass


class AdvancedSearchServiceFactory(ABC):
    """
    Abstract factory for creating advanced search service instances.
    
    Follows Open/Closed Principle for extensibility.
    """
    
    @abstractmethod
    def create_semantic_search_engine(
        self,
        config: Dict[str, Any]
    ) -> SemanticSearchEngine:
        """
        Create semantic search engine instance.
        
        Args:
            config: Engine configuration
            
        Returns:
            Configured semantic search engine
        """
        pass
    
    @abstractmethod
    def create_keyword_search_engine(
        self,
        config: Dict[str, Any]
    ) -> KeywordSearchEngine:
        """
        Create keyword search engine instance.
        
        Args:
            config: Engine configuration
            
        Returns:
            Configured keyword search engine
        """
        pass
    
    @abstractmethod
    def create_qa_engine(
        self,
        config: Dict[str, Any]
    ) -> QuestionAnsweringEngine:
        """
        Create question-answering engine instance.
        
        Args:
            config: Engine configuration
            
        Returns:
            Configured QA engine
        """
        pass
    
    @abstractmethod
    def create_search_ranker(
        self,
        config: Dict[str, Any]
    ) -> SearchRanker:
        """
        Create search ranker instance.
        
        Args:
            config: Ranker configuration
            
        Returns:
            Configured search ranker
        """
        pass
    
    @abstractmethod
    def create_search_clusterer(
        self,
        config: Dict[str, Any]
    ) -> SearchClusterer:
        """
        Create search clusterer instance.
        
        Args:
            config: Clusterer configuration
            
        Returns:
            Configured search clusterer
        """
        pass
    
    @abstractmethod
    def create_advanced_search_service(
        self,
        semantic_engine: SemanticSearchEngine,
        keyword_engine: KeywordSearchEngine,
        qa_engine: QuestionAnsweringEngine,
        ranker: SearchRanker,
        clusterer: SearchClusterer,
        config: Dict[str, Any]
    ) -> AdvancedSearchService:
        """
        Create complete advanced search service instance.
        
        Args:
            semantic_engine: Semantic search engine
            keyword_engine: Keyword search engine
            qa_engine: Question-answering engine
            ranker: Search ranker
            clusterer: Search clusterer
            config: Service configuration
            
        Returns:
            Configured advanced search service
        """
        pass
