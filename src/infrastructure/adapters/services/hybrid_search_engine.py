"""
Hybrid Search Engine Implementation
"""
import re
import logging
from typing import List, Optional, Dict, Any, Set, Tuple
from datetime import datetime
import asyncio
from collections import defaultdict

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from src.core.domain.services.advanced_search_service import (
    SemanticSearchEngine,
    KeywordSearchEngine,
    QuestionAnsweringEngine,
    SearchRanker,
    SearchClusterer,
    AdvancedSearchService,
    SearchResult,
    SearchCluster,
    SearchSuggestion,
    AdvancedSearchRequest,
    AdvancedSearchResponse,
    SearchFilter,
    SearchContext,
    SearchType,
    RankingStrategy
)
from ....core.application.use_cases.similarity_search_use_case import (
    SimilaritySearchUseCase,
    SimilaritySearchRequest
)
from src.core.domain.value_objects import DocumentId, SessionId
from src.shared.exceptions import (
    SemanticSearchError,
    KeywordSearchError,
    QuestionAnsweringError,
    RankingError,
    ClusteringError,
    AdvancedSearchError
)


class HybridSemanticSearchEngine(SemanticSearchEngine):
    """
    Semantic search engine using vector similarity.
    
    Implements semantic search operations following Single Responsibility Principle.
    """
    
    def __init__(
        self,
        similarity_search_use_case: SimilaritySearchUseCase,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize semantic search engine.
        
        Args:
            similarity_search_use_case: Similarity search use case
            similarity_threshold: Minimum similarity threshold
        """
        self._similarity_search_use_case = similarity_search_use_case
        self._similarity_threshold = similarity_threshold
        self._logger = logging.getLogger(__name__)
    
    async def semantic_search(
        self,
        query: str,
        filters: Optional[SearchFilter] = None,
        max_results: int = 20
    ) -> List[SearchResult]:
        """
        Perform enhanced semantic search with contextual filtering and deduplication.
        
        Args:
            query: Search query
            filters: Search filters
            max_results: Maximum results to return
            
        Returns:
            Semantic search results
            
        Raises:
            SemanticSearchError: If search fails
        """
        try:
            self._logger.info(f"Performing enhanced semantic search for: {query}")
            
            # Extract contextual information from query
            query_context = self._extract_query_context(query)
            
            # Convert filters to similarity search request with higher limit for filtering
            search_request = SimilaritySearchRequest(
                query_text=query,
                document_ids=filters.document_ids if filters else None,
                limit=max_results * 3,  # Get more results for better filtering
                similarity_threshold=0.18,  # Adjusted threshold for improved text processing
                include_metadata=True
            )
            
            # Execute similarity search
            search_response = await self._similarity_search_use_case.search_similar_chunks(
                search_request
            )
            
            # Convert to SearchResult objects
            raw_results = []
            for result in search_response.results:
                search_result = SearchResult(
                    id=f"semantic_{result.chunk_id}",
                    document_id=DocumentId(result.document_id),
                    document_title=result.document_title,
                    document_filename=result.document_filename,
                    chunk_id=result.chunk_id,
                    content=result.text_content,
                    snippet=self._create_snippet(result.text_content, query),
                    relevance_score=result.similarity_score,
                    confidence_score=result.similarity_score,
                    page_number=result.page_number,
                    start_position=result.start_position,
                    end_position=result.end_position,
                    highlighted_content=self._highlight_content(result.text_content, query),
                    search_type_used=SearchType.SEMANTIC,
                    ranking_factors={"similarity_score": result.similarity_score},
                    metadata=result.metadata
                )
                raw_results.append(search_result)
            
            # Apply advanced filtering and deduplication
            filtered_results = self._apply_contextual_filtering(raw_results, query_context)
            deduplicated_results = self._deduplicate_results(filtered_results)
            final_results = deduplicated_results[:max_results]
            
            self._logger.info(f"Found {len(raw_results)} raw results, filtered to {len(filtered_results)}, deduplicated to {len(deduplicated_results)}, final: {len(final_results)}")
            return final_results
            
        except Exception as e:
            self._logger.error(f"Semantic search failed: {e}")
            raise SemanticSearchError(f"Semantic search failed: {str(e)}")
    
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
        try:
            # Combine concepts into a single query
            combined_query = " ".join(concepts)
            
            # Perform semantic search with expanded query
            results = await self.semantic_search(
                query=combined_query,
                filters=filters,
                max_results=max_results
            )
            
            # Enhance results with concept matching scores
            for result in results:
                concept_scores = {}
                for concept in concepts:
                    concept_scores[concept] = self._calculate_concept_score(
                        result.content, concept
                    )
                
                result.ranking_factors.update(concept_scores)
                result.search_type_used = SearchType.CONCEPTUAL
            
            return results
            
        except Exception as e:
            self._logger.error(f"Conceptual search failed: {e}")
            raise SemanticSearchError(f"Conceptual search failed: {str(e)}")
    
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
        try:
            # Use reference content as query
            results = await self.semantic_search(
                query=reference_content,
                max_results=max_results
            )
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in results
                if result.relevance_score >= similarity_threshold
            ]
            
            return filtered_results
            
        except Exception as e:
            self._logger.error(f"Similar content search failed: {e}")
            raise SemanticSearchError(f"Similar content search failed: {str(e)}")
    
    def _create_snippet(self, content: str, query: str, snippet_length: int = 200) -> str:
        """Create a snippet from content highlighting query terms."""
        words = content.split()
        query_words = query.lower().split()
        
        # Find best position for snippet
        best_start = 0
        max_matches = 0
        
        for i in range(len(words) - snippet_length // 10):
            window = words[i:i + snippet_length // 10]
            window_text = " ".join(window).lower()
            matches = sum(1 for qw in query_words if qw in window_text)
            
            if matches > max_matches:
                max_matches = matches
                best_start = i
        
        # Create snippet
        snippet_words = words[best_start:best_start + snippet_length // 10]
        snippet = " ".join(snippet_words)
        
        if len(snippet) > snippet_length:
            snippet = snippet[:snippet_length] + "..."
        
        return snippet
    
    def _highlight_content(self, content: str, query: str) -> str:
        """Highlight query terms in content."""
        query_words = query.split()
        highlighted = content
        
        for word in query_words:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            highlighted = pattern.sub(f"<mark>{word}</mark>", highlighted)
        
        return highlighted
    
    def _calculate_concept_score(self, content: str, concept: str) -> float:
        """Calculate how well content matches a concept."""
        content_words = set(content.lower().split())
        concept_words = set(concept.lower().split())
        
        if not concept_words:
            return 0.0
        
        matches = len(content_words.intersection(concept_words))
        return matches / len(concept_words)
    
    def _extract_query_context(self, query: str) -> dict:
        """Extract contextual information from query (dates, entities, etc.)."""
        import re
        
        context = {
            'years': [],
            'entities': [],
            'keywords': [],
            'question_type': 'general'
        }
        
        # Extract years (4-digit numbers)
        years = re.findall(r'\b(19|20)\d{2}\b', query)
        context['years'] = [int(year) for year in years]
        
        # Debug logging
        self._logger.info(f"Query context extracted: {context}")
        
        # Extract company/organization indicators
        company_patterns = [
            r'\b([A-Z][A-Z\s&.]+(?:LTD|LLC|INC|CORP|COMPANY|CO\.?)?)\b',
            r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            context['entities'].extend(matches)
        
        # Determine question type
        if any(word in query.lower() for word in ['which', 'what', 'where', 'when', 'who']):
            context['question_type'] = 'specific'
        elif any(word in query.lower() for word in ['company', 'work', 'job', 'position']):
            context['question_type'] = 'employment'
        
        # Extract key terms
        important_words = ['company', 'work', 'job', 'position', 'role', 'experience', 'year']
        context['keywords'] = [word for word in important_words if word in query.lower()]
        
        return context
    
    def _apply_contextual_filtering(self, results: List[SearchResult], context: dict) -> List[SearchResult]:
        """Filter results based on query context (dates, entities, etc.)."""
        if not results:
            return results
        
        filtered_results = []
        
        self._logger.info(f"Starting contextual filtering with {len(results)} results")
        
        for i, result in enumerate(results):
            content_lower = result.content.lower()
            relevance_boost = 0.0
            should_skip = False
            
            # Filter by year if specified in query
            if context['years']:
                year_found = False
                for year in context['years']:
                    if str(year) in result.content:
                        year_found = True
                        relevance_boost += 0.4  # Boost relevance for year match
                        self._logger.info(f"Result {i}: Found matching year {year} in content")
                        break
                
                # If year specified but not found, skip this result entirely
                if not year_found:
                    # Check if result contains other years that don't match
                    other_years = re.findall(r'\b(19|20)\d{2}\b', result.content)
                    self._logger.info(f"Result {i}: No matching year found. Other years in content: {other_years}")
                    if other_years and all(int(year) not in context['years'] for year in other_years):
                        self._logger.info(f"Result {i}: Skipping due to non-matching years")
                        should_skip = True
            
            if should_skip:
                continue
            
            # Boost relevance for entity matches
            if context['entities']:
                for entity in context['entities']:
                    if entity.lower() in content_lower:
                        relevance_boost += 0.2
            
            # Boost relevance for keyword matches
            for keyword in context['keywords']:
                if keyword in content_lower:
                    relevance_boost += 0.1
            
            # Apply relevance boost
            result.relevance_score += relevance_boost
            result.confidence_score += relevance_boost
            
            # Only include results that meet minimum relevance after filtering
            if result.relevance_score >= 0.15:  # Adjusted minimum threshold
                filtered_results.append(result)
        
        # Sort by relevance score (descending)
        filtered_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return filtered_results
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate or very similar results with improved content comparison."""
        if not results:
            return results
        
        deduplicated = []
        seen_content_hashes = set()
        
        for result in results:
            # Create a content hash for exact duplicate detection
            content_hash = hash(' '.join(result.content.lower().split()))
            
            # Check for exact content duplicates first
            if content_hash in seen_content_hashes:
                continue
            
            # Check for high similarity with existing results
            is_duplicate = False
            for existing in deduplicated:
                # Compare full content similarity, not just first 20 words
                words1 = set(result.content.lower().split())
                words2 = set(existing.content.lower().split())
                
                if words1 and words2:
                    intersection = len(words1.intersection(words2))
                    union = len(words1.union(words2))
                    similarity = intersection / union if union > 0 else 0
                    
                    # If similarity > 70%, consider it a duplicate (lowered threshold)
                    if similarity > 0.7:
                        # Keep the result with higher relevance score
                        if result.relevance_score > existing.relevance_score:
                            # Replace existing with current result
                            deduplicated.remove(existing)
                            break
                        else:
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                deduplicated.append(result)
                seen_content_hashes.add(content_hash)
        
        return deduplicated


class HybridKeywordSearchEngine(KeywordSearchEngine):
    """
    Keyword search engine using text matching.
    
    Implements keyword search operations following Single Responsibility Principle.
    """
    
    def __init__(self, enable_fuzzy_matching: bool = True):
        """
        Initialize keyword search engine.
        
        Args:
            enable_fuzzy_matching: Whether to enable fuzzy matching
        """
        self._enable_fuzzy_matching = enable_fuzzy_matching
        self._logger = logging.getLogger(__name__)
        
        if SKLEARN_AVAILABLE:
            self._vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
        else:
            self._vectorizer = None
    
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
        try:
            self._logger.info(f"Performing keyword search for: {query}")
            
            # This would typically interface with a document repository
            # For now, we'll simulate keyword search results
            results = await self._simulate_keyword_search(query, filters, max_results)
            
            self._logger.info(f"Found {len(results)} keyword search results")
            return results
            
        except Exception as e:
            self._logger.error(f"Keyword search failed: {e}")
            raise KeywordSearchError(f"Keyword search failed: {str(e)}")
    
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
        try:
            # Simulate phrase search
            results = await self._simulate_phrase_search(phrase, exact_match, filters)
            return results
            
        except Exception as e:
            self._logger.error(f"Phrase search failed: {e}")
            raise KeywordSearchError(f"Phrase search failed: {str(e)}")
    
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
        try:
            # Parse boolean query
            parsed_query = self._parse_boolean_query(boolean_query)
            
            # Simulate boolean search
            results = await self._simulate_boolean_search(parsed_query, filters)
            return results
            
        except Exception as e:
            self._logger.error(f"Boolean search failed: {e}")
            raise KeywordSearchError(f"Boolean search failed: {str(e)}")
    
    async def _simulate_keyword_search(
        self,
        query: str,
        filters: Optional[SearchFilter],
        max_results: int
    ) -> List[SearchResult]:
        """Simulate keyword search results."""
        # This would be replaced with actual document repository search
        results = []
        
        # Create mock results for demonstration
        for i in range(min(max_results, 5)):
            result = SearchResult(
                id=f"keyword_{i}",
                document_id=DocumentId(f"doc_{i}"),
                document_title=f"Document {i}",
                document_filename=f"document_{i}.pdf",
                chunk_id=f"chunk_{i}",
                content=f"This is sample content containing {query} for testing purposes.",
                snippet=f"Sample content with {query}...",
                relevance_score=0.8 - (i * 0.1),
                confidence_score=0.7 - (i * 0.1),
                page_number=i + 1,
                start_position=0,
                end_position=100,
                highlighted_content=f"This is sample content containing <mark>{query}</mark> for testing.",
                search_type_used=SearchType.KEYWORD,
                ranking_factors={"keyword_matches": 1, "term_frequency": 0.1},
                metadata={"search_method": "keyword"}
            )
            results.append(result)
        
        return results
    
    async def _simulate_phrase_search(
        self,
        phrase: str,
        exact_match: bool,
        filters: Optional[SearchFilter]
    ) -> List[SearchResult]:
        """Simulate phrase search results."""
        # Mock implementation
        return await self._simulate_keyword_search(phrase, filters, 10)
    
    async def _simulate_boolean_search(
        self,
        parsed_query: Dict[str, Any],
        filters: Optional[SearchFilter]
    ) -> List[SearchResult]:
        """Simulate boolean search results."""
        # Mock implementation
        query_text = " ".join(parsed_query.get("terms", []))
        return await self._simulate_keyword_search(query_text, filters, 10)
    
    def _parse_boolean_query(self, boolean_query: str) -> Dict[str, Any]:
        """Parse boolean query into structured format."""
        # Simple boolean query parsing
        terms = []
        operators = []
        
        parts = boolean_query.split()
        for part in parts:
            if part.upper() in ["AND", "OR", "NOT"]:
                operators.append(part.upper())
            else:
                terms.append(part)
        
        return {
            "terms": terms,
            "operators": operators,
            "original_query": boolean_query
        }


class HybridQuestionAnsweringEngine(QuestionAnsweringEngine):
    """
    Question-answering engine for extractive QA.
    
    Implements QA operations following Single Responsibility Principle.
    """
    
    def __init__(self, semantic_engine: SemanticSearchEngine):
        """
        Initialize QA engine.
        
        Args:
            semantic_engine: Semantic search engine for context retrieval
        """
        self._semantic_engine = semantic_engine
        self._logger = logging.getLogger(__name__)
    
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
        try:
            self._logger.info(f"Answering question: {question}")
            
            # Use semantic search to find relevant content
            filters = SearchFilter(document_ids=context_documents) if context_documents else None
            
            results = await self._semantic_engine.semantic_search(
                query=question,
                filters=filters,
                max_results=10
            )
            
            # Filter by confidence threshold
            qa_results = []
            for result in results:
                if result.confidence_score >= confidence_threshold:
                    # Enhance result for QA
                    result.search_type_used = SearchType.QUESTION_ANSWERING
                    result.ranking_factors["qa_confidence"] = result.confidence_score
                    qa_results.append(result)
            
            self._logger.info(f"Found {len(qa_results)} QA results")
            return qa_results
            
        except Exception as e:
            self._logger.error(f"Question answering failed: {e}")
            raise QuestionAnsweringError(f"Question answering failed: {str(e)}")
    
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
        try:
            # Use semantic search to find factual content
            results = await self._semantic_engine.semantic_search(
                query=query,
                max_results=15
            )
            
            # Filter for factual content (this would be enhanced with NLP)
            factual_results = []
            for result in results:
                if self._contains_factual_content(result.content, fact_types):
                    result.search_type_used = SearchType.QUESTION_ANSWERING
                    result.ranking_factors["factual_score"] = 0.8
                    factual_results.append(result)
            
            return factual_results
            
        except Exception as e:
            self._logger.error(f"Fact extraction failed: {e}")
            raise QuestionAnsweringError(f"Fact extraction failed: {str(e)}")
    
    def _contains_factual_content(self, content: str, fact_types: Optional[List[str]]) -> bool:
        """Check if content contains factual information."""
        # Simple heuristic for factual content
        factual_indicators = [
            r'\d+%',  # Percentages
            r'\$\d+',  # Currency
            r'\d{4}',  # Years
            r'according to',  # Attribution
            r'research shows',  # Evidence
            r'study found',  # Research findings
        ]
        
        content_lower = content.lower()
        
        for pattern in factual_indicators:
            if re.search(pattern, content_lower):
                return True
        
        return False


class HybridSearchRanker(SearchRanker):
    """
    Search result ranker with multiple ranking strategies.
    
    Implements ranking operations following Single Responsibility Principle.
    """
    
    def __init__(self):
        """Initialize search ranker."""
        self._logger = logging.getLogger(__name__)
    
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
        try:
            if ranking_strategy == RankingStrategy.RELEVANCE:
                return sorted(results, key=lambda r: r.relevance_score, reverse=True)
            elif ranking_strategy == RankingStrategy.RECENCY:
                return await self._rank_by_recency(results)
            elif ranking_strategy == RankingStrategy.HYBRID_SCORE:
                return await self._rank_by_hybrid_score(results, context)
            else:
                return results
            
        except Exception as e:
            self._logger.error(f"Ranking failed: {e}")
            raise RankingError(f"Ranking failed: {str(e)}")
    
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
        try:
            # Combine multiple relevance factors
            base_score = result.relevance_score
            confidence_boost = result.confidence_score * 0.2
            
            # Query term matching
            query_terms = set(query.lower().split())
            content_terms = set(result.content.lower().split())
            term_overlap = len(query_terms.intersection(content_terms)) / len(query_terms)
            
            final_score = (base_score * 0.6) + (confidence_boost * 0.2) + (term_overlap * 0.2)
            return min(1.0, final_score)
            
        except Exception as e:
            self._logger.error(f"Relevance scoring failed: {e}")
            return result.relevance_score
    
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
        try:
            explanation = {
                "result_id": result.chunk_id,
                "final_score": result.relevance_score,
                "ranking_factors": ranking_factors,
                "explanation": f"Result ranked based on {max(ranking_factors, key=ranking_factors.get)} score"
            }
            
            return explanation
            
        except Exception as e:
            self._logger.error(f"Ranking explanation failed: {e}")
            return {"error": str(e)}
    
    async def _rank_by_recency(self, results: List[SearchResult]) -> List[SearchResult]:
        """Rank results by recency (mock implementation)."""
        # This would use actual document timestamps
        return results
    
    async def _rank_by_hybrid_score(
        self,
        results: List[SearchResult],
        context: Optional[SearchContext]
    ) -> List[SearchResult]:
        """Rank results using hybrid scoring."""
        for result in results:
            # Calculate hybrid score
            relevance_weight = 0.6
            confidence_weight = 0.3
            context_weight = 0.1
            
            hybrid_score = (
                result.relevance_score * relevance_weight +
                result.confidence_score * confidence_weight +
                (0.5 * context_weight if context else 0)  # Mock context score
            )
            
            result.ranking_factors["hybrid_score"] = hybrid_score
        
        return sorted(results, key=lambda r: r.ranking_factors.get("hybrid_score", 0), reverse=True)


class HybridSearchServiceFactory:
    """
    Factory for creating hybrid search service instances.
    
    Follows Open/Closed Principle for extensibility.
    """
    
    @staticmethod
    def create_semantic_search_engine(
        similarity_search_use_case: SimilaritySearchUseCase,
        similarity_threshold: float = 0.7
    ) -> HybridSemanticSearchEngine:
        """
        Create semantic search engine instance.
        
        Args:
            similarity_search_use_case: Similarity search use case
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            Configured semantic search engine
        """
        return HybridSemanticSearchEngine(
            similarity_search_use_case=similarity_search_use_case,
            similarity_threshold=similarity_threshold
        )
    
    @staticmethod
    def create_keyword_search_engine(
        enable_fuzzy_matching: bool = True
    ) -> HybridKeywordSearchEngine:
        """
        Create keyword search engine instance.
        
        Args:
            enable_fuzzy_matching: Whether to enable fuzzy matching
            
        Returns:
            Configured keyword search engine
        """
        return HybridKeywordSearchEngine(
            enable_fuzzy_matching=enable_fuzzy_matching
        )
    
    @staticmethod
    def create_qa_engine(
        semantic_engine: SemanticSearchEngine
    ) -> HybridQuestionAnsweringEngine:
        """
        Create question-answering engine instance.
        
        Args:
            semantic_engine: Semantic search engine
            
        Returns:
            Configured QA engine
        """
        return HybridQuestionAnsweringEngine(
            semantic_engine=semantic_engine
        )
    
    @staticmethod
    def create_search_ranker() -> HybridSearchRanker:
        """
        Create search ranker instance.
        
        Returns:
            Configured search ranker
        """
        return HybridSearchRanker()
