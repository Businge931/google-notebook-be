"""
Advanced Citation Extractor Implementation

"""
import re
import logging
from typing import List, Optional, Dict, Any, Set, Tuple
from datetime import datetime
import asyncio
from collections import defaultdict

try:
    import spacy
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    import numpy as np
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

from src.core.domain.services.citation_service import (
    CitationExtractor,
    CitationLinker,
    CitationAnalyzer,
    CitationService,
    Citation,
    CitationSpan,
    SourceLocation,
    CitationCluster,
    CitationExtractionRequest,
    CitationExtractionResponse,
    CitationType,
    CitationConfidence
)
from src.core.domain.value_objects import DocumentId, SessionId
from src.shared.exceptions import (
    CitationExtractionError,
    CitationValidationError,
    CitationClusteringError,
    ConfigurationError
)


class AdvancedCitationExtractor(CitationExtractor):
    """
    Advanced citation extractor using NLP and semantic analysis.
    
    Implements sophisticated citation extraction following Single Responsibility Principle.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.7,
        min_citation_length: int = 10,
        max_citation_length: int = 500,
        enable_clustering: bool = True
    ):
        """
        Initialize advanced citation extractor.
        
        Args:
            similarity_threshold: Minimum similarity for citations
            min_citation_length: Minimum citation text length
            max_citation_length: Maximum citation text length
            enable_clustering: Whether to enable citation clustering
        """
        if not NLP_AVAILABLE:
            raise ImportError("NLP libraries not available for advanced citation extraction")
        
        self._similarity_threshold = similarity_threshold
        self._min_citation_length = min_citation_length
        self._max_citation_length = max_citation_length
        self._enable_clustering = enable_clustering
        self._logger = logging.getLogger(__name__)
        
        # Initialize NLP components
        try:
            self._nlp = spacy.load("en_core_web_sm")
        except OSError:
            self._logger.warning("spaCy model not found, using basic extraction")
            self._nlp = None
        
        self._vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3)
        )
    
    async def extract_citations(
        self,
        request: CitationExtractionRequest
    ) -> CitationExtractionResponse:
        """
        Extract citations from response text using advanced NLP techniques.
        
        Args:
            request: Citation extraction request
            
        Returns:
            Citation extraction response with extracted citations
            
        Raises:
            CitationExtractionError: If extraction fails
        """
        start_time = datetime.utcnow()
        
        try:
            self._logger.info(f"Starting citation extraction for {len(request.source_chunks)} chunks")
            
            # Extract citations using multiple strategies
            citations = []
            
            if request.extraction_strategy == "semantic":
                citations.extend(await self._semantic_extraction(request))
            elif request.extraction_strategy == "syntactic":
                citations.extend(await self._syntactic_extraction(request))
            elif request.extraction_strategy == "hybrid":
                semantic_citations = await self._semantic_extraction(request)
                syntactic_citations = await self._syntactic_extraction(request)
                citations = await self._merge_citations(semantic_citations, syntactic_citations)
            else:
                citations.extend(await self._basic_extraction(request))
            
            # Filter by confidence
            citations = [c for c in citations if c.confidence.value in ["high", "medium"] or 
                        (c.confidence.value == "low" and request.min_confidence <= 0.5)]
            
            # Limit results
            citations = citations[:request.max_citations]
            
            # Validate citations
            validated_citations = await self.validate_citations(
                citations, request.context_documents
            )
            
            # Cluster citations if requested
            citation_clusters = []
            if request.cluster_citations and self._enable_clustering:
                citation_clusters = await self.cluster_citations(validated_citations)
            
            # Calculate confidence distribution
            confidence_dist = defaultdict(int)
            for citation in validated_citations:
                confidence_dist[citation.confidence.value] += 1
            
            # Calculate processing time
            end_time = datetime.utcnow()
            processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            self._logger.info(
                f"Extracted {len(validated_citations)} citations in {processing_time_ms}ms"
            )
            
            return CitationExtractionResponse(
                citations=validated_citations,
                citation_clusters=citation_clusters,
                extraction_metadata={
                    "strategy_used": request.extraction_strategy,
                    "source_chunks_processed": len(request.source_chunks),
                    "similarity_threshold": self._similarity_threshold,
                    "nlp_enabled": self._nlp is not None
                },
                processing_time_ms=processing_time_ms,
                total_citations_found=len(citations),
                confidence_distribution=dict(confidence_dist)
            )
            
        except Exception as e:
            self._logger.error(f"Citation extraction failed: {e}")
            raise CitationExtractionError(f"Citation extraction failed: {str(e)}")
    
    async def _semantic_extraction(
        self,
        request: CitationExtractionRequest
    ) -> List[Citation]:
        """
        Extract citations using semantic similarity analysis.
        
        Args:
            request: Citation extraction request
            
        Returns:
            List of semantically extracted citations
        """
        citations = []
        
        try:
            # Prepare texts for vectorization
            response_sentences = self._split_into_sentences(request.response_text)
            source_texts = [chunk.get('text_content', '') for chunk in request.source_chunks]
            
            if not source_texts:
                return citations
            
            # Vectorize texts
            all_texts = response_sentences + source_texts
            vectors = self._vectorizer.fit_transform(all_texts)
            
            response_vectors = vectors[:len(response_sentences)]
            source_vectors = vectors[len(response_sentences):]
            
            # Calculate similarities
            similarities = cosine_similarity(response_vectors, source_vectors)
            
            # Extract citations based on similarity
            for i, sentence in enumerate(response_sentences):
                for j, chunk in enumerate(request.source_chunks):
                    similarity_score = similarities[i][j]
                    
                    if similarity_score >= self._similarity_threshold:
                        citation = await self._create_citation(
                            sentence=sentence,
                            chunk=chunk,
                            similarity_score=similarity_score,
                            citation_type=CitationType.PARAPHRASE,
                            sentence_index=i
                        )
                        if citation:
                            citations.append(citation)
            
            return citations
            
        except Exception as e:
            self._logger.error(f"Semantic extraction failed: {e}")
            return []
    
    async def _syntactic_extraction(
        self,
        request: CitationExtractionRequest
    ) -> List[Citation]:
        """
        Extract citations using syntactic pattern matching.
        
        Args:
            request: Citation extraction request
            
        Returns:
            List of syntactically extracted citations
        """
        citations = []
        
        try:
            # Define patterns for direct quotes and references
            quote_patterns = [
                r'"([^"]+)"',  # Direct quotes
                r"'([^']+)'",  # Single quotes
                r'according to[^.]+',  # Attribution phrases
                r'as stated in[^.]+',  # Reference phrases
                r'research shows[^.]+',  # Evidence phrases
            ]
            
            for pattern in quote_patterns:
                matches = re.finditer(pattern, request.response_text, re.IGNORECASE)
                
                for match in matches:
                    matched_text = match.group(0)
                    
                    # Find best matching source chunk
                    best_chunk = None
                    best_score = 0.0
                    
                    for chunk in request.source_chunks:
                        chunk_text = chunk.get('text_content', '')
                        score = self._calculate_text_overlap(matched_text, chunk_text)
                        
                        if score > best_score:
                            best_score = score
                            best_chunk = chunk
                    
                    if best_chunk and best_score >= 0.3:
                        citation = await self._create_citation(
                            sentence=matched_text,
                            chunk=best_chunk,
                            similarity_score=best_score,
                            citation_type=CitationType.DIRECT_QUOTE,
                            start_pos=match.start(),
                            end_pos=match.end()
                        )
                        if citation:
                            citations.append(citation)
            
            return citations
            
        except Exception as e:
            self._logger.error(f"Syntactic extraction failed: {e}")
            return []
    
    async def _basic_extraction(
        self,
        request: CitationExtractionRequest
    ) -> List[Citation]:
        """
        Basic citation extraction using simple text matching.
        
        Args:
            request: Citation extraction request
            
        Returns:
            List of basically extracted citations
        """
        citations = []
        
        try:
            response_words = set(request.response_text.lower().split())
            
            for chunk in request.source_chunks:
                chunk_text = chunk.get('text_content', '')
                chunk_words = set(chunk_text.lower().split())
                
                # Calculate word overlap
                overlap = len(response_words.intersection(chunk_words))
                total_words = len(response_words.union(chunk_words))
                
                if total_words > 0:
                    similarity_score = overlap / total_words
                    
                    if similarity_score >= self._similarity_threshold:
                        citation = await self._create_citation(
                            sentence=request.response_text[:200] + "...",
                            chunk=chunk,
                            similarity_score=similarity_score,
                            citation_type=CitationType.REFERENCE
                        )
                        if citation:
                            citations.append(citation)
            
            return citations
            
        except Exception as e:
            self._logger.error(f"Basic extraction failed: {e}")
            return []
    
    async def _create_citation(
        self,
        sentence: str,
        chunk: Dict[str, Any],
        similarity_score: float,
        citation_type: CitationType,
        sentence_index: Optional[int] = None,
        start_pos: Optional[int] = None,
        end_pos: Optional[int] = None
    ) -> Optional[Citation]:
        """
        Create a citation object from extracted information.
        
        Args:
            sentence: Sentence containing the citation
            chunk: Source chunk
            similarity_score: Similarity score
            citation_type: Type of citation
            sentence_index: Index of sentence in response
            start_pos: Start position in text
            end_pos: End position in text
            
        Returns:
            Citation object or None if invalid
        """
        try:
            # Validate citation length
            if len(sentence) < self._min_citation_length or len(sentence) > self._max_citation_length:
                return None
            
            # Determine confidence based on similarity score
            if similarity_score >= 0.8:
                confidence = CitationConfidence.HIGH
            elif similarity_score >= 0.6:
                confidence = CitationConfidence.MEDIUM
            else:
                confidence = CitationConfidence.LOW
            
            # Create citation span
            citation_span = CitationSpan(
                start_position=start_pos or 0,
                end_position=end_pos or len(sentence),
                text=sentence,
                citation_type=citation_type,
                confidence=confidence
            )
            
            # Create source location
            source_location = SourceLocation(
                document_id=DocumentId(chunk.get('document_id', '')),
                document_title=chunk.get('document_title', ''),
                page_number=chunk.get('page_number'),
                chunk_id=chunk.get('chunk_id'),
                start_char=chunk.get('start_position'),
                end_char=chunk.get('end_position'),
                section_title=chunk.get('section_title'),
                metadata=chunk.get('metadata', {})
            )
            
            # Generate unique citation ID
            citation_id = f"cite_{hash(sentence + str(similarity_score))}_{datetime.utcnow().timestamp()}"
            
            return Citation(
                id=citation_id,
                citation_spans=[citation_span],
                source_location=source_location,
                source_text=chunk.get('text_content', ''),
                similarity_score=similarity_score,
                relevance_score=similarity_score,  # Can be enhanced with additional scoring
                citation_type=citation_type,
                confidence=confidence,
                generated_at=datetime.utcnow().isoformat(),
                metadata={
                    "extraction_method": "advanced",
                    "sentence_index": sentence_index,
                    "chunk_metadata": chunk.get('metadata', {})
                }
            )
            
        except Exception as e:
            self._logger.error(f"Failed to create citation: {e}")
            return None
    
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
        try:
            validated_citations = []
            source_doc_ids = {doc.value for doc in source_documents}
            
            for citation in citations:
                # Check if source document is available
                if citation.source_location.document_id.value not in source_doc_ids:
                    continue
                
                # Check citation quality
                if citation.similarity_score < 0.3:
                    continue
                
                # Check for duplicate citations
                is_duplicate = any(
                    existing.source_location.document_id == citation.source_location.document_id and
                    existing.source_location.chunk_id == citation.source_location.chunk_id and
                    abs(existing.similarity_score - citation.similarity_score) < 0.1
                    for existing in validated_citations
                )
                
                if not is_duplicate:
                    validated_citations.append(citation)
            
            return validated_citations
            
        except Exception as e:
            self._logger.error(f"Citation validation failed: {e}")
            raise CitationValidationError(f"Citation validation failed: {str(e)}")
    
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
        try:
            if len(citations) < 2:
                return []
            
            clusters = []
            
            if clustering_strategy == "semantic":
                clusters = await self._semantic_clustering(citations)
            elif clustering_strategy == "document":
                clusters = await self._document_clustering(citations)
            else:
                clusters = await self._similarity_clustering(citations)
            
            return clusters
            
        except Exception as e:
            self._logger.error(f"Citation clustering failed: {e}")
            raise CitationClusteringError(f"Citation clustering failed: {str(e)}")
    
    async def _semantic_clustering(self, citations: List[Citation]) -> List[CitationCluster]:
        """Cluster citations based on semantic similarity."""
        try:
            # Extract text from citations
            citation_texts = [' '.join([span.text for span in cite.citation_spans]) for cite in citations]
            
            # Vectorize citation texts
            vectors = self._vectorizer.fit_transform(citation_texts)
            
            # Determine optimal number of clusters
            n_clusters = min(3, len(citations) // 2 + 1)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(vectors.toarray())
            
            # Group citations by cluster
            clusters = []
            for cluster_id in range(n_clusters):
                cluster_citations = [citations[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                
                if cluster_citations:
                    # Find representative citation (highest similarity score)
                    representative = max(cluster_citations, key=lambda c: c.similarity_score)
                    
                    cluster = CitationCluster(
                        id=f"cluster_{cluster_id}",
                        citations=cluster_citations,
                        cluster_topic=f"Topic {cluster_id + 1}",
                        cluster_confidence=sum(c.similarity_score for c in cluster_citations) / len(cluster_citations),
                        representative_citation=representative,
                        metadata={"clustering_method": "semantic", "cluster_size": len(cluster_citations)}
                    )
                    clusters.append(cluster)
            
            return clusters
            
        except Exception as e:
            self._logger.error(f"Semantic clustering failed: {e}")
            return []
    
    async def _document_clustering(self, citations: List[Citation]) -> List[CitationCluster]:
        """Cluster citations by source document."""
        try:
            # Group by document
            doc_groups = defaultdict(list)
            for citation in citations:
                doc_id = citation.source_location.document_id.value
                doc_groups[doc_id].append(citation)
            
            clusters = []
            for doc_id, doc_citations in doc_groups.items():
                if len(doc_citations) > 1:  # Only cluster if multiple citations from same document
                    representative = max(doc_citations, key=lambda c: c.similarity_score)
                    
                    cluster = CitationCluster(
                        id=f"doc_cluster_{doc_id}",
                        citations=doc_citations,
                        cluster_topic=doc_citations[0].source_location.document_title,
                        cluster_confidence=sum(c.similarity_score for c in doc_citations) / len(doc_citations),
                        representative_citation=representative,
                        metadata={"clustering_method": "document", "document_id": doc_id}
                    )
                    clusters.append(cluster)
            
            return clusters
            
        except Exception as e:
            self._logger.error(f"Document clustering failed: {e}")
            return []
    
    async def _similarity_clustering(self, citations: List[Citation]) -> List[CitationCluster]:
        """Cluster citations by similarity score ranges."""
        try:
            # Group by similarity ranges
            high_sim = [c for c in citations if c.similarity_score >= 0.8]
            medium_sim = [c for c in citations if 0.6 <= c.similarity_score < 0.8]
            low_sim = [c for c in citations if c.similarity_score < 0.6]
            
            clusters = []
            
            for group, name in [(high_sim, "High Confidence"), (medium_sim, "Medium Confidence"), (low_sim, "Low Confidence")]:
                if group:
                    representative = max(group, key=lambda c: c.similarity_score)
                    
                    cluster = CitationCluster(
                        id=f"sim_cluster_{name.lower().replace(' ', '_')}",
                        citations=group,
                        cluster_topic=name,
                        cluster_confidence=sum(c.similarity_score for c in group) / len(group),
                        representative_citation=representative,
                        metadata={"clustering_method": "similarity", "confidence_level": name}
                    )
                    clusters.append(cluster)
            
            return clusters
            
        except Exception as e:
            self._logger.error(f"Similarity clustering failed: {e}")
            return []
    
    async def _merge_citations(
        self,
        semantic_citations: List[Citation],
        syntactic_citations: List[Citation]
    ) -> List[Citation]:
        """Merge citations from different extraction strategies."""
        try:
            # Combine and deduplicate citations
            all_citations = semantic_citations + syntactic_citations
            
            # Remove duplicates based on source location and similarity
            unique_citations = []
            for citation in all_citations:
                is_duplicate = any(
                    existing.source_location.document_id == citation.source_location.document_id and
                    existing.source_location.chunk_id == citation.source_location.chunk_id and
                    abs(existing.similarity_score - citation.similarity_score) < 0.05
                    for existing in unique_citations
                )
                
                if not is_duplicate:
                    unique_citations.append(citation)
            
            # Sort by similarity score
            unique_citations.sort(key=lambda c: c.similarity_score, reverse=True)
            
            return unique_citations
            
        except Exception as e:
            self._logger.error(f"Citation merging failed: {e}")
            return semantic_citations + syntactic_citations
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLP or simple splitting."""
        if self._nlp:
            doc = self._nlp(text)
            return [sent.text.strip() for sent in doc.sents]
        else:
            # Simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_text_overlap(self, text1: str, text2: str) -> float:
        """Calculate text overlap between two strings."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0


class AdvancedCitationExtractorFactory:
    """
    Factory for creating advanced citation extractor instances.
    
    Follows Open/Closed Principle for extensibility.
    """
    
    @staticmethod
    def create_citation_extractor(
        similarity_threshold: float = 0.7,
        min_citation_length: int = 10,
        max_citation_length: int = 500,
        enable_clustering: bool = True
    ) -> AdvancedCitationExtractor:
        """
        Create advanced citation extractor instance.
        
        Args:
            similarity_threshold: Minimum similarity for citations
            min_citation_length: Minimum citation text length
            max_citation_length: Maximum citation text length
            enable_clustering: Whether to enable citation clustering
            
        Returns:
            Configured advanced citation extractor
        """
        return AdvancedCitationExtractor(
            similarity_threshold=similarity_threshold,
            min_citation_length=min_citation_length,
            max_citation_length=max_citation_length,
            enable_clustering=enable_clustering
        )
