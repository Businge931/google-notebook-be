"""
Similarity Search Use Case

"""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging
from datetime import datetime

from ...domain.value_objects import DocumentId
from ...domain.repositories import VectorRepository, DocumentRepository
from ...domain.repositories.vector_repository import VectorSearchResult
from src.shared.exceptions import (
    DocumentNotFoundError,
    VectorRepositoryError,
    ValidationError
)


@dataclass
class SimilaritySearchRequest:
    """Request object for similarity search following Single Responsibility Principle."""
    query_text: str
    document_id: Optional[DocumentId] = None
    document_ids: Optional[List[DocumentId]] = None
    limit: int = 10
    similarity_threshold: float = 0.7
    include_metadata: bool = True


@dataclass
class SearchResultItem:
    """Individual search result item following Single Responsibility Principle."""
    chunk_id: str
    document_id: str
    document_title: Optional[str]
    document_filename: Optional[str]
    page_number: int
    text_content: str
    similarity_score: float
    start_position: Optional[int] = None
    end_position: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SimilaritySearchResponse:
    """Response object for similarity search following Single Responsibility Principle."""
    query_text: str
    results: List[SearchResultItem]
    total_results: int
    search_time_ms: int
    search_scope: str  # "single_document", "multiple_documents", "all_documents"
    similarity_threshold: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class DocumentSimilarityRequest:
    """Request object for document similarity search following Single Responsibility Principle."""
    query_text: str
    limit: int = 5
    exclude_document_ids: Optional[List[DocumentId]] = None
    similarity_threshold: float = 0.7


@dataclass
class DocumentSimilarityResponse:
    """Response object for document similarity search following Single Responsibility Principle."""
    query_text: str
    similar_documents: List[Dict[str, Any]]
    total_documents: int
    search_time_ms: int
    similarity_threshold: float


class SimilaritySearchUseCase:
    """
    Use case for similarity search operations following Single Responsibility Principle.
    
    Orchestrates vector-based similarity search across documents and chunks.
    """
    
    def __init__(
        self,
        vector_repository: VectorRepository,
        document_repository: DocumentRepository
    ):
        """
        Initialize similarity search use case.
        
        Args:
            vector_repository: Repository for vector operations
            document_repository: Repository for document operations
        """
        self._vector_repository = vector_repository
        self._document_repository = document_repository
        self._logger = logging.getLogger(__name__)
        self._logger.info(f"🔥 SIMILARITY SEARCH: Initialized with vector repository: {type(vector_repository).__name__}")
    
    async def search_similar_chunks(
        self,
        request: SimilaritySearchRequest
    ) -> SimilaritySearchResponse:
        """
        Search for similar text chunks.
        
        Args:
            request: Similarity search request
            
        Returns:
            Similarity search response
            
        Raises:
            ValidationError: If request is invalid
            VectorRepositoryError: If search fails
        """
        start_time = datetime.utcnow()
        
        self._logger.info(f"🔥 SIMILARITY SEARCH: search_similar_chunks called with query: '{request.query_text[:50]}...'")
        self._logger.info(f"🔥 SIMILARITY SEARCH: Vector repository type: {type(self._vector_repository).__name__}")
        print(f"🔥 PRINT DEBUG: search_similar_chunks called with query: '{request.query_text[:50]}...'")
        print(f"🔥 PRINT DEBUG: Vector repository type: {type(self._vector_repository).__name__}")
        
        try:
            # Validate request
            if not request.query_text or not request.query_text.strip():
                raise ValidationError("Query text cannot be empty")
            
            if request.limit <= 0:
                raise ValidationError("Limit must be positive")
            
            if not 0.0 <= request.similarity_threshold <= 1.0:
                raise ValidationError("Similarity threshold must be between 0.0 and 1.0")
            
            # Determine search scope and execute search
            vector_results = []
            search_scope = "all_documents"
            
            if request.document_id:
                # Search within a single document
                search_scope = "single_document"
                vector_results = await self._vector_repository.search_similar(
                    document_id=request.document_id,
                    query_text=request.query_text,
                    limit=request.limit,
                    similarity_threshold=request.similarity_threshold
                )
            elif request.document_ids:
                # Search within multiple specific documents
                search_scope = "multiple_documents"
                vector_results = await self._vector_repository.search_across_documents(
                    query_text=request.query_text,
                    document_ids=request.document_ids,
                    limit=request.limit,
                    similarity_threshold=request.similarity_threshold
                )
            else:
                # Search across all documents
                vector_results = await self._vector_repository.search_across_documents(
                    query_text=request.query_text,
                    limit=request.limit,
                    similarity_threshold=request.similarity_threshold
                )
            
            # Enrich results with document information if needed
            search_results = []
            document_cache = {}  # Cache to avoid repeated document lookups
            
            for vector_result in vector_results:
                # Get document information
                doc_id = vector_result.document_id
                document_title = None
                document_filename = None
                
                if request.include_metadata:
                    if doc_id.value not in document_cache:
                        try:
                            document = await self._document_repository.get_by_id(doc_id)
                            if document:
                                document_cache[doc_id.value] = {
                                    "title": document.file_metadata.filename,
                                    "filename": document.file_metadata.filename
                                }
                            else:
                                document_cache[doc_id.value] = {
                                    "title": None,
                                    "filename": None
                                }
                        except Exception as e:
                            self._logger.warning(f"Failed to get document {doc_id.value}: {e}")
                            document_cache[doc_id.value] = {
                                "title": None,
                                "filename": None
                            }
                    
                    doc_info = document_cache[doc_id.value]
                    document_title = doc_info["title"]
                    document_filename = doc_info["filename"]
                
                # Create search result item
                result_item = SearchResultItem(
                    chunk_id=vector_result.chunk_id,
                    document_id=doc_id.value,
                    document_title=document_title,
                    document_filename=document_filename,
                    page_number=vector_result.page_number,
                    text_content=vector_result.text_content,
                    similarity_score=vector_result.similarity_score,
                    start_position=vector_result.metadata.get("start_position"),
                    end_position=vector_result.metadata.get("end_position"),
                    metadata=vector_result.metadata if request.include_metadata else None
                )
                search_results.append(result_item)
            
            # Calculate search time
            end_time = datetime.utcnow()
            search_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            self._logger.info(
                f"Similarity search completed: {len(search_results)} results for query "
                f"'{request.query_text[:50]}...' in {search_time_ms}ms"
            )
            
            return SimilaritySearchResponse(
                query_text=request.query_text,
                results=search_results,
                total_results=len(search_results),
                search_time_ms=search_time_ms,
                search_scope=search_scope,
                similarity_threshold=request.similarity_threshold,
                metadata={
                    "documents_searched": len(request.document_ids) if request.document_ids else "all",
                    "cache_hits": len(document_cache),
                    "vector_results_count": len(vector_results)
                }
            )
            
        except (ValidationError, DocumentNotFoundError) as e:
            # Re-raise domain exceptions
            raise e
        except Exception as e:
            self._logger.error(f"Similarity search failed for query '{request.query_text}': {e}")
            raise VectorRepositoryError(f"Similarity search failed: {str(e)}")
    
    async def search_similar_documents(
        self,
        request: DocumentSimilarityRequest
    ) -> DocumentSimilarityResponse:
        """
        Search for documents similar to query text.
        
        Args:
            request: Document similarity search request
            
        Returns:
            Document similarity response
            
        Raises:
            ValidationError: If request is invalid
            VectorRepositoryError: If search fails
        """
        start_time = datetime.utcnow()
        
        try:
            # Validate request
            if not request.query_text or not request.query_text.strip():
                raise ValidationError("Query text is required")
            
            if request.limit <= 0:
                raise ValidationError("Limit must be positive")
            
            # Search across all documents with higher limit to aggregate by document
            search_limit = request.limit * 10  # Get more results to aggregate
            vector_results = await self._vector_repository.search_across_documents(
                query_text=request.query_text,
                limit=search_limit,
                similarity_threshold=request.similarity_threshold
            )
            
            # Aggregate results by document
            document_scores = {}
            document_chunks = {}
            
            for result in vector_results:
                doc_id = result.document_id.value
                
                if doc_id not in document_scores:
                    document_scores[doc_id] = []
                    document_chunks[doc_id] = []
                
                document_scores[doc_id].append(result.similarity_score)
                document_chunks[doc_id].append({
                    "chunk_id": result.chunk_id,
                    "text_content": result.text_content[:200] + "..." if len(result.text_content) > 200 else result.text_content,
                    "page_number": result.page_number,
                    "similarity_score": result.similarity_score
                })
            
            # Calculate document-level similarity scores
            document_similarities = []
            for doc_id, scores in document_scores.items():
                # Use weighted average with emphasis on top scores
                sorted_scores = sorted(scores, reverse=True)
                if len(sorted_scores) >= 3:
                    # Weight: 50% top score, 30% second, 20% third
                    avg_score = (sorted_scores[0] * 0.5 + 
                               sorted_scores[1] * 0.3 + 
                               sorted_scores[2] * 0.2)
                elif len(sorted_scores) == 2:
                    # Weight: 70% top score, 30% second
                    avg_score = sorted_scores[0] * 0.7 + sorted_scores[1] * 0.3
                else:
                    # Single score
                    avg_score = sorted_scores[0]
                
                document_similarities.append({
                    "document_id": doc_id,
                    "similarity_score": avg_score,
                    "chunk_count": len(scores),
                    "top_chunks": document_chunks[doc_id][:3]  # Top 3 chunks
                })
            
            # Sort by similarity score and limit results
            document_similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            # Filter out excluded documents
            if request.exclude_document_ids:
                excluded_ids = {doc_id.value for doc_id in request.exclude_document_ids}
                document_similarities = [
                    doc for doc in document_similarities 
                    if doc["document_id"] not in excluded_ids
                ]
            
            # Limit results
            document_similarities = document_similarities[:request.limit]
            
            # Enrich with document metadata
            similar_documents = []
            for doc_sim in document_similarities:
                try:
                    document = await self._document_repository.get_by_id(
                        DocumentId(doc_sim["document_id"])
                    )
                    
                    doc_info = {
                        "document_id": doc_sim["document_id"],
                        "title": document.file_metadata.filename if document else "Unknown",
                        "filename": document.file_metadata.filename if document else "Unknown",
                        "similarity_score": doc_sim["similarity_score"],
                        "matching_chunks": doc_sim["chunk_count"],
                        "top_chunks": doc_sim["top_chunks"],
                        "upload_date": document.upload_date.isoformat() if document and document.upload_date else None,
                        "status": document.status.value if document else "unknown"
                    }
                    similar_documents.append(doc_info)
                    
                except Exception as e:
                    self._logger.warning(f"Failed to get document {doc_sim['document_id']}: {e}")
                    # Include basic info even if document lookup fails
                    doc_info = {
                        "document_id": doc_sim["document_id"],
                        "title": "Unknown",
                        "filename": "Unknown",
                        "similarity_score": doc_sim["similarity_score"],
                        "matching_chunks": doc_sim["chunk_count"],
                        "top_chunks": doc_sim["top_chunks"],
                        "upload_date": None,
                        "status": "unknown"
                    }
                    similar_documents.append(doc_info)
            
            # Calculate search time
            end_time = datetime.utcnow()
            search_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            self._logger.info(
                f"Document similarity search completed: {len(similar_documents)} documents "
                f"for query '{request.query_text[:50]}...' in {search_time_ms}ms"
            )
            
            return DocumentSimilarityResponse(
                query_text=request.query_text,
                similar_documents=similar_documents,
                total_documents=len(similar_documents),
                search_time_ms=search_time_ms,
                similarity_threshold=request.similarity_threshold
            )
            
        except (ValidationError, DocumentNotFoundError) as e:
            # Re-raise domain exceptions
            raise e
        except Exception as e:
            self._logger.error(f"Document similarity search failed for query '{request.query_text}': {e}")
            raise VectorRepositoryError(f"Document similarity search failed: {str(e)}")
    
    async def get_search_suggestions(
        self,
        partial_query: str,
        document_id: Optional[DocumentId] = None,
        limit: int = 5
    ) -> List[str]:
        """
        Get search suggestions based on partial query.
        
        This is a simplified implementation that could be enhanced with
        more sophisticated suggestion algorithms.
        
        Args:
            partial_query: Partial query text
            document_id: Optional document to search within
            limit: Maximum number of suggestions
            
        Returns:
            List of suggested queries
        """
        try:
            if len(partial_query.strip()) < 3:
                return []
            
            # Simple implementation: search for similar chunks and extract key phrases
            search_request = SimilaritySearchRequest(
                query_text=partial_query,
                document_id=document_id,
                limit=limit * 2,  # Get more results to extract suggestions
                similarity_threshold=0.5,  # Lower threshold for suggestions
                include_metadata=False
            )
            
            search_response = await self.search_similar_chunks(search_request)
            
            # Extract potential suggestions from search results
            suggestions = set()
            for result in search_response.results:
                # Extract sentences containing the partial query
                text = result.text_content.lower()
                partial_lower = partial_query.lower()
                
                if partial_lower in text:
                    # Find sentences containing the partial query
                    sentences = text.split('.')
                    for sentence in sentences:
                        if partial_lower in sentence:
                            # Clean and add as suggestion
                            clean_sentence = sentence.strip()
                            if len(clean_sentence) > len(partial_query) and len(clean_sentence) < 100:
                                suggestions.add(clean_sentence.capitalize())
                
                if len(suggestions) >= limit:
                    break
            
            return list(suggestions)[:limit]
            
        except Exception as e:
            self._logger.error(f"Failed to get search suggestions for '{partial_query}': {e}")
            return []
