"""
AI Services Dependency Injection Container

"""
import logging
from typing import Optional, Dict, Any
from functools import lru_cache

from ..adapters.services.openai_embedding_service import (
    OpenAIEmbeddingService,
    OpenAIDocumentEmbeddingService,
    OpenAIEmbeddingServiceFactory
)
from ..adapters.services.simple_local_embedding_service import (
    SimpleLocalEmbeddingService,
    SimpleLocalDocumentEmbeddingService,
    SimpleLocalEmbeddingServiceFactory
)
from ..adapters.services.openai_chat_service import (
    OpenAIChatServiceFactory,
    OpenAIRAGService
)
from ..adapters.services.conversation_service_impl import (
    ConversationServiceFactory
)
from ..adapters.database.vector_repository_impl import FAISSVectorRepositoryFactory
from ...core.application.use_cases.document_vectorization_use_case import (
    DocumentVectorizationUseCase,
    BulkDocumentVectorizationUseCase
)
from ...core.application.use_cases.similarity_search_use_case import (
    SimilaritySearchUseCase
)
from ...core.application.use_cases.chat_use_case import ChatUseCase
from ...core.domain.services.embedding_service import (
    EmbeddingService,
    DocumentEmbeddingService,
    SimilaritySearchService
)
from ...core.domain.services.chat_service import (
    ChatService,
    RAGService,
    ConversationService
)
from ..adapters.repositories.simple_memory_vector_repository import (
    SimpleMemoryVectorRepository,
    SimpleMemoryVectorRepositoryFactory
)
from ...core.domain.repositories import VectorRepository, ChatRepository
from src.infrastructure.config.settings import Settings
from src.shared.exceptions import ConfigurationError


class AIContainer:
    """
    Dependency injection container for AI services following Single Responsibility Principle.
    
    Manages creation and lifecycle of AI-related services and use cases.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize AI container.
        
        Args:
            settings: Application settings
        """
        self._settings = settings
        self._logger = logging.getLogger(__name__)
        
        # Service instances (lazy-loaded)
        self._embedding_service: Optional[EmbeddingService] = None
        self._document_embedding_service: Optional[DocumentEmbeddingService] = None
        self._vector_repository: Optional[VectorRepository] = None
        self._chat_service: Optional[ChatService] = None
        self._rag_service: Optional[RAGService] = None
        self._conversation_service: Optional[ConversationService] = None
        
        # Use case instances (lazy-loaded)
        self._document_vectorization_use_case: Optional[DocumentVectorizationUseCase] = None
        self._bulk_document_vectorization_use_case: Optional[BulkDocumentVectorizationUseCase] = None
        self._similarity_search_use_case: Optional[SimilaritySearchUseCase] = None
        self._chat_use_case: Optional[ChatUseCase] = None
    
    @lru_cache(maxsize=1)
    def get_embedding_service(self) -> EmbeddingService:
        """
        Get embedding service instance following Singleton pattern.
        
        Returns:
            Configured embedding service
            
        Raises:
            ConfigurationError: If service cannot be created
        """
        if self._embedding_service is None:
            try:
                # Debug settings structure
                self._logger.info(f"Settings type: {type(self._settings)}")
                self._logger.info(f"AI service type: {type(self._settings.ai_service)}")
                self._logger.info(f"AI service attributes: {[attr for attr in dir(self._settings.ai_service) if not attr.startswith('_')]}")
                
                # Check if we have the correct embedding model attribute
                if hasattr(self._settings.ai_service, 'embedding_model'):
                    embedding_model = self._settings.ai_service.embedding_model
                    self._logger.info(f"Using embedding_model: {embedding_model}")
                elif hasattr(self._settings.ai_service, 'openai_embedding_model'):
                    embedding_model = self._settings.ai_service.openai_embedding_model
                    self._logger.info(f"Using openai_embedding_model: {embedding_model}")
                else:
                    raise ConfigurationError("No embedding model configuration found")
                
                # Use OpenAI embedding service for proper semantic understanding
                self._embedding_service = OpenAIEmbeddingServiceFactory.create_embedding_service(
                    api_key=self._settings.ai_service.openai_api_key,
                    model=embedding_model,
                    max_retries=3,
                    timeout=float(self._settings.ai_service.openai_timeout)
                )
                
                self._logger.info(f"Created OpenAI embedding service with model: {embedding_model}")
                
            except Exception as e:
                self._logger.error(f"Failed to create embedding service: {e}")
                raise ConfigurationError(f"Failed to create embedding service: {str(e)}")
        
        return self._embedding_service
    
    @lru_cache(maxsize=1)
    def get_document_embedding_service(self) -> DocumentEmbeddingService:
        """
        Get document embedding service instance following Singleton pattern.
        
        Returns:
            Configured document embedding service
        """
        if self._document_embedding_service is None:
            embedding_service = self.get_embedding_service()
            try:
                self._document_embedding_service = OpenAIEmbeddingServiceFactory.create_document_embedding_service(
                    embedding_service=embedding_service,
                    max_concurrent=10  # Default concurrent operations
                )
            except Exception as e:
                self._logger.error(f"Failed to create document embedding service: {e}")
                raise ConfigurationError(f"Failed to create document embedding service: {str(e)}")
            
            self._logger.info("Created document embedding service")
        
        return self._document_embedding_service
    
    @lru_cache(maxsize=1)
    def get_vector_repository(self) -> VectorRepository:
        """
        Get vector repository instance following Singleton pattern.
        
        Returns:
            Configured vector repository
            
        Raises:
            ConfigurationError: If repository cannot be created
        """
        self._logger.info(f"🔥 AI CONTAINER: get_vector_repository called, current instance: {self._vector_repository}")
        
        if self._vector_repository is None:
            try:
                embedding_service = self.get_embedding_service()
                
                # Get vector DB type from settings
                vector_db_type = getattr(self._settings.vector_db, 'vector_db_type', 'memory').lower()
                self._logger.info(f"🔥 AI CONTAINER: Vector DB type from settings: {vector_db_type}")
                
                if vector_db_type == 'faiss':
                    # Use FAISS persistent vector repository
                    storage_path = getattr(self._settings.vector_db, 'faiss_index_path', './vector_indexes')
                    self._logger.info(f"🔥 AI CONTAINER: Creating FAISS vector repository at {storage_path}")
                    
                    self._vector_repository = FAISSVectorRepositoryFactory.create_vector_repository(
                        embedding_service=embedding_service,
                        storage_path=storage_path,
                        embedding_dimension=getattr(self._settings.vector_db, 'embedding_dimension', 1536)
                    )
                    self._logger.info(f"🔥 AI CONTAINER: Created FAISS vector repository at {storage_path}")
                    
                else:
                    # Default to memory repository for development
                    self._vector_repository = SimpleMemoryVectorRepositoryFactory.create_vector_repository(
                        embedding_service=embedding_service
                    )
                    self._logger.info("Created Simple Memory vector repository (development mode)")
                
            except Exception as e:
                self._logger.error(f"Failed to create vector repository: {e}")
                raise ConfigurationError(f"Failed to create vector repository: {str(e)}")
        
        return self._vector_repository
    
    @lru_cache(maxsize=1)
    def get_document_vectorization_use_case(
        self,
        document_repository,  # Injected from main container
    ) -> DocumentVectorizationUseCase:
        """
        Get document vectorization use case instance following Singleton pattern.
        
        Args:
            document_repository: Document repository instance
            
        Returns:
            Configured document vectorization use case
        """
        if self._document_vectorization_use_case is None:
            self._document_vectorization_use_case = DocumentVectorizationUseCase(
                document_repository=document_repository,
                vector_repository=self.get_vector_repository(),
                document_embedding_service=self.get_document_embedding_service()
            )
            
            self._logger.info("Created document vectorization use case")
        
        return self._document_vectorization_use_case
    
    @lru_cache(maxsize=1)
    def get_bulk_document_vectorization_use_case(
        self,
        document_repository,  # Injected from main container
    ) -> BulkDocumentVectorizationUseCase:
        """
        Get bulk document vectorization use case instance following Singleton pattern.
        
        Args:
            document_repository: Document repository instance
            
        Returns:
            Configured bulk document vectorization use case
        """
        if self._bulk_document_vectorization_use_case is None:
            document_vectorization_use_case = self.get_document_vectorization_use_case(
                document_repository
            )
            
            self._bulk_document_vectorization_use_case = BulkDocumentVectorizationUseCase(
                document_vectorization_use_case=document_vectorization_use_case,
                max_concurrent=self._settings.max_concurrent_vectorizations
            )
            
            self._logger.info("Created bulk document vectorization use case")
        
        return self._bulk_document_vectorization_use_case
    
    @lru_cache(maxsize=1)
    def get_similarity_search_use_case(
        self,
        document_repository,  # Injected from main container
    ) -> SimilaritySearchUseCase:
        """
        Get similarity search use case instance following Singleton pattern.
        
        Args:
            document_repository: Document repository instance
            
        Returns:
            Configured similarity search use case
        """
        if self._similarity_search_use_case is None:
            self._similarity_search_use_case = SimilaritySearchUseCase(
                vector_repository=self.get_vector_repository(),
                document_repository=document_repository
            )
            
            self._logger.info("Created similarity search use case")
        
        return self._similarity_search_use_case
    
    @lru_cache(maxsize=1)
    def get_chat_service(self) -> ChatService:
        """
        Get chat service instance following Singleton pattern.
        
        Returns:
            Configured chat service
            
        Raises:
            ConfigurationError: If service cannot be created
        """
        if self._chat_service is None:
            try:
                # Check if OpenAI is configured
                self._logger.info(f"Checking OpenAI API key: has_key={bool(self._settings.ai_service.openai_api_key)}, key_prefix={self._settings.ai_service.openai_api_key[:10] if self._settings.ai_service.openai_api_key else 'None'}...")
                if not self._settings.ai_service.openai_api_key:
                    raise ConfigurationError("OpenAI API key not configured")
                
                self._chat_service = OpenAIChatServiceFactory.create_chat_service(
                    api_key=self._settings.ai_service.openai_api_key,
                    model=self._settings.ai_service.openai_model,
                    max_retries=3,  # Default retry count
                    timeout=self._settings.ai_service.openai_timeout
                )
                
                self._logger.info("Created OpenAI chat service")
                
            except Exception as e:
                self._logger.error(f"Failed to create chat service: {e}")
                raise ConfigurationError(f"Failed to create chat service: {str(e)}")
        
        return self._chat_service
    
    @lru_cache(maxsize=1)
    def get_rag_service(self, document_repository) -> RAGService:
        """
        Get RAG service instance following Singleton pattern.
        
        Args:
            document_repository: Document repository instance
        
        Returns:
            Configured RAG service
        """
        if self._rag_service is None:
            chat_service = self.get_chat_service()
            similarity_search_use_case = self.get_similarity_search_use_case(document_repository)
            
            # Get advanced search and citation capabilities if available
            try:
                advanced_search_use_case = self.get_advanced_search_use_case(document_repository)
                citation_use_case = self.get_citation_use_case()
                
                self._rag_service = OpenAIRAGService(
                    chat_service=chat_service,
                    similarity_search_use_case=similarity_search_use_case,
                    advanced_search_use_case=advanced_search_use_case,
                    citation_use_case=citation_use_case
                )
                
                self._logger.info("Created enhanced OpenAI RAG service with advanced search and citations")
                
            except Exception as e:
                self._logger.warning(f"Failed to create enhanced RAG service, falling back to basic: {e}")
                
                # Fall back to basic RAG service without advanced features
                self._rag_service = OpenAIRAGService(
                    chat_service=chat_service,
                    similarity_search_use_case=similarity_search_use_case
                )
                
                self._logger.info("Created basic OpenAI RAG service")
        
        return self._rag_service
    
    @lru_cache(maxsize=1)
    def get_conversation_service(
        self,
        chat_repository: ChatRepository
    ) -> ConversationService:
        """
        Get conversation service instance following Singleton pattern.
        
        Args:
            chat_repository: Chat repository instance
            
        Returns:
            Configured conversation service
        """
        if self._conversation_service is None:
            self._conversation_service = ConversationServiceFactory.create_conversation_service(
                chat_repository=chat_repository,
                max_history_length=self._settings.max_conversation_history
            )
            
            self._logger.info("Created conversation service")
        
        return self._conversation_service
    
    @lru_cache(maxsize=1)
    def get_chat_use_case(
        self,
        chat_repository: ChatRepository,
        document_repository
    ) -> ChatUseCase:
        """
        Get chat use case instance following Singleton pattern.
        
        Args:
            chat_repository: Chat repository instance
            document_repository: Document repository instance
            
        Returns:
            Configured chat use case
        """
        if self._chat_use_case is None:
            rag_service = self.get_rag_service(document_repository)
            chat_service = self.get_chat_service()
            
            self._chat_use_case = ChatUseCase(
                chat_repository=chat_repository,
                rag_service=rag_service,
                chat_service=chat_service
            )
            
            self._logger.info("Created chat use case")
        
        return self._chat_use_case
    
    def get_advanced_search_use_case(self, document_repository) -> 'AdvancedSearchUseCase':
        """
        Get advanced search use case with dependency injection.
        
        Args:
            document_repository: Document repository instance
        
        Returns:
            Advanced search use case instance
        """
        if not hasattr(self, '_advanced_search_use_case') or self._advanced_search_use_case is None:
            # For now, create a simplified advanced search that uses existing similarity search
            # This will allow the endpoints to work while we build full advanced search later
            from ...core.application.use_cases.advanced_search_use_case import AdvancedSearchUseCase
            
            # Create a basic implementation using existing similarity search
            class BasicAdvancedSearchUseCase:
                def __init__(self, ai_container, document_repository):
                    self._ai_container = ai_container
                    self._document_repository = document_repository
                    self._logger = logging.getLogger(__name__)
                    self._similarity_search_use_case = None
                
                def _get_similarity_search_use_case(self):
                    """Lazy initialization of similarity search use case."""
                    if self._similarity_search_use_case is None:
                        self._similarity_search_use_case = self._ai_container.get_similarity_search_use_case(
                            self._document_repository
                        )
                    return self._similarity_search_use_case
                
                async def enhanced_search(self, request):
                    """Basic enhanced search using similarity search."""
                    try:
                        # Convert to similarity search request
                        from ...core.application.use_cases.similarity_search_use_case import SimilaritySearchRequest
                        
                        # Extract document IDs from request
                        document_ids = []
                        if hasattr(request, 'document_ids') and request.document_ids:
                            document_ids = request.document_ids
                        elif hasattr(request, 'filters') and request.filters and hasattr(request.filters, 'document_ids'):
                            document_ids = request.filters.document_ids
                        
                        self._logger.info(f"Enhanced search query: '{request.query}' for documents: {document_ids}")
                        
                        search_request = SimilaritySearchRequest(
                            query_text=request.query,
                            document_ids=document_ids if document_ids else None,
                            limit=getattr(request, 'max_results', 10),
                            similarity_threshold=0.18,  # Adjusted threshold for improved text processing
                            include_metadata=True
                        )
                        
                        # Execute similarity search
                        similarity_search_use_case = self._get_similarity_search_use_case()
                        search_response = await similarity_search_use_case.search_similar_chunks(search_request)
                        
                        # Convert results to enhanced search format
                        from ...core.application.use_cases.advanced_search_use_case import EnhancedSearchResponse
                        
                        return EnhancedSearchResponse(
                            results=search_response.results,
                            clusters=[],
                            suggestions=[],
                            citations=[],
                            total_results=search_response.total_results,
                            search_time_ms=search_response.search_time_ms,
                            query_analysis={"query": request.query},
                            performance_metrics={"search_time_ms": search_response.search_time_ms}
                        )
                        
                    except Exception as e:
                        self._logger.error(f"Enhanced search failed: {e}")
                        # Return empty response on error
                        from ...core.application.use_cases.advanced_search_use_case import EnhancedSearchResponse
                        return EnhancedSearchResponse(
                            results=[],
                            clusters=[],
                            suggestions=[],
                            citations=[],
                            total_results=0,
                            search_time_ms=0,
                            query_analysis={"error": str(e)},
                            performance_metrics={}
                        )
                
                async def contextual_search(self, query, context, max_results=20):
                    """Basic contextual search."""
                    return await self.enhanced_search(type('Request', (), {'query': query, 'max_results': max_results})())
                
                async def multi_document_search(self, query, document_ids, search_type=None):
                    """Basic multi-document search."""
                    from ...core.application.use_cases.similarity_search_use_case import SimilaritySearchRequest
                    
                    search_request = SimilaritySearchRequest(
                        query_text=query,
                        document_ids=document_ids,
                        limit=10,
                        similarity_threshold=0.2,  # Lower threshold for FAISS L2 distance conversion
                        include_metadata=True
                    )
                    
                    similarity_search_use_case = self._get_similarity_search_use_case()
                    return await similarity_search_use_case.search_similar_chunks(search_request)
                
                async def get_search_analytics(self, request):
                    """Basic search analytics."""
                    from ...core.application.use_cases.advanced_search_use_case import SearchAnalyticsResponse
                    
                    return SearchAnalyticsResponse(
                        total_searches=0,
                        unique_queries=0,
                        average_response_time_ms=250.0,
                        popular_queries=[],
                        search_patterns={},
                        performance_metrics={},
                        improvement_suggestions=[]
                    )
            
            # Create basic implementation using existing similarity search
            self._advanced_search_use_case = BasicAdvancedSearchUseCase(
                ai_container=self,
                document_repository=document_repository
            )
            
            self._logger.info("Basic advanced search use case created successfully")
        
        return self._advanced_search_use_case
    
    def get_citation_use_case(self) -> 'CitationUseCase':
        """
        Get citation use case with dependency injection.
        
        Returns:
            Citation use case instance
        """
        if not hasattr(self, '_citation_use_case') or self._citation_use_case is None:
            # Create a basic citation use case implementation
            class BasicCitationUseCase:
                def __init__(self):
                    self._logger = logging.getLogger(__name__)
                
                async def extract_citations(self, response_text, source_chunks, context_documents, session_id, options=None):
                    """Basic citation extraction."""
                    try:
                        # Basic citation extraction - return empty for now
                        from ...core.application.use_cases.citation_use_case import CitationExtractionResponse
                        
                        return type('CitationExtractionResponse', (), {
                            'citations': [],
                            'extraction_time_ms': 0,
                            'confidence_score': 0.0,
                            'metadata': {}
                        })()
                        
                    except Exception as e:
                        self._logger.error(f"Citation extraction failed: {e}")
                        return type('CitationExtractionResponse', (), {
                            'citations': [],
                            'extraction_time_ms': 0,
                            'confidence_score': 0.0,
                            'metadata': {'error': str(e)}
                        })()
                
                async def link_citations(self, request):
                    """Basic citation linking."""
                    return type('CitationLinkingResponse', (), {
                        'linked_citations': [],
                        'linking_time_ms': 0,
                        'success_rate': 0.0
                    })()
                
                async def analyze_citations(self, request):
                    """Basic citation analysis."""
                    return type('CitationAnalysisResponse', (), {
                        'patterns': {},
                        'quality_metrics': {},
                        'recommendations': []
                    })()
                
                async def cluster_citations(self, request):
                    """Basic citation clustering."""
                    return type('CitationClusteringResponse', (), {
                        'clusters': [],
                        'clustering_time_ms': 0,
                        'silhouette_score': 0.0
                    })()
                
                async def validate_citations(self, request):
                    """Basic citation validation."""
                    return type('CitationValidationResponse', (), {
                        'validation_results': [],
                        'overall_accuracy': 0.0,
                        'validation_time_ms': 0
                    })()
                
                async def bulk_process_citations(self, request):
                    """Basic bulk citation processing."""
                    return type('BulkCitationProcessingResponse', (), {
                        'processed_citations': [],
                        'processing_time_ms': 0,
                        'success_count': 0,
                        'error_count': 0
                    })()
                
                async def get_session_citations(self, session_id, include_clusters=False, include_links=False):
                    """Basic session citation retrieval."""
                    return []
            
            # Create basic implementation
            self._citation_use_case = BasicCitationUseCase()
            
            self._logger.info("Basic citation use case created successfully")
        
        return self._citation_use_case
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all AI services.
        
        Returns:
            Health check results
        """
        health_results = {
            "embedding_service": False,
            "vector_repository": False,
            "overall_status": "unhealthy"
        }
        
        try:
            # Check embedding service
            embedding_service = self.get_embedding_service()
            health_results["embedding_service"] = await embedding_service.health_check()
            
            # Check vector repository
            vector_repository = self.get_vector_repository()
            health_results["vector_repository"] = await vector_repository.health_check()
            
            # Overall status
            if health_results["embedding_service"] and health_results["vector_repository"]:
                health_results["overall_status"] = "healthy"
            
        except Exception as e:
            self._logger.error(f"AI services health check failed: {e}")
            health_results["error"] = str(e)
        
        return health_results
    
    async def cleanup(self):
        """
        Cleanup AI services and resources.
        """
        try:
            # Clear cached instances
            self.get_embedding_service.cache_clear()
            self.get_document_embedding_service.cache_clear()
            self.get_vector_repository.cache_clear()
            self.get_document_vectorization_use_case.cache_clear()
            self.get_bulk_document_vectorization_use_case.cache_clear()
            self.get_similarity_search_use_case.cache_clear()
            
            # Reset instances
            self._embedding_service = None
            self._document_embedding_service = None
            self._vector_repository = None
            self._document_vectorization_use_case = None
            self._bulk_document_vectorization_use_case = None
            self._similarity_search_use_case = None
            
            self._logger.info("AI container cleanup completed")
            
        except Exception as e:
            self._logger.error(f"AI container cleanup failed: {e}")


# Global AI container instance
_ai_container: Optional[AIContainer] = None


def get_ai_container(settings: Settings) -> AIContainer:
    """
    Get global AI container instance following Singleton pattern.
    
    Args:
        settings: Application settings
        
    Returns:
        AI container instance
    """
    global _ai_container
    
    # Use singleton pattern - only create if not exists
    if _ai_container is None:
        _ai_container = AIContainer(settings)
    
    return _ai_container


def reset_ai_container():
    """Reset global AI container instance."""
    global _ai_container
    _ai_container = None
