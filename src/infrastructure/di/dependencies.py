"""
FastAPI Dependencies for AI Services

"""
from fastapi import Depends
from typing import Annotated
from sqlalchemy.ext.asyncio import AsyncSession

from .container import DIContainer

from ...core.application.use_cases.document_vectorization_use_case import (
    DocumentVectorizationUseCase,
    BulkDocumentVectorizationUseCase
)
from ...core.application.use_cases.similarity_search_use_case import (
    SimilaritySearchUseCase
)
from ...core.application.use_cases.chat_use_case import ChatUseCase
from ...core.domain.services.chat_service import (
    ChatService,
    RAGService,
    ConversationService
)
from ...core.domain.services.embedding_service import (
    EmbeddingService,
    DocumentEmbeddingService
)
from ...core.domain.repositories import VectorRepository
from .container import get_container
from .ai_container import get_ai_container
from src.shared.config import get_settings


async def get_database_session(
    container: DIContainer = Depends(get_container)
):
    """
    Get database session dependency with proper lifecycle management.
    
    Args:
        container: DI container
        
    Yields:
        Database session
    """
    db_manager = container.get_database_manager()
    async with db_manager.get_session() as session:
        yield session


def get_embedding_service() -> EmbeddingService:
    """Get embedding service dependency."""
    settings = get_settings()
    ai_container = get_ai_container(settings)
    return ai_container.get_embedding_service()


def get_document_embedding_service() -> DocumentEmbeddingService:
    """Get document embedding service dependency."""
    settings = get_settings()
    ai_container = get_ai_container(settings)
    return ai_container.get_document_embedding_service()


def get_vector_repository() -> VectorRepository:
    """Get vector repository dependency."""
    settings = get_settings()
    ai_container = get_ai_container(settings)
    return ai_container.get_vector_repository()


async def get_document_vectorization_use_case(
    container: DIContainer = Depends(get_container)
) -> DocumentVectorizationUseCase:
    """Get document vectorization use case dependency."""
    settings = get_settings()
    ai_container = get_ai_container(settings)
    db_manager = container.get_database_manager()
    
    async with db_manager.get_session() as session:
        document_repository = container.get_document_repository(session)
        return ai_container.get_document_vectorization_use_case(document_repository)


async def get_bulk_document_vectorization_use_case(
    container: DIContainer = Depends(get_container)
) -> BulkDocumentVectorizationUseCase:
    """Get bulk document vectorization use case dependency."""
    settings = get_settings()
    ai_container = get_ai_container(settings)
    db_manager = container.get_database_manager()
    
    async with db_manager.get_session() as session:
        document_repository = container.get_document_repository(session)
        return ai_container.get_bulk_document_vectorization_use_case(document_repository)


async def get_similarity_search_use_case(
    container: DIContainer = Depends(get_container)
) -> SimilaritySearchUseCase:
    """
    Get similarity search use case dependency.
    
    Args:
        container: DI container
        
    Returns:
        Similarity search use case instance
    """
    ai_container = container.get_ai_container()
    return ai_container.get_similarity_search_use_case()


async def get_advanced_search_use_case(
    container: DIContainer = Depends(get_container)
) -> 'AdvancedSearchUseCase':
    """
    Get advanced search use case dependency.
    
    Args:
        container: DI container
        
    Returns:
        Advanced search use case instance
    """
    # Use the shared AI container from the DI container instead of creating a new one
    ai_container = container.get_ai_container()
    db_manager = container.get_database_manager()
    
    async with db_manager.get_session() as session:
        document_repository = container.get_document_repository(session)
        return ai_container.get_advanced_search_use_case(document_repository)


async def get_chat_service(
    container: DIContainer = Depends(get_container)
) -> ChatService:
    """
    Get chat service dependency.
    
    Args:
        container: DI container
        
    Returns:
        Chat service instance
    """
    ai_container = container.get_ai_container()
    return ai_container.get_chat_service()


async def get_rag_service(
    container: DIContainer = Depends(get_container)
) -> RAGService:
    """
    Get RAG service dependency.
    
    Args:
        container: DI container
        
    Returns:
        RAG service instance
    """
    ai_container = container.get_ai_container()
    db_manager = container.get_database_manager()
    
    async with db_manager.get_session() as session:
        document_repository = container.get_document_repository(session)
        return ai_container.get_rag_service(document_repository)


async def get_conversation_service(
    container: DIContainer = Depends(get_container)
) -> ConversationService:
    """
    Get conversation service dependency.
    
    Args:
        container: DI container
        
    Returns:
        Conversation service instance
    """
    ai_container = container.get_ai_container()
    db_manager = container.get_database_manager()
    
    async with db_manager.get_session() as session:
        chat_repository = container.get_chat_repository(session)
        return ai_container.get_conversation_service(chat_repository)


async def get_citation_use_case(
    container: DIContainer = Depends(get_container)
) -> 'CitationUseCase':
    """
    Get citation use case dependency.
    
    Args:
        container: DI container
        
    Returns:
        Citation use case instance
    """
    db_manager = container.get_database_manager()
    
    async with db_manager.get_session() as session:
        document_repository = container.get_document_repository(session)
        chat_repository = container.get_chat_repository(session)
        from ...core.application.use_cases.citation_use_case import CitationUseCase
        return CitationUseCase(document_repository, chat_repository)


async def get_chat_use_case(
    container: DIContainer = Depends(get_container)
) -> ChatUseCase:
    """
    Get chat use case dependency.
    
    Args:
        container: DI container
        
    Returns:
        Chat use case instance
    """
    ai_container = container.get_ai_container()
    db_manager = container.get_database_manager()
    
    async with db_manager.get_session() as session:
        chat_repository = container.get_chat_repository(session)
        document_repository = container.get_document_repository(session)
        return ai_container.get_chat_use_case(chat_repository, document_repository)


# Type aliases for cleaner dependency injection
EmbeddingServiceDep = Annotated[EmbeddingService, Depends(get_embedding_service)]
DocumentEmbeddingServiceDep = Annotated[DocumentEmbeddingService, Depends(get_document_embedding_service)]
VectorRepositoryDep = Annotated[VectorRepository, Depends(get_vector_repository)]
DocumentVectorizationUseCaseDep = Annotated[DocumentVectorizationUseCase, Depends(get_document_vectorization_use_case)]
BulkDocumentVectorizationUseCaseDep = Annotated[BulkDocumentVectorizationUseCase, Depends(get_bulk_document_vectorization_use_case)]
SimilaritySearchUseCaseDep = Annotated[SimilaritySearchUseCase, Depends(get_similarity_search_use_case)]
