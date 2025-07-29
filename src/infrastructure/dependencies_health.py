"""
Dependency Injection for Health Monitoring System

Provides dependency injection for production health monitoring,
chunk storage validation, and automated repair services.
"""
from functools import lru_cache
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.application.use_cases.chunk_storage_health_use_case import ChunkStorageHealthUseCase
from ..core.domain.services.chunk_storage_service import ProductionChunkStorageService
from ..core.domain.services.large_pdf_processing_service import (
    LargePDFProcessingService,
    LargePDFProcessingServiceFactory
)
from .adapters.database.repositories.document_repository_impl import DocumentRepositoryImpl
from .adapters.database.vector_repository_impl import FAISSVectorRepository
from .di import (
    get_database_session,
    get_document_repository,
    get_vector_repository
)


@lru_cache()
def get_production_chunk_storage_service(
    document_repository=None,
    vector_repository=None
) -> ProductionChunkStorageService:
    """
    Get production chunk storage service instance.
    
    Args:
        document_repository: Document repository (injected)
        vector_repository: Vector repository (injected)
        
    Returns:
        Production chunk storage service
    """
    # Note: In a real FastAPI app, these would be properly injected
    # For now, we'll create them directly when needed
    return ProductionChunkStorageService(
        document_repository=document_repository,
        vector_repository=vector_repository
    )


async def get_chunk_storage_health_use_case(
    chunk_storage_service: ProductionChunkStorageService = Depends(get_production_chunk_storage_service),
    document_repository = Depends(get_document_repository),
    vector_repository = Depends(get_vector_repository)
) -> ChunkStorageHealthUseCase:
    """Get chunk storage health use case with dependency injection."""
    return ChunkStorageHealthUseCase(
        chunk_storage_service=chunk_storage_service,
        document_repository=document_repository,
        vector_repository=vector_repository
    )


async def get_large_pdf_processing_service(
    document_repository = Depends(get_document_repository),
    file_storage = Depends(get_file_storage),
    pdf_parsing_service = Depends(get_pdf_parsing_service),
    embedding_service = Depends(get_embedding_service),
    vector_repository = Depends(get_vector_repository)
) -> LargePDFProcessingService:
    """Get large PDF processing service with dependency injection."""
    return LargePDFProcessingServiceFactory.create_optimized(
        document_repository=document_repository,
        file_storage=file_storage,
        pdf_parsing_service=pdf_parsing_service,
        embedding_service=embedding_service,
        vector_repository=vector_repository,
        max_memory_mb=512
    )
