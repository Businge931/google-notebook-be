"""
Dependency Injection Container

"""
from typing import Dict, Any, TypeVar, Type, Optional
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession

from ..config.settings import Settings, get_settings
from ...shared.config import create_settings_from_env
from ..adapters.database.connection import DatabaseManager, get_database_manager
from ..adapters.storage.file_storage import FileStorage, FileStorageFactory
from ..adapters.database.repositories import DocumentRepositoryImpl, ChatRepositoryImpl
from ...core.domain.repositories import DocumentRepository, ChatRepository
from ...core.domain.services import DocumentProcessingService
from .ai_container import AIContainer, get_ai_container

T = TypeVar('T')


class DIContainer:
    """
    Dependency Injection Container following Single Responsibility Principle.
    
    Manages registration and resolution of dependencies while maintaining
    loose coupling between components.
    """
    
    def __init__(self):
        """Initialize the DI container."""
        self._services: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
        self._settings: Optional[Settings] = None
        self._db_manager: Optional[DatabaseManager] = None
        self._file_storage: Optional[FileStorage] = None
        self._ai_container: Optional[AIContainer] = None
    
    def register_settings(self, settings: Settings) -> None:
        """
        Register application settings.
        
        Args:
            settings: Application configuration settings
        """
        self._settings = settings
        self._services['settings'] = settings
    
    def register_database_manager(self, db_manager: DatabaseManager) -> None:
        """
        Register database manager.
        
        Args:
            db_manager: Database connection manager
        """
        self._db_manager = db_manager
        self._services['db_manager'] = db_manager
    
    def register_file_storage(self, file_storage: FileStorage) -> None:
        """
        Register file storage service.
        
        Args:
            file_storage: File storage implementation
        """
        self._file_storage = file_storage
        self._services['file_storage'] = file_storage
    
    def get_settings(self) -> Settings:
        """
        Get application settings.
        
        Returns:
            Application settings
            
        Raises:
            ValueError: If settings not registered
        """
        if self._settings is None:
            raise ValueError("Settings not registered in DI container")
        return self._settings
    
    def get_database_manager(self) -> DatabaseManager:
        """
        Get database manager.
        
        Returns:
            Database manager instance
            
        Raises:
            ValueError: If database manager not registered
        """
        if self._db_manager is None:
            raise ValueError("Database manager not registered in DI container")
        return self._db_manager
    
    def get_file_storage(self) -> FileStorage:
        """
        Get file storage service.
        
        Returns:
            File storage implementation
            
        Raises:
            ValueError: If file storage not registered
        """
        if self._file_storage is None:
            raise ValueError("File storage not registered in DI container")
        return self._file_storage
    
    def get_document_repository(self, session: AsyncSession) -> DocumentRepository:
        """
        Get document repository with database session.
        
        Args:
            session: Database session
            
        Returns:
            Document repository implementation
        """
        return DocumentRepositoryImpl(session)
    
    def get_chat_repository(self, session: AsyncSession) -> ChatRepository:
        """
        Get chat repository with database session.
        
        Args:
            session: Database session
            
        Returns:
            Chat repository implementation
        """
        return ChatRepositoryImpl(session)
    
    def get_document_processing_service(self) -> DocumentProcessingService:
        """
        Get document processing service.
        
        Returns:
            Document processing service instance
        """
        # Return singleton instance
        if 'document_processing_service' not in self._singletons:
            self._singletons['document_processing_service'] = DocumentProcessingService()
        
        return self._singletons['document_processing_service']
    
    def get_ai_container(self) -> AIContainer:
        """
        Get AI container instance.
        
        Returns:
            AI container instance
            
        Raises:
            ValueError: If settings not registered
        """
        if self._ai_container is None:
            if self._settings is None:
                raise ValueError("Settings not registered in DI container")
            self._ai_container = get_ai_container(self._settings)
        return self._ai_container
    
    @asynccontextmanager
    async def get_repositories(self):
        """
        Get repositories with managed database session.
        
        Yields:
            Tuple of (document_repository, chat_repository)
        """
        db_manager = self.get_database_manager()
        
        async with db_manager.get_session() as session:
            document_repo = self.get_document_repository(session)
            chat_repo = self.get_chat_repository(session)
            
            yield document_repo, chat_repo
    
    async def initialize(self) -> None:
        """
        Initialize the DI container with default services.
        
        Sets up all required dependencies for the application.
        """
        # Load settings (use fresh instance to avoid cache issues)
        if self._settings is None:
            settings = get_settings()  # Use infrastructure settings instead of shared config
            self.register_settings(settings)
        
        # Initialize database manager
        if self._db_manager is None:
            db_manager = get_database_manager(self._settings.database)
            self.register_database_manager(db_manager)
        
        # Initialize file storage
        if self._file_storage is None:
            file_storage = FileStorageFactory.create_storage(self._settings.file_storage)
            self.register_file_storage(file_storage)
    
    async def cleanup(self) -> None:
        """
        Cleanup resources managed by the container.
        """
        if self._db_manager:
            await self._db_manager.close()
        
        # Clear singletons
        self._singletons.clear()


# Global container instance
_container: Optional[DIContainer] = None


def get_container() -> DIContainer:
    """
    Get or create the global DI container.
    
    Returns:
        DI container instance
    """
    global _container
    
    if _container is None:
        _container = DIContainer()
    
    return _container


async def initialize_container() -> DIContainer:
    """
    Initialize the global DI container with default services.
    
    Returns:
        Initialized DI container
    """
    container = get_container()
    await container.initialize()
    return container


async def cleanup_container() -> None:
    """
    Cleanup the global DI container.
    """
    global _container
    
    if _container:
        await _container.cleanup()
        _container = None


# Dependency injection functions for FastAPI
async def get_document_repository() -> DocumentRepository:
    """
    FastAPI dependency for document repository.
    
    Returns:
        Document repository instance
    """
    container = get_container()
    db_manager = container.get_database_manager()
    
    async with db_manager.get_session() as session:
        yield container.get_document_repository(session)


async def get_chat_repository() -> ChatRepository:
    """
    FastAPI dependency for chat repository.
    
    Returns:
        Chat repository instance
    """
    container = get_container()
    db_manager = container.get_database_manager()
    
    async with db_manager.get_session() as session:
        yield container.get_chat_repository(session)


async def get_file_storage_service() -> FileStorage:
    """
    FastAPI dependency for file storage service.
    
    Returns:
        File storage service instance
    """
    container = get_container()
    return container.get_file_storage()


async def get_document_processing_service() -> DocumentProcessingService:
    """
    FastAPI dependency for document processing service.
    
    Returns:
        Document processing service instance
    """
    container = get_container()
    return container.get_document_processing_service()
