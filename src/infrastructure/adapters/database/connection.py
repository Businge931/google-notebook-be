"""
Database Connection Manager

"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from ...config.settings import DatabaseSettings
from .base import Base


class DatabaseManager:
    """
    Database connection manager following Single Responsibility Principle.
    
    Handles database engine creation, session management, and connection lifecycle.
    """
    
    def __init__(self, settings: DatabaseSettings):
        """
        Initialize database manager with settings.
        
        Args:
            settings: Database configuration settings
        """
        self._settings = settings
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None
    
    def create_engine(self) -> AsyncEngine:
        """
        Create database engine with connection pooling.
        
        Returns:
            Configured async SQLAlchemy engine
        """
        if self._engine is None:
            # Engine configuration following Open/Closed Principle
            engine_kwargs = {
                "echo": False,  # Set to True for SQL debugging
            }
            
            # Handle SQLite vs PostgreSQL configurations (Interface Segregation)
            if "sqlite" in self._settings.database_url:
                # SQLite-specific configuration
                engine_kwargs.update({
                    "poolclass": StaticPool,
                    "connect_args": {"check_same_thread": False},
                })
            else:
                # PostgreSQL-specific configuration
                engine_kwargs.update({
                    "pool_size": self._settings.max_pool_size,
                    "max_overflow": 10,  # Default overflow connections
                    "pool_timeout": self._settings.pool_timeout,
                    "pool_recycle": 3600,  # Default 1 hour recycle time
                    "pool_pre_ping": False,  # Disable to avoid async context issues
                })
            
            self._engine = create_async_engine(
                self._settings.database_url,
                future=True,
                **engine_kwargs
            )
        
        return self._engine
    
    def create_session_factory(self) -> async_sessionmaker[AsyncSession]:
        """
        Create session factory for database operations.
        
        Returns:
            Configured session factory
        """
        if self._session_factory is None:
            engine = self.create_engine()
            self._session_factory = async_sessionmaker(
                engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False,
            )
        
        return self._session_factory
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get database session with automatic cleanup.
        
        Yields:
            Database session
            
        Raises:
            DatabaseConnectionError: If session creation fails
        """
        session_factory = self.create_session_factory()
        
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def create_tables(self) -> None:
        """
        Create all database tables.
        
        Used for initial setup and testing.
        """
        engine = self.create_engine()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def drop_tables(self) -> None:
        """
        Drop all database tables.
        
        Used for testing cleanup.
        """
        engine = self.create_engine()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    
    async def close(self) -> None:
        """
        Close database connections and cleanup resources.
        """
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
    
    async def health_check(self) -> bool:
        """
        Check database connectivity.
        
        Returns:
            True if database is accessible
        """
        try:
            async with self.get_session() as session:
                await session.execute("SELECT 1")
                return True
        except Exception:
            return False


# Global database manager instance (Singleton pattern for resource management)
_db_manager: Optional[DatabaseManager] = None


def get_database_manager(settings: DatabaseSettings) -> DatabaseManager:
    """
    Get or create database manager instance.
    
    Args:
        settings: Database configuration settings
        
    Returns:
        Database manager instance
    """
    global _db_manager
    
    if _db_manager is None:
        _db_manager = DatabaseManager(settings)
    
    return _db_manager


async def get_db_session(settings: DatabaseSettings) -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency injection function for database sessions.
    
    Args:
        settings: Database configuration settings
        
    Yields:
        Database session
    """
    db_manager = get_database_manager(settings)
    async with db_manager.get_session() as session:
        yield session
