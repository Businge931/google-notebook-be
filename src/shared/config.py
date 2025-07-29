"""
Application Configuration

Follows S.O.L.I.D principles:
- Single Responsibility: Only handles configuration management
- Open/Closed: Can be extended with new configuration sections
- Liskov Substitution: All config classes follow same patterns
- Interface Segregation: Grouped by functionality
- Dependency Inversion: Depends on Pydantic abstractions
"""
import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from pydantic import field_validator
from functools import lru_cache

from .constants import (
    AIConstants,
    DatabaseConstants,
    FileConstants,
    TextConstants,
    APIConstants,
    LoggingConstants,
    EnvironmentConstants
)


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    database_url: str = Field(
        default="postgresql+asyncpg://notebooklm_user:password@localhost:5432/notebooklm_db",
        env="DATABASE_URL",
        description="Database connection URL"
    )
    pool_size: int = Field(
        default=DatabaseConstants.DEFAULT_POOL_SIZE,
        env="DB_POOL_SIZE",
        description="Database connection pool size"
    )
    max_overflow: int = Field(
        default=DatabaseConstants.DEFAULT_MAX_OVERFLOW,
        env="DB_MAX_OVERFLOW",
        description="Maximum connection pool overflow"
    )
    pool_timeout: int = Field(
        default=DatabaseConstants.DEFAULT_POOL_TIMEOUT,
        env="DB_POOL_TIMEOUT",
        description="Connection pool timeout in seconds"
    )
    pool_recycle: int = Field(
        default=DatabaseConstants.DEFAULT_POOL_RECYCLE,
        env="DB_POOL_RECYCLE",
        description="Connection pool recycle time in seconds"
    )
    echo_sql: bool = Field(
        default=False,
        env="DB_ECHO_SQL",
        description="Echo SQL queries to logs"
    )


class FileStorageSettings(BaseSettings):
    """File storage configuration settings."""
    
    storage_type: str = Field(
        default="local",
        env="STORAGE_TYPE",
        description="Storage type (local, s3, etc.)"
    )
    upload_dir: str = Field(
        default="./uploads",
        env="UPLOAD_DIR",
        description="Local upload directory for files"
    )
    base_url: str = Field(
        default="http://localhost:8000/files",
        env="STORAGE_BASE_URL",
        description="Base URL for file access"
    )
    max_file_size_mb: int = Field(
        default=FileConstants.MAX_FILE_SIZE_MB,
        env="MAX_FILE_SIZE_MB",
        description="Maximum file size in MB"
    )
    allowed_mime_types: List[str] = Field(
        default=FileConstants.ALLOWED_MIME_TYPES,
        env="ALLOWED_MIME_TYPES",
        description="Allowed MIME types for uploads"
    )


class AISettings(BaseSettings):
    """AI and vectorization configuration settings."""
    
    # OpenAI Configuration
    openai_api_key: str = Field(
        default="",
        env="OPENAI_API_KEY",
        description="OpenAI API key for embeddings and chat"
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model to use"
    )
    openai_chat_model: str = Field(
        default="gpt-3.5-turbo",
        description="OpenAI chat model to use"
    )
    openai_timeout: float = Field(
        default=AIConstants.DEFAULT_OPENAI_TIMEOUT,
        env="OPENAI_TIMEOUT",
        description="OpenAI API timeout in seconds"
    )
    openai_max_retries: int = Field(
        default=AIConstants.DEFAULT_OPENAI_MAX_RETRIES,
        env="OPENAI_MAX_RETRIES",
        description="OpenAI API maximum retries"
    )
    
    # Vector Database Configuration
    vector_storage_path: str = Field(
        default=AIConstants.DEFAULT_VECTOR_STORAGE_PATH,
        env="VECTOR_STORAGE_PATH",
        description="Vector database storage path"
    )
    vector_index_type: str = Field(
        default=AIConstants.DEFAULT_VECTOR_INDEX_TYPE,
        env="VECTOR_INDEX_TYPE",
        description="Vector index type (Flat, IVFFlat, HNSW)"
    )
    embedding_dimension: int = Field(
        default=AIConstants.EMBEDDING_DIMENSIONS[AIConstants.DEFAULT_EMBEDDING_MODEL],
        env="EMBEDDING_DIMENSION",
        description="Embedding vector dimension"
    )
    faiss_nlist: int = Field(
        default=AIConstants.DEFAULT_FAISS_NLIST,
        env="FAISS_NLIST",
        description="FAISS IVF index nlist parameter"
    )
    faiss_nprobe: int = Field(
        default=AIConstants.DEFAULT_FAISS_NPROBE,
        env="FAISS_NPROBE",
        description="FAISS IVF index nprobe parameter"
    )
    
    # Chat Configuration
    max_conversation_history: int = Field(
        default=50,
        description="Maximum conversation history length"
    )
    
    # Concurrency Configuration
    max_concurrent_embeddings: int = Field(
        default=AIConstants.DEFAULT_MAX_CONCURRENT_EMBEDDINGS,
        env="MAX_CONCURRENT_EMBEDDINGS",
        description="Maximum concurrent embedding operations"
    )
    max_concurrent_vectorizations: int = Field(
        default=AIConstants.DEFAULT_MAX_CONCURRENT_VECTORIZATIONS,
        env="MAX_CONCURRENT_VECTORIZATIONS",
        description="Maximum concurrent vectorization operations"
    )
    embedding_batch_size: int = Field(
        default=AIConstants.EMBEDDING_BATCH_SIZE,
        env="EMBEDDING_BATCH_SIZE",
        description="Batch size for embedding generation"
    )
    
    # Search Configuration
    default_similarity_threshold: float = Field(
        default=AIConstants.DEFAULT_SIMILARITY_THRESHOLD,
        env="DEFAULT_SIMILARITY_THRESHOLD",
        description="Default similarity search threshold"
    )
    default_search_limit: int = Field(
        default=AIConstants.DEFAULT_SEARCH_LIMIT,
        env="DEFAULT_SEARCH_LIMIT",
        description="Default search result limit"
    )
    max_search_limit: int = Field(
        default=AIConstants.MAX_SEARCH_LIMIT,
        env="MAX_SEARCH_LIMIT",
        description="Maximum search result limit"
    )
    
    @field_validator('vector_index_type')
    @classmethod
    def validate_vector_index_type(cls, v):
        """Validate vector index type."""
        if v not in AIConstants.AVAILABLE_VECTOR_INDEX_TYPES:
            raise ValueError(f"Invalid vector index type: {v}")
        return v
    
    @field_validator('openai_embedding_model')
    @classmethod
    def validate_embedding_model(cls, v):
        """Validate embedding model."""
        if v not in AIConstants.AVAILABLE_EMBEDDING_MODELS:
            raise ValueError(f"Invalid embedding model: {v}")
        return v


class TextProcessingSettings(BaseSettings):
    """Text processing configuration settings."""
    
    default_chunk_size: int = Field(
        default=TextConstants.DEFAULT_CHUNK_SIZE,
        env="DEFAULT_CHUNK_SIZE",
        description="Default text chunk size"
    )
    default_overlap_size: int = Field(
        default=TextConstants.DEFAULT_OVERLAP_SIZE,
        env="DEFAULT_OVERLAP_SIZE",
        description="Default text chunk overlap size"
    )
    min_chunk_size: int = Field(
        default=TextConstants.MIN_CHUNK_SIZE,
        env="MIN_CHUNK_SIZE",
        description="Minimum text chunk size"
    )
    max_chunk_size: int = Field(
        default=TextConstants.MAX_CHUNK_SIZE,
        env="MAX_CHUNK_SIZE",
        description="Maximum text chunk size"
    )
    sentence_separators: List[str] = Field(
        default=TextConstants.SENTENCE_SEPARATORS,
        env="SENTENCE_SEPARATORS",
        description="Sentence separator characters"
    )


class APISettings(BaseSettings):
    """API configuration settings."""
    
    host: str = Field(
        default="0.0.0.0",
        env="API_HOST",
        description="API host address"
    )
    port: int = Field(
        default=8000,
        env="API_PORT",
        description="API port number"
    )
    debug: bool = Field(
        default=False,
        env="API_DEBUG",
        description="Enable debug mode"
    )
    reload: bool = Field(
        default=False,
        env="API_RELOAD",
        description="Enable auto-reload"
    )
    workers: int = Field(
        default=1,
        env="API_WORKERS",
        description="Number of worker processes"
    )
    cors_origins: List[str] = Field(
        default=["http://localhost:3000"],
        env="CORS_ORIGINS",
        description="CORS allowed origins"
    )
    default_page_size: int = Field(
        default=APIConstants.DEFAULT_PAGE_SIZE,
        env="DEFAULT_PAGE_SIZE",
        description="Default pagination page size"
    )
    max_page_size: int = Field(
        default=APIConstants.MAX_PAGE_SIZE,
        env="MAX_PAGE_SIZE",
        description="Maximum pagination page size"
    )
    request_timeout: int = Field(
        default=APIConstants.DEFAULT_TIMEOUT,
        env="REQUEST_TIMEOUT",
        description="Request timeout in seconds"
    )


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""
    
    log_level: str = Field(
        default=LoggingConstants.DEFAULT_LOG_LEVEL,
        env="LOG_LEVEL",
        description="Logging level"
    )
    log_format: str = Field(
        default=LoggingConstants.LOG_FORMAT,
        env="LOG_FORMAT",
        description="Log message format"
    )
    log_file: Optional[str] = Field(
        default=None,
        env="LOG_FILE",
        description="Log file path"
    )
    log_rotation: bool = Field(
        default=True,
        env="LOG_ROTATION",
        description="Enable log file rotation"
    )
    max_log_size_mb: int = Field(
        default=LoggingConstants.MAX_LOG_FILE_SIZE_MB,
        env="MAX_LOG_SIZE_MB",
        description="Maximum log file size in MB"
    )
    backup_count: int = Field(
        default=LoggingConstants.LOG_BACKUP_COUNT,
        env="LOG_BACKUP_COUNT",
        description="Number of log backup files"
    )


class Settings(BaseSettings):
    """Main application settings."""
    
    # Environment
    environment: str = Field(
        default=EnvironmentConstants.DEFAULT_ENVIRONMENT,
        env="ENVIRONMENT",
        description="Application environment"
    )
    
    # Application Info
    app_name: str = Field(
        default="Google NotebookLM Clone",
        env="APP_NAME",
        description="Application name"
    )
    app_version: str = Field(
        default="1.0.0",
        env="APP_VERSION",
        description="Application version"
    )
    app_description: str = Field(
        default="A Google NotebookLM clone backend with PDF processing and AI chat",
        env="APP_DESCRIPTION",
        description="Application description"
    )
    
    # Secret Key
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        env="SECRET_KEY",
        description="Application secret key"
    )
    
    # CORS Settings
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        env="ALLOWED_ORIGINS",
        description="Allowed CORS origins"
    )
    
    # Component Settings
    database: DatabaseSettings = DatabaseSettings()
    file_storage: FileStorageSettings = FileStorageSettings()
    ai: AISettings = AISettings()
    text_processing: TextProcessingSettings = TextProcessingSettings()
    api: APISettings = APISettings()
    logging: LoggingSettings = LoggingSettings()
    
    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v):
        """Validate environment value."""
        valid_environments = [
            EnvironmentConstants.DEVELOPMENT,
            EnvironmentConstants.TESTING,
            EnvironmentConstants.STAGING,
            EnvironmentConstants.PRODUCTION
        ]
        if v not in valid_environments:
            raise ValueError(f"Invalid environment: {v}")
        return v
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == EnvironmentConstants.DEVELOPMENT
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment == EnvironmentConstants.TESTING
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment in EnvironmentConstants.PRODUCTION_ENVIRONMENTS
    
    @property
    def debug_enabled(self) -> bool:
        """Check if debug mode should be enabled."""
        return self.environment in EnvironmentConstants.DEBUG_ENVIRONMENTS
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"  # Allow extra fields to prevent validation errors


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings with caching.
    
    Returns:
        Application settings instance
    """
    return Settings()


def create_settings_from_env() -> Settings:
    """
    Create settings instance from environment variables.
    
    Returns:
        Fresh settings instance
    """
    return Settings()


def validate_settings(settings: Settings) -> None:
    """
    Validate application settings.
    
    Args:
        settings: Settings instance to validate
        
    Raises:
        ValueError: If settings are invalid
    """
    # Validate required settings for production
    if settings.is_production:
        if not settings.ai.openai_api_key:
            raise ValueError("OpenAI API key is required in production")
        
        if settings.secret_key == "your-secret-key-change-in-production":
            raise ValueError("Secret key must be changed in production")
        
        if "localhost" in settings.database.database_url:
            raise ValueError("Database URL should not use localhost in production")
    
    # Validate AI settings
    if settings.ai.openai_api_key:
        if not settings.ai.openai_api_key.startswith("sk-"):
            raise ValueError("Invalid OpenAI API key format")
    
    # Validate file storage paths
    storage_path = settings.file_storage.storage_path
    if not os.path.isabs(storage_path):
        # Convert to absolute path
        settings.file_storage.storage_path = os.path.abspath(storage_path)
    
    vector_path = settings.ai.vector_storage_path
    if not os.path.isabs(vector_path):
        # Convert to absolute path
        settings.ai.vector_storage_path = os.path.abspath(vector_path)


def setup_directories(settings: Settings) -> None:
    """
    Create necessary directories for the application.
    
    Args:
        settings: Settings instance
    """
    directories = [
        settings.file_storage.storage_path,
        settings.ai.vector_storage_path,
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def get_database_url(settings: Settings) -> str:
    """
    Get database URL with proper formatting.
    
    Args:
        settings: Settings instance
        
    Returns:
        Formatted database URL
    """
    return settings.database.database_url


def get_cors_settings(settings: Settings) -> dict:
    """
    Get CORS settings for FastAPI.
    
    Args:
        settings: Settings instance
        
    Returns:
        CORS configuration dictionary
    """
    return {
        "allow_origins": settings.api.cors_origins,
        "allow_credentials": True,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }
