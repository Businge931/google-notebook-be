"""
Application Settings Configuration
"""
from functools import lru_cache
from typing import List, Optional
from pydantic import validator, Field
from pydantic_settings import BaseSettings

from src.shared.constants import (
    FileConstants,
    ProcessingConstants,
    ChatConstants,
    VectorConstants,
    DatabaseConstants,
    SecurityConstants,
    EnvironmentConstants,
)


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    database_url: str = "postgresql+asyncpg://postgres:password@localhost:5432/notebooklm_db"
    database_url_sync: str = "postgresql://postgres:password@localhost:5432/notebooklm_db"
    min_pool_size: int = DatabaseConstants.MIN_POOL_SIZE
    max_pool_size: int = DatabaseConstants.MAX_POOL_SIZE
    pool_timeout: int = DatabaseConstants.POOL_TIMEOUT_SECONDS
    query_timeout: int = DatabaseConstants.DEFAULT_QUERY_TIMEOUT_SECONDS
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = 'ignore'


class RedisSettings(BaseSettings):
    """Redis configuration settings."""
    
    redis_url: str = "redis://localhost:6379/0"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = 'ignore'


class FileStorageSettings(BaseSettings):
    """File storage configuration settings."""
    
    storage_type: str = "local"  # Options: local, s3
    upload_dir: str = "./uploads"
    max_file_size: int = FileConstants.MAX_FILE_SIZE_BYTES
    allowed_file_types: List[str] = list(FileConstants.ALLOWED_MIME_TYPES)
    
    # AWS S3 settings (if using S3)
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"
    s3_bucket_name: Optional[str] = None
    
    @validator('storage_type')
    def validate_storage_type(cls, v):
        if v not in ['local', 's3']:
            raise ValueError('storage_type must be either "local" or "s3"')
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = 'ignore'


class AIServiceSettings(BaseSettings):
    """AI service configuration settings."""
    
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = "gpt-3.5-turbo"
    embedding_model: str = "text-embedding-ada-002"
    max_tokens: int = ChatConstants.DEFAULT_MAX_TOKENS
    temperature: float = 0.7
    openai_timeout: int = 60
    
    # LlamaIndex settings
    llama_parse_api_key: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = 'ignore'


class VectorDatabaseSettings(BaseSettings):
    """Vector database configuration settings."""
    
    vector_db_type: str = "faiss"  # Options: faiss, chroma
    
    # FAISS settings (if using FAISS)
    faiss_index_path: str = "./vector_indexes"
    embedding_dimension: int = 1536
    
    # ChromaDB settings (if using ChromaDB)
    chroma_persist_dir: str = "./chroma_db"
    
    # Pinecone settings removed - FAISS is the selected vector database solution
    
    # Search settings
    default_search_limit: int = VectorConstants.DEFAULT_SEARCH_LIMIT
    max_search_limit: int = VectorConstants.MAX_SEARCH_LIMIT
    similarity_threshold: float = VectorConstants.DEFAULT_SIMILARITY_THRESHOLD
    
    @validator('vector_db_type')
    def validate_vector_db_type(cls, v):
        if v not in ['faiss', 'chroma']:
            raise ValueError('vector_db_type must be one of "faiss" or "chroma"')
        return v
    
    @validator('similarity_threshold')
    def validate_similarity_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('similarity_threshold must be between 0.0 and 1.0')
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = 'ignore'


class CelerySettings(BaseSettings):
    """Celery configuration settings."""
    
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = 'ignore'


class SecuritySettings(BaseSettings):
    """Security configuration settings."""
    
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = SecurityConstants.ACCESS_TOKEN_EXPIRE_MINUTES
    
    @validator('secret_key')
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError('secret_key must be at least 32 characters long')
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = 'ignore'


class MonitoringSettings(BaseSettings):
    """Monitoring and logging configuration settings."""
    
    sentry_dsn: Optional[str] = None
    enable_metrics: bool = True
    log_level: str = "INFO"
    log_format: str = "json"
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f'log_level must be one of {valid_levels}')
        return v.upper()
    
    @validator('log_format')
    def validate_log_format(cls, v):
        if v not in ['json', 'text']:
            raise ValueError('log_format must be either "json" or "text"')
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = 'ignore'


class Settings(BaseSettings):
    """Main application settings."""
    
    # Application settings
    app_name: str = "Google NotebookLM Clone"
    debug: bool = False
    host: str = EnvironmentConstants.DEFAULT_HOST
    port: int = EnvironmentConstants.DEFAULT_PORT
    allowed_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    
    # Environment
    environment: str = EnvironmentConstants.DEVELOPMENT
    
    # Sub-settings
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    file_storage: FileStorageSettings = FileStorageSettings()
    ai_service: AIServiceSettings = AIServiceSettings()
    vector_db: VectorDatabaseSettings = VectorDatabaseSettings()
    celery: CelerySettings = CelerySettings()
    security: SecuritySettings = SecuritySettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    
    @validator('environment')
    def validate_environment(cls, v):
        valid_envs = {
            EnvironmentConstants.DEVELOPMENT,
            EnvironmentConstants.TESTING,
            EnvironmentConstants.STAGING,
            EnvironmentConstants.PRODUCTION
        }
        if v not in valid_envs:
            raise ValueError(f'environment must be one of {valid_envs}')
        return v
    
    @validator('allowed_origins')
    def validate_allowed_origins(cls, v):
        if not v:
            raise ValueError('allowed_origins cannot be empty')
        return v
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == EnvironmentConstants.DEVELOPMENT
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == EnvironmentConstants.PRODUCTION
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment == EnvironmentConstants.TESTING
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = 'ignore'


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings with caching.
    
    Returns:
        Settings instance
    """
    return Settings()
