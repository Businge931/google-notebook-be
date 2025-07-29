"""
Application Constants
"""
from enum import Enum
from typing import Set, Dict, Any


class FileConstants:
    """File-related constants."""
    
    # Supported file types
    ALLOWED_MIME_TYPES: Set[str] = {
        'application/pdf',
        'application/x-pdf',
        'application/acrobat',
        'applications/vnd.pdf',
        'text/pdf',
        'text/x-pdf'
    }
    
    # File size limits
    MAX_FILE_SIZE_BYTES: int = 100 * 1024 * 1024  # 100MB
    MAX_FILE_SIZE_MB: int = 100  # 100MB
    MIN_FILE_SIZE_BYTES: int = 1024  # 1KB
    
    # File name constraints
    MAX_FILENAME_LENGTH: int = 255
    INVALID_FILENAME_CHARS: Set[str] = {'<', '>', ':', '"', '|', '?', '*', '\0'}
    
    # File extensions
    ALLOWED_EXTENSIONS: Set[str] = {'.pdf'}
    
    # Storage paths
    DEFAULT_STORAGE_PATH: str = "./uploads"


class ProcessingConstants:
    """Document processing constants."""
    
    # Chunk size limits
    DEFAULT_CHUNK_SIZE: int = 1000
    MIN_CHUNK_SIZE: int = 500
    MAX_CHUNK_SIZE: int = 2000
    
    # Processing timeouts
    DEFAULT_PROCESSING_TIMEOUT_SECONDS: int = 1800  # 30 minutes
    MAX_PROCESSING_TIMEOUT_SECONDS: int = 3600  # 1 hour
    
    # Retry configuration
    MAX_PROCESSING_RETRIES: int = 3
    RETRY_DELAY_SECONDS: int = 60
    
    # Page processing
    MAX_PAGES_PER_DOCUMENT: int = 1000
    
    # Text extraction
    MIN_TEXT_LENGTH_PER_PAGE: int = 10
    MAX_TEXT_LENGTH_PER_CHUNK: int = 5000


class ChatConstants:
    """Chat-related constants."""
    
    # Message limits
    MAX_MESSAGE_LENGTH: int = 10000
    MAX_MESSAGES_PER_SESSION: int = 1000
    
    # Session limits
    MAX_SESSION_TITLE_LENGTH: int = 200
    MAX_ACTIVE_SESSIONS_PER_DOCUMENT: int = 10
    
    # Response generation
    DEFAULT_MAX_TOKENS: int = 4000
    MAX_CONTEXT_TOKENS: int = 8000
    
    # Citation limits
    MAX_CITATIONS_PER_MESSAGE: int = 10
    MIN_CITATION_CONFIDENCE: float = 0.7
    
    # Session timeout
    SESSION_TIMEOUT_HOURS: int = 24


class VectorConstants:
    """Vector database constants."""
    
    # Search parameters
    DEFAULT_SEARCH_LIMIT: int = 10
    MAX_SEARCH_LIMIT: int = 50
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.7
    MIN_SIMILARITY_THRESHOLD: float = 0.5
    
    # Vector dimensions (depends on embedding model)
    OPENAI_EMBEDDING_DIMENSIONS: int = 1536
    
    # Indexing
    MAX_VECTORS_PER_BATCH: int = 100
    VECTOR_INDEX_REFRESH_INTERVAL_SECONDS: int = 300  # 5 minutes


class APIConstants:
    """API-related constants."""
    
    # Pagination
    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int = 100
    
    # Rate limiting
    DEFAULT_RATE_LIMIT_PER_MINUTE: int = 60
    UPLOAD_RATE_LIMIT_PER_HOUR: int = 10
    
    # Request timeouts
    DEFAULT_REQUEST_TIMEOUT_SECONDS: int = 30
    UPLOAD_REQUEST_TIMEOUT_SECONDS: int = 300  # 5 minutes
    
    # Response formats
    SUPPORTED_RESPONSE_FORMATS: Set[str] = {'json', 'text'}
    
    # Timeouts
    DEFAULT_TIMEOUT: int = 300


class DatabaseConstants:
    """Database-related constants."""
    
    # Connection pool
    MIN_POOL_SIZE: int = 5
    DEFAULT_POOL_SIZE: int = 10
    MAX_POOL_SIZE: int = 20
    DEFAULT_MAX_OVERFLOW: int = 20
    POOL_TIMEOUT_SECONDS: int = 30
    DEFAULT_POOL_TIMEOUT: int = 30
    DEFAULT_POOL_RECYCLE: int = 3600
    
    # Query timeouts
    DEFAULT_QUERY_TIMEOUT_SECONDS: int = 30
    LONG_QUERY_TIMEOUT_SECONDS: int = 300
    
    # Batch operations
    DEFAULT_BATCH_SIZE: int = 100
    MAX_BATCH_SIZE: int = 1000


class SecurityConstants:
    """Security-related constants."""
    
    # Token configuration
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Password requirements
    MIN_PASSWORD_LENGTH: int = 8
    MAX_PASSWORD_LENGTH: int = 128
    
    # Rate limiting
    MAX_LOGIN_ATTEMPTS: int = 5
    LOGIN_LOCKOUT_MINUTES: int = 15


class LoggingConstants:
    """Logging-related constants."""
    
    # Log levels
    DEBUG: str = "DEBUG"
    INFO: str = "INFO"
    WARNING: str = "WARNING"
    ERROR: str = "ERROR"
    CRITICAL: str = "CRITICAL"
    DEFAULT_LOG_LEVEL: str = "INFO"
    
    # Log file settings
    MAX_LOG_FILE_SIZE_MB: int = 10
    MAX_LOG_FILES: int = 5
    LOG_BACKUP_COUNT: int = 5
    
    # Log format
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    SUPPORTED_LOG_FORMATS: Set[str] = {"json", "text"}
    
    # Log retention
    DEFAULT_LOG_RETENTION_DAYS: int = 30
    MAX_LOG_FILE_SIZE_MB: int = 100


class ErrorCodes:
    """Application error codes."""
    
    # Validation errors
    VALIDATION_ERROR: str = "VALIDATION_ERROR"
    BUSINESS_RULE_VIOLATION: str = "BUSINESS_RULE_VIOLATION"
    
    # Document errors
    DOCUMENT_NOT_FOUND: str = "DOCUMENT_NOT_FOUND"
    DOCUMENT_PROCESSING_ERROR: str = "DOCUMENT_PROCESSING_ERROR"
    INVALID_DOCUMENT_STATE: str = "INVALID_DOCUMENT_STATE"
    
    # Chat errors
    CHAT_SESSION_NOT_FOUND: str = "CHAT_SESSION_NOT_FOUND"
    MESSAGE_NOT_FOUND: str = "MESSAGE_NOT_FOUND"
    INVALID_SESSION_STATE: str = "INVALID_SESSION_STATE"
    
    # Repository errors
    REPOSITORY_ERROR: str = "REPOSITORY_ERROR"
    VECTOR_REPOSITORY_ERROR: str = "VECTOR_REPOSITORY_ERROR"
    DATABASE_CONNECTION_ERROR: str = "DATABASE_CONNECTION_ERROR"
    CONCURRENCY_ERROR: str = "CONCURRENCY_ERROR"
    
    # API errors
    UNAUTHORIZED: str = "UNAUTHORIZED"
    FORBIDDEN: str = "FORBIDDEN"
    RATE_LIMIT_EXCEEDED: str = "RATE_LIMIT_EXCEEDED"
    INVALID_REQUEST: str = "INVALID_REQUEST"


class EnvironmentConstants:
    """Environment and deployment constants."""
    
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    
    DEFAULT_ENVIRONMENT = DEVELOPMENT
    
    # Default values
    DEFAULT_HOST = "0.0.0.0"
    DEFAULT_PORT = 8000
    DEFAULT_DEBUG = False
    
    # Environment-specific settings
    DEBUG_ENVIRONMENTS = [DEVELOPMENT, TESTING]
    SECURE_ENVIRONMENTS = [STAGING, PRODUCTION]


class TextConstants:
    """Text processing constants."""
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_OVERLAP_SIZE = 200
    MIN_CHUNK_SIZE = 100
    MAX_CHUNK_SIZE = 8000
    MIN_OVERLAP_SIZE = 0
    MAX_OVERLAP_SIZE = 1000
    SENTENCE_SEPARATORS = [".", "!", "?", "\n\n"]


class AIConstants:
    """AI and vectorization constants."""
    
    # OpenAI Constants
    DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
    AVAILABLE_EMBEDDING_MODELS = [
        "text-embedding-3-small",
        "text-embedding-3-large", 
        "text-embedding-ada-002"
    ]
    DEFAULT_CHAT_MODEL = "gpt-3.5-turbo"
    AVAILABLE_CHAT_MODELS = [
        "gpt-4",
        "gpt-4-turbo",
        "gpt-3.5-turbo"
    ]
    
    # Embedding Dimensions
    EMBEDDING_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536
    }
    
    # Vector Database Constants
    DEFAULT_VECTOR_INDEX_TYPE = "IVFFlat"
    AVAILABLE_VECTOR_INDEX_TYPES = ["Flat", "IVFFlat", "HNSW"]
    DEFAULT_FAISS_NLIST = 100
    DEFAULT_FAISS_NPROBE = 10
    DEFAULT_VECTOR_STORAGE_PATH = "storage/vectors"
    
    # Search Constants
    DEFAULT_SIMILARITY_THRESHOLD = 0.7
    MIN_SIMILARITY_THRESHOLD = 0.0
    MAX_SIMILARITY_THRESHOLD = 1.0
    DEFAULT_SEARCH_LIMIT = 10
    MAX_SEARCH_LIMIT = 100
    
    # Processing Constants
    DEFAULT_MAX_CONCURRENT_EMBEDDINGS = 10
    DEFAULT_MAX_CONCURRENT_VECTORIZATIONS = 5
    EMBEDDING_BATCH_SIZE = 100
    MAX_TOKENS_PER_REQUEST = 8192
    
    # Timeout Constants
    DEFAULT_OPENAI_TIMEOUT = 30.0
    DEFAULT_OPENAI_MAX_RETRIES = 3
    DEFAULT_EMBEDDING_TIMEOUT = 60.0
    DEFAULT_VECTORIZATION_TIMEOUT = 300.0
