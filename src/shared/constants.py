"""
Shared Constants

"""
from enum import Enum


class DocumentStatus(Enum):
    """Document processing status enumeration."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    DELETED = "deleted"


class ProcessingStage(Enum):
    """Document processing stage enumeration."""
    UPLOAD = "upload"
    PARSING = "parsing"
    CHUNKING = "chunking"
    VECTORIZATION = "vectorization"
    COMPLETED = "completed"
    ERROR = "error"


class ChatMessageType(Enum):
    """Chat message type enumeration."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class CitationType(Enum):
    """Citation type enumeration."""
    DIRECT_QUOTE = "direct_quote"
    PARAPHRASE = "paraphrase"
    REFERENCE = "reference"


class FileConstants:
    """File-related constants."""
    MAX_FILE_SIZE_MB = 100
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
    ALLOWED_MIME_TYPES = ["application/pdf"]
    ALLOWED_EXTENSIONS = [".pdf"]
    UPLOAD_CHUNK_SIZE = 8192
    DEFAULT_STORAGE_PATH = "storage/documents"


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
    MAX_CONCURRENT_EMBEDDINGS = 10
    MAX_CONCURRENT_VECTORIZATIONS = 5
    EMBEDDING_BATCH_SIZE = 100
    MAX_TOKENS_PER_REQUEST = 8192
    
    # Timeout Constants
    DEFAULT_OPENAI_TIMEOUT = 30.0
    DEFAULT_OPENAI_MAX_RETRIES = 3
    DEFAULT_EMBEDDING_TIMEOUT = 60.0
    DEFAULT_VECTORIZATION_TIMEOUT = 300.0


class DatabaseConstants:
    """Database-related constants."""
    DEFAULT_POOL_SIZE = 10
    DEFAULT_MAX_OVERFLOW = 20
    DEFAULT_POOL_TIMEOUT = 30
    DEFAULT_POOL_RECYCLE = 3600
    CONNECTION_RETRY_ATTEMPTS = 3
    CONNECTION_RETRY_DELAY = 1.0


class APIConstants:
    """API-related constants."""
    DEFAULT_PAGE_SIZE = 20
    MAX_PAGE_SIZE = 100
    MIN_PAGE_SIZE = 1
    DEFAULT_TIMEOUT = 30
    MAX_REQUEST_SIZE_MB = 500  # Allow large PDF uploads
    RATE_LIMIT_REQUESTS = 100
    RATE_LIMIT_WINDOW = 60


class LoggingConstants:
    """Logging-related constants."""
    DEFAULT_LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    MAX_LOG_FILE_SIZE_MB = 10
    LOG_BACKUP_COUNT = 5


class SecurityConstants:
    """Security-related constants."""
    PASSWORD_MIN_LENGTH = 8
    TOKEN_EXPIRY_HOURS = 24
    REFRESH_TOKEN_EXPIRY_DAYS = 7
    MAX_LOGIN_ATTEMPTS = 5
    LOCKOUT_DURATION_MINUTES = 15
    ALLOWED_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]


class ValidationConstants:
    """Validation-related constants."""
    MIN_FILENAME_LENGTH = 1
    MAX_FILENAME_LENGTH = 255
    MIN_TITLE_LENGTH = 1
    MAX_TITLE_LENGTH = 500
    MIN_QUERY_LENGTH = 1
    MAX_QUERY_LENGTH = 1000
    MIN_MESSAGE_LENGTH = 1
    MAX_MESSAGE_LENGTH = 10000


class ErrorCodes:
    """Application error codes."""
    
    # General Errors
    VALIDATION_ERROR = "VALIDATION_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    
    # Document Errors
    DOCUMENT_NOT_FOUND = "DOCUMENT_NOT_FOUND"
    DOCUMENT_PROCESSING_ERROR = "DOCUMENT_PROCESSING_ERROR"
    DOCUMENT_UPLOAD_ERROR = "DOCUMENT_UPLOAD_ERROR"
    INVALID_DOCUMENT_FORMAT = "INVALID_DOCUMENT_FORMAT"
    
    # File Storage Errors
    FILE_STORAGE_ERROR = "FILE_STORAGE_ERROR"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    FILE_SIZE_EXCEEDED = "FILE_SIZE_EXCEEDED"
    INVALID_FILE_TYPE = "INVALID_FILE_TYPE"
    
    # Database Errors
    DATABASE_CONNECTION_ERROR = "DATABASE_CONNECTION_ERROR"
    DATABASE_OPERATION_ERROR = "DATABASE_OPERATION_ERROR"
    TRANSACTION_ERROR = "TRANSACTION_ERROR"
    
    # AI Service Errors
    EMBEDDING_ERROR = "EMBEDDING_ERROR"
    VECTOR_REPOSITORY_ERROR = "VECTOR_REPOSITORY_ERROR"
    AI_SERVICE_UNAVAILABLE = "AI_SERVICE_UNAVAILABLE"
    OPENAI_API_ERROR = "OPENAI_API_ERROR"
    
    # Chat Errors
    CHAT_SESSION_NOT_FOUND = "CHAT_SESSION_NOT_FOUND"
    MESSAGE_GENERATION_ERROR = "MESSAGE_GENERATION_ERROR"
    CITATION_ERROR = "CITATION_ERROR"
    
    # Authentication Errors
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    AUTHORIZATION_ERROR = "AUTHORIZATION_ERROR"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    INVALID_CREDENTIALS = "INVALID_CREDENTIALS"


class HTTPStatusMessages:
    """HTTP status code messages."""
    
    # Success Messages
    DOCUMENT_UPLOADED = "Document uploaded successfully"
    DOCUMENT_PROCESSED = "Document processed successfully"
    DOCUMENT_DELETED = "Document deleted successfully"
    VECTORIZATION_COMPLETED = "Vectorization completed successfully"
    SEARCH_COMPLETED = "Search completed successfully"
    
    # Error Messages
    DOCUMENT_NOT_FOUND = "Document not found"
    INVALID_REQUEST = "Invalid request parameters"
    PROCESSING_FAILED = "Processing failed"
    UNAUTHORIZED_ACCESS = "Unauthorized access"
    INTERNAL_SERVER_ERROR = "Internal server error"
    SERVICE_UNAVAILABLE = "Service temporarily unavailable"


class EnvironmentConstants:
    """Environment-related constants."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    
    DEFAULT_ENVIRONMENT = DEVELOPMENT
    
    # Environment-specific settings
    DEBUG_ENVIRONMENTS = [DEVELOPMENT, TESTING]
    PRODUCTION_ENVIRONMENTS = [STAGING, PRODUCTION]
