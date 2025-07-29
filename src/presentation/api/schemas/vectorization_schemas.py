"""
Vectorization API Schemas

"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime


class VectorizationRequestSchema(BaseModel):
    """Schema for vectorization request following Single Responsibility Principle."""
    
    model: Optional[str] = Field(
        None,
        description="Embedding model to use (e.g., 'text-embedding-3-small')",
        example="text-embedding-3-small"
    )
    force_regenerate: bool = Field(
        False,
        description="Force regeneration of embeddings even if they exist"
    )
    chunk_size: Optional[int] = Field(
        None,
        description="Custom chunk size for text splitting",
        ge=100,
        le=8000
    )
    overlap_size: Optional[int] = Field(
        None,
        description="Custom overlap size for text chunking",
        ge=0,
        le=1000
    )
    async_processing: bool = Field(
        False,
        description="Process vectorization in background"
    )
    
    @validator('overlap_size')
    def validate_overlap_size(cls, v, values):
        """Validate overlap size is less than chunk size."""
        if v is not None and 'chunk_size' in values and values['chunk_size'] is not None:
            if v >= values['chunk_size']:
                raise ValueError("Overlap size must be less than chunk size")
        return v


class VectorizationResponseSchema(BaseModel):
    """Schema for vectorization response following Single Responsibility Principle."""
    
    document_id: str = Field(description="Document identifier")
    success: bool = Field(description="Whether vectorization was successful")
    chunks_processed: int = Field(description="Number of chunks processed")
    embeddings_generated: int = Field(description="Number of embeddings generated")
    processing_time_ms: int = Field(description="Processing time in milliseconds")
    model_used: str = Field(description="Embedding model used")
    error_message: Optional[str] = Field(
        None,
        description="Error message if vectorization failed"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata about the vectorization"
    )


class BulkVectorizationRequestSchema(BaseModel):
    """Schema for bulk vectorization request following Single Responsibility Principle."""
    
    document_ids: List[str] = Field(
        description="List of document identifiers to vectorize",
        min_items=1,
        max_items=100
    )
    model: Optional[str] = Field(
        None,
        description="Embedding model to use for all documents"
    )
    force_regenerate: bool = Field(
        False,
        description="Force regeneration of embeddings even if they exist"
    )
    max_concurrent: int = Field(
        5,
        description="Maximum concurrent vectorization operations",
        ge=1,
        le=20
    )
    chunk_size: Optional[int] = Field(
        None,
        description="Custom chunk size for text splitting",
        ge=100,
        le=8000
    )
    overlap_size: Optional[int] = Field(
        None,
        description="Custom overlap size for text chunking",
        ge=0,
        le=1000
    )


class BulkVectorizationResponseSchema(BaseModel):
    """Schema for bulk vectorization response following Single Responsibility Principle."""
    
    total_documents: int = Field(description="Total number of documents processed")
    successful_documents: int = Field(description="Number of successfully processed documents")
    failed_documents: int = Field(description="Number of failed documents")
    total_chunks_processed: int = Field(description="Total chunks processed across all documents")
    total_embeddings_generated: int = Field(description="Total embeddings generated")
    total_processing_time_ms: int = Field(description="Total processing time in milliseconds")
    results: List[VectorizationResponseSchema] = Field(
        description="Individual results for each document"
    )
    error_summary: Optional[str] = Field(
        None,
        description="Summary of errors that occurred"
    )


class VectorizationStatusResponseSchema(BaseModel):
    """Schema for vectorization status response following Single Responsibility Principle."""
    
    document_id: str = Field(description="Document identifier")
    is_vectorized: bool = Field(description="Whether document has any vectors")
    is_complete: bool = Field(description="Whether all chunks are vectorized")
    vector_count: int = Field(description="Number of vectors stored")
    chunk_count: int = Field(description="Number of chunks in document")
    completion_percentage: float = Field(
        description="Percentage of chunks vectorized",
        ge=0.0,
        le=100.0
    )
    document_status: str = Field(description="Document processing status")
    document_title: Optional[str] = Field(None, description="Document title")
    document_filename: Optional[str] = Field(None, description="Document filename")


class SimilaritySearchRequestSchema(BaseModel):
    """Schema for similarity search request following Single Responsibility Principle."""
    
    query_text: str = Field(
        description="Text query to search for similar content",
        min_length=1,
        max_length=1000
    )
    document_id: Optional[str] = Field(
        None,
        description="Search within specific document only"
    )
    document_ids: Optional[List[str]] = Field(
        None,
        description="Search within specific documents only",
        max_items=50
    )
    limit: int = Field(
        10,
        description="Maximum number of results to return",
        ge=1,
        le=100
    )
    similarity_threshold: float = Field(
        0.7,
        description="Minimum similarity score threshold",
        ge=0.0,
        le=1.0
    )
    include_metadata: bool = Field(
        True,
        description="Include additional metadata in results"
    )
    
    @validator('document_ids')
    def validate_document_ids_not_both(cls, v, values):
        """Validate that both document_id and document_ids are not provided."""
        if v is not None and 'document_id' in values and values['document_id'] is not None:
            raise ValueError("Cannot specify both document_id and document_ids")
        return v


class SearchResultItemSchema(BaseModel):
    """Schema for individual search result following Single Responsibility Principle."""
    
    chunk_id: str = Field(description="Chunk identifier")
    document_id: str = Field(description="Document identifier")
    document_title: Optional[str] = Field(None, description="Document title")
    document_filename: Optional[str] = Field(None, description="Document filename")
    page_number: int = Field(description="Page number in document")
    text_content: str = Field(description="Text content of the chunk")
    similarity_score: float = Field(
        description="Similarity score (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    start_position: Optional[int] = Field(None, description="Start position in page")
    end_position: Optional[int] = Field(None, description="End position in page")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class SimilaritySearchResponseSchema(BaseModel):
    """Schema for similarity search response following Single Responsibility Principle."""
    
    query_text: str = Field(description="Original query text")
    results: List[SearchResultItemSchema] = Field(description="Search results")
    total_results: int = Field(description="Total number of results found")
    search_time_ms: int = Field(description="Search time in milliseconds")
    search_scope: str = Field(
        description="Search scope",
        pattern="^(single_document|multiple_documents|all_documents)$"
    )
    similarity_threshold: float = Field(description="Similarity threshold used")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional search metadata")


class DocumentSimilarityRequestSchema(BaseModel):
    """Schema for document similarity search request following Single Responsibility Principle."""
    
    query_text: str = Field(
        description="Text query to find similar documents",
        min_length=1,
        max_length=1000
    )
    limit: int = Field(
        5,
        description="Maximum number of documents to return",
        ge=1,
        le=50
    )
    exclude_document_ids: Optional[List[str]] = Field(
        None,
        description="Document IDs to exclude from search",
        max_items=100
    )
    similarity_threshold: float = Field(
        0.7,
        description="Minimum similarity score threshold",
        ge=0.0,
        le=1.0
    )


class DocumentSimilarityItemSchema(BaseModel):
    """Schema for individual document similarity result following Single Responsibility Principle."""
    
    document_id: str = Field(description="Document identifier")
    title: str = Field(description="Document title")
    filename: str = Field(description="Document filename")
    similarity_score: float = Field(
        description="Document similarity score (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    matching_chunks: int = Field(description="Number of matching chunks")
    top_chunks: List[Dict[str, Any]] = Field(
        description="Top matching chunks from the document"
    )
    upload_date: Optional[str] = Field(None, description="Document upload date")
    status: str = Field(description="Document status")


class DocumentSimilarityResponseSchema(BaseModel):
    """Schema for document similarity search response following Single Responsibility Principle."""
    
    query_text: str = Field(description="Original query text")
    similar_documents: List[DocumentSimilarityItemSchema] = Field(
        description="Similar documents found"
    )
    total_documents: int = Field(description="Total number of similar documents found")
    search_time_ms: int = Field(description="Search time in milliseconds")
    similarity_threshold: float = Field(description="Similarity threshold used")


class SearchSuggestionsResponseSchema(BaseModel):
    """Schema for search suggestions response following Single Responsibility Principle."""
    
    query: str = Field(description="Original partial query")
    suggestions: List[str] = Field(description="Search suggestions")
    total_suggestions: int = Field(description="Total number of suggestions")


class EmbeddingVectorSchema(BaseModel):
    """Schema for embedding vector following Single Responsibility Principle."""
    
    vector: List[float] = Field(description="Embedding vector values")
    dimension: int = Field(description="Vector dimension")
    model: str = Field(description="Model used to generate embedding")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class TextEmbeddingSchema(BaseModel):
    """Schema for text with embedding following Single Responsibility Principle."""
    
    text: str = Field(description="Original text")
    embedding: EmbeddingVectorSchema = Field(description="Text embedding")
    chunk_id: str = Field(description="Chunk identifier")
    document_id: str = Field(description="Document identifier")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class EmbeddingBatchSchema(BaseModel):
    """Schema for batch of embeddings following Single Responsibility Principle."""
    
    embeddings: List[TextEmbeddingSchema] = Field(description="List of text embeddings")
    model: str = Field(description="Model used for embeddings")
    total_tokens: Optional[int] = Field(None, description="Total tokens processed")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")


class VectorHealthCheckSchema(BaseModel):
    """Schema for vector database health check following Single Responsibility Principle."""
    
    status: str = Field(description="Health status", pattern="^(healthy|unhealthy)$")
    vector_count: int = Field(description="Total number of vectors stored")
    index_type: str = Field(description="Type of vector index")
    embedding_dimension: int = Field(description="Embedding vector dimension")
    last_updated: Optional[str] = Field(None, description="Last index update timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional health metadata")


class VectorStatsSchema(BaseModel):
    """Schema for vector database statistics following Single Responsibility Principle."""
    
    total_vectors: int = Field(description="Total number of vectors")
    total_documents: int = Field(description="Total number of documents with vectors")
    average_chunks_per_document: float = Field(description="Average chunks per document")
    embedding_models_used: List[str] = Field(description="List of embedding models used")
    index_size_mb: Optional[float] = Field(None, description="Index size in megabytes")
    last_vectorization: Optional[str] = Field(None, description="Last vectorization timestamp")


class ErrorResponseSchema(BaseModel):
    """Schema for error responses following Single Responsibility Principle."""
    
    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")
