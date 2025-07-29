"""
Chat API Schemas

"""
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime


class StartChatSessionRequest(BaseModel):
    """Request schema for starting a chat session following Single Responsibility Principle."""
    
    session_id: str = Field(
        ...,
        description="Unique identifier for the chat session",
        min_length=1,
        max_length=255,
        example="chat_session_123"
    )
    
    document_ids: Optional[List[str]] = Field(
        None,
        description="Optional list of document IDs to include in chat context",
        example=["doc_123", "doc_456"]
    )
    
    initial_message: Optional[str] = Field(
        None,
        description="Optional initial message to start the conversation",
        max_length=10000,
        example="What is this document about?"
    )
    
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional additional context for the chat session",
        example={"user_id": "user_123", "source": "web"}
    )
    
    @validator("session_id")
    def validate_session_id(cls, v):
        """Validate session ID format."""
        if not v or not v.strip():
            raise ValueError("Session ID cannot be empty")
        return v.strip()
    
    @validator("document_ids")
    def validate_document_ids(cls, v):
        """Validate document IDs."""
        if v is not None:
            if len(v) > 50:
                raise ValueError("Maximum 50 documents allowed per session")
            for doc_id in v:
                if not doc_id or not doc_id.strip():
                    raise ValueError("Document ID cannot be empty")
        return v
    
    @validator("initial_message")
    def validate_initial_message(cls, v):
        """Validate initial message."""
        if v is not None and len(v.strip()) == 0:
            return None
        return v


class StartChatSessionResponse(BaseModel):
    """Response schema for starting a chat session following Single Responsibility Principle."""
    
    session_id: str = Field(
        ...,
        description="Chat session identifier",
        example="chat_session_123"
    )
    
    success: bool = Field(
        ...,
        description="Whether the session was created successfully",
        example=True
    )
    
    created_at: str = Field(
        ...,
        description="Session creation timestamp in ISO format",
        example="2024-01-15T10:30:00Z"
    )
    
    document_count: int = Field(
        ...,
        description="Number of documents in the session context",
        ge=0,
        example=2
    )
    
    initial_response: Optional[str] = Field(
        None,
        description="Initial AI response if initial message was provided",
        example="This document appears to be about machine learning algorithms..."
    )
    
    message: str = Field(
        ...,
        description="Status message",
        example="Chat session started successfully"
    )


class SendMessageRequest(BaseModel):
    """Request schema for sending a message following Single Responsibility Principle."""
    
    message: str = Field(
        ...,
        description="Message content to send",
        min_length=1,
        max_length=10000,
        example="Can you explain the main concepts in this document?"
    )
    
    document_ids: Optional[List[str]] = Field(
        None,
        description="Optional document IDs to filter context",
        example=["doc_123"]
    )
    
    model: Optional[str] = Field(
        None,
        description="Optional AI model to use for response",
        example="gpt-4"
    )
    
    temperature: float = Field(
        0.7,
        description="Response creativity level (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
        example=0.7
    )
    
    @validator("message")
    def validate_message(cls, v):
        """Validate message content."""
        if not v or not v.strip():
            raise ValueError("Message content cannot be empty")
        return v.strip()
    
    @validator("document_ids")
    def validate_document_ids(cls, v):
        """Validate document IDs."""
        if v is not None:
            if len(v) > 20:
                raise ValueError("Maximum 20 documents allowed per message")
            for doc_id in v:
                if not doc_id or not doc_id.strip():
                    raise ValueError("Document ID cannot be empty")
        return v
    
    @validator("model")
    def validate_model(cls, v):
        """Validate model name."""
        if v is not None:
            allowed_models = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
            if v not in allowed_models:
                raise ValueError(f"Model must be one of: {', '.join(allowed_models)}")
        return v


class CitationSchema(BaseModel):
    """Schema for citation information following Single Responsibility Principle."""
    
    id: str = Field(
        ...,
        description="Citation identifier",
        example="citation_1"
    )
    
    document_id: str = Field(
        ...,
        description="Source document identifier",
        example="doc_123"
    )
    
    document_title: Optional[str] = Field(
        None,
        description="Source document title",
        example="Machine Learning Fundamentals"
    )
    
    page_number: Optional[int] = Field(
        None,
        description="Page number in source document",
        ge=1,
        example=5
    )
    
    chunk_id: Optional[str] = Field(
        None,
        description="Source chunk identifier",
        example="chunk_456"
    )
    
    text_snippet: str = Field(
        ...,
        description="Relevant text snippet from source",
        max_length=500,
        example="Machine learning is a subset of artificial intelligence..."
    )
    
    similarity_score: float = Field(
        ...,
        description="Similarity score for this citation",
        ge=0.0,
        le=1.0,
        example=0.85
    )
    
    citation_type: str = Field(
        "reference",
        description="Type of citation",
        example="reference"
    )


class SendMessageResponse(BaseModel):
    """Response schema for sending a message following Single Responsibility Principle."""
    
    session_id: str = Field(
        ...,
        description="Chat session identifier",
        example="chat_session_123"
    )
    
    message_id: str = Field(
        ...,
        description="Generated message identifier",
        example="msg_789"
    )
    
    user_message: str = Field(
        ...,
        description="Original user message",
        example="Can you explain the main concepts?"
    )
    
    assistant_response: str = Field(
        ...,
        description="AI assistant response",
        example="The main concepts in this document include..."
    )
    
    citations: List[CitationSchema] = Field(
        ...,
        description="List of citations supporting the response",
        example=[]
    )
    
    context_used: Optional[Dict[str, Any]] = Field(
        None,
        description="Context information used for generating response",
        example={"total_chunks": 3, "search_query": "main concepts"}
    )
    
    processing_time_ms: int = Field(
        ...,
        description="Response processing time in milliseconds",
        ge=0,
        example=1500
    )
    
    model_used: str = Field(
        ...,
        description="AI model used for response generation",
        example="gpt-4"
    )
    
    usage: Optional[Dict[str, Any]] = Field(
        None,
        description="Token usage information",
        example={"prompt_tokens": 150, "completion_tokens": 200, "total_tokens": 350}
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata",
        example={"finish_reason": "stop"}
    )
    
    timestamp: str = Field(
        ...,
        description="Response timestamp in ISO format",
        example="2024-01-15T10:35:00Z"
    )


class StreamingMessageChunk(BaseModel):
    """Schema for streaming message chunks following Single Responsibility Principle."""
    
    content: str = Field(
        ...,
        description="Chunk content",
        example="The main"
    )
    
    is_complete: bool = Field(
        ...,
        description="Whether this is the final chunk",
        example=False
    )
    
    chunk_index: int = Field(
        ...,
        description="Index of this chunk in the stream",
        ge=0,
        example=0
    )
    
    session_id: str = Field(
        ...,
        description="Chat session identifier",
        example="chat_session_123"
    )
    
    timestamp: str = Field(
        ...,
        description="Chunk timestamp in ISO format",
        example="2024-01-15T10:35:00Z"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional chunk metadata",
        example={"model": "gpt-4"}
    )


class ChatMessageSchema(BaseModel):
    """Schema for chat message following Single Responsibility Principle."""
    
    id: str = Field(
        ...,
        description="Message identifier",
        example="msg_123"
    )
    
    content: str = Field(
        ...,
        description="Message content",
        example="What is machine learning?"
    )
    
    role: str = Field(
        ...,
        description="Message role (user, assistant, system)",
        example="user"
    )
    
    timestamp: str = Field(
        ...,
        description="Message timestamp in ISO format",
        example="2024-01-15T10:30:00Z"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional message metadata",
        example={"model": "gpt-4", "citations": []}
    )
    
    citations: Optional[List[CitationSchema]] = Field(
        None,
        description="Citations for this message (if applicable)",
        example=[]
    )


class ChatHistoryRequest(BaseModel):
    """Request schema for getting chat history following Single Responsibility Principle."""
    
    limit: Optional[int] = Field(
        50,
        description="Maximum number of messages to return",
        ge=1,
        le=200,
        example=50
    )
    
    offset: int = Field(
        0,
        description="Number of messages to skip",
        ge=0,
        example=0
    )
    
    include_context: bool = Field(
        False,
        description="Whether to include context and citations",
        example=False
    )


class ChatHistoryResponse(BaseModel):
    """Response schema for chat history following Single Responsibility Principle."""
    
    session_id: str = Field(
        ...,
        description="Chat session identifier",
        example="chat_session_123"
    )
    
    messages: List[ChatMessageSchema] = Field(
        ...,
        description="List of messages in chronological order",
        example=[]
    )
    
    total_messages: int = Field(
        ...,
        description="Total number of messages in the session",
        ge=0,
        example=10
    )
    
    has_more: bool = Field(
        ...,
        description="Whether there are more messages available",
        example=False
    )
    
    session_info: Dict[str, Any] = Field(
        ...,
        description="Session information and metadata",
        example={
            "created_at": "2024-01-15T10:00:00Z",
            "document_count": 2
        }
    )
    
    pagination: Dict[str, Any] = Field(
        ...,
        description="Pagination information",
        example={
            "limit": 50,
            "offset": 0,
            "returned_count": 10
        }
    )


class ChatSessionSummaryResponse(BaseModel):
    """Response schema for chat session summary following Single Responsibility Principle."""
    
    session_id: str = Field(
        ...,
        description="Chat session identifier",
        example="chat_session_123"
    )
    
    created_at: str = Field(
        ...,
        description="Session creation timestamp",
        example="2024-01-15T10:00:00Z"
    )
    
    updated_at: str = Field(
        ...,
        description="Session last update timestamp",
        example="2024-01-15T11:00:00Z"
    )
    
    document_count: int = Field(
        ...,
        description="Number of documents in session context",
        ge=0,
        example=2
    )
    
    total_messages: int = Field(
        ...,
        description="Total number of messages in session",
        ge=0,
        example=10
    )
    
    user_messages: int = Field(
        ...,
        description="Number of user messages",
        ge=0,
        example=5
    )
    
    assistant_messages: int = Field(
        ...,
        description="Number of assistant messages",
        ge=0,
        example=5
    )
    
    total_citations: int = Field(
        ...,
        description="Total number of citations generated",
        ge=0,
        example=15
    )
    
    duration_seconds: float = Field(
        ...,
        description="Session duration in seconds",
        ge=0.0,
        example=3600.0
    )
    
    average_response_time: float = Field(
        ...,
        description="Average response time in seconds",
        ge=0.0,
        example=2.5
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional session metadata",
        example={"user_id": "user_123"}
    )


class ErrorResponse(BaseModel):
    """Error response schema following Single Responsibility Principle."""
    
    error: str = Field(
        ...,
        description="Error type or code",
        example="VALIDATION_ERROR"
    )
    
    message: str = Field(
        ...,
        description="Human-readable error message",
        example="Invalid request parameters"
    )
    
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details",
        example={"field": "message", "issue": "cannot be empty"}
    )
    
    timestamp: str = Field(
        ...,
        description="Error timestamp in ISO format",
        example="2024-01-15T10:30:00Z"
    )
    
    request_id: Optional[str] = Field(
        None,
        description="Request identifier for tracking",
        example="req_123"
    )


class ChatHealthResponse(BaseModel):
    """Health check response schema following Single Responsibility Principle."""
    
    status: str = Field(
        ...,
        description="Overall health status",
        example="healthy"
    )
    
    timestamp: str = Field(
        ...,
        description="Health check timestamp",
        example="2024-01-15T10:30:00Z"
    )
    
    services: Dict[str, str] = Field(
        ...,
        description="Individual service health status",
        example={
            "chat_service": "healthy",
            "rag_service": "healthy",
            "vector_database": "healthy"
        }
    )
    
    version: Optional[str] = Field(
        None,
        description="API version",
        example="1.0.0"
    )
    
    uptime_seconds: Optional[float] = Field(
        None,
        description="Service uptime in seconds",
        ge=0.0,
        example=86400.0
    )
