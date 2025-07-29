"""
Chat Service Interface

"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, AsyncIterator
from dataclasses import dataclass
from datetime import datetime

from ..value_objects import SessionId, DocumentId
from ..entities import Message


@dataclass
class ChatContext:
    """
    Represents context for chat conversation following Single Responsibility Principle.
    """
    relevant_chunks: List[Dict[str, Any]]
    document_ids: List[DocumentId]
    similarity_scores: List[float]
    total_chunks: int
    search_query: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ChatMessage:
    """
    Represents a chat message with metadata following Single Responsibility Principle.
    """
    content: str
    role: str  # "user", "assistant", "system"
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ChatResponse:
    """
    Represents a chat response following Single Responsibility Principle.
    """
    message: str
    citations: List[Dict[str, Any]]
    context_used: ChatContext
    model: str
    usage: Optional[Dict[str, Any]] = None
    processing_time_ms: int = 0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class StreamingChatChunk:
    """
    Represents a streaming chat chunk following Single Responsibility Principle.
    """
    content: str
    is_complete: bool
    chunk_index: int
    metadata: Optional[Dict[str, Any]] = None


class ChatService(ABC):
    """
    Abstract interface for chat services.
    
    Defines the contract for chat message generation following
    Interface Segregation Principle.
    """
    
    @abstractmethod
    async def generate_response(
        self,
        messages: List[ChatMessage],
        context: Optional[ChatContext] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> ChatResponse:
        """
        Generate a chat response.
        
        Args:
            messages: Conversation history
            context: Optional context from document retrieval
            model: Optional model name to use
            temperature: Response creativity (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            
        Returns:
            Chat response with citations
            
        Raises:
            ChatError: If response generation fails
        """
        pass
    
    @abstractmethod
    async def generate_streaming_response(
        self,
        messages: List[ChatMessage],
        context: Optional[ChatContext] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> AsyncIterator[StreamingChatChunk]:
        """
        Generate a streaming chat response.
        
        Args:
            messages: Conversation history
            context: Optional context from document retrieval
            model: Optional model name to use
            temperature: Response creativity (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            
        Yields:
            Streaming chat chunks
            
        Raises:
            ChatError: If response generation fails
        """
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Get list of available chat models.
        
        Returns:
            List of model names
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the chat service is healthy.
        
        Returns:
            True if service is healthy
        """
        pass


class RAGService(ABC):
    """
    Abstract interface for Retrieval-Augmented Generation services.
    
    Handles document retrieval and context preparation for chat following
    Interface Segregation Principle.
    """
    
    @abstractmethod
    async def retrieve_context(
        self,
        query: str,
        document_ids: Optional[List[DocumentId]] = None,
        max_chunks: int = 5,
        similarity_threshold: float = 0.7
    ) -> ChatContext:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: User query
            document_ids: Optional document filter
            max_chunks: Maximum chunks to retrieve
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            Chat context with relevant chunks
            
        Raises:
            RAGError: If context retrieval fails
        """
        pass
    
    @abstractmethod
    async def generate_rag_response(
        self,
        query: str,
        conversation_history: List[ChatMessage],
        document_ids: Optional[List[DocumentId]] = None,
        model: Optional[str] = None,
        temperature: float = 0.7
    ) -> ChatResponse:
        """
        Generate a RAG-enhanced response.
        
        Args:
            query: User query
            conversation_history: Previous messages
            document_ids: Optional document filter
            model: Optional model name
            temperature: Response creativity
            
        Returns:
            Chat response with citations
            
        Raises:
            RAGError: If RAG response generation fails
        """
        pass
    
    @abstractmethod
    async def extract_citations(
        self,
        response_text: str,
        context: ChatContext
    ) -> List[Dict[str, Any]]:
        """
        Extract citations from response text based on context.
        
        Args:
            response_text: Generated response text
            context: Context used for generation
            
        Returns:
            List of citations with source information
            
        Raises:
            CitationError: If citation extraction fails
        """
        pass


class ConversationService(ABC):
    """
    Abstract interface for conversation management services.
    
    Handles conversation flow and context management following
    Interface Segregation Principle.
    """
    
    @abstractmethod
    async def start_conversation(
        self,
        session_id: SessionId,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Start a new conversation session.
        
        Args:
            session_id: Session identifier
            initial_context: Optional initial context
            
        Returns:
            True if conversation started successfully
            
        Raises:
            ConversationError: If conversation start fails
        """
        pass
    
    @abstractmethod
    async def add_message(
        self,
        session_id: SessionId,
        message: ChatMessage
    ) -> bool:
        """
        Add a message to conversation history.
        
        Args:
            session_id: Session identifier
            message: Message to add
            
        Returns:
            True if message added successfully
            
        Raises:
            ConversationError: If message addition fails
        """
        pass
    
    @abstractmethod
    async def get_conversation_history(
        self,
        session_id: SessionId,
        limit: Optional[int] = None
    ) -> List[ChatMessage]:
        """
        Get conversation history.
        
        Args:
            session_id: Session identifier
            limit: Optional limit on number of messages
            
        Returns:
            List of messages in chronological order
            
        Raises:
            ConversationError: If history retrieval fails
        """
        pass
    
    @abstractmethod
    async def clear_conversation(
        self,
        session_id: SessionId
    ) -> bool:
        """
        Clear conversation history.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if conversation cleared successfully
            
        Raises:
            ConversationError: If conversation clearing fails
        """
        pass
    
    @abstractmethod
    async def get_conversation_summary(
        self,
        session_id: SessionId
    ) -> Dict[str, Any]:
        """
        Get conversation summary and statistics.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Conversation summary with statistics
            
        Raises:
            ConversationError: If summary generation fails
        """
        pass


class ChatServiceFactory(ABC):
    """
    Abstract factory for creating chat service instances.
    
    Follows Open/Closed Principle for extensibility.
    """
    
    @abstractmethod
    def create_chat_service(self, config: Dict[str, Any]) -> ChatService:
        """
        Create chat service instance.
        
        Args:
            config: Service configuration
            
        Returns:
            Configured chat service
        """
        pass
    
    @abstractmethod
    def create_rag_service(
        self,
        chat_service: ChatService,
        config: Dict[str, Any]
    ) -> RAGService:
        """
        Create RAG service instance.
        
        Args:
            chat_service: Base chat service
            config: Service configuration
            
        Returns:
            Configured RAG service
        """
        pass
    
    @abstractmethod
    def create_conversation_service(
        self,
        config: Dict[str, Any]
    ) -> ConversationService:
        """
        Create conversation service instance.
        
        Args:
            config: Service configuration
            
        Returns:
            Configured conversation service
        """
        pass
