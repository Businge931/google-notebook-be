"""
Chat Use Cases

"""
from typing import List, Optional, Dict, Any, AsyncIterator
from dataclasses import dataclass
import logging
import uuid
from datetime import datetime, timezone

from ...domain.entities import ChatSession, Message
from ...domain.entities.message import MessageRole, MessageStatus
from ...domain.value_objects import SessionId, DocumentId
from ...domain.repositories import ChatRepository
from ...domain.services.chat_service import (
    ChatService,
    RAGService,
    ChatMessage,
    ChatResponse,
    ChatContext,
    StreamingChatChunk
)
from src.shared.exceptions import (
    ChatSessionNotFoundError,
    MessageGenerationError,
    ValidationError
)


@dataclass
class StartChatRequest:
    """Request object for starting chat session following Single Responsibility Principle."""
    session_id: SessionId
    document_ids: Optional[List[DocumentId]] = None
    initial_message: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class StartChatResponse:
    """Response object for starting chat session following Single Responsibility Principle."""
    session_id: str
    success: bool
    created_at: str
    document_count: int
    initial_response: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class SendMessageRequest:
    """Request object for sending message following Single Responsibility Principle."""
    session_id: SessionId
    message: str
    document_ids: Optional[List[DocumentId]] = None
    model: Optional[str] = None
    temperature: float = 0.7
    stream: bool = False


@dataclass
class SendMessageResponse:
    """Response object for sending message following Single Responsibility Principle."""
    session_id: str
    message_id: str
    response: str
    citations: List[Dict[str, Any]]
    context_used: Optional[Dict[str, Any]]
    processing_time_ms: int
    model_used: str
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ChatHistoryRequest:
    """Request object for getting chat history following Single Responsibility Principle."""
    session_id: SessionId
    limit: Optional[int] = None
    offset: int = 0
    include_context: bool = False


@dataclass
class ChatHistoryResponse:
    """Response object for chat history following Single Responsibility Principle."""
    session_id: str
    messages: List[Dict[str, Any]]
    total_messages: int
    has_more: bool
    session_info: Dict[str, Any]


class ChatUseCase:
    """
    Use case for chat operations following Single Responsibility Principle.
    
    Orchestrates chat sessions, message handling, and conversation management.
    """
    
    def __init__(
        self,
        chat_repository: ChatRepository,
        rag_service: RAGService,
        chat_service: ChatService
    ):
        """
        Initialize chat use case.
        
        Args:
            chat_repository: Repository for chat operations
            rag_service: RAG service for context retrieval
            chat_service: Chat service for message generation
        """
        self._chat_repository = chat_repository
        self._rag_service = rag_service
        self._chat_service = chat_service
        self._logger = logging.getLogger(__name__)
    
    async def start_chat_session(
        self,
        request: StartChatRequest
    ) -> StartChatResponse:
        """
        Start a new chat session.
        
        Args:
            request: Start chat request
            
        Returns:
            Start chat response
            
        Raises:
            ValidationError: If request is invalid
            MessageGenerationError: If initial response generation fails
        """
        try:
            # Validate request
            if not request.session_id:
                raise ValidationError("Session ID is required")
            
            # Check if session already exists
            existing_session = await self._chat_repository.get_session_by_id(request.session_id)
            if existing_session:
                self._logger.warning(f"Chat session {request.session_id.value} already exists")
                return StartChatResponse(
                    session_id=request.session_id.value,
                    success=False,
                    created_at=existing_session.created_at.isoformat(),
                    document_count=1 if existing_session.document_id else 0,
                    error_message="Session already exists"
                )
            
            # Create new chat session
            # Use first document ID if provided, otherwise None for document-less sessions
            document_id = request.document_ids[0] if request.document_ids else None
            
            chat_session = ChatSession(
                session_id=request.session_id,
                document_id=document_id,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            
            # Save session
            await self._chat_repository.create_session(chat_session)
            
            # Generate initial response if initial message provided
            initial_response = None
            if request.initial_message:
                try:
                    message_request = SendMessageRequest(
                        session_id=request.session_id,
                        message=request.initial_message,
                        document_ids=request.document_ids
                    )
                    message_response = await self.send_message(message_request)
                    initial_response = message_response.response
                except Exception as e:
                    self._logger.error(f"Failed to generate initial response: {e}")
                    # Don't fail the session creation if initial response fails
            
            self._logger.info(f"Started chat session {request.session_id.value}")
            
            return StartChatResponse(
                session_id=request.session_id.value,
                success=True,
                created_at=chat_session.created_at.isoformat(),
                document_count=1 if chat_session.document_id else 0,
                initial_response=initial_response
            )
            
        except ValidationError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to start chat session: {e}")
            return StartChatResponse(
                session_id=request.session_id.value,
                success=False,
                created_at=datetime.now(timezone.utc).isoformat(),
                document_count=0,
                error_message=str(e)
            )
    
    async def send_message(
        self,
        request: SendMessageRequest
    ) -> SendMessageResponse:
        """
        Send a message and get response.
        
        Args:
            request: Send message request
            
        Returns:
            Send message response
            
        Raises:
            ChatSessionNotFoundError: If session doesn't exist
            ValidationError: If request is invalid
            MessageGenerationError: If response generation fails
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Validate request
            if not request.session_id:
                raise ValidationError("Session ID is required")
            
            if not request.message or not request.message.strip():
                raise ValidationError("Message content is required")
            
            # Get chat session
            chat_session = await self._chat_repository.get_session_by_id(request.session_id)
            if not chat_session:
                raise ChatSessionNotFoundError(f"Chat session {request.session_id.value} not found")
            
            # Get conversation history
            conversation_history = await self._get_conversation_history(request.session_id)
            
            # Use document IDs from request or session
            document_ids = request.document_ids or ([chat_session.document_id] if chat_session.document_id else [])
            
            # Generate RAG response
            rag_response = await self._rag_service.generate_rag_response(
                query=request.message,
                conversation_history=conversation_history,
                document_ids=document_ids,
                model=request.model,
                temperature=request.temperature
            )
            
            # Create user message
            user_message = Message(
                message_id=str(uuid.uuid4()),
                session_id=request.session_id,
                role=MessageRole.USER,
                content=request.message,
                status=MessageStatus.COMPLETED,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            
            # Create assistant message
            assistant_message = Message(
                message_id=str(uuid.uuid4()),
                session_id=request.session_id,
                role=MessageRole.ASSISTANT,
                content=rag_response.message,
                status=MessageStatus.COMPLETED,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                citations=[]
            )
            
            # Save messages
            await self._chat_repository.add_message(user_message)
            assistant_message_with_id = await self._chat_repository.add_message(assistant_message)
            
            # Update session timestamp
            chat_session.updated_at = datetime.now(timezone.utc)
            await self._chat_repository.update_session(chat_session)
            
            # Calculate processing time
            end_time = datetime.now(timezone.utc)
            processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            self._logger.info(
                f"Generated response for session {request.session_id.value} in {processing_time_ms}ms"
            )
            
            return SendMessageResponse(
                session_id=request.session_id.value,
                message_id=assistant_message_with_id.message_id,
                response=rag_response.message,
                citations=rag_response.citations,
                context_used=rag_response.context_used.__dict__ if rag_response.context_used else None,
                processing_time_ms=processing_time_ms,
                model_used=rag_response.model,
                usage=rag_response.usage,
                metadata=rag_response.metadata
            )
            
        except (ChatSessionNotFoundError, ValidationError):
            raise
        except Exception as e:
            self._logger.error(f"Failed to send message: {e}")
            raise MessageGenerationError(f"Failed to send message: {str(e)}")
    
    async def send_streaming_message(
        self,
        request: SendMessageRequest
    ) -> AsyncIterator[StreamingChatChunk]:
        """
        Send a message and get streaming response.
        
        Args:
            request: Send message request
            
        Yields:
            Streaming chat chunks
            
        Raises:
            ChatSessionNotFoundError: If session doesn't exist
            ValidationError: If request is invalid
            MessageGenerationError: If response generation fails
        """
        try:
            # Validate request
            if not request.session_id:
                raise ValidationError("Session ID is required")
            
            if not request.message or not request.message.strip():
                raise ValidationError("Message content is required")
            
            # Get chat session
            chat_session = await self._chat_repository.get_session_by_id(request.session_id)
            if not chat_session:
                raise ChatSessionNotFoundError(f"Chat session {request.session_id.value} not found")
            
            # Get conversation history
            conversation_history = await self._get_conversation_history(request.session_id)
            
            # Retrieve context
            document_ids = request.document_ids or ([chat_session.document_id] if chat_session.document_id else [])
            context = await self._rag_service.retrieve_context(
                query=request.message,
                document_ids=document_ids,
                max_chunks=5,
                similarity_threshold=0.7
            )
            
            # Add user message to history
            messages = conversation_history + [
                ChatMessage(
                    content=request.message,
                    role="user",
                    timestamp=datetime.now(timezone.utc)
                )
            ]
            
            # Generate streaming response
            full_response = ""
            async for chunk in self._chat_service.generate_streaming_response(
                messages=messages,
                context=context,
                model=request.model,
                temperature=request.temperature
            ):
                if not chunk.is_complete:
                    full_response += chunk.content
                
                yield chunk
            
            # Save messages after streaming is complete
            user_message = Message(
                message_id=str(uuid.uuid4()),
                session_id=request.session_id,
                role=MessageRole.USER,
                content=request.message,
                status=MessageStatus.COMPLETED,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            
            assistant_message = Message(
                message_id=str(uuid.uuid4()),
                session_id=request.session_id,
                role=MessageRole.ASSISTANT,
                content=full_response,
                status=MessageStatus.COMPLETED,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                citations=[]
            )
            
            await self._chat_repository.add_message(user_message)
            await self._chat_repository.add_message(assistant_message)
            
            # Update session timestamp
            chat_session.updated_at = datetime.now(timezone.utc)
            await self._chat_repository.update_session(chat_session)
            
        except (ChatSessionNotFoundError, ValidationError):
            raise
        except Exception as e:
            self._logger.error(f"Failed to send streaming message: {e}")
            raise MessageGenerationError(f"Failed to send streaming message: {str(e)}")
    
    async def get_chat_history(
        self,
        request: ChatHistoryRequest
    ) -> ChatHistoryResponse:
        """
        Get chat history for a session.
        
        Args:
            request: Chat history request
            
        Returns:
            Chat history response
            
        Raises:
            ChatSessionNotFoundError: If session doesn't exist
            ValidationError: If request is invalid
        """
        try:
            # Validate request
            if not request.session_id:
                raise ValidationError("Session ID is required")
            
            # Get chat session
            chat_session = await self._chat_repository.get_session_by_id(request.session_id)
            if not chat_session:
                raise ChatSessionNotFoundError(f"Chat session {request.session_id.value} not found")
            
            # Get messages
            messages = await self._chat_repository.get_messages_by_session(
                session_id=request.session_id,
                limit=request.limit,
                offset=request.offset
            )
            
            # Convert messages to response format
            message_dicts = []
            for message in messages:
                message_dict = {
                    "id": message.message_id,
                    "content": message.content,
                    "role": message.role.value,  # Convert enum to string for JSON serialization
                    "timestamp": message.created_at.isoformat(),
                    "metadata": {}  # Use empty dict since Message entity doesn't have metadata
                }
                
                # Include context if requested - use citations from Message entity
                if request.include_context and message.citations:
                    message_dict["citations"] = [
                        {
                            "citation_id": citation.citation_id,
                            "page_number": citation.page_number.value,
                            "text_snippet": citation.text_snippet,
                            "start_position": citation.start_position,
                            "end_position": citation.end_position,
                            "confidence_score": citation.confidence_score
                        }
                        for citation in message.citations
                    ]
                
                message_dicts.append(message_dict)
            
            # Check if there are more messages
            total_messages = len(await self._chat_repository.get_messages_by_session(request.session_id))
            has_more = (request.offset + len(messages)) < total_messages
            
            # Session info
            session_info = {
                "session_id": chat_session.id.value,
                "created_at": chat_session.created_at.isoformat(),
                "updated_at": chat_session.updated_at.isoformat(),
                "document_count": 1 if chat_session.document_id else 0,
                "document_ids": [chat_session.document_id.value] if chat_session.document_id else [],
                "metadata": {}
            }
            
            return ChatHistoryResponse(
                session_id=request.session_id.value,
                messages=message_dicts,
                total_messages=total_messages,
                has_more=has_more,
                session_info=session_info
            )
            
        except (ChatSessionNotFoundError, ValidationError):
            raise
        except Exception as e:
            self._logger.error(f"Failed to get chat history: {e}")
            raise MessageGenerationError(f"Failed to get chat history: {str(e)}")
    
    async def delete_chat_session(
        self,
        session_id: SessionId
    ) -> bool:
        """
        Delete a chat session and all its messages.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deletion was successful
            
        Raises:
            ChatSessionNotFoundError: If session doesn't exist
        """
        try:
            # Check if session exists
            chat_session = await self._chat_repository.get_session_by_id(session_id)
            if not chat_session:
                raise ChatSessionNotFoundError(f"Chat session {session_id.value} not found")
            
            # Delete session (should cascade delete messages)
            success = await self._chat_repository.delete_session(session_id)
            
            if success:
                self._logger.info(f"Deleted chat session {session_id.value}")
            else:
                self._logger.warning(f"Failed to delete chat session {session_id.value}")
            
            return success
            
        except ChatSessionNotFoundError:
            raise
        except Exception as e:
            self._logger.error(f"Error deleting chat session {session_id.value}: {e}")
            return False
    
    async def get_session_summary(
        self,
        session_id: SessionId
    ) -> Dict[str, Any]:
        """
        Get summary and statistics for a chat session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session summary with statistics
            
        Raises:
            ChatSessionNotFoundError: If session doesn't exist
        """
        try:
            # Get chat session
            chat_session = await self._chat_repository.get_session_by_id(session_id)
            if not chat_session:
                raise ChatSessionNotFoundError(f"Chat session {session_id.value} not found")
            
            # Get all messages
            messages = await self._chat_repository.get_messages_by_session(session_id)
            
            # Calculate statistics
            user_messages = [m for m in messages if m.role == "user"]
            assistant_messages = [m for m in messages if m.role == "assistant"]
            
            total_citations = 0
            for message in assistant_messages:
                if message.metadata and "citations" in message.metadata:
                    total_citations += len(message.metadata["citations"])
            
            # Session duration
            if messages:
                duration_seconds = (messages[-1].timestamp - messages[0].timestamp).total_seconds()
            else:
                duration_seconds = 0
            
            return {
                "session_id": session_id.value,
                "created_at": chat_session.created_at.isoformat(),
                "updated_at": chat_session.updated_at.isoformat(),
                "document_count": 1 if chat_session.document_id else 0,
                "total_messages": len(messages),
                "user_messages": len(user_messages),
                "assistant_messages": len(assistant_messages),
                "total_citations": total_citations,
                "duration_seconds": duration_seconds,
                "average_response_time": self._calculate_average_response_time(messages),
                "metadata": {}
            }
            
        except ChatSessionNotFoundError:
            raise
        except Exception as e:
            self._logger.error(f"Error getting session summary for {session_id.value}: {e}")
            raise MessageGenerationError(f"Failed to get session summary: {str(e)}")
    
    async def _get_conversation_history(
        self,
        session_id: SessionId,
        limit: int = 10
    ) -> List[ChatMessage]:
        """
        Get conversation history as ChatMessage objects.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages
            
        Returns:
            List of chat messages
        """
        try:
            messages = await self._chat_repository.get_messages_by_session(
                session_id=session_id,
                limit=limit
            )
            
            # Convert to ChatMessage objects
            chat_messages = []
            for message in messages:
                chat_message = ChatMessage(
                    content=message.content,
                    role=message.role,
                    timestamp=message.created_at,
                    metadata={}  # Use empty dict since Message entity doesn't have metadata
                )
                chat_messages.append(chat_message)
            
            return chat_messages
            
        except Exception as e:
            self._logger.error(f"Failed to get conversation history: {e}")
            return []
    
    def _calculate_average_response_time(self, messages: List[Message]) -> float:
        """
        Calculate average response time between user and assistant messages.
        
        Args:
            messages: List of messages
            
        Returns:
            Average response time in seconds
        """
        try:
            response_times = []
            
            for i in range(len(messages) - 1):
                current_msg = messages[i]
                next_msg = messages[i + 1]
                
                if current_msg.role == "user" and next_msg.role == "assistant":
                    response_time = (next_msg.timestamp - current_msg.timestamp).total_seconds()
                    response_times.append(response_time)
            
            return sum(response_times) / len(response_times) if response_times else 0.0
            
        except Exception:
            return 0.0
