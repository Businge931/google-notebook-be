"""
Conversation Service Implementation

"""
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

from src.core.domain.services.chat_service import (
    ConversationService,
    ChatMessage
)
from src.core.domain.value_objects import SessionId
from src.core.domain.repositories import ChatRepository
from src.shared.exceptions import (
    ConversationError,
    ChatSessionNotFoundError
)


class ConversationServiceImpl(ConversationService):
    """
    Implementation of ConversationService following Single Responsibility Principle.
    
    Handles conversation flow and context management using chat repository.
    """
    
    def __init__(
        self,
        chat_repository: ChatRepository,
        max_history_length: int = 50
    ):
        """
        Initialize conversation service.
        
        Args:
            chat_repository: Repository for chat operations
            max_history_length: Maximum conversation history length
        """
        self._chat_repository = chat_repository
        self._max_history_length = max_history_length
        self._logger = logging.getLogger(__name__)
    
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
        try:
            # Check if session already exists
            existing_session = await self._chat_repository.get_session_by_id(session_id)
            if existing_session:
                self._logger.warning(f"Conversation session {session_id.value} already exists")
                return False
            
            # Create new session with initial context
            from src.core.domain.entities import ChatSession
            
            chat_session = ChatSession(
                id=session_id,
                document_ids=[],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                metadata=initial_context or {}
            )
            
            # Save session
            await self._chat_repository.create_session(chat_session)
            
            self._logger.info(f"Started conversation session {session_id.value}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to start conversation {session_id.value}: {e}")
            raise ConversationError(f"Failed to start conversation: {str(e)}")
    
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
        try:
            # Check if session exists
            chat_session = await self._chat_repository.get_session_by_id(session_id)
            if not chat_session:
                raise ChatSessionNotFoundError(f"Chat session {session_id.value} not found")
            
            # Create domain message
            from src.core.domain.entities import Message
            from src.core.domain.entities.message import MessageStatus
            import uuid
            
            domain_message = Message(
                message_id=str(uuid.uuid4()),
                session_id=session_id,
                content=message.content,
                role=message.role,
                status=MessageStatus.COMPLETED,
                created_at=message.timestamp,
                updated_at=message.timestamp,
                citations=[]
            )
            
            # Add message to repository
            await self._chat_repository.add_message(domain_message)
            
            # Update session timestamp
            chat_session.updated_at = datetime.utcnow()
            await self._chat_repository.update_session(chat_session)
            
            # Cleanup old messages if history is too long
            await self._cleanup_old_messages(session_id)
            
            self._logger.debug(f"Added message to conversation {session_id.value}")
            return True
            
        except ChatSessionNotFoundError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to add message to conversation {session_id.value}: {e}")
            raise ConversationError(f"Failed to add message: {str(e)}")
    
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
        try:
            # Check if session exists
            chat_session = await self._chat_repository.get_session_by_id(session_id)
            if not chat_session:
                raise ChatSessionNotFoundError(f"Chat session {session_id.value} not found")
            
            # Get messages from repository
            messages = await self._chat_repository.get_messages_by_session(
                session_id=session_id,
                limit=limit or self._max_history_length
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
            
        except ChatSessionNotFoundError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to get conversation history for {session_id.value}: {e}")
            raise ConversationError(f"Failed to get conversation history: {str(e)}")
    
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
        try:
            # Check if session exists
            chat_session = await self._chat_repository.get_session_by_id(session_id)
            if not chat_session:
                raise ChatSessionNotFoundError(f"Chat session {session_id.value} not found")
            
            # Get all messages
            messages = await self._chat_repository.get_messages_by_session(session_id)
            
            # Delete all messages
            for message in messages:
                await self._chat_repository.delete_message(message.message_id)
            
            # Update session timestamp
            chat_session.updated_at = datetime.utcnow()
            await self._chat_repository.update_session(chat_session)
            
            self._logger.info(f"Cleared conversation history for session {session_id.value}")
            return True
            
        except ChatSessionNotFoundError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to clear conversation {session_id.value}: {e}")
            raise ConversationError(f"Failed to clear conversation: {str(e)}")
    
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
        try:
            # Check if session exists
            chat_session = await self._chat_repository.get_session_by_id(session_id)
            if not chat_session:
                raise ChatSessionNotFoundError(f"Chat session {session_id.value} not found")
            
            # Get all messages
            messages = await self._chat_repository.get_messages_by_session(session_id)
            
            # Calculate statistics
            user_messages = [m for m in messages if m.role == "user"]
            assistant_messages = [m for m in messages if m.role == "assistant"]
            system_messages = [m for m in messages if m.role == "system"]
            
            # Calculate conversation duration
            if messages:
                duration_seconds = (messages[-1].created_at - messages[0].created_at).total_seconds()
            else:
                duration_seconds = 0
            
            # Calculate average message length
            total_chars = sum(len(m.content) for m in messages)
            avg_message_length = total_chars / len(messages) if messages else 0
            
            # Count citations
            total_citations = 0
            for message in assistant_messages:
                if message.metadata and "citations" in message.metadata:
                    total_citations += len(message.metadata["citations"])
            
            # Calculate response times
            response_times = []
            for i in range(len(messages) - 1):
                current_msg = messages[i]
                next_msg = messages[i + 1]
                
                if current_msg.role == "user" and next_msg.role == "assistant":
                    response_time = (next_msg.created_at - current_msg.created_at).total_seconds()
                    response_times.append(response_time)
            
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            
            return {
                "session_id": session_id.value,
                "created_at": chat_session.created_at.isoformat(),
                "updated_at": chat_session.updated_at.isoformat(),
                "total_messages": len(messages),
                "user_messages": len(user_messages),
                "assistant_messages": len(assistant_messages),
                "system_messages": len(system_messages),
                "duration_seconds": duration_seconds,
                "average_message_length": avg_message_length,
                "total_citations": total_citations,
                "average_response_time": avg_response_time,
                "document_count": 1 if chat_session.document_id else 0,
                "metadata": {}
            }
            
        except ChatSessionNotFoundError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to get conversation summary for {session_id.value}: {e}")
            raise ConversationError(f"Failed to get conversation summary: {str(e)}")
    
    async def _cleanup_old_messages(self, session_id: SessionId) -> None:
        """
        Clean up old messages if conversation history is too long.
        
        Args:
            session_id: Session identifier
        """
        try:
            # Get all messages
            messages = await self._chat_repository.get_messages_by_session(session_id)
            
            # Check if cleanup is needed
            if len(messages) <= self._max_history_length:
                return
            
            # Sort messages by timestamp (oldest first)
            messages.sort(key=lambda m: m.created_at)
            
            # Calculate how many messages to delete
            messages_to_delete = len(messages) - self._max_history_length
            
            # Delete oldest messages
            for i in range(messages_to_delete):
                await self._chat_repository.delete_message(messages[i].message_id)
            
            self._logger.info(
                f"Cleaned up {messages_to_delete} old messages from session {session_id.value}"
            )
            
        except Exception as e:
            self._logger.error(f"Failed to cleanup old messages for {session_id.value}: {e}")
            # Don't raise exception for cleanup failures


class ConversationServiceFactory:
    """
    Factory for creating conversation service instances.
    
    Follows Open/Closed Principle for extensibility.
    """
    
    @staticmethod
    def create_conversation_service(
        chat_repository: ChatRepository,
        max_history_length: int = 50
    ) -> ConversationServiceImpl:
        """
        Create conversation service instance.
        
        Args:
            chat_repository: Chat repository
            max_history_length: Maximum conversation history length
            
        Returns:
            Configured conversation service
        """
        return ConversationServiceImpl(
            chat_repository=chat_repository,
            max_history_length=max_history_length
        )
