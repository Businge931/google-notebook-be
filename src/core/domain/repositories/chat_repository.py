"""
Chat Repository Interface

"""
from abc import ABC, abstractmethod
from typing import List, Optional

from ..entities import ChatSession, Message, SessionStatus
from ..value_objects import SessionId, DocumentId


class ChatRepository(ABC):
    """
    Abstract repository interface for chat persistence operations.
    
    Defines the contract for chat session and message storage without
    coupling to specific persistence technologies.
    """
    
    @abstractmethod
    async def save_session(self, session: ChatSession) -> ChatSession:
        """
        Save a chat session to the repository.
        
        Args:
            session: Chat session entity to save
            
        Returns:
            Saved chat session entity
            
        Raises:
            RepositoryError: If save operation fails
        """
        pass
    
    @abstractmethod
    async def find_session_by_id(self, session_id: SessionId) -> Optional[ChatSession]:
        """
        Find a chat session by its ID.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Chat session entity if found, None otherwise
            
        Raises:
            RepositoryError: If query operation fails
        """
        pass
    
    @abstractmethod
    async def find_sessions_by_document(
        self,
        document_id: DocumentId,
        status: Optional[SessionStatus] = None
    ) -> List[ChatSession]:
        """
        Find chat sessions for a specific document.
        
        Args:
            document_id: Document identifier
            status: Optional status filter
            
        Returns:
            List of chat sessions for the document
            
        Raises:
            RepositoryError: If query operation fails
        """
        pass
    
    @abstractmethod
    async def update_session(self, session: ChatSession) -> ChatSession:
        """
        Update an existing chat session.
        
        Args:
            session: Chat session entity with updated data
            
        Returns:
            Updated chat session entity
            
        Raises:
            RepositoryError: If update operation fails
            SessionNotFoundError: If session doesn't exist
        """
        pass
    
    @abstractmethod
    async def delete_session(self, session_id: SessionId) -> bool:
        """
        Delete a chat session by its ID.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            True if session was deleted, False if not found
            
        Raises:
            RepositoryError: If delete operation fails
        """
        pass
    
    @abstractmethod
    async def save_message(self, message: Message) -> Message:
        """
        Save a message to the repository.
        
        Args:
            message: Message entity to save
            
        Returns:
            Saved message entity
            
        Raises:
            RepositoryError: If save operation fails
        """
        pass
    
    @abstractmethod
    async def find_message_by_id(self, message_id: str) -> Optional[Message]:
        """
        Find a message by its ID.
        
        Args:
            message_id: Unique message identifier
            
        Returns:
            Message entity if found, None otherwise
            
        Raises:
            RepositoryError: If query operation fails
        """
        pass
    
    @abstractmethod
    async def find_messages_by_session(
        self,
        session_id: SessionId,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[Message]:
        """
        Find messages for a specific session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to return
            offset: Number of messages to skip
            
        Returns:
            List of messages for the session, ordered by creation time
            
        Raises:
            RepositoryError: If query operation fails
        """
        pass
    
    @abstractmethod
    async def update_message(self, message: Message) -> Message:
        """
        Update an existing message.
        
        Args:
            message: Message entity with updated data
            
        Returns:
            Updated message entity
            
        Raises:
            RepositoryError: If update operation fails
            MessageNotFoundError: If message doesn't exist
        """
        pass
    
    @abstractmethod
    async def delete_message(self, message_id: str) -> bool:
        """
        Delete a message by its ID.
        
        Args:
            message_id: Unique message identifier
            
        Returns:
            True if message was deleted, False if not found
            
        Raises:
            RepositoryError: If delete operation fails
        """
        pass
    
    @abstractmethod
    async def get_session_message_count(self, session_id: SessionId) -> int:
        """
        Get the total number of messages in a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Total number of messages in the session
            
        Raises:
            RepositoryError: If count operation fails
        """
        pass
