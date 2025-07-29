"""
Chat Repository Implementation

"""
from typing import List, Optional
from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.core.domain.entities import ChatSession, Message, Citation, SessionStatus, MessageRole
from src.core.domain.repositories import ChatRepository
from src.core.domain.value_objects import DocumentId, SessionId, PageNumber
from src.shared.exceptions import ChatSessionNotFoundError, MessageNotFoundError, RepositoryError
from ..models import ChatSessionModel, MessageModel, CitationModel


class ChatRepositoryImpl(ChatRepository):
    """
    SQLAlchemy implementation of ChatRepository.
    
    Adapter that implements the domain repository interface using SQLAlchemy.
    Follows Dependency Inversion Principle by implementing the abstract interface.
    """
    
    def __init__(self, session: AsyncSession):
        """
        Initialize repository with database session.
        
        Args:
            session: Database session for operations
        """
        self._session = session
    
    async def save_session(self, chat_session: ChatSession) -> ChatSession:
        """
        Save a chat session to the repository.
        
        Args:
            chat_session: Chat session entity to save
            
        Returns:
            Saved chat session entity
            
        Raises:
            RepositoryError: If save operation fails
        """
        try:
            # Convert domain entity to database model
            session_model = self._session_domain_to_model(chat_session)
            
            # Add to session and flush
            self._session.add(session_model)
            await self._session.flush()
            
            # Convert back to domain entity
            return await self._session_model_to_domain(session_model)
            
        except Exception as e:
            raise RepositoryError(f"Failed to save chat session: {str(e)}")
    
    async def create_session(self, chat_session: ChatSession) -> ChatSession:
        """
        Create a new chat session in the repository (alias for save_session).
        
        Args:
            chat_session: Chat session entity to create
            
        Returns:
            Created chat session entity
            
        Raises:
            RepositoryError: If create operation fails
        """
        return await self.save_session(chat_session)
    
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
        try:
            stmt = (
                select(ChatSessionModel)
                .options(selectinload(ChatSessionModel.messages).selectinload(MessageModel.citations))
                .where(ChatSessionModel.id == session_id.value)
            )
            
            result = await self._session.execute(stmt)
            session_model = result.scalar_one_or_none()
            
            if session_model is None:
                return None
            
            return await self._session_model_to_domain(session_model)
            
        except Exception as e:
            raise RepositoryError(f"Failed to find chat session by ID: {str(e)}")
    
    async def get_session_by_id(self, session_id: SessionId) -> Optional[ChatSession]:
        """
        Get a chat session by its ID (alias for find_session_by_id).
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Chat session entity if found, None otherwise
            
        Raises:
            RepositoryError: If query operation fails
        """
        return await self.find_session_by_id(session_id)
    
    async def find_sessions_by_document(self, document_id: DocumentId) -> List[ChatSession]:
        """
        Find all chat sessions for a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            List of chat sessions for the document
            
        Raises:
            RepositoryError: If query operation fails
        """
        try:
            stmt = (
                select(ChatSessionModel)
                .options(selectinload(ChatSessionModel.messages).selectinload(MessageModel.citations))
                .where(ChatSessionModel.document_id == document_id.value)
                .order_by(ChatSessionModel.last_activity_at.desc())
            )
            
            result = await self._session.execute(stmt)
            session_models = result.scalars().all()
            
            sessions = []
            for model in session_models:
                domain_session = await self._session_model_to_domain(model)
                sessions.append(domain_session)
            
            return sessions
            
        except Exception as e:
            raise RepositoryError(f"Failed to find sessions by document: {str(e)}")
    
    async def update_session(self, chat_session: ChatSession) -> ChatSession:
        """
        Update an existing chat session.
        
        Args:
            chat_session: Chat session entity with updated data
            
        Returns:
            Updated chat session entity
            
        Raises:
            RepositoryError: If update operation fails
            ChatSessionNotFoundError: If session doesn't exist
        """
        try:
            # Check if session exists
            existing = await self.find_session_by_id(chat_session.session_id)
            if existing is None:
                raise ChatSessionNotFoundError(chat_session.session_id.value)
            
            # Update the existing model
            session_model = self._session_domain_to_model(chat_session)
            
            # Merge changes
            await self._session.merge(session_model)
            await self._session.flush()
            
            # Return updated domain entity
            return await self._session_model_to_domain(session_model)
            
        except ChatSessionNotFoundError:
            raise
        except Exception as e:
            raise RepositoryError(f"Failed to update chat session: {str(e)}")
    
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
        try:
            stmt = delete(ChatSessionModel).where(ChatSessionModel.id == session_id.value)
            result = await self._session.execute(stmt)
            
            return result.rowcount > 0
            
        except Exception as e:
            raise RepositoryError(f"Failed to delete chat session: {str(e)}")
    
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
        try:
            # Convert domain entity to database model
            message_model = self._message_domain_to_model(message)
            
            # Add to session and flush
            self._session.add(message_model)
            await self._session.flush()
            
            # Return the original message with updated database fields to avoid lazy loading issues
            # Create a new message with the same data but updated timestamps from the database
            return Message(
                message_id=message_model.id,
                session_id=message.session_id,
                role=message.role,
                content=message.content,
                status=message.status,
                created_at=message_model.created_at,
                updated_at=message_model.updated_at,
                citations=message.citations,  # Use original citations to avoid lazy loading
                token_count=message.token_count,
                processing_time_ms=message.processing_time_ms,
                error_message=message.error_message,
            )
            
        except Exception as e:
            raise RepositoryError(f"Failed to save message: {str(e)}")
    
    async def add_message(self, message: Message) -> Message:
        """
        Add a message to the repository (alias for save_message).
        
        Args:
            message: Message entity to add
            
        Returns:
            Added message entity
            
        Raises:
            RepositoryError: If add operation fails
        """
        return await self.save_message(message)
    
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
        try:
            stmt = (
                select(MessageModel)
                .options(selectinload(MessageModel.citations))
                .where(MessageModel.id == message_id)
            )
            
            result = await self._session.execute(stmt)
            message_model = result.scalar_one_or_none()
            
            if message_model is None:
                return None
            
            return await self._message_model_to_domain(message_model)
            
        except Exception as e:
            raise RepositoryError(f"Failed to find message by ID: {str(e)}")
    
    async def find_messages_by_session(
        self,
        session_id: SessionId,
        limit: Optional[int] = None
    ) -> List[Message]:
        """
        Find messages for a chat session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to return
            
        Returns:
            List of messages for the session
            
        Raises:
            RepositoryError: If query operation fails
        """
        try:
            stmt = (
                select(MessageModel)
                .options(selectinload(MessageModel.citations))
                .where(MessageModel.session_id == session_id.value)
                .order_by(MessageModel.created_at.asc())
            )
            
            if limit:
                stmt = stmt.limit(limit)
            
            result = await self._session.execute(stmt)
            message_models = result.scalars().all()
            
            messages = []
            for model in message_models:
                domain_message = await self._message_model_to_domain(model)
                messages.append(domain_message)
            
            return messages
            
        except Exception as e:
            raise RepositoryError(f"Failed to find messages by session: {str(e)}")
    
    async def get_messages_by_session(self, session_id: SessionId, limit: Optional[int] = None) -> List[Message]:
        """
        Get messages for a chat session (alias for find_messages_by_session).
        
        Args:
            session_id: Session identifier
            limit: Optional limit on number of messages
            
        Returns:
            List of messages for the session
            
        Raises:
            RepositoryError: If query operation fails
        """
        return await self.find_messages_by_session(session_id, limit)
    
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
        try:
            # Check if message exists
            existing = await self.find_message_by_id(message.message_id)
            if existing is None:
                raise MessageNotFoundError(message.message_id)
            
            # Update the existing model
            message_model = self._message_domain_to_model(message)
            
            # Merge changes
            await self._session.merge(message_model)
            await self._session.flush()
            
            # Return updated domain entity
            return await self._message_model_to_domain(message_model)
            
        except MessageNotFoundError:
            raise
        except Exception as e:
            raise RepositoryError(f"Failed to update message: {str(e)}")
    
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
        try:
            stmt = delete(MessageModel).where(MessageModel.id == message_id)
            result = await self._session.execute(stmt)
            
            return result.rowcount > 0
            
        except Exception as e:
            raise RepositoryError(f"Failed to delete message: {str(e)}")
    
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
        try:
            stmt = (
                select(func.count(MessageModel.id))
                .where(MessageModel.session_id == session_id.value)
            )
            
            result = await self._session.execute(stmt)
            count = result.scalar_one()
            
            return count or 0
            
        except Exception as e:
            raise RepositoryError(f"Failed to get session message count: {str(e)}")
    
    def _session_domain_to_model(self, chat_session: ChatSession) -> ChatSessionModel:
        """
        Convert domain ChatSession entity to database model.
        
        Args:
            chat_session: Domain chat session entity
            
        Returns:
            Database chat session model
        """
        return ChatSessionModel(
            id=chat_session.session_id.value,
            document_id=chat_session.document_id.value if chat_session.document_id else None,
            status=chat_session.status,
            title=chat_session.title,
            message_count=chat_session.message_count,
            last_activity_at=chat_session.last_activity_at,
            created_at=chat_session.created_at,
            updated_at=chat_session.updated_at,
        )
    
    async def _session_model_to_domain(self, model: ChatSessionModel) -> ChatSession:
        """
        Convert database model to domain ChatSession entity.
        
        Args:
            model: Database chat session model
            
        Returns:
            Domain chat session entity
        """
        # Create chat session entity
        chat_session = ChatSession(
            session_id=SessionId(model.id),
            document_id=DocumentId(model.document_id) if model.document_id else None,
            status=model.status,
            created_at=model.created_at,
            updated_at=model.updated_at,
            title=model.title,
            message_count=model.message_count,
            last_activity_at=model.last_activity_at,
        )
        
        # Note: Messages are handled separately in this implementation
        
        return chat_session
    
    def _message_domain_to_model(self, message: Message) -> MessageModel:
        """
        Convert domain Message entity to database model.
        
        Args:
            message: Domain message entity
            
        Returns:
            Database message model
        """
        model = MessageModel(
            id=message.message_id,
            session_id=message.session_id.value,
            role=message.role,
            content=message.content,
            status=message.status,
            token_count=message.token_count,
            processing_time_ms=message.processing_time_ms,
            error_message=message.error_message,
            created_at=message.created_at,
            updated_at=message.updated_at,
        )
        
        # Convert citations
        for citation in message.citations:
            citation_model = CitationModel(
                id=citation.citation_id,
                message_id=message.message_id,
                page_number=citation.page_number.value,
                text_snippet=citation.text_snippet,
                start_position=citation.start_position,
                end_position=citation.end_position,
                confidence_score=citation.confidence_score,
            )
            model.citations.append(citation_model)
        
        return model
    
    async def _message_model_to_domain(self, model: MessageModel) -> Message:
        """
        Convert database model to domain Message entity.
        
        Args:
            model: Database message model
            
        Returns:
            Domain message entity
        """
        # Convert citations first - handle empty citations gracefully
        citations = []
        # Access citations safely, handling lazy loading
        citation_models = model.citations if hasattr(model, 'citations') and model.citations is not None else []
        for citation_model in citation_models:
            citation = Citation(
                citation_id=citation_model.id,
                page_number=PageNumber(citation_model.page_number),
                text_snippet=citation_model.text_snippet,
                start_position=citation_model.start_position,
                end_position=citation_model.end_position,
                confidence_score=citation_model.confidence_score,
            )
            citations.append(citation)
        
        # Create message entity with citations
        message = Message(
            message_id=model.id,
            session_id=SessionId(model.session_id),
            role=model.role,
            content=model.content,
            status=model.status,
            created_at=model.created_at,
            updated_at=model.updated_at,
            citations=citations,
            token_count=model.token_count,
            processing_time_ms=model.processing_time_ms,
            error_message=model.error_message,
        )
        
        return message
