"""
Chat API Endpoints

"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import StreamingResponse
import json
import logging
from datetime import datetime

from ..schemas.chat_schemas import (
    StartChatSessionRequest,
    StartChatSessionResponse,
    SendMessageRequest,
    SendMessageResponse,
    ChatHistoryRequest,
    ChatHistoryResponse,
    ChatSessionSummaryResponse,
    StreamingMessageChunk,
    ErrorResponse
)
from ....core.application.use_cases.chat_use_case import (
    ChatUseCase,
    StartChatRequest,
    SendMessageRequest as UseCaseSendMessageRequest,
    ChatHistoryRequest as UseCaseChatHistoryRequest
)
from src.core.domain.value_objects import SessionId, DocumentId
from ....infrastructure.di.dependencies import get_chat_use_case
from src.shared.exceptions import (
    ChatSessionNotFoundError,
    MessageGenerationError,
    ValidationError
)


router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)


@router.post(
    "/sessions",
    response_model=StartChatSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start a new chat session",
    description="Create a new chat session with optional document context and initial message"
)
async def start_chat_session(
    request: StartChatSessionRequest,
    chat_use_case: ChatUseCase = Depends(get_chat_use_case)
) -> StartChatSessionResponse:
    """
    Start a new chat session.
    
    Args:
        request: Chat session start request
        chat_use_case: Chat use case dependency
        
    Returns:
        Chat session start response
        
    Raises:
        HTTPException: If session creation fails
    """
    try:
        logger.info(f"Starting chat session: {request.session_id}")
        
        # Convert request to use case format
        use_case_request = StartChatRequest(
            session_id=SessionId(request.session_id),
            document_ids=[DocumentId(doc_id) for doc_id in request.document_ids] if request.document_ids else None,
            initial_message=request.initial_message,
            context=request.context
        )
        
        # Execute use case
        response = await chat_use_case.start_chat_session(use_case_request)
        
        if not response.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=response.error_message or "Failed to start chat session"
            )
        
        return StartChatSessionResponse(
            session_id=response.session_id,
            success=response.success,
            created_at=response.created_at,
            document_count=response.document_count,
            initial_response=response.initial_response,
            message="Chat session started successfully"
        )
        
    except ValidationError as e:
        logger.error(f"Validation error starting chat session: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error starting chat session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error starting chat session"
        )


@router.post(
    "/sessions/{session_id}/messages",
    response_model=SendMessageResponse,
    summary="Send a message to chat session",
    description="Send a message to an existing chat session and get AI response with citations"
)
async def send_message(
    session_id: str,
    request: SendMessageRequest,
    chat_use_case: ChatUseCase = Depends(get_chat_use_case)
) -> SendMessageResponse:
    """
    Send a message to a chat session.
    
    Args:
        session_id: Chat session identifier
        request: Message send request
        chat_use_case: Chat use case dependency
        
    Returns:
        Message send response with AI response and citations
        
    Raises:
        HTTPException: If message sending fails
    """
    try:
        logger.info(f"Sending message to session {session_id}")
        
        # Convert request to use case format
        use_case_request = UseCaseSendMessageRequest(
            session_id=SessionId(session_id),
            message=request.message,
            document_ids=[DocumentId(doc_id) for doc_id in request.document_ids] if request.document_ids else None,
            model=request.model,
            temperature=request.temperature,
            stream=False
        )
        
        # Execute use case
        response = await chat_use_case.send_message(use_case_request)
        
        return SendMessageResponse(
            session_id=response.session_id,
            message_id=response.message_id,
            user_message=request.message,
            assistant_response=response.response,
            citations=response.citations,
            context_used=response.context_used,
            processing_time_ms=response.processing_time_ms,
            model_used=response.model_used,
            usage=response.usage,
            metadata=response.metadata,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except ChatSessionNotFoundError as e:
        logger.error(f"Chat session not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session {session_id} not found"
        )
    except ValidationError as e:
        logger.error(f"Validation error sending message: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except MessageGenerationError as e:
        logger.error(f"Message generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate response"
        )
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error sending message"
        )


@router.post(
    "/sessions/{session_id}/messages/stream",
    summary="Send a message with streaming response",
    description="Send a message to chat session and get streaming AI response"
)
async def send_streaming_message(
    session_id: str,
    request: SendMessageRequest,
    chat_use_case: ChatUseCase = Depends(get_chat_use_case)
) -> StreamingResponse:
    """
    Send a message to a chat session with streaming response.
    
    Args:
        session_id: Chat session identifier
        request: Message send request
        chat_use_case: Chat use case dependency
        
    Returns:
        Streaming response with AI response chunks
        
    Raises:
        HTTPException: If message sending fails
    """
    try:
        logger.info(f"Sending streaming message to session {session_id}")
        
        # Convert request to use case format
        use_case_request = UseCaseSendMessageRequest(
            session_id=SessionId(session_id),
            message=request.message,
            document_ids=[DocumentId(doc_id) for doc_id in request.document_ids] if request.document_ids else None,
            model=request.model,
            temperature=request.temperature,
            stream=True
        )
        
        async def generate_stream():
            """Generate streaming response."""
            try:
                async for chunk in chat_use_case.send_streaming_message(use_case_request):
                    chunk_data = StreamingMessageChunk(
                        content=chunk.content,
                        is_complete=chunk.is_complete,
                        chunk_index=chunk.chunk_index,
                        session_id=session_id,
                        timestamp=datetime.utcnow().isoformat(),
                        metadata=chunk.metadata
                    )
                    
                    # Send as Server-Sent Events format
                    yield f"data: {chunk_data.model_dump_json()}\n\n"
                
                # Send completion signal
                completion_data = StreamingMessageChunk(
                    content="",
                    is_complete=True,
                    chunk_index=-1,
                    session_id=session_id,
                    timestamp=datetime.utcnow().isoformat(),
                    metadata={"stream_completed": True}
                )
                yield f"data: {completion_data.model_dump_json()}\n\n"
                
            except Exception as e:
                logger.error(f"Error in streaming response: {e}")
                error_data = {
                    "error": str(e),
                    "session_id": session_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
                yield f"data: {json.dumps(error_data)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*"
            }
        )
        
    except ChatSessionNotFoundError as e:
        logger.error(f"Chat session not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session {session_id} not found"
        )
    except ValidationError as e:
        logger.error(f"Validation error sending streaming message: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error sending streaming message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error sending streaming message"
        )


@router.get(
    "/sessions/{session_id}/history",
    response_model=ChatHistoryResponse,
    summary="Get chat session history",
    description="Retrieve message history for a chat session with pagination"
)
async def get_chat_history(
    session_id: str,
    limit: Optional[int] = 50,
    offset: int = 0,
    include_context: bool = False,
    chat_use_case: ChatUseCase = Depends(get_chat_use_case)
) -> ChatHistoryResponse:
    """
    Get chat history for a session.
    
    Args:
        session_id: Chat session identifier
        limit: Maximum number of messages to return
        offset: Number of messages to skip
        include_context: Whether to include context and citations
        chat_use_case: Chat use case dependency
        
    Returns:
        Chat history response
        
    Raises:
        HTTPException: If history retrieval fails
    """
    try:
        logger.info(f"Getting chat history for session {session_id}")
        
        # Convert request to use case format
        use_case_request = UseCaseChatHistoryRequest(
            session_id=SessionId(session_id),
            limit=limit,
            offset=offset,
            include_context=include_context
        )
        
        # Execute use case
        response = await chat_use_case.get_chat_history(use_case_request)
        
        return ChatHistoryResponse(
            session_id=response.session_id,
            messages=response.messages,
            total_messages=response.total_messages,
            has_more=response.has_more,
            session_info=response.session_info,
            pagination={
                "limit": limit,
                "offset": offset,
                "returned_count": len(response.messages)
            }
        )
        
    except ChatSessionNotFoundError as e:
        logger.error(f"Chat session not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session {session_id} not found"
        )
    except ValidationError as e:
        logger.error(f"Validation error getting chat history: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error getting chat history"
        )


@router.get(
    "/sessions/{session_id}/messages",
    response_model=List[Dict[str, Any]],
    summary="Get chat session messages",
    description="Retrieve all messages for a chat session"
)
async def get_chat_messages(
    session_id: str,
    chat_use_case: ChatUseCase = Depends(get_chat_use_case)
) -> List[Dict[str, Any]]:
    """
    Get all messages for a chat session.
    
    Args:
        session_id: Chat session identifier
        chat_use_case: Chat use case dependency
        
    Returns:
        List of chat messages
        
    Raises:
        HTTPException: If message retrieval fails
    """
    try:
        logger.info(f"Getting messages for session {session_id}")
        
        # Use the chat repository from the use case to get messages
        messages = await chat_use_case._chat_repository.get_messages_by_session(
            SessionId(session_id), 
            limit=None
        )
        
        # Convert messages to dictionary format for JSON response
        message_dicts = []
        for message in messages:
            message_dict = {
                "message_id": message.message_id,
                "session_id": message.session_id.value,
                "content": message.content,
                "role": message.role.value,
                "timestamp": message.created_at.isoformat(),
                "citations": []
            }
            message_dicts.append(message_dict)
        
        return message_dicts
        
    except ChatSessionNotFoundError as e:
        logger.error(f"Chat session not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session {session_id} not found"
        )
    except Exception as e:
        logger.error(f"Error getting chat messages: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error getting chat messages"
        )


@router.get(
    "/sessions/{session_id}/summary",
    response_model=ChatSessionSummaryResponse,
    summary="Get chat session summary",
    description="Get summary and statistics for a chat session"
)
async def get_session_summary(
    session_id: str,
    chat_use_case: ChatUseCase = Depends(get_chat_use_case)
) -> ChatSessionSummaryResponse:
    """
    Get summary and statistics for a chat session.
    
    Args:
        session_id: Chat session identifier
        chat_use_case: Chat use case dependency
        
    Returns:
        Chat session summary
        
    Raises:
        HTTPException: If summary retrieval fails
    """
    try:
        logger.info(f"Getting summary for session {session_id}")
        
        # Execute use case
        summary = await chat_use_case.get_session_summary(SessionId(session_id))
        
        return ChatSessionSummaryResponse(
            session_id=summary["session_id"],
            created_at=summary["created_at"],
            updated_at=summary["updated_at"],
            document_count=summary["document_count"],
            total_messages=summary["total_messages"],
            user_messages=summary["user_messages"],
            assistant_messages=summary["assistant_messages"],
            total_citations=summary["total_citations"],
            duration_seconds=summary["duration_seconds"],
            average_response_time=summary["average_response_time"],
            metadata=summary["metadata"]
        )
        
    except ChatSessionNotFoundError as e:
        logger.error(f"Chat session not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session {session_id} not found"
        )
    except Exception as e:
        logger.error(f"Error getting session summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error getting session summary"
        )


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete chat session",
    description="Delete a chat session and all its messages"
)
async def delete_chat_session(
    session_id: str,
    chat_use_case: ChatUseCase = Depends(get_chat_use_case)
) -> None:
    """
    Delete a chat session and all its messages.
    
    Args:
        session_id: Chat session identifier
        chat_use_case: Chat use case dependency
        
    Raises:
        HTTPException: If deletion fails
    """
    try:
        logger.info(f"Deleting chat session {session_id}")
        
        # Execute use case
        success = await chat_use_case.delete_chat_session(SessionId(session_id))
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete chat session"
            )
        
        logger.info(f"Successfully deleted chat session {session_id}")
        
    except ChatSessionNotFoundError as e:
        logger.error(f"Chat session not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session {session_id} not found"
        )
    except Exception as e:
        logger.error(f"Error deleting chat session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error deleting chat session"
        )


@router.get(
    "/health",
    summary="Chat service health check",
    description="Check the health of chat services"
)
async def chat_health_check(
    chat_use_case: ChatUseCase = Depends(get_chat_use_case)
) -> Dict[str, Any]:
    """
    Check the health of chat services.
    
    Args:
        chat_use_case: Chat use case dependency
        
    Returns:
        Health check results
    """
    try:
        # This would ideally check the health of underlying services
        # For now, just return basic status
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "chat_use_case": "healthy",
                "chat_service": "healthy",
                "rag_service": "healthy"
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat services are unhealthy"
        )
