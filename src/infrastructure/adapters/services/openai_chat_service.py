"""
OpenAI Chat Service Implementation
"""
from typing import List, Optional, Dict, Any, AsyncIterator
import asyncio
import logging
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, AsyncIterator

try:
    import openai
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from src.core.domain.services.chat_service import (
    ChatService,
    RAGService,
    ChatMessage,
    ChatResponse,
    ChatContext,
    StreamingChatChunk
)
from src.core.domain.value_objects import DocumentId
from ....core.application.use_cases.similarity_search_use_case import (
    SimilaritySearchUseCase,
    SimilaritySearchRequest
)
from src.shared.exceptions import (
    MessageGenerationError,
    CitationError,
    ConfigurationError
)
from src.shared.constants import AIConstants


class OpenAIChatService(ChatService):
    """
    OpenAI implementation of ChatService following Single Responsibility Principle.
    
    Handles chat message generation using OpenAI's chat models.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        max_retries: int = 3,
        timeout: float = 30.0
    ):
        """
        Initialize OpenAI chat service.
        
        Args:
            api_key: OpenAI API key
            model: Default chat model
            max_retries: Maximum number of retries
            timeout: Request timeout in seconds
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available")
        
        if not api_key:
            raise ConfigurationError("OpenAI API key is required")
        
        self._client = AsyncOpenAI(
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout
        )
        self._default_model = model
        self._logger = logging.getLogger(__name__)
        
        # Model configurations
        self._model_configs = {
            "gpt-4": {"max_tokens": 4096, "context_window": 8192},
            "gpt-4-turbo": {"max_tokens": 4096, "context_window": 128000},
            "gpt-3.5-turbo": {"max_tokens": 4096, "context_window": 16385}
        }
    
    async def generate_response(
        self,
        messages: List[ChatMessage],
        context: Optional[ChatContext] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> ChatResponse:
        """
        Generate a chat response using OpenAI.
        
        Args:
            messages: Conversation history
            context: Optional context from document retrieval
            model: Optional model name to use
            temperature: Response creativity (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            
        Returns:
            Chat response with citations
            
        Raises:
            MessageGenerationError: If response generation fails
        """
        start_time = datetime.utcnow()
        
        try:
            model_name = model or self._default_model
            
            # Validate model
            if model_name not in self._model_configs:
                raise MessageGenerationError(f"Unsupported model: {model_name}")
            
            # Prepare messages for OpenAI
            openai_messages = self._prepare_messages(messages, context)
            
            # Set max_tokens if not provided
            if max_tokens is None:
                max_tokens = self._model_configs[model_name]["max_tokens"]
            
            # Generate response
            response = await self._client.chat.completions.create(
                model=model_name,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            
            # Extract response content
            response_content = response.choices[0].message.content
            
            # Extract citations if context was provided
            citations = []
            if context:
                citations = await self._extract_citations_from_response(
                    response_content, context
                )
            
            # Calculate processing time
            end_time = datetime.utcnow()
            processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            return ChatResponse(
                message=response_content,
                citations=citations,
                context_used=context,
                model=model_name,
                usage=response.usage.model_dump() if response.usage else None,
                processing_time_ms=processing_time_ms,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "created_at": datetime.utcnow().isoformat()
                }
            )
            
        except openai.APIError as e:
            self._logger.error(f"OpenAI API error: {e}")
            raise MessageGenerationError(f"OpenAI API error: {str(e)}")
        except Exception as e:
            self._logger.error(f"Chat response generation failed: {e}")
            raise MessageGenerationError(f"Chat response generation failed: {str(e)}")
    
    async def generate_streaming_response(
        self,
        messages: List[ChatMessage],
        context: Optional[ChatContext] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> AsyncIterator[StreamingChatChunk]:
        """
        Generate a streaming chat response using OpenAI.
        
        Args:
            messages: Conversation history
            context: Optional context from document retrieval
            model: Optional model name to use
            temperature: Response creativity (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            
        Yields:
            Streaming chat chunks
            
        Raises:
            MessageGenerationError: If response generation fails
        """
        try:
            model_name = model or self._default_model
            
            # Validate model
            if model_name not in self._model_configs:
                raise MessageGenerationError(f"Unsupported model: {model_name}")
            
            # Prepare messages for OpenAI
            openai_messages = self._prepare_messages(messages, context)
            
            # Set max_tokens if not provided
            if max_tokens is None:
                max_tokens = self._model_configs[model_name]["max_tokens"]
            
            # Generate streaming response
            stream = await self._client.chat.completions.create(
                model=model_name,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            chunk_index = 0
            full_content = ""
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_content += content
                    
                    yield StreamingChatChunk(
                        content=content,
                        is_complete=False,
                        chunk_index=chunk_index,
                        metadata={
                            "model": model_name,
                            "chunk_id": chunk.id if hasattr(chunk, 'id') else None
                        }
                    )
                    chunk_index += 1
            
            # Send final chunk with completion status
            yield StreamingChatChunk(
                content="",
                is_complete=True,
                chunk_index=chunk_index,
                metadata={
                    "model": model_name,
                    "full_content": full_content,
                    "total_chunks": chunk_index
                }
            )
            
        except openai.APIError as e:
            self._logger.error(f"OpenAI streaming API error: {e}")
            raise MessageGenerationError(f"OpenAI streaming API error: {str(e)}")
        except Exception as e:
            self._logger.error(f"Streaming response generation failed: {e}")
            raise MessageGenerationError(f"Streaming response generation failed: {str(e)}")
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available chat models.
        
        Returns:
            List of model names
        """
        return list(self._model_configs.keys())
    
    async def health_check(self) -> bool:
        """
        Check if the OpenAI chat service is healthy.
        
        Returns:
            True if service is healthy
        """
        try:
            # Test with a simple chat request
            test_messages = [
                {"role": "user", "content": "Hello"}
            ]
            
            response = await self._client.chat.completions.create(
                model=self._default_model,
                messages=test_messages,
                max_tokens=10
            )
            
            return len(response.choices) > 0 and response.choices[0].message.content
            
        except Exception as e:
            self._logger.error(f"Health check failed: {e}")
            return False
    
    def _prepare_messages(
        self,
        messages: List[ChatMessage],
        context: Optional[ChatContext] = None
    ) -> List[Dict[str, str]]:
        """
        Prepare messages for OpenAI API format.
        
        Args:
            messages: Chat messages
            context: Optional context
            
        Returns:
            List of messages in OpenAI format
        """
        openai_messages = []
        
        # Add system message with context if provided
        if context:
            system_content = self._build_system_message_with_context(context)
            openai_messages.append({
                "role": "system",
                "content": system_content
            })
        
        # Convert chat messages to OpenAI format
        for message in messages:
            openai_messages.append({
                "role": message.role.value if hasattr(message.role, 'value') else message.role,  # Convert enum to string
                "content": message.content
            })
        
        return openai_messages
    
    def _build_system_message_with_context(self, context: ChatContext) -> str:
        """
        Build system message with context information.
        
        Args:
            context: Chat context
            
        Returns:
            System message content
        """
        system_message = """You are a helpful AI assistant that answers questions based on provided document content. 

IMPORTANT INSTRUCTIONS:
1. Use ONLY the information provided in the context below to answer questions
2. If the context doesn't contain enough information to answer a question, say so clearly
3. When referencing information, include citations in the format [Source: Document Title, Page X]
4. Be accurate and precise in your responses
5. If asked about something not in the context, politely explain that you can only answer based on the provided documents

CONTEXT INFORMATION:
"""
        
        # Add relevant chunks to context
        for i, chunk in enumerate(context.relevant_chunks):
            chunk_info = f"""
Document: {chunk.get('document_title', 'Unknown')}
Page: {chunk.get('page_number', 'Unknown')}
Content: {chunk.get('text_content', '')}
---"""
            system_message += chunk_info
        
        system_message += f"""

Total relevant chunks found: {context.total_chunks}
Search query used: "{context.search_query}"

Please provide helpful and accurate responses based on this context."""
        
        return system_message
    
    async def _extract_citations_from_response(
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
            List of citations
        """
        citations = []
        
        try:
            # Simple citation extraction based on context chunks
            for i, chunk in enumerate(context.relevant_chunks):
                # Check if chunk content is referenced in response
                chunk_text = chunk.get('text_content', '')
                
                # Simple heuristic: if significant overlap exists
                if self._has_content_overlap(response_text, chunk_text):
                    citation = {
                        "id": f"citation_{i}",
                        "document_id": chunk.get('document_id'),
                        "document_title": chunk.get('document_title'),
                        "page_number": chunk.get('page_number'),
                        "chunk_id": chunk.get('chunk_id'),
                        "text_snippet": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                        "similarity_score": context.similarity_scores[i] if i < len(context.similarity_scores) else 0.0,
                        "citation_type": "reference"
                    }
                    citations.append(citation)
            
            return citations
            
        except Exception as e:
            self._logger.error(f"Citation extraction failed: {e}")
            return []
    
    def _has_content_overlap(self, response_text: str, chunk_text: str, min_words: int = 3) -> bool:
        """
        Check if response has significant overlap with chunk content.
        
        Args:
            response_text: Response text
            chunk_text: Chunk text
            min_words: Minimum word overlap
            
        Returns:
            True if significant overlap exists
        """
        try:
            # Simple word-based overlap detection
            response_words = set(response_text.lower().split())
            chunk_words = set(chunk_text.lower().split())
            
            # Remove common words
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
            
            response_words -= common_words
            chunk_words -= common_words
            
            overlap = response_words.intersection(chunk_words)
            return len(overlap) >= min_words
            
        except Exception:
            return False


class OpenAIRAGService(RAGService):
    """
    OpenAI implementation of RAGService with universal document response synthesis.
    
    Handles retrieval-augmented generation using OpenAI and similarity search,
    with support for multiple document types and enhanced response synthesis.
    """
    
    # Document type detection patterns
    DOCUMENT_TYPE_PATTERNS = {
        'research_paper': [
            r'abstract', r'doi:', r'\b(?:et al\.|fig\.|table|section|references)\b',
            r'\b(?:introduction|methodology|results|discussion|conclusion)\b'
        ],
        'legal_document': [
            r'\b(section|article|clause|subsection|paragraph)\b',
            r'\b(whereas|hereby|herein|hereinafter|notwithstanding|pursuant to)\b',
            r'\b(agreement|contract|license|amendment|exhibit|schedule)\b'
        ],
        'technical_manual': [
            r'\b(chapter|section|figure|table|note|warning|caution|important)\b',
            r'\b(install|configure|usage|troubleshooting|specifications|requirements)\b',
            r'\b(step \d+|procedure|prerequisites|troubleshooting|faq)\b'
        ],
        'book': [
            r'\b(chapter|part|volume|edition|preface|foreword|introduction|epilogue)\b',
            r'\b(bibliography|index|glossary|appendix|footnote|endnote|citation)\b',
            r'\b(published by|copyright|all rights reserved|isbn|issn)\b'
        ]
    }
    
    def __init__(
        self,
        chat_service: OpenAIChatService,
        similarity_search_use_case: SimilaritySearchUseCase,
        advanced_search_use_case=None,
        citation_use_case=None,
        default_document_type: str = 'general'
    ):
        """
        Initialize OpenAI RAG service with universal document support.
        
        Args:
            chat_service: OpenAI chat service
            similarity_search_use_case: Similarity search use case
            advanced_search_use_case: Optional advanced search use case for enhanced retrieval
            citation_use_case: Optional citation use case for response citations
            default_document_type: Default document type if auto-detection fails
        """
        self._chat_service = chat_service
        self._similarity_search_use_case = similarity_search_use_case
        self._advanced_search_use_case = advanced_search_use_case
        self._citation_use_case = citation_use_case
        self._default_document_type = default_document_type
        self._logger = logging.getLogger(__name__)
    
    def _detect_document_type(self, text: str) -> str:
        """
        Detect document type based on text patterns.
        
        Args:
            text: Document text to analyze
            
        Returns:
            Detected document type (e.g., 'research_paper', 'legal_document')
        """
        if not text:
            return self._default_document_type
            
        text_lower = text.lower()
        
        # Check for document type patterns
        for doc_type, patterns in self.DOCUMENT_TYPE_PATTERNS.items():
            if any(re.search(pattern, text_lower) for pattern in patterns):
                self._logger.debug(f"Detected document type: {doc_type}")
                return doc_type
                
        return self._default_document_type
    
    def _prepare_context_for_document_type(self, context: ChatContext, doc_type: str) -> str:
        """
        Prepare context text based on document type.
        
        Args:
            context: Chat context with retrieved chunks
            doc_type: Detected document type
            
        Returns:
            Formatted context string for the language model
        """
        context_parts = []
        
        for i, chunk in enumerate(context.relevant_chunks, 1):
            # Common metadata
            source = chunk.get('source', 'Document')
            page = chunk.get('page_number', 'N/A')
            
            # Type-specific formatting
            if doc_type == 'research_paper':
                section = chunk.get('section', 'Content')
                context_parts.append(
                    f"[Source: {source}, Page {page}, Section: {section}]\n"
                    f"{chunk.get('text', '')}\n"
                    f"[End of excerpt {i}]\n"
                )
            elif doc_type == 'legal_document':
                section = chunk.get('section', 'Section')
                context_parts.append(
                    f"[{section} {chunk.get('section_number', '')}]\n"
                    f"{chunk.get('text', '')}\n"
                    f"[Source: {source}, Page {page}]\n"
                )
            elif doc_type == 'technical_manual':
                section = chunk.get('section', 'Content')
                context_parts.append(
                    f"### {section} {chunk.get('section_number', '')}\n"
                    f"{chunk.get('text', '')}\n"
                    f"*Source: {source}, Page {page}*\n\n"
                )
            else:  # General/default format
                context_parts.append(
                    f"--- Excerpt {i} ---\n"
                    f"Source: {source}, Page {page}\n"
                    f"{chunk.get('text', '')}\n\n"
                )
        
        return "\n".join(context_parts)
    
    async def retrieve_context(
        self,
        query: str,
        document_ids: Optional[List[DocumentId]] = None,
        max_chunks: int = 5,
        similarity_threshold: float = 0.7,
        document_text: Optional[str] = None
    ) -> ChatContext:
        """
        Retrieve relevant context for a query with document type awareness.
        
        Args:
            query: User query
            document_ids: Optional document filter
            max_chunks: Maximum chunks to retrieve
            similarity_threshold: Minimum similarity threshold
            document_text: Optional document text for type detection
            
        Returns:
            Chat context with relevant chunks and document type information
        """
        try:
            # Detect document type if text is provided
            doc_type = self._default_document_type
            if document_text:
                doc_type = self._detect_document_type(document_text)
                self._logger.info(f"Processing document as type: {doc_type}")
            
            # Use advanced search if available, otherwise fall back to similarity search
            if self._advanced_search_use_case:
                self._logger.info("Using advanced search for enhanced retrieval")
                
                # Create enhanced search request with document type awareness
                enhanced_request = type('EnhancedSearchRequest', (), {
                    'query': query,
                    'max_results': max_chunks,
                    'document_ids': document_ids,
                    'similarity_threshold': similarity_threshold,
                    'document_type': doc_type  # Pass document type for specialized search
                })()
                
                # Execute enhanced search
                enhanced_response = await self._advanced_search_use_case.enhanced_search(enhanced_request)
                
                # Convert enhanced search results to similarity search format for compatibility
                search_response = type('SearchResponse', (), {
                    'results': enhanced_response.results,
                    'total_results': enhanced_response.total_results,
                    'search_time_ms': enhanced_response.search_time_ms,
                    'document_type': doc_type  # Include document type in response
                })()
                
            else:
                self._logger.info("Using basic similarity search")
                search_request = SimilaritySearchRequest(
                    query=query,
                    document_ids=document_ids,
                    max_results=max_chunks,
                    similarity_threshold=similarity_threshold,
                    metadata_filters={"document_type": doc_type} if doc_type != self._default_document_type else None
                )
                search_response = await self._similarity_search_use_case.execute(search_request)
                
                # Add document type to response
                search_response.document_type = doc_type
                
            # Prepare context with document type awareness
            relevant_chunks = [result.to_dict() for result in search_response.results]
            
            # Add document type to each chunk's metadata
            for chunk in relevant_chunks:
                chunk['document_type'] = getattr(search_response, 'document_type', self._default_document_type)
            
            context = ChatContext(
                relevant_chunks=relevant_chunks,
                document_ids=document_ids or [],
                similarity_scores=[result.score for result in search_response.results],
                total_chunks=search_response.total_results,
                search_query=query,
                metadata={
                    'document_type': getattr(search_response, 'document_type', self._default_document_type),
                    'search_strategy': 'advanced' if self._advanced_search_use_case else 'basic'
                }
            )
            
            return context
        
        except Exception as e:
            self._logger.error(f"Context retrieval failed: {e}")
            # Return empty context on failure
            return ChatContext(
                relevant_chunks=[],
                document_ids=[],
                similarity_scores=[],
                total_chunks=0,
                search_query=query,
                metadata={"error": str(e)}
            )
    
    async def generate_rag_response(
        self,
        query: str,
        conversation_history: List[ChatMessage],
        document_ids: Optional[List[DocumentId]] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        document_text: Optional[str] = None
    ) -> ChatResponse:
        """
        Generate a RAG-enhanced response with document type awareness.
        
        Args:
            query: User query
            conversation_history: Previous messages
            document_ids: Optional document filter
            model: Optional model name
            temperature: Response creativity (0.0 to 1.0)
            document_text: Optional document text for type detection
            
        Returns:
            Chat response with citations and document type awareness
        """
        try:
            # Retrieve relevant context with document type awareness
            context = await self.retrieve_context(
                query=query,
                document_ids=document_ids,
                document_text=document_text
            )
            
            # Get document type from context or detect from text
            doc_type = (
                context.metadata.get('document_type') if context.metadata 
                else self._detect_document_type(document_text or '')
            )
            
            # Prepare context based on document type
            prepared_context = self._prepare_context_for_document_type(context, doc_type)
            
            # Prepare conversation history with document-type specific system message
            system_message = self._build_system_message_with_context(context)
            
            # Add document type specific instructions
            doc_type_instructions = {
                'research_paper': (
                    "You are a research assistant. Focus on providing accurate, citable information. "
                    "When possible, reference specific sections, figures, or tables from the research. "
                    "Maintain an academic tone and be precise with technical details."
                ),
                'legal_document': (
                    "You are a legal assistant. Be precise with legal terminology and citations. "
                    "Reference specific sections, clauses, or articles when possible. "
                    "Maintain a formal and neutral tone appropriate for legal contexts."
                ),
                'technical_manual': (
                    "You are a technical support specialist. Provide clear, step-by-step guidance. "
                    "Reference specific sections, figures, or steps when possible. "
                    "Be concise and focus on practical, actionable information."
                ),
                'book': (
                    "You are a knowledgeable guide. Provide context and insights from the text. "
                    "Reference specific chapters, sections, or pages when relevant. "
                    "Maintain an engaging and informative tone."
                )
            }
            
            # Add document type specific instructions if available
            if doc_type in doc_type_instructions:
                system_message += f"\n\n{doc_type_instructions[doc_type]}"
            
            messages = [
                ChatMessage(
                    role="system",
                    content=system_message,
                    timestamp=datetime.utcnow()
                )
            ]
            
            # Add user query to conversation history
            messages += conversation_history + [
                ChatMessage(
                    content=query,
                    role="user",
                    timestamp=datetime.utcnow()
                )
            ]
            
            # Generate response with document type awareness
            response = await self._chat_service.generate_response(
                messages=messages,
                context=prepared_context,
                model=model,
                temperature=temperature
            )
            
            # Enhance response with document type information
            if hasattr(response, 'metadata') and response.metadata:
                response.metadata['document_type'] = doc_type
            else:
                response.metadata = {'document_type': doc_type}
                
            # Format citations based on document type
            if hasattr(response, 'citations') and response.citations:
                response.citations = self._format_citations(response.citations, doc_type)
                
            # Enhance response with citations if citation use case is available
            if self._citation_use_case and context.relevant_chunks:
                try:
                    self._logger.info("Extracting citations for enhanced response")
                    
                    # Extract citations from the response
                    citation_response = await self._citation_use_case.extract_citations(
                        response_text=response.content,
                        source_chunks=context.relevant_chunks,
                    )
                    response.citations = citations
                except Exception as e:
                    self._logger.error(f"Citation extraction failed: {e}")
                    # Continue without citations rather than failing the entire response
            
            return response
            
        except Exception as e:
            self._logger.error(f"RAG response generation failed: {e}")
            raise MessageGenerationError(f"Failed to generate RAG response: {str(e)}")
        
    def _format_citations(self, citations: List[Dict[str, Any]], doc_type: str) -> List[Dict[str, Any]]:
        """
        Format citations based on document type.
        
        Args:
            citations: List of citation dictionaries
            doc_type: Document type
            
        Returns:
            Formatted citations with type-specific formatting
        """
        formatted_citations = []
        
        for citation in citations:
            formatted = citation.copy()
            
            # Common fields
            source = citation.get('source', 'Document')
            page = citation.get('page_number', 'N/A')
            
            # Type-specific formatting
            if doc_type == 'research_paper':
                formatted['display'] = f"{source}, p. {page}"
            elif doc_type == 'legal_document':
                section = citation.get('section', 'Section')
                section_num = citation.get('section_number', '')
                formatted['display'] = f"{section} {section_num}, {source}"
            elif doc_type == 'technical_manual':
                section = citation.get('section', 'Section')
                section_num = citation.get('section_number', '')
                formatted['display'] = f"{source}, {section} {section_num}, p. {page}"
            else:  # General/default format
                formatted['display'] = f"{source}, Page {page}"
                
            formatted_citations.append(formatted)
            
        return formatted_citations
    
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
        """
        try:
            # Use the chat service's citation extraction with document type awareness
            doc_type = context.metadata.get('document_type', self._default_document_type) if context.metadata else self._default_document_type
            citations = await self._chat_service._extract_citations_from_response(
                response_text, context
            )
            
            # Format citations based on document type
            return self._format_citations(citations, doc_type)
            
        except Exception as e:
            self._logger.error(f"Citation extraction failed: {e}")
            return []  # Return empty list if citation extraction fails


class OpenAIChatServiceFactory:
    """
    Factory for creating OpenAI chat service instances.
    
    Follows Open/Closed Principle for extensibility.
    """
    
    @staticmethod
    def create_chat_service(
        api_key: str,
        model: str = "gpt-3.5-turbo",
        max_retries: int = 3,
        timeout: float = 30.0
    ) -> OpenAIChatService:
        """
        Create OpenAI chat service instance.
        
        Args:
            api_key: OpenAI API key
            model: Default chat model
            max_retries: Maximum number of retries
            timeout: Request timeout in seconds
            
        Returns:
            Configured OpenAI chat service
        """
        return OpenAIChatService(
            api_key=api_key,
            model=model,
            max_retries=max_retries,
            timeout=timeout
        )
    
    @staticmethod
    def create_rag_service(
        chat_service: OpenAIChatService,
        similarity_search_use_case: SimilaritySearchUseCase
    ) -> OpenAIRAGService:
        """
        Create OpenAI RAG service instance.
        
        Args:
            chat_service: OpenAI chat service
            similarity_search_use_case: Similarity search use case
            
        Returns:
            Configured OpenAI RAG service
        """
        return OpenAIRAGService(
            chat_service=chat_service,
            similarity_search_use_case=similarity_search_use_case
        )
