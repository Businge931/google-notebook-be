"""
PDF Parsing Service

"""
from abc import ABC, abstractmethod
from typing import List, BinaryIO, Optional, Dict, Any
from dataclasses import dataclass
import io
import logging

try:
    import PyPDF2
    import pdfplumber
    PARSING_AVAILABLE = True
except ImportError:
    PARSING_AVAILABLE = False

from src.core.domain.entities import DocumentChunk
from src.core.domain.value_objects import PageNumber
from src.shared.exceptions import DocumentProcessingError
from src.shared.constants import ProcessingConstants


@dataclass
class ParsedPage:
    """
    Represents a parsed page from a PDF document.
    
    Follows Single Responsibility Principle by only containing page data.
    """
    page_number: int
    text_content: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ParsedDocument:
    """
    Represents a fully parsed PDF document.
    
    Follows Single Responsibility Principle by only containing document data.
    """
    pages: List[ParsedPage]
    total_pages: int
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def total_text_length(self) -> int:
        """Get total text length across all pages."""
        return sum(len(page.text_content) for page in self.pages)


class PDFParsingStrategy(ABC):
    """
    Abstract strategy for PDF parsing following Strategy Pattern.
    
    Enables different parsing approaches while maintaining Interface Segregation.
    """
    
    @abstractmethod
    async def parse_document(self, file_data: BinaryIO) -> ParsedDocument:
        """
        Parse PDF document and extract text.
        
        Args:
            file_data: Binary PDF file data
            
        Returns:
            Parsed document with pages and metadata
            
        Raises:
            DocumentProcessingError: If parsing fails
        """
        pass
    
    @abstractmethod
    def supports_file(self, mime_type: str) -> bool:
        """
        Check if this strategy supports the given file type.
        
        Args:
            mime_type: MIME type of the file
            
        Returns:
            True if strategy supports this file type
        """
        pass


class PyPDF2ParsingStrategy(PDFParsingStrategy):
    """
    PDF parsing strategy using PyPDF2 library.
    
    Follows Single Responsibility Principle by focusing on PyPDF2-specific parsing.
    """
    
    def __init__(self):
        """Initialize PyPDF2 parsing strategy."""
        if not PARSING_AVAILABLE:
            raise ImportError("PyPDF2 library not available")
        
        self._logger = logging.getLogger(__name__)
    
    async def parse_document(self, file_data: BinaryIO) -> ParsedDocument:
        """
        Parse PDF using PyPDF2.
        
        Args:
            file_data: Binary PDF file data
            
        Returns:
            Parsed document
            
        Raises:
            DocumentProcessingError: If parsing fails
        """
        try:
            # Reset file pointer
            file_data.seek(0)
            
            # Create PDF reader
            pdf_reader = PyPDF2.PdfReader(file_data)
            
            # Extract metadata
            metadata = {
                'parser': 'PyPDF2',
                'encrypted': pdf_reader.is_encrypted,
                'pages': len(pdf_reader.pages)
            }
            
            # Add document info if available
            if pdf_reader.metadata:
                metadata.update({
                    'title': pdf_reader.metadata.get('/Title', ''),
                    'author': pdf_reader.metadata.get('/Author', ''),
                    'subject': pdf_reader.metadata.get('/Subject', ''),
                    'creator': pdf_reader.metadata.get('/Creator', ''),
                })
            
            # Extract text from each page
            pages = []
            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    text_content = page.extract_text()
                    
                    # Clean and validate text
                    text_content = self._clean_text(text_content)
                    
                    page_metadata = {
                        'page_number': page_num,
                        'text_length': len(text_content),
                        'extraction_method': 'PyPDF2'
                    }
                    
                    parsed_page = ParsedPage(
                        page_number=page_num,
                        text_content=text_content,
                        metadata=page_metadata
                    )
                    pages.append(parsed_page)
                    
                except Exception as e:
                    self._logger.warning(f"Failed to extract text from page {page_num}: {e}")
                    # Add empty page to maintain page numbering
                    pages.append(ParsedPage(
                        page_number=page_num,
                        text_content="",
                        metadata={'error': str(e), 'extraction_method': 'PyPDF2'}
                    ))
            
            return ParsedDocument(
                pages=pages,
                total_pages=len(pages),
                metadata=metadata
            )
            
        except Exception as e:
            raise DocumentProcessingError(f"PyPDF2 parsing failed: {str(e)}")
    
    def supports_file(self, mime_type: str) -> bool:
        """Check if PyPDF2 supports this file type."""
        return mime_type.lower() in ['application/pdf', 'application/x-pdf']
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)


class PDFPlumberParsingStrategy(PDFParsingStrategy):
    """
    PDF parsing strategy using pdfplumber library.
    
    Provides more advanced text extraction capabilities.
    """
    
    def __init__(self):
        """Initialize pdfplumber parsing strategy."""
        if not PARSING_AVAILABLE:
            raise ImportError("pdfplumber library not available")
        
        self._logger = logging.getLogger(__name__)
    
    async def parse_document(self, file_data: BinaryIO) -> ParsedDocument:
        """
        Parse PDF using pdfplumber.
        
        Args:
            file_data: Binary PDF file data
            
        Returns:
            Parsed document
            
        Raises:
            DocumentProcessingError: If parsing fails
        """
        try:
            # Reset file pointer
            file_data.seek(0)
            
            with pdfplumber.open(file_data) as pdf:
                # Extract metadata
                metadata = {
                    'parser': 'pdfplumber',
                    'pages': len(pdf.pages)
                }
                
                # Add document metadata if available
                if pdf.metadata:
                    metadata.update({
                        'title': pdf.metadata.get('Title', ''),
                        'author': pdf.metadata.get('Author', ''),
                        'subject': pdf.metadata.get('Subject', ''),
                        'creator': pdf.metadata.get('Creator', ''),
                    })
                
                # Extract text from each page
                pages = []
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        text_content = page.extract_text()
                        
                        # Clean and validate text
                        text_content = self._clean_text(text_content)
                        
                        # Extract additional page metadata
                        page_metadata = {
                            'page_number': page_num,
                            'text_length': len(text_content),
                            'extraction_method': 'pdfplumber',
                            'width': page.width,
                            'height': page.height,
                        }
                        
                        # Add table and image counts if available
                        try:
                            tables = page.extract_tables()
                            page_metadata['table_count'] = len(tables) if tables else 0
                        except:
                            page_metadata['table_count'] = 0
                        
                        parsed_page = ParsedPage(
                            page_number=page_num,
                            text_content=text_content,
                            metadata=page_metadata
                        )
                        pages.append(parsed_page)
                        
                    except Exception as e:
                        self._logger.warning(f"Failed to extract text from page {page_num}: {e}")
                        # Add empty page to maintain page numbering
                        pages.append(ParsedPage(
                            page_number=page_num,
                            text_content="",
                            metadata={'error': str(e), 'extraction_method': 'pdfplumber'}
                        ))
                
                return ParsedDocument(
                    pages=pages,
                    total_pages=len(pages),
                    metadata=metadata
                )
                
        except Exception as e:
            raise DocumentProcessingError(f"pdfplumber parsing failed: {str(e)}")
    
    def supports_file(self, mime_type: str) -> bool:
        """Check if pdfplumber supports this file type."""
        return mime_type.lower() in ['application/pdf', 'application/x-pdf']
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text with advanced processing.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace and normalize
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 1:  # Skip very short lines
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)


class PDFParsingService:
    """
    Service for parsing PDF documents using different strategies.
    
    Follows Strategy Pattern and Dependency Inversion Principle.
    """
    
    def __init__(self, strategies: Optional[List[PDFParsingStrategy]] = None):
        """
        Initialize PDF parsing service.
        
        Args:
            strategies: List of parsing strategies to use
        """
        self._strategies = strategies or []
        self._logger = logging.getLogger(__name__)
        
        # Add default strategies if none provided
        if not self._strategies and PARSING_AVAILABLE:
            try:
                self._strategies.append(PDFPlumberParsingStrategy())
            except ImportError:
                pass
            
            try:
                self._strategies.append(PyPDF2ParsingStrategy())
            except ImportError:
                pass
    
    async def parse_document(
        self,
        file_data: BinaryIO,
        mime_type: str,
        preferred_strategy: Optional[str] = None
    ) -> ParsedDocument:
        """
        Parse PDF document using available strategies.
        
        Args:
            file_data: Binary PDF file data
            mime_type: MIME type of the file
            preferred_strategy: Preferred parsing strategy name
            
        Returns:
            Parsed document
            
        Raises:
            DocumentProcessingError: If parsing fails with all strategies
        """
        if not self._strategies:
            raise DocumentProcessingError("No PDF parsing strategies available")
        
        # Filter strategies that support this file type
        compatible_strategies = [
            strategy for strategy in self._strategies
            if strategy.supports_file(mime_type)
        ]
        
        if not compatible_strategies:
            raise DocumentProcessingError(f"No strategies support MIME type: {mime_type}")
        
        # Try preferred strategy first if specified
        if preferred_strategy:
            for strategy in compatible_strategies:
                if strategy.__class__.__name__.lower().startswith(preferred_strategy.lower()):
                    compatible_strategies.remove(strategy)
                    compatible_strategies.insert(0, strategy)
                    break
        
        # Try each strategy until one succeeds
        last_error = None
        for strategy in compatible_strategies:
            try:
                self._logger.info(f"Attempting to parse with {strategy.__class__.__name__}")
                
                # Reset file pointer for each attempt
                file_data.seek(0)
                
                parsed_document = await strategy.parse_document(file_data)
                
                # Validate parsed document
                if self._validate_parsed_document(parsed_document):
                    self._logger.info(f"Successfully parsed with {strategy.__class__.__name__}")
                    return parsed_document
                else:
                    raise DocumentProcessingError("Parsed document validation failed")
                    
            except Exception as e:
                last_error = e
                self._logger.warning(f"Strategy {strategy.__class__.__name__} failed: {e}")
                continue
        
        # All strategies failed
        raise DocumentProcessingError(
            f"All parsing strategies failed. Last error: {last_error}"
        )
    
    async def create_document_chunks(
        self,
        parsed_document: ParsedDocument,
        chunk_size: int = ProcessingConstants.DEFAULT_CHUNK_SIZE
    ) -> List[DocumentChunk]:
        """
        Create document chunks from parsed document.
        
        Args:
            parsed_document: Parsed document to chunk
            chunk_size: Maximum size of each chunk
            
        Returns:
            List of document chunks
        """
        chunks = []
        
        for page in parsed_document.pages:
            if not page.text_content.strip():
                continue
            
            # Split page text into chunks
            page_chunks = self._split_text_into_chunks(
                page.text_content,
                chunk_size
            )
            
            # Create DocumentChunk entities with correct character positions
            current_position = 0
            for i, chunk_text in enumerate(page_chunks):
                start_pos = current_position
                end_pos = current_position + len(chunk_text)
                
                chunk = DocumentChunk(
                    chunk_id=f"page_{page.page_number}_chunk_{i}",
                    page_number=PageNumber(page.page_number),
                    text_content=chunk_text,
                    start_position=start_pos,
                    end_position=end_pos
                )
                chunks.append(chunk)
                
                # Update position for next chunk (account for sentence separators)
                current_position = end_pos
                if i < len(page_chunks) - 1:  # Not the last chunk
                    current_position += 2  # Add '. ' separator length
        
        return chunks
    
    def _validate_parsed_document(self, document: ParsedDocument) -> bool:
        """
        Validate parsed document.
        
        Args:
            document: Parsed document to validate
            
        Returns:
            True if document is valid
        """
        if not document or not document.pages:
            return False
        
        if document.total_pages <= 0:
            return False
        
        # Check if at least some pages have content
        pages_with_content = sum(
            1 for page in document.pages
            if page.text_content and page.text_content.strip()
        )
        
        # At least 10% of pages should have content
        min_content_pages = max(1, document.total_pages // 10)
        return pages_with_content >= min_content_pages
    
    def _split_text_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """
        Split text into chunks of specified size with improved word boundary handling.
        
        Args:
            text: Text to split
            chunk_size: Maximum size of each chunk
            
        Returns:
            List of text chunks
        """
        # Clean and normalize text first
        text = self._clean_extracted_text(text)
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        # Try multiple splitting strategies for better results
        
        # First try sentence splitting
        sentences = [s.strip() for s in text.split('. ') if s.strip()]
        if len(sentences) > 1:
            current_chunk = ""
            
            for sentence in sentences:
                # Add sentence to current chunk if it fits
                if len(current_chunk) + len(sentence) + 2 <= chunk_size:
                    if current_chunk:
                        current_chunk += ". " + sentence
                    else:
                        current_chunk = sentence
                else:
                    # Start new chunk
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence
            
            # Add remaining chunk
            if current_chunk:
                chunks.append(current_chunk)
        else:
            # Fallback to word-based splitting for better boundaries
            words = text.split()
            current_chunk = ""
            
            for word in words:
                if len(current_chunk) + len(word) + 1 <= chunk_size:
                    if current_chunk:
                        current_chunk += " " + word
                    else:
                        current_chunk = word
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = word
            
            if current_chunk:
                chunks.append(current_chunk)
        
        return chunks
    
    def _clean_extracted_text(self, text: str) -> str:
        """
        Clean and normalize extracted PDF text to improve quality.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text with proper spacing
        """
        import re
        
        # Add spaces between concatenated words (common PDF extraction issue)
        # Look for lowercase letter followed by uppercase letter
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Add spaces between letters and numbers
        text = re.sub(r'([a-zA-Z])([0-9])', r'\1 \2', text)
        text = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', text)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive newlines but preserve paragraph breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'\n(?!\n)', ' ', text)
        
        # Apply final cleanup
        return text.strip()


    def _ai_enhanced_text_cleaning(self, text: str) -> str:
        """
        AI-enhanced text cleaning with improved accuracy.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned and normalized text
        """
        import re
        
        # Enhanced text cleaning patterns for technical documents
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Fix missing spaces after sentences
        text = re.sub(r'([a-z])([A-Z][a-z])', r'\1 \2', text)  # CamelCase separation
        text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)  # Remove duplicate words
        
        # Remove excessive whitespace
        text = re.sub(r'\s{3,}', '  ', text)  # Limit consecutive spaces
        text = re.sub(r'\n{3,}', '\n\n', text)  # Limit consecutive newlines
        
        return text.strip()


class PDFParsingServiceFactory:
    """
    Factory for creating PDF parsing service instances.
    
    Follows Open/Closed Principle for extensibility.
    """
    
    @staticmethod
    def create_default() -> PDFParsingService:
        """
        Create PDF parsing service with default strategies.
        
        Returns:
            Configured PDF parsing service
        """
        return PDFParsingService()
    
    @staticmethod
    def create_with_strategies(strategies: List[PDFParsingStrategy]) -> PDFParsingService:
        """
        Create PDF parsing service with custom strategies.
        
        Args:
            strategies: List of parsing strategies
            
        Returns:
            Configured PDF parsing service
        """
        return PDFParsingService(strategies)
