"""
LlamaParse PDF Service - Advanced AI-Driven PDF Analysis

Implements LlamaParse by LlamaIndex for superior PDF-to-markdown conversion
and enhanced vectorization capabilities for RAG systems.

"""
import asyncio
import logging
from typing import List, BinaryIO, Optional, Dict, Any
from dataclasses import dataclass
import io
import tempfile
import os

try:
    from llama_parse import LlamaParse
    from llama_index.core import Document
    LLAMAPARSE_AVAILABLE = True
except ImportError:
    LLAMAPARSE_AVAILABLE = False

from src.infrastructure.adapters.services.pdf_parsing_service import (
    PDFParsingStrategy, 
    ParsedDocument, 
    ParsedPage
)
from src.shared.exceptions import DocumentProcessingError
from src.infrastructure.config.settings import get_settings


@dataclass
class EnhancedParsedPage:
    """
    Enhanced parsed page with AI-extracted structure and metadata.
    
    Includes semantic information extracted by LlamaParse.
    """
    page_number: int
    text_content: str
    markdown_content: str
    structured_elements: Dict[str, Any]  # Headers, tables, lists, etc.
    semantic_metadata: Dict[str, Any]    # AI-extracted semantic info
    extraction_confidence: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EnhancedParsedDocument:
    """
    Enhanced parsed document with AI-driven structure analysis.
    
    Provides rich semantic information for improved RAG performance.
    """
    pages: List[EnhancedParsedPage]
    total_pages: int
    document_structure: Dict[str, Any]   # Overall document structure
    semantic_summary: str                # AI-generated document summary
    key_topics: List[str]               # Extracted key topics
    document_type: str                  # Detected document type
    extraction_quality_score: float    # Overall extraction quality
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def total_text_length(self) -> int:
        """Get total text length across all pages."""
        return sum(len(page.text_content) for page in self.pages)
    
    @property
    def total_markdown_length(self) -> int:
        """Get total markdown length across all pages."""
        return sum(len(page.markdown_content) for page in self.pages)


class LlamaParseStrategy(PDFParsingStrategy):
    """
    Advanced PDF parsing strategy using LlamaParse AI service.
    
    Provides superior text extraction, structure recognition, and
    markdown conversion for enhanced RAG performance.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LlamaParse strategy.
        
        Args:
            api_key: LlamaParse API key (optional, will use settings if not provided)
        """
        if not LLAMAPARSE_AVAILABLE:
            raise ImportError("LlamaParse library not available. Install with: pip install llama-parse")
        
        self._logger = logging.getLogger(__name__)
        
        # Get API key from parameter or settings
        if api_key:
            self._api_key = api_key
        else:
            settings = get_settings()
            self._api_key = getattr(settings, 'llama_parse_api_key', None)
            
        if not self._api_key:
            raise ValueError("LlamaParse API key is required. Set LLAMA_PARSE_API_KEY in environment or pass as parameter.")
        
        # Initialize LlamaParse with optimal settings for PDF analysis
        self._parser = LlamaParse(
            api_key=self._api_key,
            result_type="markdown",  # Get markdown for better structure
            verbose=True,
            language="en",
            parsing_instruction=self._get_parsing_instructions()
        )
    
    def _get_parsing_instructions(self) -> str:
        """
        Get optimized parsing instructions for document analysis.
        
        Returns:
            Detailed parsing instructions for LlamaParse
        """
        return """
        Extract and preserve the complete document structure including:
        1. All text content with proper formatting
        2. Headers and subheaders with hierarchy
        3. Tables with proper column/row structure
        4. Lists and bullet points
        5. Citations and references
        6. Mathematical formulas and equations
        7. Figure captions and descriptions
        8. Footnotes and endnotes
        
        Maintain semantic relationships between sections.
        Preserve technical terminology and proper nouns.
        Convert complex layouts to clean, structured markdown.
        Ensure all content is searchable and vectorization-ready.
        """
    
    async def parse_document(self, file_data: BinaryIO) -> ParsedDocument:
        """
        Parse PDF using LlamaParse AI service.
        
        Args:
            file_data: Binary PDF file data
            
        Returns:
            Enhanced parsed document with AI-extracted structure
            
        Raises:
            DocumentProcessingError: If parsing fails
        """
        try:
            # Reset file pointer
            file_data.seek(0)
            
            # Create temporary file for LlamaParse
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(file_data.read())
                temp_file_path = temp_file.name
            
            try:
                self._logger.info("🤖 Starting LlamaParse AI-driven PDF analysis...")
                
                # Parse document with LlamaParse
                documents = await asyncio.to_thread(
                    self._parser.load_data, 
                    temp_file_path
                )
                
                if not documents:
                    raise DocumentProcessingError("LlamaParse returned no documents")
                
                # Process parsed documents
                enhanced_pages = []
                total_pages = 0
                
                for doc_idx, document in enumerate(documents):
                    # Extract structured content
                    markdown_content = document.text
                    
                    # Analyze document structure
                    structured_elements = self._analyze_document_structure(markdown_content)
                    
                    # Extract semantic metadata
                    semantic_metadata = self._extract_semantic_metadata(markdown_content, structured_elements)
                    
                    # Convert markdown to plain text for compatibility
                    plain_text = self._markdown_to_text(markdown_content)
                    
                    # Calculate extraction confidence
                    confidence = self._calculate_extraction_confidence(
                        plain_text, 
                        markdown_content, 
                        structured_elements
                    )
                    
                    # Create enhanced page
                    enhanced_page = EnhancedParsedPage(
                        page_number=doc_idx + 1,
                        text_content=plain_text,
                        markdown_content=markdown_content,
                        structured_elements=structured_elements,
                        semantic_metadata=semantic_metadata,
                        extraction_confidence=confidence,
                        metadata={
                            'parser': 'LlamaParse',
                            'document_id': document.doc_id,
                            'source_file': temp_file_path
                        }
                    )
                    
                    enhanced_pages.append(enhanced_page)
                    total_pages += 1
                
                # Analyze overall document structure
                document_structure = self._analyze_overall_structure(enhanced_pages)
                
                # Generate semantic summary
                semantic_summary = self._generate_document_summary(enhanced_pages)
                
                # Extract key topics
                key_topics = self._extract_key_topics(enhanced_pages)
                
                # Detect document type
                document_type = self._detect_document_type(enhanced_pages, document_structure)
                
                # Calculate overall quality score
                quality_score = self._calculate_overall_quality(enhanced_pages)
                
                # Create enhanced parsed document
                enhanced_document = EnhancedParsedDocument(
                    pages=enhanced_pages,
                    total_pages=total_pages,
                    document_structure=document_structure,
                    semantic_summary=semantic_summary,
                    key_topics=key_topics,
                    document_type=document_type,
                    extraction_quality_score=quality_score,
                    metadata={
                        'parser': 'LlamaParse',
                        'total_documents': len(documents),
                        'processing_time': 'calculated_separately'
                    }
                )
                
                # Convert to standard ParsedDocument for compatibility
                standard_pages = [
                    ParsedPage(
                        page_number=page.page_number,
                        text_content=page.text_content,
                        metadata={
                            **page.metadata,
                            'markdown_content': page.markdown_content,
                            'structured_elements': page.structured_elements,
                            'semantic_metadata': page.semantic_metadata,
                            'extraction_confidence': page.extraction_confidence
                        }
                    )
                    for page in enhanced_pages
                ]
                
                standard_document = ParsedDocument(
                    pages=standard_pages,
                    total_pages=total_pages,
                    metadata={
                        **enhanced_document.metadata,
                        'document_structure': document_structure,
                        'semantic_summary': semantic_summary,
                        'key_topics': key_topics,
                        'document_type': document_type,
                        'extraction_quality_score': quality_score
                    }
                )
                
                self._logger.info(f"✅ LlamaParse analysis complete: {total_pages} pages, quality score: {quality_score:.2f}")
                
                return standard_document
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            self._logger.error(f"LlamaParse parsing failed: {e}")
            raise DocumentProcessingError(f"LlamaParse parsing failed: {str(e)}")
    
    def supports_file(self, mime_type: str) -> bool:
        """Check if LlamaParse supports this file type."""
        supported_types = [
            'application/pdf',
            'application/x-pdf',
            'application/acrobat',
            'applications/vnd.pdf',
            'text/pdf',
            'text/x-pdf'
        ]
        return mime_type.lower() in supported_types
    
    def _analyze_document_structure(self, markdown_content: str) -> Dict[str, Any]:
        """
        Analyze document structure from markdown content.
        
        Args:
            markdown_content: Markdown content from LlamaParse
            
        Returns:
            Dictionary containing structural analysis
        """
        import re
        
        structure = {
            'headers': [],
            'tables': 0,
            'lists': 0,
            'code_blocks': 0,
            'links': 0,
            'images': 0,
            'sections': []
        }
        
        # Extract headers
        header_pattern = r'^(#{1,6})\s+(.+)$'
        headers = re.findall(header_pattern, markdown_content, re.MULTILINE)
        structure['headers'] = [
            {'level': len(level), 'text': text.strip()}
            for level, text in headers
        ]
        
        # Count structural elements
        structure['tables'] = len(re.findall(r'\|.*\|', markdown_content))
        structure['lists'] = len(re.findall(r'^[\*\-\+]\s+', markdown_content, re.MULTILINE))
        structure['code_blocks'] = len(re.findall(r'```', markdown_content)) // 2
        structure['links'] = len(re.findall(r'\[([^\]]+)\]\(([^)]+)\)', markdown_content))
        structure['images'] = len(re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', markdown_content))
        
        # Extract sections based on headers
        if structure['headers']:
            current_section = None
            sections = []
            
            for header in structure['headers']:
                if header['level'] == 1:
                    if current_section:
                        sections.append(current_section)
                    current_section = {
                        'title': header['text'],
                        'level': header['level'],
                        'subsections': []
                    }
                elif current_section and header['level'] > 1:
                    current_section['subsections'].append({
                        'title': header['text'],
                        'level': header['level']
                    })
            
            if current_section:
                sections.append(current_section)
            
            structure['sections'] = sections
        
        return structure
    
    def _extract_semantic_metadata(self, markdown_content: str, structured_elements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract semantic metadata from content.
        
        Args:
            markdown_content: Markdown content
            structured_elements: Structural analysis results
            
        Returns:
            Dictionary containing semantic metadata
        """
        import re
        
        metadata = {
            'word_count': len(markdown_content.split()),
            'paragraph_count': len(re.findall(r'\n\s*\n', markdown_content)) + 1,
            'technical_terms': [],
            'citations': [],
            'key_phrases': [],
            'complexity_score': 0.0
        }
        
        # Extract technical terms (capitalized words, acronyms)
        technical_pattern = r'\b[A-Z]{2,}(?:[A-Z][a-z]+)*\b|\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b'
        technical_terms = list(set(re.findall(technical_pattern, markdown_content)))
        metadata['technical_terms'] = technical_terms[:20]  # Limit to top 20
        
        # Extract citations (basic pattern)
        citation_patterns = [
            r'\[(\d+)\]',  # [1], [2], etc.
            r'\(([^)]+,\s*\d{4})\)',  # (Author, 2023)
            r'et al\.',  # et al.
        ]
        
        citations = []
        for pattern in citation_patterns:
            citations.extend(re.findall(pattern, markdown_content))
        metadata['citations'] = list(set(citations))[:10]  # Limit to top 10
        
        # Calculate complexity score based on various factors
        complexity_factors = [
            len(structured_elements.get('headers', [])) * 0.1,
            structured_elements.get('tables', 0) * 0.2,
            len(technical_terms) * 0.05,
            len(citations) * 0.1,
            (metadata['word_count'] / 1000) * 0.1  # Normalize by 1000 words
        ]
        
        metadata['complexity_score'] = min(1.0, sum(complexity_factors))
        
        return metadata
    
    def _markdown_to_text(self, markdown_content: str) -> str:
        """
        Convert markdown to plain text while preserving structure.
        
        Args:
            markdown_content: Markdown content
            
        Returns:
            Plain text content
        """
        import re
        
        # Remove markdown formatting but preserve structure
        text = markdown_content
        
        # Convert headers to plain text with spacing
        text = re.sub(r'^#{1,6}\s+(.+)$', r'\1\n', text, flags=re.MULTILINE)
        
        # Remove markdown links but keep text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Remove markdown images
        text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)
        
        # Remove code block markers
        text = re.sub(r'```[^\n]*\n', '', text)
        text = re.sub(r'```', '', text)
        
        # Remove bold/italic markers
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        
        # Clean up table formatting
        text = re.sub(r'\|', ' ', text)
        text = re.sub(r'^[-\s|]+$', '', text, flags=re.MULTILINE)
        
        # Clean up whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text.strip()
    
    def _calculate_extraction_confidence(self, plain_text: str, markdown_content: str, structured_elements: Dict[str, Any]) -> float:
        """
        Calculate confidence score for extraction quality.
        
        Args:
            plain_text: Extracted plain text
            markdown_content: Extracted markdown
            structured_elements: Structural analysis
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence_factors = []
        
        # Text length factor (longer text usually means better extraction)
        text_length_factor = min(1.0, len(plain_text) / 1000)
        confidence_factors.append(text_length_factor * 0.3)
        
        # Structure factor (more structure usually means better extraction)
        structure_count = (
            len(structured_elements.get('headers', [])) +
            structured_elements.get('tables', 0) +
            structured_elements.get('lists', 0)
        )
        structure_factor = min(1.0, structure_count / 10)
        confidence_factors.append(structure_factor * 0.3)
        
        # Markdown richness factor
        markdown_richness = len(markdown_content) / max(1, len(plain_text))
        richness_factor = min(1.0, (markdown_richness - 1.0) * 2)  # Expect markdown to be longer
        confidence_factors.append(richness_factor * 0.2)
        
        # Content quality factor (presence of technical terms, proper formatting)
        quality_indicators = [
            bool(structured_elements.get('headers')),
            structured_elements.get('tables', 0) > 0,
            structured_elements.get('lists', 0) > 0,
            len(plain_text.split()) > 50,  # Reasonable content length
        ]
        quality_factor = sum(quality_indicators) / len(quality_indicators)
        confidence_factors.append(quality_factor * 0.2)
        
        return sum(confidence_factors)
    
    def _analyze_overall_structure(self, pages: List[EnhancedParsedPage]) -> Dict[str, Any]:
        """Analyze overall document structure across all pages."""
        total_headers = sum(len(page.structured_elements.get('headers', [])) for page in pages)
        total_tables = sum(page.structured_elements.get('tables', 0) for page in pages)
        total_lists = sum(page.structured_elements.get('lists', 0) for page in pages)
        
        return {
            'total_headers': total_headers,
            'total_tables': total_tables,
            'total_lists': total_lists,
            'avg_confidence': sum(page.extraction_confidence for page in pages) / len(pages) if pages else 0,
            'structure_density': (total_headers + total_tables + total_lists) / len(pages) if pages else 0
        }
    
    def _generate_document_summary(self, pages: List[EnhancedParsedPage]) -> str:
        """Generate a semantic summary of the document."""
        # Simple implementation - could be enhanced with AI summarization
        total_words = sum(len(page.text_content.split()) for page in pages)
        avg_confidence = sum(page.extraction_confidence for page in pages) / len(pages) if pages else 0
        
        return f"Document with {len(pages)} pages, {total_words} words, average extraction confidence: {avg_confidence:.2f}"
    
    def _extract_key_topics(self, pages: List[EnhancedParsedPage]) -> List[str]:
        """Extract key topics from the document."""
        # Aggregate technical terms from all pages
        all_terms = []
        for page in pages:
            all_terms.extend(page.semantic_metadata.get('technical_terms', []))
        
        # Count frequency and return top terms
        from collections import Counter
        term_counts = Counter(all_terms)
        return [term for term, count in term_counts.most_common(10)]
    
    def _detect_document_type(self, pages: List[EnhancedParsedPage], structure: Dict[str, Any]) -> str:
        """Detect the type of document based on content and structure."""
        # Simple heuristic-based detection
        if structure.get('total_tables', 0) > 5:
            return "data_report"
        elif structure.get('total_headers', 0) > 10:
            return "technical_document"
        elif any('abstract' in page.text_content.lower()[:500] for page in pages):
            return "academic_paper"
        elif any('introduction' in page.text_content.lower()[:1000] for page in pages):
            return "manual_or_guide"
        else:
            return "general_document"
    
    def _calculate_overall_quality(self, pages: List[EnhancedParsedPage]) -> float:
        """Calculate overall extraction quality score."""
        if not pages:
            return 0.0
        
        return sum(page.extraction_confidence for page in pages) / len(pages)


class LlamaParseServiceFactory:
    """
    Factory for creating LlamaParse service instances.
    
    Follows Open/Closed Principle for extensibility.
    """
    
    @staticmethod
    def create_llamaparse_strategy(api_key: Optional[str] = None) -> LlamaParseStrategy:
        """
        Create LlamaParse strategy instance.
        
        Args:
            api_key: Optional API key (will use settings if not provided)
            
        Returns:
            Configured LlamaParse strategy
        """
        return LlamaParseStrategy(api_key=api_key)
    
    @staticmethod
    def is_available() -> bool:
        """
        Check if LlamaParse is available.
        
        Returns:
            True if LlamaParse can be used
        """
        return LLAMAPARSE_AVAILABLE
