"""
Enhanced PDF Parser for Improved Text Extraction and Citation Quality

"""
import re
import logging
from typing import List, Dict, Any, Optional, BinaryIO
from dataclasses import dataclass
import io

try:
    import PyPDF2
    import pdfplumber
    PARSING_AVAILABLE = True
except ImportError:
    PARSING_AVAILABLE = False


@dataclass
class EnhancedParsedPage:
    """Enhanced parsed page with better structure preservation."""
    page_number: int
    text_content: str
    cleaned_text: str
    paragraphs: List[str]
    sentences: List[str]
    metadata: Dict[str, Any]


@dataclass
class EnhancedParsedDocument:
    """Enhanced parsed document with improved structure."""
    pages: List[EnhancedParsedPage]
    total_pages: int
    document_title: Optional[str]
    document_metadata: Dict[str, Any]
    full_text: str
    cleaned_full_text: str


class EnhancedPDFParser:
    """
    Enhanced PDF parser with improved text extraction and citation quality.
    
    Phase 1 improvements:
    - Better text cleaning and spacing
    - Contextual citation extraction
    - Document structure preservation
    """
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        if not PARSING_AVAILABLE:
            raise ImportError("PDF parsing libraries not available")
    
    async def parse_document_enhanced(self, file_data: BinaryIO) -> EnhancedParsedDocument:
        """
        Parse PDF with enhanced text extraction and structure preservation.
        
        Args:
            file_data: Binary PDF file data
            
        Returns:
            Enhanced parsed document with better text quality
        """
        try:
            # Reset file pointer
            file_data.seek(0)
            
            # Try pdfplumber first for better text extraction
            try:
                return await self._parse_with_pdfplumber(file_data)
            except Exception as e:
                self._logger.warning(f"pdfplumber parsing failed: {e}, falling back to PyPDF2")
                file_data.seek(0)
                return await self._parse_with_pypdf2(file_data)
                
        except Exception as e:
            self._logger.error(f"Enhanced PDF parsing failed: {e}")
            raise Exception(f"Failed to parse PDF: {e}")
    
    async def _parse_with_pdfplumber(self, file_data: BinaryIO) -> EnhancedParsedDocument:
        """Parse PDF using pdfplumber for better text extraction."""
        pages = []
        document_metadata = {}
        
        with pdfplumber.open(file_data) as pdf:
            # Extract document metadata
            if pdf.metadata:
                document_metadata = {
                    'title': pdf.metadata.get('Title', ''),
                    'author': pdf.metadata.get('Author', ''),
                    'subject': pdf.metadata.get('Subject', ''),
                    'creator': pdf.metadata.get('Creator', ''),
                    'producer': pdf.metadata.get('Producer', ''),
                    'creation_date': pdf.metadata.get('CreationDate', ''),
                    'modification_date': pdf.metadata.get('ModDate', ''),
                }
            
            document_title = document_metadata.get('title', '')
            
            # Process each page
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    # Extract text with layout preservation
                    raw_text = page.extract_text(layout=True) or ""
                    
                    # Enhanced text cleaning
                    cleaned_text = self._enhanced_text_cleaning(raw_text)
                    
                    # Extract paragraphs and sentences
                    paragraphs = self._extract_paragraphs(cleaned_text)
                    sentences = self._extract_sentences(cleaned_text)
                    
                    # Page metadata
                    page_metadata = {
                        'page_number': page_num,
                        'bbox': page.bbox,
                        'width': page.width,
                        'height': page.height,
                        'rotation': page.rotation,
                        'char_count': len(cleaned_text),
                        'word_count': len(cleaned_text.split()),
                        'paragraph_count': len(paragraphs),
                        'sentence_count': len(sentences)
                    }
                    
                    enhanced_page = EnhancedParsedPage(
                        page_number=page_num,
                        text_content=raw_text,
                        cleaned_text=cleaned_text,
                        paragraphs=paragraphs,
                        sentences=sentences,
                        metadata=page_metadata
                    )
                    
                    pages.append(enhanced_page)
                    
                except Exception as e:
                    self._logger.warning(f"Failed to parse page {page_num}: {e}")
                    continue
        
        # Create full document text
        full_text = "\n\n".join(page.text_content for page in pages)
        cleaned_full_text = "\n\n".join(page.cleaned_text for page in pages)
        
        return EnhancedParsedDocument(
            pages=pages,
            total_pages=len(pages),
            document_title=document_title,
            document_metadata=document_metadata,
            full_text=full_text,
            cleaned_full_text=cleaned_full_text
        )
    
    async def _parse_with_pypdf2(self, file_data: BinaryIO) -> EnhancedParsedDocument:
        """Fallback parsing with PyPDF2."""
        pages = []
        document_metadata = {}
        
        pdf_reader = PyPDF2.PdfReader(file_data)
        
        # Extract metadata
        if pdf_reader.metadata:
            document_metadata = {
                'title': pdf_reader.metadata.get('/Title', ''),
                'author': pdf_reader.metadata.get('/Author', ''),
                'subject': pdf_reader.metadata.get('/Subject', ''),
                'creator': pdf_reader.metadata.get('/Creator', ''),
                'producer': pdf_reader.metadata.get('/Producer', ''),
            }
        
        document_title = document_metadata.get('title', '')
        
        # Process each page
        for page_num, page in enumerate(pdf_reader.pages, 1):
            try:
                raw_text = page.extract_text()
                cleaned_text = self._enhanced_text_cleaning(raw_text)
                
                paragraphs = self._extract_paragraphs(cleaned_text)
                sentences = self._extract_sentences(cleaned_text)
                
                page_metadata = {
                    'page_number': page_num,
                    'char_count': len(cleaned_text),
                    'word_count': len(cleaned_text.split()),
                    'paragraph_count': len(paragraphs),
                    'sentence_count': len(sentences)
                }
                
                enhanced_page = EnhancedParsedPage(
                    page_number=page_num,
                    text_content=raw_text,
                    cleaned_text=cleaned_text,
                    paragraphs=paragraphs,
                    sentences=sentences,
                    metadata=page_metadata
                )
                
                pages.append(enhanced_page)
                
            except Exception as e:
                self._logger.warning(f"Failed to parse page {page_num}: {e}")
                continue
        
        full_text = "\n\n".join(page.text_content for page in pages)
        cleaned_full_text = "\n\n".join(page.cleaned_text for page in pages)
        
        return EnhancedParsedDocument(
            pages=pages,
            total_pages=len(pages),
            document_title=document_title,
            document_metadata=document_metadata,
            full_text=full_text,
            cleaned_full_text=cleaned_full_text
        )
    
    def _enhanced_text_cleaning(self, text: str) -> str:
        """
        Enhanced text cleaning to fix common PDF extraction issues.
        
        Addresses the spacing issues we saw in the response:
        "PrinceisacomputerprogramthatconvertsXMLandHTMLintoPDF"
        """
        if not text:
            return ""
        
        # Step 1: Fix concatenated words (most critical issue)
        # Add spaces between lowercase and uppercase letters
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Add spaces between letters and numbers
        text = re.sub(r'([a-zA-Z])([0-9])', r'\1 \2', text)
        text = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', text)
        
        # Add spaces between words and punctuation (common in PDFs)
        text = re.sub(r'([a-zA-Z])([.!?,:;])', r'\1\2', text)
        text = re.sub(r'([.!?,:;])([a-zA-Z])', r'\1 \2', text)
        
        # Step 2: Fix common PDF extraction artifacts
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix line breaks and paragraph separation
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r'\n(?!\n)', ' ', text)  # Single newlines to spaces
        
        # Step 3: Fix common PDF text issues
        # Remove page headers/footers patterns
        text = re.sub(r'\n\d+\n', '\n', text)  # Page numbers on separate lines
        text = re.sub(r'\nPage \d+\n', '\n', text)  # "Page X" patterns
        
        # Fix hyphenated words split across lines
        text = re.sub(r'-\s+', '', text)
        
        # Step 4: Normalize spacing and punctuation
        # Ensure proper spacing around punctuation
        text = re.sub(r'\s*([.!?])\s*', r'\1 ', text)
        text = re.sub(r'\s*([,;:])\s*', r'\1 ', text)
        
        # Clean up any remaining multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _extract_paragraphs(self, text: str) -> List[str]:
        """Extract paragraphs from cleaned text."""
        if not text:
            return []
        
        # Split by double newlines or significant spacing
        paragraphs = re.split(r'\n\n+', text)
        
        # Filter out very short paragraphs (likely artifacts)
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 20]
        
        return paragraphs
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from cleaned text."""
        if not text:
            return []
        
        # Split by sentence-ending punctuation
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Filter out very short sentences
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return sentences
    
    def create_enhanced_citations(self, search_results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Create enhanced citations with better context and formatting.
        
        Phase 1 improvement: Better citation quality with context.
        """
        enhanced_citations = []
        
        for i, result in enumerate(search_results[:5], 1):  # Top 5 results
            try:
                # Extract contextual information
                content = result.get('content', '')
                page_number = result.get('page_number', 1)
                document_title = result.get('document_title', 'Unknown Document')
                
                # Create intelligent snippet (not just character truncation)
                snippet = self._create_intelligent_snippet(content, query)
                
                # Extract surrounding context if available
                context_before, context_after = self._extract_context(content, query)
                
                # Calculate relevance indicators
                relevance_indicators = self._calculate_relevance_indicators(content, query)
                
                enhanced_citation = {
                    "id": f"citation_{i}",
                    "document_id": result.get('document_id'),
                    "document_title": document_title,
                    "page_number": page_number,
                    "snippet": snippet,
                    "context_before": context_before,
                    "context_after": context_after,
                    "relevance_score": result.get('relevance_score', 0.0),
                    "relevance_indicators": relevance_indicators,
                    "citation_type": self._determine_citation_type(content, query),
                    "key_phrases": self._extract_key_phrases(content, query),
                    "formatted_citation": self._format_academic_citation(
                        document_title, page_number, snippet
                    )
                }
                
                enhanced_citations.append(enhanced_citation)
                
            except Exception as e:
                self._logger.warning(f"Failed to create enhanced citation {i}: {e}")
                continue
        
        return enhanced_citations
    
    def _create_intelligent_snippet(self, content: str, query: str, max_length: int = 200) -> str:
        """Create an intelligent snippet that captures the most relevant part."""
        if not content:
            return ""
        
        # Clean the content first
        content = self._enhanced_text_cleaning(content)
        
        # Find the most relevant sentence containing query terms
        query_terms = query.lower().split()
        sentences = self._extract_sentences(content)
        
        best_sentence = ""
        best_score = 0
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = sum(1 for term in query_terms if term in sentence_lower)
            
            if score > best_score and len(sentence) <= max_length:
                best_score = score
                best_sentence = sentence
        
        # If no good sentence found, use the beginning
        if not best_sentence:
            best_sentence = content[:max_length]
            if len(content) > max_length:
                best_sentence += "..."
        
        return best_sentence.strip()
    
    def _extract_context(self, content: str, query: str) -> tuple[Optional[str], Optional[str]]:
        """Extract context before and after the relevant snippet."""
        sentences = self._extract_sentences(content)
        
        if len(sentences) < 3:
            return None, None
        
        # Find the sentence most relevant to the query
        query_terms = query.lower().split()
        relevant_idx = -1
        best_score = 0
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            score = sum(1 for term in query_terms if term in sentence_lower)
            if score > best_score:
                best_score = score
                relevant_idx = i
        
        if relevant_idx == -1:
            return None, None
        
        # Extract context
        context_before = sentences[relevant_idx - 1] if relevant_idx > 0 else None
        context_after = sentences[relevant_idx + 1] if relevant_idx < len(sentences) - 1 else None
        
        return context_before, context_after
    
    def _calculate_relevance_indicators(self, content: str, query: str) -> Dict[str, Any]:
        """Calculate various relevance indicators for the citation."""
        query_terms = query.lower().split()
        content_lower = content.lower()
        
        return {
            "query_term_matches": sum(1 for term in query_terms if term in content_lower),
            "query_term_density": sum(content_lower.count(term) for term in query_terms) / len(content.split()),
            "content_length": len(content),
            "sentence_count": len(self._extract_sentences(content)),
            "contains_numbers": bool(re.search(r'\d+', content)),
            "contains_dates": bool(re.search(r'\b\d{4}\b', content))
        }
    
    def _determine_citation_type(self, content: str, query: str) -> str:
        """Determine the type of citation based on content analysis."""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['definition', 'define', 'means', 'refers to']):
            return "definition"
        elif any(word in content_lower for word in ['example', 'instance', 'such as', 'including']):
            return "example"
        elif any(word in content_lower for word in ['result', 'conclusion', 'finding', 'shows']):
            return "evidence"
        elif any(word in content_lower for word in ['according to', 'states', 'claims', 'argues']):
            return "reference"
        else:
            return "general"
    
    def _extract_key_phrases(self, content: str, query: str, max_phrases: int = 3) -> List[str]:
        """Extract key phrases from the content relevant to the query."""
        # Simple key phrase extraction based on query terms
        query_terms = set(query.lower().split())
        sentences = self._extract_sentences(content)
        
        key_phrases = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(term in sentence_lower for term in query_terms):
                # Extract noun phrases (simplified)
                words = sentence.split()
                for i, word in enumerate(words):
                    if word.lower() in query_terms and i < len(words) - 2:
                        phrase = " ".join(words[i:i+3])
                        if len(phrase) > 10:  # Minimum phrase length
                            key_phrases.append(phrase)
        
        return key_phrases[:max_phrases]
    
    def _format_academic_citation(self, title: str, page: int, snippet: str) -> str:
        """Format citation in academic style."""
        title = title or "Unknown Document"
        snippet_preview = snippet[:50] + "..." if len(snippet) > 50 else snippet
        
        return f'"{snippet_preview}" ({title}, p. {page})'


# Factory function for easy integration
def create_enhanced_pdf_parser() -> EnhancedPDFParser:
    """Create an enhanced PDF parser instance."""
    return EnhancedPDFParser()
