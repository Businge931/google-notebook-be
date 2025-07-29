"""
AI-Enhanced PDF Processing Pipeline

"""
import asyncio
import logging
import time
from typing import List, BinaryIO, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from src.infrastructure.adapters.services.performance_optimized_pdf_service import (
    PerformanceOptimizedPDFService,
    PerformanceOptimizedPDFServiceFactory,
    ProcessingMetrics
)
from src.infrastructure.adapters.services.llamaparse_pdf_service import (
    LlamaParseServiceFactory
)
# Enhanced PDF parser import removed - using existing services
from src.infrastructure.adapters.services.enhanced_response_synthesis import (
    create_enhanced_response_synthesis
)
from src.core.domain.value_objects import DocumentId
from src.shared.exceptions import DocumentProcessingError
from src.infrastructure.config.settings import get_settings


@dataclass
class AIProcessingResult:
    """Result of AI-enhanced PDF processing."""
    document_id: DocumentId
    parsed_document: Any  # ParsedDocument
    optimized_chunks: List[Dict[str, Any]]
    processing_metrics: ProcessingMetrics
    ai_analysis: Dict[str, Any]
    vectorization_ready: bool
    quality_score: float


class AIEnhancedPDFPipeline:
    """
    Complete AI-enhanced PDF processing pipeline.
    
    Combines all improvements:
    - LlamaParse for superior text extraction
    - Performance optimizations for large files
    - AI-driven analysis and understanding
    - Optimized vectorization for RAG
    """
    
    def __init__(self):
        """Initialize the AI-enhanced PDF pipeline."""
        self._logger = logging.getLogger(__name__)
        self._settings = get_settings()
        
        # Initialize services
        self._performance_service = None
        self._pdf_parser = None
        self._response_synthesizer = None
        
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize all required services."""
        try:
            # Initialize performance optimized service
            self._performance_service = PerformanceOptimizedPDFServiceFactory.create_optimized_service(
                cache_dir="./ai_pdf_cache",
                max_workers=4,
                chunk_size=1000,
                enable_caching=True,
                use_llamaparse=True
            )
            
            # Use existing PDF parsing service
            from src.infrastructure.adapters.services.pdf_parsing_service import PDFParsingService
            self._pdf_parser = PDFParsingService()
            
            # Initialize enhanced response synthesizer
            openai_api_key = getattr(self._settings.ai_service, 'openai_api_key', None)
            self._response_synthesizer = create_enhanced_response_synthesis(openai_api_key)
            
            self._logger.info("✅ AI-Enhanced PDF Pipeline initialized successfully")
            
        except Exception as e:
            self._logger.error(f"❌ Failed to initialize AI-Enhanced PDF Pipeline: {e}")
            raise DocumentProcessingError(f"Pipeline initialization failed: {str(e)}")
    
    async def process_pdf_with_ai_analysis(
        self,
        file_data: BinaryIO,
        document_id: DocumentId,
        filename: Optional[str] = None,
        analysis_level: str = "comprehensive"
    ) -> AIProcessingResult:
        """
        Process PDF with complete AI analysis.
        
        Args:
            file_data: Binary PDF file data
            document_id: Document identifier
            filename: Optional filename
            analysis_level: Level of analysis (basic, standard, comprehensive)
            
        Returns:
            Complete AI processing result
        """
        start_time = time.time()
        
        try:
            self._logger.info(f"🚀 Starting AI-enhanced PDF processing for {filename or document_id.value}")
            
            # Step 1: Performance-optimized parsing
            self._logger.info("📄 Step 1: Performance-optimized PDF parsing")
            parsed_document = await self._performance_service.process_pdf_optimized(
                file_data=file_data,
                filename=filename
            )
            
            # Step 2: Create optimized chunks for vectorization
            self._logger.info("🔧 Step 2: Creating optimized chunks for vectorization")
            optimized_chunks = await self._performance_service.create_optimized_chunks_for_vectorization(
                parsed_document
            )
            
            # Step 3: AI-driven content analysis
            self._logger.info("🤖 Step 3: AI-driven content analysis")
            ai_analysis = await self._perform_ai_analysis(
                parsed_document, 
                optimized_chunks, 
                analysis_level
            )
            
            # Step 4: Quality assessment
            self._logger.info("📊 Step 4: Quality assessment")
            quality_score = self._calculate_overall_quality_score(
                parsed_document,
                optimized_chunks,
                ai_analysis
            )
            
            # Step 5: Vectorization readiness check
            vectorization_ready = self._check_vectorization_readiness(
                optimized_chunks,
                quality_score
            )
            
            # Get performance metrics
            processing_metrics = self._performance_service.get_performance_metrics()
            
            # Create result
            result = AIProcessingResult(
                document_id=document_id,
                parsed_document=parsed_document,
                optimized_chunks=optimized_chunks,
                processing_metrics=processing_metrics,
                ai_analysis=ai_analysis,
                vectorization_ready=vectorization_ready,
                quality_score=quality_score
            )
            
            total_time = (time.time() - start_time) * 1000
            
            self._logger.info(
                f"✅ AI-enhanced processing complete: {total_time:.1f}ms, "
                f"quality: {quality_score:.2f}, "
                f"chunks: {len(optimized_chunks)}, "
                f"vectorization ready: {vectorization_ready}"
            )
            
            return result
            
        except Exception as e:
            self._logger.error(f"❌ AI-enhanced PDF processing failed: {e}")
            raise DocumentProcessingError(f"AI processing failed: {str(e)}")
    
    async def _perform_ai_analysis(
        self,
        parsed_document: Any,
        optimized_chunks: List[Dict[str, Any]],
        analysis_level: str
    ) -> Dict[str, Any]:
        """
        Perform AI-driven content analysis.
        
        Args:
            parsed_document: Parsed document
            optimized_chunks: Optimized chunks
            analysis_level: Level of analysis
            
        Returns:
            AI analysis results
        """
        analysis = {
            'document_type': 'unknown',
            'key_topics': [],
            'complexity_score': 0.0,
            'technical_terms': [],
            'structure_analysis': {},
            'content_summary': '',
            'extraction_confidence': 0.0,
            'analysis_level': analysis_level
        }
        
        try:
            # Extract document metadata for analysis
            document_text = ' '.join([
                page.text_content for page in parsed_document.pages
            ])
            
            # Basic analysis (always performed)
            analysis.update(await self._perform_basic_analysis(document_text, optimized_chunks))
            
            if analysis_level in ['standard', 'comprehensive']:
                # Standard analysis
                analysis.update(await self._perform_standard_analysis(document_text, optimized_chunks))
            
            if analysis_level == 'comprehensive':
                # Comprehensive analysis with AI
                analysis.update(await self._perform_comprehensive_analysis(document_text, optimized_chunks))
            
        except Exception as e:
            self._logger.warning(f"AI analysis partially failed: {e}")
            analysis['analysis_error'] = str(e)
        
        return analysis
    
    async def _perform_basic_analysis(
        self,
        document_text: str,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform basic document analysis."""
        import re
        from collections import Counter
        
        # Basic statistics
        word_count = len(document_text.split())
        sentence_count = len(re.findall(r'[.!?]+', document_text))
        
        # Extract technical terms (simple heuristic)
        technical_pattern = r'\b[A-Z]{2,}(?:[A-Z][a-z]+)*\b|\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b'
        technical_terms = list(set(re.findall(technical_pattern, document_text)))[:20]
        
        # Calculate basic complexity
        avg_sentence_length = word_count / max(1, sentence_count)
        complexity_score = min(1.0, (avg_sentence_length / 20) + (len(technical_terms) / 50))
        
        # Detect document type (basic heuristics)
        document_type = self._detect_document_type_basic(document_text)
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'technical_terms': technical_terms,
            'complexity_score': complexity_score,
            'document_type': document_type,
            'avg_sentence_length': avg_sentence_length
        }
    
    async def _perform_standard_analysis(
        self,
        document_text: str,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform standard document analysis."""
        # Analyze chunk quality distribution
        chunk_qualities = [
            chunk['metadata'].get('semantic_weight', 0.5)
            for chunk in chunks
        ]
        
        avg_chunk_quality = sum(chunk_qualities) / len(chunk_qualities) if chunk_qualities else 0
        
        # Extract key topics using frequency analysis
        key_topics = self._extract_key_topics_frequency(document_text)
        
        # Analyze document structure
        structure_analysis = self._analyze_document_structure_basic(document_text)
        
        return {
            'avg_chunk_quality': avg_chunk_quality,
            'key_topics': key_topics,
            'structure_analysis': structure_analysis,
            'high_quality_chunks': len([q for q in chunk_qualities if q > 0.7])
        }
    
    async def _perform_comprehensive_analysis(
        self,
        document_text: str,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform comprehensive AI-driven analysis."""
        try:
            # Use AI for content summarization and analysis
            if self._response_synthesizer:
                # Create a sample query for analysis
                analysis_query = "Analyze this document and provide key insights, topics, and structure."
                
                # Use top chunks for analysis (limit for performance)
                top_chunks = sorted(
                    chunks,
                    key=lambda x: x['metadata'].get('semantic_weight', 0),
                    reverse=True
                )[:10]
                
                # Convert chunks to search results format for synthesis
                from src.core.domain.services.advanced_search_service import SearchResult
                from src.core.domain.value_objects import DocumentId
                
                search_results = []
                for i, chunk in enumerate(top_chunks):
                    search_result = SearchResult(
                        id=f"analysis_{i}",
                        document_id=DocumentId("analysis_doc"),
                        document_title="Document Analysis",
                        document_filename="analysis.pdf",
                        chunk_id=f"chunk_{i}",
                        content=chunk['text'],
                        snippet=chunk['text'][:200],
                        relevance_score=chunk['metadata'].get('semantic_weight', 0.5),
                        confidence_score=chunk['metadata'].get('semantic_weight', 0.5),
                        page_number=chunk.get('page_number', 1),
                        section_title="Analysis Section",
                        start_position=chunk.get('start_position', 0),
                        end_position=chunk.get('end_position', 0),
                        highlighted_content=chunk['text'][:100],
                        metadata=chunk['metadata']
                    )
                    search_results.append(search_result)
                
                # Get AI analysis
                ai_response = await self._response_synthesizer.synthesize_comprehensive_response(
                    query=analysis_query,
                    search_results=search_results,
                    max_response_length=1000
                )
                
                return {
                    'ai_summary': ai_response.get('response', ''),
                    'ai_confidence': ai_response.get('confidence', 0.0),
                    'ai_analysis_type': ai_response.get('analysis_type', 'general'),
                    'comprehensive_analysis': True
                }
        
        except Exception as e:
            self._logger.warning(f"Comprehensive AI analysis failed: {e}")
        
        return {
            'ai_summary': 'AI analysis not available',
            'ai_confidence': 0.0,
            'comprehensive_analysis': False
        }
    
    def _detect_document_type_basic(self, text: str) -> str:
        """Detect document type using basic heuristics."""
        text_lower = text.lower()
        
        if 'abstract' in text_lower[:1000]:
            return 'academic_paper'
        elif 'table of contents' in text_lower[:2000]:
            return 'manual_or_guide'
        elif any(term in text_lower for term in ['api', 'endpoint', 'function', 'class']):
            return 'technical_documentation'
        elif any(term in text_lower for term in ['report', 'analysis', 'findings']):
            return 'report'
        else:
            return 'general_document'
    
    def _extract_key_topics_frequency(self, text: str) -> List[str]:
        """Extract key topics using frequency analysis."""
        import re
        from collections import Counter
        
        # Extract meaningful words (filter out common words)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # Simple stopword filtering
        stopwords = {
            'that', 'this', 'with', 'from', 'they', 'been', 'have', 'were', 'said',
            'each', 'which', 'their', 'time', 'will', 'about', 'would', 'there',
            'could', 'other', 'after', 'first', 'well', 'also', 'some', 'what',
            'only', 'when', 'here', 'how', 'our', 'out', 'up', 'may', 'way'
        }
        
        filtered_words = [word for word in words if word not in stopwords]
        
        # Get most common words
        word_counts = Counter(filtered_words)
        return [word for word, count in word_counts.most_common(10)]
    
    def _analyze_document_structure_basic(self, text: str) -> Dict[str, Any]:
        """Analyze basic document structure."""
        import re
        
        # Count structural elements
        headers = len(re.findall(r'^[A-Z][^.!?]*$', text, re.MULTILINE))
        numbered_lists = len(re.findall(r'^\d+\.', text, re.MULTILINE))
        bullet_points = len(re.findall(r'^[•\-\*]', text, re.MULTILINE))
        
        return {
            'estimated_headers': headers,
            'numbered_lists': numbered_lists,
            'bullet_points': bullet_points,
            'structure_score': min(1.0, (headers + numbered_lists + bullet_points) / 20)
        }
    
    def _calculate_overall_quality_score(
        self,
        parsed_document: Any,
        optimized_chunks: List[Dict[str, Any]],
        ai_analysis: Dict[str, Any]
    ) -> float:
        """Calculate overall quality score for the processing."""
        quality_factors = []
        
        # Document completeness
        if parsed_document.total_pages > 0:
            avg_page_length = parsed_document.total_text_length / parsed_document.total_pages
            completeness_score = min(1.0, avg_page_length / 1000)
            quality_factors.append(completeness_score * 0.3)
        
        # Chunk quality
        if optimized_chunks:
            avg_chunk_quality = sum(
                chunk['metadata'].get('semantic_weight', 0.5)
                for chunk in optimized_chunks
            ) / len(optimized_chunks)
            quality_factors.append(avg_chunk_quality * 0.3)
        
        # AI analysis confidence
        ai_confidence = ai_analysis.get('ai_confidence', ai_analysis.get('extraction_confidence', 0.5))
        quality_factors.append(ai_confidence * 0.2)
        
        # Structure analysis
        structure_score = ai_analysis.get('structure_analysis', {}).get('structure_score', 0.5)
        quality_factors.append(structure_score * 0.2)
        
        return sum(quality_factors) if quality_factors else 0.5
    
    def _check_vectorization_readiness(
        self,
        optimized_chunks: List[Dict[str, Any]],
        quality_score: float
    ) -> bool:
        """Check if the document is ready for vectorization."""
        # Minimum requirements for vectorization
        min_chunks = 3
        min_quality = 0.4
        min_chunk_length = 50
        
        if len(optimized_chunks) < min_chunks:
            return False
        
        if quality_score < min_quality:
            return False
        
        # Check chunk quality
        valid_chunks = [
            chunk for chunk in optimized_chunks
            if len(chunk['text']) >= min_chunk_length
        ]
        
        return len(valid_chunks) >= min_chunks
    
    async def process_for_vectorization(
        self,
        file_data: BinaryIO,
        document_id: DocumentId,
        filename: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process PDF specifically for vectorization.
        
        Args:
            file_data: Binary PDF file data
            document_id: Document identifier
            filename: Optional filename
            
        Returns:
            Tuple of (optimized_chunks, metadata)
        """
        # Process with comprehensive analysis
        result = await self.process_pdf_with_ai_analysis(
            file_data=file_data,
            document_id=document_id,
            filename=filename,
            analysis_level="comprehensive"
        )
        
        if not result.vectorization_ready:
            raise DocumentProcessingError(
                f"Document not ready for vectorization. Quality score: {result.quality_score:.2f}"
            )
        
        # Prepare metadata for vectorization
        vectorization_metadata = {
            'document_id': document_id.value,
            'filename': filename,
            'total_pages': result.parsed_document.total_pages,
            'total_chunks': len(result.optimized_chunks),
            'quality_score': result.quality_score,
            'processing_time_ms': result.processing_metrics.processing_time_ms,
            'optimization_level': result.processing_metrics.optimization_level,
            'ai_analysis': result.ai_analysis,
            'vectorization_ready': result.vectorization_ready
        }
        
        return result.optimized_chunks, vectorization_metadata


class AIEnhancedPDFPipelineFactory:
    """Factory for creating AI-enhanced PDF pipeline instances."""
    
    @staticmethod
    def create_pipeline() -> AIEnhancedPDFPipeline:
        """Create AI-enhanced PDF pipeline."""
        return AIEnhancedPDFPipeline()
    
    @staticmethod
    def create_for_development() -> AIEnhancedPDFPipeline:
        """Create pipeline optimized for development."""
        # Could add development-specific configurations
        return AIEnhancedPDFPipeline()
    
    @staticmethod
    def create_for_production() -> AIEnhancedPDFPipeline:
        """Create pipeline optimized for production."""
        # Could add production-specific configurations
        return AIEnhancedPDFPipeline()
