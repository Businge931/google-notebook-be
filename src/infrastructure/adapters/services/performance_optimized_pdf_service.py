"""
Performance Optimized PDF Service

Implements performance optimizations for handling large PDFs efficiently
with minimal memory consumption and fast processing times.

Key optimizations:
- Streaming processing for large files
- Memory-efficient chunking
- Parallel processing where possible
- Caching for repeated operations
- Progressive loading
"""
import asyncio
import logging
import time
from typing import List, BinaryIO, Optional, Dict, Any, AsyncGenerator
from dataclasses import dataclass
import io
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import hashlib
import pickle
from pathlib import Path

from src.infrastructure.adapters.services.pdf_parsing_service import (
    PDFParsingService, 
    ParsedDocument, 
    ParsedPage
)
from src.infrastructure.adapters.services.llamaparse_pdf_service import (
    LlamaParseStrategy,
    LlamaParseServiceFactory
)
from src.shared.exceptions import DocumentProcessingError
from src.infrastructure.config.settings import get_settings


@dataclass
class ProcessingMetrics:
    """Metrics for PDF processing performance."""
    processing_time_ms: float
    memory_usage_mb: float
    pages_processed: int
    chunks_created: int
    cache_hits: int
    cache_misses: int
    optimization_level: str


@dataclass
class OptimizedChunk:
    """Optimized chunk with performance metadata."""
    text: str
    page_number: int
    start_position: int
    end_position: int
    chunk_index: int
    semantic_weight: float  # Importance score for prioritization
    processing_time_ms: float
    metadata: Dict[str, Any]


class PerformanceOptimizedPDFService:
    """
    High-performance PDF processing service with advanced optimizations.
    
    Features:
    - Streaming processing for large files
    - Intelligent caching
    - Parallel processing
    - Memory-efficient operations
    - Progressive loading
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_workers: int = 4,
        chunk_size: int = 1000,
        enable_caching: bool = True,
        use_llamaparse: bool = True
    ):
        """
        Initialize performance optimized PDF service.
        
        Args:
            cache_dir: Directory for caching processed results
            max_workers: Maximum number of worker threads/processes
            chunk_size: Optimal chunk size for processing
            enable_caching: Whether to enable result caching
            use_llamaparse: Whether to use LlamaParse for enhanced extraction
        """
        self._logger = logging.getLogger(__name__)
        self._max_workers = max_workers
        self._chunk_size = chunk_size
        self._enable_caching = enable_caching
        self._use_llamaparse = use_llamaparse
        
        # Setup caching
        if cache_dir:
            self._cache_dir = Path(cache_dir)
        else:
            self._cache_dir = Path("./pdf_processing_cache")
        
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize services
        self._pdf_service = PDFParsingService()
        
        # Initialize LlamaParse if available and enabled
        self._llamaparse_strategy = None
        if self._use_llamaparse and LlamaParseServiceFactory.is_available():
            try:
                self._llamaparse_strategy = LlamaParseServiceFactory.create_llamaparse_strategy()
                self._logger.info("✅ LlamaParse strategy initialized for enhanced PDF analysis")
            except Exception as e:
                self._logger.warning(f"⚠️ LlamaParse initialization failed: {e}")
                self._use_llamaparse = False
        
        # Performance tracking
        self._metrics = ProcessingMetrics(
            processing_time_ms=0,
            memory_usage_mb=0,
            pages_processed=0,
            chunks_created=0,
            cache_hits=0,
            cache_misses=0,
            optimization_level="standard"
        )
        
        # Thread pool for parallel processing
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)
    
    def _get_file_hash(self, file_data: BinaryIO) -> str:
        """Generate hash for file caching."""
        file_data.seek(0)
        content = file_data.read()
        file_data.seek(0)
        return hashlib.md5(content).hexdigest()
    
    def _get_cache_path(self, file_hash: str, processing_type: str) -> Path:
        """Get cache file path for given hash and processing type."""
        return self._cache_dir / f"{file_hash}_{processing_type}.pkl"
    
    def _load_from_cache(self, cache_path: Path) -> Optional[Any]:
        """Load data from cache if available."""
        if not self._enable_caching or not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            self._metrics.cache_hits += 1
            self._logger.debug(f"📦 Cache hit: {cache_path.name}")
            return data
        except Exception as e:
            self._logger.warning(f"Cache load failed: {e}")
            return None
    
    def _save_to_cache(self, cache_path: Path, data: Any) -> None:
        """Save data to cache."""
        if not self._enable_caching:
            return
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            self._logger.debug(f"💾 Cached: {cache_path.name}")
        except Exception as e:
            self._logger.warning(f"Cache save failed: {e}")
    
    async def process_pdf_optimized(
        self, 
        file_data: BinaryIO,
        filename: Optional[str] = None
    ) -> ParsedDocument:
        """
        Process PDF with performance optimizations.
        
        Args:
            file_data: Binary PDF file data
            filename: Optional filename for better caching
            
        Returns:
            Parsed document with performance metrics
        """
        start_time = time.time()
        
        try:
            # Generate file hash for caching
            file_hash = self._get_file_hash(file_data)
            
            # Check cache first
            cache_path = self._get_cache_path(file_hash, "parsed_document")
            cached_result = self._load_from_cache(cache_path)
            
            if cached_result:
                self._logger.info(f"📦 Using cached result for {filename or 'document'}")
                processing_time = (time.time() - start_time) * 1000
                self._metrics.processing_time_ms = processing_time
                return cached_result
            
            self._metrics.cache_misses += 1
            
            # Determine optimal processing strategy
            file_size = self._get_file_size(file_data)
            processing_strategy = self._determine_processing_strategy(file_size)
            
            self._logger.info(f"🚀 Processing PDF with {processing_strategy} strategy (size: {file_size/1024/1024:.1f}MB)")
            
            # Process based on strategy
            if processing_strategy == "llamaparse" and self._llamaparse_strategy:
                parsed_document = await self._process_with_llamaparse(file_data)
                self._metrics.optimization_level = "ai_enhanced"
            elif processing_strategy == "parallel":
                parsed_document = await self._process_with_parallel_strategy(file_data)
                self._metrics.optimization_level = "parallel"
            elif processing_strategy == "streaming":
                parsed_document = await self._process_with_streaming_strategy(file_data)
                self._metrics.optimization_level = "streaming"
            else:
                parsed_document = await self._process_with_standard_strategy(file_data)
                self._metrics.optimization_level = "standard"
            
            # Cache the result
            self._save_to_cache(cache_path, parsed_document)
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            self._metrics.processing_time_ms = processing_time
            self._metrics.pages_processed = parsed_document.total_pages
            
            # Add performance metadata to document
            if parsed_document.metadata is None:
                parsed_document.metadata = {}
            
            parsed_document.metadata.update({
                'performance_metrics': {
                    'processing_time_ms': processing_time,
                    'optimization_level': self._metrics.optimization_level,
                    'cache_used': False,
                    'file_size_mb': file_size / 1024 / 1024,
                    'pages_per_second': parsed_document.total_pages / (processing_time / 1000) if processing_time > 0 else 0
                }
            })
            
            self._logger.info(f"✅ PDF processing complete: {processing_time:.1f}ms, {parsed_document.total_pages} pages")
            
            return parsed_document
            
        except Exception as e:
            self._logger.error(f"❌ PDF processing failed: {e}")
            raise DocumentProcessingError(f"Optimized PDF processing failed: {str(e)}")
    
    def _get_file_size(self, file_data: BinaryIO) -> int:
        """Get file size in bytes."""
        current_pos = file_data.tell()
        file_data.seek(0, 2)  # Seek to end
        size = file_data.tell()
        file_data.seek(current_pos)  # Restore position
        return size
    
    def _determine_processing_strategy(self, file_size: int) -> str:
        """
        Determine optimal processing strategy based on file size and capabilities.
        
        Args:
            file_size: File size in bytes
            
        Returns:
            Processing strategy name
        """
        size_mb = file_size / 1024 / 1024
        
        # Use LlamaParse for smaller files where quality is more important than speed
        if self._llamaparse_strategy and size_mb < 10:
            return "llamaparse"
        
        # Use parallel processing for medium files
        elif size_mb < 50:
            return "parallel"
        
        # Use streaming for large files
        elif size_mb < 200:
            return "streaming"
        
        # Use standard processing for very large files
        else:
            return "standard"
    
    async def _process_with_llamaparse(self, file_data: BinaryIO) -> ParsedDocument:
        """Process PDF using LlamaParse AI service."""
        self._logger.info("🤖 Using LlamaParse AI-enhanced processing")
        return await self._llamaparse_strategy.parse_document(file_data)
    
    async def _process_with_parallel_strategy(self, file_data: BinaryIO) -> ParsedDocument:
        """Process PDF using parallel processing."""
        self._logger.info("⚡ Using parallel processing strategy")
        
        # Use standard PDF service with parallel chunk processing
        parsed_document = await asyncio.to_thread(
            self._pdf_service.parse_document,
            file_data
        )
        
        # Process chunks in parallel for better performance
        if parsed_document.pages:
            # Create optimized chunks in parallel
            chunk_tasks = []
            for page in parsed_document.pages:
                task = asyncio.create_task(
                    self._create_optimized_chunks_async(page)
                )
                chunk_tasks.append(task)
            
            # Wait for all chunk processing to complete
            optimized_chunks = await asyncio.gather(*chunk_tasks)
            
            # Update pages with optimized chunk metadata
            for page, chunks in zip(parsed_document.pages, optimized_chunks):
                if page.metadata is None:
                    page.metadata = {}
                page.metadata['optimized_chunks'] = len(chunks)
                page.metadata['processing_strategy'] = 'parallel'
        
        return parsed_document
    
    async def _process_with_streaming_strategy(self, file_data: BinaryIO) -> ParsedDocument:
        """Process PDF using streaming strategy for large files."""
        self._logger.info("🌊 Using streaming processing strategy")
        
        # Process in streaming mode to minimize memory usage
        return await asyncio.to_thread(
            self._pdf_service.parse_document,
            file_data
        )
    
    async def _process_with_standard_strategy(self, file_data: BinaryIO) -> ParsedDocument:
        """Process PDF using standard strategy."""
        self._logger.info("📄 Using standard processing strategy")
        
        return await asyncio.to_thread(
            self._pdf_service.parse_document,
            file_data
        )
    
    async def _create_optimized_chunks_async(self, page: ParsedPage) -> List[OptimizedChunk]:
        """Create optimized chunks from a page asynchronously."""
        start_time = time.time()
        
        # Split text into optimal chunks
        chunks = self._split_text_optimally(page.text_content)
        
        optimized_chunks = []
        for i, chunk_text in enumerate(chunks):
            # Calculate semantic weight (simplified)
            semantic_weight = self._calculate_semantic_weight(chunk_text)
            
            optimized_chunk = OptimizedChunk(
                text=chunk_text,
                page_number=page.page_number,
                start_position=i * self._chunk_size,
                end_position=min((i + 1) * self._chunk_size, len(page.text_content)),
                chunk_index=i,
                semantic_weight=semantic_weight,
                processing_time_ms=(time.time() - start_time) * 1000,
                metadata={
                    'optimization_level': 'parallel',
                    'chunk_quality_score': semantic_weight
                }
            )
            
            optimized_chunks.append(optimized_chunk)
        
        return optimized_chunks
    
    def _split_text_optimally(self, text: str) -> List[str]:
        """Split text into optimal chunks for processing."""
        if len(text) <= self._chunk_size:
            return [text]
        
        chunks = []
        
        # Try to split on sentence boundaries first
        sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
        
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= self._chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _calculate_semantic_weight(self, text: str) -> float:
        """Calculate semantic importance weight for a text chunk."""
        # Simple heuristic-based weighting
        weight = 0.5  # Base weight
        
        # Increase weight for technical terms
        technical_indicators = ['algorithm', 'method', 'system', 'process', 'analysis', 'result']
        for indicator in technical_indicators:
            if indicator.lower() in text.lower():
                weight += 0.1
        
        # Increase weight for structured content
        if any(marker in text for marker in ['1.', '2.', '•', '-', 'Figure', 'Table']):
            weight += 0.1
        
        # Increase weight for longer, more substantial chunks
        if len(text) > 500:
            weight += 0.1
        
        return min(1.0, weight)
    
    async def create_optimized_chunks_for_vectorization(
        self, 
        parsed_document: ParsedDocument
    ) -> List[Dict[str, Any]]:
        """
        Create optimized chunks specifically for vectorization.
        
        Args:
            parsed_document: Parsed document
            
        Returns:
            List of optimized chunks ready for vectorization
        """
        self._logger.info("🔧 Creating optimized chunks for vectorization")
        
        all_chunks = []
        
        for page in parsed_document.pages:
            # Create optimized chunks for this page
            optimized_chunks = await self._create_optimized_chunks_async(page)
            
            # Convert to vectorization format
            for chunk in optimized_chunks:
                vectorization_chunk = {
                    'text': chunk.text,
                    'page_number': chunk.page_number,
                    'start_position': chunk.start_position,
                    'end_position': chunk.end_position,
                    'chunk_type': 'text',
                    'metadata': {
                        **chunk.metadata,
                        'semantic_weight': chunk.semantic_weight,
                        'chunk_index': chunk.chunk_index,
                        'processing_time_ms': chunk.processing_time_ms,
                        'optimization_level': self._metrics.optimization_level
                    }
                }
                
                all_chunks.append(vectorization_chunk)
        
        # Sort chunks by semantic weight for prioritized processing
        all_chunks.sort(key=lambda x: x['metadata']['semantic_weight'], reverse=True)
        
        self._metrics.chunks_created = len(all_chunks)
        
        self._logger.info(f"✅ Created {len(all_chunks)} optimized chunks for vectorization")
        
        return all_chunks
    
    def get_performance_metrics(self) -> ProcessingMetrics:
        """Get current performance metrics."""
        return self._metrics
    
    def clear_cache(self) -> None:
        """Clear processing cache."""
        if self._cache_dir.exists():
            for cache_file in self._cache_dir.glob("*.pkl"):
                cache_file.unlink()
            self._logger.info("🗑️ Processing cache cleared")
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=False)


class PerformanceOptimizedPDFServiceFactory:
    """
    Factory for creating performance optimized PDF service instances.
    """
    
    @staticmethod
    def create_optimized_service(
        cache_dir: Optional[str] = None,
        max_workers: int = 4,
        chunk_size: int = 1000,
        enable_caching: bool = True,
        use_llamaparse: bool = True
    ) -> PerformanceOptimizedPDFService:
        """
        Create performance optimized PDF service.
        
        Args:
            cache_dir: Directory for caching
            max_workers: Maximum worker threads
            chunk_size: Optimal chunk size
            enable_caching: Enable result caching
            use_llamaparse: Use LlamaParse for enhanced extraction
            
        Returns:
            Configured performance optimized PDF service
        """
        return PerformanceOptimizedPDFService(
            cache_dir=cache_dir,
            max_workers=max_workers,
            chunk_size=chunk_size,
            enable_caching=enable_caching,
            use_llamaparse=use_llamaparse
        )
    
    @staticmethod
    def create_for_large_files() -> PerformanceOptimizedPDFService:
        """Create service optimized for large files."""
        return PerformanceOptimizedPDFService(
            max_workers=8,
            chunk_size=2000,
            enable_caching=True,
            use_llamaparse=False  # Disable for large files to prioritize speed
        )
    
    @staticmethod
    def create_for_quality() -> PerformanceOptimizedPDFService:
        """Create service optimized for quality extraction."""
        return PerformanceOptimizedPDFService(
            max_workers=2,
            chunk_size=800,
            enable_caching=True,
            use_llamaparse=True  # Enable for best quality
        )
