"""
Document Processing Domain Service
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime

from ..entities import Document, DocumentChunk, ProcessingStage
from ..value_objects import DocumentId, PageNumber


class DocumentProcessingService:
    """
    Domain service for document processing business logic.
    
    Encapsulates complex business rules that don't naturally fit
    within a single entity.
    """
    
    def validate_document_for_processing(self, document: Document) -> bool:
        """
        Validate if a document is ready for processing.
        
        Args:
            document: Document entity to validate
            
        Returns:
            True if document can be processed
            
        Raises:
            ValueError: If document is not valid for processing
        """
        if document.is_deleted:
            raise ValueError("Cannot process deleted document")
        
        if document.is_failed:
            raise ValueError("Cannot process failed document")
        
        if document.status.value not in ["uploaded", "processing"]:
            raise ValueError(f"Document in {document.status.value} status cannot be processed")
        
        if not document.file_metadata.is_pdf:
            raise ValueError("Only PDF documents can be processed")
        
        return True
    
    def calculate_optimal_chunk_size(self, document: Document) -> int:
        """
        Calculate optimal chunk size based on document characteristics.
        
        Args:
            document: Document entity
            
        Returns:
            Optimal chunk size in characters
        """
        base_chunk_size = 1000  # Base chunk size
        
        # Adjust based on document size
        if document.file_metadata.file_size_mb > 50:
            # Larger documents get bigger chunks for efficiency
            return min(base_chunk_size * 2, 2000)
        elif document.file_metadata.file_size_mb < 5:
            # Smaller documents get smaller chunks for precision
            return max(base_chunk_size // 2, 500)
        
        return base_chunk_size
    
    def validate_chunk_consistency(self, document: Document, chunks: List[DocumentChunk]) -> bool:
        """
        Validate that chunks are consistent with document metadata.
        
        Args:
            document: Document entity
            chunks: List of document chunks
            
        Returns:
            True if chunks are consistent
            
        Raises:
            ValueError: If chunks are inconsistent
        """
        if not chunks:
            raise ValueError("Document must have at least one chunk")
        
        # Check page number consistency
        if document.page_count:
            max_page = max(chunk.page_number.value for chunk in chunks)
            if max_page > document.page_count:
                raise ValueError(f"Chunk page number {max_page} exceeds document page count {document.page_count}")
        
        # Check for duplicate chunk IDs
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Duplicate chunk IDs found")
        
        # Validate chunk ordering within pages
        page_chunks = {}
        for chunk in chunks:
            page_num = chunk.page_number.value
            if page_num not in page_chunks:
                page_chunks[page_num] = []
            page_chunks[page_num].append(chunk)
        
        for page_num, page_chunk_list in page_chunks.items():
            # Sort chunks by start position
            sorted_chunks = sorted(page_chunk_list, key=lambda c: c.start_position)
            
            # Check for overlapping positions
            for i in range(len(sorted_chunks) - 1):
                current_chunk = sorted_chunks[i]
                next_chunk = sorted_chunks[i + 1]
                
                if current_chunk.end_position > next_chunk.start_position:
                    raise ValueError(f"Overlapping chunks found on page {page_num}")
        
        return True
    
    def determine_processing_priority(self, document: Document) -> int:
        """
        Determine processing priority based on document characteristics.
        
        Args:
            document: Document entity
            
        Returns:
            Priority score (higher = more priority)
        """
        priority = 0
        
        # Smaller documents get higher priority
        if document.file_metadata.file_size_mb < 10:
            priority += 10
        elif document.file_metadata.file_size_mb < 50:
            priority += 5
        
        # Recently uploaded documents get higher priority
        time_since_upload = datetime.utcnow() - document.created_at
        if time_since_upload.total_seconds() < 300:  # 5 minutes
            priority += 15
        elif time_since_upload.total_seconds() < 1800:  # 30 minutes
            priority += 10
        
        # Failed documents that are retrying get lower priority
        if document.processing_error:
            priority -= 5
        
        return max(priority, 1)  # Minimum priority of 1
    
    def can_advance_to_stage(self, document: Document, target_stage: ProcessingStage) -> bool:
        """
        Check if document can advance to a specific processing stage.
        
        Args:
            document: Document entity
            target_stage: Target processing stage
            
        Returns:
            True if document can advance to target stage
        """
        if document.is_failed or document.is_deleted:
            return False
        
        # Define valid stage progressions
        stage_order = [
            ProcessingStage.UPLOAD_COMPLETE,
            ProcessingStage.TEXT_EXTRACTION,
            ProcessingStage.CHUNKING,
            ProcessingStage.VECTORIZATION,
            ProcessingStage.INDEXING,
            ProcessingStage.COMPLETE
        ]
        
        try:
            current_index = stage_order.index(document.processing_stage)
            target_index = stage_order.index(target_stage)
            
            # Can only advance to next stage or stay at current stage
            return target_index <= current_index + 1
        except ValueError:
            return False
    
    def estimate_processing_time(self, document: Document) -> int:
        """
        Estimate processing time in seconds based on document characteristics.
        
        Args:
            document: Document entity
            
        Returns:
            Estimated processing time in seconds
        """
        base_time = 30  # Base processing time in seconds
        
        # Time increases with file size
        size_factor = document.file_metadata.file_size_mb * 2
        
        # Time increases with page count if known
        page_factor = 0
        if document.page_count:
            page_factor = document.page_count * 1.5
        
        total_time = base_time + size_factor + page_factor
        
        # Cap at reasonable maximum
        return min(int(total_time), 1800)  # Max 30 minutes
    
    def validate_processing_results(self, document: Document, page_count: int, chunk_count: int) -> bool:
        """
        Validate the results of document processing.
        
        Args:
            document: Document entity
            page_count: Number of pages processed
            chunk_count: Number of chunks created
            
        Returns:
            True if processing results are valid
            
        Raises:
            ValueError: If processing results are invalid
        """
        if page_count <= 0:
            raise ValueError("Page count must be positive")
        
        if chunk_count <= 0:
            raise ValueError("Chunk count must be positive")
        
        # Validate reasonable chunk-to-page ratio
        chunks_per_page = chunk_count / page_count
        if chunks_per_page > 50:  # Too many chunks per page
            raise ValueError(f"Too many chunks per page: {chunks_per_page:.1f}")
        
        if chunks_per_page < 0.1:  # Too few chunks per page
            raise ValueError(f"Too few chunks per page: {chunks_per_page:.1f}")
        
        # Update document page count if not set
        if not document.page_count:
            document.set_page_count(page_count)
        elif document.page_count != page_count:
            # Page count mismatch - log warning but don't fail
            # This can happen with different parsing strategies
            pass
        
        return True
