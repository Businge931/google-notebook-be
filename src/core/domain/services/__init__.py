"""
Domain Services

Domain services encapsulate business logic that doesn't naturally fit
within a single entity or value object.
"""
from .document_processing_service import DocumentProcessingService

__all__ = [
    "DocumentProcessingService",
]