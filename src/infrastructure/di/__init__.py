"""
Dependency Injection Module

"""
from .container import (
    DIContainer,
    get_container,
    initialize_container,
    cleanup_container,
    get_document_repository,
    get_chat_repository,
    get_file_storage_service,
    get_document_processing_service,
)
from .dependencies import (
    get_document_vectorization_use_case,
    get_bulk_document_vectorization_use_case,
    get_similarity_search_use_case,
)

__all__ = [
    "DIContainer",
    "get_container",
    "initialize_container", 
    "cleanup_container",
    "get_document_repository",
    "get_chat_repository",
    "get_file_storage_service",
    "get_document_processing_service",
    "get_document_vectorization_use_case",
    "get_bulk_document_vectorization_use_case",
    "get_similarity_search_use_case",
]
