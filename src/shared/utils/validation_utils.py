"""
Validation Utilities

Follows S.O.L.I.D principles:
- Single Responsibility: Only handles validation operations
- Open/Closed: Can be extended with new validation functions
- Liskov Substitution: All validation functions follow same contract
- Interface Segregation: Specific validation functions for specific needs
- Dependency Inversion: No dependencies on concrete implementations
"""
import re
from typing import Any, List, Optional, Union
from pathlib import Path

from ..constants import FileConstants


def validate_email(email: str) -> bool:
    """
    Validate email format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if email is valid
    """
    if not email or not isinstance(email, str):
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_filename(filename: str) -> bool:
    """
    Validate filename format and constraints.
    
    Args:
        filename: Filename to validate
        
    Returns:
        True if filename is valid
    """
    if not filename or not isinstance(filename, str):
        return False
    
    # Check length
    if len(filename) > FileConstants.MAX_FILENAME_LENGTH:
        return False
    
    # Check for invalid characters
    if any(char in filename for char in FileConstants.INVALID_FILENAME_CHARS):
        return False
    
    # Check extension
    path = Path(filename)
    if path.suffix.lower() not in FileConstants.ALLOWED_EXTENSIONS:
        return False
    
    return True


def validate_file_size(file_size: int) -> bool:
    """
    Validate file size constraints.
    
    Args:
        file_size: File size in bytes
        
    Returns:
        True if file size is valid
    """
    if not isinstance(file_size, int):
        return False
    
    return FileConstants.MIN_FILE_SIZE_BYTES <= file_size <= FileConstants.MAX_FILE_SIZE_BYTES


def validate_mime_type(mime_type: str) -> bool:
    """
    Validate MIME type for allowed file types.
    
    Args:
        mime_type: MIME type to validate
        
    Returns:
        True if MIME type is allowed
    """
    if not mime_type or not isinstance(mime_type, str):
        return False
    
    return mime_type in FileConstants.ALLOWED_MIME_TYPES


def validate_uuid_format(uuid_string: str) -> bool:
    """
    Validate UUID format.
    
    Args:
        uuid_string: UUID string to validate
        
    Returns:
        True if UUID format is valid
    """
    if not uuid_string or not isinstance(uuid_string, str):
        return False
    
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    return bool(re.match(uuid_pattern, uuid_string.lower()))


def validate_page_number(page_number: Union[int, str]) -> bool:
    """
    Validate page number.
    
    Args:
        page_number: Page number to validate
        
    Returns:
        True if page number is valid
    """
    try:
        page_num = int(page_number)
        return page_num >= 1
    except (ValueError, TypeError):
        return False


def validate_similarity_score(score: Union[float, int]) -> bool:
    """
    Validate similarity score range.
    
    Args:
        score: Similarity score to validate
        
    Returns:
        True if score is valid (between 0.0 and 1.0)
    """
    try:
        score_float = float(score)
        return 0.0 <= score_float <= 1.0
    except (ValueError, TypeError):
        return False


def validate_text_content(content: str, min_length: int = 1, max_length: int = 10000) -> bool:
    """
    Validate text content length and format.
    
    Args:
        content: Text content to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        
    Returns:
        True if content is valid
    """
    if not isinstance(content, str):
        return False
    
    content_length = len(content.strip())
    return min_length <= content_length <= max_length


def validate_list_not_empty(items: List[Any]) -> bool:
    """
    Validate that a list is not empty.
    
    Args:
        items: List to validate
        
    Returns:
        True if list is not empty
    """
    return isinstance(items, list) and len(items) > 0


def validate_positive_integer(value: Union[int, str]) -> bool:
    """
    Validate that a value is a positive integer.
    
    Args:
        value: Value to validate
        
    Returns:
        True if value is a positive integer
    """
    try:
        int_value = int(value)
        return int_value > 0
    except (ValueError, TypeError):
        return False


def validate_non_negative_integer(value: Union[int, str]) -> bool:
    """
    Validate that a value is a non-negative integer.
    
    Args:
        value: Value to validate
        
    Returns:
        True if value is a non-negative integer
    """
    try:
        int_value = int(value)
        return int_value >= 0
    except (ValueError, TypeError):
        return False


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters.
    
    Args:
        filename: Filename to sanitize
        
    Returns:
        Sanitized filename
    """
    if not filename:
        return "unnamed_file.pdf"
    
    # Remove invalid characters
    sanitized = filename
    for char in FileConstants.INVALID_FILENAME_CHARS:
        sanitized = sanitized.replace(char, "_")
    
    # Ensure it has a valid extension
    path = Path(sanitized)
    if path.suffix.lower() not in FileConstants.ALLOWED_EXTENSIONS:
        sanitized = f"{path.stem}.pdf"
    
    # Truncate if too long
    if len(sanitized) > FileConstants.MAX_FILENAME_LENGTH:
        name_part = sanitized[:FileConstants.MAX_FILENAME_LENGTH - 4]
        sanitized = f"{name_part}.pdf"
    
    return sanitized


def validate_chunk_position(start_position: int, end_position: int) -> bool:
    """
    Validate chunk position values.
    
    Args:
        start_position: Start position in text
        end_position: End position in text
        
    Returns:
        True if positions are valid
    """
    if not isinstance(start_position, int) or not isinstance(end_position, int):
        return False
    
    return start_position >= 0 and end_position > start_position
