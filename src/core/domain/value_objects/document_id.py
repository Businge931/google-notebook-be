"""
Document ID Value Object

"""
from __future__ import annotations
import uuid
from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class DocumentId:
    """
    Value object representing a unique document identifier.
    
    Immutable and validates UUID format to ensure data integrity.
    """
    value: str
    
    def __post_init__(self) -> None:
        """Validate the document ID format."""
        if not self.value:
            raise ValueError("Document ID cannot be empty")
        
        try:
            # Validate UUID format
            uuid.UUID(self.value)
        except ValueError as e:
            raise ValueError(f"Invalid document ID format: {self.value}") from e
    
    @classmethod
    def generate(cls) -> DocumentId:
        """Generate a new unique document ID."""
        return cls(str(uuid.uuid4()))
    
    @classmethod
    def from_string(cls, value: str) -> DocumentId:
        """Create DocumentId from string value."""
        return cls(value)
    
    def __str__(self) -> str:
        """String representation of the document ID."""
        return self.value
    
    def __eq__(self, other: Union[DocumentId, str]) -> bool:
        """Compare with another DocumentId or string."""
        if isinstance(other, DocumentId):
            return self.value == other.value
        if isinstance(other, str):
            return self.value == other
        return False
    
    def __hash__(self) -> int:
        """Hash implementation for use in sets and dictionaries."""
        return hash(self.value)
