"""
Session ID Value Object

"""
from __future__ import annotations
import uuid
from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class SessionId:
    """
    Value object representing a unique chat session identifier.
    
    Immutable and validates UUID format to ensure data integrity.
    """
    value: str
    
    def __post_init__(self) -> None:
        """Validate the session ID format."""
        if not self.value:
            raise ValueError("Session ID cannot be empty")
        
        try:
            # Validate UUID format
            uuid.UUID(self.value)
        except ValueError as e:
            raise ValueError(f"Invalid session ID format: {self.value}") from e
    
    @classmethod
    def generate(cls) -> SessionId:
        """Generate a new unique session ID."""
        return cls(str(uuid.uuid4()))
    
    @classmethod
    def from_string(cls, value: str) -> SessionId:
        """Create SessionId from string value."""
        return cls(value)
    
    def __str__(self) -> str:
        """String representation of the session ID."""
        return self.value
    
    def __eq__(self, other: Union[SessionId, str]) -> bool:
        """Compare with another SessionId or string."""
        if isinstance(other, SessionId):
            return self.value == other.value
        if isinstance(other, str):
            return self.value == other
        return False
    
    def __hash__(self) -> int:
        """Hash implementation for use in sets and dictionaries."""
        return hash(self.value)
