"""
Page Number Value Object

"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class PageNumber:
    """
    Value object representing a PDF page number.
    
    Ensures page numbers are valid (positive integers starting from 1).
    """
    value: int
    
    def __post_init__(self) -> None:
        """Validate the page number."""
        if not isinstance(self.value, int):
            raise TypeError("Page number must be an integer")
        
        if self.value < 1:
            raise ValueError("Page number must be positive (starting from 1)")
    
    @classmethod
    def from_int(cls, value: int) -> PageNumber:
        """Create PageNumber from integer value."""
        return cls(value)
    
    def next_page(self) -> PageNumber:
        """Get the next page number."""
        return PageNumber(self.value + 1)
    
    def previous_page(self) -> PageNumber:
        """Get the previous page number."""
        if self.value <= 1:
            raise ValueError("Cannot get previous page for page 1")
        return PageNumber(self.value - 1)
    
    def is_first_page(self) -> bool:
        """Check if this is the first page."""
        return self.value == 1
    
    def __str__(self) -> str:
        """String representation of the page number."""
        return str(self.value)
    
    def __int__(self) -> int:
        """Integer representation of the page number."""
        return self.value
    
    def __eq__(self, other: Union[PageNumber, int]) -> bool:
        """Compare with another PageNumber or integer."""
        if isinstance(other, PageNumber):
            return self.value == other.value
        if isinstance(other, int):
            return self.value == other
        return False
    
    def __lt__(self, other: Union[PageNumber, int]) -> bool:
        """Less than comparison."""
        if isinstance(other, PageNumber):
            return self.value < other.value
        if isinstance(other, int):
            return self.value < other
        return NotImplemented
    
    def __le__(self, other: Union[PageNumber, int]) -> bool:
        """Less than or equal comparison."""
        if isinstance(other, PageNumber):
            return self.value <= other.value
        if isinstance(other, int):
            return self.value <= other
        return NotImplemented
    
    def __gt__(self, other: Union[PageNumber, int]) -> bool:
        """Greater than comparison."""
        if isinstance(other, PageNumber):
            return self.value > other.value
        if isinstance(other, int):
            return self.value > other
        return NotImplemented
    
    def __ge__(self, other: Union[PageNumber, int]) -> bool:
        """Greater than or equal comparison."""
        if isinstance(other, PageNumber):
            return self.value >= other.value
        if isinstance(other, int):
            return self.value >= other
        return NotImplemented
    
    def __hash__(self) -> int:
        """Hash implementation for use in sets and dictionaries."""
        return hash(self.value)
