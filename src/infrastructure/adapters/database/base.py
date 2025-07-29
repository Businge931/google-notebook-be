"""
Database Base Configuration

"""
from datetime import datetime
from sqlalchemy import DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Mapped, mapped_column


class TimestampMixin:
    """
    Mixin for adding timestamp fields to models.
    
    Follows Single Responsibility Principle by only handling timestamps.
    """
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )


# Base class for all database models
Base = declarative_base()


class BaseModel(Base, TimestampMixin):
    """
    Abstract base model for all database entities.
    
    Provides common functionality while allowing extension (Open/Closed Principle).
    """
    __abstract__ = True
    
    def to_dict(self) -> dict:
        """Convert model to dictionary representation."""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }
    
    def update_from_dict(self, data: dict) -> None:
        """Update model from dictionary data."""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def __repr__(self) -> str:
        """String representation of the model."""
        class_name = self.__class__.__name__
        return f"<{class_name}({self.to_dict()})>"
