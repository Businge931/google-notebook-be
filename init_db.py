#!/usr/bin/env python3
"""
Database Initialization Script

Simple script to create database tables using SQLAlchemy's built-in functionality.
This bypasses Alembic migration issues and directly creates the required tables.
"""
import asyncio
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.infrastructure.config.settings import get_settings
from src.infrastructure.adapters.database.connection import get_database_manager
from src.infrastructure.adapters.database.base import Base

# Import all models to ensure they're registered with SQLAlchemy
from src.infrastructure.adapters.database.models.document_model import DocumentModel, DocumentChunkModel
from src.infrastructure.adapters.database.models.chat_model import ChatSessionModel, MessageModel, CitationModel


async def init_database():
    """Initialize database with all required tables."""
    print("🔧 Initializing database...")
    
    # Get settings and database manager
    settings = get_settings()
    db_manager = get_database_manager(settings.database)
    
    try:
        # Create all tables
        print("📊 Creating database tables...")
        await db_manager.create_tables()
        print("✅ Database tables created successfully!")
        
        # Test database connectivity
        print("🔍 Testing database connectivity...")
        health_check = await db_manager.health_check()
        
        if health_check:
            print("✅ Database connectivity test passed!")
        else:
            print("❌ Database connectivity test failed!")
            return False
            
        # List created tables
        print("\n📋 Created tables:")
        for table_name in Base.metadata.tables.keys():
            print(f"  - {table_name}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error initializing database: {str(e)}")
        return False
        
    finally:
        # Cleanup
        await db_manager.close()


def main():
    """Main entry point."""
    print("🚀 Google NotebookLM Database Initialization")
    print("=" * 50)
    
    # Run async initialization
    success = asyncio.run(init_database())
    
    if success:
        print("\n🎉 Database initialization completed successfully!")
        print("You can now start the application with: make dev-reload")
        sys.exit(0)
    else:
        print("\n💥 Database initialization failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
