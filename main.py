"""
Main application entry point for Google NotebookLM Clone Backend
"""
import asyncio
import greenlet
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.presentation.api.endpoints import documents, chat, advanced_search, citations, vectorization, async_documents
from src.infrastructure.adapters.web.routes import enhanced_document_routes
from src.presentation.middleware.error_handler import ErrorHandlerMiddleware
from src.infrastructure.config.settings import get_settings
from src.infrastructure.di.container import initialize_container, cleanup_container

# Load environment variables from .env file
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events"""
    # Initialize greenlet context for SQLAlchemy async operations
    try:
        # Ensure greenlet context is properly initialized
        loop = asyncio.get_event_loop()
        greenlet.greenlet(lambda: None).switch()
    except Exception:
        pass  # Ignore if already initialized
    
    # Startup: Initialize DI container
    await initialize_container()
    yield
    # Shutdown: Cleanup DI container
    await cleanup_container()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    settings = get_settings()
    
    app = FastAPI(
        title="Google NotebookLM Clone API",
        description="Backend API for PDF document interaction and chat interface",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000", 
            "http://127.0.0.1:3000", 
            "http://localhost:5173",  
            "http://127.0.0.1:5173",  
            "*"
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add custom middleware
    app.add_middleware(ErrorHandlerMiddleware)
    
    # Include routers
    app.include_router(documents.router, prefix="/api/v1")
    app.include_router(async_documents.router, prefix="/api/v1")  # Async processing for large PDFs
    app.include_router(enhanced_document_routes.router, prefix="/api/v1")  # Enhanced upload with large file support
    app.include_router(chat.router, prefix="/api/v1")
    app.include_router(advanced_search.router, prefix="/api/v1")
    app.include_router(citations.router, prefix="/api/v1")
    app.include_router(vectorization.router, prefix="/api/v1")
    
    # Large PDF optimization is now integrated into existing document processing pipeline
    
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info",
        # Allow large file uploads
        limit_max_requests=1000,
        limit_concurrency=1000,
        timeout_keep_alive=300,
        # Increase request size limits for large PDF uploads
        h11_max_incomplete_event_size=5000 * 1024 * 1024  # 5TB
    )
