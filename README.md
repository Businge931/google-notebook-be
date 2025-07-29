# Google NotebookLM Clone - Backend

A production-ready backend API for a Google NotebookLM clone built with Python, FastAPI, and hexagonal architecture.

## Features

- **PDF Upload & Processing**: Handle large PDF files with efficient parsing and text extraction
- **AI-Powered Chat Interface**: Interactive Q&A about PDF contents using RAG (Retrieval Augmented Generation)
- **Smart Citations**: Clickable citations linking to specific PDF pages
- **Vector Search**: Semantic search within documents using embeddings
- **Background Processing**: Async PDF processing and vectorization
- **Production Ready**: Comprehensive logging, monitoring, and error handling

## Architecture

This project follows **Hexagonal Architecture** (Ports and Adapters) principles:

- **Core Layer**: Pure business logic (Domain + Application layers)
- **Infrastructure Layer**: External integrations (Database, AI services, Storage)
- **Presentation Layer**: REST API endpoints and middleware

See [ARCHITECTURE.md](./ARCHITECTURE.md) for detailed architecture documentation.

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- OpenAI API key

### Full setup with PostgreSQL and Redis

```bash
cd backend
make setup         # Complete setup (install deps, setup services, init db)
make dev          # Start the development server
```

### Manual Setup (Step by Step)

1. **Install dependencies**:

   ```bash
   make install      # Creates venv and installs all dependencies
   # OR manually:
   # python -m venv venv
   # source venv/bin/activate
   # pip install -r requirements.txt
   ```

2. **Environment configuration**:

   ```bash
   # .env file is already configured with development defaults
   # Update OpenAI API key and other settings as needed
   ```

3. **Database setup** (Optional - for PostgreSQL):

   ```bash
   make setup-postgres  # Setup PostgreSQL database
   make setup-redis     # Setup Redis cache
   make init-db         # Initialize database with migrations
   ```

4. **Start the application**:

   ```bash
   make dev             # Start development server
   # OR with auto-reload:
   make dev-reload      # Start with auto-reload on file changes
   ```

   **🚀 API will be available at:**

   - **Main API**: http://localhost:8000
   - **Interactive Docs**: http://localhost:8000/api/docs
   - **ReDoc**: http://localhost:8000/api/redoc

### Available Make Commands

```bash
make help           # Show all available commands
make install        # Install dependencies
make dev           # Start development server
make dev-reload    # Start with auto-reload
make test          # Run all tests
make lint          # Run code linting
make format        # Format code
make clean         # Clean up temporary files
make check-services # Check if PostgreSQL/Redis are running
```

### Docker Development

1. **Start all services**:

   ```bash
   docker compose up -d
   ```

2. **View logs**:

   ```bash
   docker compose logs -f app
   ```

3. **Stop services**:
   ```bash
   docker compose down
   ```

## API Endpoints

### Document Management

- `POST /api/v1/documents/upload` - Upload PDF file
- `GET /api/v1/documents/{document_id}` - Get document metadata
- `GET /api/v1/documents/{document_id}/content` - Get processed content
- `DELETE /api/v1/documents/{document_id}` - Delete document

### Chat Interface

- `POST /api/v1/chat/sessions` - Create chat session
- `POST /api/v1/chat/sessions/{session_id}/messages` - Send message
- `GET /api/v1/chat/sessions/{session_id}/messages` - Get chat history

### Search & Retrieval

- `POST /api/v1/documents/{document_id}/search` - Semantic search
- `GET /api/v1/documents/{document_id}/citations/{citation_id}` - Get citation details

### Health & Monitoring

- `GET /api/v1/health` - Health check
- `GET /api/v1/status/{task_id}` - Background task status

## Development

### Code Structure

```
src/
├── core/                    # Business logic
│   ├── domain/             # Entities, value objects, repositories
│   └── application/        # Use cases, DTOs, interfaces
├── infrastructure/         # External integrations
│   ├── adapters/          # Database, AI, storage implementations
│   ├── config/            # Configuration management
│   └── external_services/ # Third-party API clients
├── presentation/          # API layer
│   ├── api/v1/           # REST endpoints
│   └── middleware/       # Custom middleware
└── shared/               # Common utilities
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test types
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/
```

### Code Quality

```bash
# Format code
black src/
isort src/

# Lint code
flake8 src/
mypy src/
```

## Configuration

Key environment variables:

- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `OPENAI_API_KEY`: OpenAI API key for AI features
- `STORAGE_TYPE`: File storage type (local/s3)
- `VECTOR_DB_TYPE`: Vector database type (chroma/pinecone)

See `.env.example` for complete configuration options.

## Deployment

### Production Checklist

- [ ] Set `DEBUG=False`
- [ ] Configure production database
- [ ] Set up file storage (S3 recommended)
- [ ] Configure monitoring (Sentry)
- [ ] Set up SSL/TLS
- [ ] Configure rate limiting
- [ ] Set up backup strategies

### Docker Production

```bash
# Build production image
docker build -t notebooklm-backend .

# Run with production settings
docker run -p 8000:8000 --env-file .env.prod notebooklm-backend
```

## Contributing

1. Follow hexagonal architecture principles
2. Write tests for new features
3. Update documentation
4. Follow code style guidelines (black, isort, flake8)
