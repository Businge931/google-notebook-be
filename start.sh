#!/bin/bash

# Production startup script for Google NotebookLM Clone Backend

echo "Starting Google NotebookLM Clone Backend..."

# Run database migrations
echo "Running database migrations..."
alembic upgrade head

# Create necessary directories
mkdir -p uploads
mkdir -p vector_indexes
mkdir -p logs

# Start the application
echo "Starting FastAPI application..."
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
