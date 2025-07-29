
.PHONY: help install setup dev test clean docker build deploy

# Default target
.DEFAULT_GOAL := help

# CONFIGURATION
PYTHON := python3
PIP := pip
VENV := venv
VENV_BIN := $(VENV)/bin
PROJECT_NAME := google-notebooklm-backend
DOCKER_IMAGE := $(PROJECT_NAME)
DOCKER_TAG := latest

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# HELP
help: ## Show this help message
	@echo "$(BLUE)Google NotebookLM Clone Backend - Available Commands$(NC)"
	@echo "=================================================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ENVIRONMENT SETUP
check-python: ## Check if Python 3 is installed
	@echo "$(BLUE)Checking Python installation...$(NC)"
	@$(PYTHON) --version || (echo "$(RED)Python 3 is required but not installed$(NC)" && exit 1)
	@echo "$(GREEN)Python is installed$(NC)"

create-venv: check-python ## Create virtual environment
	@echo "$(BLUE)Creating virtual environment...$(NC)"
	@$(PYTHON) -m venv $(VENV)
	@echo "$(GREEN)Virtual environment created$(NC)"

install-deps: ## Install Python dependencies
	@echo "$(BLUE)Installing Python dependencies...$(NC)"
	@$(VENV_BIN)/pip install --upgrade pip
	@$(VENV_BIN)/pip install -r requirements.txt
	@echo "$(GREEN)Dependencies installed$(NC)"

install: create-venv install-deps ## Create venv and install all dependencies
	@echo "$(GREEN)Installation complete!$(NC)"

# DATABASE SETUP
test-setup-postgres: ## Set up PostgreSQL database
	@echo "$(BLUE)Setting up PostgreSQL...$(NC)"
	@echo "$(YELLOW)Please ensure PostgreSQL is installed and running$(NC)"
	@echo "$(YELLOW)Creating database and user...$(NC)"
	@sudo -u postgres psql -c "CREATE DATABASE notebooklm_db;" || echo "Database may already exist"
	@sudo -u postgres psql -c "CREATE USER notebooklm_user WITH PASSWORD 'your_password';" || echo "User may already exist"
	@sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE notebooklm_db TO notebooklm_user;" || echo "Privileges may already be granted"
	@echo "$(GREEN)PostgreSQL setup complete$(NC)"

setup-redis: ## Set up Redis
	@echo "$(BLUE)Setting up Redis...$(NC)"
	@echo "$(YELLOW)Installing Redis if not present...$(NC)"
	@sudo apt update && sudo apt install -y redis-server || echo "Redis may already be installed"
	@sudo systemctl start redis-server || echo "Redis may already be running"
	@sudo systemctl enable redis-server || echo "Redis may already be enabled"
	@echo "$(GREEN)Redis setup complete$(NC)"

init-db: ## Initialize database with migrations
	@echo "$(BLUE)Initializing database...$(NC)"
	@$(VENV_BIN)/alembic upgrade head
	@echo "$(GREEN)Database initialized$(NC)"

# DEVELOPMENT
dev: ## Start development server
	@echo "$(BLUE)Starting development server...$(NC)"
	@$(VENV_BIN)/python main.py

dev-reload: ## Start development server with auto-reload
	@echo "$(BLUE)Starting development server with auto-reload...$(NC)"
	@$(VENV_BIN)/uvicorn main:app --host 0.0.0.0 --port 8000 --reload

test-app: ## Start test application
	@echo "$(BLUE)Starting test application...$(NC)"
	@$(VENV_BIN)/python test_app.py

# TESTING
test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	@$(VENV_BIN)/pytest tests/ -v

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	@$(VENV_BIN)/pytest tests/unit/ -v

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	@$(VENV_BIN)/pytest tests/integration/ -v

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	@$(VENV_BIN)/pytest tests/ --cov=src --cov-report=html --cov-report=term

# CODE QUALITY
lint: ## Run linting
	@echo "$(BLUE)Running linting...$(NC)"
	@$(VENV_BIN)/flake8 src/ tests/
	@$(VENV_BIN)/black --check src/ tests/
	@$(VENV_BIN)/isort --check-only src/ tests/

format: ## Format code
	@echo "$(BLUE)Formatting code...$(NC)"
	@$(VENV_BIN)/black src/ tests/
	@$(VENV_BIN)/isort src/ tests/
	@echo "$(GREEN)Code formatted$(NC)"

# DOCKER COMMANDS
docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	@docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	@echo "$(GREEN)Docker image built: $(DOCKER_IMAGE):$(DOCKER_TAG)$(NC)"

docker-run: ## Run application in Docker container
	@echo "$(BLUE)Running Docker container...$(NC)"
	@docker run -p 8000:8000 --env-file .env $(DOCKER_IMAGE):$(DOCKER_TAG)

docker-compose-up: ## Start all services with docker-compose
	@echo "$(BLUE)Starting services with docker-compose...$(NC)"
	@docker compose up -d
	@echo "$(GREEN)Services started$(NC)"

docker-compose-down: ## Stop all services
	@echo "$(BLUE)Stopping services...$(NC)"
	@docker compose down
	@echo "$(GREEN)Services stopped$(NC)"

docker-compose-logs: ## View docker-compose logs
	@docker compose logs -f

# DATABASE MIGRATIONS
migration: ## Create new database migration
	@echo "$(BLUE)Creating new migration...$(NC)"
	@read -p "Enter migration message: " message; \
	$(VENV_BIN)/alembic revision --autogenerate -m "$$message"

migrate: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(NC)"
	@$(VENV_BIN)/alembic upgrade head
	@echo "$(GREEN)Migrations complete$(NC)"

migrate-down: ## Rollback last migration
	@echo "$(BLUE)Rolling back last migration...$(NC)"
	@$(VENV_BIN)/alembic downgrade -1

# UTILITIES
clean: ## Clean up temporary files and caches
	@echo "$(BLUE)Cleaning up...$(NC)"
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
	@rm -rf .pytest_cache/
	@rm -rf htmlcov/
	@rm -rf .coverage
	@echo "$(GREEN)Cleanup complete$(NC)"

setup-env: ## Create .env file from template
	@echo "$(BLUE)Setting up environment file...$(NC)"
	@if [ ! -f .env ]; then \
		echo "$(YELLOW).env file already exists$(NC)"; \
	else \
		echo "$(GREEN).env file already configured$(NC)"; \
	fi

generate-secret: ## Generate a secure secret key
	@echo "$(BLUE)Generating secure secret key...$(NC)"
	@$(VENV_BIN)/python -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(32))"

check-services: ## Check if required services are running
	@echo "$(BLUE)Checking required services...$(NC)"
	@echo "PostgreSQL:"
	@pg_isready -h localhost -p 5432 && echo "$(GREEN)✓ PostgreSQL is running$(NC)" || echo "$(RED)✗ PostgreSQL is not running$(NC)"
	@echo "Redis:"
	@redis-cli ping > /dev/null 2>&1 && echo "$(GREEN)✓ Redis is running$(NC)" || echo "$(RED)✗ Redis is not running$(NC)"

# FULL SETUP
setup: install setup-postgres setup-redis init-db ## Complete setup (install deps, setup services, init db)
	@echo "$(GREEN)Complete setup finished!$(NC)"
	@echo "$(YELLOW)Don't forget to:$(NC)"
	@echo "$(YELLOW)1. Update your .env file with actual API keys$(NC)"
	@echo "$(YELLOW)2. Run 'make check-services' to verify services$(NC)"
	@echo "$(YELLOW)3. Run 'make dev' to start the development server$(NC)"

quick-setup: install ## Quick setup for development (no external services)
	@echo "$(BLUE)Setting up for development with SQLite...$(NC)"
	@mkdir -p uploads logs vector_indexes
	@echo "$(GREEN)Quick setup complete!$(NC)"
	@echo "$(YELLOW)Using SQLite for development. Run 'make test-app' to start.$(NC)"

# MONITORING
logs: ## View application logs
	@tail -f logs/app.log

health: ## Check application health
	@curl -s http://localhost:8000/health | python -m json.tool || echo "$(RED)Application not running$(NC)"

# DEPLOYMENT
deploy-check: ## Check if ready for deployment
	@echo "$(BLUE)Checking deployment readiness...$(NC)"
	@$(VENV_BIN)/python -c "from src.infrastructure.config.settings import get_settings; s = get_settings(); print('✓ Configuration loaded')"
	@echo "$(GREEN)Deployment check complete$(NC)"

# DEVELOPMENT SHORTCUTS
all: setup dev ## Full setup and start development server

restart: docker-compose-down docker-compose-up ## Restart all services

fresh-start: clean setup dev ## Clean, setup, and start fresh
