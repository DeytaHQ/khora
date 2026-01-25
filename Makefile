# Khora Makefile
#
# Provides shortcuts for common development and deployment tasks.
#
# Usage:
#   make help              # Show all available commands
#   make dev               # Start development environment
#   make test              # Run tests with coverage

.PHONY: help dev dev-down test lint format prek clean \
        docker-build docker-run docker-clean

# Default target
help:
	@echo "Khora Development Commands"
	@echo "=========================="
	@echo ""
	@echo "Development:"
	@echo "  make dev              Start local dev environment (databases only)"
	@echo "  make dev-down         Stop local dev environment"
	@echo "  make test             Run tests with coverage"
	@echo "  make lint             Run linting (ruff)"
	@echo "  make format           Format code (black, isort)"
	@echo "  make prek             Run prek hooks"
	@echo "  make clean            Clean build artifacts"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     Build production Docker image"
	@echo "  make docker-run       Run full stack with docker-compose"
	@echo "  make docker-clean     Remove Docker images and volumes"
	@echo ""
	@echo "Environment Variables:"
	@echo "  KHORA_PORT            API port (default: 8000)"

# ==============================================================================
# Development Commands
# ==============================================================================

# Start local development databases
dev:
	docker compose -f docker-compose.dev.yml up -d postgres
	@echo ""
	@echo "Services started. Waiting for health checks..."
	@sleep 5
	docker compose -f docker-compose.dev.yml ps
	@echo ""
	@echo "PostgreSQL: localhost:5433"
	@echo ""
	@echo "Start API:     uv run khora serve --reload"

# Stop local development databases
dev-down:
	docker compose -f docker-compose.dev.yml down

# Run tests with coverage
test:
	uv run pytest --cov=src/khora --cov-branch --cov-report=term-missing --cov-fail-under=30

# Run linting
lint:
	uv run ruff check .
	uv run black --check .
	uv run isort --check .

# Format code
format:
	uv run black .
	uv run isort .
	uv run ruff check --fix .

# Run prek hooks
prek:
	uv run prek run --all-files

# Clean build artifacts
clean:
	rm -rf .pytest_cache .coverage htmlcov .ruff_cache
	rm -rf src/*.egg-info build dist
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ==============================================================================
# Docker Commands
# ==============================================================================

# Build production Docker image
docker-build:
	docker build \
		-t khora:latest \
		-t ghcr.io/$(shell git remote get-url origin | sed 's/.*github.com[:/]\(.*\)\.git/\1/' | tr '[:upper:]' '[:lower:]'):latest \
		.

# Run full stack with docker-compose (production-like)
docker-run:
	docker compose up -d

# Clean Docker resources
docker-clean:
	docker compose down -v --remove-orphans
	docker compose -f docker-compose.dev.yml down -v --remove-orphans 2>/dev/null || true
	docker rmi khora:latest 2>/dev/null || true
	docker volume prune -f
