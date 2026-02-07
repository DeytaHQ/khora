# Khora Makefile
#
# Provides shortcuts for common development and deployment tasks.
#
# Usage:
#   make help              # Show all available commands
#   make dev               # Start development environment
#   make test              # Run tests with coverage

.PHONY: help dev dev-down serve test lint format typecheck prek clean \
        rust-build rust-dev rust-test rust-bench rust-clean \
        docker-build docker-run docker-down docker-clean

# Default target
help:
	@echo "Khora Development Commands"
	@echo "=========================="
	@echo ""
	@echo "Development:"
	@echo "  make dev              Start databases (postgres + neo4j)"
	@echo "  make dev-down         Stop databases"
	@echo "  make serve            Start API with hot-reload (requires databases)"
	@echo "  make test             Run tests with coverage"
	@echo "  make lint             Run linting (ruff, black, isort, ty)"
	@echo "  make typecheck        Run type checking (ty)"
	@echo "  make format           Format code (black, isort, ruff)"
	@echo "  make prek             Run pre-commit hooks"
	@echo "  make clean            Clean build artifacts"
	@echo ""
	@echo "Rust Acceleration:"
	@echo "  make rust-build       Build Rust extension in release mode"
	@echo "  make rust-dev         Build and install Rust extension for development"
	@echo "  make rust-test        Run Rust tests"
	@echo "  make rust-bench       Run Rust benchmarks"
	@echo "  make rust-clean       Clean Rust build artifacts"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     Build production Docker image"
	@echo "  make docker-run       Run full stack (api + databases)"
	@echo "  make docker-down      Stop full stack"
	@echo "  make docker-clean     Remove images and volumes"
	@echo ""
	@echo "Connection Strings (for .env):"
	@echo "  KHORA_DATABASE_URL=postgresql://khora:khora@localhost:5434/khora"
	@echo "  KHORA_NEO4J_URL=bolt://neo4j:khora@localhost:7688"

# ==============================================================================
# Development Commands
# ==============================================================================

# Start local development databases
dev:
	docker compose up -d
	@echo ""
	@echo "Waiting for services to be healthy..."
	@sleep 3
	@docker compose ps
	@echo ""
	@echo "PostgreSQL: localhost:5434 (user: khora, pass: khora)"
	@echo "Neo4j:      http://localhost:7475 (user: neo4j, pass: khora)"
	@echo ""
	@echo "Add to .env:"
	@echo "  KHORA_DATABASE_URL=postgresql://khora:khora@localhost:5434/khora"
	@echo ""
	@echo "Start API: make serve"

# Stop local development databases
dev-down:
	docker compose down

# Start API server with hot-reload
serve:
	uv run khora serve --reload --no-auth

# Run tests with coverage
test:
	uv run pytest --cov=src/khora --cov-branch --cov-report=term-missing --cov-fail-under=30

# Run type checking
typecheck:
	uv run ty check src/

# Run linting
lint:
	uv run ruff check .
	uv run black --check .
	uv run isort --check .
	uv run ty check src/

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
# Rust Acceleration Commands
# ==============================================================================

# Build Rust extension in release mode
rust-build:
	cd rust/khora-accel && maturin build --release

# Build and install Rust extension for development
rust-dev:
	cd rust/khora-accel && maturin develop --release

# Run Rust tests
rust-test:
	cd rust/khora-accel && cargo test

# Run Rust benchmarks
rust-bench:
	cd rust/khora-accel && cargo bench

# Clean Rust build artifacts
rust-clean:
	cd rust/khora-accel && cargo clean

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
	docker compose -f compose.full.yaml up -d

# Stop full stack
docker-down:
	docker compose -f compose.full.yaml down

# Clean Docker resources
docker-clean:
	docker compose down -v --remove-orphans 2>/dev/null || true
	docker compose -f compose.full.yaml down -v --remove-orphans 2>/dev/null || true
	docker rmi khora:latest 2>/dev/null || true
	docker volume prune -f
