# Khora Makefile
#
# Provides shortcuts for common development and deployment tasks.
#
# Usage:
#   make help              # Show all available commands
#   make dev               # Start development environment
#   make test              # Run tests with coverage

.PHONY: help dev dev-down db-up db-down test test-unit test-embedded test-integration test-soak lint format typecheck prek clean \
        rust-build rust-dev rust-test rust-bench rust-clean \
        docker-run docker-down docker-clean

# Load local environment overrides (.env is git-ignored; silently skipped in CI)
-include .env
export KHORA_DATABASE_URL
export KHORA_NEO4J_URL

# Stop ``uv run`` from re-syncing the venv on every invocation. The
# ``crewai`` and ``google-adk`` extras are declared as conflicts in
# pyproject.toml (they pin incompatible opentelemetry-api ranges), which
# splits the uv.lock into two resolution-marker branches. A bare
# ``uv run`` re-resolves against the lockfile defaults and silently swaps
# the installed otel-sdk between 1.34 and 1.40, breaking logfire's
# in-tree imports. ``uv sync --all-extras --no-extra <one>`` pins the
# combo correctly; UV_NO_SYNC keeps subsequent ``uv run`` calls from
# undoing it. Mirrored in .github/workflows/ci.yml.
export UV_NO_SYNC := 1

# Default target
help:
	@echo "Khora Development Commands"
	@echo "=========================="
	@echo ""
	@echo "Development:"
	@echo "  make install          Sync venv with all extras (crewai combo by default)"
	@echo "  make install-adk      Sync venv with the google-adk extras combo"
	@echo "  make dev              Start databases (postgres + neo4j) [alias: db-up]"
	@echo "  make dev-down         Stop databases [alias: db-down]"
	@echo "  make test             Run tests with coverage (unit parallel + integration serial)"
	@echo "  make test-unit        Run unit tests in parallel (-n auto)"
	@echo "  make test-embedded    Run SQLite+LanceDB embedded-stack tests (no Docker)"
	@echo "  make test-integration Run integration tests serially (needs make dev)"
	@echo "  make test-soak        Run long-running soak/burn-in tests"
	@echo "  make lint             Run linting (ruff, ty)"
	@echo "  make typecheck        Run type checking (ty)"
	@echo "  make format           Format code (ruff)"
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
	@echo "  make docker-run       Run full stack (databases)"
	@echo "  make docker-down      Stop full stack"
	@echo "  make docker-clean     Remove images and volumes"
	@echo ""
	@echo "Connection Strings (for .env):"
	@echo "  KHORA_DATABASE_URL=postgresql://khora:khora@localhost:5434/khora"
	@echo "  KHORA_NEO4J_URL=bolt://neo4j:pleaseletmein@localhost:7688"

# ==============================================================================
# Development Commands
# ==============================================================================

# Sync the venv with the dev dependencies + all adapter extras.
# Bare ``uv sync --all-extras`` is rejected by uv because the ``crewai`` and
# ``google-adk`` extras have an opentelemetry-api conflict declared in
# pyproject.toml's ``[tool.uv].conflicts`` (crewai pins <1.35, google-adk
# pins >=1.36). The default ``install`` target picks the crewai combo, which
# matches the CI ``test`` job. Use ``install-adk`` for the google-adk combo.
install:
	uv sync --all-extras --no-extra google-adk

install-adk:
	uv sync --all-extras --no-extra crewai

# Start local development databases
dev:
	docker compose up -d
	@echo ""
	@echo "Waiting for services to be healthy..."
	@sleep 3
	@docker compose ps
	@echo ""
	@echo "PostgreSQL: localhost:5434 (user: khora, pass: khora)"
	@echo "Neo4j:      http://localhost:7475 (user: neo4j, pass: pleaseletmein)"
	@echo ""
	@echo "Add to .env:"
	@echo "  KHORA_DATABASE_URL=postgresql://khora:khora@localhost:5434/khora"
	@echo ""
	@echo "Run migrations: uv run alembic upgrade head"

# Stop local development databases
dev-down:
	docker compose down

# Aliases for dev / dev-down
db-up: dev
db-down: dev-down

# Run unit tests in parallel (xdist) with coverage; integration tests stay serial
# because tests/integration/matrix/* fixtures DROP SCHEMA on shared PostgreSQL.
test: test-unit test-integration

# Run unit tests parallel; clear report so coverage is finalized by test-integration.
# tests/recall/ holds the recall-filter suite (validator, AST, compiler-dispatch) and
# is fully hermetic — no Postgres needed; collect it here so the filter tests gate.
# The live-pg row-narrowing proof lives in the filter-conformance corpus (its own job).
test-unit:
	uv run pytest tests/unit/ tests/recall/ tests/security/ --cov=src/khora --cov-branch --cov-report= --cov-fail-under=0 -n auto

# Run integration tests serial; appends to .coverage from test-unit and emits the report.
test-integration:
	uv run pytest tests/integration/ --cov=src/khora --cov-branch --cov-append --cov-report=term-missing --cov-fail-under=77 -m integration

# Run SQLite+LanceDB embedded-stack tests only (no Docker required).
# Useful for fast feedback on the embedded path without spinning up Postgres/Neo4j.
test-embedded:
	uv run pytest -m embedded -v

# Run long-running soak/burn-in tests. Coverage is disabled because these tests
# already run long and we don't want coverage instrumentation skewing timing.
test-soak:
	uv run pytest -m soak --no-cov

# Run type checking
typecheck:
	uv run ty check src/

# Run linting
lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run ty check src/
	uv run python tools/check_optional_imports.py

# Format code
format:
	uv run ruff format .
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
