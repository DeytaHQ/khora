# Khora

Deyta's memory lake and materialization of knowledge.

## Overview

Khora is an async FastAPI service for managing knowledge artifacts and materializing data transformations. It uses SQLAlchemy with async PostgreSQL support and Alembic for migrations.

## Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) for package management
- PostgreSQL (for production)

### Installation

```bash
# Install dependencies
uv sync --all-extras

# Run pre-commit hooks setup
uv run prek install
```

### Development

```bash
# Start the server with hot-reload
uv run khora serve --reload --no-auth

# Run tests
make test

# Format code
make format

# Run linting
make lint

# Run all pre-commit hooks
make prek
```

### API Endpoints

- `GET /status` - Service status check
- `GET /health` - Health check for load balancers
- `GET /health/ready` - Readiness check for orchestration
- `GET /health/live` - Liveness check for orchestration

## Project Structure

```
khora/
├── src/khora/           # Main application source
│   ├── api/             # FastAPI application
│   │   ├── app.py       # App factory
│   │   └── routes/      # API endpoints
│   ├── cli/             # Click CLI commands
│   ├── config/          # Configuration
│   ├── db/              # Database layer
│   └── logging_config.py
├── tests/               # Test suite
├── alembic/             # Database migrations
├── config/              # Configuration files
└── pyproject.toml       # Project configuration
```

## Configuration

Configuration is loaded from environment variables with the `KHORA_` prefix:

| Variable | Description | Default |
|----------|-------------|---------|
| `KHORA_DATABASE_URL` | PostgreSQL connection URL | - |
| `KHORA_DEBUG` | Enable debug mode | `false` |
| `KHORA_API_HOST` | API server host | `127.0.0.1` |
| `KHORA_API_PORT` | API server port | `8000` |
| `KHORA_AUTH_ENABLED` | Enable authentication | `true` |

## Database Migrations

```bash
# Run migrations
KHORA_DATABASE_URL=postgresql://... uv run alembic upgrade head

# Create a new migration
KHORA_DATABASE_URL=postgresql://... uv run alembic revision --autogenerate -m "description"
```

## License

Copyright (c) 2024 Deyta. All rights reserved.
