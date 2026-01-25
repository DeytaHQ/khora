# Khora - Development Guide

Khora is Deyta's memory lake and materialization of knowledge. This is an async FastAPI service with SQLAlchemy/Alembic for database management.

## Quick Reference

### Commands
```bash
# Development
uv run khora serve --reload      # Start dev server with hot-reload
uv run khora serve --no-auth     # Start without authentication
make test                         # Run tests with coverage
make prek                         # Run pre-commit hooks
make format                       # Format code (black, isort, ruff)
make lint                         # Check linting

# Database
uv run alembic upgrade head       # Run migrations
uv run alembic revision --autogenerate -m "description"  # Create migration
```

### Project Structure
```
src/khora/
├── __init__.py          # Package init, exports main()
├── __main__.py          # Entry point for python -m khora
├── api/                 # FastAPI application
│   ├── app.py           # App factory with lifespan
│   └── routes/          # API endpoints
│       └── status.py    # /status endpoint
├── cli/                 # Click CLI commands
│   ├── __init__.py      # CLI group, main()
│   └── server.py        # serve command
├── config/              # Configuration
│   ├── __init__.py      # load_config()
│   └── schema.py        # Pydantic settings
├── db/                  # Database layer
│   ├── __init__.py      # Model exports
│   ├── models.py        # SQLAlchemy models
│   └── session.py       # Async session management
└── logging_config.py    # Loguru setup
```

## Architecture

### Async-First Design
- All I/O operations are async
- SQLAlchemy with asyncpg for PostgreSQL
- FastAPI with async routes
- AsyncSession for database operations

### Configuration
- Pydantic BaseSettings with `KHORA_` prefix
- Environment variables or YAML config file
- `KHORA_DATABASE_URL` for PostgreSQL connection

### Database
- Alembic for migrations (async)
- SQLAlchemy 2.0+ with async support
- Run migrations on startup in production

## Code Style

- Python 3.13+
- Line length: 120 characters
- Black for formatting
- isort with black profile
- ruff for linting
- Type hints throughout

## Testing

- pytest with pytest-asyncio
- Coverage minimum: 30%
- Markers: @pytest.mark.unit, @pytest.mark.integration, @pytest.mark.e2e
- Fixtures in tests/conftest.py

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| KHORA_DATABASE_URL | PostgreSQL connection URL | - |
| KHORA_DEBUG | Enable debug mode | false |
| KHORA_API_HOST | API server host | 127.0.0.1 |
| KHORA_API_PORT | API server port | 8000 |
| KHORA_AUTH_ENABLED | Enable authentication | true |
