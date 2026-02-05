"""FastAPI application factory for Khora."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from .routes import memory, namespaces, status, sync

if TYPE_CHECKING:
    from ..config import KhoraConfig


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all requests and responses."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        method = request.method
        path = request.url.path
        query = str(request.url.query) if request.url.query else ""
        client_host = request.client.host if request.client else "unknown"

        # Log incoming request with client info
        query_str = f"?{query}" if query else ""
        logger.info(f"-> {method} {path}{query_str} from {client_host}")

        try:
            response = await call_next(request)
            duration = (time.time() - start_time) * 1000

            # Log response with status code
            if response.status_code < 400:
                logger.info(f"<- {method} {path} - {response.status_code} ({duration:.1f}ms)")
            elif response.status_code < 500:
                logger.warning(f"<- {method} {path} - {response.status_code} ({duration:.1f}ms)")
            else:
                logger.error(f"<- {method} {path} - {response.status_code} ({duration:.1f}ms)")

            return response
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            logger.exception(f"<- {method} {path} - ERROR: {e} ({duration:.1f}ms)")
            raise


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Application lifespan manager for startup/shutdown events."""
    from ..db.session import close_db, run_migrations
    from ..memory_lake import MemoryLake
    from .deps import set_memory_lake

    # Startup
    logger.info("Starting Khora API server...")

    # Run database migrations
    await run_migrations()

    # Initialize Memory Lake
    config = app.state.config
    lake = MemoryLake(config=config)
    try:
        await lake.connect()
        set_memory_lake(lake)
        app.state.memory_lake = lake
        logger.info("Memory Lake initialized")
    except Exception as e:
        logger.warning(f"Memory Lake initialization failed (service will run with limited functionality): {e}")
        app.state.memory_lake = None

    yield

    # Shutdown
    logger.info("Shutting down Khora API server...")
    if hasattr(app.state, "memory_lake") and app.state.memory_lake:
        await app.state.memory_lake.disconnect()
    else:
        # If MemoryLake wasn't initialized, still shut down telemetry
        from ..telemetry import shutdown_telemetry

        await shutdown_telemetry()
    await close_db()


def create_app(config: KhoraConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Optional application configuration

    Returns:
        Configured FastAPI application
    """
    # Setup logging (important for reload mode where CLI setup doesn't carry over)
    from ..logging_config import setup_logging

    setup_logging(level="INFO")

    if config is None:
        from ..config import load_config

        config = load_config()

    app = FastAPI(
        title="Khora",
        description="Deyta's memory lake and materialization of knowledge",
        version="0.1.4",
        lifespan=lifespan,
        debug=config.debug,
    )

    # Store config in app state
    app.state.config = config

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if config.debug else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add request logging
    app.add_middleware(LoggingMiddleware)

    # Register routes
    # Status endpoint is public (no auth)
    app.include_router(status.router, tags=["status"])

    # Memory Lake API routes
    app.include_router(memory.router)
    app.include_router(namespaces.router)
    app.include_router(sync.router)

    return app
