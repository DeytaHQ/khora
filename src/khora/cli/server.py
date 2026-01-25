"""Server command for Khora CLI."""

from __future__ import annotations

import os
from pathlib import Path

import click


@click.command(name="serve")
@click.option(
    "--host",
    default="127.0.0.1",
    help="Host to bind the server to",
)
@click.option(
    "--port",
    default=8000,
    type=int,
    help="Port to bind the server to",
)
@click.option(
    "--reload",
    is_flag=True,
    help="Enable auto-reload for development",
)
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file",
)
@click.option(
    "--no-auth",
    is_flag=True,
    help="Disable authentication (for local development)",
)
@click.pass_context
def serve(ctx: click.Context, host: str, port: int, reload: bool, config: Path | None, no_auth: bool) -> None:
    """Start the FastAPI server for API access.

    The server provides:
    - REST API for knowledge management
    - Health check endpoints
    """
    import uvicorn

    log_level = ctx.obj.get("log_level", "info").lower() if ctx.obj else "info"

    # Set auth enabled flag via environment for reload mode
    if no_auth:
        os.environ["KHORA_AUTH_ENABLED"] = "false"
        click.echo(click.style("Warning: ", fg="yellow") + "Authentication is disabled")

    if reload:
        # When reload is enabled, uvicorn needs an import string, not the app object
        uvicorn.run(
            "khora.api.app:create_app",
            host=host,
            port=port,
            reload=reload,
            factory=True,
            log_level=log_level,
        )
    else:
        # Without reload, we can pass the app object directly
        from ..api.app import create_app
        from ..config import load_config

        app_config = load_config(config) if config else load_config()

        # Override auth_enabled if --no-auth is set
        if no_auth:
            # Create a new config with auth disabled
            app_config = app_config.model_copy(update={"auth_enabled": False})

        app = create_app(app_config)

        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level=log_level,
        )
