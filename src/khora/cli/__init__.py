"""Command-line interface for Khora."""

from __future__ import annotations

from pathlib import Path

import click

from ..logging_config import setup_logging


@click.group()
@click.version_option(version="0.1.5")
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    help="Set logging level",
)
@click.option(
    "--json-logs",
    is_flag=True,
    help="Output logs in JSON format for structured logging",
)
@click.option(
    "--log-file",
    type=click.Path(path_type=Path),
    help="Write logs to file (in addition to console)",
)
@click.pass_context
def cli(ctx: click.Context, log_level: str, json_logs: bool, log_file: Path | None) -> None:
    """Khora - Deyta's memory lake and materialization of knowledge."""
    setup_logging(level=log_level.upper(), json_logs=json_logs, log_file=log_file)
    ctx.ensure_object(dict)
    ctx.obj["log_level"] = log_level
    ctx.obj["json_logs"] = json_logs


def _register_subcommands() -> None:
    """Register subcommand groups (called after cli is defined)."""
    from khora.cli.extract.commands import extract
    from khora.cli.search.commands import search

    cli.add_command(extract)
    cli.add_command(search)


_register_subcommands()


def main() -> None:
    """Main entry point."""
    cli()


__all__ = ["cli", "main"]
