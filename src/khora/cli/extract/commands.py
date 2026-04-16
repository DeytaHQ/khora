"""Implementation of khora extract command."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import click
from loguru import logger

from khora.cli._common import (
    EXIT_CONFIG_ERROR,
    EXIT_CONNECTION_ERROR,
    EXIT_PARTIAL_FAILURE,
    EXIT_SUCCESS,
    detect_output_format,
    resolve_lake_kwargs,
    resolve_namespace,
    write_json,
)


def _read_file_content(path: Path) -> tuple[str, str, dict[str, Any]]:
    """Read a file and return (content, title, metadata)."""
    title = path.stem
    metadata: dict[str, Any] = {"source_path": str(path), "format": path.suffix.lstrip(".")}

    binary_formats = {".pdf", ".xlsx", ".xls", ".docx", ".parquet"}
    if path.suffix.lower() in binary_formats:
        from khora.extraction.binary_readers import extract_if_needed

        extracted = extract_if_needed(path)
        if extracted:
            content = extracted.read_text(encoding="utf-8", errors="replace")
        else:
            content = ""
            logger.warning("Could not extract text from {}", path.name)
        return content, title, metadata

    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to read {}: {}", path.name, e)
        content = ""

    return content, title, metadata


def _collect_sources(sources: tuple[str, ...]) -> list[Path]:
    """Expand source arguments into file paths."""
    paths: list[Path] = []
    for src in sources:
        if src == "-":
            continue
        p = Path(src)
        if p.is_file():
            paths.append(p)
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and not f.name.startswith("."):
                    paths.append(f)
        else:
            logger.warning("Source not found: {}", src)
    return paths


@click.command(name="extract")
@click.argument("sources", nargs=-1)
@click.option("--database-url", envvar="KHORA_DATABASE_URL", help="Database URL")
@click.option("--graph-url", envvar="KHORA_NEO4J_URL", help="Neo4j/graph URL")
@click.option("--config", "config_path", type=click.Path(exists=True), envvar="KHORA_CONFIG_PATH", help="YAML config")
@click.option("-n", "--namespace", default=None, help="Namespace UUID (auto-created if omitted)")
@click.option("-e", "--engine", default="vectorcypher", help="Engine [default: vectorcypher]")
@click.option("-m", "--model", default=None, envvar="KHORA_LLM_MODEL", help="LLM model")
@click.option("--expertise", type=click.Path(exists=True), help="Ontology/expertise YAML")
@click.option(
    "--entity-types", default="PERSON,ORGANIZATION,CONCEPT,LOCATION,EVENT", help="Comma-separated entity types"
)
@click.option("--relationship-types", default=None, help="Comma-separated relationship types (auto if omitted)")
@click.option(
    "--chunk-strategy", default="semantic", type=click.Choice(["fixed", "semantic", "recursive", "conversation"])
)
@click.option("--chunk-size", default=512, type=int, help="Tokens per chunk")
@click.option("--batch-size", default=50, type=int, help="Docs per batch")
@click.option("--dry-run", is_flag=True, help="Show stats without extracting")
@click.option("--format", "output_format", default=None, type=click.Choice(["json", "text"]))
@click.option("--progress", is_flag=True, help="Emit JSONL progress to stderr")
@click.option("-v", "--verbose", is_flag=True)
def extract(
    sources,
    database_url,
    graph_url,
    config_path,
    namespace,
    engine,
    model,
    expertise,
    entity_types,
    relationship_types,
    chunk_strategy,
    chunk_size,
    batch_size,
    dry_run,
    output_format,
    progress,
    verbose,
):
    """Ingest files into the Khora knowledge graph.

    SOURCES: file paths, directory paths, or "-" for stdin.

    \b
    Examples:
      khora extract report.pdf
      khora extract ./docs/ --namespace <uuid>
      cat data.json | khora extract -
      khora extract *.csv --entity-types PERSON,ORGANIZATION
    """
    fmt = detect_output_format(output_format)

    try:
        lake_kwargs = resolve_lake_kwargs(
            config_path=config_path,
            database_url=database_url,
            graph_url=graph_url,
            model=model,
            engine=engine,
        )
    except Exception as e:
        if fmt == "json":
            write_json({"status": "error", "error": f"Config error: {e}"})
        else:
            click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_CONFIG_ERROR)

    e_types = [t.strip() for t in entity_types.split(",")]
    r_types = [t.strip() for t in relationship_types.split(",")] if relationship_types else []

    expertise_config = None
    if expertise:
        import yaml

        from khora.extraction.skills.base import ExpertiseConfig

        with open(expertise) as f:
            data = yaml.safe_load(f)
        expertise_config = ExpertiseConfig.model_validate(data)

    files = _collect_sources(sources)
    stdin_content = None
    if "-" in sources or (not sources and not sys.stdin.isatty()):
        stdin_content = sys.stdin.read()

    if not files and not stdin_content:
        if fmt == "json":
            write_json({"status": "error", "error": "No input sources provided"})
        else:
            click.echo("Error: No input sources. Provide file paths or pipe stdin.", err=True)
        sys.exit(EXIT_CONFIG_ERROR)

    if dry_run:
        result = {
            "status": "dry_run",
            "files": len(files),
            "stdin": bool(stdin_content),
            "engine": lake_kwargs.get("engine", "vectorcypher"),
            "entity_types": e_types,
            "chunk_strategy": chunk_strategy,
            "chunk_size": chunk_size,
        }
        if fmt == "json":
            write_json(result)
        else:
            click.echo(
                f"Dry run: {len(files)} file(s), engine={lake_kwargs.get('engine', 'vectorcypher')}, entities={e_types}"
            )
        sys.exit(EXIT_SUCCESS)

    exit_code = asyncio.run(
        _extract_async(
            lake_kwargs=lake_kwargs,
            files=files,
            stdin_content=stdin_content,
            namespace_arg=namespace,
            e_types=e_types,
            r_types=r_types,
            expertise_config=expertise_config,
            chunk_strategy=chunk_strategy,
            fmt=fmt,
            show_progress=progress,
        )
    )
    sys.exit(exit_code)


async def _extract_async(
    *,
    lake_kwargs: dict[str, Any],
    files: list[Path],
    stdin_content: str | None,
    namespace_arg: str | None,
    e_types: list[str],
    r_types: list[str],
    expertise_config: Any,
    chunk_strategy: str,
    fmt: str,
    show_progress: bool,
) -> int:
    from khora import MemoryLake

    t0 = time.monotonic()
    total_docs = 0
    total_chunks = 0
    total_entities = 0
    total_rels = 0
    errors: list[dict[str, str]] = []

    try:
        async with MemoryLake(**lake_kwargs, run_migrations=True) as lake:
            ns_id = await resolve_namespace(lake, namespace_arg)

            remember_base: dict[str, Any] = {
                "namespace": ns_id,
                "entity_types": e_types,
                "relationship_types": r_types,
                "chunk_strategy": chunk_strategy,
            }
            if expertise_config:
                remember_base["expertise"] = expertise_config

            # stdin
            if stdin_content:
                try:
                    result = await lake.remember(stdin_content, **remember_base)
                    total_docs += 1
                    total_chunks += result.chunks_created
                    total_entities += result.entities_extracted
                    total_rels += result.relationships_created
                    if show_progress:
                        _progress({"event": "document", "source": "stdin", "chunks": result.chunks_created})
                except Exception as e:
                    errors.append({"source": "stdin", "error": str(e)})

            # files
            for fp in files:
                content, title, metadata = _read_file_content(fp)
                if not content:
                    errors.append({"source": str(fp), "error": "Empty or unreadable"})
                    continue
                try:
                    result = await lake.remember(content, title=title, metadata=metadata, **remember_base)
                    total_docs += 1
                    total_chunks += result.chunks_created
                    total_entities += result.entities_extracted
                    total_rels += result.relationships_created
                    if show_progress:
                        _progress({"event": "document", "source": str(fp), "chunks": result.chunks_created})
                except Exception as e:
                    errors.append({"source": str(fp), "error": str(e)})

    except Exception as e:
        if fmt == "json":
            write_json({"status": "error", "error": f"Connection error: {e}"})
        else:
            click.echo(f"Connection error: {e}", err=True)
        return EXIT_CONNECTION_ERROR

    duration_ms = int((time.monotonic() - t0) * 1000)
    status = "success" if not errors else ("partial_failure" if total_docs > 0 else "error")

    output = {
        "status": status,
        "namespace_id": str(ns_id),
        "documents": total_docs,
        "chunks": total_chunks,
        "entities": total_entities,
        "relationships": total_rels,
        "duration_ms": duration_ms,
    }
    if errors:
        output["errors"] = errors

    if fmt == "json":
        write_json(output)
    else:
        click.echo(
            f"Extracted {total_docs} doc(s): {total_chunks} chunks, "
            f"{total_entities} entities, {total_rels} relationships ({duration_ms}ms)"
        )
        for err in errors:
            click.echo(f"  FAIL: {err['source']}: {err['error']}", err=True)

    return EXIT_SUCCESS if not errors else EXIT_PARTIAL_FAILURE


def _progress(data: dict[str, Any]) -> None:
    """Emit a JSONL progress event to stderr."""
    print(json.dumps(data, default=str), file=sys.stderr)
