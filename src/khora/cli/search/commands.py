"""Implementation of khora search command."""

from __future__ import annotations

import asyncio
import sys
import time
from typing import Any
from uuid import UUID

import click

from khora.cli._common import (
    EXIT_CONFIG_ERROR,
    EXIT_CONNECTION_ERROR,
    EXIT_SUCCESS,
    detect_output_format,
    resolve_lake_kwargs,
    write_json,
)


@click.command(name="search")
@click.argument("query", required=False)
@click.option("--database-url", envvar="KHORA_DATABASE_URL", help="Database URL")
@click.option("--graph-url", envvar="KHORA_NEO4J_URL", help="Neo4j/graph URL")
@click.option("--config", "config_path", type=click.Path(exists=True), envvar="KHORA_CONFIG_PATH", help="YAML config")
@click.option("-n", "--namespace", required=True, envvar="KHORA_NAMESPACE", help="Namespace UUID to search")
@click.option("-l", "--limit", default=10, type=int, help="Max results")
@click.option("--mode", default="hybrid", type=click.Choice(["vector", "graph", "hybrid", "all"]))
@click.option("-e", "--engine", default="vectorcypher", help="Engine")
@click.option("-m", "--model", default=None, envvar="KHORA_LLM_MODEL", help="LLM model")
@click.option("--raw", is_flag=True, help="No HyDE, no reranking, no query expansion")
@click.option("--min-similarity", default=0.0, type=float)
@click.option("--fields", default="chunks,entities,context", help="Output fields (comma-separated)")
@click.option("--format", "output_format", default=None, type=click.Choice(["json", "text"]))
@click.option("-v", "--verbose", is_flag=True)
def search(
    query,
    database_url,
    graph_url,
    config_path,
    namespace,
    limit,
    mode,
    engine,
    model,
    raw,
    min_similarity,
    fields,
    output_format,
    verbose,
):
    """Search the Khora knowledge graph.

    QUERY from argument or stdin.

    \b
    Examples:
      khora search "Who worked on the API design?" -n <namespace-uuid>
      echo "quarterly revenue" | khora search -n <uuid> --raw
      khora search "team changes" -n <uuid> --mode vector --limit 5
      khora search "auth flow" -n <uuid> --format json | jq '.chunks'
    """
    fmt = detect_output_format(output_format)

    if not query:
        if sys.stdin.isatty():
            if fmt == "json":
                write_json({"status": "error", "error": "No query provided"})
            else:
                click.echo("Error: Provide a query as argument or pipe to stdin.", err=True)
            sys.exit(EXIT_CONFIG_ERROR)
        query = sys.stdin.read().strip()

    if not query:
        if fmt == "json":
            write_json({"status": "error", "error": "Empty query"})
        else:
            click.echo("Error: Empty query.", err=True)
        sys.exit(EXIT_CONFIG_ERROR)

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

    # Parse namespace as UUID
    try:
        ns_id = UUID(namespace)
    except ValueError:
        if fmt == "json":
            write_json({"status": "error", "error": f"Invalid namespace UUID: {namespace}"})
        else:
            click.echo(f"Error: Invalid namespace UUID: {namespace}", err=True)
        sys.exit(EXIT_CONFIG_ERROR)

    exit_code = asyncio.run(
        _search_async(
            lake_kwargs=lake_kwargs,
            query=query,
            ns_id=ns_id,
            limit=limit,
            mode=mode,
            raw=raw,
            min_similarity=min_similarity,
            fields=fields,
            fmt=fmt,
            verbose=verbose,
        )
    )
    sys.exit(exit_code)


async def _search_async(
    *,
    lake_kwargs: dict[str, Any],
    query: str,
    ns_id: UUID,
    limit: int,
    mode: str,
    raw: bool,
    min_similarity: float,
    fields: str,
    fmt: str,
    verbose: bool,
) -> int:
    from khora import MemoryLake, SearchMode

    mode_map = {
        "vector": SearchMode.VECTOR,
        "graph": SearchMode.GRAPH,
        "hybrid": SearchMode.HYBRID,
        "all": SearchMode.ALL,
    }
    search_mode = mode_map[mode]
    field_set = set(fields.split(","))

    t0 = time.monotonic()

    try:
        async with MemoryLake(**lake_kwargs) as lake:
            result = await lake.recall(
                query,
                namespace=ns_id,
                limit=limit,
                mode=search_mode,
                raw=raw,
                min_similarity=min_similarity,
            )
    except Exception as e:
        if fmt == "json":
            write_json({"status": "error", "error": f"Connection error: {e}"})
        else:
            click.echo(f"Error: {e}", err=True)
        return EXIT_CONNECTION_ERROR

    duration_ms = int((time.monotonic() - t0) * 1000)

    output: dict[str, Any] = {
        "query": query,
        "mode": mode,
        "result_count": len(result.chunks),
        "duration_ms": duration_ms,
    }

    if "chunks" in field_set:
        output["chunks"] = [
            {
                "id": str(chunk.id),
                "content": chunk.content,
                "score": round(score, 4),
                **({"document_id": str(chunk.document_id)} if verbose else {}),
            }
            for chunk, score in result.chunks
        ]

    if "entities" in field_set:
        output["entities"] = [
            {
                "id": str(entity.id),
                "name": entity.name,
                "type": entity.entity_type,
                "score": round(score, 4),
            }
            for entity, score in result.entities
        ]

    if "context" in field_set:
        output["context_text"] = result.context_text

    if fmt == "json":
        write_json(output)
    else:
        click.echo(f"Found {len(result.chunks)} result(s) ({duration_ms}ms)")
        if result.context_text:
            click.echo(f"\n{result.context_text}")

    return EXIT_SUCCESS
