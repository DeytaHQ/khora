"""Demonstrate KHORA_NEO4J_LOG_LEVEL=DEBUG surfacing driver-internal logs.

Run with::

    uv run python examples/neo4j_debug_logging.py

The script sets ``KHORA_NEO4J_LOG_LEVEL=DEBUG`` in-process, initializes
khora's logging, and then constructs a ``Neo4jBackend`` pointing at the
local compose stack. The driver emits DEBUG lines as soon as a connection
is attempted (TLS handshake, pool bootstrap, routing). Those lines are
routed through khora's ``InterceptHandler`` into loguru and printed to
stdout.

If no Neo4j is reachable the connect call raises — that's fine; the DEBUG
lines produced before the failure are the point of the demo.
"""

from __future__ import annotations

import asyncio
import os

# Must be set BEFORE setup_logging() / Neo4jBackend() so the helper sees it.
os.environ.setdefault("KHORA_NEO4J_LOG_LEVEL", "DEBUG")

from khora.logging_config import setup_logging  # noqa: E402
from khora.storage.backends.neo4j import Neo4jBackend  # noqa: E402


async def main() -> None:
    setup_logging(level="DEBUG")

    backend = Neo4jBackend(
        url=os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7688"),
        user=os.environ.get("KHORA_NEO4J_USER", "neo4j"),
        password=os.environ.get("KHORA_NEO4J_PASSWORD", "pleaseletmein"),  # noqa: S106
    )

    try:
        await backend.connect()
        print("Connected to Neo4j; driver DEBUG logs should appear above.")
    except Exception as exc:  # noqa: BLE001 - demo script, surface anything
        print(f"Connect failed ({type(exc).__name__}: {exc}).")
        print("Any 'neo4j' DEBUG lines above were produced before the failure.")
    finally:
        await backend.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
