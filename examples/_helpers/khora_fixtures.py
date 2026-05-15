"""Zero-infrastructure ``Khora`` fixture for the integration examples.

Yields a ``Khora`` instance bound to a ``sqlite_lance`` backend in a
temporary directory. The SQLite file is migrated via Alembic on entry
and the temp dir is removed on exit. No Postgres, no Neo4j, no Docker.
"""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from khora import Khora
from khora.config.schema import KhoraConfig, SQLiteLanceConfig


@asynccontextmanager
async def embedded_khora(
    *,
    embedding_dimension: int = 1536,
    engine: str = "vectorcypher",
) -> AsyncIterator[Khora]:
    """Async context manager yielding a connected ``Khora`` on sqlite_lance.

    Args:
        embedding_dimension: embedding vector dimension. Must match the
            value the mock LLM was installed with. Default 1536.
        engine: khora engine name. Default ``"vectorcypher"``.

    Example::

        from examples._helpers import embedded_khora, install_mock_llm

        install_mock_llm()
        async with embedded_khora() as kb:
            ns = await kb.create_namespace()
            await kb.remember("hello world", namespace=ns.namespace_id)
            result = await kb.recall("hello", namespace=ns.namespace_id)
    """
    with tempfile.TemporaryDirectory(prefix="khora-example-") as tmp:
        tmp_path = Path(tmp)
        config = KhoraConfig()
        config.storage.backend = "sqlite_lance"
        config.storage.sqlite_lance = SQLiteLanceConfig(
            db_path=str(tmp_path / "khora.db"),
            lance_path=str(tmp_path / "khora.lance"),
            embedding_dimension=embedding_dimension,
        )
        config.llm.embedding_dimension = embedding_dimension

        kb = Khora(config, engine=engine, run_migrations=True)
        await kb.connect()
        try:
            yield kb
        finally:
            await kb.disconnect()
