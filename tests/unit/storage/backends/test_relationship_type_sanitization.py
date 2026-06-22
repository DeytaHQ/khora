"""Cross-backend regression tests for relationship_type sanitization (issue #749).

Before #749 the Cypher backends (Neo4j, Memgraph, Neptune, AGE, sqlite_lance)
silently UPPER_SNAKE_CASEd ``Relationship.relationship_type`` while SurrealDB
stored the raw user string verbatim.  Feeding ``"lives in"`` to both and
reading it back returned ``"LIVES_IN"`` from one and ``"lives in"`` from the
other — same input, two semantically different values.

This module asserts that every graph backend now:

1. Normalises the user-supplied ``relationship_type`` through the shared
   :func:`sanitize_cypher_label` helper before persisting it; and
2. Mirrors the sanitised form back onto the caller's :class:`Relationship`
   so the in-memory object matches what is on disk.

Real backend round-trips (assert ``get_relationship(...).relationship_type ==
"LIVES_IN"``) live in the integration suite — these unit tests stay backend-
independent by mocking the driver / connection and inspecting the SQL the
adapter would emit.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.core.models import Relationship
from khora.storage.backends.mixins import sanitize_cypher_label

# ---------------------------------------------------------------------------
# Sanity: the shared helper still does what every backend now agrees on.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSanitizerContract:
    """Pin the cross-backend contract.  Every backend's ``create_relationship``
    path must funnel ``relationship_type`` through this exact mapping."""

    def test_lives_in_is_canonical(self) -> None:
        assert sanitize_cypher_label("lives in") == "LIVES_IN"

    def test_mixed_case_uppercased(self) -> None:
        assert sanitize_cypher_label("Knows") == "KNOWS"

    def test_already_canonical_unchanged(self) -> None:
        assert sanitize_cypher_label("WORKS_FOR") == "WORKS_FOR"

    def test_punctuation_replaced(self) -> None:
        assert sanitize_cypher_label("at-risk!") == "AT_RISK_"


# ---------------------------------------------------------------------------
# Per-backend assertions: caller's object is rewritten in place.
# ---------------------------------------------------------------------------


def _make_rel(rel_type: str = "lives in") -> Relationship:
    return Relationship(
        namespace_id=uuid4(),
        source_entity_id=uuid4(),
        target_entity_id=uuid4(),
        relationship_type=rel_type,
    )


@pytest.mark.unit
async def test_sqlite_lance_create_relationship_mirrors_sanitized_type() -> None:
    """sqlite_lance was already mirroring; pinned here so the behaviour does
    not silently regress."""
    pytest.importorskip("aiosqlite")
    pytest.importorskip("lancedb")
    from khora.storage.backends.sqlite_lance.graph import SQLiteLanceGraphAdapter

    handle = MagicMock()
    sqlite_conn = MagicMock()
    sqlite_conn.execute = AsyncMock(return_value=MagicMock())
    sqlite_conn.commit = AsyncMock()
    handle.sqlite = sqlite_conn

    adapter = SQLiteLanceGraphAdapter(handle)
    rel = _make_rel()
    out = await adapter.create_relationship(rel)
    assert out is rel
    assert rel.relationship_type == "LIVES_IN"


@pytest.mark.unit
async def test_neo4j_create_relationship_mirrors_sanitized_type() -> None:
    pytest.importorskip("neo4j")
    from khora.storage.backends.neo4j import Neo4jBackend

    backend = Neo4jBackend.__new__(Neo4jBackend)
    # Stub out the session context manager and write transaction.
    fake_session = MagicMock()
    fake_session.__aenter__ = AsyncMock(return_value=fake_session)
    fake_session.__aexit__ = AsyncMock(return_value=None)
    fake_session.execute_write = AsyncMock(
        return_value={"doc_dropped": 0, "chunk_dropped": 0, "doc_rows": 0, "chunk_rows": 0}
    )
    backend._session = MagicMock(return_value=fake_session)  # type: ignore[attr-defined]
    backend._relationship_source_document_ids_max = 100  # type: ignore[attr-defined]
    backend._relationship_source_chunk_ids_max = 250  # type: ignore[attr-defined]

    rel = _make_rel()
    await backend.create_relationship(rel)
    assert rel.relationship_type == "LIVES_IN"


@pytest.mark.unit
async def test_neo4j_create_relationships_batch_mirrors_sanitized_type() -> None:
    pytest.importorskip("neo4j")
    from khora.storage.backends.neo4j import Neo4jBackend

    backend = Neo4jBackend.__new__(Neo4jBackend)
    fake_session = MagicMock()
    fake_session.__aenter__ = AsyncMock(return_value=fake_session)
    fake_session.__aexit__ = AsyncMock(return_value=None)
    fake_session.execute_write = AsyncMock(
        return_value={
            "created": 1,
            "doc_dropped": 0,
            "chunk_dropped": 0,
            "doc_rows": 0,
            "chunk_rows": 0,
            "edge_rows": [],
        }
    )
    backend._session = MagicMock(return_value=fake_session)  # type: ignore[attr-defined]
    # _ensure_relationship_type_indexes touches the live driver — short-circuit.
    backend._ensure_relationship_type_indexes = AsyncMock(return_value=None)  # type: ignore[attr-defined]
    # The batch path uses an asyncio.Semaphore — provide a real one.
    import asyncio

    backend._relationship_write_sem = asyncio.Semaphore(8)  # type: ignore[attr-defined]
    backend._driver = MagicMock()  # type: ignore[attr-defined]
    backend._relationship_source_document_ids_max = 100  # type: ignore[attr-defined]
    backend._relationship_source_chunk_ids_max = 250  # type: ignore[attr-defined]

    rels = [_make_rel("lives in"), _make_rel("works AT")]
    await backend.create_relationships_batch(rels)
    assert rels[0].relationship_type == "LIVES_IN"
    assert rels[1].relationship_type == "WORKS_AT"


@pytest.mark.unit
async def test_memgraph_create_relationship_mirrors_sanitized_type() -> None:
    pytest.importorskip("neo4j")
    from khora.storage.backends.memgraph import MemgraphBackend

    backend = MemgraphBackend.__new__(MemgraphBackend)
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    session.run = AsyncMock(return_value=MagicMock())
    driver = MagicMock()
    driver.session = MagicMock(return_value=session)
    backend._get_driver = MagicMock(return_value=driver)  # type: ignore[attr-defined]

    rel = _make_rel()
    await backend.create_relationship(rel)
    assert rel.relationship_type == "LIVES_IN"


@pytest.mark.unit
async def test_neptune_create_relationship_mirrors_sanitized_type() -> None:
    pytest.importorskip("neo4j")
    from khora.storage.backends.neptune import NeptuneBackend

    backend = NeptuneBackend.__new__(NeptuneBackend)
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    session.run = AsyncMock(return_value=MagicMock())
    driver = MagicMock()
    driver.session = MagicMock(return_value=session)
    backend._get_driver = MagicMock(return_value=driver)  # type: ignore[attr-defined]

    rel = _make_rel()
    await backend.create_relationship(rel)
    assert rel.relationship_type == "LIVES_IN"


@pytest.mark.unit
async def test_age_create_relationship_mirrors_sanitized_type() -> None:
    """AGE previously kept the user case (``reports_to`` not ``REPORTS_TO``).
    Post-#749 it joins the rest of the family."""
    pytest.importorskip("sqlalchemy")
    from contextlib import asynccontextmanager

    from khora.storage.backends.age import AGEBackend

    backend = AGEBackend("postgresql://localhost/test")

    result = MagicMock()
    result.fetchall = MagicMock(return_value=[])
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)

    @asynccontextmanager
    async def _begin():  # type: ignore[no-untyped-def]
        yield None

    session.begin = MagicMock(side_effect=_begin)

    @asynccontextmanager
    async def _session_ctx():  # type: ignore[no-untyped-def]
        yield session

    backend._session_factory = MagicMock(side_effect=_session_ctx)  # type: ignore[assignment]

    rel = _make_rel()
    await backend.create_relationship(rel)
    assert rel.relationship_type == "LIVES_IN"


@pytest.mark.unit
async def test_surrealdb_create_relationship_mirrors_sanitized_type() -> None:
    """The regression case from the user's bug report.  Before #749 SurrealDB
    stored ``"lives in"`` verbatim; everyone else stored ``"LIVES_IN"``."""
    pytest.importorskip("surrealdb")
    from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

    conn = MagicMock()
    conn.connected = True
    conn.execute = AsyncMock(return_value=None)
    adapter = SurrealDBGraphAdapter(conn)

    rel = _make_rel()
    out = await adapter.create_relationship(rel)
    assert out is rel
    assert rel.relationship_type == "LIVES_IN"


@pytest.mark.unit
async def test_surrealdb_create_relationships_batch_mirrors_sanitized_type() -> None:
    pytest.importorskip("surrealdb")
    from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

    conn = MagicMock()
    conn.connected = True
    conn.execute = AsyncMock(return_value=None)
    adapter = SurrealDBGraphAdapter(conn)

    rels = [_make_rel("lives in"), _make_rel("works AT")]
    results = await adapter.create_relationships_batch(rels)
    # #1320: bare RELATE always creates - best-effort is_new=True per edge.
    assert results == [(rels[0], True), (rels[1], True)]
    assert rels[0].relationship_type == "LIVES_IN"
    assert rels[1].relationship_type == "WORKS_AT"
