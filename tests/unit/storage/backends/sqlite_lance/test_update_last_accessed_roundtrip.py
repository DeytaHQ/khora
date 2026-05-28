"""SQL round-trip test for ``update_last_accessed`` on sqlite_lance (#855).

The two engine-level reinforcement tests use a recording coordinator -
they verify the helper is *called*, not that the UPDATE *lands*. This
test exercises the actual SQL: insert two chunks, update one, read both
back, assert the timestamps round-trip correctly through ``_dt_to_str``
+ ``_parse_dt`` and that the namespace scope filters the other chunk out.

Covers the Devil's Advocate concern that ``_dt_to_str`` could strip UTC
or truncate microseconds without anyone noticing - the engine path never
reads ``last_accessed_at`` back, so without this test the field could
silently lose timezone info and the decay-uses-max-timestamp test (which
constructs synthetic timestamps in memory) would still pass.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

try:
    import lancedb  # noqa: F401
    import pyarrow  # noqa: F401

    _HAS_LANCEDB = True
except ImportError:
    _HAS_LANCEDB = False

from khora.core.models import Chunk

pytestmark = pytest.mark.skipif(not _HAS_LANCEDB, reason="lancedb not installed")

if _HAS_LANCEDB:
    from khora.storage.backends.sqlite_lance.connection import (
        EmbeddedStorageHandle,
        EmbeddedStorageHandleConfig,
    )
    from khora.storage.backends.sqlite_lance.vector import SQLiteLanceVectorAdapter


async def _build_handle(db_path: Path, lance_path: Path) -> EmbeddedStorageHandle:
    cfg = EmbeddedStorageHandleConfig(
        db_path=str(db_path),
        lance_path=str(lance_path),
        embedding_dimension=8,
    )
    h = EmbeddedStorageHandle(cfg)
    await h.connect()
    # Skip FK checks - we don't seed memory_namespaces / documents here.
    await h.sqlite.execute("PRAGMA foreign_keys = OFF")
    await h.sqlite.commit()
    return h


@pytest.fixture
async def adapter(migrated_sqlite_db: Path, tmp_path: Path):
    h = await _build_handle(migrated_sqlite_db, tmp_path / "k.lance")
    try:
        yield SQLiteLanceVectorAdapter(h)
    finally:
        await h.disconnect()


def _make_chunk(ns, doc, *, content: str, index: int) -> Chunk:
    return Chunk(
        id=uuid4(),
        namespace_id=ns,
        document_id=doc,
        content=content,
        chunk_index=index,
        chunker_info={},
        embedding=None,
        embedding_model="",
        created_at=datetime.now(UTC),
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_update_last_accessed_roundtrips_via_sqlite(
    adapter: SQLiteLanceVectorAdapter,
) -> None:
    """UPDATE lands; selected chunk has tz-aware ts, sibling stays NULL.

    Catches three potential round-trip bugs at once:
      1) ``_dt_to_str`` stripping UTC ``tzinfo``.
      2) ``_parse_dt`` returning a naive datetime.
      3) The UPDATE missing the namespace scope and stamping every chunk.
    """
    ns, doc = uuid4(), uuid4()
    c1 = _make_chunk(ns, doc, content="chunk one", index=0)
    c2 = _make_chunk(ns, doc, content="chunk two", index=1)
    await adapter.create_chunks_batch([c1, c2])

    # Sanity: both rows start with last_accessed_at = NULL.
    fetched1_before = await adapter.get_chunk(c1.id, namespace_id=ns)
    fetched2_before = await adapter.get_chunk(c2.id, namespace_id=ns)
    assert fetched1_before is not None and fetched1_before.last_accessed_at is None
    assert fetched2_before is not None and fetched2_before.last_accessed_at is None

    # Stamp only c1.
    ts = datetime.now(UTC)
    rowcount = await adapter.update_last_accessed(ns, [c1.id], ts)
    assert rowcount == 1

    # c1 must round-trip with UTC tzinfo intact and within ~1ms of the
    # original ts (ISO format preserves microseconds).
    fetched1_after = await adapter.get_chunk(c1.id, namespace_id=ns)
    assert fetched1_after is not None
    got_ts = fetched1_after.last_accessed_at
    assert got_ts is not None
    assert got_ts.tzinfo is not None, "last_accessed_at lost timezone info on round-trip"
    assert abs((got_ts - ts).total_seconds()) < 1e-3

    # c2 must still be NULL - the UPDATE's namespace + id scope didn't
    # over-stamp sibling chunks.
    fetched2_after = await adapter.get_chunk(c2.id, namespace_id=ns)
    assert fetched2_after is not None
    assert fetched2_after.last_accessed_at is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_update_last_accessed_rejects_cross_namespace_writes(
    adapter: SQLiteLanceVectorAdapter,
) -> None:
    """A chunk id from ns_a passed with ns_b must NOT be stamped.

    Defense against forged-id IDOR through the reinforcement path.
    """
    ns_a, ns_b, doc = uuid4(), uuid4(), uuid4()
    chunk_a = _make_chunk(ns_a, doc, content="ns a chunk", index=0)
    await adapter.create_chunks_batch([chunk_a])

    ts = datetime.now(UTC)
    rowcount = await adapter.update_last_accessed(ns_b, [chunk_a.id], ts)
    assert rowcount == 0

    fetched = await adapter.get_chunk(chunk_a.id, namespace_id=ns_a)
    assert fetched is not None
    assert fetched.last_accessed_at is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_update_last_accessed_empty_list_is_noop(
    adapter: SQLiteLanceVectorAdapter,
) -> None:
    """Empty chunk_ids short-circuits without an SQL roundtrip."""
    rowcount = await adapter.update_last_accessed(uuid4(), [], datetime.now(UTC))
    assert rowcount == 0
