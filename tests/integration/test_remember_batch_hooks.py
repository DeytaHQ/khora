"""#1401: remember_batch() must dispatch entity/relationship hooks like remember().

The single-document path (`_run_skeleton_extraction`) dispatches an
`entity.created`/`entity.updated` hook per upserted entity and a
`relationship.created`/`relationship.updated` hook per edge. The streaming
batch path (`_remember_batch_impl`) discarded the `upsert_entities_batch` /
`create_relationships_batch` return values and dispatched nothing, so every
`remember_batch()` ingest (including the daemon's `POST /ingest`) silently
emitted zero semantic hooks.

Hermetic: sqlite_lance + the deterministic `stub_llm` extractor/embedder (no
network, no API key). A marker in the document content yields a fixed set of
entities + a typed relationship, so the hook counts are exact.
"""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False

from khora.core.models.event import MemoryEvent
from tests.test_helpers.filter_spy import EMBED_DIM, plan_extraction, stub_llm

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed"),
]

_MARKER = "hook-batch-marker"
_ENTITIES = [("Marie Curie", "PERSON"), ("Radium", "ELEMENT"), ("Polonium", "ELEMENT")]
_RELATIONSHIPS = [("Marie Curie", "Radium", "DISCOVERED"), ("Marie Curie", "Polonium", "DISCOVERED")]
# Must exceed VectorCypherConfig.min_extraction_tokens (50) so the streaming
# batch path's per-document token gate sends the chunk to extraction; the stub
# matches on _MARKER regardless of the surrounding filler.
_DOC = (
    f"{_MARKER}. "
    "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity. "
    "Working in Paris with her husband Pierre Curie she discovered the elements radium and polonium. "
    "She was the first woman to win a Nobel Prize and remains the only person to win Nobel Prizes "
    "in two different scientific fields, physics and chemistry. Her notebooks are still radioactive "
    "today and her discoveries laid the groundwork for modern nuclear physics and cancer radiotherapy."
)


def _config(tmp_path: Path):
    from khora.config import KhoraConfig
    from khora.config.schema import SQLiteLanceConfig

    config = KhoraConfig()
    config.storage.backend = "sqlite_lance"
    config.storage.sqlite_lance = SQLiteLanceConfig(
        db_path=str(tmp_path / "k.db"),
        lance_path=str(tmp_path / "k.lance"),
        embedding_dimension=EMBED_DIM,
    )
    config.storage.embedding_dimension = EMBED_DIM
    config.llm.embedding_dimension = EMBED_DIM
    # Every chunk goes to the (stub) extractor so the entity set is exact.
    config.pipelines.selective_extraction = False
    return config


async def _run(tmp_path: Path, *, batch: bool) -> tuple[int, int]:
    from khora import Khora

    entity_hooks: list[MemoryEvent] = []
    rel_hooks: list[MemoryEvent] = []

    async def on_entity(event: MemoryEvent) -> None:
        entity_hooks.append(event)

    async def on_rel(event: MemoryEvent) -> None:
        rel_hooks.append(event)

    async with Khora(_config(tmp_path), run_migrations=True) as kb:
        ns = await kb.create_namespace()
        kb.subscribe("entity.created", on_entity)
        kb.subscribe("relationship.created", on_rel)
        kwargs = dict(
            namespace=ns.namespace_id,
            entity_types=["PERSON", "ELEMENT"],
            relationship_types=["DISCOVERED"],
        )
        if batch:
            await kb.remember_batch([{"content": _DOC}], **kwargs)
        else:
            await kb.remember(_DOC, **kwargs)

    return len(entity_hooks), len(rel_hooks)


async def test_remember_batch_fires_entity_and_relationship_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1401: the batch path fires a hook per upserted entity and relationship."""
    stub_llm(monkeypatch, dim=EMBED_DIM)
    plan_extraction(_MARKER, entities=_ENTITIES, relationships=_RELATIONSHIPS)

    ent_cb, rel_cb = await _run(tmp_path, batch=True)

    assert ent_cb == len(_ENTITIES), f"expected {len(_ENTITIES)} entity.created hooks, got {ent_cb}"
    assert rel_cb >= len(_RELATIONSHIPS), f"expected >= {len(_RELATIONSHIPS)} relationship.created hooks, got {rel_cb}"


async def test_batch_and_single_doc_hook_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """remember_batch() fires the same entity hooks as remember() for the same doc (#1401)."""
    stub_llm(monkeypatch, dim=EMBED_DIM)
    plan_extraction(_MARKER, entities=_ENTITIES, relationships=_RELATIONSHIPS)

    single_dir = tmp_path / "single"
    batch_dir = tmp_path / "batch"
    single_dir.mkdir()
    batch_dir.mkdir()
    single_ent, single_rel = await _run(single_dir, batch=False)
    batch_ent, batch_rel = await _run(batch_dir, batch=True)

    assert single_ent > 0, "control: single-doc path must fire entity hooks"
    assert batch_ent == single_ent, f"batch entity hooks ({batch_ent}) must match single-doc ({single_ent})"
    assert batch_rel == single_rel, f"batch relationship hooks ({batch_rel}) must match single-doc ({single_rel})"


async def test_batch_entity_hooks_dispatch_concurrently(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """#1471: the batch path gathers the per-entity hook dispatches so they run
    concurrently. With a callback that yields the event loop, all entity
    callbacks must have started before any finishes (impossible under a serial
    ``await`` loop, where each callback runs to completion before the next
    starts)."""
    import asyncio

    stub_llm(monkeypatch, dim=EMBED_DIM)
    plan_extraction(_MARKER, entities=_ENTITIES, relationships=_RELATIONSHIPS)

    from khora import Khora

    started = 0
    max_concurrent = 0
    delivered = 0

    async def on_entity(event: MemoryEvent) -> None:
        nonlocal started, max_concurrent, delivered
        started += 1
        max_concurrent = max(max_concurrent, started)
        # Yield so sibling dispatches interleave when gathered.
        await asyncio.sleep(0.02)
        started -= 1
        delivered += 1

    async with Khora(_config(tmp_path), run_migrations=True) as kb:
        ns = await kb.create_namespace()
        kb.subscribe("entity.created", on_entity)
        await kb.remember_batch(
            [{"content": _DOC}],
            namespace=ns.namespace_id,
            entity_types=["PERSON", "ELEMENT"],
            relationship_types=["DISCOVERED"],
        )

    assert delivered == len(_ENTITIES), f"every entity hook must be delivered, got {delivered}"
    assert max_concurrent > 1, (
        f"entity hook dispatches must run concurrently (peak in-flight={max_concurrent}); "
        "a serial await loop would peak at 1"
    )


async def test_batch_skips_dispatch_when_unsubscribed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """#1471: with no subscribers the batch path must not call dispatch_hook at
    all (skips MemoryEvent construction), and ingest must still succeed."""
    stub_llm(monkeypatch, dim=EMBED_DIM)
    plan_extraction(_MARKER, entities=_ENTITIES, relationships=_RELATIONSHIPS)

    from khora import Khora
    from khora.storage.coordinator import StorageCoordinator

    dispatch_calls = 0
    orig_dispatch = StorageCoordinator.dispatch_hook

    async def _counting_dispatch(self: StorageCoordinator, event: object) -> None:
        nonlocal dispatch_calls
        dispatch_calls += 1
        await orig_dispatch(self, event)

    monkeypatch.setattr(StorageCoordinator, "dispatch_hook", _counting_dispatch)

    async with Khora(_config(tmp_path), run_migrations=True) as kb:
        ns = await kb.create_namespace()
        # No kb.subscribe(...) calls -> zero subscribers.
        result = await kb.remember_batch(
            [{"content": _DOC}],
            namespace=ns.namespace_id,
            entity_types=["PERSON", "ELEMENT"],
            relationship_types=["DISCOVERED"],
        )

    assert result.processed == 1, "ingest must still succeed with no subscribers"
    assert dispatch_calls == 0, f"no subscribers -> dispatch_hook must not be called, got {dispatch_calls}"
