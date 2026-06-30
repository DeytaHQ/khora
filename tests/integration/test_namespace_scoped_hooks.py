"""#1399: namespace-scoped hook subscriptions must fire on ingest events.

``kb.subscribe("entity.created", cb, namespace_id=ns.namespace_id)`` registers
the scope with the *stable* namespace_id that ``create_namespace`` returns.
Ingest, however, resolves that stable id to the active version's *row* id
before extraction (``remember()`` -> ``_resolve_namespace``), so every emitted
``MemoryEvent`` carries the row id. Before #1399 the dispatcher compared the
two raw and they never matched, so scoped subscriptions silently fired zero
callbacks.

This test drives the REAL ``Khora`` dispatcher wiring on the embedded
sqlite_lance stack (no LLM, no external services): it dispatches a synthetic
``entity.created`` event built with the resolved row id through the exact path
ingest uses (``storage.dispatch_hook``) and asserts the stable-scoped callback
fires. The unit-level mechanism is covered in ``tests/unit/test_hooks.py``;
this guards the end-to-end resolver wiring.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False

from khora.core.models.event import MemoryEvent

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed"),
]


async def test_namespace_scoped_subscription_fires_on_resolved_row_id(tmp_path: Path) -> None:
    from khora import Khora
    from khora.config import KhoraConfig
    from khora.config.schema import SQLiteLanceConfig

    config = KhoraConfig()
    config.storage.backend = "sqlite_lance"
    config.storage.sqlite_lance = SQLiteLanceConfig(
        db_path=str(tmp_path / "k.db"),
        lance_path=str(tmp_path / "k.lance"),
        embedding_dimension=32,
    )
    config.storage.embedding_dimension = 32
    config.llm.embedding_dimension = 32

    async with Khora(config, run_migrations=True) as kb:
        ns = await kb.create_namespace()

        # Anti-vacuity: the stable namespace_id and the resolved row id really
        # differ — otherwise this test would pass even with the bug present.
        row_id = await kb.storage.resolve_namespace(ns.namespace_id)
        assert row_id != ns.namespace_id, "expected the stable namespace_id to differ from the row id"

        fired: list[MemoryEvent] = []

        async def cb(event: MemoryEvent) -> None:
            fired.append(event)

        # Subscribe with the STABLE id (what a caller gets from create_namespace).
        kb.subscribe("entity.created", cb, namespace_id=ns.namespace_id)

        # Ingest emits with the RESOLVED ROW id — dispatch via the exact path.
        await kb.storage.dispatch_hook(
            MemoryEvent.entity_created(namespace_id=row_id, entity_id=uuid4(), data={"name": "Marie Curie"})
        )
        assert len(fired) == 1, "stable-scoped subscription must fire on the row-id event (#1399)"

        # A foreign namespace's event must still be rejected.
        fired.clear()
        await kb.storage.dispatch_hook(
            MemoryEvent.entity_created(namespace_id=uuid4(), entity_id=uuid4(), data={"name": "elsewhere"})
        )
        assert fired == [], "events from another namespace must not reach a scoped subscription"
