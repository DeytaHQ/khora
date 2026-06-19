"""End-to-end reachability of chronicle ``fact_compaction`` via ``kb.dream`` (#1067).

Two coupled defects this exercises against the real ``sqlite_lance`` stack
(production adapter + mock LLM, no external infra):

1.  **Reachability.** ``CHRONICLE_FACT_COMPACTION`` was absent from the
    chronicle plugin's ``dream_capabilities`` and never emitted from
    ``plan_dream``, so ``kb.dream(mode="apply", ops.compact_facts=True)``
    never planned it and tombstoned ``memory_facts`` were never reclaimed.
    An explicit ``scope=DreamScope(op_kinds=[CHRONICLE_FACT_COMPACTION])``
    was dropped as ``op_not_supported_by_engine``.

2.  **UUID bind mismatch.** ``fact_compaction._bind_uuid`` bound dashed
    UUID strings, but the ``sqlite_lance`` store writes 32-char hex
    (SQLAlchemy ``Uuid(as_uuid=True)`` on SQLite). Dashed != hex -> the
    ``WHERE namespace_id``/``WHERE id`` clauses matched 0 rows, so the
    planner selected nothing and apply deleted nothing.

The fixtures seed rows through the production relational adapter
(``write_facts`` -> hex storage), NOT via hand-rolled ``str(uuid)``
inserts, so they catch the bind mismatch the old unit fixtures missed.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest
import sqlalchemy as sa

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples._helpers import embedded_khora, install_mock_llm  # noqa: E402
from khora.dream.config import DreamConfig, DreamOpsConfig  # noqa: E402
from khora.dream.plan import DreamScope, OpKind  # noqa: E402
from khora.engines.chronicle.compression import MemoryFact  # noqa: E402

pytestmark = pytest.mark.embedded


async def _seed_tombstoned_fact(kb, namespace_id: UUID, *, age_days: int = 400) -> UUID:
    """Insert one tombstoned ``memory_facts`` row via the production adapter.

    Writes through ``write_facts`` (hex-UUID storage, exactly what the
    sqlite_lance adapter does in production) under the *resolved* row-level
    namespace id - the form ``Khora.remember`` persists facts under (the
    ``memory_facts.namespace_id`` FK references ``memory_namespaces.id``).
    Then backdates ``updated_at`` and flips ``is_active`` to false so the
    row is past the retention floor. Returns the fact id.
    """
    resolved_ns = await kb.storage.resolve_namespace(namespace_id)
    fact = MemoryFact(
        namespace_id=resolved_ns,
        subject="Alice",
        predicate="lives_in",
        object_="Paris",
        fact_text="Alice lives in Paris",
        confidence=0.9,
        is_active=True,
    )
    ids = await kb.storage.write_facts([fact], namespace_id=resolved_ns)
    fact_id = ids[0]

    old = (datetime.now(UTC) - timedelta(days=age_days)).isoformat()
    async with kb.storage.transaction() as txn:
        # Bind the id as hex - the column was written hex by write_facts.
        await txn.session.execute(
            sa.text("UPDATE memory_facts SET is_active = 0, updated_at = :old WHERE id = :id"),
            {"old": old, "id": fact_id.hex},
        )
    return fact_id


async def _fact_present(kb, namespace_id: UUID, fact_id: UUID) -> bool:
    async with kb.storage.transaction() as txn:
        row = (
            await txn.session.execute(
                sa.text("SELECT 1 FROM memory_facts WHERE id = :id"),
                {"id": fact_id.hex},
            )
        ).first()
    return row is not None


async def test_default_apply_reclaims_tombstoned_fact() -> None:
    """``kb.dream(mode="apply", ops.compact_facts=True)`` hard-deletes the row.

    Fails on main: the op is never planned (capability gap) and even if it
    were, the UUID bind mismatch matches 0 rows on sqlite_lance.
    """
    install_mock_llm(dim=8)
    async with embedded_khora(embedding_dimension=8, engine="chronicle") as kb:
        ns = await kb.create_namespace()
        fact_id = await _seed_tombstoned_fact(kb, ns.namespace_id)
        assert await _fact_present(kb, ns.namespace_id, fact_id)

        config = DreamConfig(enabled=True, ops=DreamOpsConfig(compact_facts=True))
        result = await kb.dream(ns.namespace_id, mode="apply", config=config)

        op_types = {op.op_type for op in result.ops}
        assert "chronicle_fact_compaction" in op_types, f"fact_compaction never planned; ops={op_types}"
        assert not await _fact_present(kb, ns.namespace_id, fact_id), "tombstoned fact was not reclaimed"


async def test_compact_facts_off_leaves_fact_intact() -> None:
    """With ``compact_facts`` off the tombstoned row is left alone (config gate)."""
    install_mock_llm(dim=8)
    async with embedded_khora(embedding_dimension=8, engine="chronicle") as kb:
        ns = await kb.create_namespace()
        fact_id = await _seed_tombstoned_fact(kb, ns.namespace_id)

        config = DreamConfig(enabled=True, ops=DreamOpsConfig(compact_facts=False))
        result = await kb.dream(ns.namespace_id, mode="apply", config=config)

        op_types = {op.op_type for op in result.ops}
        assert "chronicle_fact_compaction" not in op_types
        assert await _fact_present(kb, ns.namespace_id, fact_id)


async def test_explicit_scope_no_longer_dropped() -> None:
    """An explicit op-scope request is no longer ``op_not_supported_by_engine``."""
    install_mock_llm(dim=8)
    async with embedded_khora(embedding_dimension=8, engine="chronicle") as kb:
        ns = await kb.create_namespace()
        fact_id = await _seed_tombstoned_fact(kb, ns.namespace_id)

        config = DreamConfig(enabled=True, ops=DreamOpsConfig(compact_facts=True))
        result = await kb.dream(
            ns.namespace_id,
            mode="apply",
            scope=DreamScope(op_kinds=(OpKind.CHRONICLE_FACT_COMPACTION,)),
            config=config,
        )

        skip_reasons = result.metadata.get("skip_reasons", [])
        dropped = [
            r
            for r in skip_reasons
            if r.get("op_kind") == "chronicle_fact_compaction" and r.get("reason") == "op_not_supported_by_engine"
        ]
        assert not dropped, f"op was dropped: {skip_reasons}"
        assert not await _fact_present(kb, ns.namespace_id, fact_id)
