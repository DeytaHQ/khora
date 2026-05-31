"""End-to-end repro of issue #716 against an in-memory SurrealDB instance.

After ``Khora.remember(...)`` succeeds with ``chunks_created=1`` on the
``skeleton`` or ``vectorcypher`` engine with ``backend=surrealdb``, the
following ``Khora.recall(...)`` returned 0 chunks — silent retrieval
failure.

Root cause: the skeleton-engine temporal store filtered chunks with a
SurrealQL *literal* ``memory_namespace:⟨{uuid}⟩`` which produces a
**string**-typed RecordID on the parser side, whereas the chunk rows
were written via the SDK's ``RecordID(table, uuid)`` constructor and
therefore carry a **UUID**-typed RecordID for the ``namespace`` field.
SurrealDB's equality is type-strict, so the filter never matched.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

pytest.importorskip("surrealdb")

from khora import Khora  # noqa: E402
from khora.config import KhoraConfig  # noqa: E402

pytestmark = [
    pytest.mark.integration,
    # Share one event loop across the whole module so the
    # ``_schema_init_lock`` inside ``SurrealDBConnection`` (a module-level
    # ``asyncio.Lock`` instantiated at import time) stays bound to a single
    # loop across the parametrized engine cases.
    pytest.mark.asyncio(loop_scope="module"),
]


_EMBED_DIM = 32
_KEYWORDS = {
    "pagerduty": 0,
    "payments": 1,
    "service": 2,
    "triggered": 3,
}


def _embed_for(text: str) -> list[float]:
    """Deterministic 32-dim unit vector keyed off a tiny keyword vocabulary."""
    vec = [0.0] * _EMBED_DIM
    vec[_EMBED_DIM - 1] = 0.01  # nonzero baseline so the all-miss case still has a vector
    lower = text.lower()
    for kw, slot in _KEYWORDS.items():
        if kw in lower:
            vec[slot] = 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


async def _stub_embed_batch(self: Any, texts: list[str]) -> list[list[float]]:
    return [_embed_for(t) for t in texts]


async def _stub_embed(self: Any, text: str) -> list[float]:
    return _embed_for(text)


@pytest.fixture(autouse=True)
def _patch_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "khora.extraction.embedders.litellm.LiteLLMEmbedder.embed_batch",
        _stub_embed_batch,
    )
    monkeypatch.setattr(
        "khora.extraction.embedders.litellm.LiteLLMEmbedder.embed",
        _stub_embed,
    )


@pytest.fixture
def _surrealdb_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure a fresh in-memory SurrealDB stack per test.

    Uses the canonical single-underscore env-var spelling (Issue #789).
    The legacy double-underscore spelling is regression-covered by
    ``test_recall_with_legacy_double_underscore_env_vars`` below.
    """
    monkeypatch.setenv("KHORA_STORAGE_BACKEND", "surrealdb")
    monkeypatch.setenv("KHORA_STORAGE_SURREALDB_MODE", "memory")


@pytest.fixture
def _surrealdb_env_legacy_double_underscore(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy double-underscore env-var spelling — kept for regression coverage."""
    monkeypatch.setenv("KHORA_STORAGE__BACKEND", "surrealdb")
    monkeypatch.setenv("KHORA_STORAGE__SURREALDB__MODE", "memory")


@pytest.mark.parametrize("engine", ["skeleton", "vectorcypher"])
async def test_recall_returns_chunks_after_remember_surrealdb_memory(
    engine: str,
    _surrealdb_env: None,
) -> None:
    """Issue #716: recall must return >=1 chunk after a successful remember.

    Both ``skeleton`` and ``vectorcypher`` engines on ``backend=surrealdb``
    in memory mode must round-trip a single document.  Pre-fix this returned
    0 chunks because the namespace filter on the temporal-chunk SELECT used
    a string-typed RecordID literal while the writer used a UUID-typed
    RecordID via the SDK.
    """
    config = KhoraConfig()
    config.storage.embedding_dimension = _EMBED_DIM
    config.llm.embedding_dimension = _EMBED_DIM

    async with Khora(config, engine=engine, run_migrations=False) as kb:
        ns = await kb.create_namespace()
        # NOTE: entity_types/relationship_types are deliberately not passed because the
        # skeleton parametrization rejects them (#890); vectorcypher does not need them
        # for the round-trip assertion.
        rem = await kb.remember(
            "PagerDuty triggered for the payments service.",
            namespace=ns.namespace_id,
        )
        assert rem.chunks_created >= 1, f"[{engine}] remember reported 0 chunks created — fixture pre-condition not met"

        recalled = await kb.recall(
            "payments service",
            namespace=ns.namespace_id,
        )

        assert len(recalled.chunks) >= 1, (
            f"[{engine}] recall returned 0 chunks after a successful remember "
            f"(chunks_created={rem.chunks_created}, namespace={ns.namespace_id}) — "
            f"issue #716 regression"
        )
        # Sanity: result must be from the same namespace we wrote into.
        assert recalled.namespace_id == ns.namespace_id, (
            f"[{engine}] recalled result has namespace_id={recalled.namespace_id} but expected {ns.namespace_id}"
        )


async def test_recall_namespace_isolation_surrealdb_memory(
    _surrealdb_env: None,
) -> None:
    """Two namespaces, queries don't cross-bleed (skeleton on SurrealDB).

    Guards against the obvious over-corrective fix: dropping the namespace
    filter entirely.  If that ever happens, this test fires immediately.
    """
    config = KhoraConfig()
    config.storage.embedding_dimension = _EMBED_DIM
    config.llm.embedding_dimension = _EMBED_DIM

    async with Khora(config, engine="skeleton", run_migrations=False) as kb:
        ns_a = (await kb.create_namespace()).namespace_id
        ns_b = (await kb.create_namespace()).namespace_id

        await kb.remember(
            "PagerDuty triggered for the payments service.",
            namespace=ns_a,
        )
        await kb.remember(
            "An unrelated note that does not mention the keyword.",
            namespace=ns_b,
        )

        result_a = await kb.recall("payments service", namespace=ns_a)
        result_b = await kb.recall("payments service", namespace=ns_b)

        # ns_a returns its own chunk; ns_b's irrelevant doc must not leak in.
        a_contents = " ".join(c.content for c in result_a.chunks)
        b_contents = " ".join(c.content for c in result_b.chunks)
        assert "payments" in a_contents.lower(), "ns_a should match its own document"
        assert "payments" not in b_contents.lower(), f"ns_b leaked content from ns_a: {b_contents!r}"


async def test_recall_with_legacy_double_underscore_env_vars(
    _surrealdb_env_legacy_double_underscore: None,
) -> None:
    """Legacy ``KHORA_STORAGE__BACKEND`` / ``KHORA_STORAGE__SURREALDB__MODE`` still work.

    Regression coverage for the back-compat alias preserved by #789 —
    existing operator ``.env`` files using the double-underscore form
    must keep configuring the in-memory SurrealDB stack correctly.
    """
    config = KhoraConfig()
    config.storage.embedding_dimension = _EMBED_DIM
    config.llm.embedding_dimension = _EMBED_DIM

    async with Khora(config, engine="skeleton", run_migrations=False) as kb:
        ns = await kb.create_namespace()
        rem = await kb.remember(
            "PagerDuty triggered for the payments service.",
            namespace=ns.namespace_id,
        )
        assert rem.chunks_created >= 1
