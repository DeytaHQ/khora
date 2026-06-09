"""Full-stack integration tests for the recall-filter on Skeleton-pgvector.

INTERIM TEST — SUPERSEDED BY THE FILTER-CONFORMANCE SUITE. This is an interim
smoke test for the Skeleton-pgvector filter path. Per the project's
filter-verification strategy, this assertion reduces to the dedicated
filter-conformance corpus: its permanent home is a separate filter-conformance
CI job (``tests/integration/matrix/``, its own conformance marker, per-engine
databases) that is not built yet. That job is deliberately excluded from the
main test job so the conformance cases never double-run. This file lives in
``tests/recall/`` and is collected by the main test job's unit step, where it
SELF-SKIPS because that job provisions no Postgres — it gates locally via
``make dev`` only. That is acceptable for an interim smoke test; it is NOT the
permanent CI gate. When the filter-conformance suite lands, MIGRATE/REMOVE this
file so the assertion does not double-run against the conformance corpus. Do not
entrench it (no Postgres service should be added to the main test job for this
test).

Where ``tests/integration/test_compile_postgres_rowset.py`` exercises the
compiler against a hand-seeded temp table, this file drives the *public* API
end-to-end: real ``Khora.remember()`` writes through the Skeleton engine into a
real ``khora_chunks`` table, and ``Khora.recall(filter=...)`` must narrow the
result to exactly the in-scope rows. The engine→backend wiring threads
``filter_ast`` from the facade through the skeleton engine into the pgvector
store's WHERE predicate, so row filtering is effective end-to-end — these tests
pin that contract.

Three scenarios:

* S1 — row-set: 5 in-scope + 5 out-of-scope chunks, each out-of-scope row
  violating EXACTLY ONE of the three predicates (wrong ``source_name``;
  ``occurred_at`` too old; ``metadata.tag`` out of set / missing). A single
  three-predicate ``recall(filter=...)`` must return EXACTLY the 5 in-scope
  chunk ids. The per-row single-violation design proves each predicate bites
  independently.
* S2 — engine_info + validation: ``engine_info["filter"]`` reports the skeleton
  support row, and an invalid filter raises ``RecallFilterValidationError``
  through the ``recall()`` facade.
* S5 — backward-compat: the deprecated ``start_time``/``end_time`` shim filters
  rows end-to-end on the live lake (the unit suite covers the shim *mechanics*
  with a mock engine; this adds the live-row gap).

ENVIRONMENT: needs the Docker Compose Postgres from THIS repo (``make dev``,
port 5434, user/pass/db all ``khora``). The module skips cleanly when Postgres
is unreachable (e.g. the CI lint sandbox) and RUNS as the real gate in the CI
integration job. The public API is HARD-imported (not ``importorskip``) so a
broken import is a LOUD error, never a silent skip.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import pytest

# Hard import (NOT importorskip): the filter surface and the Khora façade are on
# the branch, so an import failure must be a LOUD test error — never a silent
# module skip. (This module still skips when Postgres is unreachable, via the
# pytestmark skipif below.)
from khora.config import KhoraConfig
from khora.filter import RecallFilterValidationError
from khora.khora import Khora
from khora.query import SearchMode

# Postgres backend supports only embedding_dimension=1536 (schema.py:1096-1100),
# so the stub embedder must emit 1536-dim vectors to match the deployed schema.
EMBED_DIM = 1536


# This repo's compose puts Postgres on 5434 (see compose.yaml). Honor an explicit
# override, else default to the compose port — never another project's container.
_DEFAULT_URL = "postgresql+asyncpg://khora:khora@localhost:5434/khora"


def _database_url() -> str:
    url = os.environ.get("KHORA_DATABASE_URL", _DEFAULT_URL)
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


def _pg_reachable() -> bool:
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(_database_url().replace("+asyncpg", ""))
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _pg_reachable(),
        # Not accidental missing coverage: this is an interim smoke test whose
        # permanent CI coverage is the dedicated filter-conformance job.
        # It self-skips without local Postgres (run `make dev`).
        reason=(
            "interim smoke test; permanent CI coverage is the dedicated "
            "filter-conformance job. Self-skips without local Postgres "
            "(run `make dev`)."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Deterministic embedder stub — no OPENAI_API_KEY required.
# ---------------------------------------------------------------------------
#
# Every text maps to the SAME 1536-dim unit vector, so query and all chunk
# embeddings are identical → cosine similarity is 1.0 for every row. Retrieval
# is therefore filter-bound, not similarity-bound: the WHERE predicate is the
# only thing that narrows the candidate set, which is exactly what S1 must
# isolate. Skeleton has no LLM extraction path, so only the embedder is stubbed.


def _unit_vector() -> list[float]:
    return [1.0] + [0.0] * (EMBED_DIM - 1)


async def _stub_embed_batch(self: Any, texts: list[str]) -> list[list[float]]:
    return [_unit_vector() for _ in texts]


async def _stub_embed(self: Any, text_in: str) -> list[float]:
    return _unit_vector()


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


# ---------------------------------------------------------------------------
# Skeleton-pgvector Khora fixture.
# ---------------------------------------------------------------------------


@pytest.fixture
async def kb() -> AsyncIterator[Khora]:
    """A connected Khora on the Skeleton engine over the compose Postgres.

    ``neo4j_url=None`` keeps it graph-less (skeleton skips the graph backend
    anyway); ``run_migrations=True`` materializes the schema, including
    ``khora_chunks``, on a fresh DB.
    """
    config = KhoraConfig(database_url=_database_url(), neo4j_url=None)
    instance = Khora(config, engine="skeleton", run_migrations=True)
    await instance.connect()
    try:
        yield instance
    finally:
        try:
            await instance.disconnect()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Seed corpus — 5 in-scope, 5 out-of-scope (one predicate violation each).
# ---------------------------------------------------------------------------
#
# The three-predicate filter under test:
#   source_name == "linear"  AND  occurred_at >= 2026-04-05
#   AND  metadata.tag IN {"urgent", "release"}
#
# Mapping onto khora_chunks columns (skeleton write-path, engine.py:457-477):
#   * source_name kwarg            -> khora_chunks.source_name
#   * metadata["occurred_at"] ISO  -> khora_chunks.occurred_at (engine.py:355-368)
#   * metadata pass-through        -> khora_chunks.metadata (so metadata.tag lands)
# Content is DISTINCT per row: skeleton dedupes by content checksum
# (engine.py:316-331), so identical content would collapse to one chunk.

_IN_BOUND = "2026-04-05T00:00:00Z"  # the occurred_at lower bound (inclusive)

# label -> (source_name, occurred_at ISO, metadata.tag | None)
_IN_SCOPE: dict[str, tuple[str, str, str | None]] = {
    # All three predicates satisfied. Boundary, mid, and future occurred_at;
    # both allowed tag values are represented.
    "in_boundary": ("linear", _IN_BOUND, "urgent"),  # exactly at the >= bound
    "in_recent": ("linear", "2026-05-01T12:00:00Z", "release"),
    "in_future": ("linear", "2026-06-01T00:00:00Z", "urgent"),
    "in_release": ("linear", "2026-04-10T00:00:00Z", "release"),
    "in_urgent": ("linear", "2026-04-20T09:30:00Z", "urgent"),
}

# Each out-of-scope row violates EXACTLY ONE predicate; every other field is
# in-scope, so a leak pins which predicate failed to bite.
_OUT_OF_SCOPE: dict[str, tuple[str | None, str, str | None]] = {
    # Predicate 1 (source_name) violated; date + tag are in-scope.
    "out_wrong_source": ("slack", "2026-05-15T00:00:00Z", "urgent"),
    # Predicate 2 (occurred_at) violated: one day before the inclusive bound.
    "out_too_old": ("linear", "2026-04-04T23:59:59Z", "release"),
    # Predicate 3 (tag) violated: tag present but outside the allowed set.
    "out_wrong_tag": ("linear", "2026-05-20T00:00:00Z", "backlog"),
    # Predicate 3 (tag) violated: tag key absent entirely.
    "out_missing_tag": ("linear", "2026-05-25T00:00:00Z", None),
    # Predicate 1 (source_name) violated via NULL: source_name omitted at write.
    "out_null_source": (None, "2026-05-30T00:00:00Z", "release"),
}

_RECALL_FILTER = {
    "source_name": "linear",
    "occurred_at": {"$gte": _IN_BOUND},
    "metadata.tag": {"$in": ["urgent", "release"]},
}


def _content_for(label: str) -> str:
    # Distinct, short (single-chunk) content per row, with a stable token so a
    # keyword channel would also surface it if a mode ever uses BM25.
    return f"chunk {label} alpha bravo charlie {hashlib.sha256(label.encode()).hexdigest()[:8]}"


async def _seed(kb: Khora, namespace_id: UUID) -> dict[str, UUID]:
    """Remember every row and return a ``label -> chunk_id`` map.

    Each ``remember`` produces exactly one chunk (content < chunk_size). The
    chunk id is read back from ``khora_chunks`` by ``document_id`` so the tests
    assert on real, server-assigned chunk ids rather than guessing them.
    """
    import sqlalchemy as sa
    from sqlalchemy.ext.asyncio import create_async_engine

    doc_to_label: dict[UUID, str] = {}
    for label, (source_name, occurred_at, tag) in {**_IN_SCOPE, **_OUT_OF_SCOPE}.items():
        metadata: dict[str, Any] = {"occurred_at": occurred_at}
        if tag is not None:
            metadata["tag"] = tag
        result = await kb.remember(
            content=_content_for(label),
            namespace=namespace_id,
            title=label,
            source_name=source_name,
            metadata=metadata,
            entity_types=[],  # skeleton refuses non-empty entity/relationship types
            relationship_types=[],
        )
        doc_to_label[result.document_id] = label

    # khora_chunks stores the row-level namespace id (not the stable public id).
    # Resolve before the direct SQL query.
    resolved_ns = await kb.storage.resolve_namespace(namespace_id)

    # Resolve each document's single chunk id from the live khora_chunks table.
    engine = create_async_engine(_database_url())
    try:
        async with engine.connect() as conn:
            rows = (
                await conn.execute(
                    sa.text("SELECT id, document_id FROM khora_chunks WHERE namespace_id = :ns"),
                    {"ns": resolved_ns},
                )
            ).fetchall()
    finally:
        await engine.dispose()

    label_to_chunk: dict[str, UUID] = {}
    for chunk_id, document_id in rows:
        label_to_chunk[doc_to_label[document_id]] = chunk_id

    assert len(label_to_chunk) == len(doc_to_label), "each remembered doc must yield exactly one chunk"
    return label_to_chunk


def _ids_for(chunk_ids: dict[str, UUID], labels: dict[str, Any]) -> set[UUID]:
    return {chunk_ids[label] for label in labels}


# ===========================================================================
# S1 — row-set: the three-predicate filter returns EXACTLY the in-scope ids.
# ===========================================================================


async def test_filter_returns_exactly_in_scope_chunks(kb: Khora) -> None:
    """``recall(filter=...)`` narrows the live result to exactly the 5 in-scope
    chunks — no out-of-scope leak.

    This is the end-to-end proof that the engine→backend wiring actually filters
    rows: the facade builds the AST, the skeleton engine threads it to the
    pgvector store, and the compiler emits a ``khora_chunks`` WHERE predicate.
    Every out-of-scope row violates exactly one predicate, so a leak would name
    the broken one.
    """
    ns = await kb.create_namespace()
    namespace_id: UUID = ns.namespace_id

    chunk_ids = await _seed(kb, namespace_id)
    in_scope = _ids_for(chunk_ids, _IN_SCOPE)
    out_of_scope = _ids_for(chunk_ids, _OUT_OF_SCOPE)
    assert len(in_scope) == 5, "expected 5 in-scope chunks seeded"
    assert len(out_of_scope) == 5, "expected 5 out-of-scope chunks seeded"

    # limit comfortably exceeds the corpus so the filter, not the limit, bounds
    # the result. VECTOR mode keeps retrieval purely embedding+filter (no BM25
    # keyword channel) so the filter is the only narrowing force.
    result = await kb.recall(
        "alpha bravo charlie",
        namespace=namespace_id,
        limit=20,
        mode=SearchMode.VECTOR,
        filter=_RECALL_FILTER,
    )

    returned = {c.id for c in result.chunks}
    assert returned == in_scope, (
        f"filter must return EXACTLY the in-scope chunk ids; "
        f"leaked={returned & out_of_scope}, missing={in_scope - returned}"
    )


async def test_no_filter_returns_all_chunks(kb: Khora) -> None:
    """Control: with no filter the same recall returns the whole corpus.

    Proves the in-scope/out-of-scope split is a property of the FILTER, not of
    retrieval reachability — every seeded chunk is recallable absent the filter,
    so S1's narrowing is attributable to the predicate alone.
    """
    ns = await kb.create_namespace()
    namespace_id: UUID = ns.namespace_id

    chunk_ids = await _seed(kb, namespace_id)

    result = await kb.recall(
        "alpha bravo charlie",
        namespace=namespace_id,
        limit=20,
        mode=SearchMode.VECTOR,
    )

    returned = {c.id for c in result.chunks}
    assert returned == set(chunk_ids.values()), "unfiltered recall must reach every seeded chunk"


# ===========================================================================
# S2 — engine_info handling + facade-level filter validation.
# ===========================================================================


async def test_engine_info_reports_skeleton_filter_row(kb: Khora) -> None:
    """``engine_info['filter']`` carries the skeleton support row with HONEST
    ``pushed_down=True``, and the filtered row-set is correct.

    What this pins is the *carrier contract* with honest pushdown reporting:
    ``engine_info['filter']`` is present on a filtered recall and
    reports ``engine="skeleton"`` / ``supported=True`` / ``pushed_down=True``.
    The skeleton-pgvector backend compiles the whole filter to a SQL WHERE
    predicate, so the pushdown is genuine — the facade surfaces the
    engine-reported flag rather than hardcoding ``False``. The assertion that
    actually matters — that the filter narrows to exactly the in-scope rows — is
    re-checked here too.
    """
    ns = await kb.create_namespace()
    namespace_id: UUID = ns.namespace_id
    chunk_ids = await _seed(kb, namespace_id)

    result = await kb.recall(
        "alpha bravo charlie",
        namespace=namespace_id,
        limit=20,
        mode=SearchMode.VECTOR,
        filter=_RECALL_FILTER,
    )

    # Carrier present with the stable skeleton support row. The full filter is
    # compiled to a khora_chunks WHERE predicate, so ``pushed_down`` is honestly
    # True for the skeleton-pgvector path.
    info = (result.engine_info or {}).get("filter")
    assert info is not None, "engine_info['filter'] carrier must be present on a filtered recall"
    assert info["engine"] == "skeleton"
    assert info["supported"] is True
    assert info["pushed_down"] is True

    # The assertion that actually matters: the filter narrows to exactly the
    # in-scope rows (same contract as S1, re-pinned alongside the carrier).
    assert {c.id for c in result.chunks} == _ids_for(chunk_ids, _IN_SCOPE)


async def test_engine_info_no_filter_reports_pushed_down_false(kb: Khora) -> None:
    """A no-filter recall reports ``pushed_down=False`` on the live lake.

    Pushdown is reported honestly per call. With no ``filter=`` there
    is nothing to push down, so the carrier — still present on every recall —
    reports ``pushed_down=False`` even on the skeleton-pgvector path that DOES
    push filters down when one is supplied.
    """
    ns = await kb.create_namespace()
    namespace_id: UUID = ns.namespace_id
    await _seed(kb, namespace_id)

    result = await kb.recall(
        "alpha bravo charlie",
        namespace=namespace_id,
        limit=20,
        mode=SearchMode.VECTOR,
    )

    info = (result.engine_info or {}).get("filter")
    assert info is not None, "engine_info['filter'] carrier must be present even with no filter"
    assert info["engine"] == "skeleton"
    assert info["pushed_down"] is False


async def test_engine_info_bare_filter_reports_pushed_down_false(kb: Khora) -> None:
    """A bare ``filter={}`` reports ``pushed_down=False`` on the live lake.

    Edge case: ``{}`` is a non-None match-everything AST (so it reaches
    the engine), but it carries zero predicates and narrows nothing. The live
    skeleton-pgvector derivation requires the filter to actually have
    constraints (``bool(filter_ast.children)``), so a constraint-free filter
    honestly reports ``pushed_down=False`` — no pushdown bit, nothing to claim.
    The unfiltered row-set is returned (same as the control), confirming the
    bare filter does not narrow.
    """
    ns = await kb.create_namespace()
    namespace_id: UUID = ns.namespace_id
    chunk_ids = await _seed(kb, namespace_id)

    result = await kb.recall(
        "alpha bravo charlie",
        namespace=namespace_id,
        limit=20,
        mode=SearchMode.VECTOR,
        filter={},
    )

    info = (result.engine_info or {}).get("filter")
    assert info is not None, "engine_info['filter'] carrier must be present on a bare-filter recall"
    assert info["engine"] == "skeleton"
    assert info["pushed_down"] is False, "a constraint-free filter narrows nothing → not pushed down"

    # The bare filter matches everything, so the full corpus comes back.
    assert {c.id for c in result.chunks} == set(chunk_ids.values())


async def test_invalid_filter_raises_validation_error(kb: Khora) -> None:
    """An invalid filter raises ``RecallFilterValidationError`` through recall().

    Covers both validation failure shapes: an unknown top-level key and an
    illegal operator for an otherwise-valid key. Both must surface the typed
    error from ``khora.filter`` before any retrieval work.
    """
    ns = await kb.create_namespace()
    namespace_id: UUID = ns.namespace_id

    with pytest.raises(RecallFilterValidationError):
        await kb.recall(
            "alpha bravo charlie",
            namespace=namespace_id,
            mode=SearchMode.VECTOR,
            filter={"not_a_real_key": "x"},
        )

    with pytest.raises(RecallFilterValidationError):
        await kb.recall(
            "alpha bravo charlie",
            namespace=namespace_id,
            mode=SearchMode.VECTOR,
            # $contains is not a legal operator on the occurred_at system key.
            filter={"occurred_at": {"$contains": "x"}},
        )


# ===========================================================================
# S5 — backward-compat: the deprecated start_time/end_time bounds, live path.
# ===========================================================================
#
# The shim MECHANICS (DeprecationWarning emission, fold-to-AST, the
# filter=+bounds ValueError) are already covered against a mock engine in
# tests/unit/test_khora.py::TestRecallTemporalBounds and ::TestRecallFilterKwarg.
# These tests add the gap that mocks can't reach: that the deprecated bounds
# actually FILTER ROWS end-to-end on the live skeleton-pgvector lake.


async def test_start_time_bound_filters_rows_live(kb: Khora) -> None:
    """``start_time=`` (deprecated) filters rows end-to-end on the live lake.

    The bound folds to ``occurred_at >= start`` and AND-s into the same
    khora_chunks predicate path the public ``filter=`` uses. With the bound set
    to the in-scope lower edge, only rows at/after it survive — the same
    occurred_at split S1 exercises, but reached through the legacy kwarg. The
    DeprecationWarning is asserted here too so the live path is exercised under
    the warning contract, not just the mocked unit path.
    """
    ns = await kb.create_namespace()
    namespace_id: UUID = ns.namespace_id
    chunk_ids = await _seed(kb, namespace_id)

    # occurred_at >= 2026-04-05 excludes only "out_too_old" (2026-04-04). Every
    # other seeded row (in-scope + the other out-of-scope rows) is at/after the
    # bound, so 9 of 10 survive — this isolates the temporal predicate alone.
    start = datetime(2026, 4, 5, tzinfo=UTC)

    with pytest.warns(DeprecationWarning):
        result = await kb.recall(
            "alpha bravo charlie",
            namespace=namespace_id,
            limit=20,
            mode=SearchMode.VECTOR,
            start_time=start,
        )

    returned = {c.id for c in result.chunks}
    expected = set(chunk_ids.values()) - {chunk_ids["out_too_old"]}
    assert chunk_ids["out_too_old"] not in returned, "start_time= must exclude the pre-bound row"
    assert returned == expected, f"expected 9 of 10 rows at/after the bound, got {len(returned)}"


async def test_filter_and_bounds_conflict_raises_live(kb: Khora) -> None:
    """Passing BOTH filter= and the deprecated bounds raises ValueError.

    Asserted on the live façade (khora.py:2040-2041) to confirm the guard fires
    before any engine/DB work on the real path, complementing the mocked unit
    coverage.
    """
    ns = await kb.create_namespace()
    namespace_id: UUID = ns.namespace_id

    with pytest.raises(ValueError, match="filter= or the deprecated start_time/end_time"):
        await kb.recall(
            "alpha bravo charlie",
            namespace=namespace_id,
            mode=SearchMode.VECTOR,
            filter=_RECALL_FILTER,
            start_time=datetime(2026, 4, 5, tzinfo=UTC),
        )
