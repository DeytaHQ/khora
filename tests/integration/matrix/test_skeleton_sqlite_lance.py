"""Skeleton SQLite + LanceDB integration tests.

Mirrors ``tests/integration/matrix/test_skeleton_pg.py`` (PR #474) for the
embedded stack. Skeleton has no graph component, so the embedded subset is
**SQLite (relational + FTS5 + graph adapter) + LanceDB (vectors)**.

How the LLM is mocked:
* ``LiteLLMEmbedder.embed_batch`` and ``embed`` return content-derived unit
  vectors. The test rig uses a 32-dim embedding so LanceDB indexes stay
  fast in tmpdir; ``config.llm.embedding_dimension`` and
  ``config.storage.embedding_dimension`` are aligned to 32.
* Skeleton does **no** entity extraction, so no extractor stub is needed.

How to run locally::

    uv run pytest tests/integration/matrix/test_skeleton_sqlite_lance.py \\
        -v -m integration --no-cov
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False

from khora.config import KhoraConfig
from khora.config.schema import SQLiteLanceConfig
from khora.engines.skeleton.backends import TemporalFilter
from khora.khora import Khora
from khora.query import SearchMode

EMBED_DIM = 32  # small dim keeps LanceDB index build cheap in tmp_path


# ---------------------------------------------------------------------------
# Fixtures: skip-if-no-embedded-deps, deterministic embedder
# ---------------------------------------------------------------------------


pytestmark = [
    pytest.mark.embedded,
    pytest.mark.integration,
    pytest.mark.skipif(
        not _HAS_EMBEDDED,
        reason="aiosqlite/lancedb not installed (pip install khora[sqlite_lance])",
    ),
]


# Keyword vocabulary: deterministic, content-aware embeddings. Slot count
# stays well below EMBED_DIM=32 so we never collide.
_KEYWORD_SLOTS: dict[str, int] = {
    "alpha": 0,
    "bravo": 1,
    "charlie": 2,
    "delta": 3,
    "echo": 4,
    "kangaroo": 5,
    "kangaroos": 5,
    "penguin": 6,
    "penguins": 6,
    "widget": 7,
    "falcon": 8,
    "launch": 9,
    "rocket": 10,
    "tag": 11,
    "metadata": 12,
    "filter": 13,
    "concurrent": 14,
    "batch": 15,
    "bulk": 16,
    "recent": 17,
    "old": 18,
    "first": 19,
    "second": 20,
    "third": 21,
    "fourth": 22,
    "fifth": 23,
    "animals": 24,
    "document": 25,
    "group": 26,
}


def _embed_for(text_in: str) -> list[float]:
    """Deterministic 32-dim unit vector derived from ``text_in``.

    Small constant baseline component so the all-zero edge case (a query
    that matches no vocabulary) still gets a defined vector.
    """
    vec = [0.0] * EMBED_DIM
    vec[EMBED_DIM - 1] = 0.01

    lower = text_in.lower()
    for kw, slot in _KEYWORD_SLOTS.items():
        if kw in lower:
            vec[slot] = 1.0

    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


async def _stub_embed_batch(self: Any, texts: list[str]) -> list[list[float]]:
    return [_embed_for(t) for t in texts]


async def _stub_embed(self: Any, text_in: str) -> list[float]:
    return _embed_for(text_in)


@pytest.fixture(autouse=True)
def _patch_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the embedder so no real LLM is called.

    Skeleton does no entity extraction, so the extractor doesn't need a stub.
    """
    monkeypatch.setattr(
        "khora.extraction.embedders.litellm.LiteLLMEmbedder.embed_batch",
        _stub_embed_batch,
    )
    monkeypatch.setattr(
        "khora.extraction.embedders.litellm.LiteLLMEmbedder.embed",
        _stub_embed,
    )


@pytest.fixture
async def kb(tmp_path: Path) -> AsyncIterator[Khora]:
    """Per-test Skeleton Khora bound to an embedded SQLite+LanceDB pair.

    The fixture allocates a fresh tmp_path per test; ``run_migrations=True``
    builds the alembic schema in the SQLite file before the coordinator
    opens it (mirrors :func:`build_sqlite_lance_coordinator` in
    ``tests/integration/_sqlite_lance_fixtures.py``).
    """
    db_path = str(tmp_path / "khora.db")
    lance_path = str(tmp_path / "khora.lance")

    config = KhoraConfig()
    config.storage.backend = "sqlite_lance"
    config.storage.sqlite_lance = SQLiteLanceConfig(
        db_path=db_path,
        lance_path=lance_path,
        embedding_dimension=EMBED_DIM,
    )
    config.llm.embedding_dimension = EMBED_DIM
    config.storage.embedding_dimension = EMBED_DIM
    # Single-chunk documents keep the test deterministic.
    config.pipelines.chunk_size = 1024

    kb = Khora(config, engine="skeleton", run_migrations=True)
    await kb.connect()
    try:
        yield kb
    finally:
        await kb.disconnect()


@pytest.fixture
async def namespace_id(kb: Khora) -> UUID:
    ns = await kb.create_namespace()
    return ns.namespace_id


async def _remember(
    kb: Khora,
    *,
    namespace_id: UUID,
    content: str,
    title: str = "",
    metadata: dict[str, Any] | None = None,
) -> Any:
    return await kb.remember(
        content=content,
        namespace=namespace_id,
        title=title,
        metadata=metadata,
        # #890: Skeleton refuses non-empty entity_types / relationship_types
        # because it has no entity extraction. Pass empty lists; the
        # protocol still requires the kwarg but the engine no longer
        # silently swallows non-empty values.
        entity_types=[],
        relationship_types=[],
    )


async def _recall(kb: Khora, query: str, **kwargs: Any) -> Any:
    """Recall wrapper that pins ``mode=SearchMode.VECTOR`` for deterministic ranking.

    Mirrors ``_recall`` in ``test_skeleton_pg.py``: the wrapper sidesteps
    BM25 weighting under HYBRID so top-k ordering tests aren't affected by
    the blend weight. ``test_skeleton_recall_default_hybrid_mode`` exercises
    the default-HYBRID path explicitly.
    """
    kwargs.setdefault("mode", SearchMode.VECTOR)
    return await kb.recall(query, **kwargs)


# ---------------------------------------------------------------------------
# Tests — mirror test_skeleton_pg.py one-for-one
# ---------------------------------------------------------------------------


async def test_skeleton_remember_recall_roundtrip(kb: Khora, namespace_id: UUID) -> None:
    """Ingest 3 docs, recall, assert ingested text appears in context."""
    contents = [
        "alpha document mentions the falcon launch in detail.",
        "bravo document covers a different rocket programme entirely.",
        "charlie document is a side note unrelated to anything else.",
    ]
    for c in contents:
        await _remember(kb, namespace_id=namespace_id, content=c)

    result = await _recall(kb, "falcon launch", namespace=namespace_id, limit=10)

    # Skeleton reports its backend type in metadata. Once the LanceDB-backed
    # vector path lands, the value should be "lancedb" (or whatever name the
    # new backend registers under) — assert non-empty rather than pin the
    # string.
    assert result.engine_info.get("backend") is not None
    assert len(result.chunks) >= 1, "expected at least one chunk back"
    assert any("falcon" in c.content.lower() for c in result.chunks)


async def test_skeleton_namespace_isolation(kb: Khora) -> None:
    """Two namespaces, queries don't cross-bleed."""
    ns_a = (await kb.create_namespace()).namespace_id
    ns_b = (await kb.create_namespace()).namespace_id

    await _remember(kb, namespace_id=ns_a, content="alpha document about kangaroos in the outback.")
    await _remember(kb, namespace_id=ns_b, content="bravo document about penguins on the ice.")

    result_a = await _recall(kb, "animals", namespace=ns_a, limit=10)
    result_b = await _recall(kb, "animals", namespace=ns_b, limit=10)

    a_text = " ".join(c.content for c in result_a.chunks)
    b_text = " ".join(c.content for c in result_b.chunks)

    assert "kangaroos" in a_text
    assert "penguins" not in a_text, "namespace_b content leaked into namespace_a"
    assert "penguins" in b_text
    assert "kangaroos" not in b_text, "namespace_a content leaked into namespace_b"


async def test_skeleton_recall_top_k_ordering(kb: Khora, namespace_id: UUID) -> None:
    """Results ordered by descending similarity (combined_score)."""
    await _remember(
        kb,
        namespace_id=namespace_id,
        content="alpha bravo charlie delta echo (high overlap with query)",
    )
    await _remember(
        kb,
        namespace_id=namespace_id,
        content="alpha bravo charlie (medium overlap with query)",
    )
    await _remember(
        kb,
        namespace_id=namespace_id,
        content="alpha (low overlap with query)",
    )

    result = await _recall(kb, "alpha bravo charlie delta echo", namespace=namespace_id, limit=10)

    assert len(result.chunks) >= 3
    scores = [c.score for c in result.chunks]
    for prev, curr in zip(scores, scores[1:]):
        assert prev >= curr, f"similarity ordering violated: {prev} < {curr} in {scores}"


async def test_skeleton_recall_with_metadata_filter(kb: Khora, namespace_id: UUID) -> None:
    """Tag filter restricts recall to chunks carrying the requested tag.

    Unlike the PG sibling, the embedded path doesn't hit the ARRAY
    incompatibility (SQLite serializes ``tags`` as JSON-text — no
    ``ARRAY(String).contains`` incompatibility). Because this test bypasses
    :meth:`Khora.recall` and calls ``engine.recall`` directly, it resolves the
    stable namespace id to the row-level id itself (see below).
    """
    await _remember(
        kb,
        namespace_id=namespace_id,
        content="alpha document tagged group A",
        metadata={"tags": ["group-A"]},
    )
    await _remember(
        kb,
        namespace_id=namespace_id,
        content="alpha document tagged group B",
        metadata={"tags": ["group-B"]},
    )

    engine = kb._get_engine()  # type: ignore[attr-defined]
    # Engine-layer recall expects the row-level namespace id (the FK target on
    # ``khora_chunks.namespace_id``); the public ``namespace_id`` fixture is
    # the stable namespace identifier. ``Khora.recall`` resolves this for
    # us automatically — when calling the engine directly we have to do it
    # ourselves.
    row_namespace_id = await kb._resolve_namespace(namespace_id)  # type: ignore[attr-defined]
    result = await engine.recall(
        "alpha document",
        row_namespace_id,
        limit=10,
        temporal_filter=TemporalFilter(tags=["group-A"]),
        hybrid_alpha=1.0,
    )

    assert len(result.chunks) >= 1
    for chunk in result.chunks:
        assert "group A" in chunk.content, f"non-group-A leaked in: {chunk.content!r}"
    assert all("group B" not in c.content for c in result.chunks), "group-B chunk leaked through tag filter"


async def test_skeleton_temporal_filter(kb: Khora, namespace_id: UUID) -> None:
    """Two docs 5d vs 20d apart, "last 7 days" → recent only.

    Skeleton's single-doc ``remember()`` ignores ``metadata['occurred_at']``
    (only ``remember_batch`` reads it). To dodge that here we
    use ``remember_batch``, which reads the field and passes it through to
    ``TemporalChunk.occurred_at``. Mirrors the dodge documented in
    ``test_skeleton_pg.py`` but avoids the SQL-update workaround since
    backdating chunks via direct SQL would couple this test to the
    sqlite_lance internals.
    """
    now = datetime.now(UTC)
    ts_recent = (now - timedelta(days=5)).isoformat()
    ts_old = (now - timedelta(days=20)).isoformat()

    batch = await kb.remember_batch(
        [
            {
                "content": "recent document about the falcon launch",
                "title": "recent",
                "metadata": {"occurred_at": ts_recent},
            },
            {
                "content": "old document about the falcon launch",
                "title": "old",
                "metadata": {"occurred_at": ts_old},
            },
        ],
        namespace=namespace_id,
        entity_types=["PERSON"],
        relationship_types=["RELATES_TO"],
    )
    assert batch.processed == 2
    assert batch.failed == 0

    seven_days_ago = now - timedelta(days=7)
    result = await _recall(
        kb,
        "falcon launch",
        namespace=namespace_id,
        limit=10,
        start_time=seven_days_ago,
    )

    contents = [c.content for c in result.chunks]
    assert any("recent" in c for c in contents), f"recent doc not returned: {contents}"
    assert not any("old document" in c for c in contents), (
        f"20-day-old document leaked through occurred_after filter: {contents}"
    )


async def test_skeleton_remember_batch(kb: Khora, namespace_id: UUID) -> None:
    """Bulk-ingest 20 docs in a single ``remember_batch`` call."""
    documents = [
        {
            "content": f"batch document number {i} contains widget-{i} content",
            "title": f"doc-{i}",
        }
        for i in range(20)
    ]
    batch = await kb.remember_batch(
        documents,
        namespace=namespace_id,
        entity_types=["PERSON"],
        relationship_types=["RELATES_TO"],
    )

    assert batch.processed == 20, f"expected 20 processed, got {batch}"
    assert batch.failed == 0, f"unexpected failures: {batch}"
    assert batch.chunks >= 20, f"expected ≥20 chunks (one per doc), got {batch.chunks}"

    result = await _recall(kb, "widget batch document", namespace=namespace_id, limit=25)
    contents_returned = {c.content for c in result.chunks}
    assert len(contents_returned) >= 20, f"expected ≥20 distinct chunks returned, got {len(contents_returned)}"


async def test_skeleton_recall_empty_namespace(kb: Khora) -> None:
    """Recall against an empty namespace returns an empty chunks list."""
    ns = (await kb.create_namespace()).namespace_id

    result = await _recall(kb, "anything at all", namespace=ns, limit=10)

    assert result.chunks == []
    assert result.entities == []  # Skeleton never returns entities anyway.
    assert result.engine_info.get("backend") is not None


async def test_skeleton_recall_metadata_keys(kb: Khora, namespace_id: UUID) -> None:
    """RecallResult.metadata exposes the keys the Skeleton engine documents."""
    await _remember(kb, namespace_id=namespace_id, content="alpha simple sentence")

    result = await _recall(kb, "alpha", namespace=namespace_id, limit=5)

    md = result.engine_info
    expected = {"backend", "hybrid_alpha", "temporal_filter"}
    missing = expected - md.keys()
    assert not missing, f"missing skeleton metadata keys: {missing}"
    assert md["hybrid_alpha"] == 1.0
    assert md["temporal_filter"] is None


async def test_skeleton_recall_default_hybrid_mode(kb: Khora, namespace_id: UUID) -> None:
    """Default ``Khora.recall(...)`` works on HYBRID — regression.

    Pre-fix, ``SkeletonConstructionEngine.recall`` referenced a non-existent
    ``SearchMode.KEYWORD`` member, crashing on default HYBRID.
    added the enum member; this test exercises the default path on the
    embedded backend so the regression coverage extends past PG.
    """
    await _remember(kb, namespace_id=namespace_id, content="alpha simple sentence")
    result = await kb.recall("alpha", namespace=namespace_id, limit=5)

    assert result.engine_info.get("backend") is not None
    # HYBRID maps to ``hybrid_alpha=0.7`` per engine.py:444.
    assert result.engine_info.get("hybrid_alpha") == 0.7


async def test_skeleton_concurrent_remember(kb: Khora, namespace_id: UUID) -> None:
    """5 concurrent ingests in one namespace, no integrity errors."""
    contents = [f"document number {i} mentions widget-{i}" for i in range(5)]
    results = await asyncio.gather(
        *(_remember(kb, namespace_id=namespace_id, content=c) for c in contents),
        return_exceptions=True,
    )

    errors = [r for r in results if isinstance(r, Exception)]
    assert not errors, f"concurrent remember raised: {errors}"

    doc_ids = {r.document_id for r in results}  # type: ignore[union-attr]
    assert len(doc_ids) == 5, f"expected 5 distinct documents, got {doc_ids}"

    result = await _recall(kb, "widget", namespace=namespace_id, limit=20)
    contents_returned = {c.content for c in result.chunks}
    assert len(contents_returned) >= 5


async def test_skeleton_recall_handles_punctuated_query(kb: Khora, namespace_id: UUID) -> None:
    """Regression for issue #526 at the **engine layer**.

    PR #528's escape_fts5_query fix was verified at the storage adapter
    layer; this test pushes punctuated / FTS5-operator queries through
    the full Khora.recall() → skeleton engine path. Catches a future
    regression that introduces a new fusion path bypassing the escape.
    """
    await _remember(
        kb,
        namespace_id=namespace_id,
        content="alpha document with widget reference and falcon launch.",
    )
    for query in (
        "What about widget?",
        "widget: please",
        "widget (please)",
        "widget AND alpha",
        'say "hello" widget',
        "widget*",
    ):
        result = await _recall(kb, query, namespace=namespace_id, limit=3)
        assert isinstance(result.chunks, list), f"recall must not raise on {query!r}"


# ---------------------------------------------------------------------------
# Deterministic recall-filter — metadata pushdown + python-fallback parity.
# ---------------------------------------------------------------------------
#
# These two cases drive the *public* ``Khora.recall(filter=...)`` end-to-end
# through the skeleton engine into the sqlite_lance backend, where ``compile_lance``
# pushes the metadata predicate into the SQLite WHERE (when JSON1 is available) and
# a ``compile_python`` post-filter enforces the full AST. The seed corpus has one
# in-scope tier and three out-of-scope rows, each violating the filter a DIFFERENT
# way (wrong value / absent path / present-null), so a leak names how the predicate
# failed to bite. All rows share embedding-vocabulary so the vector channel returns
# the whole corpus and the filter is the only narrowing force.
#
# The first case exercises the pushdown path (JSON1 present); the second forces the
# python-fallback by monkeypatching the backend's ``_has_json1`` capability flag to
# False (the documented test seam — see SQLiteLanceTemporalStore.__init__) so every
# metadata leaf defers to the compile_python post-filter. The load-bearing contract
# the brief pins is that BOTH paths return the IDENTICAL in-scope chunk set.

# metadata.tier == "gold" is the in-scope predicate. Each out-of-scope row violates
# it a distinct way; every row shares "alpha gold" vocabulary so the vector channel
# does not itself narrow.
_FILTER_SEED: dict[str, dict[str, Any]] = {
    "in_one": {"content": "alpha gold document one", "metadata": {"tier": "gold"}},
    "in_two": {"content": "alpha gold document two", "metadata": {"tier": "gold"}},
    # wrong value — tier present but != "gold".
    "out_value": {"content": "alpha gold document silver tier", "metadata": {"tier": "silver"}},
    # absent path — no "tier" key at all.
    "out_absent": {"content": "alpha gold document no tier", "metadata": {"other": "x"}},
    # present-null — tier explicitly null (distinct from absent; must not match $eq).
    "out_null": {"content": "alpha gold document null tier", "metadata": {"tier": None}},
}
_FILTER_WIRE = {"metadata.tier": "gold"}


async def _seed_filter_corpus(kb: Khora, namespace_id: UUID) -> dict[str, str]:
    """Remember the filter corpus; return a ``label -> content`` map.

    Content is distinct per row (skeleton dedupes by content checksum), so each
    label round-trips to exactly one recallable chunk.
    """
    label_to_content: dict[str, str] = {}
    for label, spec in _FILTER_SEED.items():
        await _remember(
            kb,
            namespace_id=namespace_id,
            content=spec["content"],
            title=label,
            metadata=spec["metadata"],
        )
        label_to_content[label] = spec["content"]
    return label_to_content


_IN_SCOPE_LABELS = ("in_one", "in_two")


async def test_skeleton_recall_metadata_filter_pushdown(kb: Khora, namespace_id: UUID) -> None:
    """``recall(filter={"metadata.tier": "gold"})`` narrows to exactly the gold rows.

    Drives the metadata predicate through the JSON1 pushdown path. The three
    out-of-scope rows (wrong value / absent path / present-null) must all be
    excluded — proving the compiled SQLite WHERE + post-filter honor §4 exactly,
    and that present-null does NOT match a positive ``$eq`` (the s4/s7 distinction
    proven in the oracle, here end-to-end on a real store).
    """
    label_to_content = await _seed_filter_corpus(kb, namespace_id)

    result = await kb.recall(
        "alpha gold document",
        namespace=namespace_id,
        limit=20,
        mode=SearchMode.VECTOR,
        filter=_FILTER_WIRE,
    )

    returned = {c.content for c in result.chunks}
    in_scope = {label_to_content[label] for label in _IN_SCOPE_LABELS}
    out_of_scope = {label_to_content[label] for label in _FILTER_SEED if label not in _IN_SCOPE_LABELS}

    assert returned == in_scope, (
        f"metadata filter must return exactly the in-scope chunks; "
        f"leaked={returned & out_of_scope}, missing={in_scope - returned}"
    )

    # Control: with no filter the same recall reaches every seeded chunk, proving
    # the narrowing is the FILTER's doing, not retrieval reachability.
    unfiltered = await kb.recall("alpha gold document", namespace=namespace_id, limit=20, mode=SearchMode.VECTOR)
    assert {c.content for c in unfiltered.chunks} == set(label_to_content.values())


async def test_skeleton_recall_metadata_filter_python_fallback(
    kb: Khora,
    namespace_id: UUID,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forcing the python-fallback returns the IDENTICAL in-scope chunk set.

    Monkeypatching the backend's ``_has_json1`` capability flag to ``False`` makes
    ``compile_lance`` treat every metadata leaf as unsupported, so nothing about the
    metadata predicate pushes into SQLite — the ``compile_python`` post-filter alone
    enforces it against the decoded chunks. The contract: the fallback row-set is
    IDENTICAL to the pushdown row-set (the two compilers agree on the §4 contract).
    """
    label_to_content = await _seed_filter_corpus(kb, namespace_id)
    in_scope = {label_to_content[label] for label in _IN_SCOPE_LABELS}

    # Force the fallback: disable JSON1 on the live temporal store. The flag is the
    # documented test seam (SQLiteLanceTemporalStore.__init__); with it False the
    # SchemaCapabilities the backend hands compile_lance carry sqlite_json1=False,
    # so metadata leaves defer to the compile_python post-filter.
    store = kb._get_engine()._get_temporal_store()  # type: ignore[attr-defined]
    monkeypatch.setattr(store, "_has_json1", False)

    result = await kb.recall(
        "alpha gold document",
        namespace=namespace_id,
        limit=20,
        mode=SearchMode.VECTOR,
        filter=_FILTER_WIRE,
    )

    returned = {c.content for c in result.chunks}
    out_of_scope = {label_to_content[label] for label in _FILTER_SEED if label not in _IN_SCOPE_LABELS}
    assert returned == in_scope, (
        f"python-fallback must return the SAME in-scope set as pushdown; "
        f"leaked={returned & out_of_scope}, missing={in_scope - returned}"
    )


# ---------------------------------------------------------------------------
# Engine-level FilterPushdownReport — the honest #1069 report on a REAL store.
# ---------------------------------------------------------------------------
#
# These pin ``RecallResult.engine_info["filter"]`` — the canonical
# ``FilterPushdownReport.model_dump()`` the skeleton engine folds from the
# sqlite_lance backend's ``_last_filter_plan`` (the SAME compile_lance pass its
# search ran). They drive the EXACT #1069 repro
# (``filter={"source_name": "linear", "metadata.tier": "gold"}``) through the
# public ``Khora.recall`` on a fully-embedded SQLite+LanceDB engine and assert:
#
#   * JSON1 present  -> both leaves pushed (pushed_down=True), post_filtered=True
#     from the always-on compile_python defensive re-check, BOTH keys in
#     ``pushed_keys`` (NO-DEMOTE), nothing in ``post_filtered_keys``.
#   * JSON1 absent   -> the metadata leaf splits to ``post_filtered_keys`` and
#     pushed_down=False; the system-key leaf stays pushed.
#   * no-filter / filter={} -> the canonical empty carrier with ONE named
#     ``sqlite_lance`` channel entry (decision 1), NOT channels={}.
#
# Every emitted report round-trips through ``FilterPushdownReport.model_validate``.
# The JSON1 toggle uses the documented ``_has_json1`` test seam, mirroring
# ``test_skeleton_recall_metadata_filter_python_fallback`` above.

# The #1069 repro filter: a system-key leaf (source_name) + a metadata leaf
# (metadata.tier). Both leaves push down when JSON1 is present.
_REPORT_FILTER = {"source_name": "linear", "metadata.tier": "gold"}
_REPORT_LEAVES = {"metadata.tier", "source_name"}


async def _seed_report_corpus(kb: Khora, namespace_id: UUID) -> None:
    """Seed a row that satisfies the #1069 repro filter.

    Uses ``kb.remember`` directly (not the suite's ``_remember`` helper) so the
    ``source_name`` system key is stamped onto the chunk — the filter constrains
    both ``source_name`` and ``metadata.tier``.
    """
    await kb.remember(
        content="alpha gold linear ticket about the falcon launch",
        namespace=namespace_id,
        title="repro",
        metadata={"tier": "gold"},
        source_name="linear",
        entity_types=[],
        relationship_types=[],
    )


def _filter_report(result: Any) -> dict[str, Any]:
    """Read + validate the engine_info filter report off a recall result.

    Asserts the canonical model round-trips (``model_validate``) on EVERY emitted
    report, then returns the raw dict for field-level assertions.
    """
    from khora.filter import FilterPushdownReport

    info = result.engine_info.get("filter")
    assert info is not None, "skeleton engine must write engine_info['filter'] on every recall"
    # Round-trip the canonical model — proves the emitted dict is a valid report.
    assert FilterPushdownReport.model_validate(info) == FilterPushdownReport.model_validate(info)
    return info


async def test_engine_filter_report_pushed_down_with_json1(kb: Khora, namespace_id: UUID) -> None:
    """#1069 repro WITH JSON1 → both leaves pushed, pushed_down=True (NO-DEMOTE).

    The sqlite_lance backend compiles both ``source_name`` and ``metadata.tier``
    into the SQLite WHERE when JSON1 is available, then runs an always-on
    compile_python post-filter over the full AST as a safety net. The honest
    report: ``pushed_down=True`` (every leaf pushed, none post-filtered),
    ``post_filtered=True`` (the defensive re-check fired), BOTH keys in
    ``pushed_keys``, and ``post_filtered_keys`` EMPTY (NO-DEMOTE — a fully-pushed
    leaf is never demoted by the defensive re-check).
    """
    await _seed_report_corpus(kb, namespace_id)
    store = kb._get_engine()._get_temporal_store()  # type: ignore[attr-defined]
    if not store._has_json1:
        pytest.skip("SQLite build lacks JSON1; the pushdown half is the absent-JSON1 case")

    result = await kb.recall(
        "alpha gold document",
        namespace=namespace_id,
        limit=20,
        mode=SearchMode.VECTOR,
        filter=_REPORT_FILTER,
    )

    info = _filter_report(result)
    assert info["pushed_down"] is True
    assert info["post_filtered"] is True  # always-on compile_python defensive re-check
    assert set(info["pushed_keys"]) == _REPORT_LEAVES
    assert info["post_filtered_keys"] == []  # NO-DEMOTE
    assert set(info["channels"]["sqlite_lance"]["pushed_keys"]) == _REPORT_LEAVES
    assert info["channels"]["sqlite_lance"]["post_filtered_keys"] == []


async def test_engine_filter_report_split_without_json1(
    kb: Khora, namespace_id: UUID, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1069 repro WITHOUT JSON1 → metadata leaf splits to post_filtered_keys.

    Forcing the ``_has_json1=False`` seam makes compile_lance treat the metadata
    leaf as unsupported, so only the ``source_name`` system-key leaf pushes into
    SQLite; ``metadata.tier`` defers to the compile_python post-filter. The honest
    report: ``pushed_down=False`` (a leaf was post-filtered), ``source_name`` in
    ``pushed_keys`` and ``metadata.tier`` in ``post_filtered_keys`` — the two
    leaves partition cleanly.
    """
    await _seed_report_corpus(kb, namespace_id)
    store = kb._get_engine()._get_temporal_store()  # type: ignore[attr-defined]
    monkeypatch.setattr(store, "_has_json1", False)

    result = await kb.recall(
        "alpha gold document",
        namespace=namespace_id,
        limit=20,
        mode=SearchMode.VECTOR,
        filter=_REPORT_FILTER,
    )

    info = _filter_report(result)
    assert info["pushed_down"] is False
    assert info["post_filtered"] is True
    assert info["pushed_keys"] == ["source_name"]
    assert info["post_filtered_keys"] == ["metadata.tier"]
    # The two leaves partition the constraint set (each in exactly one list).
    assert set(info["pushed_keys"]) | set(info["post_filtered_keys"]) == _REPORT_LEAVES
    assert set(info["pushed_keys"]) & set(info["post_filtered_keys"]) == set()
    channel = info["channels"]["sqlite_lance"]
    assert channel["pushed_keys"] == ["source_name"]
    assert channel["post_filtered_keys"] == ["metadata.tier"]


async def test_engine_filter_report_no_filter_carrier(kb: Khora, namespace_id: UUID) -> None:
    """A no-filter recall emits the canonical empty carrier (decision 1).

    Even with no ``filter=`` kwarg the skeleton engine writes the report on every
    recall: all-False flags, empty key lists, and ONE named ``sqlite_lance``
    channel entry (the single configured backend) — NOT ``channels={}``.
    """
    await _seed_report_corpus(kb, namespace_id)

    result = await kb.recall("alpha gold document", namespace=namespace_id, limit=20, mode=SearchMode.VECTOR)

    info = _filter_report(result)
    assert info["pushed_down"] is False
    assert info["post_filtered"] is False
    assert info["pushed_keys"] == []
    assert info["post_filtered_keys"] == []
    # ONE named empty channel — the canonical carrier, not an empty dict.
    assert info["channels"] == {"sqlite_lance": {"pushed_keys": [], "post_filtered_keys": []}}


async def test_engine_filter_report_empty_filter_carrier(kb: Khora, namespace_id: UUID) -> None:
    """A bare ``filter={}`` emits the SAME canonical empty carrier (decision 1).

    ``filter={}`` lowers to a match-everything empty-AND AST that the facade
    threads to the engine as a non-None ``filter_ast`` carrying no leaves. The
    engine reports it identically to the no-filter case — nothing narrowed,
    ``pushed_down=False``, one named empty channel.
    """
    await _seed_report_corpus(kb, namespace_id)

    result = await kb.recall(
        "alpha gold document",
        namespace=namespace_id,
        limit=20,
        mode=SearchMode.VECTOR,
        filter={},
    )

    info = _filter_report(result)
    assert info["pushed_down"] is False
    assert info["post_filtered"] is False
    assert info["pushed_keys"] == []
    assert info["post_filtered_keys"] == []
    assert info["channels"] == {"sqlite_lance": {"pushed_keys": [], "post_filtered_keys": []}}
