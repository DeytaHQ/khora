"""BM25 filter-enforcement proof on the embedded Skeleton + SurrealDB stack.

The SurrealDB sibling of the embedded sqlite_lance BM25 enforcement test
(``test_filter_enforcement_sqlite_lance.py::test_bm25_channel_enforces_filter_embedded``).
It drives a real ``Khora(engine="skeleton")`` against an in-process ``memory://``
SurrealDB store to seed a BM25-searchable corpus, then exercises the public BM25
entry point (``SurrealDBTemporalStore.search_fulltext``) directly, proving a
contradiction recall filter excludes every chunk through the keyword channel.

Why drive ``search_fulltext`` directly rather than ``recall(mode=KEYWORD)`` (as
the sqlite_lance sibling does)? On this backend the two BM25 surfaces differ:
``recall(mode=KEYWORD)`` routes through ``search`` -> ``_search_inner`` (which has
long compiled ``filter_ast`` into its pure-BM25 ``WHERE``), while
``search_fulltext`` is the separate public BM25 lookup the BM25 recall channel
calls. ``search_fulltext`` is the surface that now compiles ``filter_ast`` and
ANDs the predicate into its BM25 ``WHERE`` (mirroring the vector path); a test
that only went through ``_search_inner`` would not exercise that enforcement.

A POSITIVE CONTROL (an unfiltered ``search_fulltext`` over the SAME corpus
returns NON-EMPTY results) proves the corpus is genuinely BM25-searchable, so the
zero in the enforcement assertion means "the filter excluded them", not "BM25
found nothing anyway".

The BM25 full-text index is defined after seeding (the keyword channel falls back
to ``[]`` without it). The store's ``ensure_search_indexes()`` bundles a
dimension-pinned HNSW build the small test embedding dimension cannot satisfy, so
only the BM25 index is defined here — the keyword path never uses HNSW.

Embedded ``memory://`` runs in-process — no server, no Docker. The module
self-skips when the optional ``surrealdb`` SDK is absent.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import partial
from typing import Any
from uuid import UUID

import pytest

pytest.importorskip("surrealdb")

from khora.config import KhoraConfig  # noqa: E402
from khora.config.schema import SurrealDBConfig  # noqa: E402
from khora.engines.skeleton.backends.surrealdb import SurrealDBTemporalStore  # noqa: E402
from khora.extraction.skills import ExpertiseConfig  # noqa: E402
from khora.filter import RecallFilter, parse_to_ast  # noqa: E402
from khora.khora import Khora  # noqa: E402
from tests.test_helpers.filter_spy import EMBED_DIM, seed_corpus, stub_llm  # noqa: E402

pytestmark = [
    pytest.mark.integration,
    pytest.mark.filter_enforcement,
]


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _patch_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deterministic embedder + entity extractor (no API key / network)."""
    stub_llm(monkeypatch, dim=EMBED_DIM)


@pytest.fixture
async def kb() -> AsyncIterator[Khora]:
    """Per-test Skeleton Khora on a fresh in-process SurrealDB store.

    ``mode="memory"`` runs SurrealDB in-process (no container, no on-disk file);
    each test gets its own database. Schema initialises declaratively on
    ``connect()`` (no Alembic), so ``run_migrations`` is a no-op here.
    """
    config = KhoraConfig()
    config.storage.backend = "surrealdb"
    config.storage.surrealdb = SurrealDBConfig(mode="memory", embedding_dimension=EMBED_DIM)
    config.llm.embedding_dimension = EMBED_DIM
    config.storage.embedding_dimension = EMBED_DIM
    config.pipelines.extract_entities = False

    kb = Khora(config, engine="skeleton")
    await kb.connect()
    try:
        yield kb
    finally:
        try:
            await kb.disconnect()
        except Exception:
            pass


@pytest.fixture
async def namespace_id(kb: Khora) -> UUID:
    ns = await kb.create_namespace()
    return ns.namespace_id


def _store(kb: Khora) -> SurrealDBTemporalStore:
    """The live SurrealDB temporal store backing the skeleton engine."""
    store = kb._engine._get_temporal_store()  # type: ignore[union-attr]
    assert isinstance(store, SurrealDBTemporalStore)
    return store


def _remember(kb: Khora) -> Any:
    """Bind the per-test remember wiring for ``seed_corpus``."""
    return partial(
        kb.remember,
        title="",
        entity_types=[],
        relationship_types=[],
        expertise=ExpertiseConfig(name="skeleton-surrealdb-filter-enforcement"),
    )


# Every row shares the token "Falcon" (so a "Falcon" BM25 query returns the full
# set) and carries metadata.visibility == "public" (the discrimination key).
_CORPUS = [
    {"content": "Falcon launch coordinated by SpaceX in early 2026.", "metadata": {"visibility": "public"}},
    {"content": "Falcon recovery operations reported by SpaceX teams.", "metadata": {"visibility": "public"}},
    {"content": "Falcon mission telemetry archived after the SpaceX review.", "metadata": {"visibility": "public"}},
]
_SEED_ROW_COUNT = len(_CORPUS)


async def _seed(kb: Khora, namespace_id: UUID) -> None:
    """Seed a BM25-searchable corpus and define the BM25 full-text index.

    The skeleton SurrealDB keyword channel returns ``[]`` without a BM25 index,
    so the BM25 full-text index is defined after seeding. Only the BM25 index is
    defined here (the keyword path brute-forces vectors and never uses HNSW); the
    analyzer + index DDL mirrors ``_TEMPORAL_CHUNK_SEARCH_INDEXES``.
    """
    await seed_corpus(_remember(kb), namespace_id, _CORPUS)
    await _store(kb)._conn.execute(
        "DEFINE INDEX IF NOT EXISTS idx_tc_content_ft ON temporal_chunk "
        "FIELDS content SEARCH ANALYZER khora_fulltext BM25;"
    )


# --------------------------------------------------------------------------- #
# BM25 enforcement — a contradiction filter yields an empty keyword lookup, a
# matching filter retains every row (value discrimination).
# --------------------------------------------------------------------------- #


async def test_bm25_channel_enforces_filter_embedded_surrealdb(kb: Khora, namespace_id: UUID) -> None:
    """The BM25 channel narrows by metadata VALUE, not all-or-nothing.

    The skeleton SurrealDB ``search_fulltext`` now compiles ``filter_ast`` and
    ANDs it into the BM25 ``WHERE``. Mirrors the sqlite_lance sibling's
    enforcement contract; see the module docstring for why this drives
    ``search_fulltext`` directly rather than ``recall(mode=KEYWORD)`` on this
    backend.

    The filters are DOTTED metadata predicates on ``metadata.visibility`` — every
    seeded row carries ``visibility == "public"``, so the predicate descends the
    real FLEXIBLE ``metadata_`` column. A dotted metadata key is chosen
    deliberately over a bare system key (e.g. ``source_name``): a bare key that is
    not a column on the ``temporal_chunk`` schema would exclude every row by
    NONE-comparison regardless of value, which cannot distinguish "the filter
    discriminates by value" from "the compiled clause nukes everything".

    Three checks pin value-equivalent semantics:

    * UNFILTERED control — proves the corpus is genuinely BM25-searchable, so the
      contradiction zero below is not vacuous.
    * MATCHING filter (``visibility == "public"``) — RETAINS every row; proves a
      filter is not blanket-excluding.
    * CONTRADICTION filter (``visibility == "confidential"``) — excludes every
      row; the enforcement assertion. Together with the matching case this proves
      the channel narrows by value, not "any filter empties the result".

    The query is the bare token ``"Falcon"`` (present in all three seeded chunks)
    so the BM25 result set is the full corpus and the row counts above reconcile
    exactly.
    """
    await _seed(kb, namespace_id)
    store = _store(kb)

    # Unfiltered control: BM25 finds the whole corpus.
    baseline = await store.search_fulltext(namespace_id, "Falcon", limit=10)
    assert len(baseline) == _SEED_ROW_COUNT, (
        f"unfiltered control: expected the full BM25 corpus ({_SEED_ROW_COUNT} rows), got {len(baseline)} — "
        "the corpus is not fully BM25-searchable, so the discrimination/enforcement assertions below would be vacuous"
    )

    # Discrimination positive: a MATCHING metadata value RETAINS every row.
    match_ast = parse_to_ast(RecallFilter.model_validate({"metadata.visibility": "public"}))
    matched = await store.search_fulltext(namespace_id, "Falcon", limit=10, filter_ast=match_ast)
    assert len(matched) == _SEED_ROW_COUNT, (
        f"matching filter (visibility=public) must RETAIN every seeded row ({_SEED_ROW_COUNT}), got {len(matched)} — "
        "if a matching filter drops rows the contradiction zero would be a blanket-exclude, not value discrimination"
    )

    # Enforcement: a metadata value NO seeded row carries excludes every row.
    contradiction_ast = parse_to_ast(RecallFilter.model_validate({"metadata.visibility": "confidential"}))
    filtered = await store.search_fulltext(namespace_id, "Falcon", limit=10, filter_ast=contradiction_ast)
    assert filtered == [], f"filter-excluded chunks leaked through embedded SurrealDB BM25: {len(filtered)}"
