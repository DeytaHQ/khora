"""Row-set recall-filter proof on the embedded sqlite_lance stack — no Docker.

This is the must-always-run lane: it drives the real ``Khora.remember()`` ingest
path and ``Khora.recall(filter=...)`` read path on an embedded SQLite + LanceDB
store, and asserts the filter ACTUALLY NARROWS THE ROWS end to end — the survivor
set reconciled by ``external_id`` equals each conformance case's hand-declared
``expected_ids``. It complements the WIRING spies
(``tests/integration/matrix/test_filter_enforcement_sqlite_lance.py``), which prove
the validated AST reaches each channel; here we assert ROWS, not wiring.

Determinism (no flake): ``stub_llm`` installs a network-free, content-keyed
extractor and a SHA-256-derived embedder, HyDE is ``"never"``, and every test owns
a fresh namespace. Assertions are SET equality over a ``frozenset`` of record ids
(never a ranked list), with a generous ``limit`` so the filter is the only
narrowing force — a vector-score tie can never flip an assertion.

No Docker, no Postgres, no Neo4j: collected and run by the default
``uv run pytest -m e2e`` path. The corpus is curated to the families whose filter
predicate the embedded recall path supports (the ``metadata.*`` predicates the
chunk row carries); system-key-predicate families that only narrow where the chunk
is denormalized run on the live PG/vectorcypher lane.
"""

from __future__ import annotations

from uuid import UUID

import pytest

from khora import Khora
from khora.filter import conformance
from tests.e2e import _harness

pytestmark = pytest.mark.e2e


async def _fresh_namespace(kb: Khora) -> UUID:
    """A fresh namespace on the embedded kb (no cross-test bleed).

    The shared ``namespace_id`` conftest fixture is wired to the ``ENGINE_PARAMS``
    indirection; this embedded module pins the ``sqlite_lance_kb`` engine directly,
    so it mints its own namespace rather than going through that parametrized path.
    """
    ns = await kb.create_namespace()
    return ns.namespace_id


# --------------------------------------------------------------------------- #
# AC1 — deterministic, zero-network ingest produces extractor entities.
# --------------------------------------------------------------------------- #


async def test_deterministic_ingest_produces_entities(sqlite_lance_kb) -> None:
    """A marker-bearing ``remember()`` runs the real pipeline and writes the planned entities.

    The stub extractor emits a fixed entity set for any document containing the
    marker, so the ingest pipeline (extraction -> entity embed -> graph write) lands
    exactly those entities. No network, no API key; the same content seeds the same
    entities on every run.
    """
    from tests.test_helpers.filter_spy import plan_extraction

    namespace_id = await _fresh_namespace(sqlite_lance_kb)
    plan_extraction("falconmark", [("Falcon", "PERSON"), ("Orbit", "ORG")], [("Falcon", "Orbit", "WORKS_ON")])
    await sqlite_lance_kb.remember(
        content="falconmark: Falcon works on Orbit.",
        namespace=namespace_id,
        entity_types=["PERSON", "ORG"],
        relationship_types=["WORKS_ON"],
    )

    entity_count, relationship_count = await _harness.graph_counts(sqlite_lance_kb, namespace_id)
    assert entity_count == 2, f"expected the 2 planned entities, got {entity_count}"
    assert relationship_count >= 1, f"expected the planned relationship, got {relationship_count}"

    # Pin the exact entity identities (normalized name + type), not just the count,
    # so a stray or mis-typed node cannot satisfy the assertion. The namespace is
    # fresh, so the graph holds only what this ingest wrote. The names come back
    # lowercased — proof that the production ``normalize_entity_name`` step really ran
    # (this is the real graph-write path, not a persistence bypass).
    resolved = await sqlite_lance_kb.storage.resolve_namespace(namespace_id)
    persisted = {(e.name, e.entity_type) for e in await sqlite_lance_kb.storage.list_entities(resolved)}
    assert persisted == {("falcon", "PERSON"), ("orbit", "ORG")}, f"persisted entities were {persisted}"


# --------------------------------------------------------------------------- #
# AC4 — doc-level external_id reconciliation over a curated f_*_cases subset.
# --------------------------------------------------------------------------- #


def _metadata_rowset_cases() -> list[conformance.ConformanceCase]:
    """The dotted-path metadata-predicate cases the embedded recall path can narrow on.

    The embedded ``chunks`` row carries the ``metadata`` blob but not the
    denormalized document system keys, so only ``metadata.<path>`` filter predicates
    narrow rows there. We pull the metadata-filtering families (F-COERCE / F-OBJEQ /
    F-DOTKEY) targeting ``sqlite_lance`` and keep only the cases whose seed has no
    duplicate ``external_id`` (the record id is the reconciliation key under
    UNIQUE(namespace_id, external_id)).

    Whole-metadata-blob equality (``{"metadata": {...}}``, ``exercises[1] ==
    "metadata"``) is curated OUT — the embedded JSON path does not support exact
    whole-object equality; it stays covered on the live/conformance lanes.
    """
    cases: list[conformance.ConformanceCase] = []
    for family in (conformance.f_coerce_cases, conformance.f_objeq_cases, conformance.f_dotkey_cases):
        cases.extend(c for c in family() if "sqlite_lance" in c.backends)
    return [
        c
        for c in cases
        if not _harness._has_duplicate_external_id(c.seed_records)
        and c.exercises[1] != "metadata"  # drop whole-blob equality (unsupported on embedded)
    ]


@pytest.mark.parametrize("case", _metadata_rowset_cases(), ids=lambda c: c.id)
async def test_external_id_reconciliation(sqlite_lance_kb, case) -> None:
    """Seed via real ``remember()``; the filter survivors reconcile to the case's expected ids.

    Each record is seeded through the production ingest path with
    ``external_id = record.id``; recall applies ``case.filter`` and the surviving
    documents (those that returned at least one chunk) are reconciled back to record
    ids by ``external_id``. The survivor SET must equal the hand-declared
    ``expected_ids`` — the same target the conformance oracle asserts.
    """
    namespace_id = await _fresh_namespace(sqlite_lance_kb)
    survivors = await _harness.recall_survivors(sqlite_lance_kb, case, namespace_id)

    assert survivors == case.expected_ids
    # Authoring cross-check (NOT the assertion target): the Python oracle agrees.
    assert survivors == conformance.oracle_survivors(case)


# --------------------------------------------------------------------------- #
# AC5 — F-EXISTS presence reachability across the real 8 states.
# --------------------------------------------------------------------------- #


def _exists_metadata_cases() -> list[conformance.ConformanceCase]:
    """The ``f_exists_cases()`` whose predicate is a ``metadata.*`` presence check.

    The embedded recall path narrows on the metadata blob, so the metadata and
    nested-metadata presence states are reconcilable here; the two ``source_name``
    (system-key) states run on the live PG lane. We assert the 8-state corpus has
    not silently shrunk before selecting.
    """
    all_cases = conformance.f_exists_cases()
    assert len(all_cases) == 8, f"f_exists_cases() must expose the 8 presence states, got {len(all_cases)}"
    return [c for c in all_cases if c.exercises[1].startswith("metadata") and "sqlite_lance" in c.backends]


@pytest.mark.parametrize("case", _exists_metadata_cases(), ids=lambda c: c.id)
async def test_f_exists_reachability(sqlite_lance_kb, case) -> None:
    """Each metadata presence state reconciles to its hand-declared survivor set.

    Drives ``$exists`` / present-and-null / null-or-missing predicates over the
    metadata blob through the real ingest + recall path; the survivor set must equal
    ``case.expected_ids``. The absent-vs-present-JSON-null distinction is the
    load-bearing one (a present ``None`` is PRESENT under ``$exists:true``).
    """
    namespace_id = await _fresh_namespace(sqlite_lance_kb)
    survivors = await _harness.recall_survivors(sqlite_lance_kb, case, namespace_id)

    assert survivors == case.expected_ids
    assert survivors == conformance.oracle_survivors(case)


def test_f_exists_states_are_complete() -> None:
    """The 8 presence states are all present (a silent corpus shrink fails here).

    Guards the parametrized reachability test against a future edit that trims
    ``f_exists_cases()`` — the probe would otherwise pass on a smaller set. Pins both
    the exact case ids AND the exercised key-shape / variant tags, so a corpus that
    keeps the count but drops a sub-category (e.g. removes the nested path) still fails.
    """
    cases = conformance.f_exists_cases()
    case_ids = {c.id for c in cases}
    assert case_ids == {
        "F-EXISTS-sys-true",
        "F-EXISTS-sys-false",
        "F-EXISTS-md-true",
        "F-EXISTS-md-false",
        "F-EXISTS-nested-true",
        "F-EXISTS-nested-false",
        "F-EXISTS-present-and-null",
        "F-EXISTS-null-or-missing",
    }
    # Every case is an F-EXISTS case, and the three addressed key shapes are all present.
    assert {c.exercises[0] for c in cases} == {"F-EXISTS"}
    assert {c.exercises[1] for c in cases} == {"source_name", "metadata.mk", "metadata.a.b"}


# --------------------------------------------------------------------------- #
# Determinism + anti-vacuity guards (defense-in-depth).
# --------------------------------------------------------------------------- #


def test_hyde_is_disabled(sqlite_lance_kb) -> None:
    """HyDE is explicitly off, so no LLM rewriting injects candidates into recall.

    A determinism guard: if a future default flip re-enabled query rewriting, the
    row-set proofs could pass with HyDE-generated chunks in the result. Pinning the
    config here makes that regression fail loudly rather than silently.
    """
    assert sqlite_lance_kb._config.query.enable_hyde == "never"


async def test_survivor_chunks_score_nonzero(sqlite_lance_kb) -> None:
    """A surviving chunk carries a non-zero score — the result is not an all-empty accident.

    The set-equality proofs assert WHICH records survive; this guards that the recall
    actually scored and returned real chunks (a degenerate all-zero-score result that
    happened to match ``expected_ids`` would otherwise pass vacuously). One
    representative metadata case is enough to pin the property.
    """
    case = next(c for c in _metadata_rowset_cases() if c.expected_ids)
    namespace_id = await _fresh_namespace(sqlite_lance_kb)
    await _harness.seed_records(sqlite_lance_kb, case.seed_records, namespace_id)
    result = await sqlite_lance_kb.recall(
        case.seed_records[0].content,
        namespace=namespace_id,
        limit=_harness._RECALL_LIMIT,
        min_similarity=0.0,
        filter=case.filter,
    )
    assert result.chunks, "the representative case returned no chunks at all"
    assert any(chunk.score != 0 for chunk in result.chunks), "every returned chunk scored 0 (vacuity risk)"
