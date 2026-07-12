"""Recall-parity fingerprint for the #1480 retriever refactor.

The golden-set rank tests (``test_golden_set_recall.py``) pin LOOSE rank
positions - they catch a demotion but tolerate score / tie reshuffling. The
#1480 refactor (channel protocol, scoring-as-stages, shared post-processor)
must preserve recall behavior: the same MULTISET of returned chunks with the
same display scores, the same entity / relationship projection, and the same
abstention / confidence / channel-count engine_info. This module captures a
canonical fingerprint over the golden corpus and asserts it against a
checked-in snapshot, so any observable drift fails loudly during the refactor.

Tie-order caveat (why the fingerprint is CANONICALIZED, not raw order): recall
ranks on the pre-``attach_relevance_scores`` RRF score, which produces exact
ties at k=60. Ties resolve by list-insertion order, and the insertion order of
a fresh ingest is not stable across ingests (row / extraction ordering varies).
The RETURNED order at a tied rank is therefore already non-deterministic in the
baseline - it is not a property the code has, so we must not assert it here or
the guard flakes on unmodified code. Instead:

* the fingerprint sorts chunks by ``(score, doc)`` so it compares the MULTISET
  of (doc, display-score) pairs - stable across ingests, and any added / dropped
  chunk or changed score still trips it;
* a separate test asserts the RETURNED order is deterministic WITHIN one ingest
  (repeated recalls), which is the property the code does have and the refactor
  must preserve.

Reuses the deterministic (no-LLM, no-network) sqlite_lance harness from
``test_golden_set_recall.py``. The snapshot lives in
``recall_parity_snapshot.json`` next to this file; regenerate it intentionally
with ``KHORA_REGEN_PARITY=1`` only when a behavior change is deliberate and
reviewed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from tests.integration.golden_set.test_golden_set_recall import (
    _HAS_EMBEDDED,
    _seed_and_recall,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.embedded,
    pytest.mark.skipif(not _HAS_EMBEDDED, reason="requires aiosqlite + lancedb"),
]

_SNAPSHOT_PATH = Path(__file__).with_name("recall_parity_snapshot.json")

# engine_info keys whose values are stable across runs and are part of the
# observable recall contract we must preserve byte-for-byte. Excludes
# free-form / volatile keys (timings, version_history object identity) and any
# key carrying a UUID that changes per ingest.
_STABLE_ENGINE_INFO_KEYS = (
    "engine",
    "mode",
    "channels_used",
    "rrf_k",
    "routing",
    "use_graph",
    "graph_depth",
    "raw_chunk_count",
    "validated_chunk_count",
    "confidence",
    "abstention_signals",
    "temporal_category",
    "temporal_confidence",
    "is_temporal",
    "retrieval_mean_score",
    "retrieval_score_variance",
    "retrieval_top_score_gap",
    "vector_chunk_count",
    "graph_chunk_count",
    "bm25_chunk_count",
    "entry_entities",
    "expanded_entities",
    "adaptive_depth_applied",
    "ppr_path_used",
    "session_aware_activated",
    "max_raw_vector_score",
)


def _round(value: Any) -> Any:
    """Round floats to a stable precision so cosmetic FP noise doesn't fail."""
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_round(v) for v in value]
    return value


def _fingerprint_result(result: Any, id_map: dict[str, UUID]) -> dict[str, Any]:
    """Canonical, id-map-relative fingerprint of one RecallResult.

    Document / chunk / entity UUIDs change every ingest, so we map document ids
    back to the stable corpus doc_id where possible and index chunks by rank.
    The fingerprint captures: ordered chunk (doc_id, rounded score), ordered
    entity (name, rounded score), ordered relationship (type, rounded score),
    ordered document stub (doc_id, source_type), and the stable engine_info slice.

    Coverage note: this corpus recalls vector-only on ``sqlite_lance``
    (``SearchMode.HYBRID`` with no graph backend), so ``entities`` /
    ``relationships`` are empty here and ``project_entities`` /
    ``project_relationships`` are covered by the unit suite
    (``tests/unit/core/test_recall_projection.py``), not this end-to-end net. The
    chunk-derived ``documents`` stub projection IS exercised here.
    """
    doc_to_corpus = {v: k for k, v in id_map.items()}

    # Canonicalize chunk order by (score, doc): the returned order at a tied RRF
    # rank is non-deterministic across ingests in the baseline (see module
    # docstring), so we compare the multiset. A dropped / added chunk or a
    # changed score still trips it. Entities / relationships / documents are
    # sorted the same way for the same reason.
    chunks = sorted(
        (
            {
                "doc": doc_to_corpus.get(c.document_id, str(c.document_id)),
                "score": _round(c.score),
            }
            for c in result.chunks
        ),
        key=lambda d: (d["score"], d["doc"]),
    )
    entities = sorted(
        ({"name": e.name, "score": _round(e.score)} for e in result.entities),
        key=lambda d: (d["score"], d["name"]),
    )
    relationships = sorted(
        ({"type": r.relationship_type, "score": _round(r.score)} for r in result.relationships),
        key=lambda d: (d["score"], d["type"]),
    )
    # Document-stub projection (the third extracted #1480 function). Fingerprint
    # the stable fields only: the corpus doc id + source_type. ``created_at`` is
    # excluded because entity/rel-referenced stubs stamp ``datetime.now(UTC)``
    # (volatile); the chunk-derived docs carry a stable created_at but mixing the
    # two would flake. Sorted by (doc, source_type) so the multiset is compared.
    documents = sorted(
        (
            {
                "doc": doc_to_corpus.get(d.id, str(d.id)),
                "source_type": d.source_type,
            }
            for d in result.documents
        ),
        key=lambda d: (d["doc"], d["source_type"]),
    )

    engine_info = {}
    for key in _STABLE_ENGINE_INFO_KEYS:
        if key in result.engine_info:
            engine_info[key] = _round(result.engine_info[key])

    return {
        "chunks": chunks,
        "entities": entities,
        "relationships": relationships,
        "documents": documents,
        "engine_info": engine_info,
    }


async def _compute_fingerprint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from tests.test_helpers.diagnostics import assert_no_silent_degradation

    corpus, results, id_map = await _seed_and_recall(tmp_path, monkeypatch)
    fingerprint: dict[str, Any] = {}
    for spec in corpus["queries"]:
        qid = spec["query_id"]
        # ADR-001: the fingerprint deliberately excludes ``degradations`` (it is
        # order-volatile), so guard here that the golden baseline reflects HEALTHY
        # scoring - a silently-degraded channel in the fixture would otherwise be
        # snapshotted as "golden" without anyone noticing.
        assert_no_silent_degradation(results[qid])
        fingerprint[qid] = _fingerprint_result(results[qid], id_map)
    return fingerprint


async def test_recall_fingerprint_matches_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Recall output over the golden corpus must match the checked-in snapshot.

    This is the #1480 parity net: chunk order + display scores + entity /
    relationship order + stable engine_info. A refactor that changes any
    observable recall output trips this. Regenerate deliberately with
    ``KHORA_REGEN_PARITY=1`` only for a reviewed behavior change.
    """
    fingerprint = await _compute_fingerprint(tmp_path, monkeypatch)

    if os.environ.get("KHORA_REGEN_PARITY") == "1":
        _SNAPSHOT_PATH.write_text(json.dumps(fingerprint, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pytest.skip("regenerated recall parity snapshot (KHORA_REGEN_PARITY=1)")

    assert _SNAPSHOT_PATH.exists(), (
        f"parity snapshot missing at {_SNAPSHOT_PATH}; generate it once with "
        f"KHORA_REGEN_PARITY=1 before running the refactor"
    )
    expected = json.loads(_SNAPSHOT_PATH.read_text(encoding="utf-8"))

    # Compare per query so a failure names exactly which archetype drifted.
    mismatches: list[str] = []
    for qid, expected_fp in expected.items():
        actual_fp = fingerprint.get(qid)
        if actual_fp != expected_fp:
            mismatches.append(
                f"[{qid}] recall fingerprint changed:\n"
                f"  expected: {json.dumps(expected_fp, sort_keys=True)}\n"
                f"  actual:   {json.dumps(actual_fp, sort_keys=True)}"
            )
    new_queries = set(fingerprint) - set(expected)
    if new_queries:
        mismatches.append(f"new queries not in snapshot: {sorted(new_queries)}")

    assert not mismatches, (
        "#1480 recall-parity regression - the refactor changed observable recall output:\n" + "\n".join(mismatches)
    )
