"""Shadow-scoring A/B harness tests (#1479).

Verifies the observe-only contract:

* **Flag OFF (default)** - no ``engine_info["shadow_scoring"]`` key, and the
  returned chunk order is unchanged.
* **Flag ON** - the key is present with a well-formed divergence report, and
  the returned chunk order is BYTE-IDENTICAL to the flag-OFF run (shadow never
  reorders the live result).

Hermetic: sqlite_lance + the deterministic vocab embedder/extractor from the
golden-set module (no network, no LLM).
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

from tests.integration.golden_set.test_golden_set_recall import (
    _config,
    _ingest_corpus,
    _load_corpus,
    _patch_deterministic_llm,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.embedded,
    pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed"),
]

# A hybrid, multi-entity query that exercises fusion (vector + graph channels)
# so the incumbent order and the score-sort candidate order can genuinely
# diverge - the whole point of the harness.
_QUERY = "How do the payments platform project and the search feature roadmap relate for Alice and Bob?"


async def test_shadow_off_then_on_same_ingest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Flag OFF = no key + unchanged results; flag ON = key present, results identical.

    Toggling the process-level shadow flag between two recalls on the SAME
    namespace (one ingest) isolates the observe-only guarantee from ingest
    non-determinism: the corpus, query and every chunk UUID are identical, so a
    difference in the returned order could ONLY come from the shadow path. It
    must not - shadow is observe-only. (The recall result cache is default-OFF,
    so the second recall genuinely re-runs rather than serving a cached OFF
    result.)
    """
    from khora import Khora, SearchMode

    corpus = _load_corpus()
    dim = _patch_deterministic_llm(monkeypatch, corpus)

    config = _config(tmp_path, dim)
    config.query.shadow_scoring = False

    tmp_path.mkdir(parents=True, exist_ok=True)
    async with Khora(config, run_migrations=True) as kb:
        ns_id, _ = await _ingest_corpus(kb, corpus)

        # Flag OFF (default): no shadow key, capture the baseline order.
        off = await kb.recall(_QUERY, namespace=ns_id, limit=10, mode=SearchMode.HYBRID)
        assert "shadow_scoring" not in off.engine_info, (
            "shadow-scoring must be genuinely absent when the flag is OFF (zero-cost default)"
        )
        off_order = [str(c.id) for c in off.chunks]

        # Flip the flag ON and re-recall the SAME query on the SAME data.
        config.query.shadow_scoring = True
        on = await kb.recall(_QUERY, namespace=ns_id, limit=10, mode=SearchMode.HYBRID)

    on_order = [str(c.id) for c in on.chunks]
    assert on_order == off_order, (
        "shadow scoring changed the RETURNED chunk order - it must be observe-only.\n"
        f"  off: {off_order}\n  on:  {on_order}"
    )

    report = on.engine_info.get("shadow_scoring")
    assert report is not None, "flag ON must emit engine_info['shadow_scoring']"

    # Report is well-formed and JSON-serializable.
    assert report["strategy"] == "score_sort"
    assert report["candidate_count"] == len(on.chunks)
    assert report["topk"] >= 1
    assert 0.0 <= report["topk_overlap"] <= 1.0
    assert 0.0 <= report["topk_doc_overlap"] <= 1.0
    assert isinstance(report["moved"], list)
    assert isinstance(report["identical"], bool)
    rho = report["spearman_rho"]
    assert rho is None or (-1.0 <= rho <= 1.0)
    # Every mover names a real candidate with a coherent rank delta.
    for mover in report["moved"]:
        assert mover["delta"] == mover["candidate_rank"] - mover["incumbent_rank"]
        assert mover["delta"] != 0

    import json

    json.dumps(report)  # must not raise


async def test_shadow_identity_strategy_is_a_noop_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The 'identity' candidate equals the incumbent: rho=1.0, no movers."""
    from khora import Khora, SearchMode

    corpus = _load_corpus()
    dim = _patch_deterministic_llm(monkeypatch, corpus)

    config = _config(tmp_path, dim)
    config.query.shadow_scoring = True
    config.query.shadow_scoring_strategy = "identity"

    tmp_path.mkdir(parents=True, exist_ok=True)
    async with Khora(config, run_migrations=True) as kb:
        ns_id, _ = await _ingest_corpus(kb, corpus)
        result = await kb.recall(_QUERY, namespace=ns_id, limit=10, mode=SearchMode.HYBRID)

    report = result.engine_info["shadow_scoring"]
    assert report["strategy"] == "identity"
    assert report["identical"] is True
    assert report["moved"] == []
    # rho is 1.0 when >= 2 candidates, else None (undefined for < 2).
    assert report["spearman_rho"] in (1.0, None)
