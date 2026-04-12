"""Unit tests for Chronicle engine version-aware scoring."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from khora.core.models import Chunk, ChunkMetadata
from khora.engines.chronicle.engine import _apply_version_scoring, _has_version_intent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    *,
    version: int | None = None,
    entity_refs: list[str] | None = None,
    title: str | None = None,
    document_id=None,
) -> Chunk:
    """Create a minimal Chunk with custom metadata."""
    custom: dict = {}
    if version is not None:
        custom["version"] = version
    if entity_refs is not None:
        custom["entity_refs"] = entity_refs
    if title is not None:
        custom["title"] = title

    doc_id = document_id or uuid4()
    return Chunk(
        id=uuid4(),
        namespace_id=uuid4(),
        document_id=doc_id,
        content="test content",
        metadata=ChunkMetadata(
            document_id=doc_id,
            custom=custom,
        ),
        created_at=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# _has_version_intent
# ---------------------------------------------------------------------------


class TestHasVersionIntent:
    """Tests for the intent detection helper."""

    @pytest.mark.parametrize(
        "query",
        [
            "What is the current status of the deal?",
            "Show me the latest project update",
            "What is the status now?",
            "Get the most recent version",
            "What is the active configuration?",
            "What is the present state?",
            "What's currently happening?",
            "What is the existing setup?",
        ],
    )
    def test_detects_version_intent(self, query: str):
        assert _has_version_intent(query) is True

    @pytest.mark.parametrize(
        "query",
        [
            "Tell me about the company history",
            "Who founded this organisation?",
            "What happened in March?",
            "How many employees does the company have?",
        ],
    )
    def test_no_version_intent(self, query: str):
        assert _has_version_intent(query) is False


# ---------------------------------------------------------------------------
# _apply_version_scoring
# ---------------------------------------------------------------------------


class TestApplyVersionScoring:
    """Tests for the version-aware scoring function."""

    def test_empty_list_returns_empty(self):
        assert _apply_version_scoring([], "current status") == []

    def test_no_version_intent_passthrough(self):
        """Without intent keywords the list is returned unchanged."""
        c1 = _make_chunk(version=1, entity_refs=["acme"])
        c2 = _make_chunk(version=2, entity_refs=["acme"])
        original = [(c1, 0.9), (c2, 0.8)]
        result = _apply_version_scoring(original, "history of the company")
        # Scores should be identical (no penalty applied)
        assert [s for _, s in result] == [0.9, 0.8]

    def test_no_version_metadata_passthrough(self):
        """Chunks without version metadata are left untouched."""
        c1 = _make_chunk(entity_refs=["acme"])
        c2 = _make_chunk(entity_refs=["acme"])
        original = [(c1, 0.9), (c2, 0.8)]
        result = _apply_version_scoring(original, "current status")
        assert [s for _, s in result] == [0.9, 0.8]

    def test_latest_version_not_penalized(self):
        """The chunk with the max version keeps its full score."""
        c_v1 = _make_chunk(version=1, entity_refs=["acme"])
        c_v3 = _make_chunk(version=3, entity_refs=["acme"])
        original = [(c_v3, 0.8), (c_v1, 0.9)]
        result = _apply_version_scoring(original, "current status")

        scores = {chunk.metadata.custom["version"]: score for chunk, score in result}
        # v3 keeps full score: 0.8 * (3/3)**0.5 = 0.8
        assert scores[3] == pytest.approx(0.8)
        # v1 penalized: 0.9 * (1/3)**0.5 ~ 0.5196
        assert scores[1] == pytest.approx(0.9 * (1 / 3) ** 0.5)

    def test_older_version_demoted_below_newer(self):
        """An older version with a higher raw score should be re-ranked below the newer version."""
        c_v1 = _make_chunk(version=1, entity_refs=["acme"])
        c_v5 = _make_chunk(version=5, entity_refs=["acme"])
        # v1 starts with a higher score
        original = [(c_v1, 1.0), (c_v5, 0.6)]
        result = _apply_version_scoring(original, "What is the latest status?")

        # After penalty: v1 = 1.0 * (1/5)**0.5 ~ 0.447; v5 = 0.6
        assert result[0][0].metadata.custom["version"] == 5
        assert result[1][0].metadata.custom["version"] == 1

    def test_different_entity_groups_independent(self):
        """Version scoring is per-entity-group, not global."""
        c_acme_v1 = _make_chunk(version=1, entity_refs=["acme"])
        c_acme_v2 = _make_chunk(version=2, entity_refs=["acme"])
        c_beta_v1 = _make_chunk(version=1, entity_refs=["beta"])

        original = [(c_acme_v1, 0.9), (c_acme_v2, 0.8), (c_beta_v1, 0.7)]
        result = _apply_version_scoring(original, "current status")

        scores = {}
        for chunk, score in result:
            key = (chunk.metadata.custom["entity_refs"][0], chunk.metadata.custom["version"])
            scores[key] = score

        # acme v2 is max -> no penalty
        assert scores[("acme", 2)] == pytest.approx(0.8)
        # acme v1 penalized: 0.9 * (1/2)**0.5
        assert scores[("acme", 1)] == pytest.approx(0.9 * (1 / 2) ** 0.5)
        # beta v1 is max in its group -> no penalty
        assert scores[("beta", 1)] == pytest.approx(0.7)

    def test_fallback_to_title_grouping(self):
        """Without entity_refs, chunks are grouped by title."""
        c_v1 = _make_chunk(version=1, title="Acme Deal Record")
        c_v2 = _make_chunk(version=2, title="Acme Deal Record")
        original = [(c_v1, 0.9), (c_v2, 0.8)]
        result = _apply_version_scoring(original, "current status")

        scores = {chunk.metadata.custom["version"]: score for chunk, score in result}
        assert scores[2] == pytest.approx(0.8)
        assert scores[1] == pytest.approx(0.9 * (1 / 2) ** 0.5)

    def test_result_is_sorted_descending(self):
        """Output should be sorted by descending score."""
        c_v1 = _make_chunk(version=1, entity_refs=["x"])
        c_v2 = _make_chunk(version=2, entity_refs=["x"])
        c_v3 = _make_chunk(version=3, entity_refs=["x"])
        original = [(c_v1, 0.95), (c_v2, 0.90), (c_v3, 0.85)]
        result = _apply_version_scoring(original, "current status")
        result_scores = [s for _, s in result]
        assert result_scores == sorted(result_scores, reverse=True)
