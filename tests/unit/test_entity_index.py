"""Tests for EntityIndex — the in-memory blocking index for entity resolution."""

from __future__ import annotations

from uuid import uuid4

import pytest

from khora._accel import cosine_similarity, levenshtein_similarity, normalize_entity_name
from khora.core.models.entity import Entity
from khora.extraction.expansion.entity_index import (
    EntityIndex,
    _tokenize,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entity(
    name: str = "Test Entity",
    entity_type: str = "PERSON",
    namespace_id=None,
    embedding: list[float] | None = None,
    confidence: float = 1.0,
) -> Entity:
    return Entity(
        id=uuid4(),
        namespace_id=namespace_id or uuid4(),
        name=name,
        entity_type=entity_type,
        description=f"Description of {name}",
        embedding=embedding,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------


class TestNormalizeName:
    def test_lowercase_strip(self):
        assert normalize_entity_name("  Hello World  ") == "hello world"

    def test_empty(self):
        assert normalize_entity_name("") == ""


class TestTokenize:
    def test_basic(self):
        tokens = _tokenize("John Smith")
        assert tokens == {"john", "smith"}

    def test_filters_short_tokens(self):
        tokens = _tokenize("A B CD EF")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "cd" in tokens
        assert "ef" in tokens

    def test_strips_punctuation(self):
        tokens = _tokenize("hello-world! foo.bar")
        assert "helloworld" in tokens
        assert "foobar" in tokens


class TestCosineSimilarity:
    def test_identical(self):
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_mismatched_length(self):
        assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0

    def test_zero_vector(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


class TestLevenshteinSimilarity:
    def test_identical(self):
        assert levenshtein_similarity("hello", "hello") == 1.0

    def test_completely_different(self):
        sim = levenshtein_similarity("abc", "xyz")
        assert sim < 0.5

    def test_close(self):
        sim = levenshtein_similarity("kitten", "kittens")
        assert sim > 0.8

    def test_empty(self):
        assert levenshtein_similarity("", "abc") == 0.0
        assert levenshtein_similarity("abc", "") == 0.0


# ---------------------------------------------------------------------------
# EntityIndex core
# ---------------------------------------------------------------------------


class TestEntityIndexAdd:
    def test_add_new_entity(self):
        idx = EntityIndex()
        e = _make_entity("Alice")
        result = idx.add(e)
        assert result is None
        assert len(idx) == 1
        assert e.id in idx

    def test_add_duplicate_returns_existing(self):
        idx = EntityIndex()
        e1 = _make_entity("Alice", entity_type="PERSON")
        e2 = _make_entity("alice", entity_type="PERSON")  # same name, diff case
        idx.add(e1)
        result = idx.add(e2)
        assert result is e1
        assert len(idx) == 1

    def test_different_types_not_duplicate(self):
        idx = EntityIndex()
        e1 = _make_entity("Mercury", entity_type="PERSON")
        e2 = _make_entity("Mercury", entity_type="LOCATION")
        idx.add(e1)
        result = idx.add(e2)
        assert result is None  # different type -> not a duplicate
        assert len(idx) == 2

    def test_whitespace_normalization(self):
        idx = EntityIndex()
        e1 = _make_entity("  John Smith  ")
        idx.add(e1)
        e2 = _make_entity("john smith")
        result = idx.add(e2)
        assert result is e1


class TestEntityIndexLookup:
    def test_get_by_id(self):
        idx = EntityIndex()
        e = _make_entity("Alice")
        idx.add(e)
        assert idx.get(e.id) is e

    def test_get_by_name(self):
        idx = EntityIndex()
        e = _make_entity("Alice", entity_type="PERSON")
        idx.add(e)
        assert idx.get_by_name("Alice", "PERSON") is e
        assert idx.get_by_name("alice", "PERSON") is e
        assert idx.get_by_name("Alice", "ORGANIZATION") is None


class TestFuzzyCandidates:
    def test_finds_similar_names(self):
        idx = EntityIndex()
        ns = uuid4()
        e1 = _make_entity("Microsoft Corporation", "ORGANIZATION", ns)
        e2 = _make_entity("Microsoft Corp", "ORGANIZATION", ns)
        e3 = _make_entity("Apple Inc", "ORGANIZATION", ns)
        idx.add(e1)
        idx.add(e2)
        idx.add(e3)

        candidates = idx.find_fuzzy_candidates(e1, threshold=0.6)
        candidate_ids = {c.id for c, _ in candidates}
        assert e2.id in candidate_ids
        assert e3.id not in candidate_ids  # no shared tokens

    def test_respects_type_filter(self):
        idx = EntityIndex()
        ns = uuid4()
        e1 = _make_entity("John", "PERSON", ns)
        e2 = _make_entity("Johns", "ORGANIZATION", ns)
        idx.add(e1)
        idx.add(e2)

        # e2 is a different type, shouldn't appear
        candidates = idx.find_fuzzy_candidates(e1, threshold=0.5)
        assert len(candidates) == 0

    def test_empty_index(self):
        idx = EntityIndex()
        e = _make_entity("Alice")
        assert idx.find_fuzzy_candidates(e) == []

    def test_excludes_exact_matches(self):
        idx = EntityIndex()
        e = _make_entity("Alice")
        idx.add(e)
        # Fuzzy candidates should not include exact matches
        candidates = idx.find_fuzzy_candidates(e)
        assert len(candidates) == 0


class TestEmbeddingCandidates:
    def test_finds_similar_embeddings(self):
        idx = EntityIndex()
        ns = uuid4()
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.99, 0.1, 0.0]  # very similar
        emb3 = [0.0, 0.0, 1.0]  # orthogonal

        e1 = _make_entity("Entity A", "CONCEPT", ns, embedding=emb1)
        e2 = _make_entity("Entity B", "CONCEPT", ns, embedding=emb2)
        e3 = _make_entity("Entity C", "CONCEPT", ns, embedding=emb3)
        idx.add(e1)
        idx.add(e2)
        idx.add(e3)

        candidates = idx.find_embedding_candidates(e1, threshold=0.9)
        candidate_ids = {c.id for c, _ in candidates}
        assert e2.id in candidate_ids
        assert e3.id not in candidate_ids

    def test_no_embedding_returns_empty(self):
        idx = EntityIndex()
        e = _make_entity("Alice")  # no embedding
        idx.add(e)
        assert idx.find_embedding_candidates(e) == []

    def test_excludes_same_type_without_shared_tokens(self):
        """Embedding candidates use token blocking - entities without shared tokens are excluded.

        This is intentional: token blocking gives O(k) instead of O(n) performance.
        Entities must share at least one name token to be considered as candidates.
        """
        idx = EntityIndex()
        ns = uuid4()
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.99, 0.1, 0.0]

        e1 = _make_entity("Alpha Beta", "CONCEPT", ns, embedding=emb1)
        e2 = _make_entity("Gamma Delta", "CONCEPT", ns, embedding=emb2)  # no shared tokens
        idx.add(e1)
        idx.add(e2)

        # Token blocking excludes e2 since it shares no tokens with e1
        candidates = idx.find_embedding_candidates(e1, threshold=0.9)
        candidate_ids = {c.id for c, _ in candidates}
        assert e2.id not in candidate_ids


class TestEntityIndexBulk:
    def test_get_all_entities(self):
        idx = EntityIndex()
        entities = [_make_entity(f"Entity {i}") for i in range(5)]
        for e in entities:
            idx.add(e)
        all_ents = idx.get_all_entities()
        assert len(all_ents) == 5

    def test_get_entities_by_type(self):
        idx = EntityIndex()
        for i in range(3):
            idx.add(_make_entity(f"Person {i}", "PERSON"))
        for i in range(2):
            idx.add(_make_entity(f"Org {i}", "ORGANIZATION"))

        assert len(idx.get_entities_by_type("PERSON")) == 3
        assert len(idx.get_entities_by_type("ORGANIZATION")) == 2
        assert len(idx.get_entities_by_type("CONCEPT")) == 0

    def test_stats(self):
        idx = EntityIndex()
        idx.add(_make_entity("John Smith", "PERSON"))
        idx.add(_make_entity("Jane Doe", "PERSON"))

        stats = idx.stats()
        assert stats["total_entities"] == 2
        assert stats["exact_keys"] == 2
        assert stats["type_groups"] == 1
        assert stats["token_keys"] > 0
