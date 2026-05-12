"""Embedding-defense tests for SQLiteLanceVectorAdapter.

LanceDB silently accepts NaN-containing vectors (poisoning ANN training)
and PyArrow raises a deep schema error for wrong-dim or empty vectors that
users struggle to trace back to their embedder. The adapter now rejects
these three classes with a typed ``EmbeddingError`` before reaching the
PyArrow / LanceDB boundary.
"""

from __future__ import annotations

import math

import pytest

from khora.exceptions import EmbeddingError
from khora.storage.backends.sqlite_lance.vector import _validate_embedding


@pytest.mark.unit
class TestValidateEmbedding:
    def test_valid_embedding_passes(self) -> None:
        _validate_embedding([1.0, 0.0, 0.0, 0.0], expected_dim=4, context="test")

    def test_empty_embedding_raises(self) -> None:
        with pytest.raises(EmbeddingError, match="empty embedding"):
            _validate_embedding([], expected_dim=4, context="test")

    def test_wrong_dim_raises_with_helpful_message(self) -> None:
        with pytest.raises(EmbeddingError, match=r"dim=3 but storage configured for dim=4"):
            _validate_embedding([1.0, 0.0, 0.0], expected_dim=4, context="test")

    def test_wrong_dim_mentions_config_key(self) -> None:
        with pytest.raises(EmbeddingError, match="config.storage.embedding_dimension"):
            _validate_embedding([1.0] * 1536, expected_dim=32, context="test")

    def test_nan_at_zero_raises_with_index(self) -> None:
        with pytest.raises(EmbeddingError, match="contains NaN at index 0"):
            _validate_embedding([math.nan, 0.0, 0.0, 0.0], expected_dim=4, context="test")

    def test_nan_in_middle_raises_with_correct_index(self) -> None:
        vec = [1.0, 0.5, math.nan, 0.0]
        with pytest.raises(EmbeddingError, match="contains NaN at index 2"):
            _validate_embedding(vec, expected_dim=4, context="test")

    def test_context_propagates_into_error_message(self) -> None:
        """The ``context`` arg should appear so users can locate the bad embedding."""
        with pytest.raises(EmbeddingError, match=r"chunk id=abc-123"):
            _validate_embedding([], expected_dim=4, context="chunk id=abc-123")

    @pytest.mark.parametrize(
        "bad_value",
        [math.nan, float("nan")],
    )
    def test_all_nan_representations_caught(self, bad_value: float) -> None:
        with pytest.raises(EmbeddingError, match="NaN"):
            _validate_embedding([1.0, bad_value, 0.0, 0.0], expected_dim=4, context="test")

    def test_infinity_is_allowed(self) -> None:
        """``math.isnan`` doesn't catch Inf; the validator's job is NaN-only. Inf
        is unusual but doesn't corrupt ANN training the way NaN does — leave it
        to LanceDB's own handling rather than over-policing here.
        """
        _validate_embedding([float("inf"), 0.0, 0.0, 0.0], expected_dim=4, context="test")
