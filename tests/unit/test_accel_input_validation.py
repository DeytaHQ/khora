"""Input-validation tests for `khora._accel` Rust kernels.

Regression coverage for issue #902: two Rust pyfunctions (`mmr_diversity_select`
and `resolve_entities_enhanced`) previously panicked on parallel-array length
mismatches. Rust panics surface as `pyo3_runtime.PanicException` (a
`BaseException` subclass) which is awkward to catch and signals an internal
bug rather than a user error. The fix returns a `PyResult` and raises
`PyValueError` instead - this file pins that contract.

The tests skip when the Rust accel wheel is unavailable, since the
NumPy / pure-Python fallbacks have always validated lengths in Python.
"""

from __future__ import annotations

import numpy as np
import pytest

import khora._accel as accel

# Skip the whole module when the Rust wheel is unavailable - this regression
# only applies to the Rust kernel.
pytestmark = pytest.mark.skipif(
    not accel._HAS_RUST,
    reason="khora_accel Rust wheel not built; ValueError contract only applies to Rust path",
)


# --------------------------------------------------------------------------
# mmr_diversity_select
# --------------------------------------------------------------------------


class TestMMRInputValidation:
    """Rust `mmr_diversity_select` must raise ValueError on length mismatch."""

    def test_scores_longer_than_embeddings_raises_value_error(self):
        from khora_accel import mmr_diversity_select as rust_mmr

        embeddings = np.random.rand(3, 4).astype(np.float32)
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]  # 5 scores, 3 rows

        with pytest.raises(ValueError, match="scores length 5"):
            rust_mmr(embeddings, scores, 0.5, 2)

    def test_scores_shorter_than_embeddings_raises_value_error(self):
        from khora_accel import mmr_diversity_select as rust_mmr

        embeddings = np.random.rand(5, 4).astype(np.float32)
        scores = [0.1, 0.2]  # 2 scores, 5 rows

        with pytest.raises(ValueError, match="scores length 2"):
            rust_mmr(embeddings, scores, 0.5, 2)

    def test_valid_inputs_succeed(self):
        from khora_accel import mmr_diversity_select as rust_mmr

        embeddings = np.eye(4, dtype=np.float32)
        scores = [0.9, 0.5, 0.7, 0.3]

        result = rust_mmr(embeddings, scores, 0.5, 2)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(0 <= i < 4 for i in result)


# --------------------------------------------------------------------------
# resolve_entities_enhanced
# --------------------------------------------------------------------------


class TestResolveEntitiesEnhancedInputValidation:
    """Rust `resolve_entities_enhanced` must raise ValueError on length mismatch."""

    def test_existing_types_shorter_raises_value_error(self):
        """The original #902 bug: types list shorter than names list -> indexing panic."""
        from khora_accel import resolve_entities_enhanced as rust_resolve

        with pytest.raises(ValueError, match="existing_types length 1"):
            rust_resolve(
                ["alice", "bob"],  # new_names
                ["PERSON", "PERSON"],  # new_types
                ["alice", "carol"],  # existing_names (len 2)
                [[], []],  # existing_aliases (len 2)
                ["PERSON"],  # existing_types (len 1 - MISMATCH)
                [],  # type_thresholds_keys
                [],  # type_thresholds_vals
                0.85,  # default_threshold
            )

    def test_existing_aliases_mismatch_raises_value_error(self):
        from khora_accel import resolve_entities_enhanced as rust_resolve

        with pytest.raises(ValueError, match="existing_aliases length 1"):
            rust_resolve(
                ["alice"],
                ["PERSON"],
                ["alice", "carol"],  # existing_names (len 2)
                [[]],  # existing_aliases (len 1 - MISMATCH)
                ["PERSON", "PERSON"],  # existing_types (len 2)
                [],
                [],
                0.85,
            )

    def test_new_names_types_mismatch_raises_value_error(self):
        from khora_accel import resolve_entities_enhanced as rust_resolve

        with pytest.raises(ValueError, match="new_names length 2"):
            rust_resolve(
                ["alice", "bob"],  # new_names (len 2)
                ["PERSON"],  # new_types (len 1 - MISMATCH)
                ["alice"],
                [[]],
                ["PERSON"],
                [],
                [],
                0.85,
            )

    def test_threshold_kv_mismatch_raises_value_error(self):
        from khora_accel import resolve_entities_enhanced as rust_resolve

        with pytest.raises(ValueError, match="type_thresholds_keys length 2"):
            rust_resolve(
                ["alice"],
                ["PERSON"],
                ["alice"],
                [[]],
                ["PERSON"],
                ["PERSON", "DATE"],  # keys (len 2)
                [0.92],  # vals (len 1 - MISMATCH)
                0.85,
            )

    def test_valid_inputs_succeed(self):
        from khora_accel import resolve_entities_enhanced as rust_resolve

        result = rust_resolve(
            ["alice"],  # new_names
            ["PERSON"],  # new_types
            ["alice", "carol"],  # existing_names
            [[], []],  # existing_aliases
            ["PERSON", "PERSON"],  # existing_types
            [],
            [],
            0.85,
        )
        assert len(result) == 1
        # Should resolve to existing alice (index 0) as exact match
        assert result[0] is not None
        idx, score, match_type = result[0]
        assert idx == 0
        assert score == pytest.approx(1.0)
        assert match_type == "exact"


# --------------------------------------------------------------------------
# Not-a-BaseException sanity check
# --------------------------------------------------------------------------


def test_mmr_error_is_catchable_as_exception():
    """A regular `except Exception:` clause must catch the failure.

    `pyo3_runtime.PanicException` inherits from `BaseException` directly, so a
    `try / except Exception` will miss it. This test pins that the new error
    is a normal `Exception`.
    """
    from khora_accel import mmr_diversity_select as rust_mmr

    embeddings = np.random.rand(3, 4).astype(np.float32)
    scores = [0.1, 0.2]

    try:
        rust_mmr(embeddings, scores, 0.5, 2)
        pytest.fail("expected ValueError")
    except Exception as e:
        assert isinstance(e, ValueError), f"got {type(e).__name__}: {e}"


def test_resolve_enhanced_error_is_catchable_as_exception():
    from khora_accel import resolve_entities_enhanced as rust_resolve

    try:
        rust_resolve(
            ["alice"],
            ["PERSON"],
            ["alice", "carol"],
            [[], []],
            ["PERSON"],
            [],
            [],
            0.85,
        )
        pytest.fail("expected ValueError")
    except Exception as e:
        assert isinstance(e, ValueError), f"got {type(e).__name__}: {e}"
