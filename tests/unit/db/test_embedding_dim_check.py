"""#1260: the configured embedding dimension is checked against the live schema.

Since the pgvector columns are sized from config at migration time, an existing
database keeps the width it was created with (Alembic never re-runs an applied
migration, and a populated pgvector column cannot be resized in place). These
tests pin the guard that turns that drift into an actionable error instead of an
opaque bind failure on the first write.
"""

from __future__ import annotations

import pytest

from khora.db.embedding_dim_check import describe_dimension_mismatch, parse_declared_dim

pytestmark = pytest.mark.unit


class TestParseDeclaredDim:
    @pytest.mark.parametrize(
        ("column_type", "expected"),
        [
            ("vector(1536)", 1536),
            ("halfvec(3072)", 3072),
            ("vector(4)", 4),
            ("  vector(768)  ", 768),
            ("vector", None),  # declared without a dimension - nothing to compare
            ("halfvec", None),
            ("text", None),
            ("", None),
        ],
    )
    def test_parses_declared_dimension(self, column_type: str, expected: int | None) -> None:
        assert parse_declared_dim(column_type) == expected


class TestDescribeDimensionMismatch:
    def test_fresh_database_has_no_mismatch(self) -> None:
        # No pgvector columns yet: migrations are about to create them at the
        # configured dimension, so there is nothing to disagree with.
        assert describe_dimension_mismatch([], 3072) is None

    def test_matching_dimensions_pass(self) -> None:
        rows = [
            ("chunks", "embedding", "vector(1536)"),
            ("entities", "embedding", "vector(1536)"),
        ]
        assert describe_dimension_mismatch(rows, 1536) is None

    def test_mismatch_names_columns_and_both_dimensions(self) -> None:
        rows = [
            ("chunks", "embedding", "vector(1536)"),
            ("entities", "embedding", "vector(1536)"),
        ]
        message = describe_dimension_mismatch(rows, 3072)
        assert message is not None
        assert "1536" in message
        assert "3072" in message
        assert "chunks.embedding" in message
        assert "entities.embedding" in message
        # Actionable: names the knob and the fresh-database escape hatch.
        assert "llm.embedding_dimension" in message

    def test_mixed_width_schema_reports_all(self) -> None:
        # The partial-migration hazard: older columns at one width, newer at another.
        rows = [
            ("chunks", "embedding", "vector(1536)"),
            ("chronicle_events", "embedding", "vector(3072)"),
        ]
        message = describe_dimension_mismatch(rows, 768)
        assert message is not None
        assert "one of [1536, 3072]" in message

    def test_dimensionless_columns_are_ignored(self) -> None:
        # A bare ``vector`` column carries no width, so it cannot conflict.
        rows = [("chunks", "embedding", "vector")]
        assert describe_dimension_mismatch(rows, 3072) is None

    def test_halfvec_column_is_compared(self) -> None:
        rows = [("chunks", "embedding", "halfvec(1536)")]
        assert describe_dimension_mismatch(rows, 1536) is None
        assert describe_dimension_mismatch(rows, 3072) is not None
