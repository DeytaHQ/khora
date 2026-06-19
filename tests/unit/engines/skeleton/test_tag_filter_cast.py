"""Tag filter literal must be cast to ``VARCHAR[]`` for PostgreSQL.

The ``khora_chunks.tags`` column is ``ARRAY(String)`` (compiles to
``character varying[]``), but asyncpg infers Python ``list[str]`` literals as
``text[]``. PostgreSQL has no ``varchar[] @> text[]`` operator, so without an
explicit cast the engine raises and the query falls back to returning zero
rows. This test pins the compiled SQL so the cast can't regress.
"""

from __future__ import annotations

from sqlalchemy.dialects import postgresql

from khora.storage.temporal import TemporalFilter
from khora.storage.temporal.pgvector import PgVectorTemporalStore


def test_tag_filter_compiles_with_varchar_array_cast() -> None:
    """The ``f.tags`` literal must compile to a ``VARCHAR[]`` cast."""
    store = PgVectorTemporalStore.__new__(PgVectorTemporalStore)
    conditions = store._build_filter_conditions(TemporalFilter(tags=["group-A", "group-B"]))

    assert len(conditions) == 1
    compiled = conditions[0].compile(
        dialect=postgresql.dialect(),
        compile_kwargs={"literal_binds": True},
    )
    sql = str(compiled)

    # The literal must be wrapped in a CAST to VARCHAR[] so that the
    # ``@>`` (contains) operator matches the column type.
    assert "CAST(" in sql, f"expected explicit CAST in SQL, got: {sql}"
    assert "VARCHAR[]" in sql.upper(), f"expected VARCHAR[] cast in SQL, got: {sql}"
    assert "@>" in sql or "tags" in sql.lower()
