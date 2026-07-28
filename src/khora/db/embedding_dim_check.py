"""Guard against a configured embedding dimension that disagrees with the database.

Since #1260 the pgvector embedding columns and HNSW indexes are sized from the
configured (embedder-facing) dimension at migration time. Alembic tracks
revision ids, not migration bodies, so an *existing* database keeps whatever
width it was created with - a pgvector column cannot be resized in place while
it holds data. That leaves two ways for config and schema to drift apart:

1. An existing database created at one dimension is later opened by a process
   configured for a different one. Writes would then bind vectors at the new
   width against columns declared at the old one.
2. A partially-migrated database (or a future migration that adds an embedding
   column) would create the *new* column at the configured dimension beside
   older columns at the original one, yielding a mixed-width schema.

Both used to be impossible because the width was the hardcoded ``vector(1536)``
constant. Neither is caught by the config guard, which only validates the
dimension against pgvector's index ceilings, not against the live schema. So we
compare the configured dimension to the dimensions actually declared on the
database's pgvector columns and fail fast with an actionable message instead of
surfacing a confusing bind/cast error on the first write (or silently building a
mixed-width schema).

The check is Postgres-only and skips a database that has no pgvector columns yet
(a fresh database, where migrations are about to create them at the configured
dimension).
"""

from __future__ import annotations

import re

# Every pgvector column in the public schema that declares a dimension. Matching
# on the type rather than a hardcoded table list means a column added by a future
# migration is covered without touching this module.
COLUMN_DIMS_SQL = """
SELECT c.relname AS table_name,
       a.attname  AS column_name,
       format_type(a.atttypid, a.atttypmod) AS column_type
FROM pg_attribute a
JOIN pg_class c ON a.attrelid = c.oid
JOIN pg_namespace n ON c.relnamespace = n.oid
WHERE n.nspname = 'public'
  AND c.relkind = 'r'
  AND a.attnum > 0
  AND NOT a.attisdropped
  AND format_type(a.atttypid, a.atttypmod) ~ '^(vector|halfvec)\\([0-9]+\\)$'
"""

_DECLARED_DIM_RE = re.compile(r"^(?:vector|halfvec)\((\d+)\)$")


def parse_declared_dim(column_type: str) -> int | None:
    """Return the declared dimension of a pgvector column type, else ``None``.

    ``format_type`` renders a dimensioned column as ``vector(1536)`` /
    ``halfvec(1536)``. A column declared without a dimension renders as bare
    ``vector``, which carries no width to compare against.
    """
    match = _DECLARED_DIM_RE.match(column_type.strip())
    return int(match.group(1)) if match else None


def describe_dimension_mismatch(
    rows: list[tuple[str, str, str]],
    configured_dimension: int,
) -> str | None:
    """Return an actionable error message when the schema disagrees, else ``None``.

    Args:
        rows: ``(table_name, column_name, column_type)`` triples from
            ``COLUMN_DIMS_SQL``.
        configured_dimension: The effective embedding dimension
            (``KhoraConfig.get_effective_embedding_dimension()``).

    Returns:
        ``None`` when every dimensioned pgvector column matches (including the
        no-columns-yet case), otherwise a message naming the offending columns.
    """
    mismatched: dict[str, int] = {}
    for table_name, column_name, column_type in rows:
        declared = parse_declared_dim(column_type)
        if declared is not None and declared != configured_dimension:
            mismatched[f"{table_name}.{column_name}"] = declared

    if not mismatched:
        return None

    found = ", ".join(f"{name}=vector({dim})" for name, dim in sorted(mismatched.items()))
    existing_dims = sorted(set(mismatched.values()))
    existing = existing_dims[0] if len(existing_dims) == 1 else f"one of {existing_dims}"
    return (
        f"Embedding dimension mismatch: this database's pgvector columns are declared at "
        f"{existing}, but the configured embedding dimension is {configured_dimension} "
        f"({found}). A pgvector column cannot be resized in place while it holds data, so "
        f"writes and recall would fail against the existing columns. Either set the "
        f"embedding dimension back to the database's width (llm.embedding_dimension, or "
        f"KHORA_LLM_EMBEDDING_DIMENSION) - keeping it consistent with the embedding model - "
        f"or point at a fresh database and re-embed the corpus at {configured_dimension}."
    )


__all__ = [
    "COLUMN_DIMS_SQL",
    "describe_dimension_mismatch",
    "parse_declared_dim",
]
