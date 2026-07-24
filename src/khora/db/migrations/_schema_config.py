"""Runtime-config helpers for schema DDL (embedding dimension / halfvec).

Imported by migration version modules to size the pgvector embedding columns
and HNSW indexes from the *configured* embedding dimension instead of a
hardcoded ``1536`` constant. ``run_migrations()`` injects the effective
dimension (``llm.embedding_dimension``) and the ``use_halfvec`` flag into
``config.attributes``; these helpers read them back.

This lives in its own module (not ``env.py``) on purpose: ``env.py`` runs the
Alembic migration environment as a side effect of import, so it cannot be
imported safely by version modules. This module has no import-time side
effects — it only touches the Alembic ``context`` from inside the functions,
which are called during ``upgrade()`` when the context is configured.
"""

from __future__ import annotations

from alembic import context

# Default embedding dimension. Preserves the historical ``vector(1536)`` schema
# for CLI/standalone ``alembic`` runs and for existing databases (which already
# applied these revisions and never re-run them). Only FRESH creates pick up an
# injected, non-default dimension.
DEFAULT_EMBEDDING_DIMENSION = 1536

# pgvector index dimension ceilings. The ``vector`` HNSW opclass caps at 2000
# dims; the ``halfvec`` opclass caps at 4000. Above 2000 only the halfvec
# expression index (migration 018) can be built, so the full-precision
# ``vector`` HNSW indexes (migrations 002 / 005 / 007) are skipped there.
VECTOR_HNSW_MAX_DIM = 2000
HALFVEC_HNSW_MAX_DIM = 4000


def _attr(name: str, default: object) -> object:
    """Read an injected Alembic attribute, defaulting when absent/unavailable."""
    try:
        value = context.config.attributes.get(name)
    except Exception:
        return default
    return default if value is None else value


def configured_embedding_dimension() -> int:
    """Effective embedding dimension for schema DDL (default ``1536``)."""
    return int(_attr("embedding_dimension", DEFAULT_EMBEDDING_DIMENSION))  # type: ignore[arg-type]


def configured_use_halfvec() -> bool:
    """Whether halfvec HNSW indexes should be created (default ``True``)."""
    return bool(_attr("use_halfvec", True))


def full_precision_hnsw_supported() -> bool:
    """True when a full-precision ``vector`` HNSW index is buildable.

    pgvector caps the ``vector`` HNSW opclass at 2000 dims; above that only the
    ``halfvec`` expression index (migration 018) can be built.
    """
    return configured_embedding_dimension() <= VECTOR_HNSW_MAX_DIM
