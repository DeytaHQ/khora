"""Deprecated shim — moved to :mod:`khora.storage.temporal.pgvector`.

Importing from this path still works but is deprecated. Import from
``khora.storage.temporal.pgvector`` instead. All names below are re-exported
verbatim (same objects), so identity is preserved across both paths.
"""

from __future__ import annotations

import warnings

from khora.storage.temporal.pgvector import (
    PgVectorTemporalStore,
    khora_chunks_table,
)

warnings.warn(
    "Importing from khora.engines.skeleton.backends.pgvector is deprecated; "
    "import from khora.storage.temporal.pgvector instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "PgVectorTemporalStore",
    "khora_chunks_table",
]
