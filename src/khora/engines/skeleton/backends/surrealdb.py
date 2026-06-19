"""Deprecated shim — moved to :mod:`khora.storage.temporal.surrealdb`.

Importing from this path still works but is deprecated. Import from
``khora.storage.temporal.surrealdb`` instead. All names below are re-exported
verbatim (same objects), so identity is preserved across both paths.
"""

from __future__ import annotations

import warnings

from khora.storage.temporal.surrealdb import (
    _BACKED_SYSTEM_KEYS,
    _TEMPORAL_CHUNK_SCHEMA,
    SurrealDBTemporalStore,
)

warnings.warn(
    "Importing from khora.engines.skeleton.backends.surrealdb is deprecated; "
    "import from khora.storage.temporal.surrealdb instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "SurrealDBTemporalStore",
    "_BACKED_SYSTEM_KEYS",
    "_TEMPORAL_CHUNK_SCHEMA",
]
