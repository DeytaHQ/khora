"""Deprecated shim — moved to :mod:`khora.storage.temporal.turbopuffer`.

Importing from this path still works but is deprecated. Import from
``khora.storage.temporal.turbopuffer`` instead. All names below are re-exported
verbatim (same objects), so identity is preserved across both paths.
"""

from __future__ import annotations

import warnings

from khora.storage.temporal.turbopuffer import (
    TurbopufferBackendConfig,
    TurbopufferTemporalStore,
    _build_turbopuffer_filter,
    _chunk_to_row,
    _coerce_datetime,
    _row_to_chunk,
    _rrf_fuse,
)

warnings.warn(
    "Importing from khora.engines.skeleton.backends.turbopuffer is deprecated; "
    "import from khora.storage.temporal.turbopuffer instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "TurbopufferBackendConfig",
    "TurbopufferTemporalStore",
    "_build_turbopuffer_filter",
    "_chunk_to_row",
    "_coerce_datetime",
    "_row_to_chunk",
    "_rrf_fuse",
]
