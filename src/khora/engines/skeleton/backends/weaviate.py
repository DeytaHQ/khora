"""Deprecated shim — moved to :mod:`khora.storage.temporal.weaviate`.

Importing from this path still works but is deprecated. Import from
``khora.storage.temporal.weaviate`` instead. All names below are re-exported
verbatim (same objects), so identity is preserved across both paths.
"""

from __future__ import annotations

import warnings

from khora.storage.temporal.weaviate import (
    _DENORM_TEXT_KEYS,
    _FILTER_OVERFETCH,
    COLLECTION_NAME,
    WeaviateBackendConfig,
    WeaviateTemporalStore,
    _chunk_to_properties,
    _coerce_backend_config,
    _coerce_datetime,
    _denorm_properties,
    _extract_vector,
    _parse_host_port,
)

warnings.warn(
    "Importing from khora.engines.skeleton.backends.weaviate is deprecated; "
    "import from khora.storage.temporal.weaviate instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "COLLECTION_NAME",
    "WeaviateBackendConfig",
    "WeaviateTemporalStore",
    "_DENORM_TEXT_KEYS",
    "_FILTER_OVERFETCH",
    "_chunk_to_properties",
    "_coerce_backend_config",
    "_coerce_datetime",
    "_denorm_properties",
    "_extract_vector",
    "_parse_host_port",
]
