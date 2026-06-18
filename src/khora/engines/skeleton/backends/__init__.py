"""Backend implementations for the Skeleton engine.

.. deprecated::
    The temporal vector store now lives in :mod:`khora.storage.temporal`.
    The protocol (``TemporalVectorStore``), factory (``create_temporal_store``),
    the backend implementation modules, and the re-exported neutral temporal
    data types (``TemporalChunk``, ``TemporalFilter``, ``TemporalSearchResult``,
    ``document_denorm_fields``, ``temporal_chunk_to_chunk`` — which originate in
    :mod:`khora.core.temporal`) all moved there. Importing from this module
    still works but is deprecated; import from :mod:`khora.storage.temporal`
    instead.
"""

from __future__ import annotations

import warnings

from khora.storage.temporal import (
    TemporalChunk,
    TemporalFilter,
    TemporalSearchResult,
    TemporalVectorStore,
    create_temporal_store,
    document_denorm_fields,
    temporal_chunk_to_chunk,
)

warnings.warn(
    "Importing from khora.engines.skeleton.backends is deprecated; import from khora.storage.temporal instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "TemporalChunk",
    "TemporalFilter",
    "TemporalSearchResult",
    "TemporalVectorStore",
    "create_temporal_store",
    "document_denorm_fields",
    "temporal_chunk_to_chunk",
]
