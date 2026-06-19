"""Deprecated shim — moved to :mod:`khora.storage.temporal.sqlite_lance`.

Importing from this path still works but is deprecated. Import from
``khora.storage.temporal.sqlite_lance`` instead. The name below is re-exported
verbatim (same object), so identity is preserved across both paths.
"""

from __future__ import annotations

import warnings

from khora.storage.temporal.sqlite_lance import SQLiteLanceTemporalStore

warnings.warn(
    "Importing from khora.engines.skeleton.backends.sqlite_lance is deprecated; "
    "import from khora.storage.temporal.sqlite_lance instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "SQLiteLanceTemporalStore",
]
