"""Public deterministic recall-filter surface.

Exports the typed filter document (:class:`RecallFilter`), its operator
submodels (:class:`StringOps`, :class:`DateOps`), the structured error types
(:class:`RecallFilterValidationError`, :class:`RecallFilterUnsupportedError`),
the operator vocabulary (:class:`Op`), and the system-key whitelist
(:data:`SYSTEM_KEYS`).

``FieldError`` is importable from here for callers inspecting a validation
error's structured ``errors`` list, but is not part of the public ``__all__``.
"""

from __future__ import annotations

from khora.filter.model import (
    SYSTEM_KEYS,
    DateOps,
    Op,
    RecallFilter,
    RecallFilterUnsupportedError,
    RecallFilterValidationError,
    StringOps,
)
from khora.filter.model import FieldError as FieldError  # re-export (not in __all__)

__all__ = [
    "DateOps",
    "Op",
    "RecallFilter",
    "RecallFilterUnsupportedError",
    "RecallFilterValidationError",
    "StringOps",
    "SYSTEM_KEYS",
]
