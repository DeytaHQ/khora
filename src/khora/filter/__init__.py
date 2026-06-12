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

# Internal compilation seam (the AST + compiler layers). These are reachable as
# ``khora.filter.X`` for khora's own engines/compilers, but are deliberately NOT
# in ``__all__`` and NOT re-exported from ``khora.__init__`` — promoting them to
# a public plugin seam is deferred to a future improvement.
from khora.filter.ast import (  # noqa: F401
    DateLiteral as DateLiteral,
)
from khora.filter.ast import (
    FilterClause as FilterClause,
)
from khora.filter.ast import (
    FilterNode as FilterNode,
)
from khora.filter.ast import (
    FilterOp as FilterOp,
)
from khora.filter.ast import (
    canonical_hash as canonical_hash,
)
from khora.filter.ast import (
    metadata_leaf_count as metadata_leaf_count,
)
from khora.filter.ast import (
    parse_to_ast as parse_to_ast,
)
from khora.filter.context import (  # noqa: F401
    CompileContext as CompileContext,
)
from khora.filter.context import (
    CompileError as CompileError,
)
from khora.filter.context import (
    SchemaCapabilities as SchemaCapabilities,
)
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
from khora.filter.registry import (  # noqa: F401
    CompiledFilter as CompiledFilter,
)
from khora.filter.registry import (
    CompilerFn as CompilerFn,
)
from khora.filter.registry import (
    CompilerRegistry as CompilerRegistry,
)

__all__ = [
    "DateOps",
    "Op",
    "RecallFilter",
    "RecallFilterUnsupportedError",
    "RecallFilterValidationError",
    "StringOps",
    "SYSTEM_KEYS",
]
