"""Compiler dispatch registry (Layer 4 seam) — ``@internal``.

The :class:`CompilerRegistry` maps ``(engine_id, storage_target)`` to the
stateless compiler function that lowers a :class:`~khora.filter.ast.FilterNode`
to that backend's query fragment. It is the **second** internal seam: adding a
backend or an alternative compiler does not require touching engine code, and
adding an engine does not require touching compiler code — within khora's own
codebase.

``@internal``. The registry is ``__all__``'d under :mod:`khora.filter` only —
**not** :mod:`khora.__init__`. Exposing ``register()`` as a public extension
point for third-party engine/backend authors is deferred to a future improvement
(no current caller authors compilers; khora's five built-in compilers are the
only consumers).

The registry is **empty at import** — khora's engines register their compilers
at engine import time, not here. This module imports no compiler.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from khora.exceptions import KhoraError
from khora.filter.ast import FilterNode
from khora.filter.context import CompileContext

__all__ = [
    "CompiledFilter",
    "CompilerFn",
    "CompilerRegistry",
    "UnknownCompilerError",
]


T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Compiler output.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CompiledFilter(Generic[T]):
    """A backend compiler's output.

    ``@internal``. ``T`` varies by backend (a SQLAlchemy expression, a Cypher
    fragment, a LanceDB filter string, a ``callable(record) -> bool``, ...).

    * ``predicate`` — the compiled backend predicate (typed ``T``).
    * ``params`` — bind parameters for backends that bind (Postgres, Cypher);
      empty for backends that inline literals.
    * ``consumed_keys`` — the AST leaves this compiler handled, for partial
      pushdown: when ``CompileContext.on_unsupported == "split"`` the engine
      post-filters whatever is *not* in this set.
    * ``canonical_hash`` — the stable hash of the consumed slice, the engine's
      cache-key source.
    """

    predicate: T
    params: dict[str, Any]
    consumed_keys: frozenset[str]
    canonical_hash: str


# A compiler is a stateless function of ``(ast, ctx)``.
CompilerFn = Callable[[FilterNode, CompileContext], CompiledFilter[Any]]


# --------------------------------------------------------------------------- #
# Errors.
# --------------------------------------------------------------------------- #


class UnknownCompilerError(KhoraError):
    """No compiler is registered for the requested ``(engine_id, storage_target)``.

    ``@internal``. A :class:`KhoraError` subclass so callers can catch it
    narrowly or via the base.
    """

    def __init__(self, engine_id: str, storage_target: str) -> None:
        self.engine_id = engine_id
        self.storage_target = storage_target
        super().__init__(f"no compiler registered for ({engine_id!r}, {storage_target!r})")


class CompilerConflictError(KhoraError):
    """A different compiler is already registered for the same key.

    ``@internal``. Re-registering the *same* function for a key is idempotent and
    allowed; registering a *different* function for an occupied key is a
    programming error and raises this.
    """

    def __init__(self, engine_id: str, storage_target: str) -> None:
        self.engine_id = engine_id
        self.storage_target = storage_target
        super().__init__(
            f"a different compiler is already registered for ({engine_id!r}, {storage_target!r}); "
            "re-registering with a different function is not allowed"
        )


# --------------------------------------------------------------------------- #
# The registry.
# --------------------------------------------------------------------------- #


class CompilerRegistry:
    """Thread-safe process-wide compiler registry.

    ``@internal``. State and operations are class-level (a single process-wide
    registry — the canonical usage is ``CompilerRegistry.register(...)`` /
    ``CompilerRegistry.get(...)``), guarded by a class lock so concurrent
    engine-import-time registration is safe.
    """

    _lock: threading.Lock = threading.Lock()
    _registry: dict[tuple[str, str], CompilerFn] = {}

    def __init__(self) -> None:  # pragma: no cover - guard against instantiation
        raise TypeError("CompilerRegistry is a process-wide singleton; use its classmethods directly")

    @classmethod
    def register(cls, engine_id: str, storage_target: str, compiler: CompilerFn) -> None:
        """Register ``compiler`` for ``(engine_id, storage_target)``.

        Idempotent: re-registering the *same* function object for an already-bound
        key is a no-op. Registering a *different* function for an occupied key
        raises :class:`CompilerConflictError` — a registration is never silently
        overwritten.
        """
        key = (engine_id, storage_target)
        with cls._lock:
            existing = cls._registry.get(key)
            if existing is not None and existing is not compiler:
                raise CompilerConflictError(engine_id, storage_target)
            cls._registry[key] = compiler

    @classmethod
    def get(cls, engine_id: str, storage_target: str) -> CompilerFn:
        """Return the compiler for ``(engine_id, storage_target)``.

        Raises :class:`UnknownCompilerError` (a :class:`KhoraError` subclass) if
        no compiler is registered for the key.
        """
        key = (engine_id, storage_target)
        with cls._lock:
            compiler = cls._registry.get(key)
        if compiler is None:
            raise UnknownCompilerError(engine_id, storage_target)
        return compiler

    @classmethod
    def _clear(cls) -> None:
        """Drop every registration. Test-only escape hatch (not public API)."""
        with cls._lock:
            cls._registry.clear()
