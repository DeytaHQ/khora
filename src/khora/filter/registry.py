"""Compiler dispatch registry (Layer 4 seam) — ``@internal``.

The :class:`CompilerRegistry` maps ``(engine_id, storage_target)`` to the
stateless compiler function that lowers a :class:`~khora.filter.ast.FilterNode`
to that backend's query fragment. It is the **second** internal seam: adding a
backend or an alternative compiler does not require touching engine code, and
adding an engine does not require touching compiler code — within khora's own
codebase. The first key component names the query-path OWNER, which is not
always an engine: an engine id on the recall path (``"chronicle"``,
``"skeleton.pgvector"``), a ``relational.<dialect>`` store id on the
document-enumeration path.

``@internal``. The registry is ``__all__``'d under :mod:`khora.filter` only —
**not** :mod:`khora.__init__`. Exposing ``register()`` as a public extension
point for third-party engine/backend authors is deferred to a future improvement
(no current caller authors compilers; khora's own built-in compilers are the
only consumers — seven ``compile_*`` functions ship, of which five are ever
registered: ``compile_python`` is the in-memory oracle and ``compile_cypher``
has no registered target).

Registrants are **engine and relational-storage-backend modules**, each
registering at its own import time; this module imports no compiler. The
registry is therefore **not** empty after a bare ``import khora``: the
PostgreSQL and raw-SQLite backends sit on :mod:`khora.storage.backends`' eager
import path, so their two documents-tier entries are already present. The
sqlite_lance and SurrealDB backends are imported lazily by ``StorageFactory``,
so their entries appear only once those backends are first constructed —
anything enumerating :meth:`CompilerRegistry.registered_keys` for a full
picture must import all four backend modules explicitly.
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
    * ``consumed_keys`` — the dotted paths this compiler fully handled, for
      partial pushdown: when ``CompileContext.on_unsupported == "split"`` the
      engine post-filters whatever is *not* in this set. Membership is per
      **occurrence**, not per leaf: a path counts as consumed only when EVERY one
      of its occurrences in the AST was pushed, so a key pushed in a conjunctive
      leaf but deferred inside an ``$or`` / ``$not`` the gate deferred wholesale is
      reported *absent* and stays post-filtered. Erring the other way would tell a
      caller differencing ``leaf_keys - consumed_keys`` that a deferred occurrence
      was already enforced, and the query would return rows the filter excludes.
    * ``consumed_slice_hash`` — a stable **plan-identity** hash: it identifies the
      predicate the backend actually received, not the filter the caller asked
      for. Under ``"split"`` two filters differing only in a deferred subtree emit
      the same predicate and therefore share this hash **while returning different
      rows**, because the deferred remainder is still enforced by the engine's
      post-filter. It is consequently **never a result-cache key** — key those on
      ``canonical_hash(filter_ast)`` over the whole AST, as
      ``engines/vectorcypher/recall_cache.py`` does. Postgres / lance / surrealdb
      derive it from the reconstructed slice; cypher / weaviate / python /
      chronicle currently supply the whole-AST hash, which is a *conservative*
      plan identity (equal hash there implies equal AST, hence equal slice — it
      can only over-distinguish two identical plans, never conflate two different
      ones), and ``compile_weaviate`` explains its choice at the construction
      site. In ``"raise"`` mode the slice IS the whole AST, so every compiler
      agrees and raise-mode callers see no difference.
    """

    predicate: T
    params: dict[str, Any]
    consumed_keys: frozenset[str]
    consumed_slice_hash: str


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
    def registered_keys(cls) -> frozenset[tuple[str, str]]:
        """Snapshot of every registered ``(engine_id, storage_target)`` key.

        ``@internal``. A thread-safe read of the full key set, used by the
        conformance drift guard to assert the registry holds *exactly* the
        compilers the corpus expects — so a new, unlisted registration fails
        loudly instead of being silently excluded from the conformance matrix.
        """
        with cls._lock:
            return frozenset(cls._registry)

    @classmethod
    def _clear(cls) -> None:
        """Drop every registration. Test-only escape hatch (not public API)."""
        with cls._lock:
            cls._registry.clear()
