"""Compile context + compiler error types (Layer 4 seam) — ``@internal``.

A :class:`CompileContext` is the engine-supplied, per-compile-pass context that
a backend compiler reads to know *where* and *how* to emit a predicate — the
table/alias to attach a WHERE to, the bind-param prefix, an optional system-key
→ column-name mapping, the backend's native capabilities, and the
unsupported-node policy. It is the **first** of the two internal seams (the
other is the :class:`~khora.filter.registry.CompilerRegistry`): a single
``compile_postgres`` serves multiple engines/schemas just by varying the context
— **no** ``if engine == "..."`` ever appears in a compiler.

``@internal``. ``CompileContext`` / ``SchemaCapabilities`` are ``__all__``'d under
:mod:`khora.filter` only; promoting them to a public plugin seam for third-party
compiler authors is deferred to a future improvement.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar, Literal

from khora.exceptions import KhoraError

# Re-export (do NOT redefine) — a compiler that cannot honor a predicate raises
# the *public* error so callers catch the same type regardless of backend.
from khora.filter.model import RecallFilterUnsupportedError

__all__ = [
    "CompileContext",
    "CompileError",
    "RecallFilterUnsupportedError",
    "SchemaCapabilities",
]


# --------------------------------------------------------------------------- #
# Compiler error base.
# --------------------------------------------------------------------------- #


class CompileError(KhoraError):
    """Base for errors raised inside the compile step.

    ``@internal``. Distinct from :class:`RecallFilterUnsupportedError` (a
    *contract* outcome — the backend simply cannot express a predicate, which an
    engine may legitimately handle by splitting or post-filtering). A
    ``CompileError`` signals an internal compiler fault (a malformed AST, an
    unreachable branch) — a bug, not a capability gap.
    """


# --------------------------------------------------------------------------- #
# Backend capability flags.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SchemaCapabilities:
    """What a backend can do natively, for compiler feature-gating.

    ``@internal``. A compiler reads these to choose an emission strategy (e.g.
    nested JSONB path-query vs. flat ``#>>`` extraction) or to decide a predicate
    is unsupported on this backend. Flags are coarse, capability-shaped booleans
    — not a fine-grained SQL feature matrix.

    * ``jsonb_path_query`` — backend can evaluate a nested JSON path natively
      (Postgres ``jsonb_path_query`` / ``#>>``, SurrealQL dot-descent), vs. only
      top-level key access.
    * ``full_text`` — backend has a native full-text predicate.
    * ``native_map_metadata`` — backend stores metadata as a real nested map/object
      (SurrealDB) rather than a serialized JSON string (Neo4j), so a metadata
      predicate can push down instead of falling to a post-filter.
    * ``sqlite_json1`` — SQLite backend has the JSON1 functions (``json_extract`` /
      ``json_type`` / ``json_each``) available, so a metadata predicate can push
      down into the ``khora_chunks.metadata`` JSON-TEXT column; ``False`` makes the
      SQLite compiler treat every metadata leaf as unsupported (post-filtered).

    ``DEFAULTS`` is the conservative all-``False`` instance — a backend with no
    declared capabilities. Engines pass a richer instance for backends that have
    more; ``CompileContext`` defaults to ``DEFAULTS`` so callers need not
    construct one.
    """

    jsonb_path_query: bool = False
    full_text: bool = False
    native_map_metadata: bool = False
    sqlite_json1: bool = False

    # Populated after the class is defined (cannot reference the class in its own
    # body). ClassVar is excluded from ``__slots__``.
    DEFAULTS: ClassVar[SchemaCapabilities]


SchemaCapabilities.DEFAULTS = SchemaCapabilities()


# --------------------------------------------------------------------------- #
# Compile context.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CompileContext:
    """Engine-supplied context for a single compile pass.

    ``@internal``. A compiler is a stateless function of ``(ast, ctx)``; this is
    the ``ctx``. Frozen + slotted: it is created once per recall and read by the
    compiler, never mutated.

    * ``backend_target`` — the concrete target the predicate attaches to
      (``"khora_chunks"``, ``"documents"``, ``"Chunk"``, ``"events"``, ...).
    * ``table_alias`` — SQL: the alias of the table the WHERE attaches to
      (``None`` when the target is unaliased or the backend is not SQL).
    * ``param_namespace`` — bind-param prefix so compiled params cannot collide
      with the engine's own query parameters.
    * ``field_mapping`` — optional system-key → backend column/property name map.
      Lets one compiler serve a different schema (e.g. the legacy
      ``chunks``/``documents`` tables) without per-engine branching. ``None`` =
      identity mapping (system key name == column name). A compiler MAY also treat
      the KEY SET as the backend's declared+pushable property whitelist — a key
      absent from ``field_mapping`` is then "undeclared" and not pushed down (as
      :func:`~khora.filter.compilers.weaviate.compile_weaviate` does, leaving
      undeclared keys to the post-filter).
    * ``schema_capabilities`` — what the backend can do natively
      (:class:`SchemaCapabilities`).
    * ``on_unsupported`` — ``"raise"`` stops on the first node the backend cannot
      express; ``"split"`` compiles what it can and returns the handled leaves via
      ``CompiledFilter.consumed_keys`` for the engine to post-filter the rest.
    """

    backend_target: str
    table_alias: str | None = None
    param_namespace: str = "f"
    field_mapping: Mapping[str, str] | None = None
    schema_capabilities: SchemaCapabilities = field(default=SchemaCapabilities.DEFAULTS)
    on_unsupported: Literal["raise", "split"] = "raise"
