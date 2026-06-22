"""Signature gate for the cross-namespace IDOR invariant (the IDOR family / the IDOR family).

The IDOR family (the IDOR family/213/214 → the IDOR family/223/224 → the IDOR family) keeps
recurring because no test asserts the invariant directly. This module
enforces it structurally: for every concrete backend implementation,
every read, existence, write or delete method must declare
``namespace_id`` as a parameter (keyword-only after the IDOR family/the IDOR family for
reads and the IDOR family for writes/deletes).

If a new backend method is added without ``namespace_id``, this test fails
at collection time with the offending class / method named, so the
regression is impossible to land without a security-review prompt.

The behavioural gate (wrong-namespace returns empty) is covered by
per-backend ``test_namespace_scoped_reads.py`` modules under
``tests/unit/storage/backends/`` for the the IDOR family/223 surface; once the
behavioural matrix runs across the integration backends the per-backend
modules can be replaced by a parametrized matrix fixture here.
"""

from __future__ import annotations

import inspect
import re
from typing import Any

import pytest

from khora.storage.backends.base import (
    EventStoreProtocol,
    GraphBackendProtocol,
    RelationalBackendProtocol,
    VectorBackendProtocol,
)

# ---------------------------------------------------------------------------
# Backend registry — every concrete class implementing one of the four
# storage protocols. Lazy-imported so optional-extras backends (lancedb,
# surrealdb, neo4j, neptune) don't break this collection.
# ---------------------------------------------------------------------------


def _try_import(module_path: str, class_name: str) -> Any | None:
    try:
        mod = __import__(module_path, fromlist=[class_name])
    except Exception:
        return None
    return getattr(mod, class_name, None)


_RELATIONAL_BACKENDS = [
    ("khora.storage.backends.postgresql", "PostgreSQLBackend"),
    ("khora.storage.backends.sqlite", "SQLiteRelationalBackend"),
    ("khora.storage.backends.sqlite_lance.relational", "SQLiteLanceRelationalAdapter"),
    ("khora.storage.backends.surrealdb.relational", "SurrealDBRelationalAdapter"),
]

_VECTOR_BACKENDS = [
    ("khora.storage.backends.pgvector", "PgVectorBackend"),
    ("khora.storage.backends.sqlite", "SQLiteVectorBackend"),
    ("khora.storage.backends.sqlite_lance.vector", "SQLiteLanceVectorAdapter"),
    ("khora.storage.backends.surrealdb.vector", "SurrealDBVectorAdapter"),
]

_GRAPH_BACKENDS = [
    ("khora.storage.backends.neo4j", "Neo4jBackend"),
    ("khora.storage.backends.age", "AGEBackend"),
    ("khora.storage.backends.memgraph", "MemgraphBackend"),
    ("khora.storage.backends.neptune", "NeptuneBackend"),
    ("khora.storage.backends.sqlite_lance.graph", "SQLiteLanceGraphAdapter"),
    ("khora.storage.backends.surrealdb.graph", "SurrealDBGraphAdapter"),
]

_EVENT_STORE_BACKENDS = [
    ("khora.storage.backends.sqlite_lance.event_store", "SQLiteLanceEventStoreAdapter"),
    ("khora.storage.backends.surrealdb.event_store", "SurrealDBEventStoreAdapter"),
]

# ---------------------------------------------------------------------------
# Read- and write-method patterns. A method that matches one of these AND
# takes an identifier-typed positional argument must declare ``namespace_id``.
# ---------------------------------------------------------------------------

# Methods that read or test the existence of a row by ID (the IDOR family/223),
# plus methods that mutate or delete a row by ID (IDOR family). Includes
# traversal methods that walk the graph from a seed entity ID.
_READ_METHOD_PATTERN = re.compile(
    r"^("
    r"get_(?!session\b|connection\b|engine\b|driver\b)\w+"  # get_<x>, but not infrastructure helpers
    r"|entity_exists"
    r"|relationship_exists"
    r"|find_paths"
    r"|get_neighborhood"
    r"|get_neighborhoods_batch"
    # Write / delete surface (IDOR family):
    r"|delete_\w+"
    r"|update_entity(_\w+)?"
    r"|supersede_\w+"
    r")$"
)

# Explicitly excluded: pure-query helpers that aren't ID-based and don't
# leak rows across namespaces. The matcher catches `get_session` /
# `get_connection` / `get_engine` / `get_driver` already; add named
# exemptions here as new false-positives appear.
_EXEMPT_METHODS: frozenset[str] = frozenset(
    {
        "get_health_summary",
        "get_pool_metrics",
        "get_status",
        "get_namespace",  # the *resolver* methods take a name, not a foreign id
        "get_namespace_by_name",
        "get_namespace_versions",  # namespace registry helpers
        "get_active_namespace_id",
        "get_active_namespace",
        # Namespace-side maintenance methods that delete namespaces wholesale —
        # by definition not "scoped to a namespace_id" because they ARE
        # operating on the namespace row itself.
        "delete_namespace",
    }
)


def _is_id_param(name: str, annotation: Any) -> bool:
    """Best-effort: does this parameter look like a row id?"""
    if name in ("self", "cls"):
        return False
    if name == "namespace_id":
        return False
    if name.endswith("_id") or name.endswith("_ids"):
        return True
    # ``inspect.signature`` may stringify annotations under ``from __future__``;
    # checking the str form is sufficient to catch the common cases.
    ann = str(annotation) if annotation is not inspect.Parameter.empty else ""
    return "UUID" in ann or "uuid.UUID" in ann


# Names whose mere existence implies a row-mutation (so the gate runs even
# if their first positional argument isn't UUID-typed — e.g.
# ``update_entity(entity: Entity, ...)``).
_FORCE_INCLUDE_NAMES = re.compile(r"^(delete_\w+|update_entity(_\w+)?|supersede_\w+)$")


def _enumerate_read_methods(cls: type) -> list[tuple[str, inspect.Signature]]:
    """Return [(method_name, signature), ...] for ID-based read/write methods.

    A method is considered ID-scoped if it has at least one *required*
    id-typed parameter (after ``self``), OR if its name matches one of the
    write/delete patterns where the namespace_id requirement is unconditional
    regardless of the first-arg shape (``update_entity(entity, ...)`` is
    still namespace-scoped even though ``entity`` is not an ID).

    List-style methods like ``get_events(namespace_id, *, resource_id=None)``
    where the id-typed parameters are all optional are excluded — those are
    filtered scans, not targeted lookups, and the namespace gate in the SQL
    WHERE clause is the primary IDOR control.
    """
    out: list[tuple[str, inspect.Signature]] = []
    for name, member in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name.startswith("_"):
            continue
        if name in _EXEMPT_METHODS:
            continue
        if not _READ_METHOD_PATTERN.match(name):
            continue
        try:
            sig = inspect.signature(member)
        except (TypeError, ValueError):
            continue
        params = list(sig.parameters.values())
        has_id_param = any(
            _is_id_param(p.name, p.annotation) and p.default is inspect.Parameter.empty for p in params[1:]
        )
        if not has_id_param and not _FORCE_INCLUDE_NAMES.match(name):
            continue
        out.append((name, sig))
    return out


def _resolve(group: list[tuple[str, str]]) -> list[type]:
    """Import every backend class; skip any whose extra isn't installed."""
    resolved: list[type] = []
    for mod_path, class_name in group:
        cls = _try_import(mod_path, class_name)
        if cls is not None:
            resolved.append(cls)
    return resolved


_ALL_BACKENDS: list[tuple[type, type]] = []
_ALL_BACKENDS.extend((cls, RelationalBackendProtocol) for cls in _resolve(_RELATIONAL_BACKENDS))
_ALL_BACKENDS.extend((cls, VectorBackendProtocol) for cls in _resolve(_VECTOR_BACKENDS))
_ALL_BACKENDS.extend((cls, GraphBackendProtocol) for cls in _resolve(_GRAPH_BACKENDS))
_ALL_BACKENDS.extend((cls, EventStoreProtocol) for cls in _resolve(_EVENT_STORE_BACKENDS))


def _backend_id(value: tuple[type, type]) -> str:
    cls, protocol = value
    return f"{protocol.__name__}::{cls.__name__}"


@pytest.mark.security
@pytest.mark.unit
@pytest.mark.parametrize("backend", _ALL_BACKENDS, ids=_backend_id)
def test_all_read_methods_declare_namespace_id(backend: tuple[type, type]) -> None:
    """Every ID-based read method on a concrete backend takes ``namespace_id=``.

    This is the structural gate that prevents the IDOR family from recurring.
    If a new ``get_<X>`` / ``entity_exists`` / ``get_neighborhood`` method is
    added without a ``namespace_id`` parameter, this test fails with the
    offending class and method named so a reviewer can flag it.

    Add the method name to ``_EXEMPT_METHODS`` only when the method is
    provably not namespace-scoped (e.g. namespace-resolver helpers that
    look rows up *by name* rather than *by foreign id*).
    """
    cls, _protocol = backend
    offenders: list[str] = []
    for name, sig in _enumerate_read_methods(cls):
        if "namespace_id" not in sig.parameters:
            offenders.append(f"{cls.__name__}.{name}{sig}")
    assert not offenders, (
        "Read methods missing ``namespace_id`` parameter (cross-namespace IDOR family):\n"
        + "\n".join(f"  - {o}" for o in offenders)
        + "\n\nFix: add ``*, namespace_id: UUID`` (kwarg-only, required) to the signature "
        "and filter at the query layer. If the method is genuinely not namespace-scoped, "
        "add it to ``_EXEMPT_METHODS`` in this file with a one-line justification."
    )


@pytest.mark.security
@pytest.mark.unit
@pytest.mark.parametrize("backend", _ALL_BACKENDS, ids=_backend_id)
def test_namespace_id_is_keyword_only(backend: tuple[type, type]) -> None:
    """``namespace_id`` must be keyword-only on all read methods.

    Positional ``namespace_id`` lets callers accidentally pass a row id in
    that slot. Kwarg-only forces an explicit ``namespace_id=…`` at every
    call site, which is what the the IDOR family/the IDOR family migration enforced across
    24 in-tree caller sites.
    """
    cls, _protocol = backend
    offenders: list[str] = []
    for name, sig in _enumerate_read_methods(cls):
        param = sig.parameters.get("namespace_id")
        if param is None:
            # The preceding test will catch this; don't double-report.
            continue
        if param.kind not in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD):
            offenders.append(f"{cls.__name__}.{name}: namespace_id kind={param.kind.name}")
    assert not offenders, (
        "``namespace_id`` must be keyword-only on read methods (place after ``*,`` in the signature):\n"
        + "\n".join(f"  - {o}" for o in offenders)
    )


@pytest.mark.security
@pytest.mark.unit
def test_backend_registry_not_empty() -> None:
    """Defensive: at least the in-tree-always backends must be picked up.

    If this fires, ``_try_import`` is swallowing a real import error or the
    backend class list above is stale. Either way the parametrized gate
    would silently pass with zero cases.
    """
    assert _ALL_BACKENDS, "Backend registry is empty — _try_import or the class list is broken"
    # The pgvector/postgresql/sqlite_lance backends ship with the base extra
    # set; assert they're present so a venv missing the lancedb/surrealdb
    # optional extras doesn't degrade this gate into a tautology.
    names = {cls.__name__ for cls, _ in _ALL_BACKENDS}
    assert "PgVectorBackend" in names
    assert "PostgreSQLBackend" in names
    # The legacy embedded ``sqlite`` backend ships with the base extras too
    # (aiosqlite only, no lancedb); assert it so the IDOR coverage added for
    # it can't silently disappear if its import ever starts failing.
    assert "SQLiteRelationalBackend" in names
    assert "SQLiteVectorBackend" in names
