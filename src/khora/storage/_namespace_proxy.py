"""Deprecation shim for direct backend access via the coordinator.

``StorageCoordinator.{graph,vector,relational,event_store}`` used to expose
the raw backend object — any callsite with a ``Khora`` handle could bypass
the coordinator's namespace-enforcement layer (the IDOR family/213/214). This
module's ``NamespaceRequiredProxy`` wraps each backend so that:

1. First attribute access (per role, per process) emits a
   ``DeprecationWarning`` pointing callers at the coordinator facade.
2. Read methods tightened by the IDOR family/the IDOR family (``get_document``,
   ``get_entity``, ``get_neighborhood``, …) refuse to dispatch without an
   explicit ``namespace_id=`` kwarg, even though the underlying signature
   already requires it. The proxy gives a clearer error message and a
   stable enforcement seam for future write-method hardening (IDOR family).

Internal coordinator code uses ``self._{graph,vector,relational,event_store}``
directly and never goes through this proxy — only external callers do.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any

# Read methods tightened by the IDOR family (RelationalBackend, VectorBackend) and
# the IDOR family (GraphBackend). Calls to these via the proxy must include
# ``namespace_id=`` or the proxy raises ``TypeError`` before dispatching.
_NS_REQUIRED_METHODS: dict[str, frozenset[str]] = {
    "graph": frozenset(
        {
            "get_entity",
            "get_entities_batch",
            "get_relationship",
            "get_episode",
            "get_entity_relationships",
            "get_neighborhood",
            "get_neighborhoods_batch",
        }
    ),
    "vector": frozenset(
        {
            "entity_exists",
            "get_entity",
            "get_entities_batch",
        }
    ),
    "relational": frozenset(
        {
            "get_document",
            "get_documents_batch",
            "get_document_sources_batch",
        }
    ),
    "event_store": frozenset(
        {
            "get_events_for_resource",
            "get_latest_event",
        }
    ),
}


class NamespaceRequiredProxy:
    """Wraps a storage backend; emits deprecation + enforces namespace_id on reads."""

    __slots__ = ("_backend", "_role")

    _warned_roles: set[str] = set()

    def __init__(self, backend: Any, role: str) -> None:
        object.__setattr__(self, "_backend", backend)
        object.__setattr__(self, "_role", role)

    def __repr__(self) -> str:
        return f"NamespaceRequiredProxy(role={self._role!r}, backend={self._backend!r})"

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)

        if self._role not in NamespaceRequiredProxy._warned_roles:
            NamespaceRequiredProxy._warned_roles.add(self._role)
            warnings.warn(
                f"Direct access to StorageCoordinator.{self._role} is deprecated; "
                f"use the coordinator's namespace-scoped facade methods instead. "
                f"This attribute will be removed in v0.17.",
                DeprecationWarning,
                stacklevel=2,
            )

        attr = getattr(self._backend, name)

        if name in _NS_REQUIRED_METHODS.get(self._role, frozenset()) and callable(attr):
            role = self._role

            @functools.wraps(attr)
            def _ns_guarded(*args: Any, **kwargs: Any) -> Any:
                if "namespace_id" not in kwargs:
                    raise TypeError(f"{role}.{name}() requires keyword argument 'namespace_id'")
                return attr(*args, **kwargs)

            return _ns_guarded

        return attr
