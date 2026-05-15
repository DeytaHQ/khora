"""Type-mapping helpers between LangGraph's ``BaseStore`` shape and khora.

LangGraph identifies an item by ``(namespace: tuple[str, ...], key: str)``.
khora identifies a document by ``(namespace_id: UUID, external_id: str)``.
This module converts in both directions:

* ``flatten_namespace`` joins the tuple into a single string using a
  configurable separator and rejects any tuple segment containing that
  separator (we can't round-trip a label that already contains it).
* ``namespace_uuid`` derives a deterministic ``UUID5`` from a stable
  root + ``user_id`` so two ``KhoraStore`` instances on the same user
  see the same khora memory namespace.
* ``composite_external_id`` packs the flattened namespace + key into a
  single string capped at 512 chars — the khora ``Document.external_id``
  column length. We hash the prefix so we never exceed the cap.
* ``item_metadata`` / ``item_from_metadata`` round-trips the LangGraph
  ``value`` dict + tuple namespace through ``Document.metadata.custom``.

These helpers are private to the LangGraph adapter — not part of the
public ``khora.integrations`` API. Adapter-internal refactors are free.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import NAMESPACE_DNS, UUID, uuid5

if TYPE_CHECKING:
    from khora.core.models import Document

# Stable root used to derive the khora namespace UUID from a (root, user_id)
# pair. Lives in DNS-namespaced UUID5 space so it never collides with any
# randomly-generated UUID4 the caller might supply. Versioned so a future
# scheme change can rotate without colliding with shipped data.
_NAMESPACE_ROOT = uuid5(NAMESPACE_DNS, "khora.integrations.langgraph.v1")

# Document.external_id is a VARCHAR(512) column. We pack
# (flattened_namespace, key) into it. Long namespaces or keys would blow
# past 512 chars, so we hash-prefix when the raw form is too long.
_EXTERNAL_ID_MAX_LEN = 512


def flatten_namespace(namespace: tuple[str, ...], sep: str = "/") -> str:
    """Join a LangGraph namespace tuple into a single string.

    Args:
        namespace: A LangGraph namespace as a tuple of non-empty strings.
        sep: Separator used to join segments. Must not appear inside any
            segment.

    Returns:
        ``sep.join(namespace)``.

    Raises:
        ValueError: If ``namespace`` is empty, any segment is empty, or
            any segment contains ``sep``.
    """
    if not namespace:
        raise ValueError("LangGraph namespace cannot be empty.")
    for segment in namespace:
        if not segment:
            raise ValueError(f"LangGraph namespace segment cannot be empty: {namespace!r}")
        if sep in segment:
            raise ValueError(
                f"LangGraph namespace segment {segment!r} contains the configured "
                f"separator {sep!r}. Pick a different separator in KhoraStore(namespace_sep=...) "
                f"or rename the segment."
            )
    return sep.join(namespace)


def namespace_uuid(*, namespace_root: str, user_id: str) -> UUID:
    """Derive a stable khora namespace UUID from (root, user_id).

    Two ``KhoraStore`` instances built with the same arguments map to the
    same khora memory namespace. Used to allocate per-user long-term
    memory without storing a separate (user → namespace) registry.

    The root distinguishes adapters writing into the same khora deployment
    (``user_id``, ``thread_id``, ``app_id`` patterns) so naming collisions
    across apps don't fuse memories together.
    """
    return uuid5(_NAMESPACE_ROOT, f"{namespace_root}:{user_id}")


def composite_external_id(flat_namespace: str, key: str, sep: str = "/") -> str:
    """Pack ``(flat_namespace, key)`` into a Document.external_id string.

    Short forms pass through unchanged. Long forms hash-prefix the
    namespace portion to fit under the 512-char DB column cap. The
    raw namespace + key always live in ``Document.metadata.custom`` so
    nothing is lost — this is only the lookup index.
    """
    raw = f"{flat_namespace}::{key}"
    if len(raw) <= _EXTERNAL_ID_MAX_LEN:
        return raw
    digest = hashlib.sha1(flat_namespace.encode("utf-8"), usedforsecurity=False).hexdigest()
    # Reserve 40 chars for the digest + "::" separator + key. If the key alone
    # exceeds the cap we let the caller hit the underlying DB constraint —
    # they supplied a pathological key.
    prefix = f"h{digest}::"
    return prefix + key


def item_metadata(
    namespace: tuple[str, ...],
    key: str,
    value: dict[str, Any],
    *,
    sep: str = "/",
) -> dict[str, Any]:
    """Build the ``metadata`` dict that ``Khora.remember`` should stamp.

    The fields prefixed ``lg_`` round-trip the LangGraph identity (tuple
    namespace, string key, dict value) through ``Document.metadata.custom``
    so :func:`item_from_metadata` can reconstruct an :class:`Item` later.

    Kept simple — no JSON re-encoding, no flattening of ``value``. khora
    stores the metadata dict verbatim in a JSONB column.
    """
    return {
        "lg_namespace": list(namespace),  # tuple → list for JSON round-trip
        "lg_namespace_flat": sep.join(namespace),
        "lg_key": key,
        "lg_value": value,
    }


def item_from_metadata(
    document: Document,
) -> tuple[tuple[str, ...], str, dict[str, Any], datetime, datetime] | None:
    """Project a ``Document`` back to ``(namespace, key, value, created, updated)``.

    Returns ``None`` if the document was not written by this adapter
    (``lg_namespace`` / ``lg_key`` missing from ``metadata.custom``).
    Used by ``aget`` and ``asearch`` to skip foreign documents that share
    the khora namespace (e.g. data ingested through a non-LangGraph path).
    """
    custom = document.metadata.custom if document.metadata else {}
    ns_raw = custom.get("lg_namespace")
    key = custom.get("lg_key")
    if ns_raw is None or key is None:
        return None
    try:
        namespace = tuple(str(seg) for seg in ns_raw)
    except TypeError:
        return None
    value = custom.get("lg_value", {})
    if not isinstance(value, dict):
        value = {"value": value}
    created_at = document.created_at if document.created_at else datetime.now(UTC)
    updated_at = document.updated_at if document.updated_at else created_at
    return namespace, str(key), value, created_at, updated_at


def value_to_content(value: dict[str, Any]) -> str:
    """Render a LangGraph ``value`` dict to the ``content`` that khora chunks.

    LangGraph's vector-search contract embeds the rendered text of an
    item, so we need a deterministic string. Convention:

    * If ``value["text"]`` is set, use it (most natural for memory blobs).
    * Otherwise concatenate every string-valued field in key order.
    * Falling through to ``repr(value)`` keeps embeddings stable for
      structured data without dropping it on the floor.
    """
    if "text" in value and isinstance(value["text"], str) and value["text"]:
        return value["text"]
    strings = [str(v) for v in value.values() if isinstance(v, (str, int, float, bool))]
    if strings:
        return "\n".join(strings)
    return repr(value)


__all__ = [
    "composite_external_id",
    "flatten_namespace",
    "item_from_metadata",
    "item_metadata",
    "namespace_uuid",
    "value_to_content",
]
