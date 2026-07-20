"""Canonical metadata contract for connector authors.

This module defines the public shape connector authors should populate in
``metadata.custom`` so the ingestion pipeline can extract a meaningful
source-system timestamp (used by temporal recency scoring and per-source
decay). See ``https://docs.deyta.ai/khora/pipeline/ingestion`` §canonical-fields
for the per-source mapping.

The TypedDict is ``total=False``: connectors may set whichever subset of
fields they have. ``validate_connector_metadata()`` is advisory only and
never raises — it is meant to be called by connector CI to surface common
mistakes before chunks reach khora.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal, NotRequired, TypedDict, get_args

__all__ = [
    "CANONICAL_TIMESTAMP_FIELDS",
    "ConnectorMetadata",
    "SourceSystem",
    "coerce_source_timestamp",
    "extract_source_timestamp",
    "validate_connector_metadata",
]

# Source-system identifiers khora's per-source decay table recognizes.
# See KhoraConfig.query.temporal_default_decay_by_source.
SourceSystem = Literal[
    "slack",
    "email",
    "calendar",
    "salesforce",
    "jira",
    "linear",
    "manual",
]

# Timestamp field names recognized by the ingestion extractor, in priority
# order. ``extract_source_timestamp`` matches these against metadata keys
# case/separator-insensitively, so connectors emitting ``occurredAt`` /
# ``occurred-at`` / ``OCCURRED_AT`` / ``OccurredAt`` resolve to the same
# canonical field; ``CANONICAL_TIMESTAMP_FIELDS`` remains the documented
# canonical set. Public so connector CIs can iterate.
CANONICAL_TIMESTAMP_FIELDS: tuple[str, ...] = (
    "sent_at",
    "occurred_at",
    "created_at",
    "updated_at",
    "started_at",
)

# Source types where temporal semantics favor occurred_at over sent_at.
_TIME_SENSITIVE_SOURCE_TYPES = frozenset({"slack", "email", "calendar", "salesforce"})


class ConnectorMetadata(TypedDict, total=False):
    """Canonical metadata shape for connector authors.

    All fields are optional, but at least one timestamp field SHOULD be
    set for any chunk that has a meaningful source-system time. See
    ``https://docs.deyta.ai/khora/pipeline/ingestion`` §canonical-fields for the
    per-source mapping.
    """

    # Timestamp fields, in priority order. See extract_source_timestamp() for matching.
    # For meetings/calendar, set occurred_at; for messages/email/Slack, sent_at.
    sent_at: str  # ISO-8601 UTC, e.g. "2026-05-13T14:00:00Z"
    occurred_at: str  # ISO-8601 UTC; for calendar events use start time
    created_at: str  # ISO-8601 UTC
    updated_at: str  # ISO-8601 UTC (for edits)
    started_at: str  # ISO-8601 UTC (for events/meetings)

    # Source identity — drives per-source decay + future per-source routing.
    source_system: SourceSystem
    source_id: str  # native ID in the source system (Slack ts, Gmail msg id, etc.)
    source_type: NotRequired[str]  # finer-grained: "message", "thread", "event", "activity"

    # Authorship and channel — used in span attrs + dedup heuristics.
    author: str
    channel: str
    thread_id: str  # for message-thread reconstruction

    # Edit semantics (Phase B6 — optional bitemporal mirror of entity tables).
    valid_from: str
    valid_until: str

    # Tags — free-form labels surfaced in metadata for downstream filtering.
    tags: list[str]


def coerce_source_timestamp(value: datetime | str | None) -> datetime | None:
    """Coerce a source-timestamp value to a ``datetime``.

    Returns an existing ``datetime`` unchanged, parses ISO-8601 strings
    (trailing ``Z``, explicit offset, or date-only ``YYYY-MM-DD``), and
    returns ``None`` for ``None`` / empty / unparseable input *without*
    raising. The public ``source_timestamp`` kwarg is typed ``datetime``,
    but upstream connectors and adapters routinely hand us ISO strings;
    coercing here keeps a stray string from crashing ingestion.

    Accepted string forms (for adapter authors):

    - ``"2026-01-15T10:30:00Z"``      — UTC, trailing ``Z``
    - ``"2026-01-15T10:30:00+00:00"`` — explicit offset
    - ``"2026-01-15"``                — date-only (parsed at midnight UTC)
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    try:
        if "T" in value:
            # ISO format with or without timezone.
            if value.endswith("Z"):
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            return datetime.fromisoformat(value)
        # Date-only format.
        return datetime.fromisoformat(value + "T00:00:00+00:00")
    except (ValueError, TypeError):
        return None


def _normalize_ts_key(key: str) -> str:
    return re.sub(r"[\s_\-]+", "", key.lower())


def extract_source_timestamp(metadata: Mapping[str, Any]) -> datetime | None:
    """Extract the original timestamp from source metadata.

    Looks for common timestamp fields and parses them. The priority list
    is implementation-detail; the public contract is
    ``khora.pipelines.ConnectorMetadata``.

    For event-shaped sources (calendar/meeting/event) ``occurred_at`` is
    preferred over ``sent_at`` since the event time, not the dispatch
    time, is the meaningful temporal anchor.

    Key matching is case/separator-insensitive: a connector emitting
    ``occurredAt`` / ``occurred-at`` / ``OCCURRED_AT`` resolves to the
    canonical ``occurred_at`` field. An exact key match always wins over a
    normalized variant when both are present.
    """
    source_type = metadata.get("source_type")
    if source_type in {"calendar", "meeting", "event"}:
        timestamp_fields = [
            "occurred_at",
            "started_at",
            "sent_at",
            "created_at",
            "timestamp",
            "date",
            "updated_at",
        ]
    else:
        timestamp_fields = [
            "sent_at",
            "created_at",
            "timestamp",
            "date",
            "occurred_at",
            "started_at",
            "updated_at",
        ]

    # Normalized view of metadata, first occurrence wins on collision.
    normalized_map: dict[str, Any] = {}
    for key, value in metadata.items():
        norm = _normalize_ts_key(key)
        if norm not in normalized_map:
            normalized_map[norm] = value

    for field in timestamp_fields:
        # Exact key match wins over a normalized variant.
        if field in metadata and metadata[field]:
            value = metadata[field]
        else:
            value = normalized_map.get(_normalize_ts_key(field))
            if not value:
                continue
        parsed = coerce_source_timestamp(value)
        if parsed is not None:
            return parsed
    return None


# Timestamp fields the extractor probes, in canonical-set order plus the two
# extra implementation-detail keys (``timestamp`` / ``date``) the extractor
# recognizes. Used by the validator to mirror extractor presence detection.
_EXTRACTOR_TIMESTAMP_FIELDS: tuple[str, ...] = CANONICAL_TIMESTAMP_FIELDS + ("timestamp", "date")


def validate_connector_metadata(
    metadata: Mapping[str, Any],
    source_type: str | None = None,
) -> list[str]:
    """Return a list of warning strings for connector authors.

    Empty list means the metadata is well-formed. Non-empty list is
    advisory only — none of these are blocking. Connector CI is the
    expected caller.

    Checks:
      * If ``source_type`` is a time-sensitive source (slack/email/calendar/
        salesforce) AND metadata has no ``sent_at``/``occurred_at``/
        ``created_at`` — warn that the chunk will fall back to ingest time
        (silent regression for recency scoring).
      * Timestamp values that look like strings but don't parse as ISO-8601.
      * ``source_system`` not in the canonical ``SourceSystem`` enum
        (warn, don't reject — operators can legitimately add new sources).
      * Empty-string values for any timestamp field (treated as missing).
      * ``tags`` is set but not a list (typing nudge).

    Args:
        metadata: The ``metadata.custom`` mapping a connector would pass
            into ``Khora.remember()``.
        source_type: Optional connector source type ("slack", "calendar",
            etc.). Used to decide which timestamp fields are required.

    Returns:
        A list of human-readable warning strings. Empty if all checks pass.
    """
    warnings: list[str] = []

    # Honor ConnectorMetadata.source_type when not passed explicitly, so
    # connector CI can hand the whole metadata dict in. Explicit param wins.
    if source_type is None:
        raw_source_type = metadata.get("source_type")
        source_type = raw_source_type if isinstance(raw_source_type, str) else None

    # Normalized view of metadata keys so case/separator-insensitive variants
    # (camelCase/kebab/SCREAMING/TitleCase) count as present, mirroring the
    # extractor. First occurrence wins on collision.
    normalized_map: dict[str, Any] = {}
    for key, value in metadata.items():
        norm = _normalize_ts_key(key)
        if norm not in normalized_map:
            normalized_map[norm] = value

    # Track which timestamp fields resolve to a non-empty, parseable value.
    present_timestamps: list[str] = []
    for field_name in _EXTRACTOR_TIMESTAMP_FIELDS:
        # Exact key match wins over a normalized variant.
        if field_name in metadata:
            value = metadata[field_name]
        elif _normalize_ts_key(field_name) in normalized_map:
            value = normalized_map[_normalize_ts_key(field_name)]
        else:
            continue
        if isinstance(value, str) and value == "":
            warnings.append(
                f"metadata field {field_name!r} is an empty string; "
                "treat empty as missing — either drop the field or set a value"
            )
            continue
        if isinstance(value, str):
            if coerce_source_timestamp(value) is None:
                warnings.append(
                    f"metadata field {field_name!r} value {value!r} does not parse as "
                    "ISO-8601 — chunk will fall back to ingest time"
                )
                continue
        present_timestamps.append(field_name)

    # Time-sensitive sources should always carry at least one timestamp.
    if source_type is not None and source_type in _TIME_SENSITIVE_SOURCE_TYPES and not present_timestamps:
        warnings.append(
            f"source_type={source_type!r} has no sent_at/occurred_at/created_at — "
            "chunk will fall back to ingest time (recency scoring will be wrong)"
        )

    # source_system enum check (advisory).
    raw_source_system = metadata.get("source_system")
    if raw_source_system is not None and raw_source_system not in get_args(SourceSystem):
        warnings.append(
            f"source_system={raw_source_system!r} is not in the canonical SourceSystem "
            f"enum {list(get_args(SourceSystem))!r} — per-source decay will fall back "
            "to default"
        )

    # tags typing nudge.
    if "tags" in metadata and not isinstance(metadata["tags"], list):
        warnings.append(f"metadata field 'tags' should be list[str], got {type(metadata['tags']).__name__}")

    return warnings
