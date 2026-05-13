"""Canonical metadata contract for connector authors.

This module defines the public shape connector authors should populate in
``metadata.custom`` so the ingestion pipeline can extract a meaningful
source-system timestamp (used by temporal recency scoring and per-source
decay). See ``docs/extraction/ingestion-pipeline.md`` §canonical-fields
for the per-source mapping.

The TypedDict is ``total=False``: connectors may set whichever subset of
fields they have. ``validate_connector_metadata()`` is advisory only and
never raises — it is meant to be called by connector CI to surface common
mistakes before chunks reach khora.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal, NotRequired, TypedDict, get_args

__all__ = [
    "CANONICAL_TIMESTAMP_FIELDS",
    "ConnectorMetadata",
    "SourceSystem",
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
# order. Kept in sync with ``_extract_source_timestamp`` in
# ``khora.pipelines.flows.ingest``. Public so connector CIs can iterate.
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
    ``docs/extraction/ingestion-pipeline.md`` §canonical-fields for the
    per-source mapping.
    """

    # Timestamp fields, in priority order matching _extract_source_timestamp.
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


def _parse_iso_or_none(value: str) -> datetime | None:
    """Return parsed datetime if value is ISO-8601, else None.

    Mirrors the parsing logic used by ``_extract_source_timestamp`` so the
    validator catches exactly what the pipeline will reject.
    """
    try:
        if "T" in value:
            if value.endswith("Z"):
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            return datetime.fromisoformat(value)
        return datetime.fromisoformat(value + "T00:00:00+00:00")
    except (ValueError, TypeError):
        return None


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

    # Track which timestamp fields are present with a non-empty string value.
    present_timestamps: list[str] = []
    for field_name in CANONICAL_TIMESTAMP_FIELDS:
        if field_name not in metadata:
            continue
        value = metadata[field_name]
        if isinstance(value, str) and value == "":
            warnings.append(
                f"metadata field {field_name!r} is an empty string; "
                "treat empty as missing — either drop the field or set a value"
            )
            continue
        if isinstance(value, str):
            if _parse_iso_or_none(value) is None:
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
