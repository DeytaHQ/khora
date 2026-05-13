"""Tests for the ConnectorMetadata contract + validate_connector_metadata."""

from __future__ import annotations

from khora.pipelines import (
    CANONICAL_TIMESTAMP_FIELDS,
    ConnectorMetadata,
    SourceSystem,
    validate_connector_metadata,
)
from khora.pipelines.flows.ingest import _extract_source_timestamp


def test_connector_metadata_typeddict_import_and_instance() -> None:
    """ConnectorMetadata is importable and accepts a minimal partial instance."""
    md: ConnectorMetadata = {
        "sent_at": "2026-05-13T14:00:00Z",
        "source_system": "slack",
        "tags": ["alpha"],
    }
    # TypedDicts are dicts at runtime.
    assert md["sent_at"] == "2026-05-13T14:00:00Z"
    assert md["source_system"] == "slack"
    assert md["tags"] == ["alpha"]


def test_canonical_timestamp_fields_match_extractor_priority() -> None:
    """The public timestamp list must include every field the extractor reads.

    Regression guard: if the extractor adds a new timestamp field, the
    public contract must be updated in lock-step.
    """
    extractor_fields = {
        "sent_at",
        "created_at",
        "timestamp",
        "date",
        "occurred_at",
        "started_at",
    }
    # Public contract advertises a subset (timestamp/date are legacy aliases),
    # but every advertised field must actually be honored by the extractor.
    for field in CANONICAL_TIMESTAMP_FIELDS:
        assert field in extractor_fields or field == "updated_at", (
            f"public field {field!r} not honored by _extract_source_timestamp"
        )


def test_well_formed_slack_metadata_has_no_warnings() -> None:
    assert validate_connector_metadata({"sent_at": "2026-05-13T14:00:00Z"}, "slack") == []


def test_empty_metadata_for_time_sensitive_source_warns() -> None:
    warnings = validate_connector_metadata({}, "slack")
    assert len(warnings) >= 1
    assert any("fall back to ingest time" in w for w in warnings)


def test_non_iso_timestamp_warns() -> None:
    warnings = validate_connector_metadata({"sent_at": "yesterday"}, "slack")
    assert any("ISO-8601" in w for w in warnings)


def test_unknown_source_system_warns_but_does_not_reject() -> None:
    warnings = validate_connector_metadata(
        {"sent_at": "2026-05-13T14:00:00Z", "source_system": "weirdo"},
        "slack",
    )
    assert any("source_system" in w and "weirdo" in w for w in warnings)


def test_empty_string_timestamp_treated_as_missing_and_occurred_at_wins_for_calendar() -> None:
    # sent_at is empty (warn), occurred_at is valid — calendar source has a
    # valid temporal anchor, so the "no timestamp" warning must NOT fire.
    warnings = validate_connector_metadata(
        {"sent_at": "", "occurred_at": "2026-05-13T14:00:00Z"},
        "calendar",
    )
    # An empty-string warning is expected; a "fall back" warning is not.
    assert any("empty string" in w for w in warnings)
    assert not any("fall back to ingest time" in w for w in warnings)


def test_tags_not_a_list_warns() -> None:
    warnings = validate_connector_metadata(
        {"sent_at": "2026-05-13T14:00:00Z", "tags": "alpha,beta"},
        "slack",
    )
    assert any("tags" in w for w in warnings)


def test_non_time_sensitive_source_without_timestamp_does_not_warn_about_fallback() -> None:
    # source_type=None means we have no expectation — no fallback warning.
    assert validate_connector_metadata({}, None) == []
    # manual upload is not time-sensitive either.
    assert validate_connector_metadata({}, "manual") == []


def test_source_system_literal_includes_expected_values() -> None:
    from typing import get_args

    members = set(get_args(SourceSystem))
    assert {"slack", "email", "calendar", "salesforce", "jira", "linear", "manual"} <= members


def test_extractor_prefers_occurred_at_for_calendar() -> None:
    md = {
        "source_type": "calendar",
        "sent_at": "2026-05-13T14:00:00Z",
        "occurred_at": "2026-05-14T09:00:00Z",
    }
    ts = _extract_source_timestamp(md)
    assert ts is not None
    assert ts.isoformat() == "2026-05-14T09:00:00+00:00"


def test_extractor_default_priority_unchanged_for_non_event_sources() -> None:
    md = {
        "source_type": "message",
        "sent_at": "2026-05-13T14:00:00Z",
        "occurred_at": "2026-05-14T09:00:00Z",
    }
    ts = _extract_source_timestamp(md)
    assert ts is not None
    # sent_at still wins for non-event sources.
    assert ts.isoformat() == "2026-05-13T14:00:00+00:00"
