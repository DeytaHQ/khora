"""Unit tests for #1528 NUL-byte (0x00) sanitization helpers."""

from __future__ import annotations

from khora.core.text import strip_nul, strip_nul_json
from khora.extraction.extractors.base import (
    ExtractedEntity,
    ExtractedEvent,
    ExtractedRelationship,
    ExtractionResult,
    sanitize_extraction_result,
)


def test_strip_nul_removes_nul_bytes():
    assert strip_nul("a\x00b\x00c") == "abc"
    assert strip_nul("clean") == "clean"
    assert strip_nul("") == ""


def test_strip_nul_json_recurses_into_dicts_and_lists():
    value = {
        "k\x00ey": "va\x00lue",
        "nested": {"a": ["x\x00", "y", 1, None, True]},
        "num": 3,
    }
    out = strip_nul_json(value)
    assert out == {
        "key": "value",
        "nested": {"a": ["x", "y", 1, None, True]},
        "num": 3,
    }


def test_strip_nul_json_passes_through_non_json_scalars():
    assert strip_nul_json(42) == 42
    assert strip_nul_json(None) is None
    assert strip_nul_json(2.5) == 2.5


def test_sanitize_extraction_result_strips_all_text_fields():
    result = ExtractionResult(
        entities=[
            ExtractedEntity(
                name="Ac\x00me",
                entity_type="ORG",
                description="wid\x00get",
                attributes={"a\x00": "b\x00", "n": 1},
                aliases=["A\x00C"],
            )
        ],
        relationships=[
            ExtractedRelationship(
                source_entity="Ac\x00me",
                target_entity="Ri\x00ver",
                relationship_type="NEAR\x00",
                description="near the\x00 river",
                properties={"p\x00": "q\x00"},
            )
        ],
        events=[
            ExtractedEvent(
                description="foun\x00ded",
                participants=["Ac\x00me", "Bo\x00b"],
            )
        ],
    )

    sanitize_extraction_result(result)

    ent = result.entities[0]
    assert ent.name == "Acme"
    assert ent.description == "widget"
    assert ent.attributes == {"a": "b", "n": 1}
    assert ent.aliases == ["AC"]

    rel = result.relationships[0]
    assert rel.source_entity == "Acme"
    assert rel.target_entity == "River"
    assert rel.relationship_type == "NEAR"
    assert rel.description == "near the river"
    assert rel.properties == {"p": "q"}

    evt = result.events[0]
    assert evt.description == "founded"
    assert evt.participants == ["Acme", "Bob"]


def test_sanitize_extraction_result_tolerates_none_text_fields():
    """Extractor output is loosely typed; None fields must not raise (#1528 regression)."""
    result = ExtractionResult(
        entities=[ExtractedEntity(name="X", entity_type="ORG", description=None)],  # type: ignore[arg-type]
        relationships=[
            ExtractedRelationship(
                source_entity="X",
                target_entity="Y",
                relationship_type="R",
                description=None,  # type: ignore[arg-type]
            )
        ],
    )

    sanitize_extraction_result(result)  # must not raise

    assert result.entities[0].description is None
    assert result.relationships[0].description is None
