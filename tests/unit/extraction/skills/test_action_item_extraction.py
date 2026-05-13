"""Unit tests for the opt-in meetings/action-item extraction skill.

Issue #569: ``ACTION_ITEM`` / ``DECISION`` / ``BLOCKER`` / ``RISK`` are
typed entity definitions in ``builtin/meetings.yaml``. They are opt-in
(operators reference ``builtin:meetings`` in their ExpertiseConfig) and
emphasize precision over recall in their prompts.

These tests cover:
  - the YAML loads into a valid ExpertiseConfig
  - the typed attributes survive a round-trip through ExpertiseConfig
  - the LLM extractor returns the typed entity when the stubbed LLM emits one
  - the LLM extractor does NOT fabricate one when the stubbed LLM emits none
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from khora.extraction.extractors.llm import LLMEntityExtractor
from khora.extraction.skills.base import (
    EntityTypeConfig,
    ExpertiseConfig,
)
from khora.extraction.skills.loader import ExpertiseLoader

# ---------------------------------------------------------------------------
# Builtin YAML loads cleanly
# ---------------------------------------------------------------------------


@pytest.fixture
def meetings_config() -> ExpertiseConfig:
    """Load the bundled ``builtin:meetings`` expertise config."""
    loader = ExpertiseLoader()
    return loader.load_builtin("meetings", use_cache=False)


def test_meetings_skill_loads(meetings_config: ExpertiseConfig) -> None:
    """The builtin:meetings YAML loads and parses without error."""
    assert meetings_config.name == "meetings"
    type_names = meetings_config.get_entity_type_names()
    assert "ACTION_ITEM" in type_names
    assert "DECISION" in type_names
    assert "BLOCKER" in type_names
    assert "RISK" in type_names


def test_action_item_typed_attributes(meetings_config: ExpertiseConfig) -> None:
    """ACTION_ITEM advertises ``assignee``, ``due_by``, ``status`` as optional attrs."""
    action_item = meetings_config.get_entity_type("ACTION_ITEM")
    assert action_item is not None
    optional = action_item.attributes.get("optional", [])
    assert "assignee" in optional
    assert "due_by" in optional
    assert "status" in optional


def test_decision_typed_attributes(meetings_config: ExpertiseConfig) -> None:
    decision = meetings_config.get_entity_type("DECISION")
    assert decision is not None
    optional = decision.attributes.get("optional", [])
    assert "decided_on" in optional
    assert "rationale" in optional


def test_blocker_typed_attributes(meetings_config: ExpertiseConfig) -> None:
    blocker = meetings_config.get_entity_type("BLOCKER")
    assert blocker is not None
    optional = blocker.attributes.get("optional", [])
    assert "blocking_for" in optional
    assert "severity" in optional


def test_risk_typed_attributes(meetings_config: ExpertiseConfig) -> None:
    risk = meetings_config.get_entity_type("RISK")
    assert risk is not None
    optional = risk.attributes.get("optional", [])
    assert "likelihood" in optional
    assert "impact" in optional


def test_meetings_skill_has_higher_confidence_floor(meetings_config: ExpertiseConfig) -> None:
    """Devil's Advocate demanded precision over recall — verify floor was raised."""
    assert meetings_config.confidence.min_entity >= 0.7


# ---------------------------------------------------------------------------
# Programmatic ExpertiseConfig validates with typed entity types
# ---------------------------------------------------------------------------


def test_programmatic_action_item_config_validates() -> None:
    """An operator can construct ACTION_ITEM types programmatically."""
    config = ExpertiseConfig(
        name="custom_meetings",
        entity_types=[
            EntityTypeConfig(
                name="ACTION_ITEM",
                description="A specific task.",
                attributes={"required": ["name"], "optional": ["assignee", "due_by", "status"]},
            ),
        ],
    )
    # Round-trip through to_dict / from_dict
    restored = ExpertiseConfig.from_dict(config.to_dict())
    action_item = restored.get_entity_type("ACTION_ITEM")
    assert action_item is not None
    assert "assignee" in action_item.attributes["optional"]


# ---------------------------------------------------------------------------
# LLM extractor — happy path: stubbed response containing an action item
# ---------------------------------------------------------------------------


def _build_litellm_response(content: str) -> MagicMock:
    """Build a MagicMock that matches the litellm.acompletion response shape."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.choices[0].finish_reason = "stop"
    response.usage = MagicMock(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    response.model = "test-model"
    return response


@pytest.mark.asyncio
async def test_extracts_action_item_from_stubbed_response(
    meetings_config: ExpertiseConfig,
) -> None:
    """When the LLM emits a typed ACTION_ITEM, the extractor returns it
    with the typed attributes preserved on ``.attributes``.
    """
    stub_response_json = json.dumps(
        {
            "entities": [
                {
                    "name": "Investigate Postgres outage",
                    "entity_type": "ACTION_ITEM",
                    "description": "Bob to investigate the Postgres outage by EOW.",
                    "attributes": {
                        "assignee": "Bob",
                        "due_by": "2026-05-17",
                        "status": "open",
                    },
                    "confidence": 0.9,
                },
                {
                    "name": "Bob",
                    "entity_type": "PERSON",
                    "description": "Engineer assigned the investigation.",
                    "confidence": 0.95,
                },
            ],
            "relationships": [
                {
                    "source_entity": "Investigate Postgres outage",
                    "target_entity": "Bob",
                    "relationship_type": "ASSIGNED_TO",
                    "description": "Assigned to Bob.",
                    "confidence": 0.9,
                },
            ],
        }
    )

    extractor = LLMEntityExtractor(model="test-model")

    with patch(
        "litellm.acompletion",
        return_value=_build_litellm_response(stub_response_json),
    ) as mock_acompletion:
        result = await extractor.extract(
            "Bob: I'll investigate the Postgres outage by EOW.",
            expertise=meetings_config,
        )

    assert mock_acompletion.called
    action_items = [e for e in result.entities if e.entity_type == "ACTION_ITEM"]
    assert len(action_items) == 1
    ai = action_items[0]
    assert ai.name == "Investigate Postgres outage"
    assert ai.attributes.get("assignee") == "Bob"
    assert ai.attributes.get("due_by") == "2026-05-17"
    assert ai.attributes.get("status") == "open"


# ---------------------------------------------------------------------------
# LLM extractor — precision check: no action item is NOT fabricated
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_does_not_fabricate_action_item_when_llm_returns_none(
    meetings_config: ExpertiseConfig,
) -> None:
    """When the LLM correctly returns no ACTION_ITEM for a transcript that
    contains none, the extractor surfaces that. This is the precision-side
    guarantee: the extractor must not invent typed entities post-hoc.
    """
    # Transcript with general discussion but no explicit commitment.
    transcript = (
        "Alice: We should probably think about the pricing review at some point.\n"
        "Bob: Yeah, it'd be good. Maybe next quarter?\n"
    )
    stub_response_json = json.dumps(
        {
            "entities": [
                {"name": "Alice", "entity_type": "PERSON", "confidence": 0.95},
                {"name": "Bob", "entity_type": "PERSON", "confidence": 0.95},
                {"name": "pricing review", "entity_type": "CONCEPT", "confidence": 0.7},
            ],
            "relationships": [],
        }
    )

    extractor = LLMEntityExtractor(model="test-model")

    with patch(
        "litellm.acompletion",
        return_value=_build_litellm_response(stub_response_json),
    ):
        result = await extractor.extract(transcript, expertise=meetings_config)

    action_items = [e for e in result.entities if e.entity_type == "ACTION_ITEM"]
    assert action_items == [], "extractor must not fabricate ACTION_ITEM entities"
