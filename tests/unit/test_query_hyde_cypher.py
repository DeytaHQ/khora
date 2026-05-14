"""Unit tests for HyDE-Cypher (#595, Phase D2)."""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from khora.query.hyde_cypher import (
    TEMPLATES,
    HyDECypherSelection,
    HyDECypherValidationError,
    generate_cypher,
    select_template,
    validate_selection,
)

# ---------------------------------------------------------------------------
# Template registry sanity
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTemplateRegistry:
    def test_known_template_ids(self) -> None:
        assert set(TEMPLATES.keys()) == {"recent_by_type", "entity_relationships", "cooccurrence"}

    def test_every_template_uses_param_placeholders_only(self) -> None:
        """Injection-safety invariant: no template should ever inline a slot
        value via string interpolation. Every slot listed in the registry
        must appear as ``$slot`` in the Cypher source."""
        for tpl in TEMPLATES.values():
            for slot in tpl.slots:
                assert f"${slot}" in tpl.cypher, f"template {tpl.id} slot {slot!r} missing from Cypher source"
            # And the boilerplate parameters every template binds.
            assert "$namespace_id" in tpl.cypher
            assert "$limit" in tpl.cypher


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateSelection:
    def test_unknown_template_id_raises(self) -> None:
        sel = HyDECypherSelection(template_id="fictional", slots={})
        with pytest.raises(HyDECypherValidationError, match="unknown template"):
            validate_selection(sel)

    def test_missing_required_slot_raises(self) -> None:
        sel = HyDECypherSelection(template_id="recent_by_type", slots={"days": 7})
        with pytest.raises(HyDECypherValidationError, match="missing slot 'entity_type'"):
            validate_selection(sel)

    def test_valid_selection_returns_template(self) -> None:
        sel = HyDECypherSelection(
            template_id="recent_by_type",
            slots={"entity_type": "ACTION_ITEM", "days": 7},
        )
        tpl = validate_selection(sel)
        assert tpl.id == "recent_by_type"

    def test_entity_type_whitelist_rejects_off_whitelist_value(self) -> None:
        from khora.extraction.skills.base import EntityTypeConfig, ExpertiseConfig

        expertise = ExpertiseConfig(
            name="test",
            entity_types=[EntityTypeConfig(name="ACTION_ITEM")],
        )
        sel = HyDECypherSelection(
            template_id="recent_by_type",
            slots={"entity_type": "ATTACK_VECTOR", "days": 1},
        )
        with pytest.raises(HyDECypherValidationError, match="not in ExpertiseConfig whitelist"):
            validate_selection(sel, expertise=expertise)

    def test_relationship_type_whitelist_rejects_off_whitelist_value(self) -> None:
        from khora.extraction.skills.base import (
            EntityTypeConfig,
            ExpertiseConfig,
            RelationshipTypeConfig,
        )

        expertise = ExpertiseConfig(
            name="test",
            entity_types=[EntityTypeConfig(name="PERSON")],
            relationship_types=[RelationshipTypeConfig(name="OWNS")],
        )
        sel = HyDECypherSelection(
            template_id="entity_relationships",
            slots={"name": "Alice", "relationship_type": "DEMANDS"},
        )
        with pytest.raises(HyDECypherValidationError, match="relationship_type"):
            validate_selection(sel, expertise=expertise)


# ---------------------------------------------------------------------------
# Cypher generation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGenerateCypher:
    def test_recent_by_type_binds_all_slots(self) -> None:
        ns = uuid4()
        sel = HyDECypherSelection(
            template_id="recent_by_type",
            slots={"entity_type": "ACTION_ITEM", "days": "14"},
        )
        cypher, params = generate_cypher(sel, ns, limit=50)
        assert params["namespace_id"] == str(ns)
        assert params["entity_type"] == "ACTION_ITEM"
        assert params["days"] == 14  # coerced from string
        assert params["limit"] == 50
        # Cypher source unchanged — only parameters move.
        assert cypher == TEMPLATES["recent_by_type"].cypher

    def test_days_must_be_coercible_to_int(self) -> None:
        sel = HyDECypherSelection(
            template_id="recent_by_type",
            slots={"entity_type": "ACTION_ITEM", "days": "fortnight"},
        )
        with pytest.raises(HyDECypherValidationError, match="must be an integer"):
            generate_cypher(sel, uuid4())

    def test_slot_value_with_cypher_special_chars_is_parameter_only(self) -> None:
        """Slot values are bound as Neo4j parameters, never interpolated.
        A slot value containing Cypher syntax should pass straight into the
        params dict and never reach the Cypher source string.
        """
        sel = HyDECypherSelection(
            template_id="entity_relationships",
            slots={"name": "Alice') RETURN 1 //", "relationship_type": "OWNS"},
        )
        cypher, params = generate_cypher(sel, uuid4())
        # The value is in params, not in the Cypher source.
        assert params["name"] == "Alice') RETURN 1 //"
        assert "Alice') RETURN 1 //" not in cypher

    def test_nested_slot_value_stringified_not_interpolated(self) -> None:
        """An LLM returning a list/dict for a string slot must not crash
        the generator. We stringify defensively — the resulting param
        won't match anything in Neo4j but the call is safe."""
        sel = HyDECypherSelection(
            template_id="entity_relationships",
            slots={"name": ["A", "B"], "relationship_type": "OWNS"},
        )
        _, params = generate_cypher(sel, uuid4())
        assert isinstance(params["name"], str)


# ---------------------------------------------------------------------------
# LLM selector
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSelectTemplate:
    @pytest.mark.asyncio
    async def test_llm_picks_recent_by_type(self, monkeypatch) -> None:
        async def fake(prompt, config, **kwargs):  # noqa: ANN001
            return '{"id":"recent_by_type","slots":{"entity_type":"ACTION_ITEM","days":7}}'

        monkeypatch.setattr("khora.config.llm.acompletion", fake)
        selection = await select_template("what are the latest action items")
        assert selection.template_id == "recent_by_type"
        assert selection.slots["entity_type"] == "ACTION_ITEM"
        assert selection.slots["days"] == 7

    @pytest.mark.asyncio
    async def test_llm_returns_none_falls_back(self, monkeypatch) -> None:
        async def fake(prompt, config, **kwargs):  # noqa: ANN001
            return '{"id":"none","slots":{}}'

        monkeypatch.setattr("khora.config.llm.acompletion", fake)
        selection = await select_template("how does HyDE work")
        assert selection.template_id == "none"

    @pytest.mark.asyncio
    async def test_hallucinated_id_degrades_to_none(self, monkeypatch) -> None:
        async def fake(prompt, config, **kwargs):  # noqa: ANN001
            return '{"id":"made_up_template","slots":{"foo":"bar"}}'

        monkeypatch.setattr("khora.config.llm.acompletion", fake)
        selection = await select_template("anything")
        assert selection.template_id == "none"

    @pytest.mark.asyncio
    async def test_llm_exception_degrades_to_none(self, monkeypatch) -> None:
        async def fake(*args, **kwargs):  # noqa: ANN001
            raise RuntimeError("provider unavailable")

        monkeypatch.setattr("khora.config.llm.acompletion", fake)
        selection = await select_template("anything")
        assert selection.template_id == "none"

    @pytest.mark.asyncio
    async def test_malformed_json_degrades_to_none(self, monkeypatch) -> None:
        async def fake(prompt, config, **kwargs):  # noqa: ANN001
            return "not json at all"

        monkeypatch.setattr("khora.config.llm.acompletion", fake)
        selection = await select_template("anything")
        assert selection.template_id == "none"

    @pytest.mark.asyncio
    async def test_uses_telemetry_op(self, monkeypatch) -> None:
        captured: dict = {}

        async def fake(prompt, config, **kwargs):  # noqa: ANN001
            captured["telemetry_op"] = kwargs.get("_telemetry_op")
            return '{"id":"none","slots":{}}'

        monkeypatch.setattr("khora.config.llm.acompletion", fake)
        await select_template("q", llm_config=AsyncMock())
        assert captured["telemetry_op"] == "hyde_cypher.select"
