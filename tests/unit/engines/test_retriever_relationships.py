"""Unit tests for VectorCypher retriever relationship construction logic."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from khora.core.models.entity import Relationship


@pytest.mark.unit
class TestRelationshipScoring:
    """Tests for relationship score computation from entity_scores_by_id."""

    def _build_relationship(
        self,
        *,
        raw: dict,
        namespace_id: UUID,
        entity_scores_by_id: dict[UUID, float],
        entity_names_by_id: dict[UUID, str],
    ) -> tuple[Relationship, float]:
        """Replicate the relationship construction logic from retriever.py ~lines 654-676."""
        src_id = UUID(raw["source_entity_id"])
        tgt_id = UUID(raw["target_entity_id"])
        rel_score = (entity_scores_by_id.get(src_id, 0.0) + entity_scores_by_id.get(tgt_id, 0.0)) / 2
        rel = Relationship(
            id=UUID(raw["id"]) if raw.get("id") else uuid4(),
            namespace_id=namespace_id,
            source_entity_id=src_id,
            target_entity_id=tgt_id,
            relationship_type=raw.get("relationship_type", "RELATES_TO"),
            description=raw.get("description", "") or "",
            source_entity_name=entity_names_by_id.get(src_id, ""),
            target_entity_name=entity_names_by_id.get(tgt_id, ""),
            source_document_ids=[UUID(d) for d in (raw.get("source_document_ids") or [])],
            source_chunk_ids=[UUID(c) for c in (raw.get("source_chunk_ids") or [])],
            confidence=raw.get("confidence") if raw.get("confidence") is not None else 1.0,
            weight=raw.get("weight") if raw.get("weight") is not None else 1.0,
        )
        return rel, rel_score

    def test_relationship_score_both_endpoints_present(self) -> None:
        """Both source and target in entity_scores_by_id, score = average."""
        src_id = uuid4()
        tgt_id = uuid4()
        ns_id = uuid4()

        entity_scores = {src_id: 0.8, tgt_id: 0.6}
        entity_names = {src_id: "Alice", tgt_id: "Bob"}

        raw = {
            "id": str(uuid4()),
            "source_entity_id": str(src_id),
            "target_entity_id": str(tgt_id),
            "relationship_type": "KNOWS",
        }

        rel, score = self._build_relationship(
            raw=raw,
            namespace_id=ns_id,
            entity_scores_by_id=entity_scores,
            entity_names_by_id=entity_names,
        )

        assert score == pytest.approx(0.7)  # (0.8 + 0.6) / 2
        assert rel.source_entity_name == "Alice"
        assert rel.target_entity_name == "Bob"

    def test_relationship_score_one_endpoint_missing(self) -> None:
        """One endpoint not in entity_scores_by_id defaults to 0.0."""
        src_id = uuid4()
        tgt_id = uuid4()
        ns_id = uuid4()

        # Only source in scores
        entity_scores = {src_id: 0.8}
        entity_names = {src_id: "Alice", tgt_id: "Bob"}

        raw = {
            "id": str(uuid4()),
            "source_entity_id": str(src_id),
            "target_entity_id": str(tgt_id),
            "relationship_type": "KNOWS",
        }

        rel, score = self._build_relationship(
            raw=raw,
            namespace_id=ns_id,
            entity_scores_by_id=entity_scores,
            entity_names_by_id=entity_names,
        )

        assert score == pytest.approx(0.4)  # (0.8 + 0.0) / 2

    def test_relationship_names_from_entity_results(self) -> None:
        """Names populated from entity_names_by_id."""
        src_id = uuid4()
        tgt_id = uuid4()
        ns_id = uuid4()

        entity_scores = {src_id: 0.5, tgt_id: 0.5}
        entity_names = {src_id: "Acme Corp", tgt_id: "Widget Inc"}

        raw = {
            "id": str(uuid4()),
            "source_entity_id": str(src_id),
            "target_entity_id": str(tgt_id),
            "relationship_type": "PARTNERS_WITH",
            "description": "Strategic partnership",
        }

        rel, _ = self._build_relationship(
            raw=raw,
            namespace_id=ns_id,
            entity_scores_by_id=entity_scores,
            entity_names_by_id=entity_names,
        )

        assert rel.source_entity_name == "Acme Corp"
        assert rel.target_entity_name == "Widget Inc"
        assert rel.relationship_type == "PARTNERS_WITH"
        assert rel.description == "Strategic partnership"

    def test_relationship_names_missing_fallback(self) -> None:
        """Endpoint not in entity_names_by_id gets empty string."""
        src_id = uuid4()
        tgt_id = uuid4()
        ns_id = uuid4()

        entity_scores = {}
        entity_names = {src_id: "Alice"}  # tgt_id missing

        raw = {
            "id": str(uuid4()),
            "source_entity_id": str(src_id),
            "target_entity_id": str(tgt_id),
            "relationship_type": "KNOWS",
        }

        rel, _ = self._build_relationship(
            raw=raw,
            namespace_id=ns_id,
            entity_scores_by_id=entity_scores,
            entity_names_by_id=entity_names,
        )

        assert rel.source_entity_name == "Alice"
        assert rel.target_entity_name == ""
