"""Tests for the public ``khora.context_text`` helper.

Verifies the typed ``RecallResult`` rendering contract — chunk grouping
by document title, entities section, relationships section with name
resolution, dedup behavior, and ``max_chunks`` slicing.

A separate test asserts byte-identical output against captured baseline
strings to lock the rendered format.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from khora import context_text
from khora.core.models.recall import (
    DocumentProjection,
    RecallChunk,
    RecallEntity,
    RecallRelationship,
    RecallResult,
)

# Captured baseline strings for the byte-equivalence scenarios. The
# helper's output for these fixtures is deterministic (no UUIDs leak
# into the rendered string) so the goldens stay reproducible across
# runs.
GOLDEN_FULL_PAYLOAD_CONTEXT_TEXT = (
    "--- From: Founding Story ---\n"
    "Alice founded Acme Corp."
    "\n\n--- Entities ---\n\n"
    "- Alice (PERSON): Founder\n"
    "- Acme Corp (ORGANIZATION): A company"
    "\n\n--- Relationships ---\n\n"
    "- Alice --FOUNDED--> Acme Corp: Founded the company"
)

GOLDEN_CHUNKS_ONLY_UNTITLED_CONTEXT_TEXT = "alpha text\n\n---\n\nbeta text"

GOLDEN_ENTITIES_ONLY_CONTEXT_TEXT = "\n\n--- Entities ---\n\n- Solo (CONCEPT): The only one"


def _mk_doc(*, doc_id=None, title=None) -> DocumentProjection:
    return DocumentProjection(
        id=doc_id or uuid4(),
        created_at=datetime.now(UTC),
        title=title,
    )


def _mk_chunk(*, document_id, content, score=0.9) -> RecallChunk:
    return RecallChunk(
        id=uuid4(),
        document_id=document_id,
        content=content,
        score=score,
        created_at=datetime.now(UTC),
    )


def _mk_entity(*, name, entity_type, description="", entity_id=None, score=0.9) -> RecallEntity:
    return RecallEntity(
        id=entity_id or uuid4(),
        name=name,
        entity_type=entity_type,
        description=description,
        score=score,
        attributes={},
        mention_count=1,
        source_document_ids=[],
        source_chunk_ids=[],
    )


def _mk_rel(*, source_id, target_id, relationship_type, description="", score=0.9) -> RecallRelationship:
    return RecallRelationship(
        id=uuid4(),
        source_entity_id=source_id,
        target_entity_id=target_id,
        relationship_type=relationship_type,
        description=description,
        score=score,
        valid_from=None,
        valid_until=None,
        source_document_ids=[],
    )


def _mk_result(
    *,
    documents=None,
    chunks=None,
    entities=None,
    relationships=None,
) -> RecallResult:
    return RecallResult(
        query="q",
        namespace_id=uuid4(),
        documents=documents or [],
        chunks=chunks or [],
        entities=entities or [],
        relationships=relationships or [],
    )


@pytest.mark.unit
class TestContextTextHelper:
    """The public helper renders the typed RecallResult contract."""

    def test_empty_result_is_empty_string(self) -> None:
        assert context_text(_mk_result()) == ""

    def test_chunks_only_no_titles_joins_with_separator(self) -> None:
        doc = _mk_doc(title=None)
        chunks = [
            _mk_chunk(document_id=doc.id, content="alpha"),
            _mk_chunk(document_id=doc.id, content="beta"),
        ]
        out = context_text(_mk_result(documents=[doc], chunks=chunks))
        # Untitled chunks land in the same group (key=""), joined by \n\n
        assert "alpha" in out
        assert "beta" in out
        assert "--- From:" not in out

    def test_chunks_grouped_by_document_title(self) -> None:
        d1 = _mk_doc(title="Doc One")
        d2 = _mk_doc(title="Doc Two")
        chunks = [
            _mk_chunk(document_id=d1.id, content="alpha"),
            _mk_chunk(document_id=d2.id, content="gamma"),
            _mk_chunk(document_id=d1.id, content="beta"),
        ]
        out = context_text(_mk_result(documents=[d1, d2], chunks=chunks))
        assert "--- From: Doc One ---" in out
        assert "--- From: Doc Two ---" in out
        # d1 group contains both alpha + beta
        d1_section = out.split("--- From: Doc One ---")[1].split("---")[0]
        assert "alpha" in d1_section
        assert "beta" in d1_section

    def test_chunks_mixed_titled_and_untitled(self) -> None:
        """Mixed titled/untitled chunks: titled groups carry a `--- From: <title> ---` header;
        untitled chunks render as bare content. Both flavors coexist in the same output.
        """
        d_titled = _mk_doc(title="Source A")
        d_untitled = _mk_doc(title=None)
        chunks = [
            _mk_chunk(document_id=d_titled.id, content="titled-alpha"),
            _mk_chunk(document_id=d_untitled.id, content="untitled-body"),
            _mk_chunk(document_id=d_titled.id, content="titled-beta"),
        ]
        out = context_text(_mk_result(documents=[d_titled, d_untitled], chunks=chunks))

        assert "--- From: Source A ---" in out
        assert "titled-alpha" in out
        assert "titled-beta" in out
        # Untitled body appears as bare content — no `--- From: ` header attached.
        assert "untitled-body" in out
        assert "--- From: untitled-body" not in out
        # Sections are separated by the group separator.
        assert "\n\n---\n\n" in out

    def test_max_chunks_slices(self) -> None:
        doc = _mk_doc(title="T")
        chunks = [_mk_chunk(document_id=doc.id, content=f"c{i}") for i in range(10)]
        out = context_text(_mk_result(documents=[doc], chunks=chunks), max_chunks=3)
        assert "c0" in out
        assert "c1" in out
        assert "c2" in out
        assert "c3" not in out

    def test_entities_section_appended(self) -> None:
        e = _mk_entity(name="Alice", entity_type="PERSON", description="founder")
        out = context_text(_mk_result(entities=[e]))
        assert "--- Entities ---" in out
        assert "- Alice (PERSON): founder" in out

    def test_entities_no_description_omits_colon(self) -> None:
        e = _mk_entity(name="Bob", entity_type="PERSON", description="")
        out = context_text(_mk_result(entities=[e]))
        assert "- Bob (PERSON)" in out
        assert "- Bob (PERSON):" not in out

    def test_entities_deduplicated_by_id(self) -> None:
        eid = uuid4()
        e1 = _mk_entity(entity_id=eid, name="Dup", entity_type="C", description="a")
        e2 = _mk_entity(entity_id=eid, name="Dup", entity_type="C", description="b")
        out = context_text(_mk_result(entities=[e1, e2]))
        assert out.count("Dup") == 1

    def test_relationships_resolve_names_from_entities(self) -> None:
        alice_id = uuid4()
        acme_id = uuid4()
        e_alice = _mk_entity(entity_id=alice_id, name="Alice", entity_type="PERSON")
        e_acme = _mk_entity(entity_id=acme_id, name="Acme Corp", entity_type="ORG")
        rel = _mk_rel(
            source_id=alice_id,
            target_id=acme_id,
            relationship_type="FOUNDED",
            description="founded it",
        )
        out = context_text(_mk_result(entities=[e_alice, e_acme], relationships=[rel]))
        assert "--- Relationships ---" in out
        assert "- Alice --FOUNDED--> Acme Corp: founded it" in out

    def test_relationships_uuid_fallback_when_endpoint_missing(self) -> None:
        alice_id = uuid4()
        acme_id = uuid4()
        rel = _mk_rel(source_id=alice_id, target_id=acme_id, relationship_type="KNOWS")
        out = context_text(_mk_result(relationships=[rel]))
        assert str(alice_id) in out
        assert str(acme_id) in out
        assert "--KNOWS-->" in out

    def test_relationships_deduplicated_by_endpoints_and_type(self) -> None:
        a, b = uuid4(), uuid4()
        rel1 = _mk_rel(source_id=a, target_id=b, relationship_type="X", description="d1")
        rel2 = _mk_rel(source_id=a, target_id=b, relationship_type="X", description="d2")
        out = context_text(_mk_result(relationships=[rel1, rel2]))
        rel_lines = [line for line in out.splitlines() if line.startswith("- ") and "--X-->" in line]
        assert len(rel_lines) == 1

    def test_ordering_chunks_then_entities_then_relationships(self) -> None:
        d = _mk_doc(title="T")
        c = _mk_chunk(document_id=d.id, content="some chunk")
        a_id, b_id = uuid4(), uuid4()
        ea = _mk_entity(entity_id=a_id, name="A", entity_type="P")
        eb = _mk_entity(entity_id=b_id, name="B", entity_type="P")
        rel = _mk_rel(source_id=a_id, target_id=b_id, relationship_type="R")
        out = context_text(_mk_result(documents=[d], chunks=[c], entities=[ea, eb], relationships=[rel]))
        chunk_pos = out.index("some chunk")
        ent_pos = out.index("--- Entities ---")
        rel_pos = out.index("--- Relationships ---")
        assert chunk_pos < ent_pos < rel_pos

    def test_entities_only_output_starts_with_leading_newlines(self) -> None:
        """Entities-only output starts with the literal `\\n\\n--- Entities ---\\n\\n` prefix.

        Mirrors the legacy field byte-for-byte — the leading newlines are
        preserved even when there are no preceding chunks.
        """
        e = _mk_entity(name="Solo", entity_type="C", description="d")
        out = context_text(_mk_result(entities=[e]))
        assert out.startswith("\n\n--- Entities ---\n\n")

    def test_relationships_only_output_starts_with_leading_newlines(self) -> None:
        """Relationships-only output starts with the literal `\\n\\n--- Relationships ---\\n\\n` prefix."""
        a, b = uuid4(), uuid4()
        rel = _mk_rel(source_id=a, target_id=b, relationship_type="KNOWS")
        out = context_text(_mk_result(relationships=[rel]))
        assert out.startswith("\n\n--- Relationships ---\n\n")

    def test_entities_and_relationships_without_chunks_keep_prefix(self) -> None:
        """No-chunk composition: entities section comes first, both keep leading newlines."""
        a, b = uuid4(), uuid4()
        ea = _mk_entity(entity_id=a, name="A", entity_type="P")
        eb = _mk_entity(entity_id=b, name="B", entity_type="P")
        rel = _mk_rel(source_id=a, target_id=b, relationship_type="R")
        out = context_text(_mk_result(entities=[ea, eb], relationships=[rel]))
        assert out.startswith("\n\n--- Entities ---")
        ent_pos = out.index("--- Entities ---")
        rel_pos = out.index("--- Relationships ---")
        assert ent_pos < rel_pos

    def test_max_chunks_caps_chunks_only_leaves_entities_intact(self) -> None:
        """`max_chunks` slices the chunk list — entity and relationship sections pass through unbounded."""
        doc = _mk_doc(title="T")
        chunks = [_mk_chunk(document_id=doc.id, content=f"c{i}") for i in range(5)]
        a, b = uuid4(), uuid4()
        ea = _mk_entity(entity_id=a, name="A", entity_type="P", description="ad")
        eb = _mk_entity(entity_id=b, name="B", entity_type="P", description="bd")
        rel = _mk_rel(source_id=a, target_id=b, relationship_type="R", description="rd")

        out = context_text(
            _mk_result(
                documents=[doc],
                chunks=chunks,
                entities=[ea, eb],
                relationships=[rel],
            ),
            max_chunks=2,
        )

        # Chunks are sliced.
        assert "c0" in out
        assert "c1" in out
        assert "c2" not in out
        # Entities and relationships are not capped — both endpoints survive.
        assert "- A (P): ad" in out
        assert "- B (P): bd" in out
        assert "- A --R--> B: rd" in out

    def test_all_three_sections_rendered_together(self) -> None:
        """Chunks (with title header) + entities + relationships all appear in one render."""
        doc = _mk_doc(title="Notes")
        chunk = _mk_chunk(document_id=doc.id, content="alpha")
        a, b = uuid4(), uuid4()
        ea = _mk_entity(entity_id=a, name="A", entity_type="P", description="ad")
        eb = _mk_entity(entity_id=b, name="B", entity_type="P")
        rel = _mk_rel(source_id=a, target_id=b, relationship_type="R", description="rd")

        out = context_text(
            _mk_result(
                documents=[doc],
                chunks=[chunk],
                entities=[ea, eb],
                relationships=[rel],
            )
        )

        assert "--- From: Notes ---" in out
        assert "alpha" in out
        assert "- A (P): ad" in out
        assert "- B (P)" in out
        assert "- A --R--> B: rd" in out


@pytest.mark.unit
class TestContextTextByteEquivalenceWithLegacy:
    """The public helper output is byte-identical to a captured baseline.

    The baseline strings at module level lock the rendered format. Each
    scenario builds a typed ``RecallResult`` whose only non-deterministic
    inputs (UUIDs) are explicitly seeded so the rendered output stays
    reproducible.
    """

    def test_byte_equivalent_full_payload(self) -> None:
        ns_id = UUID("00000000-0000-0000-0000-000000000001")
        doc_id = UUID("00000000-0000-0000-0000-000000000002")
        chunk_id = UUID("00000000-0000-0000-0000-000000000003")
        alice_id = UUID("00000000-0000-0000-0000-000000000004")
        acme_id = UUID("00000000-0000-0000-0000-000000000005")
        rel_id = UUID("00000000-0000-0000-0000-000000000006")

        typed_chunk = RecallChunk(
            id=chunk_id,
            document_id=doc_id,
            content="Alice founded Acme Corp.",
            score=0.9,
            created_at=datetime.now(UTC),
        )
        typed_alice = RecallEntity(
            id=alice_id,
            name="Alice",
            entity_type="PERSON",
            description="Founder",
            score=0.85,
            attributes={},
            mention_count=1,
            source_document_ids=[],
            source_chunk_ids=[],
        )
        typed_acme = RecallEntity(
            id=acme_id,
            name="Acme Corp",
            entity_type="ORGANIZATION",
            description="A company",
            score=0.7,
            attributes={},
            mention_count=1,
            source_document_ids=[],
            source_chunk_ids=[],
        )
        typed_rel = RecallRelationship(
            id=rel_id,
            source_entity_id=alice_id,
            target_entity_id=acme_id,
            relationship_type="FOUNDED",
            description="Founded the company",
            score=0.9,
            valid_from=None,
            valid_until=None,
            source_document_ids=[],
        )
        doc = DocumentProjection(
            id=doc_id,
            created_at=datetime.now(UTC),
            title="Founding Story",
        )
        result = RecallResult(
            query="who founded acme?",
            namespace_id=ns_id,
            documents=[doc],
            chunks=[typed_chunk],
            entities=[typed_alice, typed_acme],
            relationships=[typed_rel],
        )

        assert context_text(result, max_chunks=5) == GOLDEN_FULL_PAYLOAD_CONTEXT_TEXT

    def test_byte_equivalent_chunks_only_untitled(self) -> None:
        ns_id = UUID("00000000-0000-0000-0000-000000000001")
        doc_id = UUID("00000000-0000-0000-0000-000000000002")
        chunk_a_id = UUID("00000000-0000-0000-0000-000000000003")
        chunk_b_id = UUID("00000000-0000-0000-0000-000000000004")

        typed_chunk_a = RecallChunk(
            id=chunk_a_id,
            document_id=doc_id,
            content="alpha text",
            score=0.9,
            created_at=datetime.now(UTC),
        )
        typed_chunk_b = RecallChunk(
            id=chunk_b_id,
            document_id=doc_id,
            content="beta text",
            score=0.8,
            created_at=datetime.now(UTC),
        )

        doc = DocumentProjection(
            id=doc_id,
            created_at=datetime.now(UTC),
            title=None,
        )
        result = RecallResult(
            query="q",
            namespace_id=ns_id,
            documents=[doc],
            chunks=[typed_chunk_a, typed_chunk_b],
            entities=[],
            relationships=[],
        )

        assert context_text(result, max_chunks=5) == GOLDEN_CHUNKS_ONLY_UNTITLED_CONTEXT_TEXT

    def test_byte_equivalent_entities_only(self) -> None:
        ns_id = UUID("00000000-0000-0000-0000-000000000001")
        eid = UUID("00000000-0000-0000-0000-000000000002")
        typed_entity = RecallEntity(
            id=eid,
            name="Solo",
            entity_type="CONCEPT",
            description="The only one",
            score=0.5,
            attributes={},
            mention_count=1,
            source_document_ids=[],
            source_chunk_ids=[],
        )

        result = RecallResult(
            query="q",
            namespace_id=ns_id,
            documents=[],
            chunks=[],
            entities=[typed_entity],
            relationships=[],
        )
        assert context_text(result) == GOLDEN_ENTITIES_ONLY_CONTEXT_TEXT
