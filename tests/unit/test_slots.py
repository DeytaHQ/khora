"""Verify slots=True was applied to high-frequency dataclasses."""

from khora.core.models.document import Chunk, ChunkMetadata, DocumentMetadata
from khora.core.models.entity import Entity, Episode, Relationship
from khora.memory_lake import BatchResult, RecallResult, RememberResult, Stats


class TestSlotsApplied:
    def test_document_models_have_slots(self):
        assert hasattr(DocumentMetadata, "__slots__")
        assert hasattr(ChunkMetadata, "__slots__")
        assert hasattr(Chunk, "__slots__")

    def test_entity_models_have_slots(self):
        assert hasattr(Entity, "__slots__")
        assert hasattr(Relationship, "__slots__")
        assert hasattr(Episode, "__slots__")

    def test_result_types_have_slots(self):
        assert hasattr(RememberResult, "__slots__")
        assert hasattr(BatchResult, "__slots__")
        assert hasattr(RecallResult, "__slots__")
        assert hasattr(Stats, "__slots__")
