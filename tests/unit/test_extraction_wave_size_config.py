"""#1374: extraction wave size is configurable and flows from config.

Covers:
- ``KHORA_LLM_EXTRACTION_WAVE_SIZE`` resolves to ``config.llm.extraction_wave_size``
  (single-underscore env form).
- ``extract_entities()`` (the chokepoint both ingest paths funnel through)
  threads ``wave_size`` into the constructed ``LLMEntityExtractor``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from khora.config.schema import KhoraConfig
from khora.core.models import Chunk
from khora.pipelines.tasks.extract import extract_entities

pytestmark = pytest.mark.unit


def test_default_extraction_wave_size() -> None:
    """Default config value is 8."""
    assert KhoraConfig().llm.extraction_wave_size == 8


def test_extraction_wave_size_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """KHORA_LLM_EXTRACTION_WAVE_SIZE (single underscore) populates the field."""
    monkeypatch.setenv("KHORA_LLM_EXTRACTION_WAVE_SIZE", "13")
    assert KhoraConfig().llm.extraction_wave_size == 13


@pytest.mark.asyncio
async def test_extract_entities_threads_wave_size_to_extractor() -> None:
    """extract_entities builds the extractor with the supplied wave_size."""
    chunk = Chunk(
        id=uuid4(),
        namespace_id=uuid4(),
        document_id=uuid4(),
        content="Alice works at Acme Corp.",
        embedding=[],
    )

    captured: dict = {}

    def fake_extractor(**kwargs):
        captured.update(kwargs)
        inst = MagicMock()
        inst.extract_multi = AsyncMock(return_value=[])
        return inst

    with patch("khora.extraction.extractors.LLMEntityExtractor", side_effect=fake_extractor):
        await extract_entities(
            [chunk],
            model="test-model",
            wave_size=11,
            entity_types=["PERSON"],
            relationship_types=["WORKS_FOR"],
            selective_extraction=False,
        )

    assert captured["wave_size"] == 11
