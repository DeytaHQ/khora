"""Event decomposition for Chronicle engine.

Extracts structured event tuples from text using LLM analysis.
Each event has: subject, verb, object, datetime range, and confidence.

Based on Chronos (95.6% LongMemEval) event calendar architecture.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import litellm
from loguru import logger

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ChronicleEvent:
    """A structured event extracted from text.

    Represents a subject-verb-object tuple with temporal bounds.
    """

    id: UUID = field(default_factory=uuid4)
    chunk_id: UUID | None = None
    namespace_id: UUID | None = None

    # SVO triple
    subject: str = ""
    verb: str = ""
    object: str = ""

    # Triple timestamps (Mastra OM pattern)
    observation_date: datetime | None = None  # When this was ingested
    referenced_date: datetime | None = None  # When the event occurred
    relative_offset: str = ""  # "last week", "yesterday", etc.

    # Metadata
    confidence: float = 1.0
    source_text: str = ""  # The sentence this was extracted from

    # Embedding of the SVO summary, populated at persistence time so the
    # event-similarity channel can query chronicle_events directly. Optional
    # because the sqlite_lance / LanceDB path does not store vectors here.
    embedding: list[float] | None = None

    @property
    def summary(self) -> str:
        """One-line summary of the event."""
        parts = [self.subject, self.verb, self.object]
        return " ".join(p for p in parts if p)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "subject": self.subject,
            "verb": self.verb,
            "object": self.object,
            "observation_date": self.observation_date.isoformat() if self.observation_date else None,
            "referenced_date": self.referenced_date.isoformat() if self.referenced_date else None,
            "relative_offset": self.relative_offset,
            "confidence": self.confidence,
            "source_text": self.source_text,
        }


# ---------------------------------------------------------------------------
# LLM prompt for event extraction
# ---------------------------------------------------------------------------

_EVENT_SYSTEM = """\
You are a precise event extractor. Given text, extract structured events as \
subject-verb-object tuples with temporal information.

Return a JSON array of events:
[
  {
    "subject": "person or entity performing the action",
    "verb": "action or state change",
    "object": "target of the action",
    "referenced_date": "ISO date if mentioned (e.g. 2025-03-15), or null",
    "relative_offset": "temporal reference if present (e.g. last week, yesterday), or empty string",
    "confidence": 0.0-1.0,
    "source_text": "the original sentence"
  }
]

Rules:
- Extract ALL events, including implicit ones
- Each event must have at least a subject and verb
- Resolve pronouns to actual entity names when possible
- For dates, extract the actual date if stated, otherwise capture the relative reference
- Confidence: 1.0 for explicit statements, 0.5-0.8 for inferred events
- Return an empty array [] if no events can be extracted"""


# ---------------------------------------------------------------------------
# Event extractor
# ---------------------------------------------------------------------------


class EventExtractor:
    """Extracts structured events from text using LLM.

    Decomposes text into SVO (subject-verb-object) tuples with temporal
    information. This is the key differentiator that drives high scores
    on temporal reasoning benchmarks.
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self._model = model

    async def extract_events(
        self,
        text: str,
        *,
        chunk_id: UUID | None = None,
        namespace_id: UUID | None = None,
    ) -> list[ChronicleEvent]:
        """Extract events from a text chunk.

        Args:
            text: Source text to extract events from.
            chunk_id: Optional chunk ID for linking.
            namespace_id: Optional namespace for scoping.

        Returns:
            List of extracted ChronicleEvent objects.
        """
        if not text.strip():
            return []

        now = datetime.now(UTC)

        try:
            response = await litellm.acompletion(
                model=self._model,
                messages=[
                    {"role": "system", "content": _EVENT_SYSTEM},
                    {"role": "user", "content": text[:4000]},  # Cap input
                ],
                temperature=0.0,
                max_tokens=1000,
            )
            content = response.choices[0].message.content or "[]"
            raw_events = self._parse_events(content)
        except Exception:
            logger.debug("Event extraction failed, returning empty")
            return []

        events = []
        for raw in raw_events:
            if not isinstance(raw, dict):
                continue
            subject = raw.get("subject", "")
            verb = raw.get("verb", "")
            if not subject or not verb:
                continue

            # Parse referenced date
            ref_date = None
            ref_str = raw.get("referenced_date")
            if ref_str and isinstance(ref_str, str):
                try:
                    ref_date = datetime.fromisoformat(ref_str.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass

            events.append(
                ChronicleEvent(
                    chunk_id=chunk_id,
                    namespace_id=namespace_id,
                    subject=subject,
                    verb=verb,
                    object=raw.get("object", ""),
                    observation_date=now,
                    referenced_date=ref_date,
                    relative_offset=raw.get("relative_offset", ""),
                    confidence=float(raw.get("confidence", 0.8)),
                    source_text=raw.get("source_text", ""),
                )
            )

        logger.debug(f"Extracted {len(events)} events from {len(text)} chars")
        return events

    @staticmethod
    def _parse_events(content: str) -> list[dict[str, Any]]:
        """Parse JSON array from LLM response."""
        # Direct parse
        try:
            result = json.loads(content)
            if isinstance(result, list):
                return result
            return []
        except json.JSONDecodeError:
            pass

        # Extract from markdown
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", content, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1))
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

        # Find array in content
        start = content.find("[")
        end = content.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                result = json.loads(content[start : end + 1])
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

        return []
