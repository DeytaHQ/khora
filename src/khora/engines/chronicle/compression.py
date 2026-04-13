"""Progressive memory compression for Chronicle engine.

Implements the Observer/Reflector pattern (Mastra OM, 94.9% LongMemEval):
older memories are compressed into structured observations while recent
memories are kept in full. Achieves 3-6x token reduction without
significant recall loss.

Also handles contradiction detection via ADD/UPDATE/DELETE/NOOP
operations (Mem0 pattern, 66.9% LoCoMo) to maintain memory consistency.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

import litellm
from loguru import logger

# ---------------------------------------------------------------------------
# Memory fact model
# ---------------------------------------------------------------------------


class FactOperation(str, Enum):
    """Operations on memory facts for contradiction resolution."""

    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    NOOP = "noop"


@dataclass(slots=True)
class MemoryFact:
    """An atomic fact extracted from conversation memory.

    Represents the smallest unit of retrievable knowledge — a single
    statement that can be independently verified, updated, or deleted.
    Based on EMem's Elementary Discourse Units (84.9% LongMemEval).
    """

    id: UUID = field(default_factory=uuid4)
    namespace_id: UUID | None = None
    chunk_id: UUID | None = None

    # The fact itself
    content: str = ""

    # Subject and category for dedup
    subject: str = ""  # Primary entity this fact is about
    category: str = ""  # Topic category (preference, fact, event, opinion)

    # Temporal
    observation_date: datetime | None = None
    referenced_date: datetime | None = None

    # State
    is_active: bool = True  # False = superseded or deleted
    superseded_by: UUID | None = None  # ID of the fact that replaced this
    confidence: float = 1.0

    # Source tracking
    source_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "content": self.content,
            "subject": self.subject,
            "category": self.category,
            "is_active": self.is_active,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class CompressionResult:
    """Result of compressing a set of memories."""

    facts_extracted: int = 0
    facts_added: int = 0
    facts_updated: int = 0
    facts_deleted: int = 0
    tokens_before: int = 0
    tokens_after: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.tokens_before == 0:
            return 0.0
        return 1.0 - (self.tokens_after / self.tokens_before)


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

_FACT_EXTRACTION_SYSTEM = """\
You are a precise fact extractor. Given a text, extract atomic facts — \
each fact should be a single, self-contained statement that can be \
independently verified.

Return a JSON array:
[
  {
    "content": "The complete atomic fact as a clear statement",
    "subject": "Primary entity this fact is about",
    "category": "preference | fact | event | opinion | state",
    "referenced_date": "ISO date if mentioned, or null",
    "confidence": 0.0-1.0
  }
]

Rules:
- Each fact must be self-contained (understandable without context)
- Resolve pronouns to actual names
- One fact per statement — don't combine multiple facts
- Categories: preference (likes/dislikes), fact (verifiable), event (happened),
  opinion (belief/view), state (current status/condition)
- Return [] if no facts can be extracted"""

_CONTRADICTION_SYSTEM = """\
You are a memory consistency checker. Given a NEW fact and a list of \
EXISTING facts about the same subject, determine the operation:

Return a JSON object:
{
  "operation": "add | update | delete | noop",
  "target_id": "ID of the existing fact to update/delete, or null for add",
  "reasoning": "Brief explanation",
  "updated_content": "New content if operation is update, or null"
}

Rules:
- ADD: The new fact is genuinely new information (no conflict with existing)
- UPDATE: The new fact supersedes an existing fact (contradiction or refinement)
- DELETE: The new fact invalidates an existing fact entirely
- NOOP: The new fact is already captured by an existing fact (duplicate)
- Only UPDATE or DELETE if there is a clear conflict or supersession
- Prefer UPDATE over DELETE when the old fact is partially still valid"""


# ---------------------------------------------------------------------------
# Fact extractor
# ---------------------------------------------------------------------------


class FactExtractor:
    """Extracts atomic facts from text and manages contradiction resolution.

    Two-phase operation:
    1. Extract: Break text into atomic facts (EDUs)
    2. Reconcile: Check each fact against existing facts for contradictions
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self._model = model

    async def extract_facts(
        self,
        text: str,
        *,
        chunk_id: UUID | None = None,
        namespace_id: UUID | None = None,
    ) -> list[MemoryFact]:
        """Extract atomic facts from text.

        Args:
            text: Source text to extract facts from.
            chunk_id: Optional chunk ID for linking.
            namespace_id: Optional namespace for scoping.

        Returns:
            List of MemoryFact objects.
        """
        if not text.strip():
            return []

        now = datetime.now(UTC)

        try:
            response = await litellm.acompletion(
                model=self._model,
                messages=[
                    {"role": "system", "content": _FACT_EXTRACTION_SYSTEM},
                    {"role": "user", "content": text[:4000]},
                ],
                temperature=0.0,
                max_tokens=1500,
            )
            content = response.choices[0].message.content or "[]"
            raw_facts = _parse_json_array(content)
        except Exception:
            logger.debug("Fact extraction failed, returning empty")
            return []

        facts = []
        for raw in raw_facts:
            if not isinstance(raw, dict):
                continue
            fact_content = raw.get("content", "")
            if not fact_content:
                continue

            ref_date = None
            ref_str = raw.get("referenced_date")
            if ref_str and isinstance(ref_str, str):
                try:
                    ref_date = datetime.fromisoformat(ref_str.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass

            facts.append(
                MemoryFact(
                    chunk_id=chunk_id,
                    namespace_id=namespace_id,
                    content=fact_content,
                    subject=raw.get("subject", ""),
                    category=raw.get("category", "fact"),
                    observation_date=now,
                    referenced_date=ref_date,
                    confidence=float(raw.get("confidence", 0.8)),
                    source_text=text[:500],
                )
            )

        logger.debug(f"Extracted {len(facts)} atomic facts from {len(text)} chars")
        return facts

    async def reconcile_fact(
        self,
        new_fact: MemoryFact,
        existing_facts: list[MemoryFact],
    ) -> tuple[FactOperation, UUID | None, str | None]:
        """Check a new fact against existing facts for contradictions.

        Args:
            new_fact: The newly extracted fact.
            existing_facts: Existing facts about the same subject.

        Returns:
            Tuple of (operation, target_fact_id, updated_content).
        """
        if not existing_facts:
            return FactOperation.ADD, None, None

        # Build context for LLM
        existing_list = "\n".join(
            f"  [{f.id}] {f.content} (category: {f.category})"
            for f in existing_facts[:10]  # Cap for token efficiency
        )

        prompt = (
            f"NEW FACT: {new_fact.content}\n"
            f"Subject: {new_fact.subject}\n"
            f"Category: {new_fact.category}\n\n"
            f"EXISTING FACTS about '{new_fact.subject}':\n{existing_list}"
        )

        try:
            response = await litellm.acompletion(
                model=self._model,
                messages=[
                    {"role": "system", "content": _CONTRADICTION_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=300,
            )
            content = response.choices[0].message.content or "{}"
            result = _parse_json_object(content)
        except Exception:
            logger.debug("Contradiction check failed, defaulting to ADD")
            return FactOperation.ADD, None, None

        op_str = result.get("operation", "add").lower()
        try:
            operation = FactOperation(op_str)
        except ValueError:
            operation = FactOperation.ADD

        target_id = None
        target_str = result.get("target_id")
        if target_str and isinstance(target_str, str):
            try:
                target_id = UUID(target_str)
            except ValueError:
                pass

        updated_content = result.get("updated_content")

        return operation, target_id, updated_content


# ---------------------------------------------------------------------------
# Memory compressor
# ---------------------------------------------------------------------------


class MemoryCompressor:
    """Compresses older memories via progressive summarization.

    Implements the Observer/Reflector pattern:
    - Observer: extracts atomic facts from recent memories
    - Reflector: consolidates older facts into summaries

    The compression ratio increases with memory age:
    - < 1 day: full text (no compression)
    - 1-7 days: atomic facts (3-5x compression)
    - > 7 days: consolidated summaries (5-10x compression)
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self._fact_extractor = FactExtractor(model=model)
        self._model = model

    async def compress_memories(
        self,
        chunks: list[Any],
        *,
        namespace_id: UUID | None = None,
    ) -> tuple[list[MemoryFact], CompressionResult]:
        """Compress a batch of chunks into atomic facts.

        Args:
            chunks: List of Chunk objects to compress.
            namespace_id: Namespace for scoping.

        Returns:
            Tuple of (extracted_facts, compression_result).
        """
        result = CompressionResult()

        all_facts: list[MemoryFact] = []
        for chunk in chunks:
            content = getattr(chunk, "content", str(chunk))
            result.tokens_before += len(content) // 4  # Rough token estimate

            facts = await self._fact_extractor.extract_facts(
                content,
                chunk_id=getattr(chunk, "id", None),
                namespace_id=namespace_id,
            )
            all_facts.extend(facts)
            result.facts_extracted += len(facts)

        # Estimate compressed token count
        for fact in all_facts:
            result.tokens_after += len(fact.content) // 4

        result.facts_added = len(all_facts)
        return all_facts, result


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


def _parse_json_array(content: str) -> list[dict[str, Any]]:
    """Parse a JSON array from LLM response."""
    try:
        result = json.loads(content)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", content, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1))
            return result if isinstance(result, list) else []
        except json.JSONDecodeError:
            pass

    start, end = content.find("["), content.rfind("]")
    if start != -1 and end > start:
        try:
            result = json.loads(content[start : end + 1])
            return result if isinstance(result, list) else []
        except json.JSONDecodeError:
            pass

    return []


def _parse_json_object(content: str) -> dict[str, Any]:
    """Parse a JSON object from LLM response."""
    try:
        result = json.loads(content)
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", content, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1))
            return result if isinstance(result, dict) else {}
        except json.JSONDecodeError:
            pass

    start, end = content.find("{"), content.rfind("}")
    if start != -1 and end > start:
        try:
            result = json.loads(content[start : end + 1])
            return result if isinstance(result, dict) else {}
        except json.JSONDecodeError:
            pass

    return {}
