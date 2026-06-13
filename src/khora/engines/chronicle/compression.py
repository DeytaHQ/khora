"""Progressive memory compression for Chronicle engine.

Implements the Observer/Reflector pattern (Mastra OM, 94.9% LongMemEval):
older memories are compressed into structured observations while recent
memories are kept in full. Achieves 3-6x token reduction without
significant recall loss.

Also handles contradiction detection via ADD/UPDATE/DELETE/NOOP
operations (Mem0 pattern, 66.9% LoCoMo) to maintain memory consistency.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

import litellm
from loguru import logger

from khora.config.llm import DEFAULT_LLM_TIMEOUT_S, llm_call_timeout
from khora.core.diagnostics import ErrorRecord
from khora.engines.chronicle.events import (
    _EXTRACTION_FAILED_COUNTER as _CHRONICLE_EXTRACTION_FAILED_COUNTER,
)
from khora.engines.chronicle.events import (
    _TRANSIENT_LLM_ERRORS as _CHRONICLE_TRANSIENT_LLM_ERRORS,
)
from khora.telemetry import metric_counter

# Counter for reconcile_fact LLM failures. Issue #892: previously the broad
# ``except Exception`` silently fell back to ADD on every LLM error,
# accumulating contradictory facts. Now narrowed to transient errors; the
# fact is skipped and this counter is the operator's only signal that
# reconciliation is degrading. NO namespace_id label - cardinality rule.
_RECONCILE_FAILURES = metric_counter(
    "khora.chronicle.reconcile.failures_total",
    unit="1",
    description=(
        "Issue #892. reconcile_fact() calls where the LLM contradiction-check "
        "raised a transient error (rate limit, timeout, connection error, "
        "JSON parse error). The fact is skipped (NOT added) to prevent silent "
        "contradiction accumulation. NO namespace_id label - cardinality rule."
    ),
)

# Transient LLM-side failures we accept as "model never answered" and
# treat as a skip. Real bugs (AttributeError, TypeError, KeyError on a
# logic path) must propagate so they get noticed.
_TRANSIENT_LLM_ERRORS: tuple[type[BaseException], ...] = (
    litellm.exceptions.RateLimitError,
    litellm.exceptions.Timeout,
    litellm.exceptions.APIConnectionError,
    litellm.exceptions.APIError,
    litellm.exceptions.ServiceUnavailableError,
    litellm.exceptions.InternalServerError,
    asyncio.TimeoutError,
    json.JSONDecodeError,
)

# ---------------------------------------------------------------------------
# Memory fact model
# ---------------------------------------------------------------------------


class FactOperation(str, Enum):
    """Operations on memory facts for contradiction resolution."""

    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    NOOP = "noop"
    # Issue #892: the contradiction-check LLM call failed transiently (rate
    # limit, timeout, connection error, malformed JSON). The caller MUST
    # treat this as "do not persist this fact" - skipping is safer than
    # fail-open to ADD, which silently accumulates contradictory facts.
    SKIP = "skip"


@dataclass(slots=True)
class MemoryFact:
    """An atomic SVO fact extracted from conversation memory.

    Matches the ``memory_facts`` schema introduced in Chronicle #1
    (migration 024): subject/predicate/object_/fact_text plus supersession
    tracking. ``fact_text`` is the natural-language form used for display
    and retrieval; the SVO triple powers structured queries and reconciliation.
    """

    # Identity
    id: UUID = field(default_factory=uuid4)
    namespace_id: UUID | None = None

    # Atomic SVO claim (matches memory_facts table)
    subject: str = ""
    predicate: str = ""
    object_: str = ""
    fact_text: str = ""

    # Confidence
    confidence: float = 1.0

    # Supersession tracking
    is_active: bool = True
    superseded_by: UUID | None = None

    # Source tracking — list of chunk IDs that contributed to this fact
    source_chunk_ids: list[UUID] = field(default_factory=list)

    # Event-time anchor (#1144) — the fact's real-world time, derived at
    # extraction from the source chunk's occurred_at / source_timestamp.
    # Distinct from created_at (ingestion order). Used to order supersession
    # so a backfilled OLDER fact ingested later does not overwrite a NEWER
    # active fact. Not persisted (the memory_facts table has no event-time
    # column); facts read back from storage carry event_time=None and fall
    # back to created_at for comparison.
    event_time: datetime | None = None

    # Timestamps (set by DB on insert; optional in-memory)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object_,
            "fact_text": self.fact_text,
            "confidence": self.confidence,
            "is_active": self.is_active,
            "superseded_by": str(self.superseded_by) if self.superseded_by else None,
            "source_chunk_ids": [str(cid) for cid in self.source_chunk_ids],
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
You are a precise fact extractor. Given a text, extract atomic SVO facts — \
each fact is a (subject, predicate, object) triple expressing a single \
self-contained claim that can be independently verified.

Return a JSON array:
[
  {
    "subject": "Primary entity this fact is about",
    "predicate": "relation or attribute (e.g. works_at, lives_in, prefers, is)",
    "object": "value or related entity",
    "fact_text": "Natural-language form of the fact",
    "confidence": 0.0-1.0
  }
]

Rules:
- Each fact must be self-contained (understandable without context)
- Resolve pronouns to actual names
- One fact per statement — don't combine multiple facts
- ``predicate`` should be a short relation token (snake_case preferred)
- ``fact_text`` is the natural sentence a human would read
- Return [] if no facts can be extracted"""

_CONTRADICTION_SYSTEM = """\
You are a memory consistency checker. Given a NEW fact and a list of \
EXISTING facts about the same subject, determine the operation:

Return a JSON object:
{
  "operation": "add | update | delete | noop",
  "target_id": "ID of the existing fact to update/delete, or null for add",
  "reasoning": "Brief explanation"
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
    """Extracts atomic SVO facts from text.

    Single-purpose: text → list[MemoryFact]. Reconciliation (ADD/UPDATE/
    DELETE/NOOP) lives on ``MemoryCompressor.reconcile_fact`` so the
    extractor stays a thin wrapper around the LLM call.
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self._model = model

    async def extract_facts(
        self,
        text: str,
        *,
        chunk_id: UUID | None = None,
        namespace_id: UUID | None = None,
        errors_out: list[ErrorRecord] | None = None,
    ) -> list[MemoryFact]:
        """Extract atomic facts from text.

        Args:
            text: Source text to extract facts from.
            chunk_id: Optional chunk ID — recorded in ``source_chunk_ids``.
            namespace_id: Optional namespace for scoping.
            errors_out: Optional sink for ADR-001 ``ErrorRecord`` entries.
                When the LLM raises a transient error, an entry is appended
                so the caller can surface it on
                ``RememberResult.metadata['errors']``. Real parser bugs are
                not appended - they propagate.

        Returns:
            List of MemoryFact objects. Returns [] on transient LLM failure
            (issue #903).
        """
        if not text.strip():
            return []

        import time as _time

        from khora.telemetry import get_collector

        _t0 = _time.perf_counter()
        try:
            response = await asyncio.wait_for(
                litellm.acompletion(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": _FACT_EXTRACTION_SYSTEM},
                        {"role": "user", "content": text[:4000]},
                    ],
                    temperature=0.0,
                    max_tokens=1500,
                ),
                llm_call_timeout(DEFAULT_LLM_TIMEOUT_S),
            )
            content = response.choices[0].message.content or "[]"
            raw_facts = _parse_json_array(content)
        except _CHRONICLE_TRANSIENT_LLM_ERRORS as exc:
            # Issue #903: narrow the except clause to transient LLM errors;
            # real parser bugs (AttributeError, KeyError) propagate. Promote
            # the log to WARNING with exc_info so the traceback survives.
            logger.warning(
                "Fact extraction transient LLM failure "
                "(chunk_id={chunk_id}, namespace_id={namespace_id}): {exc_type}: {exc}",
                chunk_id=chunk_id,
                namespace_id=namespace_id,
                exc_type=type(exc).__name__,
                exc=exc,
                exc_info=True,
            )
            _CHRONICLE_EXTRACTION_FAILED_COUNTER.add(
                1,
                attributes={"kind": "facts", "reason": "llm_transient_failure"},
            )
            if errors_out is not None:
                errors_out.append(
                    ErrorRecord(
                        component="chronicle.facts_extractor",
                        reason="llm_transient_failure",
                        exception=type(exc).__name__,
                        detail=str(exc)[:200] or None,
                    )
                )
            return []

        _latency = (_time.perf_counter() - _t0) * 1000
        usage = getattr(response, "usage", None)
        get_collector().record_llm_call(
            operation="fact_extraction",
            model=self._model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
            latency_ms=_latency,
            namespace_id=namespace_id,
        )

        facts: list[MemoryFact] = []
        for raw in raw_facts:
            if not isinstance(raw, dict):
                continue
            subject = (raw.get("subject") or "").strip()
            predicate = (raw.get("predicate") or "").strip()
            obj = (raw.get("object") or "").strip()
            fact_text = (raw.get("fact_text") or "").strip()

            # Require at minimum a subject and predicate; if fact_text is
            # missing, synthesise it from the triple so downstream consumers
            # always have a readable form.
            if not subject or not predicate:
                continue
            if not fact_text:
                fact_text = " ".join(p for p in (subject, predicate, obj) if p)

            facts.append(
                MemoryFact(
                    namespace_id=namespace_id,
                    subject=subject,
                    predicate=predicate,
                    object_=obj,
                    fact_text=fact_text,
                    confidence=float(raw.get("confidence", 0.8)),
                    source_chunk_ids=[chunk_id] if chunk_id else [],
                )
            )

        logger.debug(f"Extracted {len(facts)} atomic facts from {len(text)} chars")
        return facts


# ---------------------------------------------------------------------------
# Memory compressor
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ReconcileAction:
    """Result of comparing a new fact against existing facts.

    ``op`` is the action to take. ``target`` is the existing fact to
    supersede when ``op`` is UPDATE or DELETE; ``None`` for ADD/NOOP.
    """

    op: FactOperation
    target: MemoryFact | None = None


class MemoryCompressor:
    """Compresses older memories via progressive summarization.

    Implements the Observer/Reflector pattern:
    - Observer: extracts atomic facts from recent memories
    - Reflector: consolidates older facts into summaries

    Also exposes ``reconcile_fact`` — the ADD/UPDATE/DELETE/NOOP rule that
    decides what to do with a freshly extracted fact given the active
    facts already on disk for the same subject.
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
        """Compress a batch of chunks into atomic facts."""
        result = CompressionResult()

        all_facts: list[MemoryFact] = []
        for chunk in chunks:
            content = getattr(chunk, "content", str(chunk))
            result.tokens_before += len(content) // 4

            facts = await self._fact_extractor.extract_facts(
                content,
                chunk_id=getattr(chunk, "id", None),
                namespace_id=namespace_id,
            )
            all_facts.extend(facts)
            result.facts_extracted += len(facts)

        for fact in all_facts:
            result.tokens_after += len(fact.fact_text) // 4

        result.facts_added = len(all_facts)
        return all_facts, result

    async def reconcile_fact(
        self,
        existing_facts: list[MemoryFact],
        new_fact: MemoryFact,
    ) -> ReconcileAction:
        """Decide ADD/UPDATE/DELETE/NOOP for a new fact.

        Args:
            existing_facts: Active facts already stored for the new fact's subject.
            new_fact: Newly extracted fact.

        Returns:
            ReconcileAction with the operation and (for UPDATE/DELETE) the
            target existing fact to supersede.
        """
        if not existing_facts:
            return ReconcileAction(op=FactOperation.ADD)

        # Cheap pre-check: identical (subject, predicate, object_) triple
        # already exists → NOOP without spending an LLM call.
        for f in existing_facts:
            if f.subject == new_fact.subject and f.predicate == new_fact.predicate and f.object_ == new_fact.object_:
                return ReconcileAction(op=FactOperation.NOOP, target=f)

        existing_list = "\n".join(
            f"  [{f.id}] {f.fact_text} (predicate: {f.predicate}, object: {f.object_})" for f in existing_facts[:10]
        )

        prompt = (
            f"NEW FACT: {new_fact.fact_text}\n"
            f"Subject: {new_fact.subject}\n"
            f"Predicate: {new_fact.predicate}\n"
            f"Object: {new_fact.object_}\n\n"
            f"EXISTING FACTS about '{new_fact.subject}':\n{existing_list}"
        )

        import time as _time

        from khora.telemetry import get_collector

        _t0 = _time.perf_counter()
        try:
            response = await asyncio.wait_for(
                litellm.acompletion(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": _CONTRADICTION_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=200,
                ),
                llm_call_timeout(DEFAULT_LLM_TIMEOUT_S),
            )
            content = response.choices[0].message.content or "{}"
            result = _parse_json_object(content)
        except _TRANSIENT_LLM_ERRORS as exc:
            # Issue #892: NEVER fail open to ADD here. The old behavior
            # silently accumulated contradictory facts whenever the LLM
            # was unavailable. Skip the fact instead and bump a counter
            # so operators can see reconciliation degrading.
            logger.warning(
                "reconcile_fact transient LLM failure - skipping fact "
                "(subject={subject!r}, predicate={predicate!r}, "
                "namespace_id={namespace_id}): {exc_type}: {exc}",
                subject=new_fact.subject,
                predicate=new_fact.predicate,
                namespace_id=new_fact.namespace_id,
                exc_type=type(exc).__name__,
                exc=exc,
            )
            _RECONCILE_FAILURES.add(1)
            return ReconcileAction(op=FactOperation.SKIP)

        _latency = (_time.perf_counter() - _t0) * 1000
        usage = getattr(response, "usage", None)
        get_collector().record_llm_call(
            operation="fact_reconcile",
            model=self._model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
            latency_ms=_latency,
        )

        op_str = str(result.get("operation", "add")).lower()
        try:
            operation = FactOperation(op_str)
        except ValueError:
            operation = FactOperation.ADD
        # SKIP is reserved for the transient-error path; if the LLM ever
        # returns "skip", treat it as ADD so we don't accidentally drop
        # legitimate facts.
        if operation is FactOperation.SKIP:
            operation = FactOperation.ADD

        target: MemoryFact | None = None
        target_str = result.get("target_id")
        if target_str and isinstance(target_str, str):
            try:
                target_id = UUID(target_str)
            except ValueError:
                target_id = None
            if target_id is not None:
                for f in existing_facts:
                    if f.id == target_id:
                        target = f
                        break

        # Defensive: UPDATE/DELETE/NOOP require a target. Without one, the
        # safe action is ADD (treat as a new fact) so we never lose data.
        if operation in (FactOperation.UPDATE, FactOperation.DELETE, FactOperation.NOOP) and target is None:
            operation = FactOperation.ADD

        return ReconcileAction(op=operation, target=target)


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
