"""Level 2 (nano-LLM yes/no) evaluator for semantic hook filters.

Default OFF — gated by ``SemanticHooksConfig.llm_evaluation_enabled``. Only
invoked from the dispatcher after Level 0 (type) and Level 1 (embedding
cosine) have both passed AND the filter supplied ``examples`` to anchor the
prompt. Without examples the LLM has no calibration and produces noise, so
we skip Level 2 entirely.

Design (per AI/IR expert review):

- Micro-batched: up to ``batch_size`` (default 10) entity-filter pairs are
  buffered per event-loop window of ``batch_flush_ms`` (default 100ms) and
  flushed in one ``acompletion`` call. Cuts LLM call volume ~10x without
  changing perceived latency (ingest already pays embedding + storage RTT).
- Static system prompt for prefix-cache wins on OpenAI nano tiers.
- JSON schema-shaped output: ``{"results":[{"i":int,"match":bool,"confidence":float},...]}``.
- Per-namespace rolling-hour token budget. Breach → fail open (return True)
  + ``khora.hooks.llm.throttled_total`` counter + warn-once per window.
- All failures (LLM exception, timeout, parse error) → fail open. Level 1
  already said the entity is similar; dropping the match on infrastructure
  trouble is the wrong default.

Telemetry surface (declared in ``docs/telemetry-contract.json``):

- ``khora.hooks.llm.evaluations_total`` counter, label ``category`` in
  {match, no_match, timeout, budget_exceeded}.
- ``khora.hooks.llm.tokens_total`` counter, label ``direction`` in
  {input, output}.
- ``khora.hooks.llm.throttled_total`` counter, no labels.

No ``namespace_id`` label anywhere — would blow cardinality (see CLAUDE.md
gotcha "Cardinality rule").
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

from khora.telemetry.metrics import metric_counter

from .models import SemanticFilter, SemanticHooksConfig

if TYPE_CHECKING:
    from khora.core.models.event import MemoryEvent


# Module-level OTel instruments. Created once per process so the meter
# provider can de-duplicate by name. No-op when no MeterProvider is set.
_EVAL_COUNTER = metric_counter(
    "khora.hooks.llm.evaluations_total",
    description="Level 2 hook evaluations by outcome category.",
)
_TOKEN_COUNTER = metric_counter(
    "khora.hooks.llm.tokens_total",
    description="Level 2 hook token spend by direction.",
)
_THROTTLE_COUNTER = metric_counter(
    "khora.hooks.llm.throttled_total",
    description="Level 2 hook evaluations refused by per-namespace token budget.",
)


_BUDGET_WINDOW_SECONDS = 3600.0  # 1 hour rolling window

_SYSTEM_PROMPT = (
    "You evaluate whether extracted entities match user-defined filters.\n"
    "For each pair, decide if the entity is an INSTANCE of the filter "
    "description, guided by positive examples; clearly NOT matching "
    "anti-examples.\n"
    "Output a single JSON object of the shape "
    '{"results":[{"i":<index>,"match":<bool>,"confidence":<0.0-1.0>}, ...]} '
    "with one element per pair, in the same order.\n"
    "No prose, no markdown, no code fences."
)


@dataclass
class _NamespaceBudget:
    """Per-namespace rolling-hour token bucket.

    ``namespace_id`` may be ``None`` for global-scope filters; we key on
    ``None`` directly which lumps all global filters into one bucket.
    """

    tokens_used: int = 0
    window_started_at: float = field(default_factory=time.monotonic)
    warned_in_window: bool = False


@dataclass
class _PendingEvaluation:
    """One queued (event, filter) pair awaiting a batch flush."""

    event: MemoryEvent
    filter: SemanticFilter
    future: asyncio.Future[bool]


class LLMFilterEvaluator:
    """Level 2 micro-batched LLM evaluator. See module docstring."""

    def __init__(
        self,
        config: SemanticHooksConfig,
        *,
        batch_size: int | None = None,
        batch_flush_ms: float | None = None,
    ) -> None:
        self._config = config
        self._batch_size = batch_size if batch_size is not None else config.llm_batch_size
        self._batch_flush_ms = batch_flush_ms if batch_flush_ms is not None else config.llm_batch_flush_ms
        self._budgets: dict[UUID | None, _NamespaceBudget] = {}
        self._queue: list[_PendingEvaluation] = []
        self._queue_lock = asyncio.Lock()
        self._flush_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        event: MemoryEvent,
        filter: SemanticFilter,
    ) -> bool:
        """Evaluate whether ``event`` matches ``filter`` per nano-LLM.

        Returns True on match, False on no-match, and **True (fail-open)**
        on any infrastructure trouble (budget breach, timeout, parse
        error, LLM exception). Level 1 already passed; we will not drop a
        cosine-similar match because the LLM tier is flaky.
        """
        loop = asyncio.get_event_loop()
        future: asyncio.Future[bool] = loop.create_future()

        async with self._queue_lock:
            self._queue.append(_PendingEvaluation(event=event, filter=filter, future=future))
            should_flush_now = len(self._queue) >= self._batch_size

            if should_flush_now:
                pending = self._queue
                self._queue = []
                if self._flush_task and not self._flush_task.done():
                    self._flush_task.cancel()
                    self._flush_task = None
            elif self._flush_task is None or self._flush_task.done():
                self._flush_task = loop.create_task(self._flush_after_delay())

        if should_flush_now:
            # Fire-and-forget — pending items resolve their own futures.
            loop.create_task(self._run_batch(pending))

        return await future

    # ------------------------------------------------------------------
    # Batching internals
    # ------------------------------------------------------------------

    async def _flush_after_delay(self) -> None:
        try:
            await asyncio.sleep(self._batch_flush_ms / 1000.0)
        except asyncio.CancelledError:
            return

        async with self._queue_lock:
            if not self._queue:
                return
            pending = self._queue
            self._queue = []
            self._flush_task = None

        await self._run_batch(pending)

    async def _run_batch(self, pending: list[_PendingEvaluation]) -> None:
        """Group by namespace (for budgeting) then by filter (for prompt
        reuse). For Phase 1 simplicity we run one LLM call per
        (namespace, filter) sub-batch — keeps the prompt's filter
        definition stable and avoids interleaved-namespace budget
        accounting bugs. The common case (one filter, one namespace) is
        still a single call.
        """
        # Bucket by (namespace_id, filter_id) — preserves arrival order
        # within each bucket so the returned indices map correctly.
        buckets: dict[tuple[UUID | None, UUID], list[_PendingEvaluation]] = {}
        for p in pending:
            key = (p.event.namespace_id, p.filter.id)
            buckets.setdefault(key, []).append(p)

        for items in buckets.values():
            await self._evaluate_bucket(items)

    async def _evaluate_bucket(self, items: list[_PendingEvaluation]) -> None:
        """Evaluate one (namespace, filter) bucket via one LLM call."""
        filt = items[0].filter
        namespace_id = items[0].event.namespace_id

        # Estimate tokens before charging. Char/4 is the standard rule
        # of thumb for English; an over-estimate fails closed (good).
        prompt_estimate = self._estimate_input_tokens(filt, items)
        output_estimate = 25 * len(items)  # ~25 tokens per JSON result object

        if not self._charge_budget(namespace_id, prompt_estimate, output_estimate):
            _EVAL_COUNTER.add(len(items), attributes={"category": "budget_exceeded"})
            _THROTTLE_COUNTER.add(1)
            for p in items:
                if not p.future.done():
                    p.future.set_result(True)  # fail open
            return

        try:
            results = await self._call_llm(filt, items)
        except TimeoutError:
            logger.warning("Level 2 LLM evaluation timed out for filter {}", filt.name)
            _EVAL_COUNTER.add(len(items), attributes={"category": "timeout"})
            for p in items:
                if not p.future.done():
                    p.future.set_result(True)  # fail open
            return
        except Exception as exc:
            logger.warning("Level 2 LLM evaluation failed for filter {}: {}", filt.name, exc)
            _EVAL_COUNTER.add(len(items), attributes={"category": "timeout"})
            for p in items:
                if not p.future.done():
                    p.future.set_result(True)  # fail open
            return

        threshold = filt.llm_confidence_threshold
        for idx, p in enumerate(items):
            entry = results.get(idx)
            if entry is None:
                # Missing index → fail open, but count as timeout (parser issue).
                if not p.future.done():
                    p.future.set_result(True)
                _EVAL_COUNTER.add(1, attributes={"category": "timeout"})
                continue
            match_flag = bool(entry.get("match", False))
            confidence = float(entry.get("confidence", 0.0) or 0.0)
            decision = match_flag and confidence >= threshold
            if not p.future.done():
                p.future.set_result(decision)
            _EVAL_COUNTER.add(1, attributes={"category": "match" if decision else "no_match"})

    # ------------------------------------------------------------------
    # LLM call + parsing
    # ------------------------------------------------------------------

    async def _call_llm(
        self,
        filt: SemanticFilter,
        items: list[_PendingEvaluation],
    ) -> dict[int, dict[str, Any]]:
        """Make one ``acompletion`` call and return ``{index: result_dict}``."""
        # Local import so the hooks module doesn't transitively require
        # litellm at import time.
        from khora.config.llm import LiteLLMConfig, acompletion

        model = filt.filter_model or self._config.filter_model
        llm_config = LiteLLMConfig(
            model=model,
            temperature=0.0,
            max_tokens=200 + 30 * len(items),
            timeout=10,
        )

        user_message = self._build_user_message(filt, items)

        response_text = await acompletion(
            prompt=user_message,
            config=llm_config,
            system_prompt=_SYSTEM_PROMPT,
            response_format={"type": "json_object"},
            _telemetry_op="hooks.filter_eval",
        )

        # Approximate token spend for the local bucket counter. The
        # underlying ``acompletion`` already records the real token spend
        # to ``khora.llm.tokens`` via the central LLMUsage path; this
        # local counter is the bounded-label view used for hook-specific
        # dashboards.
        input_tokens = max(1, len(user_message) // 4 + len(_SYSTEM_PROMPT) // 4)
        output_tokens = max(1, len(response_text) // 4)
        _TOKEN_COUNTER.add(input_tokens, attributes={"direction": "input"})
        _TOKEN_COUNTER.add(output_tokens, attributes={"direction": "output"})

        return self._parse_response(response_text)

    def _build_user_message(
        self,
        filt: SemanticFilter,
        items: list[_PendingEvaluation],
    ) -> str:
        lines: list[str] = []
        lines.append(f"Filter: {filt.description}")
        if filt.examples:
            lines.append(f"examples: {json.dumps(filt.examples[:5])}")
        if filt.anti_examples:
            lines.append(f"anti_examples: {json.dumps(filt.anti_examples[:5])}")
        lines.append("")
        lines.append("Evaluate:")
        for idx, p in enumerate(items):
            data = p.event.data
            name = str(data.get("name", ""))[:100]
            etype = str(data.get("entity_type", data.get("relationship_type", "")))[:50]
            descr = str(data.get("description", ""))[:200]
            lines.append(f"[{idx}] name={name!r} type={etype!r} description={descr!r}")
        return "\n".join(lines)

    @staticmethod
    def _parse_response(text: str) -> dict[int, dict[str, Any]]:
        """Parse the LLM JSON response into ``{index: result}``.

        Returns empty dict on parse failure — caller fails open.
        """
        cleaned = text.strip()
        # Tolerate fenced markdown even though the prompt forbids it.
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            return {}

        results = payload.get("results") if isinstance(payload, dict) else payload
        if not isinstance(results, list):
            return {}

        out: dict[int, dict[str, Any]] = {}
        for entry in results:
            if not isinstance(entry, dict):
                continue
            i = entry.get("i")
            if isinstance(i, int):
                out[i] = entry
        return out

    # ------------------------------------------------------------------
    # Budget bookkeeping
    # ------------------------------------------------------------------

    def _charge_budget(
        self,
        namespace_id: UUID | None,
        input_tokens: int,
        output_tokens: int,
    ) -> bool:
        """Charge the namespace bucket. Returns False if budget exceeded."""
        cap = self._config.llm_max_tokens_per_namespace_per_hour
        if cap <= 0:
            return True

        now = time.monotonic()
        bucket = self._budgets.get(namespace_id)
        if bucket is None or (now - bucket.window_started_at) >= _BUDGET_WINDOW_SECONDS:
            bucket = _NamespaceBudget(window_started_at=now)
            self._budgets[namespace_id] = bucket

        projected = bucket.tokens_used + input_tokens + output_tokens
        if projected > cap:
            if not bucket.warned_in_window:
                logger.warning(
                    "Level 2 hook LLM budget exceeded for namespace {} "
                    "({} + {} would exceed cap {}). Failing open until window resets.",
                    namespace_id,
                    bucket.tokens_used,
                    input_tokens + output_tokens,
                    cap,
                )
                bucket.warned_in_window = True
            return False

        bucket.tokens_used = projected
        return True

    def _estimate_input_tokens(
        self,
        filt: SemanticFilter,
        items: list[_PendingEvaluation],
    ) -> int:
        """Rough char/4 estimate used for the pre-call budget check."""
        n = len(_SYSTEM_PROMPT) // 4
        n += len(filt.description) // 4
        for ex in filt.examples[:5]:
            n += len(ex) // 4
        for ex in filt.anti_examples[:5]:
            n += len(ex) // 4
        for p in items:
            data = p.event.data
            n += min(100, len(str(data.get("name", "")))) // 4
            n += min(200, len(str(data.get("description", "")))) // 4
        return max(1, n)
