"""Two-LLM judge for borderline dedupe merges (#667, Phase 4.1).

The vectorcypher cross-batch dedupe planner emits one candidate merge per
pair above the per-type cosine threshold (default 0.90 / 0.85 / etc).
Pairs in the **borderline band** ``[verifier_band_low, verifier_band_high)``
(default ``[0.78, 0.95)``) are not safe to apply blindly — they're the
zone where surface form / typos / aliases push pairs into the candidate
pool but the embedding doesn't fully commit.

This module gates those pairs through two independent LLM calls:

  * **verifier** — picks the apply/skip/defer decision (default ``gpt-4o-mini``)
  * **auditor** — independent second opinion (default ``claude-haiku-4.5``)

Both must vote ``merge`` (with confidence above the configured floor)
before the apply handler proceeds. Disagreement → ``decision="defer"``,
which records the verification result on the undo record and **does not
apply** the merge — operator review territory.

Schema validation is strict: a malformed LLM response is treated as
``decision="defer"`` rather than as a parse error that would stop the
run.

Stability: **internal**. The Pydantic schema and the dispatch helper
shape may evolve as we tune the verifier on real merge traffic.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, ValidationError

from khora.telemetry import bounded_text_hash, trace_span

if TYPE_CHECKING:
    from khora.dream.config import DreamConfig


VerifierDecision = Literal["merge", "keep_separate", "defer"]


class VerifierVerdict(BaseModel):
    """Structured Pydantic output for a single LLM judge call.

    The verifier and the auditor each return one ``VerifierVerdict``.
    The dispatcher (:func:`run_two_llm_judge`) combines both before
    returning the joint :class:`JudgeResult`.
    """

    decision: VerifierDecision = Field(
        description="One of merge / keep_separate / defer.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Self-reported confidence in the decision.",
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="Optional list of evidence identifiers (chunk / doc UUIDs).",
    )
    rationale: str = Field(
        default="",
        description="Short free-text rationale (hashed before becoming a span attribute).",
    )


@dataclass(slots=True, frozen=True)
class JudgeResult:
    """Joint verdict from the two-LLM judge.

    ``decision`` is the dispatcher's combined ruling:
      * ``"merge"`` only when both judges return ``merge``.
      * ``"keep_separate"`` only when both judges return ``keep_separate``.
      * ``"defer"`` for any disagreement, any below-threshold confidence,
        or any parse / network failure.
    """

    decision: VerifierDecision
    verifier_verdict: VerifierVerdict | None
    auditor_verdict: VerifierVerdict | None
    rationale: str


@dataclass(slots=True, frozen=True)
class CandidatePair:
    """Inputs the judges need to decide a single borderline pair."""

    canonical_id: str
    canonical_name: str
    canonical_entity_type: str
    absorbed_id: str
    absorbed_name: str
    absorbed_entity_type: str
    similarity_score: float


_SYSTEM_PROMPT = (
    "You are an entity-resolution judge. Two candidate entity records are "
    "presented along with their cosine similarity. Decide whether they are "
    "the same real-world entity (merge), distinct (keep_separate), or "
    "uncertain (defer). Respond with a single JSON object matching the "
    'schema {"decision": "merge|keep_separate|defer", "confidence": '
    'float in [0,1], "evidence_ids": [string], "rationale": string}. '
    "Output JSON only — no prose, no markdown fences."
)


def _build_user_prompt(pair: CandidatePair) -> str:
    return (
        f"Pair under review (cosine similarity = {pair.similarity_score:.4f}):\n"
        f"\n"
        f"A (canonical):\n"
        f"  id: {pair.canonical_id}\n"
        f"  name: {pair.canonical_name}\n"
        f"  type: {pair.canonical_entity_type}\n"
        f"\n"
        f"B (absorbed candidate):\n"
        f"  id: {pair.absorbed_id}\n"
        f"  name: {pair.absorbed_name}\n"
        f"  type: {pair.absorbed_entity_type}\n"
        f"\n"
        f"Return the JSON verdict now."
    )


_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _parse_verdict(raw: str) -> VerifierVerdict | None:
    """Parse an LLM response into a :class:`VerifierVerdict` or ``None``.

    A defensive parser — strips leading/trailing whitespace, peels off a
    fenced ```json``` block if the model emitted one, then runs Pydantic
    validation. Returns ``None`` on any failure; the dispatcher treats a
    ``None`` as ``decision="defer"``.
    """
    if not raw:
        return None
    text = raw.strip()
    match = _FENCE_RE.match(text)
    if match is not None:
        text = match.group(1).strip()
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    try:
        return VerifierVerdict.model_validate(payload)
    except ValidationError:
        return None


async def _ask_one_judge(
    pair: CandidatePair,
    *,
    model: str,
    direction: str,
    timeout_s: float,
) -> VerifierVerdict | None:
    """Call one judge model and parse its verdict.

    ``direction`` is ``"verifier"`` or ``"auditor"`` — recorded as a
    bounded-cardinality span attribute (never a metric label) so dashboards
    can split verifier vs auditor latency.
    """
    from khora.config.llm import LiteLLMConfig, acompletion

    config = LiteLLMConfig(
        model=model,
        temperature=0.0,
        max_tokens=400,
        timeout=int(max(1.0, timeout_s)),
    )
    user_prompt = _build_user_prompt(pair)

    with trace_span(
        "khora.dream.vectorcypher.dedupe.verifier",
        direction=direction,
        model=model,
        similarity_score=str(pair.similarity_score),
        prompt_hash=bounded_text_hash(user_prompt),
    ) as span:
        try:
            raw = await asyncio.wait_for(
                acompletion(
                    user_prompt,
                    config,
                    system_prompt=_SYSTEM_PROMPT,
                    _telemetry_op=f"dream.dedupe.{direction}",
                ),
                timeout=timeout_s,
            )
        except TimeoutError as exc:
            span.set_attribute("outcome", "timeout")
            span.set_attribute("error_type", type(exc).__name__)
            return None
        except Exception as exc:  # noqa: BLE001
            # LLM transport errors degrade to defer at the dispatcher
            # level — never crash the dream run on a flaky API.
            span.set_attribute("outcome", "error")
            span.set_attribute("error_type", type(exc).__name__)
            return None

        verdict = _parse_verdict(raw)
        if verdict is None:
            span.set_attribute("outcome", "parse_error")
            return None
        span.set_attribute("outcome", "parsed")
        span.set_attribute("decision", verdict.decision)
        span.set_attribute("confidence", str(verdict.confidence))
        return verdict


async def run_two_llm_judge(
    pair: CandidatePair,
    *,
    config: DreamConfig | None = None,
    min_confidence: float = 0.6,
) -> JudgeResult:
    """Run the two-LLM judge for one borderline pair.

    Both judges are queried concurrently. The dispatcher combines their
    verdicts:

      * Both ``merge`` with confidence >= ``min_confidence`` → ``merge``.
      * Both ``keep_separate`` → ``keep_separate``.
      * Anything else (disagreement, low confidence, parse failure,
        timeout, network error) → ``defer``.

    Args:
        pair: The candidate pair under review.
        config: :class:`DreamConfig` providing the verifier / auditor
            model names. ``None`` falls back to ``DreamConfig()`` defaults
            (gpt-4o-mini + claude-haiku-4.5).
        min_confidence: Both judges must report confidence at or above
            this floor in their ``merge`` verdict before the dispatcher
            returns ``merge``. Default 0.6.

    Returns:
        :class:`JudgeResult` carrying the dispatcher decision and both
        raw verdicts (or ``None`` per side on failure).
    """
    from khora.dream.config import DreamConfig as _DreamConfig

    cfg = config if config is not None else _DreamConfig()
    verifier_model = cfg.dedupe_verifier_model
    auditor_model = cfg.dedupe_auditor_model
    timeout_s = float(cfg.dedupe_verifier_timeout_seconds)

    verifier_task = _ask_one_judge(
        pair,
        model=verifier_model,
        direction="verifier",
        timeout_s=timeout_s,
    )
    auditor_task = _ask_one_judge(
        pair,
        model=auditor_model,
        direction="auditor",
        timeout_s=timeout_s,
    )
    verifier_verdict, auditor_verdict = await asyncio.gather(verifier_task, auditor_task)

    decision, rationale = _combine_verdicts(verifier_verdict, auditor_verdict, min_confidence)
    return JudgeResult(
        decision=decision,
        verifier_verdict=verifier_verdict,
        auditor_verdict=auditor_verdict,
        rationale=rationale,
    )


def _combine_verdicts(
    verifier: VerifierVerdict | None,
    auditor: VerifierVerdict | None,
    min_confidence: float,
) -> tuple[VerifierDecision, str]:
    """Dispatcher rule: agree to merge or defer."""
    if verifier is None or auditor is None:
        return "defer", "one or both judges returned no parseable verdict"
    if verifier.decision == "merge" and auditor.decision == "merge":
        if verifier.confidence >= min_confidence and auditor.confidence >= min_confidence:
            return "merge", "both judges agreed: merge"
        return "defer", "both judges said merge but confidence below floor"
    if verifier.decision == "keep_separate" and auditor.decision == "keep_separate":
        return "keep_separate", "both judges agreed: keep_separate"
    return "defer", f"judges disagree (verifier={verifier.decision}, auditor={auditor.decision})"


__all__ = [
    "CandidatePair",
    "JudgeResult",
    "VerifierDecision",
    "VerifierVerdict",
    "run_two_llm_judge",
]
