"""LLM-based semantic filter evaluator (Level 2).

Uses a nano/mini LLM model for yes/no classification when embedding
similarity alone is insufficient. Batches multiple entity-filter pairs
into a single prompt to amortize LLM overhead.

Model resolution order:
1. ``SemanticFilter.filter_model`` (per-filter override)
2. ``SemanticHooksConfig.filter_model`` (global config / env var)
3. ``"gpt-4.1-nano"`` (hardcoded fallback)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import litellm
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .models import SemanticFilter, SemanticHooksConfig

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a precise binary classifier. Given a semantic filter description \
and one or more entities, determine whether each entity matches the filter.

Return a JSON object mapping each entity index to true/false:
{"0": true, "1": false, "2": true}

Rules:
- true = the entity clearly matches the filter description
- false = the entity does not match or is ambiguous
- Consider the entity name, type, and description
- Be conservative: only return true for clear matches"""

_USER_PROMPT = """\
Filter: {filter_description}
{examples_section}
Entities to evaluate:
{entities_section}

Return JSON mapping index to match (true/false):"""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class LLMFilterResult:
    """Result of an LLM filter evaluation for a single entity."""

    entity_index: int
    matches: bool
    confidence: float = 0.0  # From logprobs if available
    model_used: str = ""


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class LLMFilterEvaluator:
    """Evaluates entity-filter pairs using a configurable LLM model.

    Supports batching: multiple entities checked against a single filter
    in one LLM call. Model is configurable at three levels:
    per-filter > global config > default.

    Usage::

        evaluator = LLMFilterEvaluator(config)
        results = await evaluator.evaluate_batch(
            filter=my_filter,
            entities=[
                {"name": "Acme Corp", "entity_type": "ORGANIZATION", "description": "..."},
                {"name": "Alice", "entity_type": "PERSON", "description": "..."},
            ],
        )
        # results: [LLMFilterResult(entity_index=0, matches=True), ...]
    """

    def __init__(self, config: SemanticHooksConfig | None = None) -> None:
        self._config = config or SemanticHooksConfig()
        self._default_model = self._config.filter_model

    def _resolve_model(self, filter: SemanticFilter) -> str:
        """Resolve which LLM model to use for a filter.

        Priority: per-filter override > config > default.
        """
        if filter.filter_model:
            return filter.filter_model
        return self._default_model

    async def evaluate_batch(
        self,
        filter: SemanticFilter,
        entities: list[dict[str, Any]],
    ) -> list[LLMFilterResult]:
        """Evaluate multiple entities against a single filter in one LLM call.

        Args:
            filter: The semantic filter to evaluate against.
            entities: List of entity dicts with keys: name, entity_type, description.

        Returns:
            List of LLMFilterResult, one per entity.
        """
        if not entities:
            return []

        model = self._resolve_model(filter)

        # Build prompt
        examples_section = ""
        if filter.examples:
            examples = "\n".join(f"  + {ex}" for ex in filter.examples[:5])
            examples_section = f"\nExamples of matching content:\n{examples}\n"
        if filter.anti_examples:
            anti = "\n".join(f"  - {ex}" for ex in filter.anti_examples[:3])
            examples_section += f"\nExamples of non-matching content:\n{anti}\n"

        entities_section = "\n".join(
            f"  [{i}] {e.get('entity_type', 'UNKNOWN')}: {e.get('name', '?')} — {e.get('description', '')}"
            for i, e in enumerate(entities)
        )

        user_prompt = _USER_PROMPT.format(
            filter_description=filter.description,
            examples_section=examples_section,
            entities_section=entities_section,
        )

        # Call LLM
        try:
            response_data = await self._call_llm(model, user_prompt)
        except Exception:
            logger.warning("LLM filter evaluation failed for filter '{}', defaulting to no-match", filter.name)
            return [LLMFilterResult(entity_index=i, matches=False, model_used=model) for i in range(len(entities))]

        # Parse response
        results = []
        for i in range(len(entities)):
            key = str(i)
            matches = response_data.get(key, False)
            if isinstance(matches, str):
                matches = matches.lower() in ("true", "yes", "1")
            results.append(
                LLMFilterResult(
                    entity_index=i,
                    matches=bool(matches),
                    confidence=1.0 if matches else 0.0,
                    model_used=model,
                )
            )

        return results

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
        retry=retry_if_exception_type((litellm.RateLimitError, litellm.ServiceUnavailableError)),
    )
    async def _call_llm(self, model: str, user_prompt: str) -> dict[str, Any]:
        """Call the LLM and parse the JSON response."""
        response = await litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=200,
        )

        content = response.choices[0].message.content or "{}"
        return self._parse_json(content)

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling common formatting issues."""
        # Try direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding JSON object in content
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                pass

        logger.warning("Could not parse LLM filter response: {}", content[:200])
        return {}
