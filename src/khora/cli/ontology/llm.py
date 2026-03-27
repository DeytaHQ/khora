"""LLM wrapper for ontology construction with token/cost tracking."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import litellm
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class OntologyLLMError(Exception):
    """Base exception for ontology LLM operations."""


class BudgetExhaustedError(OntologyLLMError):
    """Raised when token/cost budget is exhausted."""

    def __init__(self, used: float, budget: float) -> None:
        self.used = used
        self.budget = budget
        super().__init__(f"Budget exhausted: ${used:.4f} used of ${budget:.2f} budget")


class LLMResponseError(OntologyLLMError):
    """Raised when LLM response cannot be parsed as JSON."""

    def __init__(self, raw_content: str) -> None:
        self.raw_content = raw_content
        super().__init__(f"Failed to parse LLM response as JSON ({len(raw_content)} chars)")


@dataclass
class LLMUsage:
    """Track per-call usage."""

    input_tokens: int
    output_tokens: int
    model: str
    cost_usd: float


@dataclass
class OntologyLLM:
    """Thin LiteLLM wrapper with token/cost tracking and budget enforcement."""

    model: str = "gpt-4o"
    budget_usd: float = 1.0
    interactive: bool = True
    _total_input_tokens: int = field(default=0, init=False, repr=False)
    _total_output_tokens: int = field(default=0, init=False, repr=False)
    _total_cost_usd: float = field(default=0.0, init=False, repr=False)
    _call_count: int = field(default=0, init=False, repr=False)

    async def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.3,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call LLM and return parsed JSON response."""
        # Budget check: rough estimate of input cost
        estimated_tokens = len(system + user) // 4
        estimated_cost = estimated_tokens * 0.00001  # conservative estimate
        if self._total_cost_usd + estimated_cost > self.budget_usd:
            raise BudgetExhaustedError(self._total_cost_usd, self.budget_usd)

        response = await self._call_llm(system, user, temperature=temperature, response_format=response_format)

        # Extract usage
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        try:
            cost = litellm.completion_cost(completion_response=response) or 0.0
        except Exception:
            cost = 0.0

        # Update totals
        self._call_count += 1
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._total_cost_usd += cost

        logger.debug(f"LLM call #{self._call_count}: " f"{input_tokens}in/{output_tokens}out, ${cost:.4f}")

        content = response.choices[0].message.content or ""
        return self._parse_json(content)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(
            (
                litellm.RateLimitError,
                litellm.ServiceUnavailableError,
                litellm.APIConnectionError,
            )
        ),
    )
    async def _call_llm(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.3,
        response_format: dict[str, Any] | None = None,
    ) -> Any:
        """Call litellm with retry logic for transient errors."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        return await litellm.acompletion(**kwargs)

    def _parse_json(self, content: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling common LLM quirks.

        Tries multiple strategies:
        1. Direct parse
        2. Extract from markdown code blocks (```json ... ```)
        3. Find outermost { ... } in the response
        4. Strip trailing commas (common LLM mistake)
        5. Wrap bare arrays in a dict
        """
        # 1. Direct parse
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                return {"items": parsed}
        except json.JSONDecodeError:
            pass

        # 2. Extract from markdown code blocks
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", content, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                if isinstance(parsed, dict):
                    return parsed
                if isinstance(parsed, list):
                    return {"items": parsed}
            except json.JSONDecodeError:
                # Try with trailing comma fix
                cleaned = self._fix_trailing_commas(match.group(1))
                try:
                    parsed = json.loads(cleaned)
                    return parsed if isinstance(parsed, dict) else {"items": parsed}
                except json.JSONDecodeError:
                    pass

        # 3. Find outermost { ... }
        stripped = content.strip()
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = stripped[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                # Try with trailing comma fix
                cleaned = self._fix_trailing_commas(candidate)
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    pass

        # 4. Find outermost [ ... ] (bare array)
        start = stripped.find("[")
        end = stripped.rfind("]")
        if start != -1 and end != -1 and end > start:
            candidate = stripped[start : end + 1]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list):
                    return {"items": parsed}
            except json.JSONDecodeError:
                pass

        raise LLMResponseError(content)

    @staticmethod
    def _fix_trailing_commas(text: str) -> str:
        """Remove trailing commas before } or ] (common LLM JSON error)."""
        return re.sub(r",\s*([}\]])", r"\1", text)

    @property
    def usage_summary(self) -> dict[str, Any]:
        """Return cumulative usage statistics."""
        return {
            "calls": self._call_count,
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens,
            "cost_usd": self._total_cost_usd,
            "budget_remaining_usd": self.budget_usd - self._total_cost_usd,
        }

    @property
    def budget_remaining(self) -> float:
        """Return remaining budget in USD."""
        return self.budget_usd - self._total_cost_usd
