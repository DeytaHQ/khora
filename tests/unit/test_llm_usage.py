"""Unit tests for LLMUsage dataclass and request-scoped usage accumulator."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from khora.khora import BatchResult, LLMUsage, RecallResult, RememberResult

# ---------------------------------------------------------------------------
# LLMUsage dataclass
# ---------------------------------------------------------------------------


class TestLLMUsage:
    """Tests for the LLMUsage frozen dataclass."""

    def test_fields(self) -> None:
        u = LLMUsage(
            operation="entity_extraction",
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
            latency_ms=812.3,
        )
        assert u.operation == "entity_extraction"
        assert u.model == "gpt-4o"
        assert u.prompt_tokens == 100
        assert u.completion_tokens == 200
        assert u.total_tokens == 300
        assert u.latency_ms == 812.3
        assert u.batch_size == 1  # default

    def test_batch_size(self) -> None:
        u = LLMUsage(
            operation="embedding",
            model="text-embedding-3-small",
            prompt_tokens=50,
            completion_tokens=0,
            total_tokens=50,
            latency_ms=100.0,
            batch_size=10,
        )
        assert u.batch_size == 10

    def test_frozen(self) -> None:
        u = LLMUsage(
            operation="embedding",
            model="m",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            latency_ms=0.0,
        )
        with pytest.raises(AttributeError):
            u.operation = "other"  # type: ignore[misc]

    def test_importable_from_public_api(self) -> None:
        from khora import LLMUsage as PublicLLMUsage

        assert PublicLLMUsage is LLMUsage


# ---------------------------------------------------------------------------
# Result types include llm_usage field
# ---------------------------------------------------------------------------


class TestResultLLMUsageField:
    """Verify llm_usage field on all result types defaults to empty list."""

    def test_remember_result_default(self) -> None:
        r = RememberResult(
            document_id=uuid4(),
            namespace_id=uuid4(),
            chunks_created=1,
            entities_extracted=1,
            relationships_created=0,
        )
        assert r.llm_usage == []

    def test_batch_result_default(self) -> None:
        r = BatchResult(
            total=5,
            processed=5,
            skipped=0,
            failed=0,
            chunks=10,
            entities=3,
            relationships=2,
        )
        assert r.llm_usage == []

    def test_recall_result_default(self) -> None:
        r = RecallResult(
            query="test",
            namespace_id=uuid4(),
            documents=[],
            chunks=[],
            entities=[],
            relationships=[],
        )
        assert r.llm_usage == []

    def test_remember_result_with_usage(self) -> None:
        usage = [
            LLMUsage(
                operation="entity_extraction",
                model="gpt-4o",
                prompt_tokens=100,
                completion_tokens=200,
                total_tokens=300,
                latency_ms=500.0,
            )
        ]
        r = RememberResult(
            document_id=uuid4(),
            namespace_id=uuid4(),
            chunks_created=1,
            entities_extracted=1,
            relationships_created=0,
            llm_usage=usage,
        )
        assert len(r.llm_usage) == 1
        assert r.llm_usage[0].operation == "entity_extraction"


# ---------------------------------------------------------------------------
# Usage accumulator
# ---------------------------------------------------------------------------


class TestUsageAccumulator:
    """Tests for the request-scoped usage accumulator."""

    def test_collect_without_start_returns_empty(self) -> None:
        from khora.telemetry.context import collect_usage

        assert collect_usage() == []

    def test_start_record_collect(self) -> None:
        from khora.telemetry.context import (
            collect_usage,
            record_usage,
            start_usage_collection,
        )

        start_usage_collection()
        u = LLMUsage(
            operation="test",
            model="m",
            prompt_tokens=1,
            completion_tokens=2,
            total_tokens=3,
            latency_ms=10.0,
        )
        record_usage(u)
        result = collect_usage()
        assert len(result) == 1
        assert result[0] is u

    def test_collect_resets_contextvar(self) -> None:
        from khora.telemetry.context import (
            collect_usage,
            record_usage,
            start_usage_collection,
        )

        start_usage_collection()
        record_usage(
            LLMUsage(
                operation="a",
                model="m",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_ms=0.0,
            )
        )
        collect_usage()
        # Second collect should return empty (contextvar reset)
        assert collect_usage() == []

    def test_record_without_start_is_noop(self) -> None:
        from khora.telemetry.context import collect_usage, record_usage

        # Should not raise
        record_usage(
            LLMUsage(
                operation="a",
                model="m",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_ms=0.0,
            )
        )
        assert collect_usage() == []

    @pytest.mark.asyncio
    async def test_concurrent_gather_no_loss(self) -> None:
        """Concurrent asyncio.gather() tasks all accumulate into the same queue."""
        from khora.telemetry.context import (
            collect_usage,
            record_usage,
            start_usage_collection,
        )

        start_usage_collection()

        async def record_n(n: int) -> None:
            for i in range(n):
                record_usage(
                    LLMUsage(
                        operation=f"op_{i}",
                        model="m",
                        prompt_tokens=i,
                        completion_tokens=0,
                        total_tokens=i,
                        latency_ms=float(i),
                    )
                )
                # Yield control to simulate real async work
                await asyncio.sleep(0)

        # 10 concurrent tasks, each recording 5 entries = 50 total
        await asyncio.gather(*(record_n(5) for _ in range(10)))

        result = collect_usage()
        assert len(result) == 50
