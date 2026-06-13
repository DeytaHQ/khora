"""VectorCypher ``remember_batch`` carries ``source_timestamp`` onto chunks.

The single-``remember()`` path resolves the chunk event-time via
kwarg -> metadata["occurred_at"] -> source_timestamp -> now(UTC)
(``engines/vectorcypher/engine.py`` remember()). The streaming batch path
previously read only ``metadata["occurred_at"]`` and fell back to now(UTC),
so ``kb.remember_batch(items, source_timestamp=t)`` silently dropped ``t``
and chunks landed at ingest time (#992).

These tests pin the parity end-to-end over the zero-infra sqlite_lance
backend: a doc batched with a ``source_timestamp`` distinct from now must
land with ``Chunk.source_timestamp == Chunk.occurred_at == t``, not now().
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from examples._helpers import embedded_khora, install_mock_llm


def _aware(dt: datetime) -> datetime:
    """SQLite round-trips datetimes tz-naive; pin them back to UTC for comparison."""
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


@pytest.mark.unit
class TestRememberBatchSourceTimestamp:
    """``remember_batch`` honors the batch-level ``source_timestamp`` kwarg."""

    @pytest.mark.asyncio
    async def test_batch_source_timestamp_lands_on_chunk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A doc batched with source_timestamp=t lands at t, not ingest now()."""
        install_mock_llm(monkeypatch=monkeypatch, dim=1536)
        supplied = datetime.now(UTC) - timedelta(days=30)

        async with embedded_khora(engine="vectorcypher") as kb:
            ns = (await kb.create_namespace()).namespace_id
            await kb.remember_batch(
                [{"content": "We shipped Pipeline v2.0, a major ingest upgrade."}],
                namespace=ns,
                source_timestamp=supplied,
                entity_types=["CONCEPT"],
                relationship_types=["RELATES_TO"],
            )

            docs = await kb.list_documents(namespace=ns, limit=1)
            assert docs, "batch ingest produced no document"
            doc = docs[0]
            chunks = await kb.storage.get_chunks_by_document(doc.id, namespace_id=doc.namespace_id)

        assert chunks, "batch ingest produced no chunk"
        chunk = chunks[0]

        # Document-level source_timestamp is the pass-through baseline.
        assert doc.source_timestamp is not None
        assert abs((_aware(doc.source_timestamp) - supplied).total_seconds()) < 1.0

        # The bug: chunk.occurred_at (= RecallChunk.occurred_at) fell to now().
        assert chunk.occurred_at is not None
        assert abs((_aware(chunk.occurred_at) - supplied).total_seconds()) < 1.0, (
            f"chunk.occurred_at={chunk.occurred_at} should equal source_timestamp {supplied}, not ingest now()"
        )

        # source_timestamp denorm must also carry the supplied value.
        assert chunk.source_timestamp is not None
        assert abs((_aware(chunk.source_timestamp) - supplied).total_seconds()) < 1.0

    @pytest.mark.asyncio
    async def test_per_doc_source_timestamp_overrides_batch_kwarg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A per-doc source_timestamp wins over the batch-level kwarg."""
        install_mock_llm(monkeypatch=monkeypatch, dim=1536)
        batch_ts = datetime.now(UTC) - timedelta(days=30)
        per_doc_ts = datetime.now(UTC) - timedelta(days=90)

        async with embedded_khora(engine="vectorcypher") as kb:
            ns = (await kb.create_namespace()).namespace_id
            await kb.remember_batch(
                [{"content": "Per-doc override content.", "source_timestamp": per_doc_ts}],
                namespace=ns,
                source_timestamp=batch_ts,
                entity_types=["CONCEPT"],
                relationship_types=["RELATES_TO"],
            )

            docs = await kb.list_documents(namespace=ns, limit=1)
            doc = docs[0]
            chunks = await kb.storage.get_chunks_by_document(doc.id, namespace_id=doc.namespace_id)

        chunk = chunks[0]
        assert chunk.occurred_at is not None
        assert abs((_aware(chunk.occurred_at) - per_doc_ts).total_seconds()) < 1.0

    @pytest.mark.asyncio
    async def test_metadata_occurred_at_wins_over_source_timestamp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """metadata['occurred_at'] takes precedence over source_timestamp."""
        install_mock_llm(monkeypatch=monkeypatch, dim=1536)
        occurred = datetime(2026, 1, 15, 10, 0, 0, tzinfo=UTC)
        source_ts = datetime.now(UTC) - timedelta(days=30)

        async with embedded_khora(engine="vectorcypher") as kb:
            ns = (await kb.create_namespace()).namespace_id
            await kb.remember_batch(
                [
                    {
                        "content": "Has both occurred_at and source_timestamp.",
                        "metadata": {"occurred_at": "2026-01-15T10:00:00Z"},
                    }
                ],
                namespace=ns,
                source_timestamp=source_ts,
                entity_types=["CONCEPT"],
                relationship_types=["RELATES_TO"],
            )

            docs = await kb.list_documents(namespace=ns, limit=1)
            doc = docs[0]
            chunks = await kb.storage.get_chunks_by_document(doc.id, namespace_id=doc.namespace_id)

        chunk = chunks[0]
        assert chunk.occurred_at is not None
        assert abs((_aware(chunk.occurred_at) - occurred).total_seconds()) < 1.0

    @pytest.mark.asyncio
    async def test_no_timestamp_falls_back_to_now(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No source_timestamp and no occurred_at -> chunk occurred_at is now()."""
        install_mock_llm(monkeypatch=monkeypatch, dim=1536)
        before = datetime.now(UTC) - timedelta(seconds=5)

        async with embedded_khora(engine="vectorcypher") as kb:
            ns = (await kb.create_namespace()).namespace_id
            await kb.remember_batch(
                [{"content": "No timestamp supplied at all."}],
                namespace=ns,
                entity_types=["CONCEPT"],
                relationship_types=["RELATES_TO"],
            )

            docs = await kb.list_documents(namespace=ns, limit=1)
            doc = docs[0]
            chunks = await kb.storage.get_chunks_by_document(doc.id, namespace_id=doc.namespace_id)

        after = datetime.now(UTC) + timedelta(seconds=5)
        chunk = chunks[0]
        assert chunk.occurred_at is not None
        assert before <= _aware(chunk.occurred_at) <= after
