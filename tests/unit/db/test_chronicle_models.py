"""Tests for Chronicle #1 schema (chronicle_events + memory_facts).

Covers:

- Migration up/down on a fresh SQLite database (Postgres path is exercised
  by the existing integration test suite under ``tests/integration``).
- CRUD via raw SQL: insert, read, supersede.
- ExpertiseConfig round-trip with new ``events`` / ``facts`` fields.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import create_async_engine

from khora.extraction.skills.base import (
    EventExtractionConfig,
    ExpertiseConfig,
    FactExtractionConfig,
)


def _make_config(url: str) -> Config:
    cfg = Config()
    migrations_dir = Path(__file__).resolve().parents[3] / "src" / "khora" / "db" / "migrations"
    cfg.set_main_option("script_location", str(migrations_dir))
    cfg.set_main_option("sqlalchemy.url", url)
    cfg.attributes["database_url"] = url
    return cfg


@pytest.fixture
def sqlite_url(tmp_path: Path) -> str:
    return f"sqlite+aiosqlite:///{tmp_path / 'chronicle.db'}"


@pytest.mark.unit
class TestChronicleMigration:
    def test_upgrade_creates_tables(self, sqlite_url: str) -> None:
        cfg = _make_config(sqlite_url)
        command.upgrade(cfg, "head")

        async def check() -> None:
            engine = create_async_engine(sqlite_url)
            try:
                async with engine.connect() as conn:
                    rows = await conn.execute(sa.text("SELECT name FROM sqlite_master WHERE type='table'"))
                    tables = {r[0] for r in rows}
                    assert "chronicle_events" in tables
                    assert "memory_facts" in tables

                    rows = await conn.execute(sa.text("PRAGMA table_info(chronicle_events)"))
                    event_cols = {r[1] for r in rows}
                    expected = {
                        "id",
                        "namespace_id",
                        "chunk_id",
                        "subject",
                        "verb",
                        "object",
                        "observation_date",
                        "referenced_date",
                        "relative_offset",
                        "confidence",
                        "source_text",
                        "created_at",
                    }
                    assert expected <= event_cols
                    # SQLite path: no embedding column (LanceDB owns vectors)
                    assert "embedding" not in event_cols

                    rows = await conn.execute(sa.text("PRAGMA table_info(memory_facts)"))
                    fact_cols = {r[1] for r in rows}
                    expected_facts = {
                        "id",
                        "namespace_id",
                        "subject",
                        "predicate",
                        "object",
                        "fact_text",
                        "confidence",
                        "is_active",
                        "superseded_by",
                        "source_chunk_ids",
                        "created_at",
                        "updated_at",
                    }
                    assert expected_facts <= fact_cols
            finally:
                await engine.dispose()

        asyncio.run(check())

    def test_downgrade_drops_tables(self, sqlite_url: str) -> None:
        cfg = _make_config(sqlite_url)
        command.upgrade(cfg, "head")
        command.downgrade(cfg, "023_add_document_relationship_count")

        async def check() -> None:
            engine = create_async_engine(sqlite_url)
            try:
                async with engine.connect() as conn:
                    rows = await conn.execute(sa.text("SELECT name FROM sqlite_master WHERE type='table'"))
                    tables = {r[0] for r in rows}
                    assert "chronicle_events" not in tables
                    assert "memory_facts" not in tables
            finally:
                await engine.dispose()

        asyncio.run(check())


@pytest.mark.unit
class TestChronicleCrud:
    def test_event_insert_and_read(self, sqlite_url: str) -> None:
        cfg = _make_config(sqlite_url)
        command.upgrade(cfg, "head")

        ns_id = str(uuid4())
        doc_id = str(uuid4())
        chunk_id = str(uuid4())
        event_id = str(uuid4())

        async def run() -> None:
            engine = create_async_engine(sqlite_url)
            try:
                async with engine.begin() as conn:
                    await conn.execute(
                        sa.text(
                            "INSERT INTO memory_namespaces (id, namespace_id, tenancy_mode, version, is_active) "
                            "VALUES (:id, :ns, 'shared', 1, 1)"
                        ),
                        {"id": ns_id, "ns": ns_id},
                    )
                    await conn.execute(
                        sa.text(
                            "INSERT INTO documents (id, namespace_id, content, status) "
                            "VALUES (:id, :ns, 'doc body', 'completed')"
                        ),
                        {"id": doc_id, "ns": ns_id},
                    )
                    await conn.execute(
                        sa.text(
                            "INSERT INTO chunks (id, namespace_id, document_id, content) "
                            "VALUES (:id, :ns, :doc, 'chunk body')"
                        ),
                        {"id": chunk_id, "ns": ns_id, "doc": doc_id},
                    )
                    await conn.execute(
                        sa.text(
                            "INSERT INTO chronicle_events "
                            "(id, namespace_id, chunk_id, subject, verb, object, observation_date, confidence, source_text) "
                            "VALUES (:id, :ns, :chunk, 'Alice', 'visited', 'Berlin', :obs, 0.9, 'Alice visited Berlin.')"
                        ),
                        {
                            "id": event_id,
                            "ns": ns_id,
                            "chunk": chunk_id,
                            "obs": "2026-04-16T00:00:00Z",
                        },
                    )

                async with engine.connect() as conn:
                    rows = await conn.execute(
                        sa.text("SELECT subject, verb, object, confidence FROM chronicle_events WHERE id=:id"),
                        {"id": event_id},
                    )
                    row = rows.one()
                    assert row[0] == "Alice"
                    assert row[1] == "visited"
                    assert row[2] == "Berlin"
                    assert float(row[3]) == pytest.approx(0.9)
            finally:
                await engine.dispose()

        asyncio.run(run())

    def test_fact_insert_and_supersede(self, sqlite_url: str) -> None:
        cfg = _make_config(sqlite_url)
        command.upgrade(cfg, "head")

        ns_id = str(uuid4())
        f1 = str(uuid4())
        f2 = str(uuid4())

        async def run() -> None:
            engine = create_async_engine(sqlite_url)
            try:
                async with engine.begin() as conn:
                    await conn.execute(
                        sa.text(
                            "INSERT INTO memory_namespaces (id, namespace_id, tenancy_mode, version, is_active) "
                            "VALUES (:id, :ns, 'shared', 1, 1)"
                        ),
                        {"id": ns_id, "ns": ns_id},
                    )
                    for fact_id, fact_text in [(f1, "Alice lives in Paris"), (f2, "Alice lives in Berlin")]:
                        await conn.execute(
                            sa.text(
                                "INSERT INTO memory_facts "
                                "(id, namespace_id, subject, predicate, object, fact_text, confidence, is_active, source_chunk_ids) "
                                "VALUES (:id, :ns, 'Alice', 'lives_in', 'city', :ft, 0.9, 1, '[]')"
                            ),
                            {"id": fact_id, "ns": ns_id, "ft": fact_text},
                        )

                    # Supersede f1 with f2
                    await conn.execute(
                        sa.text("UPDATE memory_facts SET is_active=0, superseded_by=:by WHERE id=:id"),
                        {"by": f2, "id": f1},
                    )

                async with engine.connect() as conn:
                    rows = await conn.execute(
                        sa.text(
                            "SELECT id, is_active, superseded_by FROM memory_facts WHERE namespace_id=:ns ORDER BY id"
                        ),
                        {"ns": ns_id},
                    )
                    rows_list = sorted(rows.all(), key=lambda r: r[0])
                    by_id = {r[0]: r for r in rows_list}
                    assert int(by_id[f1][1]) == 0
                    assert by_id[f1][2] == f2
                    assert int(by_id[f2][1]) == 1

                    # Active facts query (the supported access pattern)
                    rows = await conn.execute(
                        sa.text(
                            "SELECT id FROM memory_facts WHERE namespace_id=:ns AND subject='Alice' AND is_active=1"
                        ),
                        {"ns": ns_id},
                    )
                    active_ids = [r[0] for r in rows]
                    assert active_ids == [f2]
            finally:
                await engine.dispose()

        asyncio.run(run())


@pytest.mark.unit
class TestExpertiseConfigRoundTrip:
    def test_default_events_and_facts_present(self) -> None:
        cfg = ExpertiseConfig(name="default")
        assert isinstance(cfg.events, EventExtractionConfig)
        assert isinstance(cfg.facts, FactExtractionConfig)
        assert cfg.events.enabled is True
        assert cfg.facts.enabled is True
        assert cfg.facts.reconcile is True

    def test_to_dict_includes_new_fields(self) -> None:
        cfg = ExpertiseConfig(name="default")
        out = cfg.to_dict()
        assert out["events"] == {"enabled": True, "model": "gpt-4o-mini"}
        assert out["facts"] == {"enabled": True, "model": "gpt-4o-mini", "reconcile": True}

    def test_round_trip_overrides(self) -> None:
        cfg = ExpertiseConfig(
            name="custom",
            events=EventExtractionConfig(enabled=False, model="gpt-4o"),
            facts=FactExtractionConfig(enabled=True, model="gpt-4o-mini", reconcile=False),
        )
        restored = ExpertiseConfig.from_dict(cfg.to_dict())
        assert restored.events.enabled is False
        assert restored.events.model == "gpt-4o"
        assert restored.facts.enabled is True
        assert restored.facts.reconcile is False

    def test_legacy_payload_without_chronicle_fields(self) -> None:
        # ExpertiseConfig payloads created before Chronicle #1 must continue to
        # round-trip — the new fields must default to enabled=True.
        legacy = {"name": "legacy", "version": "1.0.0"}
        cfg = ExpertiseConfig.from_dict(legacy)
        assert cfg.events.enabled is True
        assert cfg.facts.enabled is True
