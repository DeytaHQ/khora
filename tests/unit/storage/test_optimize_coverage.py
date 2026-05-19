"""Coverage tests for khora.storage.optimize.

Exercises the DDL helpers, the per-call decision branches, and the
coordinator-aware ``optimize_storage`` entrypoint with mocked SQLAlchemy /
Neo4j-driver objects. No real database is required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from khora.storage import optimize as opt_mod

# ---------------------------------------------------------------------------
# Mock builders
# ---------------------------------------------------------------------------


class _FakeRow:
    def __init__(self, value: Any) -> None:
        self._value = value

    def scalar(self) -> Any:
        return self._value


class _FakeConn:
    """Mimics a SQLAlchemy AsyncConnection used inside ``async with engine...``.

    Each ``execute`` call returns a row whose ``scalar()`` value pops from
    ``row_values``.  When the queue empties it returns ``None`` (index out of
    range produces ``None`` via the default value).
    """

    def __init__(self, row_values: list[Any] | None = None) -> None:
        self.row_values = list(row_values or [])
        self.execute = AsyncMock(side_effect=self._execute_side_effect)
        self.execution_options = AsyncMock(return_value=None)

    async def _execute_side_effect(self, *_args: Any, **_kwargs: Any) -> _FakeRow:
        if self.row_values:
            return _FakeRow(self.row_values.pop(0))
        # When no scripted value: return a no-op row.  ``scalar()`` will be
        # called on this for SELECT statements; for DDL the result is ignored.
        return _FakeRow(None)


class _AsyncCtx:
    def __init__(self, value: Any) -> None:
        self._value = value

    async def __aenter__(self) -> Any:
        return self._value

    async def __aexit__(self, *args: Any) -> None:
        return None


def _engine_with_connect(*conns: _FakeConn) -> Any:
    """Build a fake async engine.  ``connect()`` returns conns in order."""
    queue = list(conns)

    def _connect() -> _AsyncCtx:
        if not queue:
            raise AssertionError("connect() called more times than scripted")
        return _AsyncCtx(queue.pop(0))

    def _begin() -> _AsyncCtx:
        if not queue:
            raise AssertionError("begin() called more times than scripted")
        return _AsyncCtx(queue.pop(0))

    eng = MagicMock()
    eng.connect = _connect
    eng.begin = _begin
    return eng


# ---------------------------------------------------------------------------
# Module-level DDL tables — sanity checks
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pg_indexes_have_unique_names_and_sql() -> None:
    names = [idx["name"] for idx in opt_mod.PG_INDEXES]
    assert len(names) == len(set(names))
    for idx in opt_mod.PG_INDEXES:
        assert idx["sql"].strip().upper().startswith("CREATE INDEX IF NOT EXISTS")
        assert idx["purpose"]


@pytest.mark.unit
def test_neo4j_indexes_have_unique_names_and_cypher() -> None:
    names = [idx["name"] for idx in opt_mod.NEO4J_INDEXES]
    assert len(names) == len(set(names))
    for idx in opt_mod.NEO4J_INDEXES:
        assert "IF NOT EXISTS" in idx["cypher"]
        assert idx["purpose"]


@pytest.mark.unit
def test_halfvec_indexes_have_format_placeholders() -> None:
    for idx in opt_mod.HALFVEC_INDEXES:
        assert "{dim}" in idx["sql"]
        assert "{m}" in idx["sql"]
        assert "{ef_construction}" in idx["sql"]


@pytest.mark.unit
def test_hnsw_indexes_listed() -> None:
    assert "ix_chunks_embedding_hnsw" in opt_mod.HNSW_INDEXES
    assert "ix_entities_embedding_hnsw" in opt_mod.HNSW_INDEXES


# ---------------------------------------------------------------------------
# drop_hnsw_indexes
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_drop_hnsw_indexes_drops_existing_and_skips_missing() -> None:
    # 2 base HNSW + 2 halfvec = 4 total connect() calls.
    # Script: first exists (1), second missing (None), third exists (1), fourth missing.
    conns = [
        _FakeConn(row_values=[1]),  # ix_chunks_embedding_hnsw exists
        _FakeConn(row_values=[None]),  # ix_entities_embedding_hnsw missing
        _FakeConn(row_values=[1]),  # halfvec chunks exists
        _FakeConn(row_values=[None]),  # halfvec entities missing
    ]
    engine = _engine_with_connect(*conns)

    result = await opt_mod.drop_hnsw_indexes(engine)

    assert result["indexes_dropped"] == 2
    assert result["errors"] == []


@pytest.mark.unit
async def test_drop_hnsw_indexes_records_errors() -> None:
    conn = _FakeConn(row_values=[1])
    conn.execute = AsyncMock(side_effect=RuntimeError("kaboom"))
    engine = _engine_with_connect(
        conn, _FakeConn(row_values=[None]), _FakeConn(row_values=[None]), _FakeConn(row_values=[None])
    )

    result = await opt_mod.drop_hnsw_indexes(engine)

    assert any("kaboom" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# ensure_hnsw_indexes
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_ensure_hnsw_indexes_creates_missing_skips_existing() -> None:
    # Two indexes attempted in parallel: one missing -> create, one already exists.
    conns = [
        _FakeConn(row_values=[None]),  # missing → CREATE
        _FakeConn(row_values=[1]),  # already exists → skip
    ]
    engine = _engine_with_connect(*conns)

    result = await opt_mod.ensure_hnsw_indexes(engine)

    assert result["indexes_created"] == 1
    assert result["freshly_created"] == 1
    assert result["errors"] == []


@pytest.mark.unit
async def test_ensure_hnsw_indexes_records_errors() -> None:
    bad = _FakeConn(row_values=[None])
    bad.execute = AsyncMock(side_effect=RuntimeError("ddl exploded"))
    engine = _engine_with_connect(bad, _FakeConn(row_values=[1]))

    result = await opt_mod.ensure_hnsw_indexes(engine)

    assert result["indexes_created"] == 0
    assert any("ddl exploded" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# reindex_hnsw_concurrently
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_reindex_hnsw_concurrently_skips_missing() -> None:
    # Both indexes scripted as missing -> no REINDEX, no errors, count = 0.
    engine = _engine_with_connect(_FakeConn(row_values=[None]), _FakeConn(row_values=[None]))
    result = await opt_mod.reindex_hnsw_concurrently(engine)
    assert result["indexes_reindexed"] == 0
    assert result["errors"] == []


@pytest.mark.unit
async def test_reindex_hnsw_concurrently_reindexes_existing() -> None:
    engine = _engine_with_connect(_FakeConn(row_values=[1]), _FakeConn(row_values=[1]))
    result = await opt_mod.reindex_hnsw_concurrently(engine)
    assert result["indexes_reindexed"] == 2


@pytest.mark.unit
async def test_reindex_hnsw_concurrently_collects_errors() -> None:
    bad = _FakeConn(row_values=[1])
    bad.execute = AsyncMock(side_effect=RuntimeError("reindex failed"))
    engine = _engine_with_connect(bad, _FakeConn(row_values=[None]))
    result = await opt_mod.reindex_hnsw_concurrently(engine)
    assert any("reindex failed" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# optimize_postgresql — orchestrates PG_INDEXES + ANALYZE + ensure/reindex
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_optimize_postgresql_runs_full_pipeline_with_reindex(monkeypatch: pytest.MonkeyPatch) -> None:
    """All PG_INDEXES + ANALYZE happen in a single begin() block; then
    ensure_hnsw_indexes + (conditionally) reindex_hnsw_concurrently."""
    # 1 begin() for the main DDL+ANALYZE block; ensure/reindex are stubbed.
    begin_conn = _FakeConn()
    engine = _engine_with_connect(begin_conn)

    async def _fake_ensure(_engine: Any) -> dict:
        return {"indexes_created": 0, "freshly_created": 0, "errors": []}

    async def _fake_reindex(_engine: Any) -> dict:
        return {"indexes_reindexed": 2, "errors": []}

    monkeypatch.setattr(opt_mod, "ensure_hnsw_indexes", _fake_ensure)
    monkeypatch.setattr(opt_mod, "reindex_hnsw_concurrently", _fake_reindex)

    result = await opt_mod.optimize_postgresql(engine)

    assert result["indexes_created"] == len(opt_mod.PG_INDEXES)
    assert result["tables_analyzed"] == len(opt_mod.PG_ANALYZE_TABLES)
    assert result["hnsw_reindexed"] == 2
    assert result["errors"] == []


@pytest.mark.unit
async def test_optimize_postgresql_skips_reindex_when_freshly_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine_with_connect(_FakeConn())

    async def _fake_ensure(_engine: Any) -> dict:
        return {"indexes_created": 1, "freshly_created": 1, "errors": []}

    reindex_calls: list[Any] = []

    async def _spy_reindex(_engine: Any) -> dict:
        reindex_calls.append(_engine)
        return {"indexes_reindexed": 0, "errors": []}

    monkeypatch.setattr(opt_mod, "ensure_hnsw_indexes", _fake_ensure)
    monkeypatch.setattr(opt_mod, "reindex_hnsw_concurrently", _spy_reindex)

    result = await opt_mod.optimize_postgresql(engine)

    assert result["hnsw_reindexed"] == 0
    assert reindex_calls == [], "reindex should be skipped when freshly created"
    # ensure_hnsw_indexes contributed +1
    assert result["indexes_created"] == len(opt_mod.PG_INDEXES) + 1


@pytest.mark.unit
async def test_optimize_postgresql_skip_reindex_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine_with_connect(_FakeConn())

    async def _boom(_engine: Any) -> dict:
        raise AssertionError("ensure_hnsw_indexes must not be called when reindex_hnsw=False")

    monkeypatch.setattr(opt_mod, "ensure_hnsw_indexes", _boom)
    monkeypatch.setattr(opt_mod, "reindex_hnsw_concurrently", _boom)

    result = await opt_mod.optimize_postgresql(engine, reindex_hnsw=False)
    assert result["hnsw_reindexed"] == 0


@pytest.mark.unit
async def test_optimize_postgresql_records_index_and_analyze_errors() -> None:
    conn = _FakeConn()

    async def _exec_side(stmt: Any, *args: Any, **kw: Any) -> _FakeRow:
        if "ANALYZE" in str(stmt):
            raise RuntimeError("analyze died")
        if "CREATE INDEX" in str(stmt):
            raise RuntimeError("index died")
        return _FakeRow(None)

    conn.execute = AsyncMock(side_effect=_exec_side)
    engine = _engine_with_connect(conn)

    result = await opt_mod.optimize_postgresql(engine, reindex_hnsw=False)

    assert result["indexes_created"] == 0
    assert result["tables_analyzed"] == 0
    assert any("index died" in e for e in result["errors"])
    assert any("analyze died" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# optimize_neo4j
# ---------------------------------------------------------------------------


def _make_neo4j_driver(*, dedup_count: int | None = 5, raises_on: str | None = None) -> Any:
    """Build a fake AsyncDriver returning scripted dedup + run() responses."""

    class _Result:
        def __init__(self, single_record: dict[str, Any] | None) -> None:
            self._single_record = single_record

        async def single(self) -> dict[str, Any] | None:
            return self._single_record

    class _Session:
        def __init__(self) -> None:
            self.run_calls: list[str] = []

        async def __aenter__(self) -> _Session:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def run(self, cypher: str, *args: Any, **kw: Any) -> _Result:
            self.run_calls.append(cypher)
            if raises_on and raises_on in cypher:
                raise RuntimeError(f"neo4j boom on {raises_on}")
            # First call inside dedup block returns a record
            if "MATCH (e:Entity)" in cypher and "WITH e.namespace_id" in cypher:
                return _Result({"duplicates_removed": dedup_count} if dedup_count is not None else None)
            return _Result(None)

    sessions: list[_Session] = []

    def session(*, database: str = "neo4j") -> _Session:
        s = _Session()
        sessions.append(s)
        return s

    driver = MagicMock()
    driver.session = session
    driver._sessions = sessions
    return driver


@pytest.mark.unit
async def test_optimize_neo4j_happy_path() -> None:
    driver = _make_neo4j_driver(dedup_count=3)
    result = await opt_mod.optimize_neo4j(driver, database="neo4j")
    assert result["duplicates_removed"] == 3
    assert result["indexes_created"] == len(opt_mod.NEO4J_INDEXES)
    assert result["errors"] == []


@pytest.mark.unit
async def test_optimize_neo4j_no_dedup_record() -> None:
    driver = _make_neo4j_driver(dedup_count=None)
    result = await opt_mod.optimize_neo4j(driver)
    assert result["duplicates_removed"] == 0


@pytest.mark.unit
async def test_optimize_neo4j_records_index_errors() -> None:
    # Force the first index create to blow up — the loop should record and continue.
    driver = _make_neo4j_driver(dedup_count=0, raises_on="entity_ns_name_type_unique")
    result = await opt_mod.optimize_neo4j(driver)
    # The dedup cypher also references "entity_ns_name_type_unique"? No — it doesn't.
    # The constraint cypher does, so at least one error is captured.
    assert result["indexes_created"] < len(opt_mod.NEO4J_INDEXES)
    assert any("entity_ns_name_type_unique" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# prepare_for_bulk_load + optimize_storage (coordinator dispatch)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_prepare_for_bulk_load_no_backend() -> None:
    coord = MagicMock()
    coord.vector = None
    coord.relational = None
    result = await opt_mod.prepare_for_bulk_load(coord)
    assert result == {"indexes_dropped": 0, "errors": []}


@pytest.mark.unit
async def test_prepare_for_bulk_load_no_engine_attr() -> None:
    coord = MagicMock()
    coord.vector = MagicMock(spec=[])  # no _engine attr
    coord.relational = None
    result = await opt_mod.prepare_for_bulk_load(coord)
    assert result == {"indexes_dropped": 0, "errors": []}


@pytest.mark.unit
async def test_prepare_for_bulk_load_calls_drop(monkeypatch: pytest.MonkeyPatch) -> None:
    coord = MagicMock()
    fake_engine = object()
    coord.vector = MagicMock(_engine=fake_engine)
    coord.relational = None

    called_with: list[Any] = []

    async def _fake_drop(engine: Any) -> dict:
        called_with.append(engine)
        return {"indexes_dropped": 7, "errors": []}

    monkeypatch.setattr(opt_mod, "drop_hnsw_indexes", _fake_drop)

    result = await opt_mod.prepare_for_bulk_load(coord)
    assert called_with == [fake_engine]
    assert result["indexes_dropped"] == 7


@pytest.mark.unit
async def test_optimize_storage_dispatches_pg_and_neo4j(monkeypatch: pytest.MonkeyPatch) -> None:
    coord = MagicMock()
    pg_engine = object()
    neo_driver = object()
    coord.vector = MagicMock(_engine=pg_engine)
    coord.graph = MagicMock(_driver=neo_driver, _database="neo4j")
    coord.relational = None  # surrealdb branch not exercised

    async def _fake_pg(engine: Any) -> dict:
        assert engine is pg_engine
        return {"indexes_created": 1, "errors": []}

    async def _fake_neo(driver: Any, *, database: str) -> dict:
        assert driver is neo_driver
        assert database == "neo4j"
        return {"indexes_created": 2, "errors": []}

    monkeypatch.setattr(opt_mod, "optimize_postgresql", _fake_pg)
    monkeypatch.setattr(opt_mod, "optimize_neo4j", _fake_neo)

    result = await opt_mod.optimize_storage(coord)
    assert result["postgresql"]["indexes_created"] == 1
    assert result["neo4j"]["indexes_created"] == 2
    assert result["surrealdb"] is None


@pytest.mark.unit
async def test_optimize_storage_skips_when_no_backends_present() -> None:
    coord = MagicMock()
    coord.vector = None
    coord.relational = None
    coord.graph = None
    result = await opt_mod.optimize_storage(coord)
    assert result["postgresql"] is None
    assert result["neo4j"] is None
    assert result["surrealdb"] is None


@pytest.mark.unit
async def test_optimize_storage_skips_pg_when_no_engine_attr(monkeypatch: pytest.MonkeyPatch) -> None:
    coord = MagicMock()
    # Backend present but no _engine attr -> warning path, no call.
    coord.vector = MagicMock(spec=[])
    coord.relational = None
    coord.graph = None

    async def _boom(*a: Any, **k: Any) -> dict:
        raise AssertionError("optimize_postgresql must not be called without engine")

    monkeypatch.setattr(opt_mod, "optimize_postgresql", _boom)

    result = await opt_mod.optimize_storage(coord)
    assert result["postgresql"] is None


@pytest.mark.unit
async def test_optimize_storage_skips_neo4j_when_no_driver_attr(monkeypatch: pytest.MonkeyPatch) -> None:
    coord = MagicMock()
    coord.vector = None
    coord.relational = None
    coord.graph = MagicMock(spec=[])  # no _driver

    async def _boom(*a: Any, **k: Any) -> dict:
        raise AssertionError("optimize_neo4j must not be called without driver")

    monkeypatch.setattr(opt_mod, "optimize_neo4j", _boom)

    result = await opt_mod.optimize_storage(coord)
    assert result["neo4j"] is None
