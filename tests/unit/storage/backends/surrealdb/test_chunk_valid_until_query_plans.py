"""Chunk namespace-scoped reads and the graph live filter must plan ``Iterate Index``.

Two OR shapes on ``chunk`` / ``entity`` / ``relates_to`` used to collapse the
SurrealDB 2.x planner to ``Iterate Table`` — a full cross-tenant scan of the
whole corpus on recall hot paths (#1592):

* **chunk namespace scope** — ``(namespace = $ns_rid OR namespace.namespace_id =
  $ns_str)``. The ``namespace.namespace_id`` leg is a record traversal, not a
  comparison on an indexed field, so the disjunction was unservable. Rewritten to
  the scalar ``namespace_id`` (denormalised onto every chunk since #1221 and now
  indexed by ``idx_chunk_namespace_id``), the two legs union index scans. The one
  read where a plain disjunction still table-scans regardless — ``search_similar``,
  whose ``embedding IS NOT NULL`` guard poisons the union and whose
  ``vector::dot`` errors on a NONE embedding so the guard cannot move to Python —
  is split into two OR-free legs merged in Python.
* **graph live filter** — ``(valid_until IS NONE OR valid_until > time::now())``
  on ``list_entities`` / ``list_relationships``. Split into two OR-free legs
  merged in Python, the #1590 pattern.

Like ``test_document_query_plans``, these tests do NOT retype the SQL: they drive
the real adapter methods through a recording connection and re-issue each captured
statement with ``EXPLAIN`` appended. They run against ``memory://`` — no server,
no docker.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest

pytest.importorskip("surrealdb")

from khora.core.models import Chunk, Entity, MemoryNamespace, Relationship, TenancyMode  # noqa: E402
from khora.storage.backends.surrealdb.connection import SurrealDBConnection  # noqa: E402
from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter  # noqa: E402
from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter  # noqa: E402
from khora.storage.backends.surrealdb.schema import ensure_search_indexes  # noqa: E402
from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter  # noqa: E402

pytestmark = pytest.mark.unit

_BASE = datetime(2026, 3, 1, tzinfo=UTC)
_EMB = [0.1, 0.2, 0.3, 0.9]


class _RecordingConnection:
    """Pass-through wrapper keeping every ``(sql, params)`` the adapter issues."""

    def __init__(self, conn: SurrealDBConnection) -> None:
        self._conn = conn
        self.statements: list[tuple[str, dict[str, Any]]] = []

    async def query(self, sql: str, params: dict[str, Any] | None = None) -> Any:
        self.statements.append((sql, params or {}))
        return await self._conn.query(sql, params)

    async def query_one(self, sql: str, params: dict[str, Any] | None = None) -> Any:
        self.statements.append((sql, params or {}))
        return await self._conn.query_one(sql, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)

    def reset(self) -> None:
        self.statements.clear()

    def selects(self) -> list[tuple[str, dict[str, Any]]]:
        return [(sql, params) for sql, params in self.statements if sql.lstrip().upper().startswith("SELECT")]


@pytest.fixture
async def recorder():
    conn = SurrealDBConnection(mode="memory", namespace="khora_test", database="plans", embedding_dimension=4)
    await conn.connect()
    await ensure_search_indexes(conn)
    recording = _RecordingConnection(conn)
    try:
        yield recording
    finally:
        await conn.disconnect()


@pytest.fixture
async def seeded(recorder):
    """Two namespaces, each with chunks + entities + a relationship.

    A populated, multi-tenant table is what makes the plan assertion meaningful:
    the point is that the planner *chose* an index (rows to choose over) and that
    a regression would leak the foreign namespace's rows. Entities carry a mix of
    live (``valid_until`` NONE), live-with-future-window, and retired (past
    ``valid_until``) rows so the live filter has something to hide.
    """
    rel = SurrealDBRelationalAdapter(recorder)
    graph = SurrealDBGraphAdapter(recorder)
    vector = SurrealDBVectorAdapter(recorder)

    namespaces: list[UUID] = []
    for _ in range(2):
        nid = uuid4()
        await rel.create_namespace(MemoryNamespace(id=nid, namespace_id=nid, tenancy_mode=TenancyMode.SHARED))
        namespaces.append(nid)

    doc = uuid4()
    for nid in namespaces:
        ents: list[Entity] = []
        for i in range(18):
            await vector.create_chunk(
                Chunk(
                    namespace_id=nid,
                    document_id=doc,
                    content=f"hello world chunk {i}",
                    chunk_index=i,
                    start_char=0,
                    end_char=1,
                    embedding=_EMB,
                    created_at=_BASE + timedelta(minutes=i),
                )
            )
            # i % 3 == 0 -> retired (past); i % 3 == 1 -> future window; else NONE.
            if i % 3 == 0:
                valid_until = _BASE - timedelta(days=1)
            elif i % 3 == 1:
                valid_until = _BASE + timedelta(days=365)
            else:
                valid_until = None
            e = Entity(
                namespace_id=nid,
                name=f"e-{nid}-{i}",
                entity_type="PERSON",
                valid_until=valid_until,
                created_at=_BASE + timedelta(minutes=i),
                updated_at=_BASE + timedelta(minutes=i),
            )
            ents.append((await graph.upsert_entities_batch(nid, [e]))[0][0])
        # One live and one retired relationship between the first two entities.
        await graph.create_relationship(
            Relationship(
                namespace_id=nid,
                source_entity_id=ents[1].id,
                target_entity_id=ents[2].id,
                relationship_type="KNOWS",
                created_at=_BASE,
            )
        )
        await graph.create_relationship(
            Relationship(
                namespace_id=nid,
                source_entity_id=ents[3].id,
                target_entity_id=ents[4].id,
                relationship_type="KNEW",
                valid_until=_BASE - timedelta(days=1),
                created_at=_BASE + timedelta(minutes=1),
            )
        )
    recorder.reset()
    return {"rel": rel, "graph": graph, "vector": vector, "ns": namespaces[0], "other_ns": namespaces[1], "doc": doc}


async def _explain(recorder: _RecordingConnection, *, expected_statements: int, label: str) -> list[Any]:
    recorded = recorder.selects()
    assert len(recorded) == expected_statements, (
        f"{label}: expected {expected_statements} SELECT(s), captured {len(recorded)}:\n"
        + "\n".join(sql for sql, _ in recorded)
    )
    plans = []
    for sql, params in recorded:
        plan = await recorder._conn.query(f"{sql} EXPLAIN", params)
        plans.append((sql, plan))
    return plans


def _assert_no_table_scan(sql: str, plan: Any, *, label: str) -> None:
    ops = [step["operation"] for step in plan]
    assert not any("Iterate Table" in op for op in ops), (
        f"{label}: statement fell back to a full table scan — it reads every row in the "
        f"database, all namespaces included.\n  sql:  {sql}\n  ops: {ops}"
    )
    assert any(op.startswith("Iterate Index") for op in ops), (
        f"{label}: no index iteration in plan.\n  sql: {sql}\n  ops: {ops}"
    )


# --------------------------------------------------------------------------- #
# chunk namespace scope
# --------------------------------------------------------------------------- #


async def test_search_similar_legs_plan_index_scans(seeded, recorder) -> None:
    """Semantic search is split into two namespace legs, each index-eligible."""
    res = await seeded["vector"].search_similar(seeded["ns"], _EMB, limit=5)
    assert res, "fixture produced no similarity hits"
    plans = await _explain(recorder, expected_statements=2, label="search_similar")
    for sql, plan in plans:
        _assert_no_table_scan(sql, plan, label="search_similar leg")


async def test_search_fulltext_plans_index_scan(seeded, recorder) -> None:
    res = await seeded["vector"].search_fulltext(seeded["ns"], "hello", limit=5)
    assert res, "fixture produced no BM25 hits"
    ((sql, plan),) = await _explain(recorder, expected_statements=1, label="search_fulltext")
    _assert_no_table_scan(sql, plan, label="search_fulltext")


async def test_get_chunks_by_document_plans_index_scan(seeded, recorder) -> None:
    res = await seeded["vector"].get_chunks_by_document(seeded["doc"], namespace_id=seeded["ns"])
    assert res, "fixture produced no chunks for the document"
    ((sql, plan),) = await _explain(recorder, expected_statements=1, label="get_chunks_by_document")
    _assert_no_table_scan(sql, plan, label="get_chunks_by_document")


async def test_no_chunk_read_uses_the_namespace_record_traversal(seeded, recorder) -> None:
    """Source tripwire: the unservable ``namespace.namespace_id`` form is gone.

    ``EXPLAIN`` reports the plan this engine picked; a later SurrealDB that learned
    to index the traversal would let the plan assertions pass while the statement
    still table-scans on older deployments. This checks the text instead.
    """
    v = seeded["vector"]
    ns, doc = seeded["ns"], seeded["doc"]
    await v.get_chunk(uuid4(), namespace_id=ns)
    await v.get_chunks_batch([uuid4()], namespace_id=ns)
    await v.get_chunks_by_document(doc, namespace_id=ns)
    await v.search_similar(ns, _EMB, limit=5)
    await v.search_fulltext(ns, "hello", limit=5)
    offenders = [sql for sql, _ in recorder.selects() if "namespace.namespace_id" in sql]
    assert not offenders, "a chunk read still uses the record-traversal namespace form:\n" + "\n".join(offenders)


async def test_search_similar_stays_namespace_scoped(seeded, recorder) -> None:
    """The leg split must not leak the foreign tenant's chunks into the result."""
    res = await seeded["vector"].search_similar(seeded["ns"], _EMB, limit=100)
    ns_ids = {c.namespace_id for c, _ in res}
    assert ns_ids == {seeded["ns"]}, f"search_similar leaked foreign namespaces: {ns_ids}"


async def test_delete_chunks_by_document_counts_the_real_deletion(seeded, recorder) -> None:
    """The returned count is the rows actually deleted, scoped to one tenant.

    A ``count() ... GROUP ALL`` over the namespace OR double-counts rows matching
    both legs on this engine, so the count is taken from ``DELETE ... RETURN
    BEFORE`` instead. Both namespaces hold 18 chunks for the shared document;
    deleting one tenant's must report 18 (not a doubled 36/72) and leave the
    other tenant untouched.
    """
    v, ns, other, doc = seeded["vector"], seeded["ns"], seeded["other_ns"], seeded["doc"]
    deleted = await v.delete_chunks_by_document(doc, namespace_id=ns)
    assert deleted == 18, f"delete over-/under-counted: {deleted}"
    assert await v.get_chunks_by_document(doc, namespace_id=ns) == []
    survivors = await v.get_chunks_by_document(doc, namespace_id=other)
    assert len(survivors) == 18, f"cross-tenant deletion: {len(survivors)} of the other tenant's chunks survived"


# --------------------------------------------------------------------------- #
# graph live filter (valid_until)
# --------------------------------------------------------------------------- #


async def test_list_entities_legs_plan_index_scans(seeded, recorder) -> None:
    res = await seeded["graph"].list_entities(seeded["ns"], limit=100)
    assert res, "fixture produced no live entities"
    plans = await _explain(recorder, expected_statements=2, label="list_entities")
    for sql, plan in plans:
        _assert_no_table_scan(sql, plan, label="list_entities leg")


async def test_list_relationships_legs_plan_index_scans(seeded, recorder) -> None:
    res = await seeded["graph"].list_relationships(seeded["ns"], limit=100)
    assert res, "fixture produced no live relationships"
    plans = await _explain(recorder, expected_statements=2, label="list_relationships")
    for sql, plan in plans:
        _assert_no_table_scan(sql, plan, label="list_relationships leg")


async def test_no_live_filter_read_contains_a_disjunction(seeded, recorder) -> None:
    """The live-filter reads must carry neither an ``OR`` nor the record traversal."""
    await seeded["graph"].list_entities(seeded["ns"], limit=100)
    await seeded["graph"].list_relationships(seeded["ns"], limit=100)
    for sql, _ in recorder.selects():
        assert " OR " not in sql.upper(), f"a disjunction is back in a live-filter read:\n{sql}"


async def test_list_entities_hides_retired_and_keeps_live(seeded, recorder) -> None:
    """Retired entities (past ``valid_until``) are excluded; NONE and future kept.

    The fixture retires ``i % 3 == 0`` (6 of 18) and keeps the rest live, so the
    split legs must together return exactly the 12 live rows — no retired row
    leaking through, no live row dropped by the merge.
    """
    res = await seeded["graph"].list_entities(seeded["ns"], limit=100)
    assert len(res) == 12, f"live set wrong after merge: {len(res)}"
    assert all(e.valid_until is None or e.valid_until > _BASE for e in res), "a retired entity leaked through"


async def test_list_relationships_hides_retired(seeded, recorder) -> None:
    res = await seeded["graph"].list_relationships(seeded["ns"], limit=100)
    types = sorted(r.relationship_type for r in res)
    assert types == ["KNOWS"], f"retired relationship leaked or live one dropped: {types}"


async def test_list_entities_pagination_walks_every_live_row_once(seeded, recorder) -> None:
    """Paging the merged legs visits each live entity exactly once, newest first.

    The merge fetches ``offset + limit`` from each leg and slices after re-sorting;
    a per-leg slice or a lost tenant conjunct would drop or duplicate rows across
    page boundaries. Small page size forces several boundaries over the mixed
    NONE / future-window live set.
    """
    graph, ns = seeded["graph"], seeded["ns"]
    full = await graph.list_entities(ns, limit=100)
    created = [e.created_at for e in full]
    assert created == sorted(created, reverse=True), "merge did not order created_at DESC"

    seen: list[UUID] = []
    for offset in range(0, len(full) + 5, 5):
        page = await graph.list_entities(ns, limit=5, offset=offset)
        seen.extend(e.id for e in page)
        if not page:
            break
    assert len(seen) == len(set(seen)), "pagination returned a duplicate across a page boundary"
    assert set(seen) == {e.id for e in full}, "pagination skipped a live entity"


async def test_the_disjunctive_forms_really_table_scan(seeded, recorder) -> None:
    """The premise, pinned: the replaced OR shapes really do ``Iterate Table``.

    If a future engine plans any of these on an index, the client-side splits are
    no longer forced and can be reconsidered — that is the intended signal.
    """
    conn = recorder._conn
    ns = seeded["ns"]
    from khora.storage.backends.surrealdb._helpers import _rid

    ns_rid = _rid("memory_namespace", ns)
    ns_str = str(ns)
    shapes = {
        "chunk_traversal": (
            "SELECT * FROM chunk WHERE (namespace = $ns_rid OR namespace.namespace_id = $ns_str)",
            {"ns_rid": ns_rid, "ns_str": ns_str},
        ),
        "chunk_similar_or": (
            "SELECT *, vector::dot(embedding, $q) AS similarity FROM chunk "
            "WHERE (namespace = $ns_rid OR namespace_id = $ns_str) AND embedding IS NOT NULL "
            "ORDER BY similarity DESC LIMIT 5",
            {"ns_rid": ns_rid, "ns_str": ns_str, "q": _EMB},
        ),
        "entity_live_or": (
            "SELECT * FROM entity WHERE namespace = $ns_rid "
            "AND (valid_until IS NONE OR valid_until > time::now()) ORDER BY created_at DESC LIMIT 100",
            {"ns_rid": ns_rid},
        ),
        "relationship_live_or": (
            "SELECT * FROM relates_to WHERE namespace_id = $ns "
            "AND (valid_until IS NONE OR valid_until > time::now()) ORDER BY created_at DESC LIMIT 100",
            {"ns": ns_str},
        ),
    }
    for label, (sql, params) in shapes.items():
        plan = await conn.query(f"{sql} EXPLAIN", params)
        ops = [step["operation"] for step in plan]
        assert any("Iterate Table" in op for op in ops), (
            f"{label}: this engine now indexes the disjunctive form. The client-side split is no "
            f"longer forced — revisit it.\n  ops: {ops}"
        )
