"""Unit tests for VectorCypher dual-node manager."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from neo4j.exceptions import ClientError, Neo4jError

from khora.engines.vectorcypher.dual_nodes import (
    _NEO4J_TIMEOUT_CODES,
    ChunkNode,
    DualNodeManager,
    EntityChunkLink,
    _build_neighborhood_query,
)


def _make_neo4j_error(code: str, message: str = "boom") -> ClientError:
    """Build a ClientError instance with a given server-side code.

    The driver's ``code`` attribute is a read-only property derived from
    ``_neo4j_code``, so the public constructor cannot set it. We use the
    same internal factory the driver itself uses when hydrating server
    errors, which guarantees the resulting instance is the correct
    subclass (e.g. ``ClientError`` for ``Neo.ClientError.*``) and exposes
    the requested ``code``.
    """
    exc = Neo4jError._basic_hydrate(neo4j_code=code, message=message)
    # Guard against a future driver refactor silently returning a
    # different subclass. Our except clause in dual_nodes.py catches
    # only ClientError — if _basic_hydrate starts returning a bare
    # Neo4jError (or a different hierarchy), the test would pass
    # vacuously and the production code would break in prod.
    assert isinstance(exc, ClientError), (
        f"expected ClientError for code {code}, got {type(exc).__name__} — neo4j driver may have changed internal API"
    )
    assert exc.code == code, f"expected code={code}, got {exc.code}"
    return exc


def _make_neo4j_driver() -> tuple[MagicMock, AsyncMock]:
    """Create a mock Neo4j driver with properly mocked session context manager.

    Returns:
        Tuple of (driver, session) where session is the mock session object.
    """
    driver = MagicMock()
    session = AsyncMock()

    # Neo4j driver.session() returns a sync object that supports `async with`
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    driver.session.return_value = ctx

    return driver, session


def _make_capturing_driver() -> tuple[MagicMock, AsyncMock, MagicMock]:
    """Driver whose ``execute_write`` runs the write closure against a real mock tx.

    The default ``_make_neo4j_driver`` mocks ``execute_write`` so the inner
    ``_work(tx)`` closure never runs — that proves the method completes, but it
    cannot observe what params reach ``tx.run``. The create paths bind the chunk
    properties as Cypher params, so to assert on those we must actually invoke
    the closure. Here ``execute_write`` calls ``await work_fn(tx)`` with a mock
    ``tx`` whose ``run`` is an AsyncMock; the caller inspects ``tx.run.call_args``.

    Returns:
        Tuple of (driver, session, tx). ``tx.run`` is the AsyncMock to inspect.
    """
    driver, session = _make_neo4j_driver()
    tx = MagicMock()
    tx.run = AsyncMock()

    async def _run_work(work_fn):
        return await work_fn(tx)

    session.execute_write = AsyncMock(side_effect=_run_work)
    return driver, session, tx


# The eight denormalized document-grained keys projected onto the Neo4j Chunk
# node. ``source_timestamp`` is datetime-typed and serialized as an
# ISO string at write time; the other seven are plain string properties.
_DENORM_STRING_KEYS = (
    "source_type",
    "source_name",
    "source_url",
    "external_id",
    "content_type",
    "source",
    "title",
)
_DENORM_KEYS = (*_DENORM_STRING_KEYS, "source_timestamp")


class TestChunkNode:
    """Tests for ChunkNode dataclass."""

    def test_create_chunk_node(self) -> None:
        """Test creating a ChunkNode with all fields."""
        chunk_id = uuid4()
        namespace_id = uuid4()
        document_id = uuid4()
        now = datetime.now(UTC)

        node = ChunkNode(
            id=chunk_id,
            namespace_id=namespace_id,
            document_id=document_id,
            content="test content",
            embedding=[0.1, 0.2, 0.3],
            occurred_at=now,
            created_at=now,
            metadata={"key": "value"},
        )

        assert node.id == chunk_id
        assert node.namespace_id == namespace_id
        assert node.document_id == document_id
        assert node.content == "test content"
        assert node.embedding == [0.1, 0.2, 0.3]
        assert node.occurred_at == now

    def test_chunk_node_defaults(self) -> None:
        """Test ChunkNode default values."""
        node = ChunkNode(
            id=uuid4(),
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="test",
        )
        assert node.embedding is None
        assert node.occurred_at is None
        assert node.created_at is None
        assert node.metadata is None


class TestEntityChunkLink:
    """Tests for EntityChunkLink dataclass."""

    def test_create_link(self) -> None:
        """Test creating an EntityChunkLink."""
        entity_id = uuid4()
        chunk_id = uuid4()
        link = EntityChunkLink(
            entity_id=entity_id,
            chunk_id=chunk_id,
            mention_count=3,
            context="mentioned in paragraph 2",
        )

        assert link.entity_id == entity_id
        assert link.chunk_id == chunk_id
        assert link.mention_count == 3
        assert link.context == "mentioned in paragraph 2"

    def test_link_defaults(self) -> None:
        """Test EntityChunkLink default values."""
        link = EntityChunkLink(entity_id=uuid4(), chunk_id=uuid4())
        assert link.mention_count == 1
        assert link.context == ""


class TestDualNodeManagerInit:
    """Tests for DualNodeManager initialization."""

    def test_init_default_database(self) -> None:
        """Test initialization with default database."""
        driver = MagicMock()
        manager = DualNodeManager(driver)
        assert manager._driver is driver
        assert manager._database == "neo4j"

    def test_init_custom_database(self) -> None:
        """Test initialization with custom database name."""
        driver = MagicMock()
        manager = DualNodeManager(driver, database="custom_db")
        assert manager._database == "custom_db"

    def test_init_default_query_timeout_is_none(self) -> None:
        """Default query_timeout is None (opt-in at the engine layer)."""
        driver = MagicMock()
        manager = DualNodeManager(driver)
        assert manager._query_timeout is None

    def test_init_stores_query_timeout(self) -> None:
        """A non-None query_timeout is stored on the instance."""
        driver = MagicMock()
        manager = DualNodeManager(driver, query_timeout=2.5)
        assert manager._query_timeout == 2.5

    def test_init_query_timeout_is_keyword_only(self) -> None:
        """query_timeout must be passed as a keyword argument."""
        driver = MagicMock()
        with pytest.raises(TypeError):
            DualNodeManager(driver, "neo4j", 2.5)  # type: ignore[misc]


@pytest.mark.unit
class TestDualNodeManagerEnsureIndexes:
    """Tests for ensure_indexes method."""

    @pytest.mark.asyncio
    async def test_ensure_indexes_creates_all(self) -> None:
        """Test that ensure_indexes runs all index creation queries."""
        driver, session = _make_neo4j_driver()
        session.run = AsyncMock()

        manager = DualNodeManager(driver)
        await manager.ensure_indexes()

        # Should run multiple index creation statements
        assert session.run.call_count >= 9  # At least 9 indexes defined

    @pytest.mark.asyncio
    async def test_ensure_indexes_creates_denorm_filter_indexes(self) -> None:
        """The 5 new denorm-key filter indexes are issued.

        Recall pushdown filters land on these Chunk properties, so each needs a
        backing index. We assert one ``CREATE INDEX ... ON (c.<key>)`` statement
        per new key reached ``session.run`` (matched loosely on the property
        reference so an index-name rename doesn't break the test).
        """
        driver, session = _make_neo4j_driver()
        session.run = AsyncMock()

        manager = DualNodeManager(driver)
        await manager.ensure_indexes()

        issued = " || ".join(call.args[0] for call in session.run.call_args_list)
        # source/title/source_url are filterable but were intentionally NOT
        # indexed in this change — assert exactly the 5 the spec lists.
        for key in ("source_type", "source_name", "source_timestamp", "external_id", "content_type"):
            assert f"FOR (c:Chunk) ON (c.{key})" in issued, f"ensure_indexes missing CREATE INDEX on c.{key}"

    @pytest.mark.asyncio
    async def test_ensure_indexes_handles_errors(self) -> None:
        """Test that index creation errors are handled gracefully."""
        driver, session = _make_neo4j_driver()
        session.run = AsyncMock(side_effect=Exception("index already exists"))

        manager = DualNodeManager(driver)
        # Should not raise
        await manager.ensure_indexes()


@pytest.mark.unit
class TestDualNodeManagerCreateChunkNode:
    """Tests for creating chunk nodes in Neo4j."""

    @pytest.fixture
    def manager(self) -> DualNodeManager:
        """Create a manager with mocked driver."""
        driver, session = _make_neo4j_driver()
        session.execute_write = AsyncMock()
        return DualNodeManager(driver)

    @pytest.mark.asyncio
    async def test_create_chunk_node(self, manager: DualNodeManager) -> None:
        """Test creating a single chunk node."""
        from khora.storage.temporal import TemporalChunk

        chunk = TemporalChunk(
            id=uuid4(),
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="test chunk content",
            occurred_at=datetime.now(UTC),
        )

        result_id = await manager.create_chunk_node(chunk)
        assert result_id == chunk.id

    @pytest.mark.asyncio
    async def test_create_chunk_node_generates_id(self, manager: DualNodeManager) -> None:
        """Test that a UUID is generated if chunk has no ID."""
        from khora.storage.temporal import TemporalChunk

        chunk = TemporalChunk(
            id=None,
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="test",
        )

        result_id = await manager.create_chunk_node(chunk)
        assert result_id is not None


@pytest.mark.unit
class TestDualNodeManagerBatchOperations:
    """Tests for batch chunk node operations."""

    @pytest.fixture
    def manager(self) -> DualNodeManager:
        """Create a manager with mocked driver."""
        driver, session = _make_neo4j_driver()
        session.execute_write = AsyncMock()
        return DualNodeManager(driver)

    @pytest.mark.asyncio
    async def test_create_chunk_nodes_batch(self, manager: DualNodeManager) -> None:
        """Test creating chunk nodes in batch."""
        from khora.storage.temporal import TemporalChunk

        namespace_id = uuid4()
        chunks = [
            TemporalChunk(
                id=uuid4(),
                namespace_id=namespace_id,
                document_id=uuid4(),
                content=f"chunk {i}",
                occurred_at=datetime.now(UTC),
            )
            for i in range(5)
        ]

        ids = await manager.create_chunk_nodes_batch(chunks, namespace_id)
        assert len(ids) == 5

    @pytest.mark.asyncio
    async def test_create_chunk_nodes_batch_empty(self, manager: DualNodeManager) -> None:
        """Test batch creation with empty list."""
        ids = await manager.create_chunk_nodes_batch([], uuid4())
        assert ids == []


@pytest.mark.unit
class TestDualNodeManagerDenormProjection:
    """The 8 denormalized document keys reach Neo4j on both create paths.

    A recall filter on ``source_type`` / ``title`` / ... pushes down to a Cypher
    property compare on the Chunk node (see test_compile_cypher), so the create
    paths MUST actually persist those properties — otherwise the pushed-down
    predicate always matches NULL. These tests run the write closure against a
    capturing tx and assert every denorm key is present in the bound params, with
    ``source_timestamp`` serialized as an ``.isoformat()`` string (the datetime is
    not bindable as-is and the Cypher compiler compares it lexicographically).
    """

    @staticmethod
    def _chunk(**overrides):
        from khora.storage.temporal import TemporalChunk

        base = dict(
            id=uuid4(),
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="body",
            occurred_at=datetime.now(UTC),
            source_type="document",
            source_name="linear",
            source_url="https://example.test/x",
            source_timestamp=datetime(2026, 2, 3, 4, 5, 6, tzinfo=UTC),
            external_id="ext-42",
            content_type="text/markdown",
            source="slack",
            title="Quarterly plan",
        )
        base.update(overrides)
        return TemporalChunk(**base)

    @pytest.mark.asyncio
    async def test_create_chunk_node_binds_all_denorm_keys(self) -> None:
        """Single-create binds all 8 denorm keys; source_timestamp is isoformat."""
        driver, _session, tx = _make_capturing_driver()
        manager = DualNodeManager(driver)
        ts = datetime(2026, 2, 3, 4, 5, 6, tzinfo=UTC)
        chunk = self._chunk(source_timestamp=ts)

        await manager.create_chunk_node(chunk)

        tx.run.assert_awaited_once()
        params = tx.run.await_args.kwargs
        # All 8 denorm keys reached the param dict — none silently dropped.
        for key in _DENORM_KEYS:
            assert key in params, f"create_chunk_node dropped denorm key {key!r}"
        assert params["source_type"] == "document"
        assert params["source_name"] == "linear"
        assert params["source_url"] == "https://example.test/x"
        assert params["external_id"] == "ext-42"
        assert params["content_type"] == "text/markdown"
        assert params["source"] == "slack"
        assert params["title"] == "Quarterly plan"
        # The one datetime-typed key binds as its ISO string, NOT a datetime.
        assert params["source_timestamp"] == ts.isoformat()
        assert isinstance(params["source_timestamp"], str)
        # Cypher CREATE references each denorm property by name.
        cypher = tx.run.await_args.args[0]
        for key in _DENORM_KEYS:
            assert f"{key}:" in cypher

    @pytest.mark.asyncio
    async def test_create_chunk_node_none_source_timestamp_binds_none(self) -> None:
        """A None source_timestamp binds None (not a crash on .isoformat())."""
        driver, _session, tx = _make_capturing_driver()
        manager = DualNodeManager(driver)

        await manager.create_chunk_node(self._chunk(source_timestamp=None))

        params = tx.run.await_args.kwargs
        assert params["source_timestamp"] is None

    @pytest.mark.asyncio
    async def test_create_chunk_nodes_batch_binds_all_denorm_keys(self) -> None:
        """Batch-create stamps all 8 denorm keys on each UNWIND row."""
        driver, _session, tx = _make_capturing_driver()
        manager = DualNodeManager(driver)
        ts = datetime(2026, 2, 3, 4, 5, 6, tzinfo=UTC)
        ns = uuid4()
        chunks = [self._chunk(source_timestamp=ts, title=f"doc {i}", external_id=f"ext-{i}") for i in range(3)]

        await manager.create_chunk_nodes_batch(chunks, ns)

        tx.run.assert_awaited_once()
        # Batch passes the row dicts as the ``chunks`` kwarg to UNWIND.
        rows = tx.run.await_args.kwargs["chunks"]
        assert len(rows) == 3
        for i, row in enumerate(rows):
            for key in _DENORM_KEYS:
                assert key in row, f"batch row {i} dropped denorm key {key!r}"
            assert row["source_timestamp"] == ts.isoformat()
            assert isinstance(row["source_timestamp"], str)
            assert row["title"] == f"doc {i}"
            assert row["external_id"] == f"ext-{i}"
            assert row["source_type"] == "document"
            assert row["content_type"] == "text/markdown"
        # The UNWIND CREATE references each denorm property.
        cypher = tx.run.await_args.args[0]
        for key in _DENORM_KEYS:
            assert f"{key}:" in cypher

    @pytest.mark.asyncio
    async def test_create_chunk_nodes_batch_none_source_timestamp_binds_none(self) -> None:
        """Batch path also tolerates a None source_timestamp (no .isoformat())."""
        driver, _session, tx = _make_capturing_driver()
        manager = DualNodeManager(driver)

        await manager.create_chunk_nodes_batch([self._chunk(source_timestamp=None)], uuid4())

        row = tx.run.await_args.kwargs["chunks"][0]
        assert row["source_timestamp"] is None


@pytest.mark.unit
class TestDualNodeManagerLinkOperations:
    """Tests for entity-chunk linking operations."""

    @pytest.fixture
    def manager(self) -> DualNodeManager:
        """Create a manager with mocked driver."""
        driver, session = _make_neo4j_driver()
        session.execute_write = AsyncMock()
        return DualNodeManager(driver)

    @pytest.mark.asyncio
    async def test_link_entity_to_chunk(self, manager: DualNodeManager) -> None:
        """Test creating a single MENTIONED_IN relationship."""
        entity_id = uuid4()
        chunk_id = uuid4()

        await manager.link_entity_to_chunk(entity_id, chunk_id, mention_count=2)
        # Should not raise

    @pytest.mark.asyncio
    async def test_link_entities_to_chunks_batch(self, manager: DualNodeManager) -> None:
        """Test batch MENTIONED_IN relationship creation."""
        links = [
            EntityChunkLink(entity_id=uuid4(), chunk_id=uuid4(), mention_count=1),
            EntityChunkLink(entity_id=uuid4(), chunk_id=uuid4(), mention_count=3),
        ]

        await manager.link_entities_to_chunks_batch(links)
        # Should not raise

    @pytest.mark.asyncio
    async def test_link_entities_to_chunks_batch_empty(self) -> None:
        """Test batch linking with empty list is a no-op."""
        driver = MagicMock()
        manager = DualNodeManager(driver)
        await manager.link_entities_to_chunks_batch([])
        driver.session.assert_not_called()


@pytest.mark.unit
class TestDualNodeManagerTimeLinks:
    """Tests for chunk-to-time linking operations."""

    @pytest.fixture
    def manager(self) -> DualNodeManager:
        """Create a manager with mocked driver."""
        driver, session = _make_neo4j_driver()
        session.execute_write = AsyncMock()
        return DualNodeManager(driver)

    @pytest.mark.asyncio
    async def test_link_chunk_to_time(self, manager: DualNodeManager) -> None:
        """Test creating a single AT_TIME relationship."""
        chunk_id = uuid4()
        time_node_id = uuid4()

        await manager.link_chunk_to_time(chunk_id, time_node_id)
        # Should not raise

    @pytest.mark.asyncio
    async def test_link_chunks_to_time_batch(self, manager: DualNodeManager) -> None:
        """Test batch AT_TIME relationship creation."""
        links = [(uuid4(), uuid4()), (uuid4(), uuid4())]

        await manager.link_chunks_to_time_batch(links)
        # Should not raise

    @pytest.mark.asyncio
    async def test_link_chunks_to_time_batch_empty(self) -> None:
        """Test batch time linking with empty list is a no-op."""
        driver = MagicMock()
        manager = DualNodeManager(driver)
        await manager.link_chunks_to_time_batch([])
        driver.session.assert_not_called()


@pytest.mark.unit
class TestDualNodeManagerGetChunksByEntities:
    """Tests for get_chunks_by_entities method."""

    @pytest.mark.asyncio
    async def test_get_chunks_empty_entity_ids(self) -> None:
        """Test that empty entity IDs returns empty list."""
        driver = MagicMock()
        manager = DualNodeManager(driver)

        result = await manager.get_chunks_by_entities([], uuid4())
        assert result == []
        driver.session.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_chunks_by_entities(self) -> None:
        """Test fetching chunks connected to entities."""
        driver, session = _make_neo4j_driver()

        mock_records = [
            {
                "chunk_id": str(uuid4()),
                "content": "test content",
                "document_id": str(uuid4()),
                "occurred_at": None,
                "metadata": {},
                "entity_ids": [str(uuid4())],
                "total_mentions": 2,
            }
        ]
        session.execute_read = AsyncMock(return_value=mock_records)

        manager = DualNodeManager(driver)
        result = await manager.get_chunks_by_entities(
            [uuid4()],
            uuid4(),
            limit=10,
        )

        assert len(result) == 1
        assert result[0]["content"] == "test content"


@pytest.mark.unit
class TestDualNodeManagerGetEntityNeighborhoods:
    """Tests for get_entity_neighborhoods method."""

    @pytest.mark.asyncio
    async def test_get_neighborhoods_empty_ids(self) -> None:
        """Test that empty entity IDs returns empty dict."""
        driver = MagicMock()
        manager = DualNodeManager(driver)

        result = await manager.get_entity_neighborhoods([], uuid4())
        assert result == {}

    @pytest.mark.asyncio
    async def test_get_neighborhoods_depth_clamped(self) -> None:
        """Test that depth is clamped to 1-4."""
        driver, session = _make_neo4j_driver()
        session.execute_read = AsyncMock(return_value=[])

        manager = DualNodeManager(driver)

        # Depth 0 should be clamped to 1
        await manager.get_entity_neighborhoods([uuid4()], uuid4(), depth=0)
        # Depth 10 should be clamped to 4
        await manager.get_entity_neighborhoods([uuid4()], uuid4(), depth=10)


@pytest.mark.unit
class TestDualNodeManagerGetEntityNeighborhoodsTimeout:
    """Tests for the configurable per-transaction timeout in get_entity_neighborhoods.

    The Neo4j Python driver applies transaction-level timeouts via
    ``neo4j.unit_of_work(timeout=...)``, which decorates the read closure
    that ``session.execute_read`` runs. The DualNodeManager should:
      * skip the decorator entirely when ``query_timeout`` is None,
      * apply it (with the configured value) otherwise,
      * convert the two known transaction-timeout error codes into an
        empty result dict + warning log,
      * re-raise any other error so callers still see real failures.
    """

    @pytest.mark.asyncio
    async def test_wraps_work_with_unit_of_work_when_timeout_set(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When query_timeout is set, the closure passed to execute_read is
        the decorated one — end-to-end, not just "the decorator was called".

        This guards against a refactor that drops the wrap or rebinds
        ``_work`` back to the original between the decorate step and the
        execute_read call. We verify by:
          * Spying on ``unit_of_work`` to tag its output with a sentinel
            attribute (``_is_timed_work``) AND preserve the driver's
            real ``.timeout`` attribute.
          * Capturing the function handed to ``session.execute_read`` and
            asserting the sentinel is present.
          * Actually invoking the captured function with a mock tx so we
            exercise the full wrap → execute_read → _work path.
        """
        driver, session = _make_neo4j_driver()

        # Import the real neo4j.unit_of_work so the spy delegates to
        # it (keeps the .timeout attribute the driver sets).
        from neo4j import unit_of_work as real_unit_of_work

        decorator_calls: list[float | None] = []

        def _spy_unit_of_work(*, timeout: float | None = None, **kwargs):
            decorator_calls.append(timeout)
            real_decorator = real_unit_of_work(timeout=timeout, **kwargs)

            def _marking_decorator(fn):
                wrapped = real_decorator(fn)
                wrapped._is_timed_work = True  # test-only sentinel
                return wrapped

            return _marking_decorator

        monkeypatch.setattr(
            "khora.engines.vectorcypher.dual_nodes.unit_of_work",
            _spy_unit_of_work,
        )

        # Capture whatever function is passed to execute_read, and
        # actually invoke it against a mock tx so the inner closure
        # runs (otherwise the test degenerates to "mock returned []").
        captured: dict[str, object] = {}

        async def _capture_execute_read(work_fn):
            captured["work_fn"] = work_fn

            # Build a minimal AsyncManagedTransaction-shaped mock.
            async def _empty_result_iter():
                if False:  # pragma: no cover
                    yield
                return

            class _AsyncIter:
                def __aiter__(self):
                    return self

                async def __anext__(self):
                    raise StopAsyncIteration

            result_cursor = MagicMock()
            result_cursor.__aiter__ = lambda self: _AsyncIter()
            fake_tx = AsyncMock()
            fake_tx.run = AsyncMock(return_value=result_cursor)

            inner = await work_fn(fake_tx)
            captured["inner_called"] = True
            captured["inner_result"] = inner
            return inner

        session.execute_read = AsyncMock(side_effect=_capture_execute_read)

        manager = DualNodeManager(driver, query_timeout=3.0)
        result = await manager.get_entity_neighborhoods([uuid4()], uuid4(), depth=1)

        # unit_of_work was called exactly once — during __init__'s hoist,
        # NOT per get_entity_neighborhoods call. Behaviour-locks the
        # M8 hoist against accidental un-hoisting.
        assert decorator_calls == [3.0]

        # The function reaching execute_read MUST carry:
        #   (1) our test sentinel — proves our decorator ran,
        #   (2) the driver's own .timeout attribute — proves it's still
        #       the real neo4j.unit_of_work wrapper (not something we
        #       accidentally unwrapped along the way).
        work_fn = captured["work_fn"]
        assert getattr(work_fn, "_is_timed_work", False), (
            "decorator was not applied to the function handed to execute_read"
        )
        assert getattr(work_fn, "timeout", None) == 3.0, "wrapped function missing the driver's .timeout attribute"

        # The inner Cypher closure actually ran (full end-to-end wrap).
        assert captured.get("inner_called") is True
        assert result == {}

    @pytest.mark.asyncio
    async def test_skips_unit_of_work_when_timeout_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When query_timeout is None, unit_of_work is NOT invoked."""
        driver, session = _make_neo4j_driver()
        session.execute_read = AsyncMock(return_value=[])

        unit_of_work_mock = MagicMock()
        monkeypatch.setattr(
            "khora.engines.vectorcypher.dual_nodes.unit_of_work",
            unit_of_work_mock,
        )

        manager = DualNodeManager(driver)  # default: query_timeout=None
        result = await manager.get_entity_neighborhoods([uuid4()], uuid4(), depth=1)

        assert result == {}
        unit_of_work_mock.assert_not_called()
        session.execute_read.assert_awaited_once()

    @pytest.mark.parametrize("timeout_code", _NEO4J_TIMEOUT_CODES)
    @pytest.mark.asyncio
    async def test_returns_empty_on_timeout_codes(self, timeout_code: str) -> None:
        """Both server- and client-configured timeout codes degrade to {}."""
        driver, session = _make_neo4j_driver()
        timeout_exc = _make_neo4j_error(timeout_code, message="transaction timed out")
        # _make_neo4j_error already asserts isinstance(ClientError) + code
        # internally (see the guard in the helper). Re-assert here as a
        # call-site layer of defence.
        assert isinstance(timeout_exc, ClientError)
        assert timeout_exc.code == timeout_code

        session.execute_read = AsyncMock(side_effect=timeout_exc)

        manager = DualNodeManager(driver, query_timeout=1.0)

        with patch("khora.engines.vectorcypher.dual_nodes.logger") as mock_logger:
            result = await manager.get_entity_neighborhoods([uuid4()], uuid4(), depth=2)

        assert result == {}
        mock_logger.warning.assert_called_once()

        warning_call = mock_logger.warning.call_args
        # Structured kwargs (operators rely on these for filtering).
        assert warning_call.kwargs.get("code") == timeout_code
        assert warning_call.kwargs.get("timeout") == 1.0
        assert warning_call.kwargs.get("timeout_occurred") is True
        assert warning_call.kwargs.get("n") == 1
        assert warning_call.kwargs.get("d") == 2
        # Template assertions (M7). When loguru is patched,
        # ``mock_logger.warning.call_args.args[0]`` is the RAW template
        # string (verified at runtime), NOT the interpolated form. These
        # guard against a future refactor silently dropping the structured
        # fields or switching to f-string interpolation — in either case
        # the placeholder markers disappear from args[0] and the assertion
        # fires.
        template = warning_call.args[0]
        assert "timed out" in template
        assert "{timeout}" in template
        assert "{code}" in template
        assert "{n}" in template
        assert "{d}" in template

    @pytest.mark.asyncio
    async def test_reraises_non_timeout_client_error(self) -> None:
        """Non-timeout ClientErrors (e.g. syntax errors) propagate to the caller."""
        driver, session = _make_neo4j_driver()
        syntax_exc = _make_neo4j_error(
            "Neo.ClientError.Statement.SyntaxError",
            message="Cypher syntax error",
        )
        # Make sure we built a real ClientError-shaped instance — if a
        # future driver split this code into a different class hierarchy
        # the test setup itself should fail rather than the assertion.
        assert isinstance(syntax_exc, ClientError)
        assert syntax_exc.code not in _NEO4J_TIMEOUT_CODES

        session.execute_read = AsyncMock(side_effect=syntax_exc)
        manager = DualNodeManager(driver, query_timeout=1.0)

        with pytest.raises(ClientError) as excinfo:
            await manager.get_entity_neighborhoods([uuid4()], uuid4(), depth=1)

        assert excinfo.value.code == "Neo.ClientError.Statement.SyntaxError"

    @pytest.mark.asyncio
    async def test_reraises_non_client_errors(self) -> None:
        """Errors that are not ClientError (e.g. RuntimeError) also propagate."""
        driver, session = _make_neo4j_driver()
        session.execute_read = AsyncMock(side_effect=RuntimeError("connection lost"))
        manager = DualNodeManager(driver, query_timeout=1.0)

        with pytest.raises(RuntimeError, match="connection lost"):
            await manager.get_entity_neighborhoods([uuid4()], uuid4(), depth=1)

    @pytest.mark.asyncio
    async def test_timeout_disabled_still_handles_results(self) -> None:
        """With query_timeout=None, normal execution returns parsed neighborhoods."""
        driver, session = _make_neo4j_driver()
        source_id = str(uuid4())
        related_id = str(uuid4())
        session.execute_read = AsyncMock(
            return_value=[
                {
                    "source_id": source_id,
                    "source_name": "alice",
                    "source_entity_type": "PERSON",
                    "source_description": None,
                    "source_source_tool": None,
                    "related_entities": [
                        {
                            "id": related_id,
                            "name": "bob",
                            "entity_type": "PERSON",
                            "description": None,
                            "source_tool": None,
                            "distance": 1,
                        }
                    ],
                }
            ]
        )

        manager = DualNodeManager(driver)
        result = await manager.get_entity_neighborhoods([uuid4()], uuid4(), depth=1)

        assert result == {
            source_id: [
                {
                    "id": related_id,
                    "name": "bob",
                    "entity_type": "PERSON",
                    "description": None,
                    "source_tool": None,
                    "distance": 1,
                }
            ]
        }

    @pytest.mark.asyncio
    async def test_emits_timeout_telemetry_span(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """On timeout, a dedicated trace_span is emitted with structured attrs.

        Replaces the former telemetry counter —
        operators can now alert on span name ``*.timeout`` in Logfire/OTEL
        and filter by ``timeout_s``, ``entity_count``, ``depth``, ``code``,
        and ``namespace_id`` attributes.
        """
        from contextlib import contextmanager

        driver, session = _make_neo4j_driver()
        timeout_exc = _make_neo4j_error(
            "Neo.ClientError.Transaction.TransactionTimedOut",
            message="transaction timed out",
        )
        session.execute_read = AsyncMock(side_effect=timeout_exc)

        captured_spans: list[tuple[str, dict]] = []

        @contextmanager
        def _fake_trace_span(name, **attributes):
            entry = (name, dict(attributes))
            captured_spans.append(entry)

            class _NoOp:
                def set_attribute(self, key, value):
                    captured_spans[-1][1][key] = value

                def set_attributes(self, attrs):
                    captured_spans[-1][1].update(attrs)

            yield _NoOp()

        monkeypatch.setattr(
            "khora.engines.vectorcypher.dual_nodes.trace_span",
            _fake_trace_span,
        )

        manager = DualNodeManager(driver, query_timeout=2.5)
        ns = uuid4()
        eids = [uuid4(), uuid4(), uuid4()]

        result = await manager.get_entity_neighborhoods(eids, ns, depth=3)

        assert result == {}
        # Exactly one span ending in ``.timeout`` — this is the signal
        # operators will alert on. If a refactor drops the trace_span
        # call, this test fires.
        timeout_spans = [s for s in captured_spans if s[0].endswith(".timeout")]
        assert len(timeout_spans) == 1, f"expected exactly one .timeout span, got {[s[0] for s in captured_spans]}"
        name, attrs = timeout_spans[0]
        assert name == "khora.neo4j.get_entity_neighborhoods.timeout"
        # All five attributes ops needs for dashboard filters.
        assert attrs["timeout_s"] == 2.5
        assert attrs["entity_count"] == 3
        assert attrs["depth"] == 3
        assert attrs["code"] == "Neo.ClientError.Transaction.TransactionTimedOut"
        assert attrs["namespace_id"] == str(ns)
        # The Devil's-Advocate delta: timeout_occurred lives on the
        # OTEL span as an attribute (not just in the loguru log).
        assert attrs["timeout_occurred"] is True


@pytest.mark.unit
class TestDualNodeManagerSiblingTimeouts:
    """Tests for the configurable per-transaction timeout in the 4 sibling methods:

    - get_chunks_by_entities
    - get_relationships_between
    - get_temporal_chunks
    - get_entity_channels

    All four use the same pattern as get_entity_neighborhoods:
      * apply ``_timed_unit_of_work`` when ``query_timeout`` is set,
      * convert known timeout codes into an empty result + warning log,
      * re-raise non-timeout ClientErrors,
      * work normally when ``query_timeout=None``.
    """

    # Each entry: (method_name, call_args_factory, expected_empty_sentinel)
    # call_args_factory returns (args, kwargs) for the method call.
    _METHOD_SPECS: list[tuple[str, str]] = [
        ("get_chunks_by_entities", "list"),
        ("get_relationships_between", "list"),
        ("get_temporal_chunks", "list"),
        ("get_entity_channels", "list"),
    ]

    @staticmethod
    def _call_method(manager: DualNodeManager, method_name: str):
        """Build (args, kwargs) for the given method and call it."""
        if method_name == "get_chunks_by_entities":
            return manager.get_chunks_by_entities([uuid4()], uuid4())
        elif method_name == "get_relationships_between":
            return manager.get_relationships_between([str(uuid4()), str(uuid4())], str(uuid4()))
        elif method_name == "get_temporal_chunks":
            return manager.get_temporal_chunks(uuid4(), [uuid4()])
        elif method_name == "get_entity_channels":
            return manager.get_entity_channels([str(uuid4())], str(uuid4()))
        else:
            raise ValueError(f"unknown method: {method_name}")

    @pytest.mark.parametrize(
        "method_name",
        ["get_chunks_by_entities", "get_relationships_between", "get_temporal_chunks", "get_entity_channels"],
    )
    @pytest.mark.parametrize("timeout_code", _NEO4J_TIMEOUT_CODES)
    @pytest.mark.asyncio
    async def test_returns_empty_on_timeout(self, method_name: str, timeout_code: str) -> None:
        """Each method degrades to [] on both known timeout codes."""
        driver, session = _make_neo4j_driver()
        timeout_exc = _make_neo4j_error(timeout_code, message="transaction timed out")
        session.execute_read = AsyncMock(side_effect=timeout_exc)

        manager = DualNodeManager(driver, query_timeout=1.0)

        with patch("khora.engines.vectorcypher.dual_nodes.logger") as mock_logger:
            result = await self._call_method(manager, method_name)

        assert result == []
        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args
        assert warning_call.kwargs.get("timeout_occurred") is True
        assert warning_call.kwargs.get("code") == timeout_code

    @pytest.mark.parametrize(
        "method_name",
        ["get_chunks_by_entities", "get_relationships_between", "get_temporal_chunks", "get_entity_channels"],
    )
    @pytest.mark.asyncio
    async def test_reraises_non_timeout_client_error(self, method_name: str) -> None:
        """Non-timeout ClientErrors propagate to the caller."""
        driver, session = _make_neo4j_driver()
        syntax_exc = _make_neo4j_error(
            "Neo.ClientError.Statement.SyntaxError",
            message="Cypher syntax error",
        )
        session.execute_read = AsyncMock(side_effect=syntax_exc)

        manager = DualNodeManager(driver, query_timeout=1.0)

        with pytest.raises(ClientError) as excinfo:
            await self._call_method(manager, method_name)

        assert excinfo.value.code == "Neo.ClientError.Statement.SyntaxError"

    @pytest.mark.parametrize(
        "method_name,mock_return",
        [
            ("get_chunks_by_entities", [{"chunk_id": "c1", "content": "hi", "metadata": None}]),
            ("get_relationships_between", [{"id": "r1", "source_entity_id": "a", "target_entity_id": "b"}]),
            ("get_temporal_chunks", [{"chunk_id": "c2", "content": "hello", "metadata": None}]),
            ("get_entity_channels", ["session-1", "session-2"]),
        ],
    )
    @pytest.mark.asyncio
    async def test_works_normally_with_timeout_none(self, method_name: str, mock_return: list) -> None:
        """With query_timeout=None, methods return data normally."""
        driver, session = _make_neo4j_driver()
        session.execute_read = AsyncMock(return_value=mock_return)

        manager = DualNodeManager(driver)  # default: query_timeout=None

        result = await self._call_method(manager, method_name)

        assert result == mock_return
        session.execute_read.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_emits_timeout_telemetry_span_get_chunks_by_entities(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """On timeout, get_chunks_by_entities emits a trace_span with structured attrs."""
        from contextlib import contextmanager

        driver, session = _make_neo4j_driver()
        timeout_exc = _make_neo4j_error(
            "Neo.ClientError.Transaction.TransactionTimedOut",
            message="transaction timed out",
        )
        session.execute_read = AsyncMock(side_effect=timeout_exc)

        captured_spans: list[tuple[str, dict]] = []

        @contextmanager
        def _fake_trace_span(name, **attributes):
            entry = (name, dict(attributes))
            captured_spans.append(entry)

            class _NoOp:
                def set_attribute(self, key, value):
                    captured_spans[-1][1][key] = value

                def set_attributes(self, attrs):
                    captured_spans[-1][1].update(attrs)

            yield _NoOp()

        monkeypatch.setattr(
            "khora.engines.vectorcypher.dual_nodes.trace_span",
            _fake_trace_span,
        )

        manager = DualNodeManager(driver, query_timeout=2.5)
        ns = uuid4()
        eids = [uuid4(), uuid4()]

        result = await manager.get_chunks_by_entities(eids, ns)

        assert result == []
        timeout_spans = [s for s in captured_spans if s[0].endswith(".timeout")]
        assert len(timeout_spans) == 1
        name, attrs = timeout_spans[0]
        assert name == "khora.neo4j.get_chunks_by_entities.timeout"
        assert attrs["timeout_s"] == 2.5
        assert attrs["entity_count"] == 2
        assert attrs["namespace_id"] == str(ns)
        assert attrs["code"] == "Neo.ClientError.Transaction.TransactionTimedOut"
        assert attrs["timeout_occurred"] is True

    @pytest.mark.parametrize(
        "method_name",
        ["get_chunks_by_entities", "get_relationships_between", "get_temporal_chunks", "get_entity_channels"],
    )
    @pytest.mark.asyncio
    async def test_reraises_non_client_errors(self, method_name: str) -> None:
        """Non-ClientError exceptions (e.g. RuntimeError) propagate unchanged."""
        driver, session = _make_neo4j_driver()
        session.execute_read = AsyncMock(side_effect=RuntimeError("connection lost"))

        manager = DualNodeManager(driver, query_timeout=1.0)

        with pytest.raises(RuntimeError, match="connection lost"):
            await self._call_method(manager, method_name)


@pytest.mark.unit
class TestDualNodeManagerDeleteOperations:
    """Tests for delete operations."""

    @pytest.mark.asyncio
    async def test_delete_chunks_by_document(self) -> None:
        """Test deleting chunks by document ID."""
        driver, session = _make_neo4j_driver()
        session.execute_write = AsyncMock(return_value=5)

        manager = DualNodeManager(driver)
        deleted = await manager.delete_chunks_by_document(uuid4(), uuid4())

        assert deleted == 5


@pytest.mark.unit
class TestDualNodeManagerGetEntityChannels:
    """Tests for get_entity_channels method."""

    @pytest.mark.asyncio
    async def test_get_entity_channels_returns_channels(self) -> None:
        """Test that get_entity_channels returns distinct channel strings."""
        driver, session = _make_neo4j_driver()
        session.execute_read = AsyncMock(return_value=["session-1", "session-2", "session-3"])

        manager = DualNodeManager(driver)
        channels = await manager.get_entity_channels(
            entity_ids=[str(uuid4()), str(uuid4())],
            namespace_id=str(uuid4()),
        )

        assert channels == ["session-1", "session-2", "session-3"]
        session.execute_read.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_entity_channels_empty_entity_ids(self) -> None:
        """Test that get_entity_channels returns empty list for no entities."""
        driver, session = _make_neo4j_driver()
        manager = DualNodeManager(driver)

        channels = await manager.get_entity_channels(
            entity_ids=[],
            namespace_id=str(uuid4()),
        )

        assert channels == []
        session.execute_read.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_entity_channels_single_channel(self) -> None:
        """Test that a single channel is returned correctly."""
        driver, session = _make_neo4j_driver()
        session.execute_read = AsyncMock(return_value=["only-session"])

        manager = DualNodeManager(driver)
        channels = await manager.get_entity_channels(
            entity_ids=[str(uuid4())],
            namespace_id=str(uuid4()),
        )

        assert channels == ["only-session"]


class _EmptyAsyncResult:
    """An async-iterable Neo4j result that yields no records."""

    def __aiter__(self) -> _EmptyAsyncResult:
        return self

    async def __anext__(self):
        raise StopAsyncIteration


def _make_read_capturing_driver() -> tuple[MagicMock, AsyncMock, MagicMock]:
    """Driver whose ``execute_read`` runs the read closure against a real mock tx.

    ``get_chunks_by_entities`` reads via ``session.execute_read(_work)`` where
    ``_work(tx)`` calls ``tx.run(query, **params)``. To observe the query and
    bound params we must actually invoke the closure: ``execute_read`` here calls
    ``await work_fn(tx)`` with a mock ``tx`` whose ``run`` returns an empty
    async result. The caller inspects ``tx.run.await_args``.
    """
    driver, session = _make_neo4j_driver()
    tx = MagicMock()
    tx.run = AsyncMock(return_value=_EmptyAsyncResult())

    async def _run_work(work_fn):
        return await work_fn(tx)

    session.execute_read = AsyncMock(side_effect=_run_work)
    return driver, session, tx


class TestGetChunksByEntitiesFilterPushdown:
    """The caller filter's system-key slice is pushed into the graph Cypher query."""

    @pytest.mark.asyncio
    async def test_system_key_filter_pushed_as_bound_param(self) -> None:
        """A system-key predicate is spliced into the WHERE with its value bound.

        The compiled predicate references ``c.<key>`` and the value travels as a
        Cypher ``$param`` (no string interpolation) — the injection-safe path.
        """
        from khora.filter import RecallFilter, parse_to_ast

        driver, _session, tx = _make_read_capturing_driver()
        manager = DualNodeManager(driver)
        ast = parse_to_ast(RecallFilter.model_validate({"source_name": "linear"}))

        await manager.get_chunks_by_entities([uuid4()], uuid4(), filter_ast=ast, limit=5)

        tx.run.assert_awaited_once()
        query = tx.run.await_args.args[0]
        params = tx.run.await_args.kwargs
        assert "source_name" in query, "system-key predicate was not pushed into the Cypher WHERE"
        assert any(v == "linear" for v in params.values()), "filter value was not passed as a bound parameter"

    @pytest.mark.asyncio
    async def test_metadata_only_filter_not_spliced(self) -> None:
        """A metadata-only filter consumes nothing on the Cypher side (deferred to post-filter)."""
        from khora.filter import RecallFilter, parse_to_ast

        driver, _session, tx = _make_read_capturing_driver()
        manager = DualNodeManager(driver)
        ast = parse_to_ast(RecallFilter.model_validate({"metadata.tag": "urgent"}))

        await manager.get_chunks_by_entities([uuid4()], uuid4(), filter_ast=ast, limit=5)

        tx.run.assert_awaited_once()
        params = tx.run.await_args.kwargs
        assert not any(v == "urgent" for v in params.values()), (
            "metadata leaf was wrongly pushed to Cypher instead of deferred to the in-memory post-filter"
        )

    @pytest.mark.asyncio
    async def test_pushed_keys_out_sink_receives_the_executed_compile(self) -> None:
        """The ``pushed_keys_out`` sink reports exactly what the executed WHERE spliced.

        A system-key filter splices the predicate and appends its
        ``consumed_keys`` to the sink (the report source = execution input). A
        metadata-only filter consumes nothing on the Cypher side, and a no-filter
        call compiles nothing — both leave the sink untouched, so the graph plan
        cannot over-claim a pushdown that did not happen.
        """
        from khora.filter import RecallFilter, parse_to_ast

        driver, _session, _tx = _make_read_capturing_driver()
        manager = DualNodeManager(driver)

        # System-key filter -> sink receives exactly the compile's consumed_keys.
        system_sink: list[frozenset[str]] = []
        await manager.get_chunks_by_entities(
            [uuid4()],
            uuid4(),
            filter_ast=parse_to_ast(RecallFilter.model_validate({"source_name": "linear"})),
            limit=5,
            pushed_keys_out=system_sink,
        )
        assert system_sink == [frozenset({"source_name"})]

        # Metadata-only filter -> nothing pushed -> sink untouched.
        metadata_sink: list[frozenset[str]] = []
        await manager.get_chunks_by_entities(
            [uuid4()],
            uuid4(),
            filter_ast=parse_to_ast(RecallFilter.model_validate({"metadata.tag": "urgent"})),
            limit=5,
            pushed_keys_out=metadata_sink,
        )
        assert metadata_sink == []

        # No filter -> no compile -> sink untouched.
        no_filter_sink: list[frozenset[str]] = []
        await manager.get_chunks_by_entities(
            [uuid4()],
            uuid4(),
            filter_ast=None,
            limit=5,
            pushed_keys_out=no_filter_sink,
        )
        assert no_filter_sink == []


@pytest.mark.unit
class TestBoundedNeighborhoodExpansionQuery:
    """Tests for the bounded per-hop expansion query (#1419).

    The old ``OPTIONAL MATCH path = (e)-[*1..depth]-(related:Entity)``
    enumerated every undirected trail (exponential in graph density); the
    replacement unrolls one frontier-expansion block per hop with a visited
    set and a ``$hop_limit`` fan-out cap. These tests lock the query shape;
    result parity against the legacy query is covered by the real-Neo4j
    integration test
    ``tests/integration/test_neo4j_neighborhood_expansion_parity_integration.py``.
    """

    @pytest.mark.parametrize("depth", [1, 2, 3, 4])
    @pytest.mark.parametrize("prefer_current", [False, True])
    def test_no_variable_length_pattern(self, depth: int, prefer_current: bool) -> None:
        """The exponential all-paths pattern must be gone for every shape."""
        query = _build_neighborhood_query(depth, prefer_current)
        assert "[*1.." not in query
        assert "relationships(path)" not in query

    @pytest.mark.parametrize("depth", [1, 2, 3, 4])
    def test_one_expansion_block_per_hop(self, depth: int) -> None:
        """depth N unrolls exactly N frontier-expansion blocks with 1..N distances."""
        query = _build_neighborhood_query(depth, prefer_current=False)
        assert query.count("OPTIONAL MATCH") == depth
        assert query.count("collect(DISTINCT") == depth
        for i in range(1, depth + 1):
            assert f"distance: {i}}}" in query
        assert f"distance: {depth + 1}}}" not in query

    def test_hop_and_result_limits_parameterized(self) -> None:
        """Frontier cap and per-entity result cap bind via Cypher params."""
        query = _build_neighborhood_query(2, prefer_current=False)
        assert query.count("[0..$hop_limit]") == 2  # one cap per hop
        assert "_found[0..$limit]" in query

    def test_prefer_current_filters_every_hop(self) -> None:
        """valid_until is checked per traversed relationship and per reported node.

        valid_until is now a native ZONED DATETIME (#1472), so the comparison is
        cast-free (``valid_until > _now``) - engaging the entity_ns_valid_until
        index range instead of the former non-sargable ``datetime(valid_until)``
        per-row string cast.
        """
        depth = 3
        query = _build_neighborhood_query(depth, prefer_current=True)
        assert "datetime() AS _now" in query
        # No per-row cast of the stored property (would defeat the index).
        assert "datetime(_r" not in query
        assert "datetime(x.valid_until)" not in query
        # One relationship-validity check per hop...
        for i in range(1, depth + 1):
            assert f"_r{i}.valid_until IS NULL OR _r{i}.valid_until > _now" in query
        # ...and one reported-node validity check per hop.
        assert query.count("x.valid_until IS NULL OR x.valid_until > _now") == depth

    def test_prefer_current_false_has_no_temporal_filter(self) -> None:
        query = _build_neighborhood_query(3, prefer_current=False)
        assert "valid_until" not in query
        assert "_now" not in query

    def test_reported_nodes_filtered_but_traversal_unrestricted(self) -> None:
        """Only *reported* nodes are Entity+namespace filtered - the traversal
        itself must remain label/namespace-unrestricted so entities reachable
        through :Chunk / :TimeNode intermediates keep surfacing (old-query
        parity: the ``[*1..depth]`` pattern constrained only the endpoint)."""
        query = _build_neighborhood_query(1, prefer_current=False)
        # The expansion pattern has no label on the neighbor node.
        assert "OPTIONAL MATCH (_cur1)-[_r1]-(_nb1)" in query
        # The report-side comprehension is where Entity/namespace filtering lives.
        assert "'Entity' IN labels(x)" in query
        assert "x.namespace_id = $namespace_id" in query

    @pytest.mark.asyncio
    async def test_hop_limit_param_reaches_tx_run(self) -> None:
        """The hop_limit kwarg is bound as a Cypher parameter."""
        driver, session = _make_neo4j_driver()
        captured: dict[str, object] = {}

        async def _capture_execute_read(work_fn):
            class _AsyncIter:
                def __aiter__(self):
                    return self

                async def __anext__(self):
                    raise StopAsyncIteration

            result_cursor = MagicMock()
            result_cursor.__aiter__ = lambda self: _AsyncIter()

            async def _run(query, **params):
                captured["query"] = query
                captured["params"] = params
                return result_cursor

            tx = MagicMock()
            tx.run = _run
            return await work_fn(tx)

        session.execute_read = AsyncMock(side_effect=_capture_execute_read)
        manager = DualNodeManager(driver)

        result = await manager.get_entity_neighborhoods([uuid4()], uuid4(), depth=2, limit_per_entity=7, hop_limit=55)

        assert result == {}
        params = captured["params"]
        assert params["hop_limit"] == 55
        assert params["limit"] == 7
        assert "$hop_limit" in captured["query"]


@pytest.mark.unit
class TestGetEntityNeighborhoodsTimeoutDegradation:
    """ADR-001 (#1419): timeout no longer degrades silently."""

    @pytest.mark.asyncio
    async def test_timeout_appends_degradation_and_bumps_counter(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        driver, session = _make_neo4j_driver()
        session.execute_read = AsyncMock(
            side_effect=_make_neo4j_error("Neo.ClientError.Transaction.TransactionTimedOut")
        )
        counter = MagicMock()
        monkeypatch.setattr(
            "khora.engines.vectorcypher.retriever._CYPHER_EXPAND_DEGRADED_COUNTER",
            counter,
        )

        manager = DualNodeManager(driver, query_timeout=1.5)
        sink: list = []
        result = await manager.get_entity_neighborhoods([uuid4(), uuid4()], uuid4(), depth=3, degradations=sink)

        assert result == {}
        counter.add.assert_called_once_with(1, attributes={"reason": "neo4j_timeout"})
        assert len(sink) == 1
        deg = sink[0]
        assert deg["component"] == "vectorcypher.cypher_expand"
        assert deg["reason"] == "neo4j_timeout"
        assert "1.5" in deg["detail"]
        assert "entity_count=2" in deg["detail"]
        assert "depth=3" in deg["detail"]
        assert deg["exception"] == "ClientError"

    @pytest.mark.asyncio
    async def test_timeout_without_sink_still_counts_and_returns_empty(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Callers without a sink (engine.find_related_entities) keep the old
        return contract; the counter still fires so the event is observable."""
        driver, session = _make_neo4j_driver()
        session.execute_read = AsyncMock(
            side_effect=_make_neo4j_error("Neo.ClientError.Transaction.TransactionTimedOutClientConfiguration")
        )
        counter = MagicMock()
        monkeypatch.setattr(
            "khora.engines.vectorcypher.retriever._CYPHER_EXPAND_DEGRADED_COUNTER",
            counter,
        )

        manager = DualNodeManager(driver, query_timeout=1.0)
        result = await manager.get_entity_neighborhoods([uuid4()], uuid4(), depth=1)

        assert result == {}
        counter.add.assert_called_once_with(1, attributes={"reason": "neo4j_timeout"})

    @pytest.mark.asyncio
    async def test_non_timeout_error_records_no_degradation(self) -> None:
        driver, session = _make_neo4j_driver()
        session.execute_read = AsyncMock(
            side_effect=_make_neo4j_error("Neo.ClientError.Statement.SyntaxError", message="bad cypher")
        )
        manager = DualNodeManager(driver, query_timeout=1.0)
        sink: list = []

        with pytest.raises(ClientError):
            await manager.get_entity_neighborhoods([uuid4()], uuid4(), depth=1, degradations=sink)

        assert sink == []
