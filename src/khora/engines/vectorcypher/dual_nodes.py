"""Dual-node manager for HippoRAG 2 architecture in Neo4j.

Implements the dual-node structure where:
- (:Chunk) nodes represent text chunks with content and embeddings
- (:Entity) nodes represent extracted entities
- [:MENTIONED_IN] edges link entities to chunks where they appear
- [:AT_TIME] edges link chunks/entities to time hierarchy nodes

This structure enables efficient retrieval by:
1. Finding entry entities via vector similarity
2. Expanding to related entities via graph traversal
3. Retrieving chunks via MENTIONED_IN relationships
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from loguru import logger
from neo4j import unit_of_work
from neo4j.exceptions import ClientError

from khora.core.diagnostics import Degradation
from khora.storage.backends.mixins import deserialize_dict, serialize_dict
from khora.telemetry import trace, trace_span

if TYPE_CHECKING:
    from neo4j import AsyncDriver, AsyncSession

    from khora.core.temporal import ChunkTemporalFilter, TemporalChunk
    from khora.filter import FilterNode
    from khora.storage.backends.neo4j import Neo4jBackend


# Neo4j 5.x splits timeout errors into two codes depending on whether
# the timeout fired from the server's db.transaction.timeout setting
# or from our client-configured unit_of_work(timeout=...). We catch
# both so that our configured ceiling is always respected regardless
# of which code path the server takes.
_NEO4J_TIMEOUT_CODES = (
    "Neo.ClientError.Transaction.TransactionTimedOut",
    "Neo.ClientError.Transaction.TransactionTimedOutClientConfiguration",
)

# Per-recall shared Neo4j session (issue #1468). Set by
# ``DualNodeManager.bind_session()`` for the duration of a recall's sequential
# graph reads; read by ``DualNodeManager._session()`` so those reads reuse one
# pooled connection instead of acquiring one each. Module-level (not an
# instance attribute) so it is isolated per async task-tree — a single manager
# instance serves concurrent recalls, and a plain attribute would cross-leak
# their sessions. ``None`` means "no active bind" → open a fresh session as
# before.
_BOUND_SESSION: ContextVar[AsyncSession | None] = ContextVar("khora_dualnode_bound_session", default=None)


def _build_neighborhood_query(depth: int, prefer_current: bool) -> str:
    """Build the bounded per-hop neighborhood expansion query (issue #1419).

    Replaces the former ``OPTIONAL MATCH path = (e)-[*1..depth]-(related:Entity)``
    all-paths enumeration, whose cost was exponential in graph density (every
    distinct undirected trail to the same node was enumerated just to compute
    DISTINCT + min distance). This builds an unrolled per-hop BFS instead:
    each hop expands the current frontier one relationship, dedupes with
    ``collect(DISTINCT ...)``, drops already-visited nodes, and caps the new
    frontier at ``$hop_limit`` nodes so hub nodes cannot blow up a hop. Total
    work is linear in the reachable (bounded) subgraph instead of exponential
    in path count. Single round-trip, plain Cypher - no APOC required.

    Result parity with the old query (the acceptance bar for #1419):

    - A node's reported ``distance`` is its minimum hop count from the source,
      identical to the old ``min(length(path))`` because BFS visits each node
      first at its shortest distance. The old query could additionally emit
      the *same* node again at larger distances (its DISTINCT was over the
      whole map including ``distance``); consumers take the max score =
      min distance (``retriever._cypher_expand``), so dropping those
      strictly-dominated duplicates is consumption-equivalent.
    - Traversal is undirected over ALL relationship types and ALL intermediate
      node labels/namespaces (the old pattern constrained only the endpoint),
      so entities reachable e.g. through a shared :Chunk or :TimeNode node
      still surface. Only *reported* nodes are filtered to
      ``:Entity`` + same namespace + ``<> source``.
    - ``prefer_current`` filters every traversed relationship's
      ``valid_until`` per hop (matching the old ``all(r IN relationships(path)
      ...)``) and filters *reported* entities' ``valid_until`` - but expired
      entities remain traversable-through, exactly like old intermediate
      nodes, which were never validity-checked.
    - Results are appended hop by hop, so ``_found`` is distance-ascending -
      the same ordering the old ``ORDER BY distance`` + ``[0..$limit]``
      truncation produced.

    Bounded deviation (documented, not parity-relevant on realistic graphs):
    when a hop discovers more than ``$hop_limit`` new nodes, the overflow is
    arbitrary-dropped for that hop and may be rediscovered at a later hop
    with a larger distance. The old query had no such bound - it timed out
    instead (silently dropping the whole graph channel).

    Args:
        depth: Number of hops to unroll (caller clamps to 1-4).
        prefer_current: Whether to add per-hop valid_until filtering.

    Returns:
        Cypher query string with ``$entity_ids``, ``$namespace_id``,
        ``$hop_limit`` and ``$limit`` parameters.
    """
    now_carry = ", _now" if prefer_current else ""
    lines = [
        "UNWIND $entity_ids AS eid",
        "MATCH (e:Entity {id: eid, namespace_id: $namespace_id})",
    ]
    if prefer_current:
        # Bind the server clock once per source row as a native ZONED DATETIME.
        # valid_until is now persisted as a native ZONED DATETIME (#1472), so the
        # per-hop comparison is a cast-free ``valid_until > _now`` - engaging the
        # entity_ns_valid_until index range instead of the former non-sargable
        # ``datetime(valid_until) > _now`` per-row string cast.
        lines.append("WITH e, datetime() AS _now")
    lines.append(f"WITH e{now_carry}, [e] AS _visited, [e] AS _frontier, [] AS _found")
    for i in range(1, depth + 1):
        rel_filter = f"\n  AND (_r{i}.valid_until IS NULL OR _r{i}.valid_until > _now)" if prefer_current else ""
        node_filter = "\n         AND (x.valid_until IS NULL OR x.valid_until > _now)" if prefer_current else ""
        lines.append(
            # An empty frontier must not kill the row (UNWIND [] discards it),
            # so substitute [null]; OPTIONAL MATCH on a null start node then
            # yields a null neighbor, which collect() ignores.
            f"UNWIND (CASE WHEN size(_frontier) = 0 THEN [null] ELSE _frontier END) AS _cur{i}\n"
            f"OPTIONAL MATCH (_cur{i})-[_r{i}]-(_nb{i})\n"
            f"WHERE NOT _nb{i} IN _visited{rel_filter}\n"
            f"WITH e{now_carry}, _visited, _found, collect(DISTINCT _nb{i})[0..$hop_limit] AS _next{i}\n"
            f"WITH e{now_carry}, _visited + _next{i} AS _visited, _next{i} AS _frontier,\n"
            f"     _found + [x IN _next{i}\n"
            f"       WHERE 'Entity' IN labels(x)\n"
            f"         AND x.namespace_id = $namespace_id\n"
            f"         AND x.id <> e.id{node_filter}\n"
            f"       | {{id: x.id, name: x.name, entity_type: x.entity_type,\n"
            f"          description: x.description, source_tool: x.source_tool,\n"
            f"          distance: {i}}}] AS _found"
        )
    lines.append(
        "RETURN e.id AS source_id,\n"
        "       e.name AS source_name,\n"
        "       e.entity_type AS source_entity_type,\n"
        "       e.description AS source_description,\n"
        "       e.source_tool AS source_source_tool,\n"
        "       _found[0..$limit] AS related_entities"
    )
    return "\n".join(lines)


@dataclass
class ChunkNode:
    """Chunk node representation for Neo4j."""

    id: UUID
    namespace_id: UUID
    document_id: UUID
    content: str
    embedding: list[float] | None = None
    occurred_at: datetime | None = None
    created_at: datetime | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class EntityChunkLink:
    """Link between entity and chunk."""

    entity_id: UUID
    chunk_id: UUID
    mention_count: int = 1
    context: str = ""


class DualNodeManager:
    """Manages HippoRAG 2 dual-node structure in Neo4j.

    Creates and maintains:
    - (:Chunk) nodes with content and embeddings
    - [:MENTIONED_IN] relationships from Entity to Chunk
    - [:AT_TIME] relationships to time hierarchy

    The dual-node structure enables efficient retrieval:
    - Vector search on Entity nodes for entry points
    - Graph expansion to find related entities
    - Chunk retrieval via MENTIONED_IN for context
    """

    def __init__(
        self,
        driver: AsyncDriver,
        database: str = "neo4j",
        *,
        query_timeout: float | None = None,
        pool_backend: Neo4jBackend | None = None,
    ) -> None:
        """Initialize the manager.

        Args:
            driver: Neo4j async driver
            database: Database name
            query_timeout: Optional per-transaction timeout in seconds,
                applied to ``get_entity_neighborhoods`` to bound a runaway
                Neo4j query (e.g. on an unusually dense/large reachable
                subgraph). ``None`` disables the timeout.
            pool_backend: Optional ``Neo4jBackend`` instance. When provided,
                :meth:`_session` delegates to ``pool_backend._session()`` so
                pool metric instrumentation (timeout counter, acquire
                duration) covers all traversal paths driven by this manager.
                Falls back to the raw driver session when ``None`` (e.g.
                tests or callers that do not have a backend wired).
        """
        self._driver = driver
        self._database = database
        self._query_timeout = query_timeout
        self._pool_backend = pool_backend
        # Pre-bind the unit_of_work decorator once. The factory call is
        # cheap but non-zero and the produced decorator is fully reusable
        # — see neo4j.unit_of_work: it closes over (metadata, timeout) and
        # returns a plain wrapper function that can decorate any number of
        # transaction callables. Hoisting shaves an allocation per call
        # on the hot neighborhood lookup path.
        self._timed_unit_of_work = unit_of_work(timeout=query_timeout) if query_timeout is not None else None

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[AsyncSession]:
        """Yield a Neo4j session, routed through ``pool_backend`` if set.

        All Neo4j access from this class must go through this helper so that
        ``Neo4jBackend._session`` pool metrics (timeout counter, acquire
        duration) observe traversals driven by ``DualNodeManager``.

        When a recall has an active :meth:`bind_session` (issue #1468) the
        bound session is reused instead of opening a new one, so a recall's
        sequential graph reads share one pooled connection. The bind owns the
        session's lifecycle, so this contextmanager must NOT close it here — it
        just yields the already-open session.
        """
        bound = _BOUND_SESSION.get()
        if bound is not None:
            yield bound
        elif self._pool_backend is not None:
            async with self._pool_backend._session() as session:
                yield session
        else:
            async with self._driver.session(database=self._database) as session:
                yield session

    @asynccontextmanager
    async def bind_session(self) -> AsyncIterator[None]:
        """Bind one Neo4j session for a recall's sequential graph reads (#1468).

        Opens a single session (through ``pool_backend`` when set, so pool
        metrics still observe it) and installs it in the ``_BOUND_SESSION``
        ContextVar for the duration of the ``async with`` body. Every
        :meth:`_session` call inside the body — from ``get_entity_neighborhoods``,
        the retriever's ``_version_filter_entities`` / ``_fetch_version_history``,
        and ``get_chunks_by_entities`` — reuses this session, collapsing what
        were 2-3 separate pool acquisitions per recall into one.

        SAFETY: a Neo4j ``AsyncSession`` is not safe for concurrent use, so the
        caller MUST only wrap a region whose graph reads run strictly
        sequentially (awaited one after another). The retriever's expand ->
        (version filter) -> chunk-fetch chain is exactly such a region; the
        concurrent relationship fetch runs outside it. Re-entrant binds are a
        no-op reuse of the outer bind (nested ``async with`` on the same
        manager keeps the outer session).
        """
        if _BOUND_SESSION.get() is not None:
            # Already bound by an enclosing scope; reuse it (no new session,
            # no token to reset — the outer bind owns lifecycle).
            yield
            return

        if self._pool_backend is not None:
            session_cm = self._pool_backend._session()
        else:
            session_cm = self._driver.session(database=self._database)

        # The fold is a pure connection-reuse optimization. If the underlying
        # session source is not a real async context manager (e.g. an AsyncMock
        # driver/backend in unit tests, whose ``.session()`` returns a bare
        # coroutine), skip binding and yield without a bound session — reads
        # then fall back to per-call ``_session()`` exactly as before. This
        # keeps the fold transparent to callers that mock at the manager
        # boundary rather than the driver boundary.
        if not (hasattr(session_cm, "__aenter__") and hasattr(session_cm, "__aexit__")):
            # Best-effort close of a stray coroutine to quiet "never awaited"
            # warnings when the source was an AsyncMock; a real CM is entered
            # below instead.
            close = getattr(session_cm, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()
            yield
            return

        async with session_cm as session:
            token = _BOUND_SESSION.set(session)
            try:
                yield
            finally:
                _BOUND_SESSION.reset(token)

    async def ensure_indexes(self) -> None:
        """Create indexes for Chunk and TimeNode nodes."""
        indexes = [
            "CREATE INDEX chunk_id IF NOT EXISTS FOR (c:Chunk) ON (c.id)",
            "CREATE INDEX chunk_namespace IF NOT EXISTS FOR (c:Chunk) ON (c.namespace_id)",
            "CREATE INDEX chunk_document IF NOT EXISTS FOR (c:Chunk) ON (c.document_id)",
            "CREATE INDEX chunk_occurred_at IF NOT EXISTS FOR (c:Chunk) ON (c.occurred_at)",
            # Composite for efficient namespace + time queries
            "CREATE INDEX chunk_ns_time IF NOT EXISTS FOR (c:Chunk) ON (c.namespace_id, c.occurred_at)",
            # Chunk filter indexes for structured queries
            "CREATE INDEX chunk_source_system IF NOT EXISTS FOR (c:Chunk) ON (c.source_system)",
            "CREATE INDEX chunk_author IF NOT EXISTS FOR (c:Chunk) ON (c.author)",
            "CREATE INDEX chunk_channel IF NOT EXISTS FOR (c:Chunk) ON (c.channel)",
            # Denormalized document-grained filter indexes for recall pushdown
            "CREATE INDEX chunk_source_type IF NOT EXISTS FOR (c:Chunk) ON (c.source_type)",
            "CREATE INDEX chunk_source_name IF NOT EXISTS FOR (c:Chunk) ON (c.source_name)",
            "CREATE INDEX chunk_source_timestamp IF NOT EXISTS FOR (c:Chunk) ON (c.source_timestamp)",
            "CREATE INDEX chunk_external_id IF NOT EXISTS FOR (c:Chunk) ON (c.external_id)",
            "CREATE INDEX chunk_content_type IF NOT EXISTS FOR (c:Chunk) ON (c.content_type)",
            # TimeNode indexes for time hierarchy traversal
            "CREATE INDEX timenode_id IF NOT EXISTS FOR (t:TimeNode) ON (t.id)",
            "CREATE INDEX timenode_namespace IF NOT EXISTS FOR (t:TimeNode) ON (t.namespace_id)",
        ]

        async with self._session() as session:
            for index in indexes:
                try:
                    await session.run(index)
                except Exception as e:
                    logger.debug(f"Index creation: {e}")

    async def create_chunk_node(self, chunk: TemporalChunk) -> UUID:
        """Create a single Chunk node in Neo4j.

        Args:
            chunk: Temporal chunk to create node for

        Returns:
            Chunk node ID
        """
        chunk_id = chunk.id or uuid4()

        query = """
        CREATE (c:Chunk {
            id: $id,
            namespace_id: $namespace_id,
            document_id: $document_id,
            content: $content,
            occurred_at: $occurred_at,
            created_at: $created_at,
            source_system: $source_system,
            author: $author,
            channel: $channel,
            confidence: $confidence,
            metadata: $metadata,
            chunker_info: $chunker_info,
            source_type: $source_type,
            source_name: $source_name,
            source_url: $source_url,
            source_timestamp: $source_timestamp,
            external_id: $external_id,
            content_type: $content_type,
            source: $source,
            title: $title
        })
        RETURN c.id AS id
        """

        params = dict(
            id=str(chunk_id),
            namespace_id=str(chunk.namespace_id),
            document_id=str(chunk.document_id),
            content=chunk.content,
            occurred_at=chunk.occurred_at.isoformat() if chunk.occurred_at else None,
            created_at=chunk.created_at.isoformat() if chunk.created_at else datetime.now(UTC).isoformat(),
            source_system=chunk.source_system,
            author=chunk.author,
            channel=chunk.channel,
            confidence=chunk.confidence,
            metadata=serialize_dict(chunk.metadata or {}),
            chunker_info=json.dumps(chunk.chunker_info or {}),
            source_type=chunk.source_type,
            source_name=chunk.source_name,
            source_url=chunk.source_url,
            source_timestamp=chunk.source_timestamp.isoformat() if chunk.source_timestamp else None,
            external_id=chunk.external_id,
            content_type=chunk.content_type,
            source=chunk.source,
            title=chunk.title,
        )

        async with self._session() as session:

            async def _work(tx):
                await tx.run(query, **params)

            await session.execute_write(_work)

        logger.debug(f"Created Chunk node: {chunk_id}")
        return chunk_id

    @trace("khora.neo4j.create_chunk_nodes_batch")
    async def create_chunk_nodes_batch(
        self,
        chunks: list[TemporalChunk],
        namespace_id: UUID,
    ) -> list[UUID]:
        """Create Chunk nodes in batch.

        Args:
            chunks: List of temporal chunks
            namespace_id: Namespace ID

        Returns:
            List of created chunk IDs
        """
        if not chunks:
            return []

        # Prepare batch data
        chunk_data = []
        chunk_ids = []

        for chunk in chunks:
            chunk_id = chunk.id or uuid4()
            chunk_ids.append(chunk_id)

            chunk_data.append(
                {
                    "id": str(chunk_id),
                    "namespace_id": str(namespace_id),
                    "document_id": str(chunk.document_id),
                    "content": chunk.content,
                    "occurred_at": chunk.occurred_at.isoformat() if chunk.occurred_at else None,
                    "created_at": (chunk.created_at.isoformat() if chunk.created_at else datetime.now(UTC).isoformat()),
                    "source_system": chunk.source_system,
                    "author": chunk.author,
                    "channel": chunk.channel,
                    "confidence": chunk.confidence,
                    "metadata": serialize_dict(chunk.metadata or {}),
                    "chunker_info": json.dumps(chunk.chunker_info or {}),
                    "source_type": chunk.source_type,
                    "source_name": chunk.source_name,
                    "source_url": chunk.source_url,
                    "source_timestamp": chunk.source_timestamp.isoformat() if chunk.source_timestamp else None,
                    "external_id": chunk.external_id,
                    "content_type": chunk.content_type,
                    "source": chunk.source,
                    "title": chunk.title,
                }
            )

        query = """
        UNWIND $chunks AS chunk
        CREATE (c:Chunk {
            id: chunk.id,
            namespace_id: chunk.namespace_id,
            document_id: chunk.document_id,
            content: chunk.content,
            occurred_at: chunk.occurred_at,
            created_at: chunk.created_at,
            source_system: chunk.source_system,
            author: chunk.author,
            channel: chunk.channel,
            confidence: chunk.confidence,
            metadata: chunk.metadata,
            chunker_info: chunk.chunker_info,
            source_type: chunk.source_type,
            source_name: chunk.source_name,
            source_url: chunk.source_url,
            source_timestamp: chunk.source_timestamp,
            external_id: chunk.external_id,
            content_type: chunk.content_type,
            source: chunk.source,
            title: chunk.title
        })
        """

        async with self._session() as session:

            async def _work(tx):
                await tx.run(query, chunks=chunk_data)

            await session.execute_write(_work)

        logger.debug(f"Created {len(chunk_ids)} Chunk nodes in batch")
        return chunk_ids

    async def link_entity_to_chunk(
        self,
        entity_id: UUID,
        chunk_id: UUID,
        *,
        mention_count: int = 1,
        context: str = "",
    ) -> None:
        """Create MENTIONED_IN relationship from Entity to Chunk.

        Args:
            entity_id: Entity node ID
            chunk_id: Chunk node ID
            mention_count: Number of times entity is mentioned in chunk
            context: Surrounding context of the mention
        """
        query = """
        MATCH (e:Entity {id: $entity_id})
        MATCH (c:Chunk {id: $chunk_id})
        MERGE (e)-[r:MENTIONED_IN]->(c)
        ON CREATE SET r.mention_count = $mention_count, r.context = $context
        ON MATCH SET r.mention_count = r.mention_count + $mention_count
        """

        async with self._session() as session:

            async def _work(tx):
                await tx.run(
                    query,
                    entity_id=str(entity_id),
                    chunk_id=str(chunk_id),
                    mention_count=mention_count,
                    context=context,
                )

            await session.execute_write(_work)

    @trace("khora.neo4j.link_entities_to_chunks_batch")
    async def link_entities_to_chunks_batch(
        self,
        links: list[EntityChunkLink],
    ) -> None:
        """Create MENTIONED_IN relationships in batch.

        Args:
            links: List of EntityChunkLink objects
        """
        if not links:
            return

        link_data = [
            {
                "entity_id": str(link.entity_id),
                "chunk_id": str(link.chunk_id),
                "mention_count": link.mention_count,
                "context": link.context,
            }
            for link in links
        ]

        query = """
        UNWIND $links AS link
        MATCH (e:Entity {id: link.entity_id})
        MATCH (c:Chunk {id: link.chunk_id})
        MERGE (e)-[r:MENTIONED_IN]->(c)
        ON CREATE SET r.mention_count = link.mention_count, r.context = link.context
        ON MATCH SET r.mention_count = r.mention_count + link.mention_count
        """

        async with self._session() as session:

            async def _work(tx):
                await tx.run(query, links=link_data)

            await session.execute_write(_work)

        logger.debug(f"Created {len(links)} MENTIONED_IN relationships")

    async def link_chunk_to_time(
        self,
        chunk_id: UUID,
        time_node_id: UUID,
    ) -> None:
        """Create AT_TIME relationship from Chunk to TimeNode.

        Args:
            chunk_id: Chunk node ID
            time_node_id: TimeNode ID (usually a day node)
        """
        query = """
        MATCH (c:Chunk {id: $chunk_id})
        MATCH (t:TimeNode {id: $time_node_id})
        MERGE (c)-[:AT_TIME]->(t)
        """

        async with self._session() as session:

            async def _work(tx):
                await tx.run(
                    query,
                    chunk_id=str(chunk_id),
                    time_node_id=str(time_node_id),
                )

            await session.execute_write(_work)

    async def link_chunks_to_time_batch(
        self,
        chunk_time_links: list[tuple[UUID, UUID]],
    ) -> None:
        """Create AT_TIME relationships in batch.

        Args:
            chunk_time_links: List of (chunk_id, time_node_id) tuples
        """
        if not chunk_time_links:
            return

        link_data = [
            {"chunk_id": str(chunk_id), "time_node_id": str(time_id)} for chunk_id, time_id in chunk_time_links
        ]

        query = """
        UNWIND $links AS link
        MATCH (c:Chunk {id: link.chunk_id})
        MATCH (t:TimeNode {id: link.time_node_id})
        MERGE (c)-[:AT_TIME]->(t)
        """

        async with self._session() as session:

            async def _work(tx):
                await tx.run(query, links=link_data)

            await session.execute_write(_work)

    @trace(
        "khora.neo4j.get_chunks_by_entities",
        include={"entity_ids", "namespace_id"},
        result=lambda r: {"chunk_count": len(r)},
    )
    async def get_chunks_by_entities(
        self,
        entity_ids: list[UUID],
        namespace_id: UUID,
        *,
        temporal_filter: ChunkTemporalFilter | None = None,
        temporal_sort: bool = False,
        prefer_current: bool = False,
        limit: int = 50,
        filter_ast: FilterNode | None = None,
        pushed_keys_out: list[frozenset[str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Get chunks connected to the given entities via MENTIONED_IN.

        Args:
            entity_ids: List of entity IDs to find chunks for
            namespace_id: Namespace to search within
            temporal_filter: Optional temporal constraints
            prefer_current: When True, prefer entities whose valid_until
                has not passed (for temporal queries). Entities without
                valid_until are kept (NULL = still valid).
            limit: Maximum chunks to return
            filter_ast: Canonical recall-filter AST. The system-key slice
                (the projected document/date keys on the Chunk node) is
                pushed down here as an extra ``WHERE`` predicate; any metadata
                leaf is not expressible in Cypher and is left to the engine's
                in-memory post-filter. A predicate this backend cannot honor at
                all raises :class:`RecallFilterUnsupportedError`.
            pushed_keys_out: Optional sink for the pushdown report. Receives the
                consumed-key set of the compile whose predicate was actually
                spliced into the executed ``WHERE`` — so the report derives from
                the executing compile, not a parallel one. Left untouched when
                nothing pushed (no filter, or a metadata-only filter that
                consumed zero leaves). Carries a bare ``frozenset[str]`` so this
                graph-store layer stays free of filter-report types.

        Returns:
            List of chunk dicts with entity connection info.
            Returns ``[]`` on query timeout when ``query_timeout`` is set.
        """
        if not entity_ids:
            return []

        # Build temporal filter conditions
        temporal_conditions = []
        params: dict[str, Any] = {
            "entity_ids": [str(eid) for eid in entity_ids],
            "namespace_id": str(namespace_id),
            "limit": limit,
        }

        if temporal_filter:
            if temporal_filter.occurred_after:
                temporal_conditions.append("c.occurred_at >= $occurred_after")
                params["occurred_after"] = temporal_filter.occurred_after.isoformat()
            if temporal_filter.occurred_before:
                temporal_conditions.append("c.occurred_at < $occurred_before")
                params["occurred_before"] = temporal_filter.occurred_before.isoformat()
            if temporal_filter.source_system:
                temporal_conditions.append("c.source_system = $source_system")
                params["source_system"] = temporal_filter.source_system
            if temporal_filter.author:
                temporal_conditions.append("c.author = $author")
                params["author"] = temporal_filter.author
            if temporal_filter.channel:
                temporal_conditions.append("c.channel = $channel")
                params["channel"] = temporal_filter.channel

        # For temporal queries, prefer entities whose validity hasn't expired.
        # valid_until is now a native ZONED DATETIME (#1472), so compare it
        # directly against the server clock - a cast-free indexed range.
        if prefer_current:
            temporal_conditions.append("(e.valid_until IS NULL OR e.valid_until > datetime())")

        # Push the caller filter's system-key slice down to Cypher. The MATCH
        # below binds the chunk as ``c`` (the compiler's default node variable),
        # so the compiled predicate references ``c.<key>`` directly. ``"split"``
        # mode means a leaf this backend cannot express (any metadata predicate)
        # is omitted from ``consumed_keys`` and left to the engine's full-AST
        # in-memory post-filter, which enforces every leaf — so this call does
        # not raise.
        if filter_ast is not None:
            from khora.filter.compilers.cypher import compile_cypher
            from khora.filter.execute import build_compile_context

            compiled = compile_cypher(
                filter_ast,
                build_compile_context("Chunk", table_alias="c", on_unsupported="split"),
            )
            # Only splice when at least one leaf actually pushed down. A
            # metadata-only filter consumes nothing here (every leaf collapses to
            # a non-constraining ``true``); leave it entirely to the post-filter.
            if compiled.consumed_keys:
                temporal_conditions.append(compiled.predicate)
                params.update(compiled.params)
                if pushed_keys_out is not None:
                    pushed_keys_out.append(compiled.consumed_keys)

        where_clause = ""
        if temporal_conditions:
            where_clause = "AND " + " AND ".join(temporal_conditions)

        order_clause = (
            "ORDER BY c.occurred_at DESC, total_mentions DESC" if temporal_sort else "ORDER BY total_mentions DESC"
        )

        query = f"""
        MATCH (e:Entity)-[r:MENTIONED_IN]->(c:Chunk)
        WHERE e.id IN $entity_ids
        AND c.namespace_id = $namespace_id
        {where_clause}
        RETURN c.id AS chunk_id,
               c.content AS content,
               c.document_id AS document_id,
               c.occurred_at AS occurred_at,
               c.source_timestamp AS source_timestamp,
               c.metadata AS metadata,
               c.chunker_info AS chunker_info,
               collect(DISTINCT e.id) AS entity_ids,
               sum(r.mention_count) AS total_mentions
        {order_clause}
        LIMIT $limit
        """

        async def _work(tx):
            result = await tx.run(query, **params)
            return [record.data() async for record in result]

        if self._timed_unit_of_work is not None:
            _work = self._timed_unit_of_work(_work)

        try:
            async with self._session() as session:
                records = await session.execute_read(_work)
        except ClientError as exc:
            if exc.code in _NEO4J_TIMEOUT_CODES:
                timeout = self._query_timeout
                ns = namespace_id
                n = len(entity_ids)
                with trace_span(
                    "khora.neo4j.get_chunks_by_entities.timeout",
                    timeout_s=timeout,
                    entity_count=n,
                    namespace_id=str(ns),
                    code=exc.code,
                    timeout_occurred=True,
                ):
                    pass
                logger.warning(
                    "Neo4j get_chunks_by_entities timed out after {timeout}s "
                    "(namespace_id={ns}, entity_count={n}, code={code}); "
                    "returning empty list",
                    timeout=timeout,
                    ns=ns,
                    n=n,
                    code=exc.code,
                    timeout_occurred=True,
                )
                return []
            raise

        # Deserialize metadata from JSON string back to dict
        for record in records:
            if "metadata" in record:
                record["metadata"] = deserialize_dict(record["metadata"])

        return records

    @trace(
        "khora.neo4j.get_entity_neighborhoods",
        include={"entity_ids", "depth"},
        result=lambda r: {"result_count": len(r)},
    )
    async def get_entity_neighborhoods(
        self,
        entity_ids: list[UUID],
        namespace_id: UUID,
        *,
        depth: int = 2,
        limit_per_entity: int = 20,
        prefer_current: bool = False,
        hop_limit: int = 200,
        degradations: list[Degradation] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Get neighborhood of entities via bounded per-hop graph expansion.

        OPTIMIZATION: Uses a single Cypher query with UNWIND pattern to fetch
        all entity neighborhoods in one database round-trip. The query
        (built by :func:`_build_neighborhood_query`, issue #1419):
        1. Uses IN clause for batch entity matching (index-backed)
        2. Expands each neighborhood hop by hop with a visited-set +
           ``collect(DISTINCT ...)`` frontier - linear in reachable nodes,
           unlike the former all-paths ``[*1..depth]`` enumeration whose cost
           was exponential in graph density
        3. Caps each hop's new frontier at ``hop_limit`` nodes per source
           entity so hub nodes cannot blow up a hop
        4. Limits per-entity results to avoid explosion

        Args:
            entity_ids: Starting entity IDs
            namespace_id: Namespace constraint
            depth: Maximum traversal depth (1-4)
            limit_per_entity: Max related entities per starting entity
            prefer_current: When True, filter out entities and relationships whose valid_until
                has passed (for STATE_QUERY/RECENCY/CHANGE temporal categories).
                Entities and relationships without valid_until are kept (NULL = still valid).
            hop_limit: Per-hop frontier cap per source entity (default 200).
                Bounds fan-out at hub nodes; each hop expands at most this
                many newly discovered nodes. On graphs where a hop exceeds
                the cap, overflow nodes are dropped for that hop (they may
                still be rediscovered later at a larger distance). The
                default comfortably exceeds ``limit_per_entity`` (20 at the
                retriever call site) so realistic neighborhoods are unaffected.
            degradations: Optional ADR-001 sink. When the query times out
                (``query_timeout``), a structured ``Degradation``
                (component=``vectorcypher.cypher_expand``,
                reason=``neo4j_timeout``) is appended so the dropped graph
                channel is observable instead of a silent ``{}``.

        Returns:
            Dict mapping entity_id -> list of related entity info
        """
        if not entity_ids:
            return {}

        depth = min(max(1, depth), 4)  # Clamp to 1-4

        query = _build_neighborhood_query(depth, prefer_current)

        async def _work(tx):
            result = await tx.run(
                query,
                entity_ids=[str(eid) for eid in entity_ids],
                namespace_id=str(namespace_id),
                limit=limit_per_entity,
                hop_limit=hop_limit,
            )
            return [record.data() async for record in result]

        # Apply the configured transaction timeout via the pre-bound
        # unit_of_work decorator (hoisted in __init__). The Neo4j Python
        # driver's `tx.run()` does NOT accept a timeout kwarg — timeouts
        # must be set at transaction-begin time, which unit_of_work does
        # by attaching metadata the session reads when starting the
        # managed tx. Hoisting means we pay the factory cost once per
        # DualNodeManager rather than once per query.
        if self._timed_unit_of_work is not None:
            _work = self._timed_unit_of_work(_work)

        try:
            async with self._session() as session:
                records = await session.execute_read(_work)
        except ClientError as exc:
            # Match only the two known transaction-timeout codes (explicit
            # tuple, not a prefix match) so we don't swallow syntax errors,
            # auth failures, or constraint violations that are also ClientError.
            if exc.code in _NEO4J_TIMEOUT_CODES:
                # Emit a dedicated child span so operators can alert on
                # timeout frequency in Logfire/OTEL via a span-name filter
                # (the ".timeout" suffix). The parent @trace span records
                # result_count=0 on timeout, but a distinct span carries
                # the timeout-specific attributes ops needs for dashboards
                # (configured timeout, entity count, depth, error code).
                with trace_span(
                    "khora.neo4j.get_entity_neighborhoods.timeout",
                    timeout_s=self._query_timeout,
                    entity_count=len(entity_ids),
                    depth=depth,
                    code=exc.code,
                    namespace_id=str(namespace_id),
                    timeout_occurred=True,
                ):
                    pass  # attributes set via kwargs; no inner work
                logger.warning(
                    "Neo4j get_entity_neighborhoods timed out after {timeout}s "
                    "(namespace_id={ns}, entity_count={n}, depth={d}, code={code}); "
                    "returning empty neighborhood",
                    timeout=self._query_timeout,
                    ns=namespace_id,
                    n=len(entity_ids),
                    d=depth,
                    code=exc.code,
                    timeout_occurred=True,
                )
                # ADR-001 (#1419): the silent `{}` here is how the all-paths
                # blowup stayed hidden - the timeout masked it by dropping
                # the whole graph channel with no machine-readable signal.
                # Record a structured Degradation on the caller's sink and
                # bump the shared cypher_expand degradation counter.
                # Lazy import: retriever imports this module at top level,
                # so importing the counter eagerly would be circular.
                from khora.engines.vectorcypher.retriever import (
                    _CYPHER_EXPAND_DEGRADED_COUNTER,
                )

                _CYPHER_EXPAND_DEGRADED_COUNTER.add(1, attributes={"reason": "neo4j_timeout"})
                if degradations is not None:
                    degradations.append(
                        Degradation(
                            component="vectorcypher.cypher_expand",
                            reason="neo4j_timeout",
                            detail=(
                                f"neighborhood expansion timed out after "
                                f"{self._query_timeout}s (entity_count={len(entity_ids)}, "
                                f"depth={depth}); graph channel returned empty"
                            ),
                            exception=type(exc).__name__,
                        )
                    )
                return {}
            raise

        return {record["source_id"]: record["related_entities"] for record in records}

    @trace(
        "khora.neo4j.get_relationships_between",
        include={"entity_ids", "namespace_id"},
        result=lambda r: {"relationship_count": len(r)},
    )
    async def get_relationships_between(
        self,
        entity_ids: list[str],
        namespace_id: str,
        *,
        limit: int = 90,
    ) -> list[dict[str, Any]]:
        """Get relationships between a set of entities.

        Finds all directed relationships where both source and target
        are in the given entity_ids set, within the same namespace.

        Args:
            entity_ids: Entity IDs (strings) to find relationships between
            namespace_id: Namespace constraint (string)
            limit: Maximum number of relationships to return

        Returns:
            List of relationship dicts with id, source_entity_id,
            target_entity_id, relationship_type, description, confidence, weight,
            source_document_ids, source_chunk_ids.
            Returns ``[]`` on query timeout when ``query_timeout`` is set.
        """
        if len(entity_ids) < 2:
            return []

        query = """
        UNWIND $entity_ids AS sid
        MATCH (source:Entity {id: sid, namespace_id: $namespace_id})-[r]->(target:Entity)
        WHERE target.namespace_id = $namespace_id
          AND target.id IN $entity_ids
          AND source.id <> target.id
        RETURN DISTINCT r.id AS id, source.id AS source_entity_id, target.id AS target_entity_id,
               type(r) AS relationship_type, r.description AS description,
               r.confidence AS confidence, r.weight AS weight,
               r.source_document_ids AS source_document_ids,
               r.source_chunk_ids AS source_chunk_ids
        ORDER BY (CASE WHEN relationship_type = 'ASSOCIATED_WITH' THEN 1 ELSE 0 END) ASC,
                 coalesce(confidence, 0.0) DESC
        LIMIT $limit
        """

        async def _work(tx):
            result = await tx.run(
                query,
                entity_ids=entity_ids,
                namespace_id=namespace_id,
                limit=limit,
            )
            return [record.data() async for record in result]

        if self._timed_unit_of_work is not None:
            _work = self._timed_unit_of_work(_work)

        try:
            async with self._session() as session:
                return await session.execute_read(_work)
        except ClientError as exc:
            if exc.code in _NEO4J_TIMEOUT_CODES:
                with trace_span(
                    "khora.neo4j.get_relationships_between.timeout",
                    timeout_s=self._query_timeout,
                    entity_count=len(entity_ids),
                    namespace_id=namespace_id,
                    code=exc.code,
                    timeout_occurred=True,
                ):
                    pass
                logger.warning(
                    "Neo4j get_relationships_between timed out after {timeout}s "
                    "(namespace_id={ns}, entity_count={n}, code={code}); "
                    "returning empty list",
                    timeout=self._query_timeout,
                    ns=namespace_id,
                    n=len(entity_ids),
                    code=exc.code,
                    timeout_occurred=True,
                )
                return []
            raise

    @trace(
        "khora.neo4j.get_temporal_chunks",
        include={"entity_ids", "namespace_id"},
        result=lambda r: {"chunk_count": len(r)},
    )
    async def get_temporal_chunks(
        self,
        namespace_id: UUID,
        entity_ids: list[UUID],
        *,
        after: datetime | None = None,
        before: datetime | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get chunks connected to entities via MENTIONED_IN within a time range.

        Matches (:Entity)-[:MENTIONED_IN]->(:Chunk) and filters chunks by
        occurred_at or created_at falling within the given time window.

        Args:
            namespace_id: Namespace to restrict results to
            entity_ids: Entity IDs whose connected chunks to retrieve
            after: Include only chunks with occurred_at/created_at >= this value
            before: Include only chunks with occurred_at/created_at <= this value
            limit: Maximum chunks to return

        Returns:
            List of chunk property dicts with entity connection info.
            Returns ``[]`` on query timeout when ``query_timeout`` is set.
        """
        if not entity_ids:
            return []

        params: dict[str, Any] = {
            "entity_ids": [str(eid) for eid in entity_ids],
            "namespace_id": str(namespace_id),
            "limit": limit,
        }

        conditions: list[str] = []
        if after is not None:
            conditions.append(
                "(coalesce(c.occurred_at, c.created_at) IS NULL OR coalesce(c.occurred_at, c.created_at) >= $after)"
            )
            params["after"] = after.isoformat()
        if before is not None:
            conditions.append(
                "(coalesce(c.occurred_at, c.created_at) IS NULL OR coalesce(c.occurred_at, c.created_at) <= $before)"
            )
            params["before"] = before.isoformat()

        where_extra = ""
        if conditions:
            where_extra = "\nAND " + "\nAND ".join(conditions)

        query = f"""
        MATCH (e:Entity)-[r:MENTIONED_IN]->(c:Chunk)
        WHERE e.id IN $entity_ids
        AND c.namespace_id = $namespace_id{where_extra}
        RETURN c.id AS chunk_id,
               c.content AS content,
               c.document_id AS document_id,
               c.occurred_at AS occurred_at,
               c.created_at AS created_at,
               c.metadata AS metadata,
               collect(DISTINCT e.id) AS entity_ids,
               sum(r.mention_count) AS total_mentions
        ORDER BY coalesce(c.occurred_at, c.created_at) DESC
        LIMIT $limit
        """

        async def _work(tx):
            result = await tx.run(query, **params)
            return [record.data() async for record in result]

        if self._timed_unit_of_work is not None:
            _work = self._timed_unit_of_work(_work)

        try:
            async with self._session() as session:
                records = await session.execute_read(_work)
        except ClientError as exc:
            if exc.code in _NEO4J_TIMEOUT_CODES:
                with trace_span(
                    "khora.neo4j.get_temporal_chunks.timeout",
                    timeout_s=self._query_timeout,
                    entity_count=len(entity_ids),
                    namespace_id=str(namespace_id),
                    code=exc.code,
                    timeout_occurred=True,
                ):
                    pass
                logger.warning(
                    "Neo4j get_temporal_chunks timed out after {timeout}s "
                    "(namespace_id={ns}, entity_count={n}, code={code}); "
                    "returning empty list",
                    timeout=self._query_timeout,
                    ns=namespace_id,
                    n=len(entity_ids),
                    code=exc.code,
                    timeout_occurred=True,
                )
                return []
            raise

        for record in records:
            if "metadata" in record:
                record["metadata"] = deserialize_dict(record["metadata"])

        return records

    @trace(
        "khora.neo4j.get_entity_channels",
        include={"entity_ids", "namespace_id"},
        result=lambda r: {"channel_count": len(r)},
    )
    async def get_entity_channels(
        self,
        entity_ids: list[str],
        namespace_id: str,
    ) -> list[str]:
        """Get distinct session channels from entities' connected chunks.

        Queries Neo4j for all Chunk nodes connected to the given entities
        via MENTIONED_IN relationships and returns the distinct non-null
        channel values.

        Args:
            entity_ids: Entity IDs (strings) to find channels for
            namespace_id: Namespace constraint (string)

        Returns:
            List of distinct channel strings (never contains None).
            Returns ``[]`` on query timeout when ``query_timeout`` is set.
        """
        if not entity_ids:
            return []

        query = """
        MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
        WHERE e.id IN $entity_ids
          AND c.namespace_id = $namespace_id
          AND c.channel IS NOT NULL
        RETURN DISTINCT c.channel AS channel
        """

        async def _work(tx):
            result = await tx.run(
                query,
                entity_ids=entity_ids,
                namespace_id=namespace_id,
            )
            return [record["channel"] async for record in result]

        if self._timed_unit_of_work is not None:
            _work = self._timed_unit_of_work(_work)

        try:
            async with self._session() as session:
                channels = await session.execute_read(_work)
        except ClientError as exc:
            if exc.code in _NEO4J_TIMEOUT_CODES:
                with trace_span(
                    "khora.neo4j.get_entity_channels.timeout",
                    timeout_s=self._query_timeout,
                    entity_count=len(entity_ids),
                    namespace_id=namespace_id,
                    code=exc.code,
                    timeout_occurred=True,
                ):
                    pass
                logger.warning(
                    "Neo4j get_entity_channels timed out after {timeout}s "
                    "(namespace_id={ns}, entity_count={n}, code={code}); "
                    "returning empty list",
                    timeout=self._query_timeout,
                    ns=namespace_id,
                    n=len(entity_ids),
                    code=exc.code,
                    timeout_occurred=True,
                )
                return []
            raise

        logger.debug(f"Found {len(channels)} distinct channels for {len(entity_ids)} entities")
        return channels

    async def delete_chunks_by_document(
        self,
        document_id: UUID,
        namespace_id: UUID,
    ) -> int:
        """Delete all Chunk nodes for a document.

        Also removes MENTIONED_IN and AT_TIME relationships.

        Args:
            document_id: Document ID
            namespace_id: Namespace ID

        Returns:
            Number of chunks deleted
        """
        query = """
        MATCH (c:Chunk {document_id: $document_id, namespace_id: $namespace_id})
        DETACH DELETE c
        RETURN count(c) AS deleted
        """

        async with self._session() as session:

            async def _work(tx):
                result = await tx.run(
                    query,
                    document_id=str(document_id),
                    namespace_id=str(namespace_id),
                )
                record = await result.single()
                return record["deleted"] if record else 0

            deleted = await session.execute_write(_work)
        logger.debug(f"Deleted {deleted} Chunk nodes for document {document_id}")
        return deleted


__all__ = [
    "ChunkNode",
    "DualNodeManager",
    "EntityChunkLink",
]
