"""VectorCypher engine - hybrid vector+graph retrieval with temporal support.

This engine implements the VectorCypher retrieval paradigm inspired by Graph RAG 2026:
- Vector search to find entry entities (pgvector)
- Cypher traversal to expand relationships (Neo4j)
- Query routing to optimize simple vs complex queries
- HippoRAG 2 dual-node architecture (Entity + Chunk nodes)
- Skeleton-based construction (KET-RAG) - full KG extraction for top chunks (default 70%)
- Bi-temporal edges (Graphiti-style) - occurred_at vs ingested_at with invalidation

Target: Sub-300ms P95 for simple queries, sub-800ms for complex multi-hop queries.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time as _time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar, Literal
from uuid import UUID

from loguru import logger
from neo4j import AsyncGraphDatabase
from sqlalchemy.exc import IntegrityError

from khora.config import KhoraConfig, LiteLLMConfig
from khora.core.diagnostics import Degradation
from khora.core.models import (
    Chunk,
    Document,
    Entity,
    MemoryNamespace,
    Relationship,
)
from khora.core.models.recall import (
    DocumentProjection,
    RecallChunk,
    RecallEntity,
    RecallRelationship,
)
from khora.core.recall_abstention import compute_abstention_signals
from khora.engines._forget_cascade import cascade_forget_extraction
from khora.engines._stats import gather_counts
from khora.engines._storage_config import build_storage_config
from khora.engines.skeleton.backends import TemporalChunk, TemporalFilter, create_temporal_store
from khora.engines.skeleton.skeleton import SkeletonIndexer
from khora.exceptions import EngineCapabilityError
from khora.extraction.embedders import LiteLLMEmbedder
from khora.khora import BatchResult, RecallResult, RememberResult, Stats
from khora.query import SearchMode
from khora.storage import StorageConfig, create_storage_coordinator
from khora.telemetry import trace, trace_span
from khora.telemetry.metrics import metric_counter, metric_histogram

from .dual_nodes import DualNodeManager, EntityChunkLink
from .retriever import RetrieverConfig, VectorCypherRetriever
from .router import QueryComplexityRouter, RouterConfig
from .temporal_detection import TemporalCategory, TemporalDetector, TemporalSignal

if TYPE_CHECKING:
    from neo4j import AsyncDriver

    from khora.extraction.chunkers import ChunkStrategy
    from khora.extraction.skills import ExpertiseConfig
    from khora.khora import _GlobalChunkSemaphore
    from khora.storage import StorageCoordinator


_VC_ABSTENTION_SIGNAL_COUNTER = metric_counter(
    "khora.vectorcypher.abstention_signal",
    description="VectorCypher abstention signal firings, by signal name.",
)
_VC_ABSTENTION_COMBINED_SCORE_HISTOGRAM = metric_histogram(
    "khora.vectorcypher.abstention_combined_score",
    unit="1",
    description="VectorCypher abstention combined-score (0.0=confident, 1.0=should-abstain).",
)


def _ensure_tags(value: Any) -> list[str]:
    """Coerce tags to a list — handles JSON strings from PostgreSQL metadata."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        return [value] if value else []
    return []


def _coerce_session_id_from_metadata(metadata: dict[str, Any] | None) -> UUID | None:
    """Pull ``session_id`` out of a metadata dict and coerce to UUID (#620).

    ``Khora.remember`` stamps ``session_id`` into ``metadata`` so engines
    that build :class:`Document` directly (rather than via
    ``pipelines.flows.ingest.stage_document``) can still surface it as a
    first-class column. Malformed values fall back to ``None`` rather than
    crashing ingestion.
    """
    if not metadata:
        return None
    value = metadata.get("session_id")
    if value is None or value == "":
        return None
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (ValueError, TypeError, AttributeError):
        return None


def _build_remember_metadata(extraction_diagnostics: dict[str, Any] | None) -> dict[str, Any]:
    """Project extraction diagnostics onto a ``RememberResult.metadata`` payload.

    #889: surface ``extraction_errors`` (count) and any ADR-001
    ``degradations`` list collected by ``extract_entities``. Returns an
    empty dict when no errors were observed so the happy path leaves
    ``RememberResult.metadata`` empty (matching pre-fix behavior).
    """
    if not extraction_diagnostics:
        return {}
    errors = extraction_diagnostics.get("extraction_errors", 0)
    degradations = extraction_diagnostics.get("degradations") or []
    if not errors and not degradations:
        return {}
    metadata: dict[str, Any] = {}
    if errors:
        metadata["extraction_errors"] = int(errors)
    if degradations:
        metadata["degradations"] = list(degradations)
    return metadata


_MAX_COOCCURRENCE_PER_CHUNK = 15


def _bfs_distances_from(seed: UUID, relationships: list[Any]) -> dict[UUID, int]:
    """BFS hop distance from ``seed`` over an undirected adjacency built from
    ``relationships``.

    Used by ``find_related_entities`` on graph-only backends where the
    backend's ``get_neighborhood`` returns entities + edges but no per-entity
    distance. Edges are treated as undirected (both incoming and outgoing
    expansions are considered, matching the recursive walk in the underlying
    backends).
    """
    if not relationships:
        return {}
    adj: dict[UUID, list[UUID]] = {}
    for rel in relationships:
        # Tolerate both Relationship dataclasses and dict-shaped rows that
        # some backends (e.g. surrealdb) may return.
        src = getattr(rel, "source_entity_id", None)
        tgt = getattr(rel, "target_entity_id", None)
        if src is None and isinstance(rel, dict):
            src = rel.get("source_entity_id") or rel.get("in") or rel.get("from")
            tgt = rel.get("target_entity_id") or rel.get("out") or rel.get("to")
        if src is None or tgt is None:
            continue
        adj.setdefault(src, []).append(tgt)
        adj.setdefault(tgt, []).append(src)

    distances: dict[UUID, int] = {seed: 0}
    frontier = [seed]
    depth = 0
    while frontier:
        depth += 1
        next_frontier: list[UUID] = []
        for node in frontier:
            for neighbor in adj.get(node, ()):
                if neighbor not in distances:
                    distances[neighbor] = depth
                    next_frontier.append(neighbor)
        frontier = next_frontier
    return distances


def _build_cooccurrence_relationships(
    entities: list[Entity],
    chunks: list[Chunk],
    namespace_id: UUID,
    existing_relationships: list[Relationship],
) -> list[Relationship]:
    """Create ASSOCIATED_WITH edges between entities sharing the same chunk.

    This mirrors the co-occurrence logic in ``pipelines/flows/ingest.py:369-405``
    to ensure VectorCypher builds equally dense graphs.  Capped at
    ``_MAX_COOCCURRENCE_PER_CHUNK`` per chunk to prevent quadratic explosion.

    ``chunks`` carries the (chunk_id → document_id) mapping used to stamp
    provenance (``source_chunk_ids`` / ``source_document_ids``) on each new
    edge.
    """
    # Build chunk → entities map
    chunk_entity_map: dict[UUID, list[Entity]] = {}
    for entity in entities:
        for chunk_id in entity.source_chunk_ids:
            chunk_entity_map.setdefault(chunk_id, []).append(entity)

    chunk_to_doc: dict[UUID, UUID] = {c.id: c.document_id for c in chunks}

    # Collect existing pairs to avoid duplicates
    existing_pairs: set[tuple[UUID, UUID]] = set()
    for r in existing_relationships:
        pair = (min(r.source_entity_id, r.target_entity_id), max(r.source_entity_id, r.target_entity_id))
        existing_pairs.add(pair)

    cooccurrence_rels: list[Relationship] = []
    for chunk_id, chunk_entities in chunk_entity_map.items():
        if len(chunk_entities) < 2:
            continue
        document_id = chunk_to_doc.get(chunk_id)
        if document_id is None:
            logger.warning(
                f"VectorCypher co-occurrence: chunk {chunk_id} missing from chunks list; "
                "relationship will have empty source_document_ids"
            )
            source_document_ids: list[UUID] = []
        else:
            source_document_ids = [document_id]
        chunk_count = 0
        for i, e1 in enumerate(chunk_entities):
            for e2 in chunk_entities[i + 1 :]:
                pair = (min(e1.id, e2.id), max(e1.id, e2.id))
                if pair in existing_pairs:
                    continue
                existing_pairs.add(pair)
                cooccurrence_rels.append(
                    Relationship(
                        source_entity_id=e1.id,
                        target_entity_id=e2.id,
                        relationship_type="ASSOCIATED_WITH",
                        namespace_id=namespace_id,
                        description="Co-occurs in same chunk",
                        properties={},
                        source_chunk_ids=[chunk_id],
                        source_document_ids=list(source_document_ids),
                        confidence=0.4,
                    )
                )
                chunk_count += 1
                if chunk_count >= _MAX_COOCCURRENCE_PER_CHUNK:
                    break
            if chunk_count >= _MAX_COOCCURRENCE_PER_CHUNK:
                break

    if cooccurrence_rels:
        logger.debug(f"VectorCypher: added {len(cooccurrence_rels)} co-occurrence edges")

    return cooccurrence_rels


@dataclass(slots=True)
class ExtractionQualityMetrics:
    """Track extraction quality for monitoring."""

    total_chunks: int = 0
    chunks_with_entities: int = 0
    total_entities: int = 0
    total_relationships: int = 0
    avg_entities_per_chunk: float = 0.0
    avg_confidence: float = 0.0
    entity_type_distribution: dict[str, int] = field(default_factory=dict)

    def compute_averages(self) -> None:
        """Compute average metrics from totals."""
        if self.total_chunks > 0:
            self.avg_entities_per_chunk = self.total_entities / self.total_chunks


@dataclass(slots=True)
class VectorCypherConfig:
    """VectorCypher-specific configuration."""

    # Routing
    routing_enabled: bool = True
    routing_use_llm: bool = False

    # Skeleton indexing
    skeleton_core_ratio: float = 0.70  # 70% get full KG extraction (increased for denser graphs)
    conversation_skeleton_ratio: float = 0.90  # Higher ratio for conversation batches

    # Graph traversal
    graph_default_depth: int = 2
    graph_max_depth: int = 4
    graph_max_entry_entities: int = 10

    # Fusion
    fusion_rrf_k: int = 60
    fusion_vector_weight: float = 0.6
    fusion_graph_weight: float = 0.4
    fusion_simple_vector_weight: float = 0.8
    fusion_simple_graph_weight: float = 0.2
    fusion_complex_vector_weight: float = 0.4
    fusion_complex_graph_weight: float = 0.6

    # Temporal
    temporal_recency_weight: float = 0.2
    temporal_recency_decay_days: int = 30
    recency_decay_type: str = "exponential"  # "linear" or "exponential"

    # Extraction concurrency (aligned with ingest pipeline's default of 20)
    max_concurrent_extractions: int = 20

    # Maximum texts per LLM extraction batch. Lower values reduce output token
    # requirements and avoid timeouts with strict JSON schema constrained decoding.
    extraction_batch_size: int = 5

    # Maximum number of chunks processed through stages 2–6 simultaneously.
    # Primary memory control surface: chunks are ~2 KB each (512 tokens), so chunk
    # count directly correlates with peak memory. None = process all chunks at once
    # (current behavior, backward-compatible).
    max_chunks_in_flight: int | None = None

    def __post_init__(self) -> None:
        if self.max_chunks_in_flight is not None and self.max_chunks_in_flight < 1:
            raise ValueError(f"max_chunks_in_flight must be >= 1, got {self.max_chunks_in_flight}")

    # Streaming pipeline (A-1: batch entity storage across documents)
    streaming_pipeline: bool = True
    enable_smart_resolution: bool = True

    # Skip LLM entity extraction for short messages (conversation batches).
    # Messages with all chunks ≤ this token count rely on BM25 + vector search instead.
    min_extraction_tokens: int = 50

    # Lazy entity expansion (recovers graph signal for non-core chunks)
    lazy_entity_expansion: bool = True

    # Store extracted events as EVENT entities with PARTICIPATED_IN relationships
    store_events: bool = True

    # Search thresholds
    fusion_hybrid_alpha: float = 0.7
    retriever_min_entity_similarity: float = 0.3

    # BM25 channel (independent full-text search alongside vector + graph)
    enable_bm25_channel: bool = False
    bm25_weight: float = 0.3
    bm25_top_k: int = 50

    # Session-aware parallel retrieval for cross-session temporal queries.
    # Only activates when: Neo4j is connected, query is temporal, and entry
    # entities span multiple sessions/channels.
    enable_session_aware_search: bool = True

    # Cross-encoder reranking
    enable_reranking: bool = False
    reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranking_top_n: int = 50  # How many candidates to feed to the cross-encoder
    reranking_blend_weight: float = 0.7  # Rerank vs original score blend (passed to reranker)

    # LLM reranking (applied after cross-encoder, only for temporal queries).
    #
    # When ``enable_llm_reranking=True``, the retriever still applies a
    # version-metadata precondition by default — if no chunk in the top
    # candidates carries ``metadata["version"]`` (the enterprise temporal
    # signal), the LLM rerank step is skipped because PR #364 showed it
    # regresses MRR on conversational benchmarks (LongMemEval / LoCoMo).
    # Set ``llm_reranking_mode="always"`` to disable that precondition and
    # force LLM rerank on every temporal query (the "decisive winner"
    # latency optimization still applies).
    enable_llm_reranking: bool = False
    llm_reranking_model: str = "gpt-4o-mini"
    llm_reranking_top_n: int = 5
    llm_reranking_confidence_threshold: float = 0.1
    # ``"auto"`` (default) — gate LLM rerank on the version-metadata
    # precondition (current behavior). The first time the gate fires for
    # a given namespace, a one-time WARNING is emitted so users who opted
    # in to ``enable_llm_reranking=True`` discover why the LLM is not
    # being invoked.
    # ``"always"`` — skip the version-metadata precondition; LLM rerank
    # runs on every temporal query (subject to the decisive-winner skip).
    llm_reranking_mode: Literal["auto", "always"] = "auto"


class VectorCypherEngine:
    """VectorCypher engine - hybrid vector+graph retrieval with temporal support.

    Key features:
    - Dual retrieval: Vector similarity (pgvector) + graph traversal (Neo4j)
    - Smart routing: Route queries to optimal path (simple vs complex)
    - Skeleton indexing: Full KG extraction only for core chunks (configurable ratio)
    - Bi-temporal model: Track occurred_at vs ingested_at
    - RRF fusion: Combine vector and graph scores

    Requirements:
    - PostgreSQL with pgvector extension
    - Neo4j (required, not optional like in SkeletonEngine)

    Usage:
        engine = VectorCypherEngine(config)
        await engine.connect()

        # Store with temporal context
        result = await engine.remember(
            "Meeting with John about Q1 planning",
            namespace_id,
            occurred_at=datetime(2024, 1, 15),
        )

        # Retrieve with hybrid search
        result = await engine.recall(
            "What did we discuss with John about planning?",
            namespace_id,
            graph_depth=2,
        )
    """

    # VectorCypher honestly implements all five modes. KEYWORD / VECTOR /
    # GRAPH skip the unused channels at retrieve-time; HYBRID is the default
    # (vector-weighted RRF); ALL is HYBRID with ``hybrid_alpha=0.5``
    # (balanced fusion). See ``recall`` below.
    supported_modes: ClassVar[frozenset[SearchMode]] = frozenset(
        {
            SearchMode.VECTOR,
            SearchMode.GRAPH,
            SearchMode.HYBRID,
            SearchMode.ALL,
            SearchMode.KEYWORD,
        }
    )

    def __init__(
        self,
        config: KhoraConfig,
        *,
        storage_config: StorageConfig | None = None,
        vectorcypher_config: VectorCypherConfig | None = None,
    ) -> None:
        """Initialize the VectorCypher engine.

        Args:
            config: KhoraConfig instance
            storage_config: Storage configuration (deprecated, derived from config)
            vectorcypher_config: VectorCypher-specific configuration
        """
        self._config = config
        self._vc_config = vectorcypher_config or VectorCypherConfig()

        # Build storage config (shared helper handles SurrealDB, pool_pre_ping, etc.)
        self._storage_config = storage_config or build_storage_config(config)

        # Component instances (initialized on connect)
        self._storage: StorageCoordinator | None = None
        self._temporal_store = None
        self._neo4j_driver: AsyncDriver | None = None
        self._embedder: LiteLLMEmbedder | None = None
        self._retriever: VectorCypherRetriever | None = None
        self._dual_nodes: DualNodeManager | None = None
        self._router: QueryComplexityRouter | None = None
        self._connected = False

    @staticmethod
    def _neo4j_driver_kwargs(neo4j_cfg: Any) -> dict[str, Any]:
        """Extract Neo4j driver kwargs from config.

        Centralises the config → driver mapping so new fields cannot be
        silently omitted.
        """

        def _get(attr: str, default: Any) -> Any:
            return getattr(neo4j_cfg, attr, default) if neo4j_cfg else default

        return {
            "max_connection_pool_size": _get("max_connection_pool_size", 100),
            "max_connection_lifetime": _get("max_connection_lifetime", 900),
            "liveness_check_timeout": _get("liveness_check_timeout", 30.0),
            "connection_acquisition_timeout": _get("connection_acquisition_timeout", 60.0),
        }

    async def connect(self) -> None:
        """Connect to all storage backends."""
        if self._connected:
            return

        logger.info("Connecting VectorCypher engine...")

        # Detect non-Neo4j backends — skips Neo4j and uses the unified
        # backend's graph adapter for Cypher-equivalent operations.
        backend = getattr(self._config.storage, "backend", "postgres")
        is_surrealdb = backend == "surrealdb"
        is_sqlite_lance = backend == "sqlite_lance"
        skip_neo4j = is_surrealdb or is_sqlite_lance
        neo4j_database = "neo4j"
        neo4j_query_timeout: float | None = 5.0

        if not skip_neo4j:
            # Connect to Neo4j (required for VectorCypher with traditional stack)
            neo4j_url = self._config.get_neo4j_url()
            if not neo4j_url:
                raise ValueError(
                    "Neo4j URL is required for VectorCypher engine. Set KHORA_NEO4J_URL or configure graph_config."
                )

            neo4j_cfg = self._config.get_graph_config()
            neo4j_query_timeout = getattr(neo4j_cfg, "query_timeout", 5.0) if neo4j_cfg else 5.0
            driver_kwargs = self._neo4j_driver_kwargs(neo4j_cfg)
            self._neo4j_driver = AsyncGraphDatabase.driver(
                neo4j_url,
                auth=(self._config.get_neo4j_user(), self._config.get_neo4j_password()),
                **driver_kwargs,
                keep_alive=True,
            )
            await self._neo4j_driver.verify_connectivity()

        # Create and connect relational storage
        self._storage = create_storage_coordinator(self._storage_config)

        if not skip_neo4j:
            # Share the Neo4j driver so only one connection pool is used
            from khora.storage.backends.neo4j import Neo4jBackend

            neo4j_cfg = self._config.get_graph_config()
            neo4j_database = self._config.get_neo4j_database() or "neo4j"
            if self._storage._graph is not None:
                # Route through the public attr so __setattr__ rewraps the
                # proxy alongside the private ref (IDOR family).
                self._storage.graph = Neo4jBackend.from_driver(  # type: ignore[assignment]
                    self._neo4j_driver,
                    database=neo4j_database,
                    entity_write_concurrency=getattr(neo4j_cfg, "entity_write_concurrency", 12),
                    relationship_write_concurrency=getattr(neo4j_cfg, "relationship_write_concurrency", 8),
                    query_timeout=neo4j_query_timeout,
                    pool_sampler_enabled=getattr(neo4j_cfg, "pool_sampler_enabled", False),
                    pool_sampler_interval_ms=getattr(neo4j_cfg, "pool_sampler_interval_ms", 500),
                    pool_keepalive_enabled=getattr(neo4j_cfg, "pool_keepalive_enabled", False),
                    pool_keepalive_interval_ms=getattr(neo4j_cfg, "pool_keepalive_interval_ms", 15000),
                )

        await self._storage.connect()

        # Create and connect temporal vector store
        if is_surrealdb:
            # Share the coordinator's SurrealDB connection to avoid isolated
            # embedded views (each embedded connection has its own write buffer)
            shared_conn = getattr(self._storage._relational, "_conn", None)
            from khora.engines.skeleton.backends.surrealdb import SurrealDBTemporalStore

            self._temporal_store = SurrealDBTemporalStore(
                self._config,
                connection=shared_conn,
            )
        elif is_sqlite_lance:
            # Reuse the coordinator's shared EmbeddedStorageHandle (single
            # aiosqlite + LanceDB pair across all adapters). The vector
            # adapter holds the canonical reference. Mirrors the Skeleton
            # engine wiring landed in #481.
            if self._storage._vector is None:
                raise RuntimeError("sqlite_lance coordinator did not provide a vector backend")
            sqlite_lance_handle = getattr(self._storage._vector, "_handle", None)
            if sqlite_lance_handle is None:
                raise RuntimeError("sqlite_lance vector adapter is missing its EmbeddedStorageHandle")
            self._temporal_store = create_temporal_store(
                "sqlite_lance",
                self._config,
                sqlite_lance_handle=sqlite_lance_handle,
            )
        else:
            # Share the coordinator's SQLAlchemy engine so the temporal store
            # does not create a second connection pool against the same PG.
            shared_pg_engine = None
            if self._storage._vector is not None:
                shared_pg_engine = getattr(self._storage._vector, "_engine", None)
            if shared_pg_engine is None and self._storage._relational is not None:
                shared_pg_engine = getattr(self._storage._relational, "_engine", None)
            self._temporal_store = create_temporal_store("pgvector", self._config, engine=shared_pg_engine)
        await self._temporal_store.connect()

        # Create embedder
        # Connector fields are forwarded so YAML-configured values reach the
        # shared aiohttp session via configure_litellm — without this hop they
        # would be silently dropped.
        llm_config = LiteLLMConfig(
            model=self._config.llm.model,
            embedding_model=self._config.llm.embedding_model,
            embedding_dimension=self._config.llm.embedding_dimension,
            timeout=self._config.llm.timeout,
            max_retries=self._config.llm.max_retries,
            max_total_connections=self._config.llm.max_total_connections,
            max_connections_per_host=self._config.llm.max_connections_per_host,
            keepalive_timeout_s=self._config.llm.keepalive_timeout_s,
        )
        from khora.config.llm import _init_shared_session, configure_litellm

        configure_litellm(llm_config)
        await _init_shared_session()
        self._embedder = LiteLLMEmbedder.from_config(llm_config)

        # Initialize dual node manager (Neo4j only — non-Neo4j backends
        # use the storage coordinator's graph adapter directly).
        # Route session acquisition through Neo4jBackend._session so pool
        # metrics (timeout counter + acquire_duration) observe these paths.
        if not skip_neo4j:
            neo4j_backend = self._storage._graph if self._storage is not None else None
            self._dual_nodes = DualNodeManager(
                self._neo4j_driver,
                neo4j_database,
                query_timeout=neo4j_query_timeout,
                pool_backend=neo4j_backend,
            )
            await self._dual_nodes.ensure_indexes()

        # Initialize router
        router_config = RouterConfig(
            enabled=self._vc_config.routing_enabled,
            use_llm=self._vc_config.routing_use_llm,
            moderate_depth=1,
            complex_depth=self._vc_config.graph_default_depth,
        )
        self._router = QueryComplexityRouter(router_config)

        # Initialize retriever
        retriever_config = RetrieverConfig(
            default_depth=self._vc_config.graph_default_depth,
            max_depth=self._vc_config.graph_max_depth,
            max_entry_entities=self._vc_config.graph_max_entry_entities,
            rrf_k=self._vc_config.fusion_rrf_k,
            vector_weight=self._vc_config.fusion_vector_weight,
            graph_weight=self._vc_config.fusion_graph_weight,
            simple_vector_weight=self._vc_config.fusion_simple_vector_weight,
            simple_graph_weight=self._vc_config.fusion_simple_graph_weight,
            complex_vector_weight=self._vc_config.fusion_complex_vector_weight,
            complex_graph_weight=self._vc_config.fusion_complex_graph_weight,
            recency_weight=self._vc_config.temporal_recency_weight,
            recency_decay_days=self._vc_config.temporal_recency_decay_days,
            recency_decay_type=self._vc_config.recency_decay_type,
            min_entity_similarity=self._vc_config.retriever_min_entity_similarity,
            hybrid_alpha=self._vc_config.fusion_hybrid_alpha,
            lazy_entity_expansion=self._vc_config.lazy_entity_expansion,
            skeleton_core_ratio=self._vc_config.skeleton_core_ratio,
            enable_session_aware_search=self._vc_config.enable_session_aware_search,
            enable_bm25_channel=self._vc_config.enable_bm25_channel,
            bm25_weight=self._vc_config.bm25_weight,
            bm25_top_k=self._vc_config.bm25_top_k,
            enable_reranking=self._vc_config.enable_reranking,
            reranking_model=self._vc_config.reranking_model,
            reranking_top_n=self._vc_config.reranking_top_n,
            reranking_blend_weight=self._vc_config.reranking_blend_weight,
            enable_llm_reranking=self._vc_config.enable_llm_reranking,
            llm_reranking_model=self._vc_config.llm_reranking_model,
            llm_reranking_top_n=self._vc_config.llm_reranking_top_n,
            llm_reranking_confidence_threshold=self._vc_config.llm_reranking_confidence_threshold,
            llm_reranking_mode=self._vc_config.llm_reranking_mode,
            # Issue #567 Phase A — pull temporal flags from KhoraConfig.query.
            # All default OFF; operators opt in per-namespace.
            temporal_reference_wall_clock=self._config.query.temporal_reference_wall_clock,
            temporal_recency_floor_enabled=self._config.query.temporal_recency_floor_enabled,
            temporal_per_source_decay=self._config.query.temporal_per_source_decay,
            temporal_default_decay_by_source=dict(self._config.query.temporal_default_decay_by_source),
            temporal_recency_channel_enabled=self._config.query.temporal_recency_channel_enabled,
            temporal_query_relevance_floor=self._config.query.temporal_query_relevance_floor,
            temporal_recency_channel_limit=self._config.query.temporal_recency_channel_limit,
            temporal_llm_disambiguation_enabled=self._config.query.temporal_llm_disambiguation_enabled,
            temporal_llm_disambiguation_model=self._config.query.temporal_llm_disambiguation_model,
            # Issue #542 — Personalized PageRank retrieval (HippoRAG 2).
            # Default OFF; flag flows through KhoraConfig.query.enable_ppr_retrieval.
            enable_ppr_retrieval=self._config.query.enable_ppr_retrieval,
            ppr_damping=self._config.query.ppr_damping,
            ppr_max_iter=self._config.query.ppr_max_iter,
            ppr_tol=self._config.query.ppr_tol,
            ppr_top_entities=self._config.query.ppr_top_entities,
        )
        self._retriever = VectorCypherRetriever(
            vector_store=self._temporal_store,
            neo4j_driver=self._neo4j_driver,
            embedder=self._embedder,
            database=neo4j_database,
            config=retriever_config,
            storage=self._storage,
            neo4j_query_timeout=neo4j_query_timeout,
            backend=backend,
        )

        # Initialize telemetry
        from khora.telemetry import init_telemetry
        from khora.telemetry.config import TelemetryConfig

        telemetry_cfg = TelemetryConfig(
            database_url=self._config.telemetry_database_url,
            service_name=self._config.telemetry_service_name,
        )
        await init_telemetry(telemetry_cfg)

        self._connected = True
        logger.info("VectorCypher engine connected")

    async def disconnect(self) -> None:
        """Disconnect from all storage backends."""
        if not self._connected:
            return

        logger.info("Disconnecting VectorCypher engine...")

        # Close shared aiohttp session used by litellm calls
        from khora.config.llm import close_shared_session

        await close_shared_session()

        # Shutdown telemetry
        from khora.telemetry import shutdown_telemetry

        await shutdown_telemetry()

        # Disconnect the storage coordinator (which owns the from_driver-wrapped
        # Neo4jBackend) BEFORE closing the shared driver, so the backend's
        # pool sampler task is stopped while the pool is still alive.
        if self._storage:
            await self._storage.disconnect()
            self._storage = None

        if self._temporal_store:
            await self._temporal_store.disconnect()
            self._temporal_store = None

        if self._neo4j_driver:
            await self._neo4j_driver.close()
            self._neo4j_driver = None

        self._embedder = None
        self._retriever = None
        self._dual_nodes = None
        self._router = None
        self._connected = False

        logger.info("VectorCypher engine disconnected")

    def _get_storage(self) -> StorageCoordinator:
        if self._storage is None:
            raise RuntimeError("VectorCypher engine not connected. Call connect() first.")
        return self._storage

    def _get_temporal_store(self):
        if self._temporal_store is None:
            raise RuntimeError("VectorCypher engine not connected. Call connect() first.")
        return self._temporal_store

    def _get_embedder(self) -> LiteLLMEmbedder:
        if self._embedder is None:
            raise RuntimeError("VectorCypher engine not connected. Call connect() first.")
        return self._embedder

    def _get_retriever(self) -> VectorCypherRetriever:
        if self._retriever is None:
            raise RuntimeError("VectorCypher engine not connected. Call connect() first.")
        return self._retriever

    def _get_dual_nodes(self) -> DualNodeManager | None:
        """Get the dual node manager. Returns None for SurrealDB unified backend."""
        return self._dual_nodes

    # =========================================================================
    # Core API: remember, recall, forget
    # =========================================================================

    async def remember(
        self,
        content: str,
        namespace_id: UUID,
        *,
        title: str = "",
        source: str = "",
        source_type: str = "library",
        source_name: str | None = None,
        source_url: str | None = None,
        source_timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        skill_name: str = "general_entities",
        expertise: ExpertiseConfig | str | None = None,
        extraction_model: str | None = None,
        occurred_at: datetime | None = None,
        entity_types: list[str],
        relationship_types: list[str],
        extraction_config_hash: str | None = None,
        chunk_strategy: ChunkStrategy | None = None,
        external_id: str | None = None,
    ) -> RememberResult:
        """Store content in the memory engine.

        Args:
            content: Content to store
            namespace_id: Namespace to store in
            title: Document title
            source: Document source
            metadata: Additional metadata
            skill_name: Extraction skill to use
            expertise: ExpertiseConfig, expertise name string, or file path
            extraction_model: LLM model for entity extraction (default: config model)
            occurred_at: When this content/event occurred (default: now)
            chunk_strategy: Override chunking strategy for this call.
                Valid values: "fixed", "semantic", "recursive", "conversation".
                When None (default), uses the configured pipeline default.
            external_id: Optional caller-supplied external identifier for the document.

        Returns:
            RememberResult with document_id and counts
        """
        checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
        storage = self._get_storage()

        # Resolve occurred_at: explicit kwarg wins, then metadata["occurred_at"]
        # (parity with remember_batch), then the user-supplied source_timestamp
        # (parity with the relational Document.source_timestamp field), finally
        # fall back to now(). Fixes #859 - source_timestamp was previously
        # ignored on the chunk side even though it was persisted on Document.
        if occurred_at is None:
            if metadata and "occurred_at" in metadata:
                try:
                    occurred_at = self._parse_datetime(metadata["occurred_at"])
                except ValueError:
                    pass
            if occurred_at is None and source_timestamp is not None:
                occurred_at = source_timestamp
            if occurred_at is None:
                occurred_at = datetime.now(UTC)

        # external_id dispatch — route to replace_document_extraction
        # when the caller supplied an external_id that already exists in the
        # namespace. Lookup is status-agnostic (COMPLETED / PROCESSING / FAILED)
        # so the replace path self-heals previously failed rows.
        if external_id is not None:
            existing_by_ext = await storage.get_document_by_external_id(external_id, namespace_id=namespace_id)
            if existing_by_ext is not None:
                return await self._remember_via_replace(
                    existing=existing_by_ext,
                    content=content,
                    checksum=checksum,
                    namespace_id=namespace_id,
                    title=title,
                    source=source,
                    source_type=source_type,
                    source_name=source_name,
                    source_url=source_url,
                    source_timestamp=source_timestamp,
                    metadata=metadata,
                    skill_name=skill_name,
                    expertise=expertise,
                    extraction_model=extraction_model,
                    occurred_at=occurred_at,
                    entity_types=entity_types,
                    relationship_types=relationship_types,
                    extraction_config_hash=extraction_config_hash,
                    chunk_strategy=chunk_strategy,
                    external_id=external_id,
                )

        # Check for duplicate
        existing = await storage.get_document_by_checksum(namespace_id, checksum)
        if existing:
            logger.debug(f"Document already exists (checksum={checksum[:8]}..., status={existing.status})")
            return RememberResult(
                document_id=existing.id,
                namespace_id=namespace_id,
                chunks_created=existing.chunk_count,
                entities_extracted=existing.entity_count,
                relationships_created=existing.relationship_count,
                metadata={"duplicate": True, "status": str(existing.status)},
            )

        # Create document
        document = Document(
            namespace_id=namespace_id,
            content=content,
            title=title or None,
            source=source or None,
            source_type=source_type,
            source_name=source_name or None,
            source_url=source_url or None,
            source_timestamp=source_timestamp,
            checksum=checksum,
            size_bytes=len(content.encode("utf-8")),
            metadata=dict(metadata or {}),
            extraction_config_hash=extraction_config_hash,
            external_id=external_id,
            session_id=_coerce_session_id_from_metadata(metadata),
        )
        try:
            document = await storage.create_document(document)
        except IntegrityError:
            # Concurrent race on `(namespace_id, external_id)`: another caller
            # inserted the same external_id between our lookup and this
            # create. The partial UNIQUE index ``ix_documents_namespace_external_id_unique``
            # converts the race into a deterministic conflict.
            # Retry the lookup and route to replace so the loser still
            # succeeds against the winner's row.
            if external_id is None:
                raise
            existing_after_race = await storage.get_document_by_external_id(external_id, namespace_id=namespace_id)
            if existing_after_race is None:
                raise
            return await self._remember_via_replace(
                existing=existing_after_race,
                content=content,
                checksum=checksum,
                namespace_id=namespace_id,
                title=title,
                source=source,
                source_timestamp=source_timestamp,
                metadata=metadata,
                skill_name=skill_name,
                expertise=expertise,
                extraction_model=extraction_model,
                occurred_at=occurred_at,
                entity_types=entity_types,
                relationship_types=relationship_types,
                extraction_config_hash=extraction_config_hash,
                chunk_strategy=chunk_strategy,
                external_id=external_id,
            )

        # #889: collect ADR-001 diagnostics from extraction so failures
        # are visible on RememberResult.metadata instead of looking
        # successful with entities_extracted=0.
        extraction_diagnostics: dict[str, Any] = {}

        # Process document
        chunks_created, entities_extracted, relationships_created = await self._process_document(
            document,
            skill_name=skill_name,
            expertise=expertise,
            extraction_model=extraction_model,
            occurred_at=occurred_at,
            entity_types=entity_types,
            relationship_types=relationship_types,
            chunk_strategy=chunk_strategy,
            out_diagnostics=extraction_diagnostics,
        )

        result_metadata = _build_remember_metadata(extraction_diagnostics)
        return RememberResult(
            document_id=document.id,
            namespace_id=namespace_id,
            chunks_created=chunks_created,
            entities_extracted=entities_extracted,
            relationships_created=relationships_created,
            metadata=result_metadata,
        )

    async def _process_document(
        self,
        document: Document,
        *,
        skill_name: str,
        expertise: ExpertiseConfig | str | None = None,
        extraction_model: str | None = None,
        occurred_at: datetime,
        entity_types: list[str],
        relationship_types: list[str],
        chunk_strategy: ChunkStrategy | None = None,
        max_chunks_in_flight: int | None = None,
        chunk_semaphore: _GlobalChunkSemaphore | None = None,
        out_diagnostics: dict[str, Any] | None = None,
    ) -> tuple[int, int, int]:
        """Process a document into chunks with skeleton-based entity extraction.

        The VectorCypher pipeline:
        1. Chunk the document
        2. Embed all chunks
        3. Run skeleton indexing to identify core chunks (configurable, default 70%)
        4. Extract entities only from core chunks
        5. Store chunks in pgvector and create Chunk nodes in Neo4j
        6. Link entities to chunks via MENTIONED_IN
        """
        from khora.extraction.chunkers import create_chunker

        with trace_span(
            "khora.vectorcypher.process_document",
            namespace_id=str(document.namespace_id),
            document_id=str(document.id),
        ) as span:
            storage = self._get_storage()
            embedder = self._get_embedder()
            temporal_store = self._get_temporal_store()
            dual_nodes = self._get_dual_nodes()

            # Create chunker
            strategy = chunk_strategy if chunk_strategy is not None else self._config.pipeline.chunking_strategy
            chunker = create_chunker(
                strategy=strategy,
                chunk_size=self._config.pipeline.chunk_size,
                chunk_overlap=self._config.pipeline.chunk_overlap,
            )

            # Chunk the document
            with trace_span("khora.vectorcypher.chunking"):
                raw_chunks = await asyncio.to_thread(chunker.chunk, document.content)

            if not raw_chunks:
                document.mark_completed(0, 0)
                await storage.update_document(document)
                span.set_attribute("chunk_count", 0)
                return 0, 0, 0

            # Extract metadata (computed once, not per window)
            doc_metadata = document.metadata or {}

            # Split into windows when max_chunks_in_flight is set; otherwise one window.
            # Per-call override takes precedence over the engine config.
            window_size = (
                max_chunks_in_flight if max_chunks_in_flight is not None else self._vc_config.max_chunks_in_flight
            )
            windows = (
                [raw_chunks[i : i + window_size] for i in range(0, len(raw_chunks), window_size)]
                if window_size is not None
                else [raw_chunks]
            )

            total_chunks_created = 0
            entities_extracted = 0
            relationships_created = 0
            chunk_index_offset = 0

            for window in windows:
                # Acquire global chunk semaphore before processing this window.
                # This bounds total chunks in flight across all concurrent
                # submit_batch calls to max_chunks_in_flight process-wide.
                n_window = len(window)
                n_acquired = n_window
                if chunk_semaphore is not None:
                    n_acquired = await chunk_semaphore.acquire(n_window)
                try:
                    # Embed window chunks in batch
                    with trace_span("khora.vectorcypher.embed_batch", chunk_count=len(window)):
                        chunk_texts = [c.content for c in window]
                        embeddings = await embedder.embed_batch(chunk_texts)

                    # Create temporal chunks
                    temporal_chunks = []
                    for i, (raw_chunk, embedding) in enumerate(zip(window, embeddings)):
                        temporal_chunk = TemporalChunk(
                            id=None,
                            namespace_id=document.namespace_id,
                            document_id=document.id,
                            content=raw_chunk.content,
                            embedding=embedding,
                            occurred_at=occurred_at,
                            created_at=datetime.now(UTC),
                            source_system=doc_metadata.get("source_system"),
                            author=doc_metadata.get("author"),
                            channel=doc_metadata.get("channel") or doc_metadata.get("thread_id"),
                            tags=_ensure_tags(doc_metadata.get("tags", [])),
                            confidence=1.0,
                            metadata={
                                **doc_metadata,
                                "chunk_index": chunk_index_offset + i,
                                "start_char": raw_chunk.start_char if hasattr(raw_chunk, "start_char") else 0,
                                "end_char": raw_chunk.end_char
                                if hasattr(raw_chunk, "end_char")
                                else len(raw_chunk.content),
                            },
                            chunker_info=dict(raw_chunk.metadata),
                        )
                        temporal_chunks.append(temporal_chunk)

                    # Store in pgvector
                    stored_chunks = await temporal_store.create_chunks_batch(temporal_chunks)

                    # Update temporal_chunks with assigned IDs
                    for i, stored in enumerate(stored_chunks):
                        temporal_chunks[i].id = stored.id

                    # Create Chunk nodes in Neo4j (skipped for SurrealDB — chunks in temporal store)
                    if dual_nodes is not None:
                        await dual_nodes.create_chunk_nodes_batch(temporal_chunks, document.namespace_id)

                    # Skeleton-based entity extraction (for core chunks only)
                    if self._config.pipeline.extract_entities:
                        ents, rels = await self._run_skeleton_extraction(
                            temporal_chunks,
                            document.namespace_id,
                            skill_name=skill_name,
                            expertise=expertise,
                            extraction_model=extraction_model,
                            entity_types=entity_types,
                            relationship_types=relationship_types,
                            out_diagnostics=out_diagnostics,
                        )
                        entities_extracted += ents
                        relationships_created += rels

                    total_chunks_created += len(stored_chunks)
                    chunk_index_offset += len(window)
                finally:
                    if chunk_semaphore is not None:
                        await chunk_semaphore.release(n_acquired)

            # Update document status
            document.mark_completed(total_chunks_created, entities_extracted, relationships_created)
            await storage.update_document(document)

            logger.debug(
                f"Processed document {document.id}: {total_chunks_created} chunks, "
                f"{entities_extracted} entities, {relationships_created} relationships"
            )

            span.set_attribute("chunk_count", total_chunks_created)
            span.set_attribute("entities_extracted", entities_extracted)
            span.set_attribute("relationships_created", relationships_created)
            return total_chunks_created, entities_extracted, relationships_created

    async def process_staged_document(
        self,
        document: Document,
        *,
        skill_name: str,
        occurred_at: datetime,
        entity_types: list[str],
        relationship_types: list[str],
        expertise: ExpertiseConfig | str | None = None,
        extraction_config_hash: str | None = None,
        chunk_strategy: ChunkStrategy | None = None,
        max_chunks_in_flight: int | None = None,
        chunk_semaphore: _GlobalChunkSemaphore | None = None,
    ) -> tuple[int, int, int]:
        """Process a pre-staged PENDING document through the VectorCypher pipeline.

        Called by Khora.submit_batch() for documents that were already
        persisted to the DB with PENDING status before this call. Delegates
        to _process_document; does NOT create a new document record.

        Args:
            document: Pre-created PENDING Document from storage.
            skill_name: Extraction skill to use.
            occurred_at: Temporal anchor for chunks and entities.
            entity_types: Entity types to extract.
            relationship_types: Relationship types to extract.
            expertise: Optional domain-specific extraction config.
            extraction_config_hash: Optional hash for change detection.
            chunk_strategy: Override chunking strategy.
            max_chunks_in_flight: Maximum chunks per processing window.
            chunk_semaphore: Optional global chunk semaphore (from Khora)
                shared across concurrent submit_batch calls to bound total
                chunks in flight process-wide.

        Returns:
            Tuple of (chunks_created, entities_extracted, relationships_created).
        """
        # Update the document's extraction_config_hash if provided, so it is
        # persisted when mark_completed() writes the record back.
        if extraction_config_hash is not None and document.extraction_config_hash != extraction_config_hash:
            document.extraction_config_hash = extraction_config_hash

        return await self._process_document(
            document,
            skill_name=skill_name,
            expertise=expertise,
            extraction_model=None,
            occurred_at=occurred_at,
            entity_types=entity_types,
            relationship_types=relationship_types,
            chunk_strategy=chunk_strategy,
            max_chunks_in_flight=max_chunks_in_flight,
            chunk_semaphore=chunk_semaphore,
        )

    async def clear_document_extraction_state(self, document_id: UUID, namespace_id: UUID) -> None:
        """Clear partial extraction state (khora_chunks + :Chunk nodes) for a FAILED document.

        Called by submit_batch before re-queuing a previously-FAILED document to prevent
        duplicate chunks accumulating on retry (self-heal path, H1 fix).

        Best-effort: logs and ignores storage errors so that cleanup failures do not
        block re-processing.
        """
        temporal_store = self._get_temporal_store()
        dual_nodes = self._get_dual_nodes()
        try:
            await temporal_store.delete_chunks_by_document(document_id, namespace_id)
        except Exception as exc:
            logger.warning(f"submit_batch cleanup: could not clear khora_chunks for document {document_id}: {exc}")
        if dual_nodes is not None:
            try:
                await dual_nodes.delete_chunks_by_document(document_id, namespace_id)
            except Exception as exc:
                logger.warning(f"submit_batch cleanup: could not clear :Chunk nodes for document {document_id}: {exc}")

    async def _run_skeleton_extraction(
        self,
        chunks: list[TemporalChunk],
        namespace_id: UUID,
        *,
        skill_name: str = "general_entities",
        expertise: ExpertiseConfig | str | None = None,
        extraction_model: str | None = None,
        entity_types: list[str],
        relationship_types: list[str],
        out_diagnostics: dict[str, Any] | None = None,
    ) -> tuple[int, int]:
        """Run skeleton-based entity extraction on core chunks only.

        Uses the skeleton indexer to identify the top core chunks (by PageRank),
        then runs full LLM extraction on those chunks and stores the results
        in both Neo4j (entities, relationships, MENTIONED_IN links) and pgvector.

        Args:
            chunks: All chunks from the document
            namespace_id: Namespace ID
            skill_name: Extraction skill to use
            expertise: ExpertiseConfig, expertise name string, or file path
            extraction_model: LLM model for extraction (default: config model)

        Returns:
            Tuple of (entities_extracted, relationships_created)
        """
        from khora.pipelines.tasks.extract import extract_entities

        if not chunks:
            return 0, 0

        with trace_span(
            "khora.vectorcypher.skeleton_extraction",
            namespace_id=str(namespace_id),
            total_chunks=len(chunks),
        ) as span:
            # Skip skeleton overhead for small documents (≤2 chunks)
            if len(chunks) <= 2:
                core_ids = {c.id for c in chunks}
            else:
                skeleton = SkeletonIndexer(core_ratio=self._vc_config.skeleton_core_ratio)
                skeleton.add_chunks_batch(chunks)
                with trace_span(
                    "khora.vectorcypher.skeleton_build",
                    chunk_count=len(chunks),
                    core_ratio=self._vc_config.skeleton_core_ratio,
                ):
                    core_ids = await asyncio.to_thread(skeleton.build_skeleton)

            logger.debug(f"Skeleton indexing: {len(core_ids)}/{len(chunks)} core chunks")
            span.set_attribute("core_chunks", len(core_ids))

            if not core_ids:
                return 0, 0

            # Get core chunks
            core_temporal_chunks = [c for c in chunks if c.id in core_ids]

            # Convert TemporalChunk -> Chunk for extract_entities()
            chunk_objects = []
            for tc in core_temporal_chunks:
                chunk_objects.append(
                    Chunk(
                        id=tc.id,
                        namespace_id=tc.namespace_id,
                        document_id=tc.document_id,
                        content=tc.content,
                        created_at=tc.created_at or tc.occurred_at,
                        chunker_info=dict(tc.chunker_info or {}),
                    )
                )

            # Run LLM extraction on core chunks
            model = extraction_model or self._config.llm.model
            entities, relationships = await extract_entities(
                chunk_objects,
                skill_name=skill_name,
                expertise=expertise,
                model=model,
                max_concurrent=self._vc_config.max_concurrent_extractions,
                timeout=self._config.llm.timeout,
                max_tokens=self._config.llm.max_tokens,
                extraction_batch_size=self._vc_config.extraction_batch_size,
                entity_types=entity_types,
                relationship_types=relationship_types,
                store_events=self._vc_config.store_events,
                out_diagnostics=out_diagnostics,
            )

            if not entities:
                span.set_attribute("entities_extracted", 0)
                return 0, 0

            # Compute entity embeddings (matching ingest pipeline format)
            embedder = self._get_embedder()
            entity_texts = [f"{e.name}: {e.description}" if e.description else e.name for e in entities]
            entity_embeddings = await embedder.embed_batch(entity_texts)
            for entity, embedding in zip(entities, entity_embeddings):
                entity.embedding = embedding
                entity.embedding_model = embedder.model_name

            storage = self._get_storage()
            dual_nodes = self._get_dual_nodes()

            # Snapshot pre-upsert IDs (#806). On the second ingest that
            # shares an entity, the storage backends match by
            # ``(namespace_id, name, entity_type)`` and rewrite the input
            # ``entity.id`` to the canonical persisted UUID. Any
            # relationship built from the extraction-time UUID needs to
            # be remapped to the canonical id before
            # ``create_relationships_batch`` runs - otherwise sqlite_lance
            # fires an FK violation and Neo4j silently drops the row
            # because its ``MATCH (source {id: ...})`` finds nothing.
            pre_upsert_ids = [str(e.id) for e in entities]

            # Store entities in Neo4j + pgvector
            await storage.upsert_entities_batch(namespace_id, entities)

            # Build pre-upsert -> canonical remap (only diffs).
            id_remap: dict[str, str] = {}
            for pre_id, entity in zip(pre_upsert_ids, entities):
                canonical_id = str(entity.id)
                if pre_id != canonical_id:
                    id_remap[pre_id] = canonical_id

            # Apply the remap to the LLM-extracted relationships before
            # we build co-occurrence rels (which already use the
            # canonical ids because they read ``entity.id`` post-upsert).
            if id_remap and relationships:
                from uuid import UUID as _UUID

                for rel in relationships:
                    src_str = str(rel.source_entity_id)
                    tgt_str = str(rel.target_entity_id)
                    if src_str in id_remap:
                        rel.source_entity_id = _UUID(id_remap[src_str])
                    if tgt_str in id_remap:
                        rel.target_entity_id = _UUID(id_remap[tgt_str])

            # Create co-occurrence relationships between entities in the same chunk
            cooccurrence_rels = _build_cooccurrence_relationships(entities, chunk_objects, namespace_id, relationships)
            if cooccurrence_rels:
                relationships = list(relationships) + cooccurrence_rels

            # Store relationships in Neo4j
            relationships_created = 0
            if relationships:
                relationships_created = await storage.create_relationships_batch(relationships)

            # Build entity-chunk links from source_chunk_ids
            entity_chunk_links: list[EntityChunkLink] = []
            for entity in entities:
                for chunk_id in entity.source_chunk_ids:
                    entity_chunk_links.append(
                        EntityChunkLink(
                            entity_id=entity.id,
                            chunk_id=chunk_id,
                        )
                    )

            # Create MENTIONED_IN edges in Neo4j
            if entity_chunk_links:
                if dual_nodes is not None:
                    await dual_nodes.link_entities_to_chunks_batch(entity_chunk_links)

            logger.debug(
                f"Skeleton extraction: {len(entities)} entities, "
                f"{relationships_created} relationships from {len(core_temporal_chunks)} core chunks"
            )

            span.set_attribute("entities_extracted", len(entities))
            span.set_attribute("relationships_created", relationships_created)
            return len(entities), relationships_created

    async def _run_skeleton_extraction_deferred(
        self,
        chunks: list[TemporalChunk],
        namespace_id: UUID,
        *,
        skill_name: str = "general_entities",
        expertise: ExpertiseConfig | str | None = None,
        extraction_model: str | None = None,
        entity_types: list[str],
        relationship_types: list[str],
        skeleton_ratio_override: float | None = None,
        is_conversation: bool = False,
    ) -> tuple[list[Entity], list[Relationship], list[EntityChunkLink]]:
        """Run skeleton extraction but return results instead of storing.

        Same as _run_skeleton_extraction but defers storage so the caller
        can accumulate entities across multiple documents for batch storage.

        Args:
            is_conversation: When True, use a lower min_extraction_tokens threshold
                (15 instead of 50) so short conversational messages still get entity
                extraction. This enables graph retrieval for LoCoMo-style benchmarks
                where messages are typically 10-50 words.

        Returns:
            Tuple of (entities_with_embeddings, relationships, entity_chunk_links)
        """
        from khora.pipelines.tasks.extract import extract_entities

        if not chunks:
            return [], [], []

        # Use lower threshold for conversation batches.
        # Short conversational messages (10-50 words) contain critical entities
        # (people, places, activities) that must be extracted for graph retrieval.
        # Without extraction, these chunks are invisible to graph search.
        min_tokens = self._vc_config.min_extraction_tokens
        if is_conversation:
            min_tokens = min(min_tokens, 15)
        if min_tokens > 0 and all(len(c.content.split()) <= min_tokens for c in chunks):
            logger.debug(f"Skipping entity extraction for {len(chunks)} short chunks (all ≤ {min_tokens} tokens)")
            return [], [], []

        # Skip skeleton overhead for small documents (≤2 chunks)
        if len(chunks) <= 2:
            core_ids = {c.id for c in chunks}
        else:
            effective_ratio = skeleton_ratio_override or self._vc_config.skeleton_core_ratio
            skeleton = SkeletonIndexer(core_ratio=effective_ratio)
            skeleton.add_chunks_batch(chunks)
            with trace_span(
                "khora.vectorcypher.skeleton_build",
                chunk_count=len(chunks),
                core_ratio=effective_ratio,
            ):
                core_ids = await asyncio.to_thread(skeleton.build_skeleton)

        logger.debug(f"Skeleton indexing (deferred): {len(core_ids)}/{len(chunks)} core chunks")

        if not core_ids:
            return [], [], []

        core_temporal_chunks = [c for c in chunks if c.id in core_ids]
        chunk_objects = [
            Chunk(
                id=tc.id,
                namespace_id=tc.namespace_id,
                document_id=tc.document_id,
                content=tc.content,
                created_at=tc.created_at or tc.occurred_at,
                chunker_info=dict(tc.chunker_info or {}),
            )
            for tc in core_temporal_chunks
        ]

        model = extraction_model or self._config.llm.model
        entities, relationships = await extract_entities(
            chunk_objects,
            skill_name=skill_name,
            expertise=expertise,
            model=model,
            max_concurrent=self._vc_config.max_concurrent_extractions,
            timeout=self._config.llm.timeout,
            max_tokens=self._config.llm.max_tokens,
            extraction_batch_size=self._vc_config.extraction_batch_size,
            entity_types=entity_types,
            relationship_types=relationship_types,
            store_events=self._vc_config.store_events,
        )

        if not entities:
            return [], [], []

        # Compute entity embeddings
        embedder = self._get_embedder()
        entity_texts = [f"{e.name}: {e.description}" if e.description else e.name for e in entities]
        entity_embeddings = await embedder.embed_batch(entity_texts)
        for entity, embedding in zip(entities, entity_embeddings):
            entity.embedding = embedding
            entity.embedding_model = embedder.model_name

        # Create co-occurrence relationships between entities in the same chunk
        cooccurrence_rels = _build_cooccurrence_relationships(entities, chunk_objects, namespace_id, relationships)
        if cooccurrence_rels:
            relationships = list(relationships) + cooccurrence_rels

        # Build entity-chunk links
        entity_chunk_links: list[EntityChunkLink] = []
        for entity in entities:
            for chunk_id in entity.source_chunk_ids:
                entity_chunk_links.append(EntityChunkLink(entity_id=entity.id, chunk_id=chunk_id))

        return entities, relationships, entity_chunk_links

    async def _remember_via_replace(
        self,
        *,
        existing: Document,
        content: str,
        checksum: str,
        namespace_id: UUID,
        title: str,
        source: str,
        source_type: str = "library",
        source_name: str | None = None,
        source_url: str | None = None,
        source_timestamp: datetime | None = None,
        metadata: dict[str, Any] | None,
        skill_name: str,
        expertise: ExpertiseConfig | str | None,
        extraction_model: str | None,
        occurred_at: datetime,
        entity_types: list[str],
        relationship_types: list[str],
        extraction_config_hash: str | None,
        chunk_strategy: ChunkStrategy | None,
        external_id: str,
    ) -> RememberResult:
        """Dispatch an ``external_id``-matched remember() to the replace path.

        Builds chunks / entities / relationships in-memory, then performs the
        full VectorCypher storage-side replace that ``replace_document_extraction``
        alone does not cover. The coordinator primitive handles the
        ``chunks`` table + Neo4j entity / relationship retire / remap / upsert,
        but VectorCypher also owns ``khora_chunks`` (via ``TemporalVectorStore``)
        and Neo4j ``:Chunk`` nodes (via ``DualNodeManager``). This method:

        1. Reuses ``existing.id`` — the same id is reused across replacements;
           the row is updated in place. Preserves ``created_at`` /
           ``source_timestamp`` / ``processed_at``.
        2. Chunks + embeds + extracts in-memory (mirrors the create path but
           defers all persistence).
        3. Wipes old ``khora_chunks`` rows and old ``:Chunk`` nodes, writes
           new ones with refreshed embeddings / content / metadata — BEFORE
           the coordinator call so a failure mid-wipe marks the doc FAILED
           and the next replace self-heals.
        4. Delegates to ``coordinator.replace_document_extraction`` for
           atomic PG transaction + graph retire / remap / upsert.
        5. Rebuilds ``MENTIONED_IN`` edges from upserted entities to the
           new ``:Chunk`` nodes. Without this, retired entities would still
           reference old chunks via stale edges.

        Any exception in steps 3 or 5 marks the document FAILED, best-effort
        persists the status, and re-raises unwrapped — mirroring the
        coordinator's own failure handling.
        """
        from khora.extraction.chunkers import create_chunker
        from khora.pipelines.tasks.extract import extract_entities

        storage = self._get_storage()
        embedder = self._get_embedder()

        # 1. Build the replacement Document row. Reuse existing.id; refresh
        #    content/checksum/metadata/external_id/extraction_config_hash.
        #    Preserve created_at from the existing row.
        new_doc_metadata = dict(metadata or {})
        new_document = Document(
            id=existing.id,
            namespace_id=namespace_id,
            content=content,
            title=title or None,
            source=source or None,
            source_type=source_type,
            source_name=source_name or None,
            source_url=source_url or None,
            checksum=checksum,
            size_bytes=len(content.encode("utf-8")),
            metadata=new_doc_metadata,
            extraction_config_hash=extraction_config_hash,
            external_id=external_id,
            created_at=existing.created_at,
            source_timestamp=source_timestamp if source_timestamp is not None else existing.source_timestamp,
            processed_at=existing.processed_at,
            session_id=_coerce_session_id_from_metadata(metadata) or existing.session_id,
        )

        # 2. Chunk + embed in-memory (no persistence).
        strategy = chunk_strategy if chunk_strategy is not None else self._config.pipeline.chunking_strategy
        chunker = create_chunker(
            strategy=strategy,
            chunk_size=self._config.pipeline.chunk_size,
            chunk_overlap=self._config.pipeline.chunk_overlap,
        )
        raw_chunks = await asyncio.to_thread(chunker.chunk, content)

        new_chunks: list[Chunk] = []
        if raw_chunks:
            embed_texts = [c.content for c in raw_chunks]
            embeddings = await embedder.embed_batch(embed_texts)
            now = datetime.now(UTC)
            for i, (raw_chunk, embedding) in enumerate(zip(raw_chunks, embeddings)):
                new_chunks.append(
                    Chunk(
                        namespace_id=namespace_id,
                        document_id=new_document.id,
                        content=raw_chunk.content,
                        chunk_index=i,
                        start_char=getattr(raw_chunk, "start_char", 0),
                        end_char=getattr(raw_chunk, "end_char", len(raw_chunk.content)),
                        metadata={**new_doc_metadata, "chunk_index": i},
                        chunker_info=dict(raw_chunk.metadata),
                        embedding=embedding,
                        embedding_model=embedder.model_name,
                        created_at=now,
                        source_timestamp=occurred_at,
                    )
                )

        # 3. Extract entities + relationships from core chunks (skeleton),
        #    exactly as the create path does — but deferred (no storage).
        new_entities: list[Entity] = []
        new_relationships: list[Relationship] = []
        if new_chunks and self._config.pipeline.extract_entities:
            if len(new_chunks) <= 2:
                core_chunks = new_chunks
            else:
                skeleton = SkeletonIndexer(core_ratio=self._vc_config.skeleton_core_ratio)
                skeleton_input = [
                    TemporalChunk(
                        id=c.id,
                        namespace_id=c.namespace_id,
                        document_id=c.document_id,
                        content=c.content,
                        embedding=c.embedding,
                        occurred_at=occurred_at,
                        created_at=c.created_at,
                        chunker_info=dict(c.chunker_info or {}),
                    )
                    for c in new_chunks
                ]
                skeleton.add_chunks_batch(skeleton_input)
                core_ids = await asyncio.to_thread(skeleton.build_skeleton)
                core_chunks = [c for c in new_chunks if c.id in core_ids]

            if core_chunks:
                model = extraction_model or self._config.llm.model
                extracted_entities, extracted_relationships = await extract_entities(
                    core_chunks,
                    skill_name=skill_name,
                    expertise=expertise,
                    model=model,
                    max_concurrent=self._vc_config.max_concurrent_extractions,
                    timeout=self._config.llm.timeout,
                    max_tokens=self._config.llm.max_tokens,
                    extraction_batch_size=self._vc_config.extraction_batch_size,
                    entity_types=entity_types,
                    relationship_types=relationship_types,
                    store_events=self._vc_config.store_events,
                )

                if extracted_entities:
                    entity_texts = [
                        f"{e.name}: {e.description}" if e.description else e.name for e in extracted_entities
                    ]
                    entity_embeddings = await embedder.embed_batch(entity_texts)
                    for entity, emb in zip(extracted_entities, entity_embeddings):
                        entity.embedding = emb
                        entity.embedding_model = embedder.model_name

                    new_entities = list(extracted_entities)
                    new_relationships = list(extracted_relationships)

                    cooccurrence_rels = _build_cooccurrence_relationships(
                        new_entities, new_chunks, namespace_id, new_relationships
                    )
                    if cooccurrence_rels:
                        new_relationships.extend(cooccurrence_rels)

        # 4. Wipe/write VectorCypher-owned stores (khora_chunks + :Chunk nodes)
        #    BEFORE the coordinator call. The coordinator only owns the
        #    `chunks` table + graph entities/relationships; it does NOT know
        #    about `khora_chunks` (via TemporalVectorStore) or :Chunk nodes
        #    (via DualNodeManager), which VectorCypher's create path writes
        #    directly (see `_process_document`). Without this, after a
        #    replace, retrieval returns stale content because khora_chunks
        #    still holds the old chunks and :Chunk nodes still reference
        #    old document content, with MENTIONED_IN edges pointing from
        #    retired entities to old chunks.
        temporal_store = self._get_temporal_store()
        dual_nodes = self._get_dual_nodes()
        doc_metadata = new_doc_metadata

        new_temporal_chunks: list[TemporalChunk] = []
        for i, c in enumerate(new_chunks):
            new_temporal_chunks.append(
                TemporalChunk(
                    id=c.id,
                    namespace_id=c.namespace_id,
                    document_id=c.document_id,
                    content=c.content,
                    embedding=c.embedding,
                    occurred_at=occurred_at,
                    created_at=datetime.now(UTC),
                    source_system=doc_metadata.get("source_system"),
                    author=doc_metadata.get("author"),
                    channel=doc_metadata.get("channel") or doc_metadata.get("thread_id"),
                    tags=_ensure_tags(doc_metadata.get("tags", [])),
                    confidence=1.0,
                    metadata={
                        **doc_metadata,
                        "chunk_index": i,
                        "start_char": c.start_char,
                        "end_char": c.end_char or len(c.content),
                    },
                    chunker_info=dict(c.chunker_info or {}),
                )
            )

        try:
            # Wipe old VectorCypher-owned state.
            await temporal_store.delete_chunks_by_document(existing.id, namespace_id)
            if dual_nodes is not None:
                await dual_nodes.delete_chunks_by_document(existing.id, namespace_id)

            # Write new chunks (khora_chunks + :Chunk nodes).
            if new_temporal_chunks:
                stored_temporal = await temporal_store.create_chunks_batch(new_temporal_chunks)
                # Propagate any assigned ids back so the coordinator, which
                # writes to `chunks` below, uses the same uuids as
                # khora_chunks / :Chunk nodes.
                for tc, stored, chunk in zip(new_temporal_chunks, stored_temporal, new_chunks):
                    tc.id = stored.id
                    chunk.id = stored.id
                if dual_nodes is not None:
                    await dual_nodes.create_chunk_nodes_batch(new_temporal_chunks, namespace_id)
        except Exception as e:
            # Self-heal: mark FAILED and re-raise unwrapped so the next
            # successful replace against the same external_id heals the row.
            new_document.mark_failed(str(e))
            try:
                await storage.update_document(new_document)
            except Exception as update_err:
                logger.warning(
                    f"Failed to mark document {new_document.id} FAILED during "
                    f"_remember_via_replace error handling: {update_err}"
                )
            raise

        # 5. Hand off to the coordinator — it owns the Postgres transaction,
        #    graph retire / remap / upsert, and FAILED-on-exception handling.
        #
        #    #884: catch the typed signal for "PG committed, graph mirror
        #    partial" so we can record the divergence on the user-facing
        #    RememberResult instead of presenting the failure as if PG
        #    also rolled back. Skipping steps 6-8 (MENTIONED_IN linking,
        #    source_chunk_ids reset) is intentional: those also touch the
        #    graph backend, and re-attempting them now would either raise
        #    again or write against the partial graph state. A future
        #    reconciler will replay the missing graph work.
        from khora.exceptions import GraphMirrorFailedAfterPGCommitError

        try:
            replace_result = await storage.replace_document_extraction(
                namespace_id=namespace_id,
                old_document_id=existing.id,
                new_document=new_document,
                new_chunks=new_chunks,
                new_entities=new_entities,
                new_relationships=new_relationships,
            )
        except GraphMirrorFailedAfterPGCommitError as graph_mirror_err:
            logger.warning(
                "replace_document_extraction graph-mirror phase failed after "
                "PG commit for document {} in namespace {}: {}. Returning "
                "RememberResult with degradation; the next successful "
                "replace heals the graph (#884).",
                new_document.id,
                namespace_id,
                graph_mirror_err.original_exception_type,
            )
            return RememberResult(
                document_id=new_document.id,
                namespace_id=namespace_id,
                chunks_created=len(new_chunks),
                # entities_extracted reflects what was extracted, not what
                # made it into the graph - PG-side entity counts are stamped
                # but graph state is partial. Operators see the divergence
                # via the metadata.degradations entry below.
                entities_extracted=len(new_entities),
                relationships_created=0,
                metadata={
                    "replaced": True,
                    "old_document_id": str(existing.id),
                    "degradations": [
                        {
                            "component": "coordinator.replace_document_extraction",
                            "reason": "graph_mirror_failed_after_pg_commit",
                            "exception": graph_mirror_err.original_exception_type,
                            "issue": "884",
                        }
                    ],
                },
            )

        # 6. After the coordinator has (re)written graph entities (retire /
        #    remap / upsert), relink entities → chunks via MENTIONED_IN for
        #    Neo4j-backed deployments. Mirrors `_run_skeleton_extraction`
        #    lines ~901-915. Skipped when dual_nodes is None (SurrealDB
        #    unified — its graph adapter owns entity↔chunk linkage).
        if dual_nodes is not None and new_entities:
            entity_chunk_links: list[EntityChunkLink] = []
            for entity in new_entities:
                for chunk_id in entity.source_chunk_ids:
                    entity_chunk_links.append(
                        EntityChunkLink(
                            entity_id=entity.id,
                            chunk_id=chunk_id,
                        )
                    )
            if entity_chunk_links:
                try:
                    await dual_nodes.link_entities_to_chunks_batch(entity_chunk_links)
                except Exception as e:
                    new_document.mark_failed(str(e))
                    try:
                        await storage.update_document(new_document)
                    except Exception as update_err:
                        logger.warning(
                            f"Failed to mark document {new_document.id} FAILED "
                            f"during MENTIONED_IN linking error handling: {update_err}"
                        )
                    raise

        # 7. Overwrite ``source_chunk_ids`` to the current extraction's chunk
        #    UUIDs. The replace decomposes into
        #    ``upsert_entities_batch`` (net-new only; ON MATCH *appends*
        #    source_chunk_ids[-250..]) + ``remap_source_document_ids_batch``
        #    (survivors; source_document_ids only). Neither replaces
        #    source_chunk_ids, so without this step survivor entities keep
        #    retired chunk UUIDs and net-new entities accumulate stale UUIDs
        #    from prior documents. Downstream consumers that read
        #    ``len(entity.source_chunk_ids)`` as a mention count would
        #    double-count across replaces. SET is idempotent; safe for both
        #    survivors and net-new.
        graph = storage.graph
        reset_source_chunk_ids = getattr(graph, "reset_entity_source_chunk_ids_batch", None) if graph else None
        if reset_source_chunk_ids is not None and new_entities:
            reset_rows = [
                {
                    "name": e.name,
                    "entity_type": e.entity_type,
                    "source_chunk_ids": [str(c) for c in e.source_chunk_ids],
                }
                for e in new_entities
                if e.name and e.entity_type
            ]
            if reset_rows:
                await reset_source_chunk_ids(namespace_id, reset_rows)

        # 8. Same append-with-tail concern for relationships: Neo4j's
        #    create_relationships_batch ON MATCH clause appends
        #    source_chunk_ids[-250..]. Match relationships by entity name+type
        #    (MERGE-stable across replaces) rather than UUID — survivor
        #    entities keep their persisted Neo4j id, which differs from the
        #    fresh extraction's uuid.
        reset_rel_source_chunk_ids = (
            getattr(graph, "reset_relationship_source_chunk_ids_batch", None) if graph else None
        )
        if reset_rel_source_chunk_ids is not None and new_relationships and new_entities:
            from khora.storage.backends.neo4j import _sanitize_neo4j_label

            entity_key_by_id: dict[UUID, tuple[str, str]] = {
                e.id: (e.name, e.entity_type) for e in new_entities if e.name and e.entity_type
            }
            rel_reset_rows: list[dict[str, Any]] = []
            for r in new_relationships:
                src_key = entity_key_by_id.get(r.source_entity_id)
                tgt_key = entity_key_by_id.get(r.target_entity_id)
                if src_key is None or tgt_key is None:
                    continue
                rel_reset_rows.append(
                    {
                        "source_name": src_key[0],
                        "source_type": src_key[1],
                        "target_name": tgt_key[0],
                        "target_type": tgt_key[1],
                        "rel_type": _sanitize_neo4j_label(r.relationship_type),
                        "source_chunk_ids": [str(c) for c in r.source_chunk_ids],
                    }
                )
            if rel_reset_rows:
                await reset_rel_source_chunk_ids(namespace_id, rel_reset_rows)

        return RememberResult(
            document_id=replace_result.document_id,
            namespace_id=namespace_id,
            chunks_created=replace_result.chunks_created,
            entities_extracted=replace_result.entities_created + replace_result.entities_updated,
            relationships_created=replace_result.relationships_created,
            metadata={"replaced": True, "old_document_id": str(existing.id)},
        )

    def _validate_recall_results(
        self,
        chunks: list[tuple[Chunk, float]],
        query: str,
        *,
        min_content_length: int = 10,
    ) -> list[tuple[Chunk, float]]:
        """Validate and filter retrieval results.

        Removes duplicates, filters out empty content, ensures scores are normalized,
        and logs quality warnings.

        Args:
            chunks: List of (chunk, score) tuples
            query: Original query text for logging context
            min_content_length: Minimum content length to accept

        Returns:
            Validated and filtered list of (chunk, score) tuples
        """
        validated: list[tuple[Chunk, float]] = []
        seen_ids: set[UUID] = set()
        empty_count = 0
        duplicate_count = 0

        for chunk, score in chunks:
            if not isinstance(chunk, Chunk):
                logger.warning(f"Skipping non-Chunk result: {type(chunk)}")
                continue

            # Skip duplicates
            if chunk.id in seen_ids:
                duplicate_count += 1
                continue
            seen_ids.add(chunk.id)

            # Skip empty content
            if not chunk.content or len(chunk.content.strip()) < min_content_length:
                empty_count += 1
                continue

            # Normalize score to [0, 1]
            normalized_score = min(max(score, 0.0), 1.0)

            validated.append((chunk, normalized_score))

        # Log quality warnings
        if duplicate_count > 0:
            logger.debug(f"Recall validation: removed {duplicate_count} duplicate chunks for query: {query[:50]}...")
        if empty_count > 0:
            logger.warning(f"Recall validation: filtered {empty_count} empty/short chunks for query: {query[:50]}...")

        return validated

    @trace(
        "khora.vectorcypher.recall",
        include={"namespace_id", "limit", "mode"},
        result=lambda r: {"chunk_count": len(r.chunks), "entity_count": len(r.entities)},
    )
    async def recall(
        self,
        query: str,
        namespace_id: UUID,
        *,
        limit: int = 10,
        mode: SearchMode = SearchMode.HYBRID,
        min_similarity: float = 0.0,
        # VectorCypher-specific parameters
        temporal_filter: TemporalFilter | None = None,
        graph_depth: int | None = None,
        hybrid_alpha: float | None = None,
    ) -> RecallResult:
        """Recall memories relevant to a query using VectorCypher.

        Args:
            query: Query text
            namespace_id: Namespace to search
            limit: Maximum number of results
            mode: Search mode. VectorCypher implements all five honestly:
                ``VECTOR`` (vector channel only), ``GRAPH`` (graph expansion
                only - no vector chunks, no BM25), ``KEYWORD`` (BM25 only),
                ``HYBRID`` (vector-weighted RRF, the default), and ``ALL``
                (balanced fusion with ``hybrid_alpha=0.5``). Unsupported
                modes raise ``EngineCapabilityError``.
            min_similarity: Minimum similarity threshold
            temporal_filter: Temporal constraints
            graph_depth: Override graph traversal depth
            hybrid_alpha: Blend factor (0=graph, 1=vector)

        Returns:
            RecallResult with chunks, entities, and context
        """
        # #833: validate the mode contract before doing any storage work.
        if mode not in self.supported_modes:
            raise EngineCapabilityError("vectorcypher", mode, self.supported_modes)

        retriever = self._get_retriever()

        # Cascade temporal detection: Aho-Corasick dictionary -> (optional) semantic
        # Replaces the old regex + dateparser approach with categorized signals.
        # Always run temporal detection (dictionary-based, <10μs, deterministic);
        # temporal category detection is critical for recency weighting and sort order.
        temporal_signal: TemporalSignal | None = None
        if temporal_filter is not None:
            # API-asserted bounds: synthesize an EXPLICIT signal so downstream
            # behavior (skip-fallback in retriever, version filter, recency
            # weighting) treats the caller-supplied predicate as a high-confidence
            # temporal intent. source="api" disambiguates from
            # "dictionary"/"semantic"/"none" in traces.
            temporal_signal = TemporalSignal(
                is_temporal=True,
                category=TemporalCategory.EXPLICIT,
                confidence=1.0,
                source="api",
                temporal_filter=temporal_filter,
            )
            with trace_span("khora.vectorcypher.temporal_detect") as td_span:
                td_span.set_attribute("category", temporal_signal.category.value)
                td_span.set_attribute("confidence", temporal_signal.confidence)
                td_span.set_attribute("source", temporal_signal.source)
        else:
            with trace_span("khora.vectorcypher.temporal_detect") as td_span:
                detector = TemporalDetector()
                temporal_signal = detector.detect(query)
                td_span.set_attribute("category", temporal_signal.category.value)
                td_span.set_attribute("confidence", temporal_signal.confidence)
                td_span.set_attribute("source", temporal_signal.source)
                # EXPLICIT category produces a date-range TemporalFilter for pushdown
                if temporal_signal.temporal_filter is not None:
                    temporal_filter = temporal_signal.temporal_filter

                # Resolve relative dates ("last 7 days") to SQL-pushdown filter
                # for RECENCY / STATE_QUERY / CHANGE categories that the
                # EXPLICIT extractor can't handle.
                if temporal_filter is None and temporal_signal.is_temporal:
                    from khora.query.temporal_resolver import resolve_temporal_filter

                    temporal_filter = resolve_temporal_filter(query, temporal_signal)
                    if temporal_filter:
                        logger.debug(
                            "Resolved temporal filter: {} to {}",
                            temporal_filter.occurred_after,
                            temporal_filter.occurred_before,
                        )

        # Respect SearchMode.ALL: lower hybrid_alpha to give BM25 equal weight
        # with vector similarity, enabling keyword-based retrieval alongside
        # semantic search.  An explicit hybrid_alpha kwarg takes precedence.
        # Save and restore to avoid stateful side-effects on the shared config.
        original_alpha = retriever._config.hybrid_alpha
        if hybrid_alpha is not None:
            retriever._config.hybrid_alpha = hybrid_alpha
        elif mode == SearchMode.ALL:
            retriever._config.hybrid_alpha = 0.5

        # Use VectorCypher retriever
        try:
            result = await retriever.retrieve(
                query=query,
                namespace_id=namespace_id,
                temporal_filter=temporal_filter,
                temporal_signal=temporal_signal,
                graph_depth=graph_depth,
                limit=limit,
                min_similarity=min_similarity,
                mode=mode,
            )
        finally:
            retriever._config.hybrid_alpha = original_alpha

        # Validate and filter retrieval results
        validated_chunks = self._validate_recall_results(result.chunks, query)

        # Compute retrieval confidence signals for abstention calibration
        scores = [s for _, s in validated_chunks]
        if len(scores) >= 2:
            mean_score = sum(scores) / len(scores)
            score_variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            top_score_gap = scores[0] - scores[1]  # chunks are sorted by score
        elif len(scores) == 1:
            mean_score = scores[0]
            score_variance = 0.0
            top_score_gap = 0.0
        else:
            mean_score = 0.0
            score_variance = 0.0
            top_score_gap = 0.0

        recall_chunks = [
            RecallChunk(
                id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                score=score,
                created_at=chunk.created_at,
                occurred_at=chunk.source_timestamp,
                chunker_info=chunk.chunker_info or {},
            )
            for chunk, score in validated_chunks
        ]

        recall_entities = [
            RecallEntity(
                id=entity.id,
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description or "",
                score=score,
                attributes=dict(entity.attributes or {}),
                mention_count=entity.mention_count or 0,
                source_document_ids=list(entity.source_document_ids) or list((entity.source_documents or {}).keys()),
                source_chunk_ids=list(entity.source_chunk_ids),
            )
            for entity, score in result.entities
        ]

        recall_relationships = [
            RecallRelationship(
                id=rel.id,
                source_entity_id=rel.source_entity_id,
                target_entity_id=rel.target_entity_id,
                relationship_type=rel.relationship_type,
                description=rel.description or "",
                score=score,
                valid_from=rel.valid_from,
                valid_until=rel.valid_until,
                source_document_ids=list(rel.source_document_ids),
            )
            for rel, score in result.relationships
        ]

        # Document stubs — fuller projections land with the recall-method rewrite.
        seen_doc_ids: set[UUID] = set()
        documents: list[DocumentProjection] = []
        for chunk, _ in validated_chunks:
            if chunk.document_id in seen_doc_ids:
                continue
            seen_doc_ids.add(chunk.document_id)
            src = chunk.source_document
            documents.append(
                DocumentProjection(
                    id=chunk.document_id,
                    created_at=chunk.created_at,
                    source_type=(src.source_type if src and src.source_type else "library"),
                    title=(src.title if src and src.title else None),
                    source=(src.source if src and src.source else None),
                    source_timestamp=(src.source_timestamp if src else None),
                    metadata=dict(chunk.metadata or {}),
                )
            )
        for re_ in recall_entities:
            for did in re_.source_document_ids:
                if did in seen_doc_ids:
                    continue
                seen_doc_ids.add(did)
                documents.append(DocumentProjection(id=did, created_at=datetime.now(UTC), source_type="library"))
        for rr in recall_relationships:
            for did in rr.source_document_ids:
                if did in seen_doc_ids:
                    continue
                seen_doc_ids.add(did)
                documents.append(DocumentProjection(id=did, created_at=datetime.now(UTC), source_type="library"))

        # Canonical engine_info keys for the recall response.
        #
        # Use the engine's captured pre-rerank raw vector cosine
        # (``max_raw_vector_score``) for ``top_score_low``, NOT the
        # post-fusion ``validated_chunks[0][1]``. Cross-encoder reranking
        # compresses scores into a narrow high-side band even for
        # off-topic queries; mirroring the chronicle fix (#809). When the
        # vector channel is empty (graph-only recall), this is 0.0 and
        # the signal correctly flags as "low".
        max_raw_vector_score = float(result.metadata.get("max_raw_vector_score") or 0.0)
        abstention_signals = compute_abstention_signals(
            chunk_count=len(validated_chunks),
            top_vector_score=max_raw_vector_score,
            entity_count=len(recall_entities),
            # Hardcoded to ChronicleEngine defaults — same passive-signal semantics.
            min_chunks=1,
            min_top_score=0.3,
            combined_threshold=0.5,
        )

        if abstention_signals["entities_empty"]:
            _VC_ABSTENTION_SIGNAL_COUNTER.add(1, attributes={"signal": "entities_empty"})
        if abstention_signals["chunks_empty"]:
            _VC_ABSTENTION_SIGNAL_COUNTER.add(1, attributes={"signal": "chunks_empty"})
        if abstention_signals["chunks_below_min"]:
            _VC_ABSTENTION_SIGNAL_COUNTER.add(1, attributes={"signal": "chunks_below_min"})
        if abstention_signals["top_score_low"]:
            _VC_ABSTENTION_SIGNAL_COUNTER.add(1, attributes={"signal": "top_score_low"})
        _VC_ABSTENTION_COMBINED_SCORE_HISTOGRAM.record(abstention_signals["combined_score"])

        channels_used: list[str] = []
        if result.metadata.get("vector_chunk_count", 0) > 0:
            channels_used.append("vector")
        if result.metadata.get("graph_chunk_count", 0) > 0:
            channels_used.append("graph")
        if result.metadata.get("bm25_chunk_count", 0) > 0:
            channels_used.append("bm25")

        return RecallResult(
            query=query,
            namespace_id=namespace_id,
            documents=documents,
            chunks=recall_chunks,
            entities=recall_entities,
            relationships=recall_relationships,
            engine_info={
                "engine": "vectorcypher",
                "mode": mode.name.lower(),
                "channels_used": channels_used,
                "rrf_k": self._vc_config.fusion_rrf_k,
                "temporal_signal": (
                    {"category": temporal_signal.category.value, "source": temporal_signal.source}
                    if temporal_signal is not None
                    else {"category": "none", "source": "none"}
                ),
                "abstention_signals": abstention_signals,
                "routing": result.routing_decision.complexity.value,
                "use_graph": result.routing_decision.use_graph,
                "graph_depth": result.routing_decision.graph_depth,
                "raw_chunk_count": len(result.chunks),
                "validated_chunk_count": len(validated_chunks),
                # Temporal telemetry
                "temporal_category": temporal_signal.category.value if temporal_signal else None,
                "temporal_confidence": temporal_signal.confidence if temporal_signal else None,
                "is_temporal": temporal_signal.is_temporal if temporal_signal else False,
                # Retrieval confidence signals (for abstention calibration)
                "retrieval_mean_score": round(mean_score, 4),
                "retrieval_score_variance": round(score_variance, 6),
                "retrieval_top_score_gap": round(top_score_gap, 4),
                **result.metadata,
            },
        )

    async def forget(self, document_id: UUID, namespace_id: UUID | None) -> bool:
        """Remove a memory from the engine."""
        storage = self._get_storage()
        temporal_store = self._get_temporal_store()
        dual_nodes = self._get_dual_nodes()

        # namespace_id is required for IDOR-safe lookup (IDOR family). Callers
        # going through Khora.forget always resolve it before calling here;
        # bail loudly rather than allow a cross-tenant id probe.
        if namespace_id is None:
            logger.warning(f"Cannot forget document {document_id}: namespace_id is required")
            return False

        # Verify the document exists in this namespace before doing any work.
        document = await storage.get_document(document_id, namespace_id=namespace_id)
        if document is None:
            return False

        degradations = await self._cascade_forget_extraction(document_id, namespace_id)
        if degradations:
            logger.warning("forget cascade degraded: {}", degradations)

        # Delete from Neo4j (Chunk nodes and relationships)
        if dual_nodes is not None:
            await dual_nodes.delete_chunks_by_document(document_id, namespace_id)

        # Delete from pgvector
        await temporal_store.delete_chunks_by_document(document_id, namespace_id)

        # Delete from relational storage
        return await storage.delete_document(document_id, namespace_id=namespace_id)

    async def _cascade_forget_extraction(self, document_id: UUID, namespace_id: UUID) -> list[Degradation]:
        """Drop / decrement entities and relationships extracted from a document.

        Vector-anchored refcounting (#923): hard-deletes orphans (entities /
        relationships whose only ``source_document_ids`` entry is
        ``document_id``) and strips ``document_id`` from survivors' source
        arrays. Cleanup is anchored on whichever store actually holds the
        entities - the pgvector ``entities`` table on pg-backed stacks, the
        graph adapter tables on sqlite_lance / SurrealDB / Memgraph / Neptune
        / AGE - and mirrored opportunistically to the other store so the
        graph stays consistent. Runs on every backend, not just Neo4j.
        """
        storage = self._get_storage()
        return await cascade_forget_extraction(
            graph=storage.graph,
            vector=storage.vector,
            document_id=document_id,
            namespace_id=namespace_id,
            engine="khora.vectorcypher",
        )

    @staticmethod
    def _build_conversation_context(
        sorted_docs: list[dict[str, Any]],
    ) -> dict[int, str]:
        """Build context-enriched embedding text for conversation messages.

        For each message, creates text that includes ±2 neighboring messages
        as context. This helps embeddings capture conversational flow.

        Args:
            sorted_docs: Documents sorted by occurred_at timestamp.

        Returns:
            Dict mapping document index → enriched text for embedding.
        """
        context_map: dict[int, str] = {}
        n = len(sorted_docs)
        for i in range(n):
            parts: list[str] = []
            # ±2 neighbor context window
            for j in range(max(0, i - 2), min(n, i + 3)):
                if j == i:
                    continue
                neighbor = sorted_docs[j]
                author = neighbor.get("metadata", {}).get("author", "")
                content = neighbor.get("content", "")[:100]
                prefix = "prev" if j < i else "next"
                parts.append(f"{prefix}: {author}: {content}")

            current = sorted_docs[i].get("content", "")
            if parts:
                context_str = " | ".join(parts)
                context_map[i] = f"[Context: {context_str}]\n{current}"
            else:
                context_map[i] = current
        return context_map

    async def remember_batch(
        self,
        documents: list[dict[str, Any]],
        namespace_id: UUID,
        *,
        skill_name: str = "general_entities",
        expertise: ExpertiseConfig | str | None = None,
        extraction_model: str | None = None,
        max_concurrent: int = 20,
        deduplicate: bool = True,
        infer_relationships: bool = True,
        on_progress: Callable[[int, int], None] | None = None,
        entity_types: list[str],
        relationship_types: list[str],
        extraction_config_hash: str | None = None,
        chunk_strategy: ChunkStrategy | None = None,
        source_type: str = "library",
        source_name: str | None = None,
        source_url: str | None = None,
        source_timestamp: datetime | None = None,
        bulk_mode: bool = False,
    ) -> BatchResult:
        """Store multiple documents with automatic optimization.

        Uses a staged pipeline to batch API calls across documents:
          Stage 1: Dedup, create Document objects, chunk (parallel CPU)
          Stage 2: Batch-embed ALL chunks in one API call
          Stage 3: Store chunks to pgvector + Neo4j (parallel DB)
          Stage 4: Skeleton-select core chunks, extract entities (parallel LLM)
          Stage 5: Batch-embed ALL entities in one API call
          Stage 6: Batch store entities + relationships to Neo4j + pgvector

        Args:
            documents: List of document dicts with 'content', 'title', 'source', 'metadata'
            namespace_id: Namespace to store documents in
            skill_name: Extraction skill to use
            max_concurrent: Maximum concurrent document processing
            deduplicate: Whether to skip duplicate documents
            infer_relationships: Whether to infer relationships
            on_progress: Callback for progress updates
            entity_types: Entity types to extract
            relationship_types: Relationship types to extract
            chunk_strategy: Override chunking strategy for this call.
                Valid values: "fixed", "semantic", "recursive", "conversation".
                When None (default), uses the configured pipeline default.
            bulk_mode: If True, defer HNSW indexes during load and rebuild after

        Returns:
            BatchResult with processing statistics
        """
        if not documents:
            return BatchResult(total=0, processed=0, skipped=0, failed=0, chunks=0, entities=0, relationships=0)

        storage = self._get_storage()
        if bulk_mode:
            from khora.storage.optimize import prepare_for_bulk_load

            await prepare_for_bulk_load(storage)

        try:
            return await self._remember_batch_impl(
                documents,
                namespace_id,
                skill_name=skill_name,
                expertise=expertise,
                extraction_model=extraction_model,
                max_concurrent=max_concurrent,
                deduplicate=deduplicate,
                infer_relationships=infer_relationships,
                on_progress=on_progress,
                entity_types=entity_types,
                relationship_types=relationship_types,
                extraction_config_hash=extraction_config_hash,
                chunk_strategy=chunk_strategy,
                source_type=source_type,
                source_name=source_name,
                source_url=source_url,
                source_timestamp=source_timestamp,
            )
        finally:
            if bulk_mode:
                from khora.storage.optimize import ensure_hnsw_indexes

                await ensure_hnsw_indexes(storage)

    async def _remember_batch_impl(
        self,
        documents: list[dict[str, Any]],
        namespace_id: UUID,
        *,
        skill_name: str = "general_entities",
        expertise: ExpertiseConfig | str | None = None,
        extraction_model: str | None = None,
        max_concurrent: int = 20,
        deduplicate: bool = True,
        infer_relationships: bool = True,
        on_progress: Callable[[int, int], None] | None = None,
        entity_types: list[str],
        relationship_types: list[str],
        extraction_config_hash: str | None = None,
        chunk_strategy: ChunkStrategy | None = None,
        source_type: str = "library",
        source_name: str | None = None,
        source_url: str | None = None,
        source_timestamp: datetime | None = None,
    ) -> BatchResult:
        """Internal implementation of remember_batch (separated for bulk_mode wrapping)."""
        use_streaming = self._vc_config.streaming_pipeline
        if not use_streaming:
            # Legacy path: fall back to per-document processing
            return await self._remember_batch_legacy(
                documents,
                namespace_id,
                skill_name=skill_name,
                expertise=expertise,
                extraction_model=extraction_model,
                max_concurrent=max_concurrent,
                deduplicate=deduplicate,
                on_progress=on_progress,
                entity_types=entity_types,
                relationship_types=relationship_types,
                extraction_config_hash=extraction_config_hash,
                chunk_strategy=chunk_strategy,
                source_type=source_type,
                source_name=source_name,
                source_url=source_url,
                source_timestamp=source_timestamp,
            )

        from khora.extraction.chunkers import create_chunker
        from khora.pipelines.tasks.extract import extract_entities

        storage = self._get_storage()
        embedder = self._get_embedder()
        temporal_store = self._get_temporal_store()
        dual_nodes = self._get_dual_nodes()
        total = len(documents)

        results: dict[str, int] = {
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "chunks": 0,
            "entities": 0,
            "relationships": 0,
        }
        progress_count = 0

        def _report_progress(n: int = 1) -> None:
            nonlocal progress_count
            if on_progress:
                progress_count += n
                on_progress(progress_count, total)

        # ── Stage 0a: external_id dispatch ──────────────────────────────
        # Docs with an external_id that already exists in the namespace are
        # routed to the replace path via self.remember() — which detects the
        # same existing row and calls coordinator.replace_document_extraction.
        # Unmatched / absent external_id docs fall through to the streaming
        # pipeline below, unchanged.
        #
        # Batch the existence lookup: one ``get_documents_by_external_ids``
        # call replaces N serial ``get_document_by_external_id`` round-trips.
        external_id_handled: set[int] = set()
        ext_id_to_idx: dict[str, int] = {}
        for idx, doc_data in enumerate(documents):
            ext_id = doc_data.get("external_id")
            if ext_id is None or not isinstance(ext_id, str) or not ext_id.strip():
                continue
            # Keep the last-seen idx if an external_id repeats in the batch —
            # earlier duplicates fall through to Stage 0 checksum dedup.
            ext_id_to_idx[ext_id] = idx

        existing_by_ext_map: dict[str, Any] = {}
        if ext_id_to_idx:
            existing_by_ext_map = await storage.get_documents_by_external_ids(
                list(ext_id_to_idx.keys()), namespace_id=namespace_id
            )

        for ext_id, idx in ext_id_to_idx.items():
            existing_by_ext = existing_by_ext_map.get(ext_id)
            if existing_by_ext is None:
                continue
            doc_data = documents[idx]
            try:
                doc_metadata = doc_data.get("metadata", {})
                occurred_at = (
                    self._parse_datetime(doc_metadata["occurred_at"]) if "occurred_at" in doc_metadata else None
                )
                result = await self.remember(
                    doc_data.get("content", ""),
                    namespace_id,
                    title=doc_data.get("title", ""),
                    source=doc_data.get("source", ""),
                    source_type=doc_data.get("source_type", source_type),
                    source_name=doc_data.get("source_name", source_name),
                    source_url=doc_data.get("source_url", source_url),
                    source_timestamp=doc_data.get("source_timestamp", source_timestamp),
                    metadata=doc_metadata,
                    skill_name=skill_name,
                    expertise=expertise,
                    extraction_model=extraction_model,
                    occurred_at=occurred_at,
                    entity_types=entity_types,
                    relationship_types=relationship_types,
                    extraction_config_hash=extraction_config_hash,
                    chunk_strategy=chunk_strategy,
                    external_id=ext_id,
                )
                results["processed"] += 1
                results["chunks"] += result.chunks_created
                results["entities"] += result.entities_extracted
                results["relationships"] += result.relationships_created
            except Exception as e:
                logger.error(f"Failed to replace document external_id={ext_id!r}: {e}")
                results["failed"] += 1
            external_id_handled.add(idx)
            _report_progress()

        # ── Stage 0: Dedup ──────────────────────────────────────────────
        _stage0_t0 = _time.perf_counter()
        doc_checksums = [hashlib.sha256(d.get("content", "").encode("utf-8")).hexdigest() for d in documents]
        existing_docs: dict[str, Any] = {}
        if deduplicate:
            existing_docs = await storage.get_documents_by_checksums(namespace_id, doc_checksums)

        # Filter to non-duplicate documents, preserving original index.
        # Docs already dispatched via external_id above are excluded here.
        checksums_seen: set[str] = set()
        active_indices: list[int] = []
        for idx, checksum in enumerate(doc_checksums):
            if idx in external_id_handled:
                continue
            if checksum in checksums_seen or (deduplicate and checksum in existing_docs):
                results["skipped"] += 1
                _report_progress()
            else:
                checksums_seen.add(checksum)
                active_indices.append(idx)
        _stage0_ms = (_time.perf_counter() - _stage0_t0) * 1000

        if not active_indices:
            return BatchResult(total=total, **results)

        # ── Conversation mode detection ─────────────────────────────────
        docs_with_ts = sum(1 for d in documents if "occurred_at" in d.get("metadata", {}))
        avg_content_len = sum(len(d.get("content", "")) for d in documents) / max(total, 1)
        context_by_orig: dict[int, str] = {}
        is_conversation_mode = False
        if docs_with_ts > total * 0.5 and avg_content_len < 200:
            indexed_docs = list(enumerate(documents))
            indexed_docs.sort(key=lambda x: x[1].get("metadata", {}).get("occurred_at", ""))
            sorted_docs = [d for _, d in indexed_docs]
            conversation_context = self._build_conversation_context(sorted_docs)
            orig_to_sorted = {orig_idx: sort_idx for sort_idx, (orig_idx, _) in enumerate(indexed_docs)}
            context_by_orig = {
                orig_idx: conversation_context[sort_idx]
                for orig_idx, sort_idx in orig_to_sorted.items()
                if sort_idx in conversation_context
            }
            is_conversation_mode = bool(context_by_orig)
            logger.info(
                f"Conversation mode detected: {docs_with_ts}/{total} docs with timestamps, "
                f"avg content {avg_content_len:.0f} chars, enriching embeddings"
            )

        skeleton_ratio = self._vc_config.conversation_skeleton_ratio if is_conversation_mode else None

        # ── Stage 1: Create documents + chunk in parallel (CPU) ─────────
        _stage1_t0 = _time.perf_counter()

        strategy = chunk_strategy if chunk_strategy is not None else self._config.pipeline.chunking_strategy
        chunker = create_chunker(
            strategy=strategy,
            chunk_size=self._config.pipeline.chunk_size,
            chunk_overlap=self._config.pipeline.chunk_overlap,
        )

        @dataclass
        class _DocState:
            idx: int
            doc_data: dict[str, Any]
            checksum: str
            document: Document | None = None
            raw_chunks: list = field(default_factory=list)
            embed_texts: list[str] = field(default_factory=list)
            occurred_at: datetime | None = None
            failed: bool = False

        doc_states: list[_DocState] = []
        sem = asyncio.Semaphore(max_concurrent)

        async def _create_and_chunk(idx: int) -> _DocState:
            doc_data = documents[idx]
            checksum = doc_checksums[idx]
            state = _DocState(idx=idx, doc_data=doc_data, checksum=checksum)
            try:
                doc_metadata = doc_data.get("metadata", {})
                if "occurred_at" in doc_metadata:
                    state.occurred_at = self._parse_datetime(doc_metadata["occurred_at"])

                async with sem:
                    document = Document(
                        namespace_id=namespace_id,
                        content=doc_data.get("content", ""),
                        title=doc_data.get("title") or None,
                        source=doc_data.get("source") or None,
                        checksum=checksum,
                        source_type=doc_data.get("source_type", source_type),
                        source_name=doc_data.get("source_name", source_name) or None,
                        source_url=doc_data.get("source_url", source_url) or None,
                        source_timestamp=doc_data.get("source_timestamp", source_timestamp),
                        metadata=dict(doc_metadata),
                        extraction_config_hash=extraction_config_hash,
                        external_id=doc_data.get("external_id"),
                        session_id=_coerce_session_id_from_metadata(doc_metadata),
                    )
                    document = await storage.create_document(document)
                    state.document = document

                    raw_chunks = await asyncio.to_thread(chunker.chunk, document.content)
                    state.raw_chunks = raw_chunks

                    # Determine embedding text (conversation context or raw chunk content)
                    if idx in context_by_orig:
                        state.embed_texts = [context_by_orig[idx]]
                    else:
                        state.embed_texts = [c.content for c in raw_chunks]
            except Exception as e:
                logger.error(f"Failed to create/chunk document {idx}: {e}")
                state.failed = True
            return state

        doc_states = await asyncio.gather(*[_create_and_chunk(idx) for idx in active_indices])

        # Separate successful from failed
        failed_states = [s for s in doc_states if s.failed or not s.raw_chunks]
        ok_states = [s for s in doc_states if not s.failed and s.raw_chunks]
        for s in failed_states:
            if s.failed:
                results["failed"] += 1
            elif not s.raw_chunks and s.document:
                s.document.mark_completed(0, 0)
                await storage.update_document(s.document)
                results["processed"] += 1
            _report_progress()

        _stage1_ms = (_time.perf_counter() - _stage1_t0) * 1000

        if not ok_states:
            return BatchResult(total=total, **results)

        # ── Build processing windows ─────────────────────────────────────
        # When max_chunks_in_flight is set, group documents into windows so
        # that the total chunk count per window stays ≤ the limit.  Document
        # boundaries are respected: a document's chunks are never split across
        # windows.  When the limit is None (default), a single window holds
        # all documents (current behaviour, backward-compatible).
        max_cif = self._vc_config.max_chunks_in_flight
        if max_cif is None:
            windows: list[list[_DocState]] = [ok_states]
        else:
            windows = []
            _win: list[_DocState] = []
            _win_count = 0
            for _s in ok_states:
                _n = len(_s.raw_chunks)
                if _n > max_cif:
                    logger.warning(
                        f"Document idx={_s.idx} has {_n} chunks which exceeds "
                        f"max_chunks_in_flight={max_cif}; processing as single-document window."
                    )
                if _win and _win_count + _n > max_cif:
                    windows.append(_win)
                    _win = []
                    _win_count = 0
                _win.append(_s)
                _win_count += _n
            if _win:
                windows.append(_win)
            logger.debug(
                f"Windowed processing: {len(ok_states)} docs → {len(windows)} windows (max_chunks_in_flight={max_cif})"
            )

        # Accumulated timing across windows
        _stage2_ms = 0.0
        _stage3_ms = 0.0
        _stage4_ms = 0.0
        _stage5_ms = 0.0
        _stage6_upsert_ms = 0.0
        _stage6_rels_ms = 0.0
        _stage6_links_ms = 0.0

        # Track unique entity keys across windows to avoid double-counting entities
        # that appear in multiple windows (upsert_entities_batch ensures a single DB
        # row, so BatchResult.entities must reflect unique persisted cardinality).
        _seen_entity_keys: set[tuple[str, str]] = set()

        for window_states in windows:
            # ── Stage 2: Batch-embed ALL chunk texts ────────────────────────
            _t0 = _time.perf_counter()

            # Collect all texts with provenance tracking
            all_embed_texts: list[str] = []
            text_offsets: list[tuple[int, int]] = []  # (state_index, start_offset) into all_embed_texts
            for si, state in enumerate(window_states):
                text_offsets.append((si, len(all_embed_texts)))
                all_embed_texts.extend(state.embed_texts)

            logger.debug(f"Batch embedding {len(all_embed_texts)} texts across {len(window_states)} documents")
            all_embeddings = await embedder.embed_batch(all_embed_texts)

            _stage2_ms += (_time.perf_counter() - _t0) * 1000

            # ── Stage 3: Build TemporalChunks + store to pgvector + Neo4j ───
            _t0 = _time.perf_counter()

            # Build TemporalChunk objects with pre-computed embeddings
            all_temporal_chunks: list[TemporalChunk] = []
            state_chunk_ranges: list[tuple[int, int]] = []  # (start, end) indices into all_temporal_chunks

            for si, state in enumerate(window_states):
                doc = state.document
                assert doc is not None
                doc_metadata = doc.metadata or {}
                occurred = state.occurred_at or datetime.now(UTC)

                start_idx = len(all_temporal_chunks)
                _, embed_offset = text_offsets[si]

                for ci, raw_chunk in enumerate(state.raw_chunks):
                    embedding = all_embeddings[embed_offset + min(ci, len(state.embed_texts) - 1)]
                    tc = TemporalChunk(
                        id=None,
                        namespace_id=doc.namespace_id,
                        document_id=doc.id,
                        content=raw_chunk.content,
                        embedding=embedding,
                        occurred_at=occurred,
                        created_at=datetime.now(UTC),
                        source_system=doc_metadata.get("source_system"),
                        author=doc_metadata.get("author"),
                        channel=doc_metadata.get("channel") or doc_metadata.get("thread_id"),
                        tags=_ensure_tags(doc_metadata.get("tags", [])),
                        confidence=1.0,
                        metadata={
                            **doc_metadata,
                            "chunk_index": ci,
                            "start_char": raw_chunk.start_char if hasattr(raw_chunk, "start_char") else 0,
                            "end_char": raw_chunk.end_char
                            if hasattr(raw_chunk, "end_char")
                            else len(raw_chunk.content),
                        },
                        chunker_info=dict(raw_chunk.metadata),
                    )
                    all_temporal_chunks.append(tc)

                state_chunk_ranges.append((start_idx, len(all_temporal_chunks)))

            # Batch store to pgvector
            stored = await temporal_store.create_chunks_batch(all_temporal_chunks)
            for i, s in enumerate(stored):
                all_temporal_chunks[i].id = s.id

            # Batch create Neo4j chunk nodes (skipped for SurrealDB)
            if dual_nodes is not None:
                await dual_nodes.create_chunk_nodes_batch(all_temporal_chunks, namespace_id)

            _stage3_ms += (_time.perf_counter() - _t0) * 1000

            # ── Stage 4: Skeleton extraction across ALL documents ───────────
            _t0 = _time.perf_counter()

            all_entities: list[Entity] = []
            all_relationships: list[Relationship] = []
            all_entity_chunk_links: list[EntityChunkLink] = []

            if self._config.pipeline.extract_entities:
                # Collect all chunks across documents for batch skeleton extraction
                all_core_chunk_objects: list[Chunk] = []

                for si, state in enumerate(window_states):
                    start, end = state_chunk_ranges[si]
                    doc_chunks = all_temporal_chunks[start:end]

                    if not doc_chunks:
                        continue

                    # Skeleton selection per document (maintains document-level PageRank semantics).
                    # Skip min_tokens gate for conversation mode — short messages are the norm
                    # and should always reach extraction with skeleton_ratio=0.90.
                    if not is_conversation_mode:
                        min_tokens = self._vc_config.min_extraction_tokens
                        if min_tokens > 0 and all(len(c.content.split()) <= min_tokens for c in doc_chunks):
                            continue

                    if len(doc_chunks) <= 2:
                        core_ids = {c.id for c in doc_chunks}
                    else:
                        effective_ratio = skeleton_ratio or self._vc_config.skeleton_core_ratio
                        skeleton = SkeletonIndexer(core_ratio=effective_ratio)
                        skeleton.add_chunks_batch(doc_chunks)
                        core_ids = await asyncio.to_thread(skeleton.build_skeleton)

                    for tc in doc_chunks:
                        if tc.id in core_ids:
                            all_core_chunk_objects.append(
                                Chunk(
                                    id=tc.id,
                                    namespace_id=tc.namespace_id,
                                    document_id=tc.document_id,
                                    content=tc.content,
                                    created_at=tc.created_at or tc.occurred_at,
                                    chunker_info=dict(tc.chunker_info or {}),
                                )
                            )

                if all_core_chunk_objects:
                    model = extraction_model or self._config.llm.model

                    if is_conversation_mode:
                        # In conversation mode, extract per-document to match the
                        # old pipeline's behaviour.  The old code called
                        # extract_entities once per document (1-2 chunks each),
                        # which produced more entities because the LLM saw each
                        # message in isolation.  Batching all chunks together
                        # causes cross-document entity deduplication that drops
                        # ~60% of entities for short conversation messages.
                        from collections import defaultdict

                        doc_chunks_map: dict[UUID, list[Chunk]] = defaultdict(list)
                        for chunk in all_core_chunk_objects:
                            doc_chunks_map[chunk.document_id].append(chunk)

                        # Map document_id -> occurred_at for temporal context in extraction
                        doc_occurred_at: dict[UUID, datetime | None] = {}
                        for state in window_states:
                            if state.document is not None:
                                doc_occurred_at[state.document.id] = state.occurred_at

                        logger.debug(
                            f"Conversation extraction: {len(all_core_chunk_objects)} chunks "
                            f"across {len(doc_chunks_map)} documents (per-document mode)"
                        )

                        per_doc_entities: list[Entity] = []
                        per_doc_relationships: list[Relationship] = []
                        sem = asyncio.Semaphore(self._vc_config.max_concurrent_extractions)

                        async def _extract_one_doc(
                            chunks: list[Chunk], occurred_at: datetime | None = None
                        ) -> tuple[list, list]:
                            ctx = {"document_created_at": occurred_at.isoformat()} if occurred_at else None
                            async with sem:
                                return await extract_entities(
                                    chunks,
                                    skill_name=skill_name,
                                    expertise=expertise,
                                    model=model,
                                    max_concurrent=1,
                                    context=ctx,
                                    timeout=self._config.llm.timeout,
                                    max_tokens=self._config.llm.max_tokens,
                                    extraction_batch_size=self._vc_config.extraction_batch_size,
                                    entity_types=entity_types,
                                    relationship_types=relationship_types,
                                    store_events=self._vc_config.store_events,
                                )

                        extraction_results = await asyncio.gather(
                            *[
                                _extract_one_doc(cks, doc_occurred_at.get(doc_id))
                                for doc_id, cks in doc_chunks_map.items()
                            ]
                        )
                        for ents, rels in extraction_results:
                            per_doc_entities.extend(ents)
                            per_doc_relationships.extend(rels)

                        entities = per_doc_entities
                        relationships = per_doc_relationships
                    else:
                        logger.debug(
                            f"Batch extraction: {len(all_core_chunk_objects)} core chunks from {len(window_states)} documents"
                        )
                        entities, relationships = await extract_entities(
                            all_core_chunk_objects,
                            skill_name=skill_name,
                            expertise=expertise,
                            model=model,
                            max_concurrent=self._vc_config.max_concurrent_extractions,
                            timeout=self._config.llm.timeout,
                            max_tokens=self._config.llm.max_tokens,
                            extraction_batch_size=self._vc_config.extraction_batch_size,
                            entity_types=entity_types,
                            relationship_types=relationship_types,
                            store_events=self._vc_config.store_events,
                        )

                    if entities:
                        all_entities = list(entities)
                        all_relationships = list(relationships)

                        # Build entity-chunk links
                        for entity in all_entities:
                            for chunk_id in entity.source_chunk_ids:
                                all_entity_chunk_links.append(EntityChunkLink(entity_id=entity.id, chunk_id=chunk_id))

                        # Co-occurrence relationships
                        cooccurrence_rels = _build_cooccurrence_relationships(
                            all_entities, all_core_chunk_objects, namespace_id, all_relationships
                        )
                        if cooccurrence_rels:
                            all_relationships.extend(cooccurrence_rels)

            _stage4_ms += (_time.perf_counter() - _t0) * 1000

            # ── Stage 5: Batch-embed ALL entity texts ───────────────────────
            _t0 = _time.perf_counter()

            if all_entities:
                entity_texts = [f"{e.name}: {e.description}" if e.description else e.name for e in all_entities]
                entity_embeddings = await embedder.embed_batch(entity_texts)
                for entity, emb in zip(all_entities, entity_embeddings):
                    entity.embedding = emb
                    entity.embedding_model = embedder.model_name

            _stage5_ms += (_time.perf_counter() - _t0) * 1000

            # ── Stage 6: Batch store entities + relationships ───────────────
            if all_entities:
                # Track the discarded-or-mutated entity IDs so we can
                # remap ``all_relationships`` endpoints below (#806).
                # Both the cross-document dedup AND the upsert mutate
                # ``entity.id``; without remapping, relationships built
                # from a window's throwaway extraction-time UUID become
                # FK violations (sqlite_lance) or silent MATCH drops
                # (Neo4j).
                id_remap: dict[str, str] = {}

                # Cross-document entity dedup by normalized name:type
                if self._vc_config.enable_smart_resolution:
                    from khora._accel import normalize_entity_name

                    deduped: dict[str, Entity] = {}
                    for entity in all_entities:
                        key = f"{normalize_entity_name(entity.name)}:{entity.entity_type}"
                        if key in deduped:
                            existing = deduped[key]
                            # Record the discard so dependent rels remap.
                            if entity.id != existing.id:
                                id_remap[str(entity.id)] = str(existing.id)
                            existing.mention_count += entity.mention_count
                            for doc_id in entity.source_document_ids:
                                if doc_id not in existing.source_document_ids:
                                    existing.source_document_ids.append(doc_id)
                            for chunk_id in entity.source_chunk_ids:
                                if chunk_id not in existing.source_chunk_ids:
                                    existing.source_chunk_ids.append(chunk_id)
                        else:
                            deduped[key] = entity
                    all_entities = list(deduped.values())
                    logger.debug(f"Cross-document dedup: {len(deduped)} unique entities")

                    # Rebuild entity-chunk links after dedup: the pre-dedup links
                    # reference UUIDs of discarded entities, causing MATCH failures
                    # in Neo4j (silent MENTIONED_IN edge loss).  Surviving entities
                    # already carry the merged source_chunk_ids from all duplicates.
                    all_entity_chunk_links = [
                        EntityChunkLink(entity_id=entity.id, chunk_id=chunk_id)
                        for entity in all_entities
                        for chunk_id in entity.source_chunk_ids
                    ]

                # Snapshot pre-upsert IDs so the post-upsert mutation
                # also lands in the remap.
                pre_upsert_ids = [str(e.id) for e in all_entities]

                _t0 = _time.perf_counter()
                await storage.upsert_entities_batch(namespace_id, all_entities)
                _stage6_upsert_ms += (_time.perf_counter() - _t0) * 1000

                # Extend id_remap with post-upsert canonicalisations.
                # Compose with the dedup pass: if dedup said X -> Y and
                # upsert then said Y -> Z, the relationship endpoint
                # X must end up as Z.
                for pre_id, entity in zip(pre_upsert_ids, all_entities):
                    canonical_id = str(entity.id)
                    if pre_id != canonical_id:
                        id_remap[pre_id] = canonical_id
                if id_remap:
                    for src, tgt in list(id_remap.items()):
                        if tgt in id_remap:
                            id_remap[src] = id_remap[tgt]

                # Rebuild entity-chunk links after upsert: upsert_entities_batch()
                # mutates entity.id in-place to the DB's canonical UUID when the entity
                # already exists (e.g., cross-window collision).  Links built before this
                # call carry pre-mutation UUIDs and cause silent MENTIONED_IN edge loss.
                all_entity_chunk_links = [
                    EntityChunkLink(entity_id=entity.id, chunk_id=chunk_id)
                    for entity in all_entities
                    for chunk_id in entity.source_chunk_ids
                ]

                # Apply the dedup + canonical-id remap to relationships
                # captured during window extraction (#806).
                if id_remap and all_relationships:
                    from uuid import UUID as _UUID

                    for rel in all_relationships:
                        src_str = str(rel.source_entity_id)
                        tgt_str = str(rel.target_entity_id)
                        if src_str in id_remap:
                            rel.source_entity_id = _UUID(id_remap[src_str])
                        if tgt_str in id_remap:
                            rel.target_entity_id = _UUID(id_remap[tgt_str])

                if all_relationships:
                    _t0 = _time.perf_counter()
                    await storage.create_relationships_batch(all_relationships)
                    _stage6_rels_ms += (_time.perf_counter() - _t0) * 1000

                if all_entity_chunk_links and dual_nodes is not None:
                    _t0 = _time.perf_counter()
                    await dual_nodes.link_entities_to_chunks_batch(all_entity_chunk_links)
                    _stage6_links_ms += (_time.perf_counter() - _t0) * 1000

                logger.info(
                    f"Streaming pipeline batch store: {len(all_entities)} entities, "
                    f"{len(all_relationships)} relationships, {len(all_entity_chunk_links)} links"
                )

            # ── Update document statuses + fire on_progress (per window) ────
            for si, state in enumerate(window_states):
                doc = state.document
                assert doc is not None
                start, end = state_chunk_ranges[si]
                chunks_created = end - start
                # Count entities from this document's chunks
                doc_chunk_ids = {all_temporal_chunks[i].id for i in range(start, end)}
                doc_entity_count = sum(
                    1 for e in all_entities if any(cid in doc_chunk_ids for cid in e.source_chunk_ids)
                )
                doc_relationship_count = sum(
                    1 for r in all_relationships if any(cid in doc_chunk_ids for cid in r.source_chunk_ids)
                )
                doc.mark_completed(chunks_created, doc_entity_count, doc_relationship_count)
                await storage.update_document(doc)
                results["processed"] += 1
                results["chunks"] += chunks_created
                _report_progress()

            new_entity_count = 0
            for _e in all_entities:
                _key = (_e.name, _e.entity_type)
                if _key not in _seen_entity_keys:
                    _seen_entity_keys.add(_key)
                    new_entity_count += 1
            results["entities"] += new_entity_count
            results["relationships"] += len(all_relationships)
            # end of window loop

        _stage6_total_ms = _stage6_upsert_ms + _stage6_rels_ms + _stage6_links_ms
        with trace_span(
            "khora.vectorcypher.remember_batch",
            document_count=total,
            processed=results["processed"],
            skipped=results["skipped"],
            failed=results["failed"],
            chunks=results["chunks"],
            entities=results["entities"],
            relationships=results["relationships"],
            stage0_dedup_ms=round(_stage0_ms, 2),
            stage1_chunk_ms=round(_stage1_ms, 2),
            stage2_embed_chunks_ms=round(_stage2_ms, 2),
            stage3_store_chunks_ms=round(_stage3_ms, 2),
            stage4_extraction_ms=round(_stage4_ms, 2),
            stage5_embed_entities_ms=round(_stage5_ms, 2),
            stage6_store_entities_ms=round(_stage6_total_ms, 2),
        ):
            pass

        logger.info(
            f"Staged pipeline: dedup={_stage0_ms:.0f}ms, chunk={_stage1_ms:.0f}ms, "
            f"embed_chunks={_stage2_ms:.0f}ms, store_chunks={_stage3_ms:.0f}ms, "
            f"extract={_stage4_ms:.0f}ms, embed_entities={_stage5_ms:.0f}ms, "
            f"store_entities={_stage6_total_ms:.0f}ms"
        )

        return BatchResult(
            total=total,
            processed=results["processed"],
            skipped=results["skipped"],
            failed=results["failed"],
            chunks=results["chunks"],
            entities=results["entities"],
            relationships=results["relationships"],
        )

    async def _remember_batch_legacy(
        self,
        documents: list[dict[str, Any]],
        namespace_id: UUID,
        *,
        skill_name: str = "general_entities",
        expertise: ExpertiseConfig | str | None = None,
        extraction_model: str | None = None,
        max_concurrent: int = 20,
        deduplicate: bool = True,
        on_progress: Callable[[int, int], None] | None = None,
        entity_types: list[str],
        relationship_types: list[str],
        extraction_config_hash: str | None = None,
        chunk_strategy: ChunkStrategy | None = None,
        source_type: str = "library",
        source_name: str | None = None,
        source_url: str | None = None,
        source_timestamp: datetime | None = None,
    ) -> BatchResult:
        """Legacy per-document remember_batch (non-streaming pipeline)."""
        storage = self._get_storage()
        total = len(documents)
        results: dict[str, int] = {
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "chunks": 0,
            "entities": 0,
            "relationships": 0,
        }
        results_lock = asyncio.Lock()
        progress_count = 0
        progress_lock = asyncio.Lock()
        doc_checksums = [hashlib.sha256(d.get("content", "").encode("utf-8")).hexdigest() for d in documents]
        existing_docs: dict[str, Any] = {}
        if deduplicate:
            existing_docs = await storage.get_documents_by_checksums(namespace_id, doc_checksums)
        checksums_in_flight: set[str] = set()
        checksums_lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_document(doc_data: dict[str, Any], checksum: str) -> None:
            nonlocal progress_count
            async with checksums_lock:
                if checksum in checksums_in_flight:
                    async with results_lock:
                        results["skipped"] += 1
                    return
                checksums_in_flight.add(checksum)
            if deduplicate and checksum in existing_docs:
                async with results_lock:
                    results["skipped"] += 1
                if on_progress:
                    async with progress_lock:
                        progress_count += 1
                        on_progress(progress_count, total)
                return
            async with semaphore:
                try:
                    doc_metadata = doc_data.get("metadata", {})
                    occurred_at = None
                    if "occurred_at" in doc_metadata:
                        occurred_at = self._parse_datetime(doc_metadata["occurred_at"])
                    result = await self.remember(
                        doc_data.get("content", ""),
                        namespace_id,
                        title=doc_data.get("title", ""),
                        source=doc_data.get("source", ""),
                        source_type=doc_data.get("source_type", source_type),
                        source_name=doc_data.get("source_name", source_name),
                        source_url=doc_data.get("source_url", source_url),
                        source_timestamp=doc_data.get("source_timestamp", source_timestamp),
                        metadata=doc_metadata,
                        skill_name=skill_name,
                        expertise=expertise,
                        extraction_model=extraction_model,
                        occurred_at=occurred_at,
                        entity_types=entity_types,
                        relationship_types=relationship_types,
                        extraction_config_hash=extraction_config_hash,
                        chunk_strategy=chunk_strategy,
                        external_id=doc_data.get("external_id"),
                    )
                    async with results_lock:
                        if result.metadata.get("duplicate"):
                            results["skipped"] += 1
                        else:
                            results["processed"] += 1
                            results["chunks"] += result.chunks_created
                            results["entities"] += result.entities_extracted
                            results["relationships"] += result.relationships_created
                except Exception as e:
                    logger.error(f"Failed to process document: {e}")
                    async with results_lock:
                        results["failed"] += 1
            if on_progress:
                async with progress_lock:
                    progress_count += 1
                    on_progress(progress_count, total)

        await asyncio.gather(*[process_document(doc, checksum) for doc, checksum in zip(documents, doc_checksums)])
        return BatchResult(total=total, **results)

    # Compiled regex for lightweight temporal keyword detection
    _TEMPORAL_KW_RE = re.compile(
        r"\b(when|before|after|during|since|until|last\s+(?:week|month|year|night|time)"
        r"|yesterday|today|recently|earlier|latest|newest|oldest|first|most\s+recent"
        r"|in\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)"
        r"|in\s+\d{4}|on\s+\d{1,2}[/\-]|ago)\b",
        re.IGNORECASE,
    )
    _DATE_EXTRACT_RE = re.compile(
        r"(\d{4}[/\-]\d{1,2}[/\-]\d{1,2})"
        r"|(\b(?:january|february|march|april|may|june|july|august|september|october|november|december)"
        r"\s+\d{1,2},?\s+\d{4}\b)"
        r"|(\b\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)"
        r"\s+\d{4}\b)",
        re.IGNORECASE,
    )

    def _detect_temporal_filter(self, query: str) -> TemporalFilter | None:
        """Lightweight regex-based temporal detection — no LLM call.

        Returns a TemporalFilter if temporal keywords and parseable dates
        are found, otherwise None. Cost: ~0.25ms.
        """
        if not self._TEMPORAL_KW_RE.search(query):
            return None

        # Try to extract an explicit date from the query
        date_match = self._DATE_EXTRACT_RE.search(query)
        if date_match:
            date_str = date_match.group(0)
            try:
                parsed_dt = self._parse_datetime(date_str)
                # "before" / "after" / default to "around that date" (±30 days)
                query_lower = query.lower()
                if "before" in query_lower:
                    return TemporalFilter(occurred_before=parsed_dt)
                elif "after" in query_lower or "since" in query_lower:
                    return TemporalFilter(occurred_after=parsed_dt)
                else:
                    # Within ±30 days of the mentioned date
                    from datetime import timedelta

                    return TemporalFilter(
                        occurred_after=parsed_dt - timedelta(days=30),
                        occurred_before=parsed_dt + timedelta(days=30),
                    )
            except ValueError:
                pass

        # Temporal keywords detected but no parseable date — signal to retriever
        # via a marker filter that enables recency boosting
        return None

    def _parse_datetime(self, value: Any) -> datetime:
        """Parse a datetime value from various formats."""
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=UTC)
            return value
        if isinstance(value, str):
            try:
                return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
            except ValueError:
                pass
            try:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                return dt
            except ValueError:
                pass
            # LongMemEval format: "2023/04/10 (Mon) 17:50"
            for fmt in (
                "%Y/%m/%d (%a) %H:%M",
                "%Y/%m/%d %H:%M",
                "%Y/%m/%d",
                "%B %d, %Y",
            ):
                try:
                    return datetime.strptime(value, fmt).replace(tzinfo=UTC)
                except ValueError:
                    continue
            # Last-resort: dateparser (handles a wide variety of natural-language dates)
            try:
                import dateparser

                dt = dateparser.parse(value, settings={"RETURN_AS_TIMEZONE_AWARE": True})
                if dt is not None:
                    return dt
            except Exception as e:
                logger.debug(f"dateparser failed for '{value}': {e}")
        raise ValueError(f"Cannot parse datetime: {value}")

    # =========================================================================
    # Namespace Management
    # =========================================================================

    async def create_namespace(
        self,
        *,
        config_overrides: dict[str, Any] | None = None,
    ) -> MemoryNamespace:
        """Create a new memory namespace."""
        namespace = MemoryNamespace(
            config_overrides=config_overrides or {},
        )
        return await self._get_storage().create_namespace(namespace)

    async def get_namespace(self, namespace_id: UUID) -> MemoryNamespace | None:
        """Get a namespace by ID."""
        return await self._get_storage().get_namespace(namespace_id)

    # =========================================================================
    # Entity Operations
    # =========================================================================

    async def get_entity(self, entity_id: UUID, *, namespace_id: UUID) -> Entity | None:
        """Get an entity by ID, scoped to ``namespace_id``."""
        return await self._get_storage().get_entity(entity_id, namespace_id=namespace_id)

    async def list_entities(
        self,
        namespace_id: UUID,
        *,
        entity_type: str | None = None,
        limit: int = 100,
    ) -> list[Entity]:
        """List entities in a namespace."""
        return await self._get_storage().list_entities(namespace_id, entity_type=entity_type, limit=limit)

    @trace("khora.find_related_entities", result=lambda r: {"result_count": len(r)})
    async def find_related_entities(
        self,
        entity_id: UUID,
        namespace_id: UUID,
        *,
        max_depth: int = 2,
        limit: int = 20,
    ) -> list[tuple[Entity, float]]:
        """Find entities related to a given entity via graph traversal."""
        dual_nodes = self._get_dual_nodes()
        if dual_nodes is None:
            # Graph-only backends (sqlite_lance, surrealdb): no chunk-entity
            # dual graph, so traversal goes through the GraphBackendProtocol
            # directly. ``get_neighborhood`` returns the seed plus connected
            # entities and relationships up to ``depth``; we BFS the returned
            # relationships to recover per-entity distance from the seed,
            # mirroring the Neo4j Path A scoring ``1 / (1 + distance)``.
            storage = self._get_storage()
            if storage.graph is None:
                return []
            neighborhood = await storage.graph.get_neighborhood(
                entity_id,
                namespace_id=namespace_id,
                depth=max_depth,
                limit=limit,
            )
            entities = neighborhood.get("entities", [])
            relationships = neighborhood.get("relationships", [])
            distances = _bfs_distances_from(entity_id, relationships)
            results: list[tuple[Entity, float]] = []
            for e in entities:
                if e.id == entity_id:
                    continue
                d = distances.get(e.id, 1)
                results.append((e, 1.0 / (1 + d)))
            results.sort(key=lambda pair: pair[1], reverse=True)
            return results[:limit]

        neighborhoods = await dual_nodes.get_entity_neighborhoods(
            entity_ids=[entity_id],
            namespace_id=namespace_id,
            depth=max_depth,
            limit_per_entity=limit,
        )

        results: list[tuple[Entity, float]] = []
        entity_infos = neighborhoods.get(str(entity_id), [])

        for info in entity_infos[:limit]:
            entity = await self._get_storage().get_entity(UUID(info["id"]), namespace_id=namespace_id)
            if entity:
                score = 1.0 / (1 + info.get("distance", 1))
                results.append((entity, score))

        return results

    @trace("khora.search_entities", exclude={"query"}, result=lambda r: {"result_count": len(r)})
    async def search_entities(
        self,
        query: str,
        namespace_id: UUID,
        *,
        limit: int = 10,
    ) -> list[Entity]:
        """Search entities by query text using embedding similarity."""
        embedder = self._get_embedder()
        query_embedding = await embedder.embed(query)

        # Search via storage coordinator
        storage = self._get_storage()
        entity_ids_scores = await storage.search_similar_entities(
            namespace_id,
            query_embedding,
            limit=limit,
            min_similarity=0.0,
        )

        if not entity_ids_scores:
            return []

        # Batch fetch all entities in a single query (avoids N+1)
        entity_ids = [entity_id for entity_id, _ in entity_ids_scores]
        entities_map = await storage.get_entities_batch(entity_ids, namespace_id=namespace_id)

        # Return entities in score order, filtering out any that weren't found
        return [entities_map[eid] for eid, _score in entity_ids_scores if eid in entities_map]

    # =========================================================================
    # Document Operations
    # =========================================================================

    async def get_document(self, document_id: UUID, *, namespace_id: UUID) -> Document | None:
        """Get a document by ID, scoped to ``namespace_id`` (IDOR family)."""
        return await self._get_storage().get_document(document_id, namespace_id=namespace_id)

    async def list_documents(
        self,
        namespace_id: UUID,
        *,
        limit: int = 100,
    ) -> list[Document]:
        """List documents in a namespace."""
        return await self._get_storage().list_documents(namespace_id, limit=limit)

    async def stats(self, namespace_id: UUID) -> Stats:
        """Get document/chunk/entity/relationship counts for a namespace."""
        storage = self._get_storage()

        doc_count = 0
        last_activity_at = None

        try:
            doc_count, last_activity_at = await storage.get_document_stats(namespace_id)
        except (AttributeError, NotImplementedError):
            pass

        chunk_count, entity_count, relationship_count, metadata = await gather_counts(
            storage, namespace_id, engine="vectorcypher"
        )

        return Stats(
            documents=doc_count,
            chunks=chunk_count,
            entities=entity_count,
            relationships=relationship_count,
            last_activity_at=last_activity_at,
            metadata=metadata,
        )

    async def health_check(self) -> dict[str, Any]:
        """Check health of all components."""
        if not self._connected:
            return {"status": "disconnected"}

        storage_health = await self._get_storage().health_check()
        temporal_health = await self._get_temporal_store().health_check()

        # Check Neo4j
        neo4j_healthy = False
        if self._neo4j_driver:
            try:
                await self._neo4j_driver.verify_connectivity()
                neo4j_healthy = True
            except Exception as e:
                logger.debug(f"Neo4j health check failed: {e}")

        all_healthy = storage_health.is_healthy and temporal_health.get("status") == "healthy" and neo4j_healthy

        return {
            "status": "healthy" if all_healthy else "degraded",
            "storage": storage_health.summary,
            "temporal_store": temporal_health,
            "neo4j": "healthy" if neo4j_healthy else "unhealthy",
            "engine": "vectorcypher",
        }


__all__ = ["ExtractionQualityMetrics", "VectorCypherConfig", "VectorCypherEngine"]
