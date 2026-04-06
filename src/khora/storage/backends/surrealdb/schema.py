"""SurrealDB schema definitions for Khora.

Defines all tables, indexes, and relations needed by the unified SurrealDB backend.
Uses DEFINE ... IF NOT EXISTS for idempotent schema initialization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from .connection import SurrealDBConnection

# ---------------------------------------------------------------------------
# Full-text analyzer
# ---------------------------------------------------------------------------

_ANALYZER_DEFINITIONS = """
DEFINE ANALYZER IF NOT EXISTS khora_fulltext TOKENIZERS blank, class FILTERS lowercase, snowball(english);
"""

# ---------------------------------------------------------------------------
# Table definitions
# ---------------------------------------------------------------------------

_TABLE_DEFINITIONS = """
-- Namespace / tenancy
DEFINE TABLE IF NOT EXISTS memory_namespace SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS namespace_id ON memory_namespace TYPE string;
DEFINE FIELD IF NOT EXISTS name ON memory_namespace TYPE option<string>;
DEFINE FIELD IF NOT EXISTS stable_id ON memory_namespace TYPE option<string>;
DEFINE FIELD IF NOT EXISTS tenancy_mode ON memory_namespace TYPE string DEFAULT 'shared';
DEFINE FIELD IF NOT EXISTS version ON memory_namespace TYPE int DEFAULT 1;
DEFINE FIELD IF NOT EXISTS is_active ON memory_namespace TYPE bool DEFAULT true;
DEFINE FIELD IF NOT EXISTS config ON memory_namespace FLEXIBLE TYPE option<object>;
DEFINE FIELD IF NOT EXISTS config_overrides ON memory_namespace FLEXIBLE TYPE option<object>;
DEFINE FIELD IF NOT EXISTS sync_checkpoints ON memory_namespace FLEXIBLE TYPE option<object>;
DEFINE FIELD IF NOT EXISTS metadata_ ON memory_namespace FLEXIBLE TYPE option<object>;
DEFINE FIELD IF NOT EXISTS created_at ON memory_namespace TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS updated_at ON memory_namespace TYPE datetime DEFAULT time::now();
DEFINE INDEX IF NOT EXISTS idx_memory_namespace_stable_id ON memory_namespace FIELDS stable_id;
DEFINE INDEX IF NOT EXISTS idx_memory_namespace_ns_active ON memory_namespace FIELDS namespace_id, is_active;

-- Document (aligned with SQLAlchemy DocumentModel)
DEFINE TABLE IF NOT EXISTS document SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS namespace_id ON document TYPE string;
DEFINE FIELD IF NOT EXISTS title ON document TYPE option<string>;
DEFINE FIELD IF NOT EXISTS content ON document TYPE option<string>;
DEFINE FIELD IF NOT EXISTS content_type ON document TYPE string DEFAULT 'text';
DEFINE FIELD IF NOT EXISTS source ON document TYPE option<string>;
DEFINE FIELD IF NOT EXISTS source_type ON document TYPE option<string>;
DEFINE FIELD IF NOT EXISTS status ON document TYPE string DEFAULT 'pending';
DEFINE FIELD IF NOT EXISTS author ON document TYPE option<string>;
DEFINE FIELD IF NOT EXISTS language ON document TYPE string DEFAULT 'en';
DEFINE FIELD IF NOT EXISTS checksum ON document TYPE option<string>;
DEFINE FIELD IF NOT EXISTS size_bytes ON document TYPE option<int>;
DEFINE FIELD IF NOT EXISTS chunk_count ON document TYPE int DEFAULT 0;
DEFINE FIELD IF NOT EXISTS entity_count ON document TYPE int DEFAULT 0;
DEFINE FIELD IF NOT EXISTS error_message ON document TYPE option<string>;
DEFINE FIELD IF NOT EXISTS extraction_config_hash ON document TYPE option<string>;
DEFINE FIELD IF NOT EXISTS metadata_ ON document FLEXIBLE TYPE option<object>;
DEFINE FIELD IF NOT EXISTS created_at ON document TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS updated_at ON document TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS processed_at ON document TYPE option<datetime>;
DEFINE FIELD IF NOT EXISTS source_timestamp ON document TYPE option<datetime>;
DEFINE INDEX IF NOT EXISTS idx_document_namespace ON document FIELDS namespace_id;
DEFINE INDEX IF NOT EXISTS idx_document_ns_checksum ON document FIELDS namespace_id, checksum;
DEFINE INDEX IF NOT EXISTS idx_document_ns_status ON document FIELDS namespace_id, status;

-- Chunk (with HNSW vector index and BM25 full-text index)
DEFINE TABLE IF NOT EXISTS chunk SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS namespace ON chunk TYPE option<record<memory_namespace>>;
DEFINE FIELD IF NOT EXISTS namespace_id ON chunk TYPE option<string>;
DEFINE FIELD IF NOT EXISTS document ON chunk TYPE option<record<document>>;
DEFINE FIELD IF NOT EXISTS document_id ON chunk TYPE option<string>;
DEFINE FIELD IF NOT EXISTS content ON chunk TYPE string;
DEFINE FIELD IF NOT EXISTS chunk_index ON chunk TYPE int DEFAULT 0;
DEFINE FIELD IF NOT EXISTS start_char ON chunk TYPE int DEFAULT 0;
DEFINE FIELD IF NOT EXISTS end_char ON chunk TYPE int DEFAULT 0;
DEFINE FIELD IF NOT EXISTS token_count ON chunk TYPE option<int>;
DEFINE FIELD IF NOT EXISTS embedding ON chunk TYPE option<array<float>>;
DEFINE FIELD IF NOT EXISTS embedding_model ON chunk TYPE option<string>;
DEFINE FIELD IF NOT EXISTS metadata_ ON chunk FLEXIBLE TYPE option<object>;
DEFINE FIELD IF NOT EXISTS created_at ON chunk TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS source_timestamp ON chunk TYPE option<datetime>;
DEFINE INDEX IF NOT EXISTS idx_chunk_namespace ON chunk FIELDS namespace;
DEFINE INDEX IF NOT EXISTS idx_chunk_document ON chunk FIELDS document;
DEFINE INDEX IF NOT EXISTS idx_chunk_doc_idx ON chunk FIELDS document, chunk_index;
-- HNSW + BM25 indexes deferred to ensure_search_indexes() for bulk load performance

-- Entity (with HNSW vector index and unique constraint)
DEFINE TABLE IF NOT EXISTS entity SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS namespace ON entity TYPE record<memory_namespace>;
DEFINE FIELD IF NOT EXISTS namespace_id ON entity TYPE option<string>;
DEFINE FIELD IF NOT EXISTS name ON entity TYPE string;
DEFINE FIELD IF NOT EXISTS entity_type ON entity TYPE string;
DEFINE FIELD IF NOT EXISTS description ON entity TYPE option<string>;
DEFINE FIELD IF NOT EXISTS attributes ON entity FLEXIBLE TYPE option<object>;
DEFINE FIELD IF NOT EXISTS source_document_ids ON entity TYPE option<array<string>>;
DEFINE FIELD IF NOT EXISTS source_chunk_ids ON entity TYPE option<array<string>>;
DEFINE FIELD IF NOT EXISTS source_tool ON entity TYPE option<string>;
DEFINE FIELD IF NOT EXISTS embedding ON entity TYPE option<array<float>>;
DEFINE FIELD IF NOT EXISTS embedding_model ON entity TYPE option<string>;
DEFINE FIELD IF NOT EXISTS mention_count ON entity TYPE int DEFAULT 1;
DEFINE FIELD IF NOT EXISTS valid_from ON entity TYPE option<datetime>;
DEFINE FIELD IF NOT EXISTS valid_until ON entity TYPE option<datetime>;
DEFINE FIELD IF NOT EXISTS confidence ON entity TYPE float DEFAULT 1.0;
DEFINE FIELD IF NOT EXISTS metadata_ ON entity FLEXIBLE TYPE option<object>;
DEFINE FIELD IF NOT EXISTS created_at ON entity TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS updated_at ON entity TYPE datetime DEFAULT time::now();
DEFINE INDEX IF NOT EXISTS idx_entity_namespace ON entity FIELDS namespace;
DEFINE INDEX IF NOT EXISTS idx_entity_unique ON entity FIELDS namespace, name, entity_type UNIQUE;
DEFINE INDEX IF NOT EXISTS idx_entity_ns_type ON entity FIELDS namespace, entity_type;
DEFINE INDEX IF NOT EXISTS idx_entity_ns_mention ON entity FIELDS namespace, mention_count;
DEFINE INDEX IF NOT EXISTS idx_entity_ns_created ON entity FIELDS namespace, created_at;
-- HNSW index deferred to ensure_search_indexes()

-- Relates-to (graph edge between entities)
DEFINE TABLE IF NOT EXISTS relates_to TYPE RELATION SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS namespace_id ON relates_to TYPE string;
DEFINE FIELD IF NOT EXISTS relationship_type ON relates_to TYPE string;
DEFINE FIELD IF NOT EXISTS weight ON relates_to TYPE float DEFAULT 1.0;
DEFINE FIELD IF NOT EXISTS description ON relates_to TYPE option<string>;
DEFINE FIELD IF NOT EXISTS properties ON relates_to FLEXIBLE TYPE option<object>;
DEFINE FIELD IF NOT EXISTS source_document_ids ON relates_to TYPE option<array<string>>;
DEFINE FIELD IF NOT EXISTS source_chunk_ids ON relates_to TYPE option<array<string>>;
DEFINE FIELD IF NOT EXISTS valid_from ON relates_to TYPE option<datetime>;
DEFINE FIELD IF NOT EXISTS valid_until ON relates_to TYPE option<datetime>;
DEFINE FIELD IF NOT EXISTS confidence ON relates_to TYPE float DEFAULT 1.0;
DEFINE FIELD IF NOT EXISTS metadata_ ON relates_to FLEXIBLE TYPE option<object>;
DEFINE FIELD IF NOT EXISTS created_at ON relates_to TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS updated_at ON relates_to TYPE datetime DEFAULT time::now();
DEFINE INDEX IF NOT EXISTS idx_relates_to_namespace ON relates_to FIELDS namespace_id;
DEFINE INDEX IF NOT EXISTS idx_relates_to_ns_type ON relates_to FIELDS namespace_id, relationship_type;
DEFINE INDEX IF NOT EXISTS idx_relates_to_ns_weight ON relates_to FIELDS namespace_id, relationship_type, weight;

-- Episode (aligned with SQLAlchemy Episode model)
DEFINE TABLE IF NOT EXISTS episode SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS namespace ON episode TYPE option<record<memory_namespace>>;
DEFINE FIELD IF NOT EXISTS namespace_id ON episode TYPE option<string>;
DEFINE FIELD IF NOT EXISTS name ON episode TYPE option<string>;
DEFINE FIELD IF NOT EXISTS description ON episode TYPE option<string>;
DEFINE FIELD IF NOT EXISTS episode_type ON episode TYPE string DEFAULT 'generic';
DEFINE FIELD IF NOT EXISTS occurred_at ON episode TYPE option<datetime>;
DEFINE FIELD IF NOT EXISTS duration_seconds ON episode TYPE option<int>;
DEFINE FIELD IF NOT EXISTS entity_ids ON episode TYPE option<array<string>>;
DEFINE FIELD IF NOT EXISTS source_document_ids ON episode TYPE option<array<string>>;
DEFINE FIELD IF NOT EXISTS source_chunk_ids ON episode TYPE option<array<string>>;
DEFINE FIELD IF NOT EXISTS embedding ON episode TYPE option<array<float>>;
DEFINE FIELD IF NOT EXISTS embedding_model ON episode TYPE option<string>;
DEFINE FIELD IF NOT EXISTS metadata_ ON episode FLEXIBLE TYPE option<object>;
DEFINE FIELD IF NOT EXISTS created_at ON episode TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS updated_at ON episode TYPE datetime DEFAULT time::now();
DEFINE INDEX IF NOT EXISTS idx_episode_namespace ON episode FIELDS namespace;
DEFINE INDEX IF NOT EXISTS idx_episode_occurred ON episode FIELDS namespace, occurred_at;
-- HNSW index deferred to ensure_search_indexes()

-- Involves (episode → entity edge)
DEFINE TABLE IF NOT EXISTS involves TYPE RELATION SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS namespace_id ON involves TYPE option<string>;
DEFINE FIELD IF NOT EXISTS created_at ON involves TYPE datetime DEFAULT time::now();

-- Memory event
DEFINE TABLE IF NOT EXISTS memory_event SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS namespace_id ON memory_event TYPE string;
DEFINE FIELD IF NOT EXISTS event_type ON memory_event TYPE string;
DEFINE FIELD IF NOT EXISTS timestamp ON memory_event TYPE option<datetime>;
DEFINE FIELD IF NOT EXISTS resource_type ON memory_event TYPE option<string>;
DEFINE FIELD IF NOT EXISTS resource_id ON memory_event TYPE option<string>;
DEFINE FIELD IF NOT EXISTS data ON memory_event FLEXIBLE TYPE option<object>;
DEFINE FIELD IF NOT EXISTS previous_data ON memory_event FLEXIBLE TYPE option<object>;
DEFINE FIELD IF NOT EXISTS actor_id ON memory_event TYPE option<string>;
DEFINE FIELD IF NOT EXISTS actor_type ON memory_event TYPE string DEFAULT 'system';
DEFINE FIELD IF NOT EXISTS correlation_id ON memory_event TYPE option<string>;
DEFINE FIELD IF NOT EXISTS version ON memory_event TYPE int DEFAULT 1;
DEFINE FIELD IF NOT EXISTS metadata_ ON memory_event FLEXIBLE TYPE option<object>;
DEFINE FIELD IF NOT EXISTS created_at ON memory_event TYPE datetime DEFAULT time::now();
DEFINE INDEX IF NOT EXISTS idx_memory_event_namespace ON memory_event FIELDS namespace_id;
DEFINE INDEX IF NOT EXISTS idx_memory_event_type ON memory_event FIELDS event_type;
DEFINE INDEX IF NOT EXISTS idx_memory_event_resource ON memory_event FIELDS resource_type, resource_id;
DEFINE INDEX IF NOT EXISTS idx_memory_event_timestamp ON memory_event FIELDS namespace_id, timestamp;

-- Permission
DEFINE TABLE IF NOT EXISTS permission SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS namespace_id ON permission TYPE string;
DEFINE FIELD IF NOT EXISTS principal ON permission TYPE string;
DEFINE FIELD IF NOT EXISTS principal_type ON permission TYPE string DEFAULT 'user';
DEFINE FIELD IF NOT EXISTS permission_level ON permission TYPE string DEFAULT 'read';
DEFINE FIELD IF NOT EXISTS created_at ON permission TYPE datetime DEFAULT time::now();
DEFINE INDEX IF NOT EXISTS idx_permission_namespace ON permission FIELDS namespace_id;
DEFINE INDEX IF NOT EXISTS idx_permission_principal ON permission FIELDS principal;

-- Sync checkpoint
DEFINE TABLE IF NOT EXISTS sync_checkpoint SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS namespace_id ON sync_checkpoint TYPE string;
DEFINE FIELD IF NOT EXISTS source ON sync_checkpoint TYPE string;
DEFINE FIELD IF NOT EXISTS checkpoint ON sync_checkpoint TYPE string;
DEFINE FIELD IF NOT EXISTS metadata_ ON sync_checkpoint FLEXIBLE TYPE option<object>;
DEFINE FIELD IF NOT EXISTS updated_at ON sync_checkpoint TYPE datetime DEFAULT time::now();
DEFINE INDEX IF NOT EXISTS idx_sync_checkpoint_ns_source ON sync_checkpoint FIELDS namespace_id, source UNIQUE;

-- Expertise definition
DEFINE TABLE IF NOT EXISTS expertise_definition SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS namespace_id ON expertise_definition TYPE string;
DEFINE FIELD IF NOT EXISTS name ON expertise_definition TYPE string;
DEFINE FIELD IF NOT EXISTS description ON expertise_definition TYPE option<string>;
DEFINE FIELD IF NOT EXISTS config ON expertise_definition FLEXIBLE TYPE option<object>;
DEFINE FIELD IF NOT EXISTS created_at ON expertise_definition TYPE datetime DEFAULT time::now();
DEFINE INDEX IF NOT EXISTS idx_expertise_namespace ON expertise_definition FIELDS namespace_id;

-- Time node (temporal graph)
DEFINE TABLE IF NOT EXISTS time_node SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS namespace_id ON time_node TYPE string;
DEFINE FIELD IF NOT EXISTS time_value ON time_node TYPE datetime;
DEFINE FIELD IF NOT EXISTS granularity ON time_node TYPE string DEFAULT 'day';
DEFINE FIELD IF NOT EXISTS label ON time_node TYPE option<string>;
DEFINE FIELD IF NOT EXISTS metadata_ ON time_node FLEXIBLE TYPE option<object>;
DEFINE FIELD IF NOT EXISTS created_at ON time_node TYPE datetime DEFAULT time::now();
DEFINE INDEX IF NOT EXISTS idx_time_node_namespace ON time_node FIELDS namespace_id;
DEFINE INDEX IF NOT EXISTS idx_time_node_value ON time_node FIELDS namespace_id, time_value;

-- Temporal edge (between time nodes)
DEFINE TABLE IF NOT EXISTS temporal_edge TYPE RELATION SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS namespace_id ON temporal_edge TYPE string;
DEFINE FIELD IF NOT EXISTS edge_type ON temporal_edge TYPE string;
DEFINE FIELD IF NOT EXISTS weight ON temporal_edge TYPE float DEFAULT 1.0;
DEFINE FIELD IF NOT EXISTS metadata_ ON temporal_edge FLEXIBLE TYPE option<object>;
DEFINE FIELD IF NOT EXISTS created_at ON temporal_edge TYPE datetime DEFAULT time::now();
DEFINE INDEX IF NOT EXISTS idx_temporal_edge_namespace ON temporal_edge FIELDS namespace_id;

-- Time-edge link (links entities/chunks to time nodes)
DEFINE TABLE IF NOT EXISTS time_edge_link TYPE RELATION SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS namespace_id ON time_edge_link TYPE string;
DEFINE FIELD IF NOT EXISTS link_type ON time_edge_link TYPE string DEFAULT 'occurred_at';
DEFINE FIELD IF NOT EXISTS metadata_ ON time_edge_link FLEXIBLE TYPE option<object>;
DEFINE FIELD IF NOT EXISTS created_at ON time_edge_link TYPE datetime DEFAULT time::now();
DEFINE INDEX IF NOT EXISTS idx_time_edge_link_namespace ON time_edge_link FIELDS namespace_id;

-- Next-session link (connects last chunk of session A to first chunk of session B)
DEFINE TABLE IF NOT EXISTS next_session TYPE RELATION SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS namespace_id ON next_session TYPE string;
DEFINE FIELD IF NOT EXISTS metadata_ ON next_session FLEXIBLE TYPE option<object>;
DEFINE FIELD IF NOT EXISTS created_at ON next_session TYPE datetime DEFAULT time::now();
DEFINE INDEX IF NOT EXISTS idx_next_session_namespace ON next_session FIELDS namespace_id;
"""


_SEARCH_INDEX_DEFINITIONS = """
-- HNSW vector indexes (deferred from table definitions for bulk-load performance).
-- These are expensive to maintain incrementally on every INSERT.
DEFINE INDEX IF NOT EXISTS idx_chunk_embedding ON chunk FIELDS embedding HNSW DIMENSION 1536 DIST COSINE TYPE F32 EFC 128 M 24;
DEFINE INDEX IF NOT EXISTS idx_chunk_content_ft ON chunk FIELDS content SEARCH ANALYZER khora_fulltext BM25;
DEFINE INDEX IF NOT EXISTS idx_entity_embedding ON entity FIELDS embedding HNSW DIMENSION 1536 DIST COSINE TYPE F32 EFC 128 M 24;
DEFINE INDEX IF NOT EXISTS idx_episode_embedding ON episode FIELDS embedding HNSW DIMENSION 1536 DIST COSINE TYPE F32 EFC 128 M 24;
"""


async def ensure_search_indexes(conn: SurrealDBConnection) -> None:
    """Create HNSW and BM25 search indexes on chunk/entity/episode tables.

    Call after bulk ingestion to avoid per-INSERT index maintenance
    overhead during data loading.  Idempotent (uses IF NOT EXISTS).

    Args:
        conn: An active SurrealDBConnection instance.
    """
    logger.info("Creating SurrealDB search indexes (HNSW + BM25)...")
    await conn.execute(_ANALYZER_DEFINITIONS)
    await conn.execute(_SEARCH_INDEX_DEFINITIONS)
    logger.info("SurrealDB search indexes created")


async def initialize_schema(conn: SurrealDBConnection) -> None:
    """Initialize the SurrealDB schema with all table and index definitions.

    This function is idempotent — all DEFINE statements use IF NOT EXISTS.

    Args:
        conn: An active SurrealDBConnection instance.
    """
    logger.info("Initializing SurrealDB schema...")

    # Define the full-text analyzer first (required by BM25 indexes)
    await conn.execute(_ANALYZER_DEFINITIONS)

    # Define all tables, fields, and indexes
    await conn.execute(_TABLE_DEFINITIONS)

    logger.info("SurrealDB schema initialized successfully")
