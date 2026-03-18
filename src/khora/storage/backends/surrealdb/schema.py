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
DEFINE FIELD IF NOT EXISTS name ON memory_namespace TYPE string;
DEFINE FIELD IF NOT EXISTS stable_id ON memory_namespace TYPE string;
DEFINE FIELD IF NOT EXISTS version ON memory_namespace TYPE int DEFAULT 1;
DEFINE FIELD IF NOT EXISTS is_active ON memory_namespace TYPE bool DEFAULT true;
DEFINE FIELD IF NOT EXISTS config ON memory_namespace FLEXIBLE TYPE option<object>;
DEFINE FIELD IF NOT EXISTS created_at ON memory_namespace TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS updated_at ON memory_namespace TYPE datetime DEFAULT time::now();
DEFINE INDEX IF NOT EXISTS idx_memory_namespace_stable_id ON memory_namespace FIELDS stable_id;

-- Document
DEFINE TABLE IF NOT EXISTS document SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS namespace_id ON document TYPE string;
DEFINE FIELD IF NOT EXISTS title ON document TYPE option<string>;
DEFINE FIELD IF NOT EXISTS content ON document TYPE option<string>;
DEFINE FIELD IF NOT EXISTS content_type ON document TYPE string DEFAULT 'text';
DEFINE FIELD IF NOT EXISTS source ON document TYPE option<string>;
DEFINE FIELD IF NOT EXISTS metadata ON document FLEXIBLE TYPE option<object>;
DEFINE FIELD IF NOT EXISTS created_at ON document TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS updated_at ON document TYPE datetime DEFAULT time::now();
DEFINE INDEX IF NOT EXISTS idx_document_namespace ON document FIELDS namespace_id;

-- Chunk (with HNSW vector index and BM25 full-text index)
DEFINE TABLE IF NOT EXISTS chunk SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS namespace_id ON chunk TYPE string;
DEFINE FIELD IF NOT EXISTS document_id ON chunk TYPE string;
DEFINE FIELD IF NOT EXISTS content ON chunk TYPE string;
DEFINE FIELD IF NOT EXISTS chunk_index ON chunk TYPE int DEFAULT 0;
DEFINE FIELD IF NOT EXISTS token_count ON chunk TYPE option<int>;
DEFINE FIELD IF NOT EXISTS embedding ON chunk TYPE option<array<float>>;
DEFINE FIELD IF NOT EXISTS metadata ON chunk FLEXIBLE TYPE option<object>;
DEFINE FIELD IF NOT EXISTS created_at ON chunk TYPE datetime DEFAULT time::now();
DEFINE INDEX IF NOT EXISTS idx_chunk_namespace ON chunk FIELDS namespace_id;
DEFINE INDEX IF NOT EXISTS idx_chunk_document ON chunk FIELDS document_id;
DEFINE INDEX IF NOT EXISTS idx_chunk_embedding ON chunk FIELDS embedding HNSW DIMENSION 1536 DIST COSINE TYPE F32 EFC 128 M 24;
DEFINE INDEX IF NOT EXISTS idx_chunk_content_ft ON chunk FIELDS content SEARCH ANALYZER khora_fulltext BM25;

-- Entity (with HNSW vector index and unique constraint)
DEFINE TABLE IF NOT EXISTS entity SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS namespace_id ON entity TYPE string;
DEFINE FIELD IF NOT EXISTS name ON entity TYPE string;
DEFINE FIELD IF NOT EXISTS entity_type ON entity TYPE string;
DEFINE FIELD IF NOT EXISTS description ON entity TYPE option<string>;
DEFINE FIELD IF NOT EXISTS embedding ON entity TYPE option<array<float>>;
DEFINE FIELD IF NOT EXISTS mention_count ON entity TYPE int DEFAULT 1;
DEFINE FIELD IF NOT EXISTS metadata ON entity FLEXIBLE TYPE option<object>;
DEFINE FIELD IF NOT EXISTS created_at ON entity TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS updated_at ON entity TYPE datetime DEFAULT time::now();
DEFINE INDEX IF NOT EXISTS idx_entity_unique ON entity FIELDS namespace_id, name, entity_type UNIQUE;
DEFINE INDEX IF NOT EXISTS idx_entity_namespace ON entity FIELDS namespace_id;
DEFINE INDEX IF NOT EXISTS idx_entity_embedding ON entity FIELDS embedding HNSW DIMENSION 1536 DIST COSINE TYPE F32 EFC 128 M 24;

-- Relates-to (graph edge between entities)
DEFINE TABLE IF NOT EXISTS relates_to TYPE RELATION SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS namespace_id ON relates_to TYPE string;
DEFINE FIELD IF NOT EXISTS relationship_type ON relates_to TYPE string;
DEFINE FIELD IF NOT EXISTS weight ON relates_to TYPE float DEFAULT 1.0;
DEFINE FIELD IF NOT EXISTS description ON relates_to TYPE option<string>;
DEFINE FIELD IF NOT EXISTS metadata ON relates_to FLEXIBLE TYPE option<object>;
DEFINE FIELD IF NOT EXISTS created_at ON relates_to TYPE datetime DEFAULT time::now();
DEFINE INDEX IF NOT EXISTS idx_relates_to_namespace ON relates_to FIELDS namespace_id;

-- Episode
DEFINE TABLE IF NOT EXISTS episode SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS namespace_id ON episode TYPE string;
DEFINE FIELD IF NOT EXISTS name ON episode TYPE option<string>;
DEFINE FIELD IF NOT EXISTS episode_type ON episode TYPE string DEFAULT 'generic';
DEFINE FIELD IF NOT EXISTS summary ON episode TYPE option<string>;
DEFINE FIELD IF NOT EXISTS metadata ON episode FLEXIBLE TYPE option<object>;
DEFINE FIELD IF NOT EXISTS created_at ON episode TYPE datetime DEFAULT time::now();
DEFINE INDEX IF NOT EXISTS idx_episode_namespace ON episode FIELDS namespace_id;

-- Memory event
DEFINE TABLE IF NOT EXISTS memory_event SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS namespace_id ON memory_event TYPE string;
DEFINE FIELD IF NOT EXISTS event_type ON memory_event TYPE string;
DEFINE FIELD IF NOT EXISTS payload ON memory_event FLEXIBLE TYPE option<object>;
DEFINE FIELD IF NOT EXISTS created_at ON memory_event TYPE datetime DEFAULT time::now();
DEFINE INDEX IF NOT EXISTS idx_memory_event_namespace ON memory_event FIELDS namespace_id;
DEFINE INDEX IF NOT EXISTS idx_memory_event_type ON memory_event FIELDS event_type;

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
DEFINE FIELD IF NOT EXISTS metadata ON sync_checkpoint FLEXIBLE TYPE option<object>;
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
DEFINE FIELD IF NOT EXISTS metadata ON time_node FLEXIBLE TYPE option<object>;
DEFINE FIELD IF NOT EXISTS created_at ON time_node TYPE datetime DEFAULT time::now();
DEFINE INDEX IF NOT EXISTS idx_time_node_namespace ON time_node FIELDS namespace_id;
DEFINE INDEX IF NOT EXISTS idx_time_node_value ON time_node FIELDS namespace_id, time_value;

-- Temporal edge (between time nodes)
DEFINE TABLE IF NOT EXISTS temporal_edge TYPE RELATION SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS namespace_id ON temporal_edge TYPE string;
DEFINE FIELD IF NOT EXISTS edge_type ON temporal_edge TYPE string;
DEFINE FIELD IF NOT EXISTS weight ON temporal_edge TYPE float DEFAULT 1.0;
DEFINE FIELD IF NOT EXISTS metadata ON temporal_edge FLEXIBLE TYPE option<object>;
DEFINE FIELD IF NOT EXISTS created_at ON temporal_edge TYPE datetime DEFAULT time::now();
DEFINE INDEX IF NOT EXISTS idx_temporal_edge_namespace ON temporal_edge FIELDS namespace_id;

-- Time-edge link (links entities/chunks to time nodes)
DEFINE TABLE IF NOT EXISTS time_edge_link TYPE RELATION SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS namespace_id ON time_edge_link TYPE string;
DEFINE FIELD IF NOT EXISTS link_type ON time_edge_link TYPE string DEFAULT 'occurred_at';
DEFINE FIELD IF NOT EXISTS metadata ON time_edge_link FLEXIBLE TYPE option<object>;
DEFINE FIELD IF NOT EXISTS created_at ON time_edge_link TYPE datetime DEFAULT time::now();
DEFINE INDEX IF NOT EXISTS idx_time_edge_link_namespace ON time_edge_link FIELDS namespace_id;
"""


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
