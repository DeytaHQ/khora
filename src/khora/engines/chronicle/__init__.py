"""Chronicle — temporal-semantic memory engine.

Optimized for LongMemEval, LoCoMo, and BEAM benchmarks.
No graph database required — uses PostgreSQL + pgvector only.

Key techniques:
- Event decomposition: SVO tuples with datetime ranges
- Atomic fact extraction (Elementary Discourse Units)
- Triple timestamps: observation, referenced, relative
- 4-channel parallel retrieval: semantic + BM25 + temporal + entity
- Progressive memory compression
- Temporal decay (Ebbinghaus forgetting curve)

Usage:
    async with MemoryLake(db_url, engine="chronicle") as lake:
        ns = await lake.create_namespace()
        await lake.remember(
            "Alice met Bob at the conference on March 15th.",
            namespace=ns.namespace_id,
            entity_types=["PERSON", "EVENT"],
            relationship_types=["ATTENDED"],
        )
        result = await lake.recall("Who did Alice meet?", namespace=ns.namespace_id)
"""

from __future__ import annotations

from .engine import ChronicleEngine

__all__ = ["ChronicleEngine"]
