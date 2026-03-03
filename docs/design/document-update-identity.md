# ADR: Document Update Identity

## Status

Accepted

## Context

Khora's ingestion pipeline treats every `remember()` call as a new document. The only dedup is checksum-based: identical content is skipped. When a document's content changes (same source URL, different text), the old chunks, entities, and relationships remain alongside the new ones, causing stale/contradictory query results.

We need a mechanism to detect when a re-ingested document is an update of an existing one, clean up the old data, and replace it.

Two competing approaches were considered:

- **(A) Consumer-supplied stable ID:** Consumers pass an explicit `external_id`; Khora is fully agnostic about identity.
- **(B) Auto-detect from existing metadata:** Khora uses an existing field (`source`) to detect updates without requiring new consumer-side changes.

## Decision

Use the existing `source` field (URL, file path, etc.) as an **optional update key** for v1.

- When `source` is **non-empty** and a document with the same `(namespace_id, source)` exists but has a **different checksum** → treat as an update.
- When `source` is **empty** → always create a new document (current behavior).
- An `allow_update: bool = True` parameter allows consumers to opt out.
- Checksum dedup runs **first** — if content is identical, skip regardless of source.

## Implications

- Documents ingested **without** `source` can never be auto-updated (no identity key).
- Same content re-ingested under a different source URL is **skipped** by checksum dedup (not treated as a new doc for the new source).
- If a doc is first ingested without source, then later re-ingested with a source and different content → creates a second document (no retroactive identity assignment).
- Concurrent updates to the same source race; **last writer wins** (no locking in v1).
- On update: old chunks are deleted, entity/relationship `source_document_ids` are pruned, orphaned entities/relationships (sole source was this doc) are deleted.
- Cross-backend cleanup (Neo4j + pgvector) is best-effort; if one backend fails mid-cleanup, stale references may persist until the next update of the same source.
- The document UUID is **reused** on update so external consumers referencing doc IDs see continuity.

## Future Considerations

- A dedicated `external_id` column could be added if `source` proves insufficient as an identity key.
- Backfill support (`update_document_source()`) could be added to retroactively assign sources.
- `SELECT ... FOR UPDATE` could be added if concurrent update races become problematic.
- Error handling and retry logic for partial cleanup failures across backends.
