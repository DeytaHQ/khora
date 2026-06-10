# Deterministic end-to-end recall-filter harness — design notes

> **SETTLED DECISIONS (read first — supersede any older snapshot you may be holding;
> all four reconciliation/seeding points empirically validated against a real
> sqlite_lance Khora):**
> 1. **Seed via real `Khora.remember()` everywhere** — NOT `conformance.seed_case`
>    (it bypasses the ingest pipeline, writes zero entities → defeats AC2, one chunk
>    per record → defeats AC6, and its rows are NOT recall-visible: `seed_case` →
>    `recall()` returns 0 chunks, proven).
> 2. **Reconcile DOC-LEVEL by `external_id`** (= `SeedRecord.id`, stamped at
>    `remember()` time) — NOT chunk-UUID via a `seed_case` `id_map` (there is no
>    `id_map` when seeding via `remember()`). `chunk-UUID` reconciliation is
>    **rejected** (see §3). Key off the RETURNED chunks' `document_id`s.
> 3. **Use DISTINCT content per record** (`f"{rec.content} record {rec.id}"`) — N
>    identical-content docs collapse to ONE recallable doc through vector top-k
>    (proven), which would make the row-set proof impossible. Content is not a filter
>    target for the curated families, so `expected_ids` is unchanged.
> 4. **Stub seam = three class-method patches** via `filter_spy.stub_llm`
>    (`LiteLLMEmbedder.embed_batch/.embed` + `LLMEntityExtractor.extract_multi`). The
>    SHA-256 hash embedder is fine here because distinct content gives each doc a
>    distinct vector; the graph lane floors the entry gate with
>    `min_entity_similarity=0.0` and guards with the `graph_chunk_count>0` tripwire.
> 5. Curated e2e corpus = remember-threadable F-OP families (7 string keys +
>    `source_timestamp` + metadata); `occurred_at`/`created_at`/`content_type`,
>    `external_id`-filtering, and duplicate-`external_id` families are curated OUT
>    (covered by the conformance executor suite). See §3.

`@internal`. Architecture for the `tests/e2e/` recall-filter end-to-end suite.
This is the **row-set proof** that complements the existing WIRING spies
(`tests/integration/test_filter_pushdown_graph.py`,
`tests/integration/matrix/test_filter_enforcement_sqlite_lance.py`): those prove
the validated filter AST *reaches* each channel unchanged; this suite proves the
filter *actually narrows the rows* end to end, through the real `Khora.remember()`
ingest pipeline and `Khora.recall(filter=...)` read path, with a populated graph.

Smallest thing that satisfies the six acceptance criteria. No new test framework,
no DI abstractions, no `src/` changes. Behavior-only language throughout.

## File layout + ownership (`tests/e2e/`)

```
tests/e2e/
  __init__.py                      # empty
  DESIGN_NOTES.md                  # this file (architect)
  conftest.py                      # BACKEND owns
  _harness.py                      # BACKEND owns
  test_filter_rowset_embedded.py   # QA owns. sqlite_lance, no-Docker main lane -> AC1 + AC4 + AC5 (metadata)
  test_filter_rowset_graph.py      # QA owns. live VectorCypher PG+Neo4j, self-skip -> AC3 + system-key AC4/AC5
  test_filter_rowset_chronicle.py  # QA owns. live Chronicle PG-only, self-skip -> AC3 + system-key AC4/AC5
  test_graph_contribution.py       # QA owns. live PG+Neo4j, self-skip -> AC2 + AC3 honesty (pre-flight + firing tripwire + filter-enforcement drop)
  test_multichunk_denorm.py        # QA owns. sqlite_lance, no-Docker -> AC6
```

The modules are organized by **engine lane** (the shipped structure), not one-per-AC:
the three `test_filter_rowset_*` modules drive the same `remember()`→`recall(filter=)`
row-set proof on each engine; `test_graph_contribution.py` is the non-vacuity honesty
proof; `test_multichunk_denorm.py` is the denormalization-uniformity proof. The
embedded lane carries the corpus families (AC1 deterministic ingest, AC4 external_id
reconciliation, AC5 F-EXISTS reachability) since it runs in the no-Docker main job.

- **`conftest.py` (Backend)** — per-engine `Khora` fixtures (sqlite_lance embedded
  / vectorcypher PG+Neo4j / chronicle), each installing the deterministic
  LLM/extractor stub (`stub_llm`) before building its `Khora`, with HyDE-off
  (+ reranking-off) config. `_pg_reachable()` self-skip guard so a no-Docker run
  collects-and-skips the live legs cleanly.
- **`_harness.py` (Backend)** — the deterministic extract stub, the entity-bearing
  + multi-chunk seed builders, doc-level `reconcile()`, and a `neo4j_populated()`
  probe. The single surface QA imports.
- **`test_*.py` (QA)** — the cases, the per-AC assertions, the parametrize
  hookups. QA imports from `_harness` and `khora.filter.conformance`; QA never
  reaches into `src/` for private symbols beyond what `_harness` exposes.

**Boundary rule:** Backend writes `conftest.py` + `_harness.py` ONLY. QA writes the
`test_*.py` modules ONLY.

## The seam (settled — no `src/` changes)

A monkeypatch of three class methods (network-free, undone at teardown), via
`tests/test_helpers/filter_spy.py::stub_llm(monkeypatch, dim)`. `install_mock_llm`
is NOT used and not needed — completions never fire with HyDE + reranking off and
extraction stubbed above the `litellm.acompletion` call.

- `khora.extraction.embedders.litellm.LiteLLMEmbedder.embed_batch`
- `khora.extraction.embedders.litellm.LiteLLMEmbedder.embed`
  → `fake_embedding(text, dim)` = SHA-256 → L2-normalize (deterministic per text).
  **The hash embedder is sufficient here BECAUSE seeding uses distinct content per
  record (§3):** distinct content → distinct vector → each doc is independently
  recallable, and a generous `limit` (100) makes the filter the only narrowing
  force. On the **graph lane** the entity-bearing docs have distinct content too, so
  the query↔entity cosine is not guaranteed positive — the fixture floors the
  entry-entity gate with `min_entity_similarity = 0.0` and the `graph_chunk_count>0`
  pre-flight tripwire (§"Graph-channel proof") catches any regression. (If that gate
  ever flakes, swap the graph fixture's embedder to a unit vector `[1.0]+[0.0]*(dim-1)`
  for cosine 1.0 — the reference `test_vectorcypher_filter_counters.py` does this.)
  Per-engine dim: 1536 for the PG pgvector column, 32 (`EMBED_DIM`) for sqlite_lance.
- `khora.extraction.extractors.llm.LLMEntityExtractor.extract_multi`
  → a content-keyed registry: a doc whose text **contains** a registered marker
  yields that marker's `ExtractionResult` (real `ExtractedEntity` /
  `ExtractedRelationship` dataclasses), else an empty `ExtractionResult()`.

A thin `plan_extraction(marker, entities, relationships)` helper in `_harness.py`
backs the marker→result registry the `extract_multi` stub reads. This drives
**marker-based entity emission through the REAL ingest pipeline** — the patch sits
at the LLM/embedding leaf, so everything above it runs unchanged: `extract_entities`
constructs the real `LLMEntityExtractor`, entity normalization + resolution +
dual-node mirroring run, and `upsert_entities_batch` + `create_relationships_batch`
write MENTIONED_IN edges into **both Neo4j and pgvector**. This is *not*
`extract_entities=False` and *not* a high-level patch that bypasses persistence. For
the negative tripwire the registry is left empty so every doc gets
`ExtractionResult()` and the graph stays empty.

### Config so the graph fires

- `extract_entities = True`, `selective_extraction = False` (every seed chunk goes
  to the stub, so `expected_ids` stays exact)
- `enable_reranking = False`, `enable_llm_reranking = False` (no cross-encoder load;
  scores untouched by a reranker)
- HyDE off: `enable_hyde = "never"`, `enable_hyde_cypher = False`, set on the
  `KhoraConfig` each per-engine fixture builds; `test_hyde_is_disabled` pins
  `config.query.enable_hyde == "never"` so a future default flip can't silently
  re-enable query rewriting. (There is no env-var override and no autouse fixture —
  as shipped, `stub_llm` is installed *inside* each per-engine fixture before its
  `Khora` is constructed, not via an autouse/env layer.)
- `min_entity_similarity = 0.0` (floors the entry-entity vector gate so the seeded
  entity is returned as a graph-expansion seed — see Implementation risks)
- non-empty `entity_types` / `relationship_types` on `remember()` (vectorcypher
  requires them)

> Verify the exact `KhoraConfig` attribute names against the live config /
> reference fixtures (`config.pipeline` vs `config.pipelines`, `config.query.*`) —
> do not guess.

## Reconciliation (AC4) — seed via real `remember()`, reconcile on `external_id`

**Settled:** AC4 seeds each curated case's `SeedRecord`s through the **real**
`Khora.remember()` ingest pipeline — NOT `conformance.seed_case`. This is the whole
reason the harness exists: it must exercise real chunking, extraction, embedding,
denormalization, and graph writes, then prove the public `recall(filter=...)`
narrows the rows to the same set the corpus declares. `seed_case` writes directly
through the `StorageCoordinator` (the direct-seed / conformance path) and would
bypass the ingest pipeline — re-running what the conformance suite already covers.

> **Considered & rejected: `seed_case` + chunk-UUID reconciliation.** A direct
> `conformance.seed_case` write + reconciling on its `seed_id -> chunk UUID`
> `id_map` looks simpler, but it **bypasses `remember()`** — it runs no extraction
> (writes ZERO entities → the AC2 graph channel cannot fire), writes exactly one
> `chunk_index=0` chunk per record (AC6 multi-chunk impossible), and reconciling the
> direct seed against its own `id_map` is circular (proves nothing about the ingest
> pipeline under test). The harness uses real `remember()` everywhere, so there is
> no `seed_case` `id_map` to key on.

Because `remember()` assigns **fresh chunk/doc UUIDs we don't control**, the only
stable handle to reconcile a real-ingest result against the corpus's handle-based
`expected_ids` is **`external_id`**. So:

- For each `SeedRecord` in a curated case, call
  `kb.remember(content=..., namespace=ns, external_id=<SeedRecord.id>, ...)`,
  mapping the record's filterable fields to `remember()` kwargs:
  `metadata` → `metadata=`, `source_name`/`source_url`/`source_type`/`title` →
  the same-named kwargs, and the event time via `source_timestamp=` (which the
  engine resolves to `occurred_at`).
- **Use DISTINCT content per record** — `f"{record.content} record {record.id}"`,
  NOT the shared `"conformance anchor"`. Empirically verified (Backend): N records
  with IDENTICAL content collapse to ONE recallable document through the engine's
  vector top-k (identical embedding vectors → ANN returns a single distinct row),
  so a shared-anchor seed makes a row-set proof over N records impossible on the
  recall path. Distinct content gives each document a distinct vector so all N are
  independently recallable. Content is NOT a filter target for the curated families,
  so distinct content does not change which rows the filter keeps — `expected_ids`
  is unchanged.
- `recall(query=<fixed>, namespace=ns, limit >= N, filter=case.filter,
  min_similarity=0.0)`. With distinct-content docs all present in the namespace and
  a generous `limit` (≥ the seed count; the harness uses 100), the **filter** — not
  ranking — decides the surviving set.
- `reconcile(result)` → key off the RETURNED CHUNKS, not raw `result.documents`:
  collect `returned_doc_ids = { chunk.document_id for chunk in result.chunks }`,
  then `{ doc.external_id for doc in result.documents if doc.id in returned_doc_ids }`;
  assert `== case.expected_ids`. Doc-level, any-chunk. (Keying on the returned
  chunks' `document_id`s — not blindly on every `result.documents` entry — is what
  enforces "present iff ≥1 surviving chunk"; a document with zero surviving chunks
  must not count even if it appears in the projection list.) `expected_ids` are the
  `SeedRecord` handles, which equal the stamped `external_id`s. The handle
  round-trips on the public surface: `DocumentProjection.external_id` (recall.py:31)
  carries the stamped value out — no chunk/doc-UUID bookkeeping beyond the
  `chunk.document_id → documents[].external_id` join.
- `conformance._case_namespace_id(case)` = `uuid5(...)` gives a fresh, xdist-safe
  namespace per case.

**Curated subset = families `remember()` can faithfully seed.** The AC4 subset uses
families whose filter predicate `remember()` can reproduce AND the lane supports.
`remember()` accepts `metadata`, `title`, `source`, `source_type`, `source_name`,
`source_url`, `source_timestamp`, `external_id`. The clean pick is the
**metadata-key families** — `F-COERCE`, `F-OBJEQ`, `F-DOTKEY`, and the
`metadata.*` `F-EXISTS` cases — which filter on `metadata.<key>` and seed via
`remember(metadata=...)`. Curate OUT:

- The **`external_id` F-OP family** (deliberate-duplicate cases): `external_id` is
  the test variable there, so it cannot also be the reconciliation handle. Covered
  by the direct-seed conformance suite. For every other family
  `SeedRecord.external_id` defaults to `None`, so stamping `external_id =
  SeedRecord.id` (unique within the fresh per-case namespace) is free.
- The **date-column families** (`F-DATES`, the date `F-OP` cases): `remember()` has
  no direct `occurred_at` / `created_at` kwargs — `occurred_at` derives from
  `source_timestamp`, `created_at` is stamped at ingest — so a case that filters on
  a caller-controlled date column is not faithfully reproducible. Leave to the
  conformance suite *unless* the smoke run shows the column is settable via
  `metadata.custom`.

The harness needs only a representative slice per kept family, not the full catalog.
The smoke run drives final curation.

As shipped, `_harness.seed_records(kb, records, namespace_id, ...)` is the seeding
path: it calls `kb.remember()` once per `SeedRecord` via `_remember_kwargs`, mapping
`external_id = record.id` + the threadable system-key fields. The curated case set is
`_harness.rowset_cases(backend, include_system_keys=...)` (the single selector the
three lane modules share).

## Graph-channel proof (non-vacuity)

The honest, lane-isolated signal is `engine_info["graph_chunk_count"]` (surfaced on
the public `RecallResult.engine_info`). Use `SearchMode.GRAPH` (force the graph
path — `HYBRID` could classify a short entity query as SIMPLE / `use_graph=False`
and fall to the graph-less path, making the proof vacuous) and
`min_entity_similarity = 0.0`.

- **Pre-flight (positive):** a no-filter `GRAPH` recall asserts
  `graph_chunk_count > 0` — the channel really held candidates.
- **Negative tripwire:** a filtered recall then asserts the correct retain/drop
  (e.g. a filter all graph docs violate empties the channel) — proving the filter
  is the narrowing force, not an empty-to-empty identity.

Mirror the live `kb` fixture from `tests/integration/test_filter_pushdown_graph.py`
(the live PG+Neo4j fixture around line 115) plus its `_pg_reachable()` / `_SKIP`
gate, and lower `retriever._config.min_entity_similarity = 0.0`. **This knob is test
infrastructure only — it is set on the per-test fixture's retriever instance to let
the deterministic-embedder entry entities clear the floor; the production default is
unchanged.** `neo4j_populated()` in `_harness.py` is the direct read-back probe for
the graph-write assertion.

## Multi-chunk denormalization (AC6)

AC6 seeds via **real `remember()` with LONG content** — content over the
configured `chunk_size` (default 512 tokens) so the real chunker emits **multiple
chunks per document**. This is the path that proves the denormalization contract on
real ingest (and is impossible on `seed_case`, which writes exactly one
`chunk_index=0` chunk per record). Follow the contract in
`tests/integration/matrix/test_chunk_denormalization_contract.py`: the eight
denormalized document keys + the three date columns are uniform across **all**
chunks of a document. AC6 then runs a `recall(filter=...)` on a denormalized key
and asserts the filter retains/drops the whole document consistently — i.e. the
doc-level `external_id` reconciliation holds even though several chunks back the one
document (any-chunk semantics).

## Acceptance-criteria → test-module map

| AC | module | proves |
|---|---|---|
| AC1 deterministic ingest | `test_filter_rowset_embedded.py` | real `remember()` pipeline, deterministic embeddings + extraction, stable across runs |
| AC2 graph fires / Neo4j populated | `test_graph_contribution.py` | `graph_chunk_count > 0` + `neo4j_populated()` after a real `remember()` ingest whose `stub_llm`/`plan_extraction` emits entities — the ONLY path that writes entities/edges (`seed_case` writes zero) |
| AC3 set-equality + filter enforcement | `test_graph_contribution.py` (honesty) + `test_filter_rowset_{embedded,graph,chronicle}.py` (row-set) | filtered recall retains/drops the right set; `test_graph_channel_drops_violating_chunks` proves the graph channel *enforces* `filter_ast` (entity-bearing keep/drop docs → only keep survives), distinct from the firing tripwire |
| AC4 reconciliation | `test_filter_rowset_embedded.py` (+ live graph/chronicle) | curated `f_*_cases` seeded via real `remember()`; `external_id`-keyed survivor set == `expected_ids` (see Reconciliation) |
| AC5 F-EXISTS reachability | `test_filter_rowset_embedded.py` (6 metadata states) + live graph/chronicle (2 `source_name` system-key states) | all 8 presence sub-states (`f_exists_cases()`) reconcile through real `remember()`; `test_f_exists_states_are_complete` guards against a corpus shrink |
| AC6 multi-chunk denorm | `test_multichunk_denorm.py` | denormalized doc/date keys uniform across chunks; filter is whole-doc consistent |

## CI gating + deferred lanes

- **The embedded sqlite_lance lane gates CI.** It is hermetic (no services), so it
  runs in the `test-unit` job (`pytest tests/unit/ tests/recall/ tests/e2e/ -m "not
  slow and not filter_conformance"`). The live `vectorcypher` / `chronicle` modules
  carry `pytest.mark.slow` + a reachability self-skip, so the same invocation collects
  and skips them cleanly; they execute under `make dev` + `NEO4J_INTEGRATION_TEST=1`.
- **Dedicated slow/e2e CI job for the live lanes is a follow-up** (a separate job that
  provisions PG+Neo4j and runs `-m "e2e and slow"`).
- **The skeleton-pgvector engine lane is deferred.** The shipped harness covers the
  embedded sqlite_lance, live VectorCypher (PG+Neo4j), and live Chronicle (PG-only)
  lanes. A skeleton-pgvector row-set lane is not included in this PR; it can be added
  by giving `_harness.rowset_cases` a `"skeleton_pgvector"` backend selector and a
  fixture that pins that engine.

## Implementation risks (flagged for Backend — verified against source)

These do not change the settled seam; they are sharp edges to handle in
`_harness.py` / the fixtures:

1. **Embedder dim must match the configured dimension.** The unit-vector stub must
   emit a vector of exactly `config.storage.embedding_dimension` (1536 on the PG
   pgvector column, 32 / `EMBED_DIM` on sqlite_lance). A dim mismatch fails the
   write or the search. The `graph_chunk_count > 0` pre-flight gate is the tripwire
   that catches a misconfigured entry gate — it must FAIL LOUD, never be softened.

2. **Namespace must round-trip from seed to recall.** The embedded vector channel
   (`SQLiteLanceVectorAdapter.search_similar`, vector.py:438) reads `chunks_vec`
   scoped by `namespace_id`; `remember()`d chunks land there under the same
   namespace, so seed and recall MUST use the identical namespace handle (the
   public `namespace_id`, resolved consistently). A namespace mismatch is the most
   likely cause of a zero-result recall on an otherwise-correct seed.

3. **HyDE default is `"auto"`** (not off) — the explicit `"never"` + env override +
   test-start assertion above is required, not optional.
