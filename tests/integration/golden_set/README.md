# Golden-set retrieval regression tests (#1479)

A hermetic CI safety net that pins retrieval **rank positions** so any future
ranking change that demotes a known-correct chunk fails loudly - without a paid
benchmark run. This is the guardrail the fusion/seeding changes (Wave C, #1480)
depend on.

## What's here

| File | Role |
|------|------|
| `corpus.json` | The fixed corpus (~18 short docs) + the golden queries with pinned max ranks. |
| `test_golden_set_recall.py` | Ingests the corpus into `sqlite_lance`, runs each query through the full VectorCypher recall path, asserts each gold doc appears at or above its pinned rank in the **returned order**. Also asserts run-to-run determinism. |
| `test_shadow_scoring.py` | Exercises the shadow-scoring A/B harness (flag OFF = no key + unchanged results; flag ON = report present + results identical). |

## How ranks are asserted

* **Order, not membership (#1433).** The returned `RecallResult.chunks` order
  *is* the authoritative relevance ranking (fusion + boosts + rerank + MMR).
  We assert on that order, NOT on a re-sort by `chunk.score`. `_rank_of_doc`
  computes the 1-based position of the first chunk whose `document_id` matches
  the gold doc.
* **Loose but real.** `max_rank` is "gold must be within top-N", not an exact
  position - loose enough to not be brittle across harmless reorderings, tight
  enough to catch a genuine demotion. A #1463-class MMR bug (which floated
  graph-only chunks over the true top hit) would push a gold doc out of its
  top-N and trip the test.

## Determinism (no flakiness)

* **No LLM, no network.** The entity extractor is stubbed (`stub_llm`) and the
  embedder is replaced with `vocab_embedding` - an L2-normalized bag-of-words
  over a fixed vocabulary built from the corpus. Cosine similarity therefore
  tracks lexical overlap, so a query genuinely ranks its lexically-matching
  gold doc high. Unlike the SHA-hash `fake_embedding` helper (deterministic but
  *not* semantic), this makes rank assertions meaningful.
* **No stochastic ranking stages.** The golden config forces `enable_hyde` off
  (HyDE would issue a non-stubbed chat LLM call) and disables the cross-encoder
  reranker (it downloads a real model whose near-tied tail scores resolve by
  random UUID). MMR stays ON so the #1463-class demotion is still exercised.
* **Fixed everything.** Same corpus + same query -> same order, every run
  (the vocabulary is sorted before indexing). `test_golden_set_recall_is_deterministic`
  guards this by repeating the same recall three times on one ingest. Two
  *independent* ingests are not compared: candidate ties tie-break on the
  per-ingest-random document UUID, so the tail legitimately reorders across
  ingests - but a single ingest is fully reproducible, which is what CI needs.
* **Fast.** ~18 docs, one ingest, one recall per query - a couple of seconds.
  Runs in the main `test` CI job under the `integration` + `embedded` markers
  (no Docker; self-skips if `aiosqlite`/`lancedb` aren't installed).

## Adding a new golden case

1. **Add distinctive documents** to `corpus.json` under `documents`. Give each
   a stable `doc_id` (a string; the test maps it to the per-ingest UUID) and a
   short, single-chunk `content` (a few sentences - stay under the 512-token
   chunk size so one doc = one chunk). For a new archetype, include a couple of
   *distractor* docs that share vocabulary but are not the answer, so the query
   has something to rank against. Optional `occurred_at` (ISO-8601) for
   temporal cases.
2. **Add the query** under `queries` with:
   * `query_id` - unique slug (names the case in failure output),
   * `archetype` - one of the retrieval shapes (single-needle factoid,
     multi-hop / relational, temporal / recency, multi-entity, ...),
   * `query` - the natural-language question,
   * `gold_doc_ids` - the `doc_id`(s) that MUST be retrieved,
   * `max_rank` - the loosest rank the gold may occupy (e.g. `3` = top-3).
3. **Pick `max_rank` empirically, then tighten.** Run
   `pytest tests/integration/golden_set/test_golden_set_recall.py` and read the
   reported rank. Set `max_rank` to that rank (or one above it for headroom) -
   loose enough to survive harmless reorderings, tight enough that a real
   demotion trips it. Because the embedding is lexical, make the query share
   distinctive tokens with its gold doc (and NOT with the distractors).
4. **Keep it hermetic.** Do not add anything that needs a network call, an LLM,
   or a Docker service.

## The shadow-scoring harness (#1479, deliverable 2)

`KHORA_QUERY_SHADOW_SCORING` (default OFF) makes the VectorCypher engine compute
a **candidate** ranking alongside the live **incumbent** ranking on each recall
and record the divergence under `RecallResult.engine_info["shadow_scoring"]`.
The returned results are always the incumbent's - shadow is observe-only. See
`src/khora/engines/vectorcypher/shadow_scoring.py` for the report shape and the
`KHORA_QUERY_SHADOW_SCORING_STRATEGY` knob (`score_sort` | `identity`).
