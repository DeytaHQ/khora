"""Workload 08 — Expertise config + cross-document entity resolution.

The textbook full-extraction, entity-centric workload: 50 candidate
resumes where every entity matters and **cross-document deduplication
is the whole story**. "Worked at Stripe AND knows k8s" requires
resolving "Stripe" / "Stripe Inc." / "Stripe, Inc." / "stripe.com" to
one canonical node, "Kubernetes" / "k8s" / "K8s" to another, and then
traversing skill edges between candidates and resolved entities.

The corpus (``examples/data/resumes.jsonl``, 50 entries) is constructed
with deliberate naming variants so the unifier has real work to do:

  * Stripe appears as 4 different surface forms across ~12 candidates.
  * Cloudflare, GitHub, Google, Meta, Datadog, OpenAI all have at
    least two surface forms.
  * "Rust" alternates with "the Rust programming language" and "Rust
    language".
  * "Kubernetes" alternates with "k8s" and "K8s".

This is what real ATS data looks like — the same employer rendered ten
ways across thousands of resumes — and what makes a memory layer with a
genuine entity-resolution pass earn its place over "stuff the corpus
into a single LLM context."

WHY VECTORCYPHER WITH ``skeleton_core_ratio=1.0``
=================================================
Full extraction on every chunk is what made the deprecated ``graphrag``
engine useful. VectorCypher at ``skeleton_core_ratio=1.0`` is the
documented replacement — same shape, one less codebase.

Why full extraction (vs the 0.70 default) for resumes?

  * Every skill / employer / project matters — there is no filler.
    The 70% importance heuristic is calibrated for prose where many
    chunks are background; resumes are uniformly dense.
  * Cross-document dedup is the headline. The more entities we
    extract per resume, the better the dedup signal.

EXPERTISECONFIG
===============
An inline ``ExpertiseConfig`` defines three entity types (CANDIDATE,
COMPANY, SKILL) plus the relationships connecting them, with a
purpose-built system prompt instructing the extractor to canonicalize
company names and respect negation. This is the ADR-022 stable public
API for taxonomies. In a real recruiter app you'd load this from YAML
(see ``examples/config/expertise/`` for the shape).

DUAL-BACKEND SUPPORT
====================
VectorCypher is **production-ready** on PostgreSQL + Neo4j and
**Experimental** on the embedded SQLite+LanceDB stack. The default
demo path is pg+neo4j (entity vectors are indexed there, so
write-time similarity dedup catches naming variants even without
explicit ``unify_entities``). On the embedded path the resolver has
no entity-vector index, so the explicit pipeline step is load-bearing.

Run it
======
uv run python examples/30_workloads/08_resume_search.py
python examples/30_workloads/08_resume_search.py
uv run python examples/30_workloads/08_resume_search.py --embedded
python examples/30_workloads/08_resume_search.py --embedded
uv run python examples/30_workloads/08_resume_search.py --config examples/khora.standard.yaml
python examples/30_workloads/08_resume_search.py --config examples/khora.standard.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

from loguru import logger

from khora import Khora
from khora.config import KhoraConfig
from khora.engines.vectorcypher.engine import VectorCypherConfig
from khora.extraction.skills.base import EntityTypeConfig, ExpertiseConfig, RelationshipTypeConfig

# ── Logging setup ───────────────────────────────────────────────────────
logger.remove()
logger.add("khora.log", level="TRACE", enqueue=True)
for _noisy in ("httpx", "httpcore", "LiteLLM", "openai", "sqlalchemy.engine"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)


_DEFAULT_CONFIG = Path(__file__).parent.parent / "khora.embedded.yaml"
_RESUMES_PATH = Path(__file__).parent.parent / "data" / "resumes.jsonl"

# Inline pg+neo4j builder — used when the demo runs without --config or
# --embedded. We don't load khora.standard.yaml because its documented
# env-var override (KHORA_DATABASE_URL / KHORA_NEO4J_URL) is ignored by
# from_yaml in v0.17.3 (https://github.com/DeytaHQ/khora/issues/859).
# Defaults match the docker-compose ports documented in CLAUDE.md.
_PG_URL = os.environ.get("KHORA_DATABASE_URL", "postgresql://khora:khora@localhost:5434/khora")
_NEO4J_URL = os.environ.get("KHORA_NEO4J_URL", "bolt://neo4j:pleaseletmein@localhost:7688")


def _inline_postgres_neo4j_config() -> KhoraConfig:
    return KhoraConfig.model_validate(
        {
            "storage": {"backend": "postgres", "embedding_dimension": 1536},
            "database_url": _PG_URL,
            "neo4j_url": _NEO4J_URL,
            "llm": {
                "model": "gpt-4o-mini",
                "api_key_env": "OPENAI_API_KEY",
                "embedding_model": "text-embedding-3-small",
                "embedding_dimension": 1536,
            },
        }
    )


def _load_resumes(path: Path = _RESUMES_PATH) -> list[dict]:
    """Read the resumes JSONL into a list of remember_batch-ready dicts.

    The file has one JSON object per line with shape
    ``{"id": ..., "name": ..., "source": ..., "content": ...}``. We
    pass through ``content`` (the body the extractor sees) and
    ``source`` (recorded on chunks for provenance); ``name`` is not
    a remember_batch field but ``title`` is, so we map it through.
    """
    docs: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            docs.append(
                {
                    "title": entry["name"],
                    "source": entry["source"],
                    "content": entry["content"],
                }
            )
    return docs


def _load_config() -> KhoraConfig:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML override. Without it, the demo uses an inline pg+neo4j config (env-var-overridable).",
    )
    parser.add_argument(
        "--embedded",
        action="store_true",
        help="Use the embedded sqlite_lance config instead of pg+neo4j. Default backend is pg+neo4j.",
    )
    args = parser.parse_args()
    if args.config is not None:
        return KhoraConfig.from_yaml(args.config)
    if args.embedded:
        return KhoraConfig.from_yaml(_DEFAULT_CONFIG)
    return _inline_postgres_neo4j_config()


def _build_expertise() -> ExpertiseConfig:
    """Build a domain-specific ExpertiseConfig for the recruiter use case.

    Four load-bearing rules in the system prompt:
      1. Canonicalize companies — strip legal suffixes, normalize
         punctuation. Critical: without this, "Stripe" / "Stripe Inc." /
         "Stripe, Inc." / "stripe.com" land as four rows that
         per-tuple dedup cannot collapse, because the embedded backend
         does not index entity vectors (no semantic similarity available).
      2. Canonicalize skill aliases — "k8s" / "K8s" → "Kubernetes",
         "Postgres" → "PostgreSQL", "Golang" → "Go". Same problem,
         same solution.
      3. Respect negation — gpt-4o-mini happily extracts "Rust" as a
         skill from "no Rust experience" without an explicit instruction.
      4. One CANDIDATE per name across the corpus.
    """
    return ExpertiseConfig(
        name="recruiter_demo",
        description="Recruiter-style ontology: candidates, employers, skills.",
        system_prompt=(
            "You are extracting structured information from short candidate "
            "resumes for a technical recruiter. Follow these rules strictly:\n"
            "1. Companies: ALWAYS canonicalize to the brand name with no "
            "legal suffix and no URL. 'Stripe Inc.', 'Stripe, Inc.', "
            "'Stripe', 'stripe.com' all extract as 'Stripe'. Apply the "
            "same shape to any company name with Inc./Corp./Ltd./GmbH/LLC "
            "suffixes or .com/.io/.ai TLDs.\n"
            "2. Skills: canonicalize common aliases. 'k8s' / 'K8s' → "
            "'Kubernetes'. 'Postgres' / 'pg' → 'PostgreSQL'. 'Golang' → "
            "'Go'. 'TS' → 'TypeScript'. 'the Rust programming language' / "
            "'Rust language' → 'Rust'.\n"
            "3. Negation: if a candidate explicitly LACKS a skill ('no "
            "Rust experience', 'never used X', 'has not worked with Y'), "
            "DO NOT create a HAS_SKILL relationship to that skill.\n"
            "4. Only extract these entity types: CANDIDATE, COMPANY, SKILL. "
            "Ignore standard library names, framework keywords, and other "
            "incidental technologies.\n"
            "5. One CANDIDATE entity per person — use the full name as it "
            "appears in the blurb."
        ),
        entity_types=[
            EntityTypeConfig(
                name="CANDIDATE",
                description="A job candidate referenced by name (real persons, not pronouns).",
                identifiers=["name"],
            ),
            EntityTypeConfig(
                name="COMPANY",
                description="A current or past employer organisation.",
                identifiers=["name"],
                aliases=["EMPLOYER", "ORGANIZATION"],
            ),
            EntityTypeConfig(
                name="SKILL",
                description="A technical skill, language, framework, or technology.",
                identifiers=["name"],
                aliases=["TECHNOLOGY", "TOOL"],
            ),
        ],
        relationship_types=[
            RelationshipTypeConfig(
                name="WORKED_AT",
                description="CANDIDATE was employed by COMPANY.",
                source_types=["CANDIDATE"],
                target_types=["COMPANY"],
                properties=["start_year", "end_year", "role"],
            ),
            RelationshipTypeConfig(
                name="HAS_SKILL",
                description="CANDIDATE has experience with SKILL.",
                source_types=["CANDIDATE"],
                target_types=["SKILL"],
            ),
        ],
    )


def _print_top(entities, label: str, top_n: int = 12) -> None:
    """Print top entities by mention_count with a header."""
    print(f"\n{label}:")
    by_mentions = sorted(entities, key=lambda e: -e.mention_count)[:top_n]
    if not by_mentions:
        print("  (none)")
        return
    width = max(len(repr(e.name)) for e in by_mentions)
    for ent in by_mentions:
        print(f"  {repr(ent.name):<{width}}  mentions={ent.mention_count}")
    if len(entities) > top_n:
        print(f"  … and {len(entities) - top_n} more")


def _variant_buckets(entities, variants_per_brand: dict[str, list[str]]) -> None:
    """Surface the naming-variant problem explicitly.

    For each brand we expect to be a single entity after unify, list the
    surface forms that *actually* materialized in the entity table. If
    the unifier did its job, only one row per brand survives.
    """
    by_name = {e.name.lower(): e for e in entities}
    for brand, surface_forms in variants_per_brand.items():
        present = [s for s in surface_forms if s.lower() in by_name]
        if len(present) <= 1:
            verdict = "ok (1 row)" if len(present) == 1 else "absent"
            print(f"  {brand:14s} {verdict}")
        else:
            mentions = sum(by_name[s.lower()].mention_count for s in present)
            print(f"  {brand:14s} {len(present)} variants survive — {present} ({mentions} total mentions)")


def _show_progress(done: int, total: int) -> None:
    """Print every 10th completed doc + the final count."""
    if done == total or done % 10 == 0:
        print(f"  …{done}/{total}")


async def main() -> None:
    config = _load_config()
    expertise = _build_expertise()
    resumes = _load_resumes()

    engine_kwargs = {
        "vectorcypher_config": VectorCypherConfig(
            skeleton_core_ratio=1.0,
            min_extraction_tokens=0,
        ),
    }

    # Brands we deliberately seeded with surface-form variants. The
    # printer will check whether the unifier collapsed each to a single
    # entity or left N variants standing.
    _EXPECTED_BRANDS = {
        "Stripe": ["Stripe", "Stripe Inc.", "Stripe, Inc.", "stripe.com"],
        "Cloudflare": ["Cloudflare", "Cloudflare, Inc."],
        "Google": ["Google", "Google LLC", "Alphabet (Google)"],
        "GitHub": ["GitHub", "GitHub Inc.", "GitHub at Microsoft"],
        "Meta": ["Meta", "Meta Platforms", "Facebook"],
        "Datadog": ["Datadog", "Datadog Inc."],
        "OpenAI": ["OpenAI", "Open AI"],
        "Kubernetes": ["Kubernetes", "k8s", "K8s"],
        "PostgreSQL": ["PostgreSQL", "Postgres", "pg"],
        "Rust": ["Rust", "the Rust programming language", "Rust language"],
    }

    async with Khora(config, engine="vectorcypher", engine_kwargs=engine_kwargs, run_migrations=True) as kb:
        namespace = await kb.create_namespace()
        ns_id = namespace.namespace_id

        # ── Ingest 50 resumes with the ExpertiseConfig ─────────────
        # remember_batch fans out at max_concurrent=5 by default. The
        # ExpertiseConfig + canonicalization rules above are what make
        # cross-doc dedup possible at write time on the pg+neo4j stack
        # (entity vectors indexed → similarity dedup catches variants
        # the LLM didn't already collapse).
        print(f"ingesting {len(resumes)} resumes (skeleton_core_ratio=1.0)…")
        batch = await kb.remember_batch(
            resumes,
            namespace=ns_id,
            expertise=expertise,
            entity_types=expertise.get_entity_type_names(),
            relationship_types=expertise.get_relationship_type_names(),
            on_progress=_show_progress,
        )
        print(
            f"  {batch.processed}/{batch.total} processed, "
            f"{batch.entities} entities, {batch.relationships} relationships"
        )

        # ── Snapshot BEFORE unify_entities ─────────────────────────
        print("\n── BEFORE unify_entities ─────────────────────────────")
        companies_pre = await kb.list_entities(namespace=ns_id, entity_type="COMPANY", limit=200)
        skills_pre = await kb.list_entities(namespace=ns_id, entity_type="SKILL", limit=200)
        candidates_pre = await kb.list_entities(namespace=ns_id, entity_type="CANDIDATE", limit=200)
        print(f"  totals: CANDIDATE={len(candidates_pre)}, COMPANY={len(companies_pre)}, SKILL={len(skills_pre)}")
        _print_top(companies_pre, "Top companies by mention_count", top_n=10)
        _print_top(skills_pre, "Top skills by mention_count", top_n=10)
        print("\n  Surface-form survivors per brand (pre-unify):")
        _variant_buckets(companies_pre + skills_pre, _EXPECTED_BRANDS)

        # ── Cross-document entity unification ──────────────────────
        # The 3-phase ingest pipeline (stage → enrich → EXPAND, see
        # docs) registers ``unify_entities``. Importing the flows
        # module triggers the @pipeline decorator that registers it.
        from khora.pipelines.flows import expansion  # noqa: F401
        from khora.pipelines.registry import PipelineRegistry

        print("\nrunning unify_entities pipeline (cross-document dedup)…")
        registry = PipelineRegistry()
        info = registry.get("unify_entities")
        if info is None:
            print("  (unify_entities pipeline not registered — skipping)")
        else:
            storage = kb._engine._get_storage()  # type: ignore[union-attr]
            ns_row_id = await storage.resolve_namespace(ns_id)
            unify_result = await info.func(
                namespace_id=ns_row_id,
                storage=storage,
                expertise=expertise,
            )
            print(f"  {unify_result}")

        # ── Snapshot AFTER unify_entities ──────────────────────────
        print("\n── AFTER unify_entities ──────────────────────────────")
        companies_post = await kb.list_entities(namespace=ns_id, entity_type="COMPANY", limit=200)
        skills_post = await kb.list_entities(namespace=ns_id, entity_type="SKILL", limit=200)
        candidates_post = await kb.list_entities(namespace=ns_id, entity_type="CANDIDATE", limit=200)
        print(
            f"  totals: CANDIDATE={len(candidates_post)} "
            f"({len(candidates_pre) - len(candidates_post):+d}), "
            f"COMPANY={len(companies_post)} "
            f"({len(companies_pre) - len(companies_post):+d}), "
            f"SKILL={len(skills_post)} "
            f"({len(skills_pre) - len(skills_post):+d})"
        )
        _print_top(companies_post, "Top companies by mention_count", top_n=10)
        _print_top(skills_post, "Top skills by mention_count", top_n=10)
        print("\n  Surface-form survivors per brand (post-unify):")
        _variant_buckets(companies_post + skills_post, _EXPECTED_BRANDS)

        # ── Graph traversal: "candidates with Rust experience" ────
        # The entity-centric API: pick a SKILL entity, walk HAS_SKILL
        # edges back to CANDIDATE nodes. With 50 resumes and ~18 of
        # them mentioning Rust in one form or another, this should
        # surface a meaningful population (not a single hit like the
        # 4-resume version of this demo).
        print("\n── Graph traversal: candidates with Rust experience ──")
        rust = next((s for s in skills_post if s.name.lower() == "rust"), None)
        if rust is None:
            print("  (no canonical 'Rust' SKILL — unifier may have kept variants)")
        else:
            related = await kb.find_related_entities(
                rust.id,
                namespace=ns_id,
                max_depth=2,
                limit=50,
            )
            rust_candidates = [(ent, score) for ent, score in related if ent.entity_type == "CANDIDATE"]
            print(f"  {len(rust_candidates)} candidate(s) connected to Rust:")
            for ent, score in rust_candidates[:15]:
                print(f"  [{score:.3f}] {ent.name}")
            if len(rust_candidates) > 15:
                print(f"  … and {len(rust_candidates) - 15} more")

        # ── Graph traversal: Stripe alumni who know Kubernetes ────
        # Two-step intersection: candidates with WORKED_AT → Stripe
        # AND candidates with HAS_SKILL → Kubernetes. The kind of
        # query a recruiter actually asks; impossible to answer well
        # if Stripe variants haven't collapsed.
        print("\n── Graph traversal: Stripe alumni who know Kubernetes ──")
        stripe = next((c for c in companies_post if c.name.lower() == "stripe"), None)
        k8s = next((s for s in skills_post if s.name.lower() in {"kubernetes", "k8s"}), None)
        if stripe is None or k8s is None:
            print(
                f"  (cannot run — missing canonical entity: "
                f"stripe={'ok' if stripe else 'MISSING'}, "
                f"kubernetes={'ok' if k8s else 'MISSING'})"
            )
        else:
            stripe_related = await kb.find_related_entities(stripe.id, namespace=ns_id, max_depth=2, limit=50)
            k8s_related = await kb.find_related_entities(k8s.id, namespace=ns_id, max_depth=2, limit=50)
            stripe_alumni = {e.id: e for e, _ in stripe_related if e.entity_type == "CANDIDATE"}
            k8s_users = {e.id for e, _ in k8s_related if e.entity_type == "CANDIDATE"}
            overlap_ids = stripe_alumni.keys() & k8s_users
            overlap = sorted(stripe_alumni[i].name for i in overlap_ids)
            print(
                f"  {len(stripe_alumni)} Stripe alumni × {len(k8s_users)} Kubernetes users = "
                f"{len(overlap)} candidates in the intersection:"
            )
            for name in overlap:
                print(f"    • {name}")

        # ── Free-text recall as a sanity check ─────────────────────
        # The same population also has to be findable through normal
        # recall. With 50 resumes the rerankers will return chunks
        # from multiple candidates, and the entity panel surfaces the
        # CANDIDATE entities ranked by graph signal.
        print("\n── Free-text recall: 'rust engineer with payments experience' ──")
        result = await kb.recall(
            "rust engineer with payments experience",
            namespace=ns_id,
            limit=5,
        )
        for chunk in result.chunks:
            preview = chunk.content[:90].replace("\n", " ")
            print(f"  [{chunk.score:.3f}] {preview}{'…' if len(chunk.content) > 90 else ''}")
        if result.entities:
            cand_entities = [e for e in result.entities if e.entity_type == "CANDIDATE"][:5]
            if cand_entities:
                print("  entity panel (top candidates):")
                for ent in cand_entities:
                    print(f"    • {ent.name}")


if __name__ == "__main__":
    asyncio.run(main())
