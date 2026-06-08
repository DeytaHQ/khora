"""Workload 11 — Multimodal document QA over a NASA Mars-rover corpus.

A real-world shape: a folder of markdown documents (NASA public-domain text on
the Perseverance and Curiosity rovers) where figures are embedded as ``<img>``
tags — maps, rover diagrams, traverse routes. khora indexes text, so the
pipeline is:

  1. **Parse** each ``.md`` — split into sections (text chunks) and pull out the
     ``<img>`` tags (figures).
  2. **Describe** every figure with a vision model, so the picture becomes
     searchable text. The image's path rides in metadata.
  3. **Remember** all of it — text + figure descriptions — into one namespace.
  4. **Answer** questions with retrieval-augmented generation: ``recall()`` the
     most relevant chunks, then have an LLM write a grounded answer *from that
     context only*, and report which documents / figures it drew on.

Because the corpus spans two rovers, the questions are cross-document — "which
has more cameras?", "are they in the same place?" — and some can only be
answered from a figure (the labeled instrument diagram, the traverse map).

Engine choice: **vectorcypher** — entities (rovers, instruments, craters,
measurements) are extracted from both the prose and the figure descriptions, so
one question can reach facts spread across several documents through the graph
as well as by vector similarity.

Run it
======
uv run python examples/30_workloads/11_multimodal_document_qa.py
python examples/30_workloads/11_multimodal_document_qa.py --config examples/khora.standard.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import logging
import mimetypes
import re
from pathlib import Path

from loguru import logger
from openai import AsyncOpenAI

from khora import Khora
from khora.config import KhoraConfig

logger.remove()
logger.add("khora.log", level="TRACE", enqueue=True)
for _noisy in ("httpx", "httpcore", "LiteLLM", "openai", "sqlalchemy.engine"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

_DEFAULT_CONFIG = Path(__file__).parent.parent / "khora.embedded.yaml"
_DOCS_DIR = Path(__file__).parent.parent / "data" / "mars_rovers"
_VISION_MODEL = "gpt-4o"
_ANSWER_MODEL = "gpt-4o"

_ENTITY_TYPES = ["SPACECRAFT", "INSTRUMENT", "LOCATION", "MEASUREMENT", "MISSION"]
_REL_TYPES = ["RELATES_TO", "PART_OF", "LOCATED_IN"]

_QUESTIONS = [
    "Are Perseverance and Curiosity exploring the same place on Mars?",
    "Is Curiosity bigger than Perseverance?",
    "Which rover has more cameras?",
    "Which instruments sit on Perseverance's robotic-arm turret?",
    "What has each rover discovered or demonstrated about Mars?",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    return parser.parse_args()


# ── Markdown parsing: sections → text chunks, <img> tags → figures ────────
_IMG_RE = re.compile(r"<img\b[^>]*>")
_SRC_RE = re.compile(r'src="([^"]+)"')
_ALT_RE = re.compile(r'alt="([^"]*)"')


def parse_doc(path: Path) -> tuple[dict, list[tuple[str, str]], list[tuple[str, str]]]:
    """Return (meta, text_sections, figures) for one markdown doc.

    text_sections: list of (heading, body). figures: list of (img_src, alt).
    """
    raw = path.read_text(encoding="utf-8")

    meta: dict[str, str] = {"title": path.stem, "source": ""}
    if raw.startswith("---"):
        _, front, raw = raw.split("---", 2)
        for line in front.strip().splitlines():
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip().strip('"')

    figures = [
        (m1.group(1), (m2.group(1) if m2 else ""))
        for tag in _IMG_RE.findall(raw)
        if (m1 := _SRC_RE.search(tag))
        for m2 in [_ALT_RE.search(tag)]
    ]

    # Strip img tags and figure-caption lines from the prose.
    body = _IMG_RE.sub("", raw)
    body = re.sub(r"^\*Figure:.*$", "", body, flags=re.M)

    # Split on "## " headings; keep the intro under the "# " title as Overview.
    parts = re.split(r"^##\s+(.+)$", body, flags=re.M)
    intro = re.sub(r"^#\s+.*$", "", parts[0], count=1, flags=re.M).strip()
    sections = [("Overview", intro)] if intro else []
    for i in range(1, len(parts), 2):
        text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if text:
            sections.append((parts[i].strip(), text))
    return meta, sections, figures


async def describe_image(path: Path, alt: str, *, client: AsyncOpenAI) -> str:
    """Describe a figure with a vision model; the alt text steers the focus."""
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    data_url = f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")
    resp = await client.chat.completions.create(
        model=_VISION_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Describe this figure for a search index in 2-4 factual sentences. "
                "Name the labeled parts, places, or values you can read. No preamble.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Context: {alt}"},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    )
    return (resp.choices[0].message.content or "").strip()


async def answer(question: str, recall, docs: dict, *, client: AsyncOpenAI) -> str:
    """Write a grounded answer from the recalled chunks only (RAG)."""
    blocks = []
    for c in recall.chunks:
        doc = docs.get(c.document_id)
        label = doc.title if doc else "?"
        img = (doc.metadata.get("image_path") if doc else None) or ""
        fig = f" (figure {Path(img).name})" if img else ""
        blocks.append(f"[{label}{fig}]\n{c.content}")
    context = "\n\n".join(blocks)
    resp = await client.chat.completions.create(
        model=_ANSWER_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Answer the question using ONLY the provided context about NASA's "
                "Mars rovers. Base every figure and number strictly on the context; do not add "
                "outside facts. Be concise (2-3 sentences). If the context lacks the answer, say so.",
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


async def main() -> None:
    args = _parse_args()
    docs_paths = sorted(_DOCS_DIR.glob("*.md"))
    if not docs_paths:
        print(f"No markdown docs in {_DOCS_DIR}.")
        return
    config = KhoraConfig.from_yaml(args.config)
    client = AsyncOpenAI()  # reads OPENAI_API_KEY

    # 1+2 — Parse every doc; describe its figures concurrently.
    chunks: list[dict] = []
    figure_jobs: list[tuple[str, dict, Path, str]] = []
    for path in docs_paths:
        meta, sections, figures = parse_doc(path)
        stem = path.stem
        for heading, text in sections:
            chunks.append(
                {
                    "content": f"{meta['title']} — {heading}\n\n{text}",
                    "title": f"{meta['title']}: {heading}",
                    "metadata": {"doc": stem, "source_url": meta.get("source", ""), "kind": "text"},
                }
            )
        for src, alt in figures:
            figure_jobs.append((stem, meta, (path.parent / src), alt))

    print(f"parsed {len(docs_paths)} docs → {len(chunks)} text chunks, {len(figure_jobs)} figures")
    described = await asyncio.gather(
        *(describe_image(p, alt, client=client) for _stem, _meta, p, alt in figure_jobs)
    )
    for (stem, meta, p, alt), desc in zip(figure_jobs, described):
        print(f"  described {p.name}: {desc[:70]}…")
        chunks.append(
            {
                "content": f"{meta['title']} — figure: {desc}",
                "title": f"{meta['title']}: figure",
                "metadata": {
                    "doc": stem,
                    "source_url": meta.get("source", ""),
                    "kind": "image",
                    "image_path": str(p),
                },
            }
        )

    async with Khora(config, engine="vectorcypher", run_migrations=True) as kb:
        ns_id = (await kb.create_namespace()).namespace_id

        # 3 — Remember every chunk (text + figure descriptions), capped concurrency.
        sem = asyncio.Semaphore(5)

        async def _remember(ch: dict) -> None:
            async with sem:
                await kb.remember(
                    ch["content"],
                    namespace=ns_id,
                    title=ch["title"],
                    source="mars_rovers",
                    metadata=ch["metadata"],
                    entity_types=_ENTITY_TYPES,
                    relationship_types=_REL_TYPES,
                )

        await asyncio.gather(*(_remember(c) for c in chunks))
        print(f"remembered {len(chunks)} chunks into the namespace")

        # 4 — Retrieval-augmented QA: recall → grounded LLM answer → sources.
        for question in _QUESTIONS:
            recall = await kb.recall(question, namespace=ns_id, limit=6)
            docs = {d.id: d for d in recall.documents}
            reply = await answer(question, recall, docs, client=client)
            sources = []
            for c in recall.chunks:
                d = docs.get(c.document_id)
                tag = d.metadata.get("doc") if d else None
                if tag and tag not in sources:
                    sources.append(tag)
            print(f"\nQ: {question}\nA: {reply}\n   sources: {', '.join(sources)}")


if __name__ == "__main__":
    asyncio.run(main())
