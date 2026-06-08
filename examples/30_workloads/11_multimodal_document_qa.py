"""Workload 11 — Multimodal document QA over a NASA Mars-rover corpus.

A real-world shape: a folder of markdown documents (NASA public-domain text on
the Perseverance and Curiosity rovers) where figures are embedded as Markdown
images (``![alt](path)``) — maps, rover diagrams, traverse routes. khora indexes
text, so the pipeline is:

  1. **Parse** each ``.md`` — split it into sections by heading (one chunk per
     section) and pull out the Markdown image embeds (figures).
  2. **Describe** every figure with a vision model — the alt text steers it — so
     the picture becomes searchable text. The image path rides in metadata.
  3. **Remember** all of it with ``remember_batch``, giving every chunk a stable
     ``external_id`` so recalled context can be cited.
  4. **Answer** questions with retrieval-augmented generation: ``recall()`` the
     most relevant chunks, have an LLM write a grounded answer *from that context
     only*, and print the ``external_id`` of every source it drew on.

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
_VISION_MODEL = "gpt-5.5"
_ANSWER_MODEL = "gpt-5.5"

_ENTITY_TYPES = ["SPACECRAFT", "INSTRUMENT", "LOCATION", "MEASUREMENT", "MISSION"]
_REL_TYPES = ["RELATES_TO", "PART_OF", "LOCATED_IN"]

_VISION_SYSTEM = (
    "You are describing a figure — a map, chart, or labeled diagram — for a search index, in "
    "enough detail that a reader could reconstruct it without seeing it. Be exhaustive: a long, "
    "complete description is the goal. Cover:\n"
    "1. CONTENT — transcribe every visible label, place name, axis title, legend entry, and number. "
    "Also describe physical features that are drawn but NOT labeled, and COUNT the repeated parts you "
    "can see — for example, how many wheels, antennas, cameras, booms, or masts are visible.\n"
    "2. LAYOUT — say where each element sits and how elements relate in space. For a map, give "
    "positions with compass directions (north/south/east/west) and use the scale bar to estimate "
    "distances between places (for example, 'X is north of Y, roughly 2 km away'). For a diagram, "
    "say what is mounted where and what is next to, above or below, in front of or behind, or hidden "
    "behind what.\n"
    "3. STORY — if the figure shows a route or sequence such as a rover's traverse, narrate the "
    "path in order from start to finish: the starting point, each stop in sequence, and the overall "
    "direction of travel. If it shows an object, summarize what it is and what each major part does.\n"
    "Do not limit the length — completeness is more important than brevity. No preamble."
)

_QUESTIONS = [
    "Are Perseverance and Curiosity exploring the same place on Mars?",
    "Is Curiosity bigger than Perseverance?",
    "Which rover has more cameras?",
    "Which instruments sit on Perseverance's robotic-arm turret?",
    "What has each rover discovered or demonstrated about Mars?",
    "What path did Curiosity traverse since landing?",
    "What is in front of each rover?",
    "How many wheels does Curiosity have?",
    # Figure-only: these facts appear only in the described images, not in any prose
    # doc — so correct answers prove the described images were used.
    "On Curiosity's route map, how many geologic waypoints are marked?",
    "On the Curiosity instrument diagram, which side of the rover is the DAN instrument on?",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    return parser.parse_args()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "section"


# ── Markdown parsing: sections → text chunks, ![alt](src) images → figures ──
_IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")  # markdown image: ![alt](src)


def parse_doc(path: Path) -> tuple[dict, list[tuple[str, str]], list[tuple[str, str]]]:
    """Return (meta, text_sections, figures) for one markdown doc.

    text_sections: list of (heading, body) — one per ``##`` section, plus the
    intro under the ``#`` title. figures: list of (img_src, alt).
    """
    raw = path.read_text(encoding="utf-8")

    meta: dict[str, str] = {"title": path.stem, "source": ""}
    if raw.startswith("---"):
        _, front, raw = raw.split("---", 2)
        for line in front.strip().splitlines():
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip().strip('"')

    figures = [(m.group(2), m.group(1)) for m in _IMG_RE.finditer(raw)]  # (src, alt)

    # Strip image embeds and figure-caption lines from the prose.
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
    """Describe a figure with a vision model. The doc's alt text is passed in so
    the model knows what the figure is meant to convey before reading it."""
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    data_url = f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")
    resp = await client.chat.completions.create(
        model=_VISION_MODEL,
        messages=[
            {"role": "system", "content": _VISION_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Caption / alt text for this figure: {alt}\n\nDescribe the figure."},
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
        cite = (doc.external_id if doc else None) or "?"
        blocks.append(f"[{cite}]\n{c.content}")
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
    docs_paths = sorted(p for p in _DOCS_DIR.glob("*.md") if p.name != "CREDITS.md")
    if not docs_paths:
        print(f"No markdown docs in {_DOCS_DIR}.")
        return
    config = KhoraConfig.from_yaml(args.config)
    client = AsyncOpenAI()  # reads OPENAI_API_KEY

    # 1 — Parse each doc into per-section text chunks; collect figures to describe.
    batch: list[dict] = []
    figure_jobs: list[tuple[str, dict, Path, str]] = []
    for path in docs_paths:
        meta, sections, figures = parse_doc(path)
        stem = path.stem
        for heading, text in sections:
            batch.append(
                {
                    "content": f"{meta['title']} — {heading}\n\n{text}",
                    "title": f"{meta['title']}: {heading}",
                    "source": "mars_rovers",
                    "external_id": f"{stem}#{_slug(heading)}",
                    "metadata": {"doc": stem, "source_url": meta.get("source", ""), "kind": "text"},
                }
            )
        for src, alt in figures:
            figure_jobs.append((stem, meta, path.parent / src, alt))
    print(f"parsed {len(docs_paths)} docs → {len(batch)} text chunks, {len(figure_jobs)} figures")

    # 2 — Describe every figure with the vision model (alt text included).
    described = await asyncio.gather(*(describe_image(p, alt, client=client) for _stem, _meta, p, alt in figure_jobs))
    for (stem, meta, p, alt), desc in zip(figure_jobs, described):
        print(f"\n── figure: {p.name} ──\n{desc}")
        batch.append(
            {
                "content": f"{meta['title']} — figure ({p.name}): {desc}",
                "title": f"{meta['title']}: figure",
                "source": "mars_rovers",
                "external_id": f"{stem}#figure-{p.stem}",
                "metadata": {"doc": stem, "source_url": meta.get("source", ""), "kind": "image", "image_path": str(p)},
            }
        )

    async with Khora(config, engine="vectorcypher", run_migrations=True) as kb:
        ns_id = (await kb.create_namespace()).namespace_id

        # 3 — Remember everything in one batch; each chunk keeps its external_id.
        result = await kb.remember_batch(
            batch,
            namespace=ns_id,
            max_concurrent=5,
            entity_types=_ENTITY_TYPES,
            relationship_types=_REL_TYPES,
        )
        print(f"\nremembered {result.processed} chunks (chunks={result.chunks}, entities={result.entities})")

        # 4 — Retrieval-augmented QA: recall → grounded answer → cite external_ids.
        #     limit=10 so cross-rover comparison questions surface chunks from both rovers.
        for question in _QUESTIONS:
            recall = await kb.recall(question, namespace=ns_id, limit=10)
            docs = {d.id: d for d in recall.documents}
            reply = await answer(question, recall, docs, client=client)
            sources: list[str] = []
            for c in recall.chunks:
                d = docs.get(c.document_id)
                eid = d.external_id if d else None
                if eid and eid not in sources:
                    sources.append(eid)
            print(f"\nQ: {question}\nA: {reply}\n   sources: {', '.join(sources)}")


if __name__ == "__main__":
    asyncio.run(main())
