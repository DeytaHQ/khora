"""Workload 11 — Image ingestion: describe figures, remember them, recall across them.

Khora indexes **text**. An image can't be embedded directly, so the durable
pattern is: describe each figure with a vision model, then ``remember()`` the
description as a normal chunk — with the **image's location carried in the
document metadata**. Recall then surfaces figures by *what they show*, and the
caller reads the image location off each recalled chunk's document to display
the original.

The payoff is cross-image recall: ingest a small set of climate figures (a CO2
chart, a temperature map, a carbon-cycle diagram) and ask **one question that no
single figure answers** — "what evidence links rising carbon dioxide to global
warming?". Recall pulls the relevant figures from across the set, and (on
vectorcypher) shared entities like "carbon dioxide" link them in the graph too.

This is the production image-ingestion pipeline (chunker → vision describer →
ingester) collapsed into one file. Needs ``OPENAI_API_KEY`` (for the vision call
and khora's entity extraction) and one or more images in ``--images-dir``.

Engine choice: **vectorcypher** — the descriptions are prose, so extracting
entities lets one query reach several figures through the graph as well as by
vector similarity.

Run it
======
uv run python examples/30_workloads/11_image_ingestion.py
python examples/30_workloads/11_image_ingestion.py --config examples/khora.standard.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import logging
import mimetypes
import os
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
_DEFAULT_IMAGES = Path(__file__).parent.parent / "data" / "images"
_DEFAULT_MODEL = "gpt-4o"
_DEFAULT_QUERY = "What evidence links rising carbon dioxide to global warming?"
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument(
        "--images-dir", type=Path, default=_DEFAULT_IMAGES, help="Folder of images to ingest"
    )
    parser.add_argument("--model", default=_DEFAULT_MODEL, help="Vision model")
    parser.add_argument(
        "--query", default=_DEFAULT_QUERY, help="Recall query — should span several figures"
    )
    return parser.parse_args()


async def describe_image(image_path: Path, *, client: AsyncOpenAI, model: str) -> str:
    """Describe a local image with OpenAI; return a short factual description.

    The image is sent inline as a base64 data URL, so no public hosting is
    needed for the describe step.
    """
    mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
    data_url = f"data:{mime};base64," + base64.b64encode(image_path.read_bytes()).decode("ascii")
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You describe figures for a search index. Reply with 2-3 factual "
                    "sentences about what the image shows. No preamble."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this figure."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    )
    return (resp.choices[0].message.content or "").strip()


async def main() -> None:
    args = _parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this example — the vision call needs it.")
        return
    images = sorted(p for p in args.images_dir.glob("*") if p.suffix.lower() in _IMAGE_SUFFIXES)
    if not images:
        print(f"No images in {args.images_dir}. Drop a few figures there (png/jpg) and re-run.")
        return
    config = KhoraConfig.from_yaml(args.config)
    client = AsyncOpenAI()

    async with Khora(config, engine="vectorcypher", run_migrations=True) as kb:
        ns_id = (await kb.create_namespace()).namespace_id

        # 1 — Describe each figure and remember it, with the image's location
        #     in metadata. The description is what gets embedded + entity-
        #     extracted; the location is how the recall app fetches the original.
        for img in images:
            description = await describe_image(img, client=client, model=args.model)
            print(f"\n{img.name}:\n  {description}")
            await kb.remember(
                description,
                namespace=ns_id,
                title=f"Figure — {img.stem}",
                source="examples",
                metadata={"modality": "image", "image_path": str(img)},
                external_id=f"examples/image/{img.stem}",
                entity_types=["CONCEPT", "LOCATION"],
                relationship_types=["RELATES_TO"],
            )

        # 2 — Ask one question that no single figure answers. Recall reads the
        #     image location back off each chunk's *document* projection
        #     (chunks reference a document_id into recall.documents).
        recall = await kb.recall(args.query, namespace=ns_id, limit=5)
        docs = {d.id: d for d in recall.documents}
        print(f"\nQ: {args.query}")
        spanned: list[str] = []
        for c in recall.chunks:
            doc = docs.get(c.document_id)
            location = doc.metadata.get("image_path") if doc else None
            name = Path(location).name if location else "?"
            print(f"  [{c.score:.2f}] {name} — {c.content[:80]}…")
            if location and location not in spanned:
                spanned.append(location)

        print(f"\n→ one question answered from {len(spanned)} image(s): {[Path(p).name for p in spanned]}")


if __name__ == "__main__":
    asyncio.run(main())
