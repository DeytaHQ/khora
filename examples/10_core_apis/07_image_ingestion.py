"""Core API — Image ingestion: describe figures, remember them, recall across them.

Khora indexes **text**. To make an image searchable, describe it with a vision
model and ``remember()`` the description — carrying the image's location in
metadata so the app can fetch the original to display. One question then recalls
across several figures, even though no single figure answers it.

Engine choice: **skeleton** — cost-efficient hybrid search. We embed the
descriptions and retrieve by similarity; no typed entity extraction needed
(``entity_types`` / ``relationship_types`` are passed empty).

Run it
======
uv run python examples/10_core_apis/07_image_ingestion.py
python examples/10_core_apis/07_image_ingestion.py
"""

from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
from pathlib import Path

from loguru import logger
from openai import AsyncOpenAI

from khora import Khora
from khora.config import KhoraConfig

logger.remove()
logger.add("khora.log", level="TRACE", enqueue=True)
for _noisy in ("httpx", "httpcore", "LiteLLM", "openai", "sqlalchemy.engine"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

_CONFIG = Path(__file__).parent.parent / "khora.embedded.yaml"
_IMAGES = Path(__file__).parent.parent / "data" / "images"
_MODEL = "gpt-4o"
_QUERY = "What evidence links rising carbon dioxide to global warming?"
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


async def describe_image(path: Path, *, client: AsyncOpenAI) -> str:
    """Describe a local image with OpenAI; return a short factual description."""
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    data_url = f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")
    resp = await client.chat.completions.create(
        model=_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Describe this figure for a search index in 2-3 factual sentences. No preamble.",
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
    images = sorted(p for p in _IMAGES.glob("*") if p.suffix.lower() in _IMAGE_SUFFIXES)
    if not images:
        print(f"No images in {_IMAGES}. Add a few figures (png/jpg) and re-run.")
        return
    config = KhoraConfig.from_yaml(_CONFIG)
    client = AsyncOpenAI()  # reads OPENAI_API_KEY

    async with Khora(config, engine="skeleton", run_migrations=True) as kb:
        ns_id = (await kb.create_namespace()).namespace_id

        # Describe each figure and remember it, with the image location in metadata.
        for img in images:
            description = await describe_image(img, client=client)
            print(f"\n{img.name}:\n  {description}")
            await kb.remember(
                description,
                namespace=ns_id,
                metadata={"image_path": str(img)},
                entity_types=[],
                relationship_types=[],
            )

        # One question that no single figure answers — recall spans the set.
        recall = await kb.recall(_QUERY, namespace=ns_id, limit=5)
        docs = {d.id: d for d in recall.documents}
        print(f"\nQ: {_QUERY}")
        spanned: set[str] = set()
        for c in recall.chunks:
            doc = docs.get(c.document_id)
            path = doc.metadata.get("image_path") if doc else None
            name = Path(path).name if path else "?"
            print(f"  [{c.score:.2f}] {name} — {c.content[:80]}…")
            if path:
                spanned.add(path)
        print(f"\n→ one question answered from {len(spanned)} image(s)")


if __name__ == "__main__":
    asyncio.run(main())
