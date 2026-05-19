"""Public ``context_text`` helper — render a ``RecallResult`` as an LLM context string.

The output format mirrors the legacy ``RecallResult.context_text`` field byte-for-byte:

- Chunks are grouped by document title (``DocumentProjection.title``) and
  joined with ``\\n\\n---\\n\\n`` separators; titled groups are prefixed with
  ``--- From: <title> ---``.
- An entities section (``--- Entities ---``) is appended when the result
  carries entities, deduplicated by entity id.
- A relationships section (``--- Relationships ---``) is appended when the
  result carries relationships, deduplicated by
  ``(source_entity_id, target_entity_id, relationship_type)``.

The implementation lifts the canonical contract from
``khora.query.engine.format_entity_section`` / ``format_relationship_section`` /
``QueryResult.get_context_text`` and adapts it to the typed
``RecallResult`` / ``RecallChunk`` / ``RecallEntity`` / ``RecallRelationship``
projections. Endpoint names for relationships are resolved from
``RecallResult.entities`` by id (falling back to ``str(uuid)`` when the
referenced entity is absent from the result), matching the legacy fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from khora.core.models.recall import (
        RecallEntity,
        RecallRelationship,
        RecallResult,
    )


def _format_entity_section(entities: list[RecallEntity]) -> str:
    if not entities:
        return ""
    seen: set[UUID] = set()
    lines: list[str] = []
    for entity in entities:
        if entity.id in seen:
            continue
        seen.add(entity.id)
        if entity.description:
            lines.append(f"- {entity.name} ({entity.entity_type}): {entity.description}")
        else:
            lines.append(f"- {entity.name} ({entity.entity_type})")
    if not lines:
        return ""
    return "\n\n--- Entities ---\n\n" + "\n".join(lines)


def _format_relationship_section(
    relationships: list[RecallRelationship],
    entity_names: dict[UUID, str],
) -> str:
    if not relationships:
        return ""
    seen: set[tuple[UUID, UUID, str]] = set()
    lines: list[str] = []
    for rel in relationships:
        dedup_key = (rel.source_entity_id, rel.target_entity_id, rel.relationship_type)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        source_name = entity_names.get(rel.source_entity_id) or str(rel.source_entity_id)
        target_name = entity_names.get(rel.target_entity_id) or str(rel.target_entity_id)
        line = f"- {source_name} --{rel.relationship_type}--> {target_name}"
        if rel.description:
            line += f": {rel.description}"
        lines.append(line)
    if not lines:
        return ""
    return "\n\n--- Relationships ---\n\n" + "\n".join(lines)


def context_text(result: RecallResult, *, max_chunks: int = 5) -> str:
    """Render a :class:`RecallResult` as a flat text context string for an LLM.

    Groups the first ``max_chunks`` chunks by document title and appends
    entity / relationship sections when present.

    Args:
        result: The :class:`RecallResult` returned by :meth:`Khora.recall`.
        max_chunks: Maximum number of chunks to include. Default ``5``.

    Returns:
        The concatenated context string. Empty string when there are no
        chunks, entities, or relationships to render.
    """
    titles_by_doc: dict[UUID, str] = {doc.id: (doc.title or "") for doc in result.documents}

    groups: dict[str, list[str]] = {}
    for chunk in result.chunks[:max_chunks]:
        title = titles_by_doc.get(chunk.document_id, "")
        groups.setdefault(title, []).append(chunk.content)

    sections: list[str] = []
    for title, contents in groups.items():
        if title:
            sections.append(f"--- From: {title} ---\n" + "\n\n".join(contents))
        else:
            sections.extend(contents)
    text = "\n\n---\n\n".join(sections)

    entity_section = _format_entity_section(list(result.entities))
    if entity_section:
        text = text + entity_section if text else entity_section

    entity_names: dict[UUID, str] = {e.id: e.name for e in result.entities}
    relationship_section = _format_relationship_section(list(result.relationships), entity_names)
    if relationship_section:
        text = text + relationship_section if text else relationship_section

    return text


__all__ = ["context_text"]
