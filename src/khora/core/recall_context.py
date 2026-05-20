"""Canonical ``context_text`` rendering for :class:`RecallResult`.

This module owns the byte-stable text representation that callers (LLMs,
prompt templates, snapshot tests) consume:

- Chunks are grouped by document title (``DocumentProjection.title``) and
  joined with ``\\n\\n---\\n\\n`` separators; titled groups are prefixed
  with ``--- From: <title> ---``.
- An entities section (``--- Entities ---``) is appended when the result
  carries entities, deduplicated by entity id.
- A relationships section (``--- Relationships ---``) is appended when the
  result carries relationships, deduplicated by
  ``(source_entity_id, target_entity_id, relationship_type)``.

Endpoint names for relationships are resolved from ``RecallResult.entities``
by id, falling back to ``str(uuid)`` when the referenced entity is absent
from the result.

Public surface (``__all__``): :func:`context_text`, :func:`format_entity_section`,
:func:`format_relationship_section`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from khora.core.models.recall import (
        DocumentProjection,
        RecallChunk,
        RecallEntity,
        RecallRelationship,
        RecallResult,
    )


def format_entity_section(entities: list[RecallEntity]) -> str:
    """Render the ``--- Entities ---`` section for a list of recall entities.

    Returns an empty string when ``entities`` is empty. Deduplicates by
    entity id. Lines have the form ``- <name> (<type>): <description>``
    when a description is present, otherwise ``- <name> (<type>)``.
    """
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


def format_relationship_section(
    relationships: list[RecallRelationship],
    entity_names: dict[UUID, str],
) -> str:
    """Render the ``--- Relationships ---`` section for recall relationships.

    Returns an empty string when ``relationships`` is empty. Deduplicates by
    ``(source_entity_id, target_entity_id, relationship_type)``. Endpoint
    names are resolved via ``entity_names`` (typically built from
    ``RecallResult.entities``), falling back to ``str(uuid)`` for endpoints
    that are not present in the lookup.
    """
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


def _group_chunks_by_title(
    chunks: list[RecallChunk],
    documents: list[DocumentProjection],
    max_chunks: int,
) -> str:
    """Group the first ``max_chunks`` chunks by document title and render them.

    Titled groups become ``--- From: <title> ---`` sections joined by
    ``\\n\\n---\\n\\n``; untitled chunks are concatenated without a header.
    """
    titles_by_doc: dict[UUID, str] = {doc.id: (doc.title or "") for doc in documents}

    groups: dict[str, list[str]] = {}
    for chunk in chunks[:max_chunks]:
        title = titles_by_doc.get(chunk.document_id, "")
        groups.setdefault(title, []).append(chunk.content)

    sections: list[str] = []
    for title, contents in groups.items():
        if title:
            sections.append(f"--- From: {title} ---\n" + "\n\n".join(contents))
        else:
            sections.extend(contents)
    return "\n\n---\n\n".join(sections)


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
    text = _group_chunks_by_title(list(result.chunks), list(result.documents), max_chunks)

    entity_section = format_entity_section(list(result.entities))
    if entity_section:
        text = text + entity_section if text else entity_section

    entity_names: dict[UUID, str] = {e.id: e.name for e in result.entities}
    relationship_section = format_relationship_section(list(result.relationships), entity_names)
    if relationship_section:
        text = text + relationship_section if text else relationship_section

    return text


__all__ = ["context_text", "format_entity_section", "format_relationship_section"]
