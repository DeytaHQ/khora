"""Entity extraction task."""

from __future__ import annotations

import time as _time
from datetime import UTC
from typing import TYPE_CHECKING, Any

from khora.telemetry import metric_counter

if TYPE_CHECKING:
    from datetime import datetime

    from khora.core.models import Chunk, Entity, Relationship
    from khora.extraction.skills import ExpertiseConfig


def _parse_valid_date(value: Any) -> datetime | None:
    """Parse an LLM-supplied real-world date into an aware datetime, or None.

    The extractor emits temporal bounds as ISO date / datetime strings (or a
    descriptive phrase, or null). This is a real-world (valid-time) signal, so
    it must never fall back to a khora-ops value (chunk.created_at / now()).
    Unparseable values return None so the caller can pick a same-axis floor.
    """
    from datetime import datetime as _dt

    if value is None:
        return None
    if isinstance(value, _dt):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            dt = _dt.fromisoformat(text.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
    return None


# #889: count chunks where LLM extraction returned an error metadata
# (truncated response, parse failure, retry exhaustion). The counter is
# bumped once per failed chunk. Caller-side ADR-001 ``degradations``
# entries carry the per-chunk reason; this metric is the aggregate
# operator-facing signal. NO ``namespace_id`` label - cardinality rule.
_EXTRACTION_ERRORS_COUNTER = metric_counter(
    "khora.extraction.errors_total",
    description=("LLM extraction failures per chunk (truncated, parse error, retry exhausted). Labels: reason."),
)


async def extract_entities(
    chunks: list[Chunk],
    *,
    skill_name: str = "general_entities",
    expertise: ExpertiseConfig | str | None = None,
    model: str = "gpt-4o-mini",
    max_concurrent: int = 10,
    context: dict[str, Any] | None = None,
    timeout: int = 60,
    max_retries: int = 3,
    retry_wait: float = 2.0,
    extraction_batch_size: int = 10,
    max_tokens: int | None = None,
    entity_types: list[str],
    relationship_types: list[str],
    store_events: bool = True,
    selective_extraction: bool = True,
    extraction_importance_ratio: float = 0.7,
    extraction_min_importance: float = 0.2,
    shared_extractor: Any | None = None,
    out_diagnostics: dict[str, Any] | None = None,
) -> tuple[list[Entity], list[Relationship]]:
    """Extract entities and relationships from chunks.

    Uses batch extraction for parallel processing of multiple chunks.
    Supports both legacy skills and new expertise configurations.

    When ``selective_extraction`` is enabled, chunks are scored by importance
    and only the top fraction (controlled by ``extraction_importance_ratio``)
    are sent to LLM extraction.  The remaining chunks get lightweight
    rule-based co-occurrence edges, reducing LLM cost significantly.

    Args:
        chunks: Chunks to extract from
        skill_name: Extraction skill to use (legacy, ignored if expertise provided)
        expertise: ExpertiseConfig, expertise name string, or file path
        model: LLM model for extraction
        max_concurrent: Maximum concurrent extractions
        context: Optional context dict for prompt template rendering
        timeout: Request timeout in seconds
        max_retries: Maximum retries on failure
        retry_wait: Base wait time for exponential backoff between retries
        extraction_batch_size: Maximum texts per extraction batch
        max_tokens: Maximum tokens for LLM response
        entity_types: Required entity types to extract
        relationship_types: Required relationship types to extract
        store_events: Convert extracted events to EVENT entities with PARTICIPATED_IN relationships
        selective_extraction: Enable importance-based selective extraction
        extraction_importance_ratio: Fraction of chunks to send to LLM (top-K by score)
        extraction_min_importance: Minimum importance score threshold
        shared_extractor: Optional pre-initialized LLMEntityExtractor to reuse
            across documents (shares semaphore for cross-document concurrency control)
        out_diagnostics: Optional dict the function populates with ADR-001
            failure-observability fields when one or more chunks return an
            ``ExtractionResult`` with an ``error`` metadata key. Keys written:

            - ``extraction_errors`` (int): count of chunks where LLM
              extraction failed (truncated response, parse error, request
              error, ...). Always written when the dict is supplied.
            - ``llm_chunks`` (int): count of chunks routed to the LLM (the
              denominator for ``extraction_errors``).
            - ``degradations`` (list[Degradation]): one entry per failed
              chunk, ``component="extraction.llm"``, ``reason`` taken from
              ``result.metadata["error"]`` (truncated where helpful).

            Caller is expected to forward these into a result-level
            ``metadata`` dict (e.g. ``RememberResult.metadata``). #889.

    Returns:
        Tuple of (entities, relationships)
    """
    from khora.core.models import Entity, Relationship
    from khora.extraction.extractors import LLMEntityExtractor
    from khora.extraction.skills import ExpertiseConfig
    from khora.extraction.skills.registry import get_default_registry

    if not chunks:
        return [], []

    from loguru import logger

    from khora._accel import normalize_entity_name

    # --- Selective extraction: split chunks by importance ---
    lightweight_chunks: list[Chunk] = []
    llm_chunks = chunks

    if selective_extraction and len(chunks) > 1:
        from khora.extraction.importance import ChunkImportanceScorer

        scorer = ChunkImportanceScorer()
        llm_chunks, lightweight_chunks = scorer.select_for_extraction(
            chunks,
            ratio=extraction_importance_ratio,
            min_score=extraction_min_importance,
        )
        logger.debug(
            f"Selective extraction: {len(llm_chunks)} chunks to LLM, "
            f"{len(lightweight_chunks)} chunks to lightweight edges "
            f"(ratio={extraction_importance_ratio}, min_score={extraction_min_importance})"
        )

    # Resolve expertise configuration
    resolved_expertise: ExpertiseConfig | None = None
    if expertise is not None:
        if isinstance(expertise, ExpertiseConfig):
            resolved_expertise = expertise
        elif isinstance(expertise, str):
            # Load from string (file path or builtin name)
            from khora.extraction.skills import load_expertise

            try:
                resolved_expertise = load_expertise(expertise)
            except Exception as e:
                # Fall back to registry lookup
                logger.debug(f"load_expertise('{expertise}') failed, falling back to registry: {e}")
                registry = get_default_registry()
                resolved_expertise = registry.get_expertise(expertise)

    # Get legacy skill for backward compatibility
    registry = get_default_registry()
    skill = registry.get_or_default(skill_name)

    # If expertise provided, use its confidence thresholds
    if resolved_expertise:
        min_entity_confidence = resolved_expertise.confidence.min_entity
        min_relationship_confidence = resolved_expertise.confidence.min_relationship
    else:
        min_entity_confidence = skill.min_entity_confidence
        min_relationship_confidence = skill.min_relationship_confidence

    # Reuse shared extractor if provided (shares semaphore across documents),
    # otherwise create a new one per call.
    if shared_extractor is not None:
        extractor = shared_extractor
    else:
        extractor_kwargs = dict(
            model=model,
            max_concurrent=max_concurrent,
            timeout=timeout,
            max_retries=max_retries,
            retry_wait=retry_wait,
        )
        if max_tokens is not None:
            extractor_kwargs["max_tokens"] = max_tokens
        extractor = LLMEntityExtractor(**extractor_kwargs)

    # Extract from LLM-selected chunks using adaptive token-budget-based batching
    # Groups chunks into batches that fit within the model's input token budget,
    # reducing API round-trips by up to 5x while avoiding context overflow
    texts = [chunk.content for chunk in llm_chunks]

    # Use adaptive batching based on token budget (auto-calculated from max_tokens)
    # batch_size=5 is the max texts per batch; actual batching respects token limits
    _llm_t0 = _time.perf_counter()
    results = await extractor.extract_multi(
        texts,
        entity_types=entity_types,
        relationship_types=relationship_types,
        expertise=resolved_expertise,
        context=context,
        batch_size=extraction_batch_size,
        max_input_tokens=None,  # Auto-calculate from model
    )
    _llm_extraction_ms = (_time.perf_counter() - _llm_t0) * 1000

    # Process results
    all_entities: dict[str, Entity] = {}  # name -> entity (for dedup)
    all_relationships: list[Relationship] = []
    events_converted = 0
    # #889: Track per-chunk extraction errors so callers can surface them
    # on RememberResult.metadata. ``LLMEntityExtractor`` returns an empty
    # ``ExtractionResult`` with a populated ``metadata["error"]`` on
    # truncated responses and on retry exhaustion; before this PR the
    # caller could not tell extraction failed from "the chunk had no
    # entities". We accumulate a count + ADR-001 ``degradations`` list
    # when ``out_diagnostics`` is supplied.
    extraction_error_count = 0
    extraction_degradations: list[dict[str, Any]] = []

    for chunk, result in zip(llm_chunks, results):
        # #889: detect failed extraction (truncated / parse error / request
        # error) before processing entities so we can surface it on the
        # result. The error metadata is set by ``LLMEntityExtractor`` -
        # see ``llm.py`` lines 859 (truncation) and 885 (post-retry).
        result_meta = getattr(result, "metadata", None) or {}
        if isinstance(result_meta, dict) and "error" in result_meta:
            extraction_error_count += 1
            if out_diagnostics is not None:
                error_value = result_meta.get("error")
                error_text = str(error_value) if error_value is not None else "unknown"
                # Cap detail length - some raw_response errors carry a
                # full HTTP body. Bounded text keeps the diagnostic
                # log-friendly.
                detail = error_text if len(error_text) <= 200 else error_text[:197] + "..."
                extraction_degradations.append(
                    {
                        "component": "extraction.llm",
                        "reason": "extraction_failed",
                        "detail": detail,
                        "exception": None,
                    }
                )

        # Process entities
        for extracted in result.entities:
            if extracted.confidence < min_entity_confidence:
                continue

            # Deduplicate by normalized name
            key = f"{normalize_entity_name(extracted.name)}:{extracted.entity_type}"
            if key in all_entities:
                # Merge into existing
                existing = all_entities[key]
                existing.mention_count += 1
                if chunk.document_id not in existing.source_document_ids:
                    existing.source_document_ids.append(chunk.document_id)
                if chunk.id not in existing.source_chunk_ids:
                    existing.source_chunk_ids.append(chunk.id)
                # Lower valid_from to the earliest real-world date (#1225).
                # Resolve the candidate from the same-axis signals the
                # create branch uses (LLM temporal, then the
                # chunk.source_timestamp floor), never chunk.created_at - a
                # khora-ops value that would replace a real-world date with
                # ingest time.
                merge_temporal = extracted.temporal
                merge_valid_from = _parse_valid_date(merge_temporal.valid_from) if merge_temporal else None
                if merge_valid_from is None:
                    merge_valid_from = chunk.source_timestamp
                if existing.valid_from and merge_valid_from is not None and merge_valid_from < existing.valid_from:
                    existing.valid_from = merge_valid_from
            else:
                # Create new entity — preserve original type string from LLM
                entity_type = extracted.entity_type or "CONCEPT"

                # Real-world validity bounds (#994): prefer the LLM-supplied
                # temporal signal, fall back to the same-axis chunk
                # source_timestamp floor, then None. Never chunk.created_at
                # (a khora-ops value) - that would cross the time axis.
                temporal = extracted.temporal
                valid_from = _parse_valid_date(temporal.valid_from) if temporal else None
                valid_until = _parse_valid_date(temporal.valid_until) if temporal else None
                if valid_from is None:
                    valid_from = chunk.source_timestamp

                entity = Entity(
                    namespace_id=chunk.namespace_id,
                    name=normalize_entity_name(extracted.name),
                    entity_type=entity_type,
                    description=extracted.description,
                    attributes=extracted.attributes,
                    source_document_ids=[chunk.document_id],
                    source_chunk_ids=[chunk.id],
                    confidence=extracted.confidence,
                    valid_from=valid_from,
                    valid_until=valid_until,
                )
                all_entities[key] = entity

        # Build name→key lookup for O(1) relationship resolution
        entity_name_to_key: dict[str, str] = {}
        for key in all_entities:
            name_part = key.split(":")[0]
            entity_name_to_key[name_part] = key

        # Process relationships
        for extracted_rel in result.relationships:
            if extracted_rel.confidence < min_relationship_confidence:
                continue

            # Preserve original type string from LLM
            rel_type = extracted_rel.relationship_type or "RELATES_TO"

            # Find source and target entities (normalize names to match dedup keys)
            source_key = entity_name_to_key.get(normalize_entity_name(extracted_rel.source_entity))
            target_key = entity_name_to_key.get(normalize_entity_name(extracted_rel.target_entity))

            if source_key and target_key:
                # Real-world validity bounds (#994): LLM temporal -> same-axis
                # chunk source_timestamp floor -> None. Never chunk.created_at.
                rel_temporal = extracted_rel.temporal
                rel_valid_from = _parse_valid_date(rel_temporal.valid_from) if rel_temporal else None
                rel_valid_until = _parse_valid_date(rel_temporal.valid_until) if rel_temporal else None
                if rel_valid_from is None:
                    rel_valid_from = chunk.source_timestamp

                relationship = Relationship(
                    namespace_id=chunk.namespace_id,
                    source_entity_id=all_entities[source_key].id,
                    target_entity_id=all_entities[target_key].id,
                    relationship_type=rel_type,
                    description=extracted_rel.description,
                    properties=extracted_rel.properties,
                    source_document_ids=[chunk.document_id],
                    source_chunk_ids=[chunk.id],
                    confidence=extracted_rel.confidence,
                    valid_from=rel_valid_from,
                    valid_until=rel_valid_until,
                )
                all_relationships.append(relationship)

        # Convert extracted events to EVENT entities + PARTICIPATED_IN relationships
        if store_events and result.events:
            for event in result.events:
                if event.confidence < min_entity_confidence:
                    continue

                # Build a deterministic name from description (truncated for readability)
                event_name = event.description[:120].strip()
                if not event_name:
                    continue
                normalized_name = normalize_entity_name(event_name)
                event_key = f"{normalized_name}:EVENT"

                # Build attributes from event fields
                event_attrs: dict[str, Any] = {}
                if event.event_type:
                    event_attrs["event_type"] = event.event_type
                if event.occurred_at:
                    event_attrs["occurred_at"] = event.occurred_at
                if event.participants:
                    event_attrs["participants"] = event.participants

                # Real-world event time (#994): the LLM-supplied occurred_at is
                # the EVENT's valid_from. Fall back to the same-axis chunk
                # source_timestamp floor, then None. Never chunk.created_at.
                event_valid_from = _parse_valid_date(event.occurred_at)
                if event_valid_from is None:
                    event_valid_from = chunk.source_timestamp

                if event_key in all_entities:
                    # Merge into existing event entity
                    existing_event = all_entities[event_key]
                    existing_event.mention_count += 1
                    if chunk.document_id not in existing_event.source_document_ids:
                        existing_event.source_document_ids.append(chunk.document_id)
                    if chunk.id not in existing_event.source_chunk_ids:
                        existing_event.source_chunk_ids.append(chunk.id)
                else:
                    event_entity = Entity(
                        namespace_id=chunk.namespace_id,
                        name=normalized_name,
                        entity_type="EVENT",
                        description=event.description,
                        attributes=event_attrs,
                        source_document_ids=[chunk.document_id],
                        source_chunk_ids=[chunk.id],
                        confidence=event.confidence,
                        valid_from=event_valid_from,
                    )
                    all_entities[event_key] = event_entity
                    events_converted += 1

                # Rebuild name→key lookup after adding event entity
                entity_name_to_key[normalized_name] = event_key

                # Create PARTICIPATED_IN relationships from participant entities to the event
                for participant_name in event.participants:
                    participant_key = entity_name_to_key.get(normalize_entity_name(participant_name))
                    if participant_key:
                        rel = Relationship(
                            namespace_id=chunk.namespace_id,
                            source_entity_id=all_entities[participant_key].id,
                            target_entity_id=all_entities[event_key].id,
                            relationship_type="PARTICIPATED_IN",
                            description=f"Participated in: {event.description[:80]}",
                            source_document_ids=[chunk.document_id],
                            source_chunk_ids=[chunk.id],
                            confidence=event.confidence,
                            # Participation shares the event's real-world time (#994).
                            valid_from=event_valid_from,
                        )
                        all_relationships.append(rel)

    if events_converted > 0:
        logger.debug(f"Converted {events_converted} extracted events to EVENT entities")

    # --- Apply STATE_CHANGE entities to affected entities ---
    # When a STATE_CHANGE entity is extracted (e.g. "Alice switched from piano to guitar"),
    # propagate the new_state to the affected entity's attributes.  This triggers
    # bi-temporal version creation during Neo4j upsert (SUPERSEDES edge + EntityVersion),
    # which is required for counterfactual reasoning to work.
    state_changes_applied = 0
    for key, entity in list(all_entities.items()):
        if entity.entity_type != "STATE_CHANGE":
            continue
        attrs = entity.attributes
        affected_name = attrs.get("entity_affected", "")
        new_state = attrs.get("new_state", "")
        attr_changed = attrs.get("attribute_changed", "")
        if not affected_name or not new_state:
            continue

        # Find the affected entity
        affected_norm = normalize_entity_name(affected_name)
        affected_key = entity_name_to_key.get(affected_norm)
        if not affected_key or affected_key not in all_entities:
            continue

        affected_entity = all_entities[affected_key]
        if attr_changed:
            affected_entity.attributes[attr_changed] = new_state
        # Set valid_from on the STATE_CHANGE to the transition date if available
        transition_date = attrs.get("transition_date")
        if transition_date:
            try:
                from datetime import datetime as _dt

                parsed = _dt.fromisoformat(transition_date.replace("Z", "+00:00"))
                entity.valid_from = parsed
            except (ValueError, TypeError):
                pass

        # Create INVOLVES relationship from affected entity to the STATE_CHANGE
        rel = Relationship(
            namespace_id=entity.namespace_id,
            source_entity_id=affected_entity.id,
            target_entity_id=entity.id,
            relationship_type="INVOLVES",
            description=f"State changed: {attrs.get('previous_state', '?')} → {new_state}",
            source_document_ids=entity.source_document_ids[:],
            source_chunk_ids=entity.source_chunk_ids[:],
            confidence=entity.confidence,
            valid_from=entity.valid_from,
        )
        all_relationships.append(rel)
        state_changes_applied += 1

    if state_changes_applied > 0:
        logger.debug(f"Applied {state_changes_applied} STATE_CHANGE entities to affected entities")

    # --- Process lightweight chunks (selective extraction) ---
    if lightweight_chunks:
        from khora.extraction.importance import extract_lightweight_edges

        lightweight_edge_count = 0
        for chunk in lightweight_chunks:
            edges = extract_lightweight_edges(chunk)
            for entity1_name, rel_type, entity2_name in edges:
                # Create or reuse entities for co-occurrence edges
                norm1 = normalize_entity_name(entity1_name)
                norm2 = normalize_entity_name(entity2_name)
                key1 = f"{norm1}:CONCEPT"
                key2 = f"{norm2}:CONCEPT"

                for norm_name, key, original_name in [(norm1, key1, entity1_name), (norm2, key2, entity2_name)]:
                    if key in all_entities:
                        existing = all_entities[key]
                        existing.mention_count += 1
                        if chunk.document_id not in existing.source_document_ids:
                            existing.source_document_ids.append(chunk.document_id)
                        if chunk.id not in existing.source_chunk_ids:
                            existing.source_chunk_ids.append(chunk.id)
                    else:
                        entity = Entity(
                            namespace_id=chunk.namespace_id,
                            name=norm_name,
                            entity_type="CONCEPT",
                            description="",
                            source_document_ids=[chunk.document_id],
                            source_chunk_ids=[chunk.id],
                            confidence=0.5,  # Lower confidence for rule-based extraction
                            # No LLM temporal signal on the lightweight path; use the
                            # same-axis source_timestamp floor, never created_at (#994).
                            valid_from=chunk.source_timestamp,
                        )
                        all_entities[key] = entity

                # Create CO_OCCURS_WITH relationship
                if key1 != key2:
                    relationship = Relationship(
                        namespace_id=chunk.namespace_id,
                        source_entity_id=all_entities[key1].id,
                        target_entity_id=all_entities[key2].id,
                        relationship_type=rel_type,
                        description="Co-occurs in chunk",
                        properties={"extraction_method": "lightweight"},
                        source_document_ids=[chunk.document_id],
                        source_chunk_ids=[chunk.id],
                        confidence=0.4,  # Lower confidence for rule-based edges
                        # Same-axis source_timestamp floor, never created_at (#994).
                        valid_from=chunk.source_timestamp,
                    )
                    all_relationships.append(relationship)
                    lightweight_edge_count += 1

        if lightweight_edge_count > 0:
            logger.debug(
                f"Created {lightweight_edge_count} lightweight co-occurrence edges "
                f"from {len(lightweight_chunks)} skipped chunks"
            )

    # #889: bump the extraction-errors counter and surface diagnostics
    # on the caller-supplied dict before the span closes so the span
    # attributes reflect the same view.
    if extraction_error_count > 0:
        _EXTRACTION_ERRORS_COUNTER.add(
            extraction_error_count,
            attributes={"reason": "extraction_failed"},
        )
        logger.warning(
            f"LLM extraction failed on {extraction_error_count}/{len(llm_chunks)} chunks "
            f"(see degradations metadata for per-chunk reason)"
        )
    if out_diagnostics is not None:
        out_diagnostics["extraction_errors"] = extraction_error_count
        out_diagnostics["llm_chunks"] = len(llm_chunks)
        if extraction_degradations:
            existing = out_diagnostics.setdefault("degradations", [])
            if isinstance(existing, list):
                existing.extend(extraction_degradations)

    from khora.telemetry import trace_span

    with trace_span(
        "khora.extraction.extract_entities",
        total_chunks=len(chunks),
        llm_chunks=len(llm_chunks),
        lightweight_chunks=len(lightweight_chunks),
        selective_extraction=selective_extraction,
        extraction_importance_ratio=extraction_importance_ratio,
        llm_extraction_ms=round(_llm_extraction_ms, 2),
        entities_extracted=len(all_entities),
        relationships_extracted=len(all_relationships),
        events_converted=events_converted,
        state_changes_applied=state_changes_applied,
        extraction_errors=extraction_error_count,
    ):
        pass

    return list(all_entities.values()), all_relationships
