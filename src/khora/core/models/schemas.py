"""Pydantic models for entity attribute schema validation.

Provides validated attribute schemas for standard entity types.
The attribute schemas enforce required/optional fields defined
in EntityTypeConfig YAML configurations at storage time.

The registry is extensible — downstream projects can register
additional schemas via ``register_attribute_schema()``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError


class PersonAttributes(BaseModel):
    """Validated attributes for PERSON entities."""

    name: str
    title: str | None = None
    role: str | None = None
    email: str | None = None
    organization: str | None = None
    department: str | None = None
    location: str | None = None


class OrganizationAttributes(BaseModel):
    """Validated attributes for ORGANIZATION entities."""

    name: str
    type: str | None = None  # "company", "nonprofit", "government"
    industry: str | None = None
    website: str | None = None
    founded: str | None = None
    location: str | None = None


class LocationAttributes(BaseModel):
    """Validated attributes for LOCATION entities."""

    name: str
    type: str | None = None  # "city", "country", "region", "address"
    country: str | None = None
    coordinates: str | None = None
    address: str | None = None


class ConceptAttributes(BaseModel):
    """Validated attributes for CONCEPT entities."""

    name: str
    category: str | None = None
    definition: str | None = None
    related_concepts: list[str] | None = None


class EventAttributes(BaseModel):
    """Validated attributes for EVENT entities."""

    name: str
    date: str | None = None
    location: str | None = None
    participants: list[str] | None = None
    type: str | None = None


class TechnologyAttributes(BaseModel):
    """Validated attributes for TECHNOLOGY entities."""

    name: str
    type: str | None = None  # "language", "framework", "platform", "tool"
    version: str | None = None
    vendor: str | None = None
    category: str | None = None


class ProductAttributes(BaseModel):
    """Validated attributes for PRODUCT entities."""

    name: str
    vendor: str | None = None
    category: str | None = None
    price: str | None = None
    version: str | None = None


class DateAttributes(BaseModel):
    """Validated attributes for DATE entities."""

    value: str
    type: str | None = None
    precision: str | None = None


class StateChangeAttributes(BaseModel):
    """Validated attributes for STATE_CHANGE entities.

    Captures entity state transitions with explicit pre/post states and
    transition dates.  Used by bi-temporal versioning to create SUPERSEDES
    edges and by counterfactual reasoning to verify state validity at a
    given point in time.
    """

    entity_affected: str
    previous_state: str
    new_state: str
    transition_date: str | None = None
    attribute_changed: str | None = None  # e.g. "job_title", "location", "instrument"
    reason: str | None = None


# Registry mapping entity type names to Pydantic models.
# Extensible via register_attribute_schema().
ATTRIBUTE_SCHEMAS: dict[str, type[BaseModel]] = {
    "PERSON": PersonAttributes,
    "ORGANIZATION": OrganizationAttributes,
    "LOCATION": LocationAttributes,
    "CONCEPT": ConceptAttributes,
    "EVENT": EventAttributes,
    "TECHNOLOGY": TechnologyAttributes,
    "PRODUCT": ProductAttributes,
    "DATE": DateAttributes,
    "STATE_CHANGE": StateChangeAttributes,
}


def register_attribute_schema(
    entity_type: str,
    schema: type[BaseModel],
    *,
    aliases: list[str] | None = None,
) -> None:
    """Register a Pydantic attribute schema for an entity type.

    Downstream projects use this to add domain-specific schemas without
    modifying khora core.

    Args:
        entity_type: Canonical entity type name (e.g. "TICKET")
        schema: Pydantic BaseModel subclass for attribute validation
        aliases: Optional alternative type names that share the same schema
    """
    ATTRIBUTE_SCHEMAS[entity_type.upper()] = schema
    for alias in aliases or []:
        ATTRIBUTE_SCHEMAS[alias.upper()] = schema


def validate_attributes(entity_type: str, attributes: dict[str, Any]) -> dict[str, Any]:
    """Validate and coerce attributes using the registered schema.

    Falls back to passthrough if no schema is registered for the entity type
    or if validation fails (graceful degradation).

    Args:
        entity_type: The entity type name (e.g. "PERSON", "TICKET")
        attributes: Raw attributes dict to validate

    Returns:
        Cleaned attributes dict with None values excluded
    """
    schema = ATTRIBUTE_SCHEMAS.get(entity_type.upper())
    if not schema:
        return attributes

    try:
        validated = schema.model_validate(attributes)
        return validated.model_dump(exclude_none=True)
    except ValidationError:
        # Graceful degradation: return original attributes if validation fails
        return attributes
