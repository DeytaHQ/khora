"""Ontology output: serialization and validation."""

from __future__ import annotations

from .serializer import serialize_ontology
from .validator import validate_ontology

__all__ = ["serialize_ontology", "validate_ontology"]
