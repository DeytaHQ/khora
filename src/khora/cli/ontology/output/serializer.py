"""Serialize ExpertiseConfig to clean YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from khora.extraction.skills.base import ExpertiseConfig


def _clean_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Remove None values and empty lists/dicts for cleaner YAML output."""
    cleaned: dict[str, Any] = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, list) and len(v) == 0:
            continue
        if isinstance(v, dict) and len(v) == 0:
            continue
        if isinstance(v, dict):
            v = _clean_dict(v)
        if isinstance(v, list):
            v = [_clean_dict(item) if isinstance(item, dict) else item for item in v]
        cleaned[k] = v
    return cleaned


def serialize_ontology(config: ExpertiseConfig) -> str:
    """Serialize an ExpertiseConfig to a clean YAML string.

    Removes empty/None fields and uses block style for readability.
    """
    data = _clean_dict(config.to_dict())
    return yaml.dump(
        data,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
        width=100,
    )


def write_ontology(config: ExpertiseConfig, path: Path) -> None:
    """Write an ExpertiseConfig to a YAML file."""
    content = serialize_ontology(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
