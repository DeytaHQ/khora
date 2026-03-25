"""Session state persistence for ontology construction."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from khora.extraction.skills.base import ExpertiseConfig

from .inference.domain import DomainResult

SESSION_FILE = Path(".khora_ontology_session.json")


@dataclass
class OntologySession:
    """Persistent session state for ontology construction."""

    phase: str = "init"
    sources: list[str] = field(default_factory=list)
    model: str = "gpt-4o"
    budget_usd: float = 1.0
    output_path: str = "./ontology.yaml"
    extends_skill: str | None = None

    # Phase results
    domain: dict[str, Any] | None = None
    draft: dict[str, Any] | None = None

    # Tracking
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    tokens_used: int = 0
    cost_usd: float = 0.0

    def save(self, path: Path = SESSION_FILE) -> None:
        """Save session to JSON file."""
        path.write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = SESSION_FILE) -> OntologySession:
        """Load session from JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def get_domain_result(self) -> DomainResult | None:
        """Reconstruct DomainResult from stored dict."""
        if self.domain is None:
            return None
        return DomainResult.from_dict(self.domain)

    def get_expertise_config(self) -> ExpertiseConfig | None:
        """Reconstruct ExpertiseConfig from stored dict."""
        if self.draft is None:
            return None
        return ExpertiseConfig.from_dict(self.draft)
