"""schema_drift is reachable via the public ``kb.dream()`` API (#1036).

The vectorcypher ``schema_drift`` op was gated ``if expertise is not None``
in ``engines/registry.py`` while no public surface forwarded an
``ExpertiseConfig`` - ``Khora.dream()`` / ``dream.api.dream()`` had no
``expertise`` param. So every ``kb.dream(ops=[VECTORCYPHER_SCHEMA_DRIFT_REPORT])``
call dropped the op, and the else branch recorded no ``skip_reasons`` entry,
making the drop silent (an ADR-001 observability gap).

These run a real dry-run dream on the embedded stack:

- (a) ``kb.dream(..., expertise=cfg)`` produces a schema-drift op.
- (b) omitting ``expertise`` records a ``skip_reasons`` entry
  (``reason="op_requires_expertise"``) rather than silently dropping it.

No Docker / Postgres / network - mock LLM + embedded sqlite_lance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples._helpers import embedded_khora, install_mock_llm  # noqa: E402
from khora.dream.config import DreamConfig  # noqa: E402
from khora.dream.plan import OpKind  # noqa: E402
from khora.extraction.skills.base import ExpertiseConfig  # noqa: E402

pytestmark = pytest.mark.embedded

DRIFT = OpKind.VECTORCYPHER_SCHEMA_DRIFT_REPORT


async def _remember(kb, namespace_id):
    return await kb.remember(
        "OpenAI was founded by Sam Altman.",
        namespace=namespace_id,
        entity_types=[],
        relationship_types=[],
    )


async def test_schema_drift_reachable_with_expertise() -> None:
    """kb.dream(ops=[schema_drift], expertise=cfg) produces a schema-drift op."""
    install_mock_llm(dim=8)
    async with embedded_khora(embedding_dimension=8, engine="vectorcypher") as kb:
        ns = await kb.create_namespace()
        await _remember(kb, ns.namespace_id)

        result = await kb.dream(
            ns.namespace_id,
            mode="dry-run",
            ops=[DRIFT],
            config=DreamConfig(enabled=True),
            expertise=ExpertiseConfig(name="repro"),
        )

        op_types = [o.op_type for o in result.ops]
        assert any(str(DRIFT) == str(t) for t in op_types), (
            f"schema_drift op absent from result.ops={op_types!r} when expertise was supplied"
        )


async def test_schema_drift_omitted_expertise_records_skip_reason() -> None:
    """Omitting expertise records a skip_reason rather than silently dropping."""
    install_mock_llm(dim=8)
    async with embedded_khora(embedding_dimension=8, engine="vectorcypher") as kb:
        ns = await kb.create_namespace()
        await _remember(kb, ns.namespace_id)

        result = await kb.dream(
            ns.namespace_id,
            mode="dry-run",
            ops=[DRIFT],
            config=DreamConfig(enabled=True),
        )

        op_types = [o.op_type for o in result.ops]
        assert not any(str(DRIFT) == str(t) for t in op_types), "schema_drift should not run without an ExpertiseConfig"

        skip = result.metadata.get("skip_reasons", [])
        matching = [s for s in skip if str(DRIFT) == str(s.get("op_kind"))]
        assert matching, (
            f"schema_drift requested but no skip_reasons entry recorded - silent drop. skip_reasons={skip!r}"
        )
        assert matching[0].get("reason") == "op_requires_expertise"
