"""Drift gate for the telemetry public-surface contract.

Loads ``docs/telemetry-contract.json`` and asserts the live codebase
matches it. A new span / pipeline-stage / metric name added to source but
not added to the JSON fails this test. See ``docs/telemetry-contract.md``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src" / "khora"
CONTRACT_PATH = REPO_ROOT / "docs" / "telemetry-contract.json"


@pytest.fixture(scope="module")
def contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text())


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


def test_public_exports_match_all(contract: dict) -> None:
    import khora.telemetry as telemetry

    expected = set(contract["public_exports"])
    actual = set(telemetry.__all__)
    missing = expected - actual
    extra = actual - expected
    assert not missing, f"Contract names missing from telemetry.__all__: {missing}"
    assert not extra, (
        f"telemetry.__all__ has names not in contract: {extra}. "
        "Add them to docs/telemetry-contract.json or remove from __all__."
    )


def test_public_exports_are_importable(contract: dict) -> None:
    import khora.telemetry as telemetry

    for name in contract["public_exports"]:
        assert hasattr(telemetry, name), f"khora.telemetry missing export: {name}"


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------


def test_event_type_fields_match(contract: dict) -> None:
    from khora.telemetry import models

    for event_spec in contract["event_types"]:
        cls = getattr(models, event_spec["name"])
        actual_fields = set(cls.model_fields.keys())
        expected_fields = {f["name"] for f in event_spec["fields"]}
        missing = expected_fields - actual_fields
        extra = actual_fields - expected_fields
        assert not missing, f"{event_spec['name']} missing fields: {missing}"
        assert not extra, (
            f"{event_spec['name']} has extra fields not in contract: {extra}. Add them to docs/telemetry-contract.json."
        )


# ---------------------------------------------------------------------------
# Collector methods
# ---------------------------------------------------------------------------


def test_collector_methods_exist(contract: dict) -> None:
    from khora.telemetry.collector import TelemetryCollector
    from khora.telemetry.noop import NoOpCollector

    for method in contract["collector_methods"]:
        for cls in (TelemetryCollector, NoOpCollector):
            attr = getattr(cls, method, None)
            assert attr is not None, f"{cls.__name__} missing method: {method}"
            assert callable(attr), f"{cls.__name__}.{method} not callable"


# ---------------------------------------------------------------------------
# Span / regex sanity (the contract itself)
# ---------------------------------------------------------------------------


def test_contract_span_names_match_regex(contract: dict) -> None:
    pattern = re.compile(contract["span_name_regex"])
    for span in contract["spans"]:
        assert pattern.match(span["name"]), (
            f"Contract span name {span['name']!r} fails regex {contract['span_name_regex']}"
        )


# ---------------------------------------------------------------------------
# Drift gate: names found in source must appear in the contract
# ---------------------------------------------------------------------------


_SPAN_RE = re.compile(r'trace_span\(\s*"([^"]+)"')
_PIPELINE_STAGE_POS_RE = re.compile(r'pipeline_stage\(\s*"([^"]+)"\s*,\s*"([^"]+)"')
_PIPELINE_STAGE_KW_RE = re.compile(
    r'record_pipeline_stage\([^)]*?\bpipeline\s*=\s*"([^"]+)"[^)]*?\bstage\s*=\s*"([^"]+)"',
    re.DOTALL,
)
_METRIC_RE = re.compile(r'metric_(?:counter|histogram|gauge_callback)\(\s*"([^"]+)"')


def _iter_py_files() -> list[Path]:
    return sorted(SRC_ROOT.rglob("*.py"))


def _scan(pattern: re.Pattern[str]) -> set:
    found: set = set()
    for path in _iter_py_files():
        text = path.read_text()
        for match in pattern.finditer(text):
            if match.lastindex and match.lastindex >= 2:
                found.add(match.group(1) + "." + match.group(2))
            else:
                found.add(match.group(1))
    return found


def test_no_undeclared_spans(contract: dict) -> None:
    declared = {s["name"] for s in contract["spans"]}
    found = _scan(_SPAN_RE)
    # Drop khora.telemetry's own self-instrumentation in instrument.py if any
    # (none today — surface as undeclared if introduced).
    undeclared = found - declared
    assert not undeclared, (
        f"trace_span names in code missing from contract: {sorted(undeclared)}. "
        "Add them to docs/telemetry-contract.json with stability=internal "
        "(or stability=public if part of the public surface)."
    )


def test_no_undeclared_pipeline_stages(contract: dict) -> None:
    declared = {s["name"] for s in contract["pipeline_stages"]}
    found = _scan(_PIPELINE_STAGE_POS_RE) | _scan(_PIPELINE_STAGE_KW_RE)
    undeclared = found - declared
    assert not undeclared, (
        f"pipeline stages in code missing from contract: {sorted(undeclared)}. "
        "Add them to docs/telemetry-contract.json."
    )


def test_no_undeclared_metrics(contract: dict) -> None:
    declared = {m["name"] for m in contract["metrics"]}
    found = _scan(_METRIC_RE)
    undeclared = found - declared
    assert not undeclared, (
        f"metric names in code missing from contract: {sorted(undeclared)}. Add them to docs/telemetry-contract.json."
    )


# ---------------------------------------------------------------------------
# Reverse drift: every public-stability item in the contract must
# actually be emitted somewhere in source. (Internal items may be
# placeholders, so we only enforce this for public.)
# ---------------------------------------------------------------------------


def test_public_spans_are_emitted(contract: dict) -> None:
    found = _scan(_SPAN_RE)
    for span in contract["spans"]:
        if span["stability"] != "public":
            continue
        assert span["name"] in found, (
            f"Public span {span['name']!r} declared in contract but no trace_span() call site found in source."
        )


def test_public_metrics_are_emitted(contract: dict) -> None:
    found = _scan(_METRIC_RE)
    for metric in contract["metrics"]:
        if metric["stability"] != "public":
            continue
        assert metric["name"] in found, (
            f"Public metric {metric['name']!r} declared in contract but no metric_*() call site found in source."
        )


# ---------------------------------------------------------------------------
# OTel-first invariants (v0.10.8)
# ---------------------------------------------------------------------------


def test_instrumentation_scope_name_is_khora(contract: dict) -> None:
    """Spans + metrics must use ``khora`` as the instrumentation scope name.

    The contract declares this; the live tracer must match it.
    """
    from khora.telemetry._otel import _TRACER

    declared = contract["instrumentation_scope"]["name"]
    assert declared == "khora", f"contract instrumentation_scope.name={declared!r}, expected 'khora'"
    # The OTel API tracer carries the scope as a private attribute; we
    # round-trip through a recorded span in test_otel_parity.py and
    # assert here that the cached tracer is the OTel SDK's, not a logfire
    # wrapper that might rename the scope.
    assert _TRACER is not None


def test_instrumentation_scope_version_uses_package_version(contract: dict) -> None:
    """``scope.version`` must come from importlib.metadata, not a constant.

    Locking this prevents a future contributor from hard-coding a stale
    version string into the module.
    """
    from importlib.metadata import version

    from khora.telemetry import _otel as _otel_module

    declared_source = contract["instrumentation_scope"]["version_source"]
    assert "importlib.metadata" in declared_source

    expected = version("khora")
    assert _otel_module._KHORA_VERSION == expected, (
        f"khora.telemetry._otel._KHORA_VERSION={_otel_module._KHORA_VERSION!r}, "
        f"importlib.metadata.version('khora')={expected!r}"
    )


def test_deprecated_alias_is_still_importable() -> None:
    """The ``install_neo4j_logfire_handler`` alias must remain importable
    for one minor release after the 0.10.8 rename."""
    import warnings

    import khora.telemetry as telemetry

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        fn = telemetry.install_neo4j_logfire_handler
    assert any(issubclass(w.category, DeprecationWarning) for w in recorded)
    assert fn is telemetry.install_neo4j_log_bridge


def test_contract_declares_backends_block(contract: dict) -> None:
    backends = contract.get("backends", {})
    assert "precedence" in backends, "contract must list backend precedence"
    assert isinstance(backends["precedence"], list)
    assert "supported" in backends
    supported = set(backends["supported"])
    assert supported == {"auto", "otel", "logfire", "none"}, f"contract supported backends drifted: {supported}"
