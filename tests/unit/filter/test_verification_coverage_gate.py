"""Verification-coverage meta-gate: no filter/recall test is silently disabled.

A test that is green in CI but never actually *executed* — because no job
provisions the live backend it needs, or because a ``-m`` marker expression
de-selects it — is worse than no test: it advertises coverage it does not
deliver. This hermetic meta-test makes that failure mode impossible to ship
unnoticed.

What it proves
--------------
For every filter/recall test module in the repo, the gate proves there exists
at least one CI **job leg** that (1) *effectively selects* the module (its
pytest path args include the module AND the leg's effective ``-m`` marker
expression matches the module's markers) AND (2) *provisions every backend the
module requires* (Postgres / Neo4j / Weaviate / embedded). A module that no
such leg covers is "orphaned" and fails the gate — unless it is consciously
recorded in :data:`_KNOWN_UNRUN` with a public tracking reference, so the gap
is *visible and tracked* rather than silent.

How it works (hermetic — no DB, no network, no imports of khora)
----------------------------------------------------------------
* Parses every ``.github/workflows/*.yml`` with PyYAML (discovered by glob, not a
  hard-coded list, so a new workflow file — e.g. a dedicated slow/e2e live-DB lane
  — is picked up automatically), expanding each ``matrix.include`` entry into one
  runnable leg. Per-entry ``${{ matrix.<key> }}`` references are substituted into
  the run command before tokenizing, so a matrix-driven path/selector resolves to
  concrete args. A leg's pytest ``-k`` is modelled as a substring match against the
  module path, so an embedded lane that selects by a filename token via ``-k`` is
  claimed exactly — not over-claimed (``-k`` ignored) nor dropped.
* Reads the default ``-m`` marker filter from ``pyproject.toml``
  (``[tool.pytest.ini_options].addopts``) via ``tomllib`` — never hard-coded —
  so the gate tracks drift if the default changes.
* Walks every candidate test module with ``ast`` (never imports it) to read its
  ``pytestmark`` markers, its backend-gating ``skipif`` conditions, and its
  ``xfail``/``skip`` reasons.

The marker-replacement subtlety (the central correctness risk)
--------------------------------------------------------------
pytest's ``-m`` on the command line **replaces** the ``addopts`` ``-m`` (last
``-m`` wins; they are NOT ANDed). So a leg's effective marker expression is its
own ``-m`` if it has one, else the ``addopts`` default. Marker *exclusion*
de-claims a test even when its path is included: ``test-unit`` runs
``tests/e2e/`` but with ``-m "not slow and not filter_conformance"``, so a
``slow`` e2e lane whose path is included there is still NOT effectively
selected. The gate evaluates the boolean marker expression against each test's
marker set — it does not merely check path membership.

Falsifiability
--------------
A meta-test that can never fail is worthless. :func:`test_gate_is_falsifiable`
feeds the SAME core orphan-detection function a synthetic module/leg set where a
Postgres-needing module is selected only by an embedded-only leg, asserts the
gate reports it orphaned, then adds a provisioning leg and asserts it clears —
proving the gate has teeth independent of live repo state.

How to update this gate when you add a new filter test
------------------------------------------------------
* If your test needs a live backend, make sure a CI job both *selects its path*
  (with a matching ``-m``) and *provisions that backend*. The gate certifies it
  automatically — no edit here needed.
* If your test is a backend-gap acceptance test (``xfail``) or a deferred-work
  ``skip``, give its ``reason=`` a public tracking ref (``#NNNN`` or ``ADR-NNN``).
* Only add to :data:`_KNOWN_UNRUN` for a test that is *consciously* not yet run
  by any provisioning job, and always with a tracking ref. Remove the entry the
  moment a real job covers it — the gate flags stale entries (drift in both
  directions, same discipline as the filter-enforcement audit gate).

No DB, no infra — runs in the fast unit suite (the ``test-unit`` job).
"""

from __future__ import annotations

import ast
import re
import shlex
import tomllib
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.unit]

# Repo root: tests/unit/filter/<this file> → parents[3] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKFLOWS = _REPO_ROOT / ".github" / "workflows"
_CI_YML = _WORKFLOWS / "ci.yml"  # the one workflow that must always exist (sanity anchor)
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

# ``embedded`` is the sentinel for "needs no live service" (SQLite+LanceDB /
# SurrealDB memory): every leg provides it implicitly.
_EMBEDDED = "embedded"

# The two SIBLING meta-gates plus this file are meta-tests, not
# backend-exercising filter tests — exclude them from the enumeration universe.
# They DO carry filter markers, so the marker-authority safety net exempts this
# same set (a meta-gate is not a mislocated filter test).
_META_TEST_EXCLUSIONS = frozenset(
    {
        "tests/unit/filter/test_verification_coverage_gate.py",
        "tests/unit/filter/test_system_keys_coverage.py",
        "tests/recall/test_filter_enforcement_audit_gate.py",
    }
)

# Modules whose FILENAME collides with the universe heuristic
# (``filter``/``conformance``/``compile``) but which are NOT recall-filter tests.
# Documented here so the narrowing is auditable: the universe glob is scoped to
# ``tests/recall/`` + ``tests/unit/filter/`` dirs (not these), so they are never
# enumerated — but they carry NONE of the filter markers, so the marker-authority
# safety net does not flag them either. One reason each:
_DOCUMENTED_FILENAME_COLLISIONS: dict[str, str] = {
    "tests/unit/integrations/langgraph/test_conformance.py": "LangGraph adapter protocol conformance — not a recall-filter test.",
    "tests/unit/test_graph_protocol_conformance.py": "GraphBackend protocol conformance — not a recall-filter test.",
    "tests/unit/integrations/test_protocol_conformance.py": "Integration-adapter protocol conformance — not a recall-filter test.",
    "tests/unit/core/test_recall_abstention.py": "Recall abstention-signal scoring — not the recall-filter subsystem.",
    "tests/unit/core/test_recall_scoring.py": "Recall scoring math — not the recall-filter subsystem.",
    "tests/unit/db/test_migration_037_recall_response_format.py": "A DB migration test — not a recall-filter test.",
}

# Recall/filter modules whose FILENAME lacks the ``filter``/``conformance``/
# ``compile`` heuristic tokens, but which are squarely recall-filter tests and so
# belong in the universe. Named explicitly (rather than broadening the fuzzy
# filename match) to keep the universe precise and the inclusion intentional.
# Each entry carries the reason its filename misses the heuristic. The
# marker-authority safety net (:func:`test_marked_filter_tests_are_in_universe`)
# is the backstop that proves no marked filter test escapes this curated set.
_NAMED_EXTRAS = frozenset(
    {
        # Exercises embedded VC recall behavior (temporal recall / prefer_current /
        # traversal); its backend-gap xfails MUST be tracked (4b). Filename has no
        # filter/conformance/compile token.
        "tests/integration/matrix/test_vectorcypher_sqlite_lance.py",
        # Exercises embedded Skeleton metadata-filtered recall; carries a tracked
        # backend-gap xfail (4b). Filename has no filter/conformance/compile token.
        "tests/integration/matrix/test_skeleton_sqlite_lance.py",
        # Carries the ``filter_conformance`` marker (conformance seed-map roundtrip)
        # but its filename ("seed_map_roundtrip") has no heuristic token.
        "tests/integration/matrix/test_seed_map_roundtrip.py",
        # Carries the ``filter_enforcement`` marker (unit-level filter_ast pushdown
        # partial-failure threading); lives under tests/unit/engines/, outside the
        # curated dirs, and its filename has no heuristic token.
        "tests/unit/engines/vectorcypher/test_filter_pushdown_partial_failure.py",
    }
)

# Tripwire, not an exact count: 11 recall + 13 unit/filter + ~12
# integration/matrix/e2e filter modules ≈ 36 today. A glob/path refactor that
# collapses discovery would make this gate pass vacuously; the floor refuses to.
_MIN_EXPECTED_FILTER_MODULES = 20

# ``skipif`` reasons that are ENVIRONMENT GUARDS (the store is simply not up in
# this run) rather than tracked backend GAPS. These are exempt from the
# tracking-ref requirement — they describe how to run locally, not deferred work.
_ENV_GUARD_REASON_SUBSTRINGS = (
    "not reachable",
    "not installed",
    "set neo4j_integration_test",
    "start postgres",
    "make dev",
    # runtime feature-availability guards (the build/runtime simply lacks an
    # optional capability) — same class as "not installed", not deferred work.
    "lacks",
    "not available",
)

# A tracking reference is a public GitHub issue (#NNN+) or an ADR (ADR-NNN).
_TRACKING_REF = re.compile(r"#\d{3,}\b|ADR-\d+\b")

# ---------------------------------------------------------------------------
# Consciously-unrun tests: known gaps awaiting a job that provisions them.
#
# These are KNOWN-TRACKED gaps, not silent ones — that is the whole point of the
# gate. An entry belongs here only while a test is *consciously* not yet run by
# any provisioning job, and always with a tracking ref. REMOVE an entry — and
# clear its tracking issue — the moment a real job covers it; the gate flags a
# stale entry that a job now covers (drift in both directions), so this dict
# cannot quietly accumulate resolved gaps.
#
# Currently empty: the slow/e2e live-DB rowset lanes (graph / chronicle) that
# previously lived here are now provisioned and selected by the dedicated e2e
# workflow (`.github/workflows/e2e.yml`), so the gate certifies them directly.
# ---------------------------------------------------------------------------
_KNOWN_UNRUN: dict[str, str] = {}


# ===========================================================================
# pyproject: default -m marker filter (read, never hard-coded)
# ===========================================================================
def _default_marker_expr() -> str:
    """The ``-m`` expression baked into ``addopts`` — the fallback when a leg's
    pytest invocation omits its own ``-m``.

    pytest applies ``addopts`` before the command line, and a command-line
    ``-m`` REPLACES it. So this is only the effective expression for invocations
    that do not pass their own ``-m``.
    """
    data = tomllib.loads(_PYPROJECT.read_text())
    addopts = data["tool"]["pytest"]["ini_options"]["addopts"]
    # addopts is a list like [..., "-m", "not slow and not filter_conformance"].
    for i, tok in enumerate(addopts):
        if tok == "-m":
            return addopts[i + 1]
        if tok.startswith("-m"):  # "-mEXPR" glued form, defensive
            return tok[2:].strip()
    raise AssertionError("no -m found in [tool.pytest.ini_options].addopts — addopts shape changed")


# ===========================================================================
# Marker-expression evaluator (pytest -m grammar subset)
# ===========================================================================
_ALLOWED_EXPR_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.Name,
    ast.Load,
)


def _eval_bool_expr(expr: str, leaf: Callable[[str], bool]) -> bool:
    """Evaluate a pytest ``-m`` / ``-k`` boolean expression.

    Supports ``and`` / ``or`` / ``not`` / parentheses / bare names — the subset
    pytest uses for both ``-m`` (names tested against markers) and ``-k`` (names
    tested as substrings of the test id / path). Parses to an AST and walks it,
    evaluating each ``Name`` via ``leaf``. Any node outside the whitelist raises,
    so a malformed expression fails loudly rather than silently mis-evaluating.
    """
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_EXPR_NODES):
            raise AssertionError(f"unsupported boolean-expression node {type(node).__name__!r} in {expr!r}")

    def _ev(node: ast.AST) -> bool:
        if isinstance(node, ast.Expression):
            return _ev(node.body)
        if isinstance(node, ast.Name):
            return leaf(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not _ev(node.operand)
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(_ev(v) for v in node.values)
            return any(_ev(v) for v in node.values)
        raise AssertionError(f"unexpected node {type(node).__name__!r}")

    return _ev(tree)


def _eval_marker_expr(expr: str, markers: frozenset[str]) -> bool:
    """A leg's ``-m`` expression evaluated against a test's marker set."""
    return _eval_bool_expr(expr, lambda name: name in markers)


def _k_selects(k_expr: str, module_path: str) -> bool:
    """Model pytest ``-k`` as a substring match against the module path.

    An empty ``-k`` imposes no constraint. Otherwise each name in the ``-k``
    expression is true when it is a substring of the module path
    (case-insensitive), so ``-k`` NARROWS a path-included leg's coverage rather
    than being ignored — ignoring it would over-claim coverage for every module
    under the path. The slow/e2e lanes select by a filename token via ``-k``, so
    this is how such a leg legitimately claims exactly the modules it runs. A
    ``-k`` value pytest's grammar accepts but ``ast`` cannot parse falls back to a
    plain substring test of the whole token.
    """
    if not k_expr:
        return True
    path_lo = module_path.lower()
    try:
        return _eval_bool_expr(k_expr, lambda name: name.lower() in path_lo)
    except Exception:
        return k_expr.strip().lower() in path_lo


# ===========================================================================
# Workflow parsing → job legs
# ===========================================================================
class _Leg:
    """One runnable CI unit: a plain job, or one ``matrix.include`` expansion."""

    __slots__ = ("name", "backends", "invocations")

    def __init__(self, name: str, backends: frozenset[str], invocations: list[tuple[list[str], str, str]]) -> None:
        self.name = name
        # backends this leg PROVISIONS (always includes the embedded sentinel).
        self.backends = backends | {_EMBEDDED}
        # list of (path_args, effective_marker_expr, k_expr) — one per pytest
        # invocation. ``k_expr`` is the leg's ``-k`` (empty string = no -k).
        self.invocations = invocations

    def selects(self, module_path: str, markers: frozenset[str]) -> bool:
        """True if any invocation includes ``module_path`` AND its effective
        marker expression matches ``markers`` AND its ``-k`` (if any) substring-
        matches the path (path ∩ marker ∩ -k)."""
        for paths, marker_expr, k_expr in self.invocations:
            if (
                _path_included(module_path, paths)
                and _eval_marker_expr(marker_expr, markers)
                and _k_selects(k_expr, module_path)
            ):
                return True
        return False


def _path_included(module_path: str, path_args: list[str]) -> bool:
    """A directory arg includes everything under it; a file arg matches itself."""
    mp = module_path
    for arg in path_args:
        if arg.endswith(".py"):
            if mp == arg:
                return True
        else:
            prefix = arg if arg.endswith("/") else arg + "/"
            if mp.startswith(prefix):
                return True
    return False


def _image_to_backend(image: str) -> str | None:
    lo = image.lower()
    if "pgvector" in lo or "postgres" in lo:
        return "postgres"
    if "neo4j" in lo:
        return "neo4j"
    if "weaviate" in lo:
        return "weaviate"
    return None


def _conformance_backend_token(token: str) -> str:
    """Map a conformance ``matrix.backend`` token to the provisioning vocab."""
    if token == "cypher":
        return "neo4j"
    if token in ("postgres", "neo4j", "weaviate"):
        return token
    # python / chronicle / sqlite_lance / surrealdb run in-process.
    return _EMBEDDED


_MATRIX_TOKEN = re.compile(r"\$\{\{\s*matrix\.([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def _substitute_matrix(text: str, entry: dict) -> str:
    """Replace ``${{ matrix.<key> }}`` with the include-entry's value.

    Unresolved tokens (a key absent from this entry) are left intact so they fall
    out of tokenizing harmlessly rather than being silently blanked.
    """
    return _MATRIX_TOKEN.sub(lambda m: str(entry[m.group(1)]) if m.group(1) in entry else m.group(0), text)


def _steps_pytest_invocations(
    steps: list[dict], marker_default: str, subst: dict | None = None
) -> list[tuple[list[str], str, str]]:
    """Extract (path_args, effective_marker_expr, k_expr) for every pytest run step.

    When ``subst`` (a ``matrix.include`` entry) is given, ``${{ matrix.<key> }}``
    references in the run command are resolved first, so a matrix-driven path or
    ``-k`` selector becomes concrete before tokenizing.
    """
    out: list[tuple[list[str], str, str]] = []
    for step in steps:
        run = step.get("run")
        if not run or "pytest" not in run:
            continue
        if subst:
            run = _substitute_matrix(run, subst)
        # A step may stack several lines; split on newlines AND treat the whole
        # block as one shell command (line continuations join). Tokenize with
        # shlex after stripping line-continuation backslashes.
        joined = run.replace("\\\n", " ")
        for line in joined.splitlines():
            if "pytest" not in line:
                continue
            try:
                tokens = shlex.split(line)
            except ValueError:
                continue
            if "pytest" not in tokens:
                continue
            paths = [t for t in tokens if t.startswith("tests/") and (t.endswith(".py") or "/" in t)]
            if not paths:
                continue
            marker_expr = marker_default
            k_expr = ""
            for i, t in enumerate(tokens):
                if t == "-m" and i + 1 < len(tokens):
                    marker_expr = tokens[i + 1]  # command-line -m REPLACES addopts -m
                elif t == "-k" and i + 1 < len(tokens):
                    k_expr = tokens[i + 1]  # narrow selection to path-substring matches
            out.append((paths, marker_expr, k_expr))
    return out


def _services_backends(job: dict) -> frozenset[str]:
    backends: set[str] = set()
    for svc in (job.get("services") or {}).values():
        if isinstance(svc, dict):
            image = svc.get("image", "")
            be = _image_to_backend(str(image))
            if be:
                backends.add(be)
    return frozenset(backends)


def _docker_run_backends(steps: list[dict], satisfied_flags: frozenset[str]) -> frozenset[str]:
    """Backends started by ``if: matrix.<flag>``-guarded ``docker run`` steps,
    counted only when the leg satisfies that flag (the conformance pattern)."""
    backends: set[str] = set()
    for step in steps:
        run = step.get("run") or ""
        if "docker run" not in run:
            continue
        cond = str(step.get("if") or "")
        # If guarded on a matrix flag, only count it when this leg sets the flag.
        flag_guarded = False
        for flag in ("postgres", "neo4j", "weaviate"):
            if f"matrix.{flag}" in cond:
                flag_guarded = True
                if flag in satisfied_flags:
                    backends.add(flag)
        if not flag_guarded:
            # Unguarded docker run (rare): infer from the image text directly.
            for img_be in ("pgvector", "postgres", "neo4j", "weaviate"):
                if img_be in run.lower():
                    be = _image_to_backend(img_be)
                    if be:
                        backends.add(be)
    return frozenset(backends)


def _expand_jobs(workflow: dict) -> list[_Leg]:
    legs: list[_Leg] = []
    marker_default = _default_marker_expr()
    for job_name, job in (workflow.get("jobs") or {}).items():
        steps = job.get("steps") or []
        # Detect a test-running job by the presence of a pytest run step, NOT by
        # whether paths already resolve — a matrix-driven path (``${{ matrix.select }}``)
        # only becomes concrete after per-entry substitution below.
        if not any("pytest" in (step.get("run") or "") for step in steps):
            continue
        matrix = (job.get("strategy") or {}).get("matrix") or {}
        includes = matrix.get("include")
        if includes:
            for entry in includes:
                if not isinstance(entry, dict):
                    continue
                invocations = _steps_pytest_invocations(steps, marker_default, entry)
                if not invocations:
                    continue
                # Flags this leg sets truthy (postgres/neo4j/weaviate).
                satisfied = frozenset(flag for flag in ("postgres", "neo4j", "weaviate") if entry.get(flag))
                backends = _services_backends(job) | _docker_run_backends(steps, satisfied)
                # Also map the conformance backend token (cypher→neo4j etc.) so a
                # leg's declared backend counts even if the docker-run heuristic
                # ever misses it.
                token = entry.get("backend")
                if token:
                    be = _conformance_backend_token(str(token))
                    if be != _EMBEDDED:
                        backends = backends | {be}
                leg_name = f"{job_name} ({token})" if token else job_name
                legs.append(_Leg(leg_name, frozenset(backends), invocations))
        else:
            invocations = _steps_pytest_invocations(steps, marker_default)
            if not invocations:
                continue
            backends = _services_backends(job) | _docker_run_backends(steps, frozenset())
            legs.append(_Leg(job_name, frozenset(backends), invocations))
    return legs


def _workflow_files() -> list[Path]:
    """Every workflow file, discovered by glob (not a hard-coded list) so a new
    workflow — e.g. a dedicated slow/e2e live-DB lane — is audited automatically."""
    return sorted(_WORKFLOWS.glob("*.yml")) + sorted(_WORKFLOWS.glob("*.yaml"))


def _all_legs() -> list[_Leg]:
    files = _workflow_files()
    assert files, f"no workflow files found under {_WORKFLOWS}"
    assert _CI_YML.exists(), f"missing workflow file: {_CI_YML}"
    legs: list[_Leg] = []
    for wf in files:
        doc = yaml.safe_load(wf.read_text())
        if isinstance(doc, dict):
            legs.extend(_expand_jobs(doc))
    return legs


# ===========================================================================
# Enumerate filter/recall modules + infer backends + collect skip/xfail
# ===========================================================================
class _Module:
    __slots__ = ("path", "markers", "backends", "deferrals")

    def __init__(
        self,
        path: str,
        markers: frozenset[str],
        backends: frozenset[str],
        deferrals: list[tuple[str, str]],
    ) -> None:
        self.path = path
        self.markers = markers
        self.backends = backends  # required backend SET
        # (kind, reason) for every xfail/skip/skipif marker found, kind in
        # {"xfail", "skip", "skipif"}.
        self.deferrals = deferrals


def _universe_paths() -> list[str]:
    """Filter/recall test modules (the gate's universe).

    Scoped to the recall-filter subsystem: ``tests/recall/`` and
    ``tests/unit/filter/`` whole dirs, plus any ``test_*.py`` under
    ``tests/integration/`` or ``tests/e2e/`` whose filename contains ``filter``,
    ``conformance``, or ``compile``, plus :data:`_NAMED_EXTRAS`. Excludes
    unrelated ``conformance``/``recall`` unit tests elsewhere (integration-protocol
    conformance, recall scoring, etc.) and the meta-gates themselves.
    """
    found: set[str] = set()
    for p in (_REPO_ROOT / "tests" / "recall").glob("test_*.py"):
        found.add(_rel(p))
    for p in (_REPO_ROOT / "tests" / "unit" / "filter").glob("test_*.py"):
        found.add(_rel(p))
    for base in ("integration", "e2e"):
        for p in (_REPO_ROOT / "tests" / base).rglob("test_*.py"):
            name = p.name.lower()
            if any(tok in name for tok in ("filter", "conformance", "compile")):
                found.add(_rel(p))
    for extra in _NAMED_EXTRAS:
        if (_REPO_ROOT / extra).exists():
            found.add(extra)
    found -= set(_META_TEST_EXCLUSIONS)
    return sorted(found)


def _rel(p: Path) -> str:
    return p.resolve().relative_to(_REPO_ROOT).as_posix()


def _const_str(node: ast.AST) -> str | None:
    """Extract a string literal, joining implicit-concat / parenthesized parts."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):  # f-string — take literal parts only
        return "".join(v.value for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, str))
    return None


def _marker_name(node: ast.AST) -> str | None:
    """``pytest.mark.<name>`` (attribute or call) → ``<name>``."""
    target = node.func if isinstance(node, ast.Call) else node
    # walk attribute chain: pytest.mark.<name>
    if isinstance(target, ast.Attribute):
        if isinstance(target.value, ast.Attribute) and target.value.attr == "mark":
            return target.attr
    return None


def _reason_of(node: ast.Call) -> str:
    for kw in node.keywords:
        if kw.arg == "reason":
            return _const_str(kw.value) or ""
    return ""


def _condition_source(node: ast.Call) -> str:
    """Best-effort source text of a skipif's first positional condition arg."""
    if node.args:
        try:
            return ast.unparse(node.args[0])
        except Exception:
            return ""
    return ""


def _analyze_module(path: str) -> _Module:
    tree = ast.parse((_REPO_ROOT / path).read_text())

    markers: set[str] = set()
    backends: set[str] = set()
    deferrals: list[tuple[str, str]] = []

    # Collect every pytest.mark.* call/attribute anywhere in the module
    # (module-level pytestmark lists AND function decorators AND module-level
    # _SKIP = pytest.mark.skipif(...) assignments). NOTE: markers are unioned
    # module-wide — a module that MIXED marked and unmarked functions could be
    # counted covered by a leg that runs only the marked subset. Fine today: every
    # filter/recall module declares its markers at module level (pytestmark), so
    # the union equals each test's marker set. Revisit if a module mixes them.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Call, ast.Attribute)):
            name = _marker_name(node)
            if name is None:
                continue
            if name not in ("skipif", "skip", "xfail"):
                markers.add(name)
            if name in ("xfail", "skip", "skipif") and isinstance(node, ast.Call):
                reason = _reason_of(node)
                deferrals.append((name, reason))
                cond = _condition_source(node).upper() if name == "skipif" else ""
                backends |= _backends_from_signal(cond)

    # Code backstop for backend signals that live outside a marker call — e.g. a
    # ``_SKIP = pytest.mark.skipif(...)`` condition built from a module-level
    # ``_pg_reachable()`` / ``os.environ.get("NEO4J_INTEGRATION_TEST")``. We scan
    # the env-var STRINGS and called-name identifiers in real code only (AST), NOT
    # raw source text — a docstring/comment that merely MENTIONS a signal (e.g.
    # "NOT _pg_reachable-gated") must not be mistaken for a real gate.
    backends |= _backends_from_signal(_code_signal_text(tree))

    # Embedded-only fallback: a pure-unit / embedded module needs no live service.
    if not backends:
        backends.add(_EMBEDDED)
    elif "embedded" in markers and "postgres" not in backends and "neo4j" not in backends:
        backends.add(_EMBEDDED)

    return _Module(path, frozenset(markers), frozenset(backends), deferrals)


def _backends_from_signal(text: str) -> set[str]:
    """Map an UPPER-cased signal blob to the backends it implies."""
    up = text.upper()
    out: set[str] = set()
    if "NEO4J_INTEGRATION_TEST" in up or "KHORA_E2E_NEO4J_REQUIRED" in up:
        out.add("neo4j")
    if "WEAVIATE_INTEGRATION_TEST" in up or "KHORA_E2E_WEAVIATE_REQUIRED" in up:
        out.add("weaviate")
    if "_PG_REACHABLE" in up or "KHORA_PG_REQUIRED" in up or "KHORA_E2E_PG_REQUIRED" in up:
        out.add("postgres")
    # The container-free lanes (embedded sqlite_lance / in-process SurrealDB) gate
    # on these "required" flags too; both imply the embedded sentinel, which every
    # leg provides — so a module gated on them is covered by any leg.
    if "KHORA_E2E_EMBEDDED_REQUIRED" in up or "KHORA_E2E_SURREAL_REQUIRED" in up:
        out.add(_EMBEDDED)
    return out


def _code_signal_text(tree: ast.Module) -> str:
    """Backend-gate signals from REAL CODE only (not docstrings/comments).

    Joins (a) every string literal that is an arg to ``os.environ.get`` /
    ``os.getenv`` / ``os.environ[...]`` and (b) every called-function identifier
    (so a bare ``_pg_reachable()`` call surfaces). Docstrings and comments are
    never code-called and carry no ``os.environ`` arg, so they cannot leak in.
    """
    parts: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # called identifier: foo() → "foo"; obj.meth() → "meth"
            fn = node.func
            if isinstance(fn, ast.Name):
                parts.append(fn.id)
            elif isinstance(fn, ast.Attribute):
                parts.append(fn.attr)
            # os.environ.get("X") / os.getenv("X") string args
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    parts.append(a.value)
        elif isinstance(node, ast.Subscript):  # os.environ["X"]
            sl = node.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                parts.append(sl.value)
    return " ".join(parts)


def _all_modules() -> list[_Module]:
    return [_analyze_module(p) for p in _universe_paths()]


# Markers that authoritatively declare "this IS a filter-suite test". A module
# carrying one of these MUST be inside the enumerated universe — the safety net
# below proves tightening the universe can't become a silent escape hatch.
_FILTER_SUITE_MARKERS = frozenset({"filter_conformance", "filter_enforcement"})


def _modules_carrying_filter_markers() -> list[str]:
    """Every ``test_*.py`` under ``tests/`` that carries a filter-suite marker
    (AST-detected — a real ``pytest.mark.<m>``, not a text mention)."""
    out: list[str] = []
    for p in (_REPO_ROOT / "tests").rglob("test_*.py"):
        try:
            tree = ast.parse(p.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        names: set[str] = set()
        for node in ast.walk(tree):
            name = _marker_name(node)
            if name:
                names.add(name)
        if names & _FILTER_SUITE_MARKERS:
            out.append(_rel(p))
    return out


# ===========================================================================
# Core check (pure function — reused by the real test AND the self-test)
# ===========================================================================
def _orphaned_modules(modules: list[_Module], legs: list[_Leg]) -> list[tuple[str, frozenset[str], str]]:
    """Modules no leg both selects AND fully provisions.

    Returns (module_path, required_backends, why) for each orphan. ``why``
    distinguishes "no leg selects it" from "selected but under-provisioned".
    """
    orphans: list[tuple[str, frozenset[str], str]] = []
    for m in modules:
        if m.path in _KNOWN_UNRUN:
            continue
        selecting = [leg for leg in legs if leg.selects(m.path, m.markers)]
        if not selecting:
            orphans.append((m.path, m.backends, "no CI leg effectively selects it (path ∩ marker)"))
            continue
        if any(m.backends <= leg.backends for leg in selecting):
            continue
        closest = max(selecting, key=lambda leg: len(m.backends & leg.backends))
        missing = m.backends - closest.backends
        orphans.append(
            (
                m.path,
                m.backends,
                f"selected by {closest.name!r} but it lacks {sorted(missing)} (provisions {sorted(closest.backends)})",
            )
        )
    return orphans


# ===========================================================================
# Tests
# ===========================================================================
def test_enumeration_is_not_vacuous() -> None:
    """The universe glob must find a sane floor of modules.

    Tripwire against a path/glob refactor turning the gate into a silent no-op.
    """
    modules = _all_modules()
    assert len(modules) >= _MIN_EXPECTED_FILTER_MODULES, (
        f"filter/recall enumeration collapsed to {len(modules)} modules "
        f"(< {_MIN_EXPECTED_FILTER_MODULES}) — a glob/path refactor likely broke discovery; "
        "refusing to pass vacuously."
    )


def test_marked_filter_tests_are_in_universe() -> None:
    """Marker-authority safety net: a test carrying a filter-suite marker
    (``filter_conformance`` / ``filter_enforcement``) MUST be in the enumerated
    universe.

    Markers are the authoritative "this IS a filter-suite test" signal. The
    universe is a curated glob (dirs + named extras), so this proves the
    narrowing can't become a silent escape hatch: a marked filter test that
    lives outside the curated set fails here, forcing it into the universe (add
    a named extra) — it cannot slip past the coverage gate unnoticed. Meta-gates
    carry the markers too but are consciously excluded, so they are exempt.
    """
    universe = set(_universe_paths())
    marked = set(_modules_carrying_filter_markers()) - set(_META_TEST_EXCLUSIONS)
    escaped = sorted(marked - universe)
    assert not escaped, (
        "filter-marked test(s) live OUTSIDE the enumerated universe — they would "
        "escape the coverage gate:\n"
        + "\n".join(f"  {p}" for p in escaped)
        + "\nAdd each to _NAMED_EXTRAS (with a one-line reason) so the gate covers it."
    )


def test_legs_parsed_from_workflows() -> None:
    """At least the known test-running legs were parsed, with backends attached."""
    legs = _all_legs()
    names = {leg.name for leg in legs}
    assert any(n.startswith("test-unit") for n in names), f"test-unit leg not parsed: {sorted(names)}"
    assert any(n.startswith("test-integration") for n in names), "test-integration leg not parsed"
    # test-integration must provision BOTH postgres and neo4j (services block).
    ti = next(leg for leg in legs if leg.name.startswith("test-integration"))
    assert {"postgres", "neo4j"} <= ti.backends, f"test-integration backends={sorted(ti.backends)}"
    # The conformance matrix must yield a postgres leg and a neo4j (cypher) leg.
    assert any("postgres" in leg.backends and "conformance" in leg.name for leg in legs)
    assert any("neo4j" in leg.backends and "conformance" in leg.name for leg in legs)


def test_no_orphaned_filter_test() -> None:
    """Every filter/recall module is selected + fully provisioned by some leg.

    This is the core gate. A failure means a test is silently disabled: green in
    CI but never executed because no job provisions its backend (or a ``-m``
    marker excludes it). Fix by wiring a job that selects its path with a
    matching marker AND provisions its backends — or, if consciously deferred,
    add it to ``_KNOWN_UNRUN`` with a tracking ref.
    """
    modules = _all_modules()
    legs = _all_legs()
    orphans = _orphaned_modules(modules, legs)
    assert not orphans, "filter/recall test(s) not covered by any provisioning CI leg:\n" + "\n".join(
        f"  {path}  requires {sorted(req)} — {why}" for path, req, why in orphans
    )


def test_deferrals_carry_tracking_ref() -> None:
    """Every backend-gap ``xfail`` / deferred ``skip`` cites a public tracking ref.

    Environment-guard ``skipif`` reasons (store-not-up, dep-not-installed,
    "set NEO4J_INTEGRATION_TEST=1") are exempt — they describe how to run, not
    deferred work. ``xfail`` always asserts a known-broken behavior, so it always
    needs a ref.
    """
    offenders: list[str] = []
    for m in _all_modules():
        for kind, reason in m.deferrals:
            low = reason.lower()
            is_env_guard = kind == "skipif" and any(s in low for s in _ENV_GUARD_REASON_SUBSTRINGS)
            if is_env_guard:
                continue
            if not _TRACKING_REF.search(reason):
                offenders.append(f"  {m.path}: {kind} reason lacks #NNNN / ADR-NNN ref: {reason[:90]!r}")
    assert not offenders, "backend-gap xfail/skip without a tracking reference:\n" + "\n".join(offenders)


def test_integration_e2e_filter_tests_claimed_by_a_pillar() -> None:
    """Every integration/e2e filter test is claimed by a verification pillar.

    Three pillar classes, identified structurally by what a leg selects on:
      * filter-conformance — a leg selecting the ``filter_conformance`` marker.
      * filter_enforcement — a leg whose effective selection includes the test
        (via the ``filter_enforcement`` marker or the integration/unit paths).
      * slow|e2e — a leg selecting ``slow`` or running the e2e lanes.
    A test claimed by none fails. Pure-unit ``tests/recall/`` + ``tests/unit/filter/``
    modules are exempt (covered by ``test-unit``; no backend pillar).
    """
    legs = _all_legs()
    unclaimed: list[str] = []
    for m in _all_modules():
        if not (m.path.startswith("tests/integration/") or m.path.startswith("tests/e2e/")):
            continue
        if m.path in _KNOWN_UNRUN:
            continue
        claimed = any(leg.selects(m.path, m.markers) for leg in legs)
        if not claimed:
            unclaimed.append(
                f"  {m.path} (markers={sorted(m.markers)}) — claimed by no "
                "filter-conformance / filter_enforcement / slow|e2e leg"
            )
    assert not unclaimed, "integration/e2e filter test(s) claimed by no pillar:\n" + "\n".join(unclaimed)


def test_known_unrun_entries_have_refs() -> None:
    """Every ``_KNOWN_UNRUN`` allowlist entry carries a public tracking ref."""
    for path, reason in _KNOWN_UNRUN.items():
        assert _TRACKING_REF.search(reason), f"_KNOWN_UNRUN[{path!r}] reason lacks #NNNN / ADR-NNN ref: {reason!r}"


def test_known_unrun_entries_are_not_stale() -> None:
    """An allowlisted module that a real leg now covers must be removed.

    Drift in both directions, same discipline as the filter-enforcement gate: a
    stale entry hides that the gap was closed.
    """
    legs = _all_legs()
    modules_by_path = {m.path: m for m in _all_modules()}
    stale: list[str] = []
    for path in _KNOWN_UNRUN:
        m = modules_by_path.get(path)
        if m is None:
            continue  # path drifted away; the universe-floor test covers vacuity
        if any(m.backends <= leg.backends and leg.selects(m.path, m.markers) for leg in legs):
            stale.append(f"  {path} — now covered by a provisioning leg; remove the _KNOWN_UNRUN entry")
    assert not stale, "stale _KNOWN_UNRUN entries:\n" + "\n".join(stale)


def test_gate_is_falsifiable() -> None:
    """The orphan check has teeth: a Postgres-needing module selected only by an
    embedded-only leg must be reported orphaned, and adding a Postgres leg must
    clear it.

    Reuses the SAME ``_orphaned_modules`` the real gate calls (no duplicated
    logic), proving the teeth independent of live repo state.
    """
    pg_module = _Module(
        path="tests/integration/fake/test_needs_postgres.py",
        markers=frozenset({"integration"}),
        backends=frozenset({"postgres"}),
        deferrals=[],
    )
    # A leg that SELECTS the module (path + marker match) but provisions only
    # embedded — the exact "selected but under-provisioned" orphan shape.
    embedded_only_leg = _Leg(
        name="fake-embedded",
        backends=frozenset(),  # → {embedded} after __init__
        invocations=[(["tests/integration/"], "integration", "")],
    )
    orphans = _orphaned_modules([pg_module], [embedded_only_leg])
    assert [o[0] for o in orphans] == [pg_module.path], "gate failed to flag an under-provisioned module — no teeth"

    # Add a leg that selects it AND provisions postgres → orphan clears.
    pg_leg = _Leg(
        name="fake-postgres",
        backends=frozenset({"postgres"}),
        invocations=[(["tests/integration/"], "integration", "")],
    )
    assert not _orphaned_modules([pg_module], [embedded_only_leg, pg_leg]), (
        "gate still reports orphan after a provisioning leg was added — false positive"
    )


def test_marker_expr_replacement_semantics() -> None:
    """Pins the central correctness invariant: a command-line ``-m`` REPLACES the
    addopts default, and marker EXCLUSION de-claims a path-included test.

    A ``slow`` test whose path a leg includes but whose ``-m "not slow"`` excludes
    must NOT be selected; the same test IS selected by a leg that runs ``-m slow``.
    """
    slow_e2e = _Module(
        path="tests/e2e/test_fake_slow.py",
        markers=frozenset({"e2e", "slow"}),
        backends=frozenset({_EMBEDDED}),
        deferrals=[],
    )
    excludes_slow = _Leg("excl", frozenset(), [(["tests/e2e/"], "not slow and not filter_conformance", "")])
    selects_slow = _Leg("incl", frozenset(), [(["tests/e2e/"], "slow", "")])
    assert not excludes_slow.selects(slow_e2e.path, slow_e2e.markers), "marker exclusion failed to de-claim a test"
    assert selects_slow.selects(slow_e2e.path, slow_e2e.markers), "a -m slow leg should select a slow test"


def test_workflow_discovery_is_glob_based() -> None:
    """Workflows are discovered by glob, not a hard-coded pair — so a new workflow
    file (e.g. a dedicated slow/e2e live-DB lane) is audited the moment it lands.

    A hard-coded list would silently ignore a new workflow, and the
    ``_KNOWN_UNRUN`` removal condition (a job that lives in a not-yet-read file)
    could never fire.
    """
    files = {p.name for p in _workflow_files()}
    assert "ci.yml" in files, f"ci.yml not discovered by glob: {sorted(files)}"
    assert "filter-conformance.yml" in files, f"filter-conformance.yml not discovered: {sorted(files)}"


def test_matrix_select_and_dash_k_modeling() -> None:
    """A ``matrix.include`` leg whose run uses ``${{ matrix.select }}`` expands per
    entry to concrete args, and ``-k`` narrows selection to path-substring matches.

    This is the exact shape a dedicated slow/e2e lane uses (a matrix selector plus
    a ``-k`` filename token). Without per-entry substitution the path would be an
    unresolved ``${{ ... }}`` token and the leg would be dropped; without ``-k``
    modelling the leg would over-claim every module under the path.
    """
    wf = {
        "jobs": {
            "e2e": {
                "strategy": {
                    "matrix": {
                        "include": [
                            {"select": "tests/e2e/ -k rowset_graph", "postgres": True, "neo4j": True},
                        ]
                    }
                },
                "services": {
                    "postgres": {"image": "pgvector/pgvector:pg17"},
                    "neo4j": {"image": "neo4j:2025.12.1"},
                },
                "steps": [{"run": 'uv run pytest ${{ matrix.select }} -m "e2e and slow" -n auto'}],
            }
        }
    }
    legs = _expand_jobs(wf)
    assert len(legs) == 1, f"expected one expanded leg, got {[leg.name for leg in legs]}"
    leg = legs[0]
    assert {"postgres", "neo4j"} <= leg.backends, f"e2e leg backends={sorted(leg.backends)}"
    graph_markers = frozenset({"e2e", "slow"})
    # -k rowset_graph selects the graph module (filename token is a path substring)…
    assert leg.selects("tests/e2e/test_filter_rowset_graph.py", graph_markers)
    # …but NOT the chronicle module: -k narrows, so it is not over-claimed.
    assert not leg.selects("tests/e2e/test_filter_rowset_chronicle.py", graph_markers)


def test_e2e_env_flags_infer_backends() -> None:
    """A test gated on ``KHORA_E2E_{PG,NEO4J,WEAVIATE}_REQUIRED`` is inferred to
    require that backend, so a live slow/e2e lane's modules carry the right
    provisioning requirement once it lands."""
    assert _backends_from_signal("KHORA_E2E_PG_REQUIRED") == {"postgres"}
    assert _backends_from_signal("KHORA_E2E_NEO4J_REQUIRED") == {"neo4j"}
    assert _backends_from_signal("KHORA_E2E_WEAVIATE_REQUIRED") == {"weaviate"}
    assert _backends_from_signal("KHORA_E2E_PG_REQUIRED KHORA_E2E_NEO4J_REQUIRED") == {"postgres", "neo4j"}
    # The container-free lanes' required flags both map to the embedded sentinel.
    assert _backends_from_signal("KHORA_E2E_EMBEDDED_REQUIRED") == {_EMBEDDED}
    assert _backends_from_signal("KHORA_E2E_SURREAL_REQUIRED") == {_EMBEDDED}


def test_tracking_ref_accepts_long_issue_numbers() -> None:
    """The tracking-ref pattern accepts issue numbers of 3+ digits (not just 3–4),
    so a five-digit issue is a valid ref and an ADR ref still matches."""
    assert _TRACKING_REF.search("tracked in #12345")
    assert _TRACKING_REF.search("see #806")
    assert _TRACKING_REF.search("per ADR-001")
    assert not _TRACKING_REF.search("issue #42")  # 2 digits is not a tracking ref


def test_gate_module_is_itself_selected_by_a_leg() -> None:
    """Self-coverage: this gate must itself be selected by some CI leg.

    The gate guards other tests, but is itself only useful if it runs. A workflow
    edit that drops ``tests/unit/`` from the ``test-unit`` job would deselect the
    gate, leaving nothing to notice — this assertion fails first if that happens.
    """
    this_path = _rel(Path(__file__))
    me = _analyze_module(this_path)
    legs = _all_legs()
    assert any(leg.selects(this_path, me.markers) for leg in legs), (
        f"this gate ({this_path}) is selected by no CI leg — a workflow change may "
        "have deselected it; the gate cannot guard what does not itself run."
    )
