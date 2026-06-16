"""Guard the core dependency set declared in ``pyproject.toml``.

Some modules are imported unconditionally on the eager ``import khora`` path,
so the packages they need must be declared as *direct core dependencies* — not
left to arrive transitively through some other dependency's pin. A transitive
floor can be loosened or dropped by an upstream release without warning, which
would break a bare ``import khora`` for installers that did not happen to pull
the package in any other way.

``numpy`` is the concrete case this test pins: it is imported with no guard at
import time, must be a declared core dependency, and must floor high enough to
guarantee a wheel for the project's minimum supported Python.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _core_dependencies() -> list[str]:
    """Return the ``[project].dependencies`` array (core deps, not extras)."""
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    return data["project"]["dependencies"]


def test_numpy_is_a_core_dependency() -> None:
    """numpy is imported unconditionally at import time -> must be a core dep."""
    deps = _core_dependencies()

    try:
        from packaging.requirements import Requirement

        names = {Requirement(spec).name.lower() for spec in deps}
    except Exception:  # pragma: no cover - packaging always present transitively
        names = {spec.split(">=")[0].split("==")[0].split("[")[0].strip().lower() for spec in deps}

    assert "numpy" in names, (
        "numpy must be declared as a direct core dependency in "
        "[project].dependencies — it is imported unconditionally on the eager "
        "`import khora` path and must not rely on transitive resolution."
    )


def test_numpy_floor_is_at_least_2_1() -> None:
    """The numpy floor must guarantee a wheel for the minimum supported Python."""
    deps = _core_dependencies()
    numpy_spec = next(spec for spec in deps if spec.lstrip().lower().startswith("numpy"))

    try:
        from packaging.requirements import Requirement
        from packaging.version import Version

        req = Requirement(numpy_spec)
        lower_bounds = [Version(s.version) for s in req.specifier if s.operator in (">=", "==", ">", "~=")]
        assert lower_bounds, f"numpy requirement {numpy_spec!r} has no lower bound"
        assert max(lower_bounds) >= Version("2.1"), f"numpy lower bound must be >= 2.1, got {numpy_spec!r}"
    except ImportError:  # pragma: no cover - packaging always present transitively
        assert ">=2.1" in numpy_spec or ">=2." in numpy_spec or "==2." in numpy_spec, (
            f"numpy requirement {numpy_spec!r} must floor at >= 2.1"
        )
