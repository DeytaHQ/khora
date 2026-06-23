"""Lazy-import guard for the temporal-store package.

The temporal-store relocation keeps every optional backend dependency lazy:
``import khora.storage.temporal`` exposes the protocol and the
``create_temporal_store`` factory, but must NOT eagerly drag in a backend's
heavy third-party dependency. The backend impl modules
(``khora.storage.temporal.sqlite_lance`` / ``weaviate`` / ``turbopuffer``) are
imported lazily inside ``create_temporal_store`` only when that backend is
actually selected, so a process that merely touches the package — or selects a
different backend — never pays for LanceDB / Weaviate / Turbopuffer.

This is checked in a FRESH subprocess on purpose: the parent pytest process has
already imported many of these modules for other tests, so an in-process
``sys.modules`` check would be polluted and meaningless. ``sys.executable -c``
gives a clean interpreter whose module table reflects only what
``import khora.storage.temporal`` itself pulls in.

WHAT IS (AND IS NOT) ASSERTED
-----------------------------
We assert the genuinely-lazy *backend* dependencies are absent:

* ``lancedb`` — the LanceDB vector store, imported only by the sqlite_lance
  backend impl module (and the embedded vector adapter). This is the headline
  guarantee of the relocation: touching the package must not load LanceDB.
* ``weaviate`` / ``turbopuffer`` — the remote backend SDKs, imported only inside
  their respective impl modules.

We deliberately do NOT assert ``pyarrow`` absence. ``pyarrow`` is pulled into
``sys.modules`` transitively by ``neo4j`` (``neo4j._optional_deps`` does
``import pyarrow``), and ``khora.storage.backends.__init__`` eagerly imports
``Neo4jBackend`` as part of normal package initialisation. That eager neo4j
import predates and is independent of the temporal-store relocation, so binding
this test to ``pyarrow`` absence would assert something that was never true and
has nothing to do with the move. The lazy-backend guarantee is fully captured by
the LanceDB / Weaviate / Turbopuffer assertions above.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest

# Top-level package names of the genuinely-lazy backend dependencies. Each is
# imported only inside the corresponding backend impl module, which
# ``create_temporal_store`` imports lazily — so none should appear after a bare
# ``import khora.storage.temporal``.
_FORBIDDEN_TOP_LEVEL = ("lancedb", "weaviate", "turbopuffer")


def test_import_does_not_leak_optional_backend_deps() -> None:
    """A clean ``import khora.storage.temporal`` loads no lazy backend SDK.

    The leak assertion only bites for SDKs that are actually installed — an
    absent package can never appear in ``sys.modules`` regardless of whether the
    impl modules import it lazily or eagerly, so asserting its absence would be
    vacuous. To avoid a silent vacuous pass (the anti-pattern of "skipped but
    looks green"), we (a) require the headline dep ``lancedb`` — always present
    wherever the embedded stack runs — to be installed, skipping loudly with a
    reason otherwise, and (b) surface the installed-vs-absent split so CI logs
    show exactly which assertions were meaningful. An *eager* impl-module import
    of an absent SDK is still caught independently: it would raise ImportError
    at ``import khora.storage.temporal`` and trip the non-zero-exit assertion.
    """
    installed = [p for p in _FORBIDDEN_TOP_LEVEL if importlib.util.find_spec(p) is not None]
    if "lancedb" not in installed:
        pytest.skip(
            "lancedb is not installed, so the lazy-import guard would be vacuous "
            "(no backend SDK present to leak). Install the embedded extra to run it."
        )
    absent = [p for p in _FORBIDDEN_TOP_LEVEL if p not in installed]
    # Loud visibility (shows in -s / on failure): which forbidden SDKs the leak
    # assertion actually exercised vs. which were absent (assertion vacuous).
    print(
        f"lazy-import guard — installed (asserted absent-after-import): {installed}; not installed (vacuous): {absent}"
    )

    forbidden = repr(_FORBIDDEN_TOP_LEVEL)
    script = (
        "import sys\n"
        "import khora.storage.temporal  # noqa: F401\n"
        f"forbidden = {forbidden}\n"
        "leaked = sorted(\n"
        "    m for m in sys.modules\n"
        "    if m.split('.')[0] in forbidden\n"
        ")\n"
        "if leaked:\n"
        "    sys.stderr.write('LEAKED: ' + ', '.join(leaked) + '\\n')\n"
        "    sys.exit(1)\n"
    )
    result = subprocess.run(  # noqa: S603 — test harness, sys.executable is trusted
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        "importing khora.storage.temporal eagerly loaded a lazy backend "
        f"dependency:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
