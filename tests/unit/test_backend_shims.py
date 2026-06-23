"""Deprecation + identity guard for the temporal-store relocation shims.

The temporal vector store moved out of ``khora.engines.skeleton.backends`` into
``khora.storage.temporal``. The old import paths survive as thin shims that
re-export the moved names verbatim and emit a ``DeprecationWarning`` at import
time. These tests lock in two properties for every old path:

1. importing it emits a ``DeprecationWarning`` (so downstream code is nudged to
   migrate), and
2. each re-exported object is *identical* (``is``) to the one now living in the
   corresponding ``khora.storage.temporal`` module (so the move was a pure
   re-export, not a fork).

WHY the subprocess for the warning check: the shim's ``warnings.warn(...)``
fires exactly once, at module-execution time. Python caches the module in
``sys.modules`` after the first import, so a second in-process ``import`` is a
no-op that does NOT re-run the module body and therefore does NOT re-warn —
``pytest.warns`` would see nothing and falsely fail. Running each import in a
fresh ``python -W error::DeprecationWarning`` subprocess gives a clean module
table every time, and turning the warning into an error means a missing warning
surfaces as a non-zero exit. The in-process identity check below sidesteps the
same caching by popping the modules from ``sys.modules`` before re-importing
under ``pytest.warns``.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

# (old shim path, new home, [re-exported names checked for identity]).
_SHIMS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "khora.engines.skeleton.backends",
        "khora.storage.temporal",
        (
            "TemporalChunk",
            "TemporalFilter",
            "TemporalSearchResult",
            "TemporalVectorStore",
            "create_temporal_store",
            "document_denorm_fields",
            "temporal_chunk_to_chunk",
        ),
    ),
    (
        "khora.engines.skeleton.backends.pgvector",
        "khora.storage.temporal.pgvector",
        ("PgVectorTemporalStore", "khora_chunks_table"),
    ),
    (
        "khora.engines.skeleton.backends.surrealdb",
        "khora.storage.temporal.surrealdb",
        ("SurrealDBTemporalStore", "_BACKED_SYSTEM_KEYS", "_TEMPORAL_CHUNK_SCHEMA"),
    ),
    (
        "khora.engines.skeleton.backends.weaviate",
        "khora.storage.temporal.weaviate",
        ("WeaviateTemporalStore", "WeaviateBackendConfig", "COLLECTION_NAME"),
    ),
    (
        "khora.engines.skeleton.backends.turbopuffer",
        "khora.storage.temporal.turbopuffer",
        ("TurbopufferTemporalStore", "TurbopufferBackendConfig"),
    ),
    (
        "khora.engines.skeleton.backends.sqlite_lance",
        "khora.storage.temporal.sqlite_lance",
        ("SQLiteLanceTemporalStore",),
    ),
)

_SHIM_IDS = [shim for shim, _new, _names in _SHIMS]


@pytest.mark.parametrize(("shim", "new_home", "names"), _SHIMS, ids=_SHIM_IDS)
def test_shim_import_emits_deprecation_warning(shim: str, new_home: str, names: tuple[str, ...]) -> None:
    """Importing the old path raises under ``-W error::DeprecationWarning``.

    Subprocess (not ``pytest.warns``) because the warning fires once at module
    load and ``sys.modules`` caching makes an in-process re-import silent — see
    the module docstring.
    """
    result = subprocess.run(  # noqa: S603 — test harness, sys.executable is trusted
        [sys.executable, "-W", "error::DeprecationWarning", "-c", f"import {shim}"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0, (
        f"importing {shim} under -W error::DeprecationWarning did not raise — "
        f"the shim's DeprecationWarning is missing.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "DeprecationWarning" in result.stderr, (
        f"{shim} import failed but not with a DeprecationWarning:\n{result.stderr}"
    )


@pytest.mark.parametrize(("shim", "new_home", "names"), _SHIMS, ids=_SHIM_IDS)
def test_shim_reexports_are_identical(shim: str, new_home: str, names: tuple[str, ...]) -> None:
    """Each name on the shim ``is`` the same object as on its new home.

    Drops both modules from ``sys.modules`` first so the shim body re-runs under
    ``pytest.warns`` (proving the warning again, in-process this time), then
    compares object identity name-by-name.
    """
    sys.modules.pop(shim, None)
    new_mod = importlib.import_module(new_home)

    with pytest.warns(DeprecationWarning):
        shim_mod = importlib.import_module(shim)

    for name in names:
        shim_obj = getattr(shim_mod, name)
        new_obj = getattr(new_mod, name)
        assert shim_obj is new_obj, (
            f"{shim}.{name} is not the same object as {new_home}.{name} — "
            "the shim must re-export verbatim, not redefine."
        )
