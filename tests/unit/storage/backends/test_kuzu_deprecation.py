"""Kuzu backend deprecation (removal scheduled for khora 0.10.0).

Verifies that instantiating ``KuzuBackend`` emits a ``DeprecationWarning``
even when the optional ``kuzu`` package is not installed (the warning
fires before any IO).
"""

from __future__ import annotations

import warnings

import pytest

from khora.storage.backends.kuzu import KuzuBackend


@pytest.mark.unit
def test_kuzu_backend_emits_deprecation_warning(tmp_path) -> None:
    """KuzuBackend.__init__ must emit a DeprecationWarning per instantiation.

    The warning must fire before any IO so the test does not require the
    optional ``kuzu`` package to be installed. We don't call ``connect()``.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        KuzuBackend(database_path=str(tmp_path / "kuzu_db"))

    dep_warnings = [
        w
        for w in caught
        if issubclass(w.category, DeprecationWarning) and "KuzuBackend is deprecated" in str(w.message)
    ]
    assert len(dep_warnings) == 1, (
        f"Expected one DeprecationWarning from KuzuBackend.__init__, got "
        f"{len(dep_warnings)}: {[str(w.message) for w in caught]}"
    )
    assert "0.10.0" in str(dep_warnings[0].message), "Deprecation warning must mention the removal version (0.10.0)"
