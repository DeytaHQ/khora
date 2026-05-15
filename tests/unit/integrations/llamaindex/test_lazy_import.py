"""Verify ``khora.integrations.llamaindex`` imports without ``llama_index``.

This is the adapter-local mirror of the cross-adapter probe in
``tests/unit/integrations/test_no_eager_imports.py``. Lives in the
llamaindex test dir so a failure here points directly at the adapter
that broke the discipline.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_package_import_without_llama_index() -> None:
    """``import khora.integrations.llamaindex`` must not require llama_index."""
    script = textwrap.dedent(
        """
        import sys
        # Poison: any real import of llama_index raises ImportError.
        sys.modules["llama_index"] = None
        import khora.integrations.llamaindex as adapter
        assert adapter.__all__, "adapter package must export its public surface"
        """
    )
    subprocess.run(  # noqa: S603 — test harness, sys.executable is trusted
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
