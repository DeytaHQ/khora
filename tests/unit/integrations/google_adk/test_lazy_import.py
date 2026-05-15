"""Verify ``khora.integrations.google_adk`` does not eagerly import the SDK.

The full subprocess-level harness lives in ``test_no_eager_imports.py``
and is parametrized over every adapter in ``ADAPTERS``. This file is a
local quick-check kept alongside the adapter so failures point straight
at this PR's diff rather than at the shared list.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_google_adk_submodule_imports_with_google_adk_poisoned() -> None:
    script = textwrap.dedent(
        """
        import sys
        # Poison: any real ``import google.adk`` raises ImportError.
        sys.modules['google.adk'] = None
        sys.modules['google.adk.memory'] = None
        sys.modules['google.adk.memory.base_memory_service'] = None
        sys.modules['google.adk.sessions'] = None
        sys.modules['google.adk.events'] = None

        import khora.integrations.google_adk  # must still succeed
        from khora.integrations.google_adk import memory_service  # noqa: F401
        from khora.integrations.google_adk import _mapping  # noqa: F401
        """
    )
    subprocess.run(  # noqa: S603 — sys.executable is trusted
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
