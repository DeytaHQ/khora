"""Verify ``khora.integrations.crewai`` does not eagerly import CrewAI.

The full subprocess-level harness lives in ``test_no_eager_imports.py``
and is parametrized over every adapter in ``ADAPTERS``. This file is a
local quick-check kept alongside the adapter so failures point straight
at this PR's diff rather than at the shared list.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_crewai_submodule_imports_with_crewai_poisoned() -> None:
    script = textwrap.dedent(
        """
        import sys
        # Poison: any real `import crewai` raises ImportError.
        sys.modules['crewai'] = None
        import khora.integrations.crewai  # must still succeed
        # Also verify the storage submodule can be loaded — it has the
        # most surface area where a stray top-level import could hide.
        from khora.integrations.crewai import storage  # noqa: F401
        from khora.integrations.crewai import _mapping  # noqa: F401
        """
    )
    subprocess.run(  # noqa: S603 — sys.executable is trusted
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
