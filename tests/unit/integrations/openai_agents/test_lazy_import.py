"""Verify ``khora.integrations.openai_agents`` does not eagerly import ``agents``.

The full subprocess-level harness lives in ``test_no_eager_imports.py``
and is parametrized over every adapter in ``ADAPTERS`` plus extras like
this one. This file is a local quick-check kept alongside the adapter
so failures point straight at this PR's diff rather than at the shared
list.

Two poisons:

* ``sys.modules['openai_agents']`` — defensive, matches the AST lint's
  framework key (the adapter dir name).
* ``sys.modules['agents']`` — the real upstream package name. The SDK
  publishes as ``openai-agents`` on PyPI but installs as the Python
  module ``agents``; this poison is the one that actually catches a
  regression.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_openai_agents_submodule_imports_with_agents_poisoned() -> None:
    script = textwrap.dedent(
        """
        import sys
        # Poison both the adapter-dir name AND the real framework module
        # name. Either one being imported eagerly at module top level
        # raises ImportError on these next lines.
        sys.modules['openai_agents'] = None
        sys.modules['agents'] = None
        import khora.integrations.openai_agents  # must still succeed
        # Also verify the heavy submodules can be loaded — they hold the
        # most surface area where a stray top-level import could hide.
        from khora.integrations.openai_agents import session  # noqa: F401
        from khora.integrations.openai_agents import tool  # noqa: F401
        from khora.integrations.openai_agents import hooks  # noqa: F401
        from khora.integrations.openai_agents import _mapping  # noqa: F401
        """
    )
    subprocess.run(  # noqa: S603 — sys.executable is trusted
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
