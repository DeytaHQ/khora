"""Subprocess harness that proves adapter submodules don't eagerly
import their framework.

The check itself is parameterised over ``ADAPTERS`` — currently empty
because no adapter has merged yet (#619 ships the architecture, not
the adapters). Adapter PRs add their framework name to ``ADAPTERS`` and
this test exercises them.

Mechanism: poison ``sys.modules['<framework>'] = None`` so any real
``import <framework>`` inside the adapter file raises ``ImportError``,
then try to import ``khora.integrations.<framework>``. Success means
the adapter deferred the framework import properly (function bodies or
``if TYPE_CHECKING``).

This is layer 2 of optional-install discipline. Layer 1 is the
``tools/check_optional_imports.py`` AST lint that runs in CI lint, and
it catches the same bug statically — but the subprocess probe catches
bugs the AST misses (e.g. an import buried in a decorator argument or a
default value).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

# Adapter PRs (CrewAI, LangGraph, ...) add their framework name here when
# they merge. Each entry is the submodule name; the test poisons it in
# sys.modules and asserts khora.integrations.<name> still imports.
ADAPTERS: list[str] = ["crewai", "langgraph", "google_adk", "openai_agents", "llamaindex"]

# Extra frameworks whose import name differs from the adapter dir name.
# The AST lint keys off the dir name, so for adapters whose framework
# publishes under a different name we additionally poison the real
# framework module in ``sys.modules`` and prove the adapter still imports.
# Each tuple is ``(adapter_dir_name, framework_module_name)``.
#   - openai-agents (PyPI) installs as Python module ``agents``
#   - llama-index-core installs as Python module ``llama_index``
EXTRA_FRAMEWORK_NAMES: list[tuple[str, str]] = [
    ("openai_agents", "agents"),
    ("llamaindex", "llama_index"),
]


@pytest.mark.parametrize("name", ADAPTERS)
def test_adapter_imports_without_framework(name: str) -> None:
    """Importing the adapter must not require the framework.

    Skipped until at least one adapter merges and is added to ``ADAPTERS``.
    """
    script = textwrap.dedent(
        f"""
        import sys
        # Poison: any real import of {name} raises ImportError.
        sys.modules[{name!r}] = None
        import khora.integrations.{name}  # must still succeed
        """
    )
    subprocess.run(  # noqa: S603 — test harness, sys.executable is trusted
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize(("adapter_name", "framework_module"), EXTRA_FRAMEWORK_NAMES)
def test_adapter_imports_when_renamed_framework_poisoned(adapter_name: str, framework_module: str) -> None:
    """Adapters whose framework module name differs from the dir name.

    The AST lint (``tools/check_optional_imports.py``) only catches
    ``import <dir_name>`` at module top level. For adapters whose framework
    publishes under a different name (e.g. ``openai-agents`` → ``agents``,
    or ``llamaindex`` → ``llama_index``)
    that's not enough — we additionally poison the real framework name in
    ``sys.modules`` and prove the adapter still imports cleanly.
    """
    script = textwrap.dedent(
        f"""
        import sys
        # Poison: any real import of {framework_module} raises ImportError.
        sys.modules[{framework_module!r}] = None
        import khora.integrations.{adapter_name}  # must still succeed
        """
    )
    subprocess.run(  # noqa: S603 — test harness, sys.executable is trusted
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_ast_lint_helper_runs_clean_on_current_tree():
    """Layer-1 sanity: the AST lint exits 0 on the current tree.

    Provides immediate signal in unit tests if someone lands a top-level
    framework import — they don't have to wait for the CI lint job.
    """
    result = subprocess.run(
        [sys.executable, "tools/check_optional_imports.py"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"check_optional_imports.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
