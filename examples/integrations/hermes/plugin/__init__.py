"""Khora memory provider for Hermes.

Copy this directory into ``$HERMES_HOME/plugins/khora/``. Hermes will
discover and register it automatically on next startup. Requires:

    pip install 'khora[hermes]'

and a configured Khora instance available via ``Khora.shared()`` or an
explicit kb passed through ``KHORA_HERMES_KB_FACTORY`` (callable import
path that returns a ``Khora`` instance).
"""

from __future__ import annotations

import os
from importlib import import_module
from typing import Any


def register(ctx: Any) -> None:
    """Hermes plugin entry point. Called once at plugin discovery time."""
    from khora import Khora
    from khora.integrations.hermes import KhoraMemoryProvider

    factory = os.environ.get("KHORA_HERMES_KB_FACTORY")
    if factory:
        module_path, _, attr = factory.rpartition(":")
        kb = getattr(import_module(module_path), attr)()
    else:
        kb = Khora.shared()

    ctx.register_memory_provider(KhoraMemoryProvider(kb=kb))
