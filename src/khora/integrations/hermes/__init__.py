"""``khora.integrations.hermes`` — ``hermes-agent`` ``MemoryProvider`` adapter.

End-user surface: :func:`KhoraMemoryProvider`. Build one and hand it to
the example Hermes plugin (``examples/integrations/hermes/plugin/``)
which Hermes loads via its plugin discovery::

    from khora import Khora
    from khora.integrations.hermes import KhoraMemoryProvider

    kb = Khora()
    await kb.connect()
    provider = KhoraMemoryProvider(kb=kb)

Distribution model (per issue #628): option (a) + example plugin
directory. The adapter ships with khora's optional ``[hermes]`` extra,
and the example plugin folder is what Hermes actually discovers — it
imports this module, calls the factory, and returns the resulting
provider to Hermes.

Module-load discipline: nothing from ``hermes_agent`` is imported at
this module's top level — not even transitively. :func:`KhoraMemoryProvider`
is exposed via a thin re-export that defers the actual import (and the
inevitable ABC lookup) to call time. The AST lint
(``tools/check_optional_imports.py``) does NOT catch ``import
hermes_agent`` (note the underscore distinguishing the dist name
``hermes-agent`` from the import name); the subprocess no-import probe
in ``tests/unit/integrations/test_no_eager_imports.py`` is the gate of
record.

Stability: experimental. The ``hermes-agent`` ``MemoryProvider`` ABC is
still pre-1.0; expect rework per upstream minor until it stabilises.
"""

from __future__ import annotations

from typing import Any


def KhoraMemoryProvider(**kwargs: Any) -> Any:  # noqa: N802 — factory masquerading as constructor
    """Re-export of :func:`khora.integrations.hermes.provider.KhoraMemoryProvider`.

    Wrapped so this module's top-level imports never pull in
    ``provider.py`` until a caller actually wants a provider — keeps
    ``import khora.integrations.hermes`` free of any Hermes-side
    references on systems where the optional extra is not installed.
    """
    from khora.integrations.hermes.provider import (  # noqa: PLC0415 — lazy
        KhoraMemoryProvider as _impl,
    )

    return _impl(**kwargs)


__all__ = ["KhoraMemoryProvider"]
