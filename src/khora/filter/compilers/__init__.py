"""Backend filter compilers (Layer 4 implementations) — ``@internal``.

Each module here lowers a canonical :class:`~khora.filter.ast.FilterNode` to one
backend's query fragment. A compiler is a stateless
``Callable[[FilterNode, CompileContext], CompiledFilter]`` registered against an
``(engine_id, storage_target)`` key on the
:class:`~khora.filter.registry.CompilerRegistry` at engine import time.

``@internal``. Re-exported under :mod:`khora.filter.compilers` only — not from
:mod:`khora.__init__`.
"""

from __future__ import annotations

from khora.filter.compilers.chronicle import ChronicleDateBound, compile_chronicle
from khora.filter.compilers.postgres import compile_postgres
from khora.filter.compilers.python import compile_python

__all__ = [
    "ChronicleDateBound",
    "compile_chronicle",
    "compile_postgres",
    "compile_python",
]
