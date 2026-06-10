"""Deterministic end-to-end recall-filter row-set suite.

Complements the wiring spies (which prove the validated filter AST reaches
each channel unchanged) by proving the filter actually narrows the rows end
to end, through the real ``Khora.remember()`` ingest pipeline and
``Khora.recall(filter=...)`` read path, with a populated graph. See
``DESIGN_NOTES.md``.
"""
