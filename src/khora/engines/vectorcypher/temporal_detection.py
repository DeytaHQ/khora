"""Backward-compatible re-export shim.

The canonical location is now ``khora.query.temporal_detection``.
This module re-exports everything so existing imports continue to work.
"""

from khora.query.temporal_detection import *  # noqa: F401,F403
from khora.query.temporal_detection import __all__  # noqa: F401
