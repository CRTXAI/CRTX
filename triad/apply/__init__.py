"""Apply mode — safe file writing and intelligent patching.

Applies pipeline-generated code to disk with git safety gates,
interactive diff preview, post-apply test verification, and
optional rollback.
"""

from triad.apply.engine import ApplyEngine

__all__ = ["ApplyEngine"]
