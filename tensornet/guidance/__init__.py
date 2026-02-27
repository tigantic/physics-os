"""Backward-compatibility shim — real module at tensornet.aerospace.guidance.

This shim exists so that legacy imports like::

    from tensornet.guidance import X
    from tensornet.guidance.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.aerospace.guidance``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.aerospace.guidance")
_sys.modules[__name__] = _real
