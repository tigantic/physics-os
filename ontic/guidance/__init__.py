"""Backward-compatibility shim — real module at ontic.aerospace.guidance.

This shim exists so that legacy imports like::

    from ontic.guidance import X
    from ontic.guidance.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.aerospace.guidance``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.aerospace.guidance")
_sys.modules[__name__] = _real
