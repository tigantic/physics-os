"""Backward-compatibility shim — real module at ontic.applied.emergency.

This shim exists so that legacy imports like::

    from ontic.emergency import X
    from ontic.emergency.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.applied.emergency``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.applied.emergency")
_sys.modules[__name__] = _real
