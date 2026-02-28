"""Backward-compatibility shim — real module at ontic.applied.optics.

This shim exists so that legacy imports like::

    from ontic.optics import X
    from ontic.optics.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.applied.optics``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.applied.optics")
_sys.modules[__name__] = _real
