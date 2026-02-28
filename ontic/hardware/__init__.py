"""Backward-compatibility shim — real module at ontic.engine.hardware.

This shim exists so that legacy imports like::

    from ontic.hardware import X
    from ontic.hardware.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.engine.hardware``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.engine.hardware")
_sys.modules[__name__] = _real
