"""Backward-compatibility shim — real module at ontic.applied.medical.

This shim exists so that legacy imports like::

    from ontic.medical import X
    from ontic.medical.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.applied.medical``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.applied.medical")
_sys.modules[__name__] = _real
