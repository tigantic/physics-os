"""Backward-compatibility shim — real module at ontic.quantum.statmech.

This shim exists so that legacy imports like::

    from ontic.statmech import X
    from ontic.statmech.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.quantum.statmech``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.quantum.statmech")
_sys.modules[__name__] = _real
