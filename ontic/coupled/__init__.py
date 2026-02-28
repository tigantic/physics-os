"""Backward-compatibility shim — real module at ontic.fluids.coupled.

This shim exists so that legacy imports like::

    from ontic.coupled import X
    from ontic.coupled.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.fluids.coupled``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.fluids.coupled")
_sys.modules[__name__] = _real
