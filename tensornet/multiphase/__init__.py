"""Backward-compatibility shim — real module at tensornet.fluids.multiphase.

This shim exists so that legacy imports like::

    from tensornet.multiphase import X
    from tensornet.multiphase.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.fluids.multiphase``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.fluids.multiphase")
_sys.modules[__name__] = _real
