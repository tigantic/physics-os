"""Backward-compatibility shim — real module at tensornet.fluids.fsi.

This shim exists so that legacy imports like::

    from tensornet.fsi import X
    from tensornet.fsi.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.fluids.fsi``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.fluids.fsi")
_sys.modules[__name__] = _real
