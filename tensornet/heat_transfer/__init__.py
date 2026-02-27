"""Backward-compatibility shim — real module at tensornet.fluids.heat_transfer.

This shim exists so that legacy imports like::

    from tensornet.heat_transfer import X
    from tensornet.heat_transfer.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.fluids.heat_transfer``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.fluids.heat_transfer")
_sys.modules[__name__] = _real
