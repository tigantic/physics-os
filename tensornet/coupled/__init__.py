"""Backward-compatibility shim — real module at tensornet.fluids.coupled.

This shim exists so that legacy imports like::

    from tensornet.coupled import X
    from tensornet.coupled.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.fluids.coupled``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.fluids.coupled")
_sys.modules[__name__] = _real
