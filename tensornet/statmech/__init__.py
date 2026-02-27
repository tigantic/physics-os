"""Backward-compatibility shim — real module at tensornet.quantum.statmech.

This shim exists so that legacy imports like::

    from tensornet.statmech import X
    from tensornet.statmech.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.quantum.statmech``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.quantum.statmech")
_sys.modules[__name__] = _real
