"""Backward-compatibility shim — real module at tensornet.quantum.quantum_mechanics.

This shim exists so that legacy imports like::

    from tensornet.quantum_mechanics import X
    from tensornet.quantum_mechanics.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.quantum.quantum_mechanics``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.quantum.quantum_mechanics")
_sys.modules[__name__] = _real
