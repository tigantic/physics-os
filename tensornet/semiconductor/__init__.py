"""Backward-compatibility shim — real module at tensornet.quantum.semiconductor.

This shim exists so that legacy imports like::

    from tensornet.semiconductor import X
    from tensornet.semiconductor.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.quantum.semiconductor``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.quantum.semiconductor")
_sys.modules[__name__] = _real
