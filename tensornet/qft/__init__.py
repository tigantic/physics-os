"""Backward-compatibility shim — real module at tensornet.quantum.qft.

This shim exists so that legacy imports like::

    from tensornet.qft import X
    from tensornet.qft.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.quantum.qft``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.quantum.qft")
_sys.modules[__name__] = _real
