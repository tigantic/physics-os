"""Backward-compatibility shim — real module at tensornet.quantum.qm.

This shim exists so that legacy imports like::

    from tensornet.qm import X
    from tensornet.qm.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.quantum.qm``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.quantum.qm")
_sys.modules[__name__] = _real
