"""Backward-compatibility shim — real module at ontic.quantum.qm.

This shim exists so that legacy imports like::

    from ontic.qm import X
    from ontic.qm.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.quantum.qm``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.quantum.qm")
_sys.modules[__name__] = _real
