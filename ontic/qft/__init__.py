"""Backward-compatibility shim — real module at ontic.quantum.qft.

This shim exists so that legacy imports like::

    from ontic.qft import X
    from ontic.qft.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.quantum.qft``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.quantum.qft")
_sys.modules[__name__] = _real
