"""Backward-compatibility shim — real module at ontic.quantum.electronic_structure.

This shim exists so that legacy imports like::

    from ontic.electronic_structure import X
    from ontic.electronic_structure.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.quantum.electronic_structure``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.quantum.electronic_structure")
_sys.modules[__name__] = _real
