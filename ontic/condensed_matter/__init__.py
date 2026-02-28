"""Backward-compatibility shim — real module at ontic.quantum.condensed_matter.

This shim exists so that legacy imports like::

    from ontic.condensed_matter import X
    from ontic.condensed_matter.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.quantum.condensed_matter``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.quantum.condensed_matter")
_sys.modules[__name__] = _real
