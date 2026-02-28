"""Backward-compatibility shim — real module at ontic.ml.data.

This shim exists so that legacy imports like::

    from ontic.data import X
    from ontic.data.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.ml.data``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.ml.data")
_sys.modules[__name__] = _real
