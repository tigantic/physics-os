"""Backward-compatibility shim — real module at ontic.ml.neural.

This shim exists so that legacy imports like::

    from ontic.neural import X
    from ontic.neural.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.ml.neural``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.ml.neural")
_sys.modules[__name__] = _real
