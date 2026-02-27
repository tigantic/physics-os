"""Backward-compatibility shim — real module at tensornet.ml.ml_surrogates.

This shim exists so that legacy imports like::

    from tensornet.ml_surrogates import X
    from tensornet.ml_surrogates.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.ml.ml_surrogates``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.ml.ml_surrogates")
_sys.modules[__name__] = _real
