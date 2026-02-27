"""Backward-compatibility shim — real module at tensornet.ml.data.

This shim exists so that legacy imports like::

    from tensornet.data import X
    from tensornet.data.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.ml.data``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.ml.data")
_sys.modules[__name__] = _real
