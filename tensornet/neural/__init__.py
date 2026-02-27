"""Backward-compatibility shim — real module at tensornet.ml.neural.

This shim exists so that legacy imports like::

    from tensornet.neural import X
    from tensornet.neural.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.ml.neural``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.ml.neural")
_sys.modules[__name__] = _real
