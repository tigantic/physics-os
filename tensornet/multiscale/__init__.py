"""Backward-compatibility shim — real module at tensornet.fluids.multiscale.

This shim exists so that legacy imports like::

    from tensornet.multiscale import X
    from tensornet.multiscale.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.fluids.multiscale``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.fluids.multiscale")
_sys.modules[__name__] = _real
