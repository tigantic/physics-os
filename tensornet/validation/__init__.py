"""Backward-compatibility shim — real module at tensornet.sim.validation.

This shim exists so that legacy imports like::

    from tensornet.validation import X
    from tensornet.validation.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.sim.validation``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.sim.validation")
_sys.modules[__name__] = _real
