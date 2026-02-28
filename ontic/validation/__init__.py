"""Backward-compatibility shim — real module at ontic.sim.validation.

This shim exists so that legacy imports like::

    from ontic.validation import X
    from ontic.validation.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.sim.validation``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.sim.validation")
_sys.modules[__name__] = _real
