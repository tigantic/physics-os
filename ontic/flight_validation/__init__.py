"""Backward-compatibility shim — real module at ontic.sim.flight_validation.

This shim exists so that legacy imports like::

    from ontic.flight_validation import X
    from ontic.flight_validation.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.sim.flight_validation``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.sim.flight_validation")
_sys.modules[__name__] = _real
