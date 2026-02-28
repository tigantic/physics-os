"""Backward-compatibility shim — real module at ontic.sim.simulation.

This shim exists so that legacy imports like::

    from ontic.simulation import X
    from ontic.simulation.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.sim.simulation``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.sim.simulation")
_sys.modules[__name__] = _real
