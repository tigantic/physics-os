"""Backward-compatibility shim — real module at tensornet.sim.simulation.

This shim exists so that legacy imports like::

    from tensornet.simulation import X
    from tensornet.simulation.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.sim.simulation``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.sim.simulation")
_sys.modules[__name__] = _real
