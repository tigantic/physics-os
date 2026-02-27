"""Backward-compatibility shim — real module at tensornet.sim.visualization.

This shim exists so that legacy imports like::

    from tensornet.visualization import X
    from tensornet.visualization.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.sim.visualization``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.sim.visualization")
_sys.modules[__name__] = _real
