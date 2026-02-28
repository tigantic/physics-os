"""Backward-compatibility shim — real module at ontic.sim.visualization.

This shim exists so that legacy imports like::

    from ontic.visualization import X
    from ontic.visualization.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.sim.visualization``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.sim.visualization")
_sys.modules[__name__] = _real
