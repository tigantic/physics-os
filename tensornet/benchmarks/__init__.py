"""Backward-compatibility shim — real module at tensornet.sim.benchmarks.

This shim exists so that legacy imports like::

    from tensornet.benchmarks import X
    from tensornet.benchmarks.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.sim.benchmarks``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.sim.benchmarks")
_sys.modules[__name__] = _real
