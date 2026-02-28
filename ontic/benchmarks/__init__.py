"""Backward-compatibility shim — real module at ontic.sim.benchmarks.

This shim exists so that legacy imports like::

    from ontic.benchmarks import X
    from ontic.benchmarks.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.sim.benchmarks``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.sim.benchmarks")
_sys.modules[__name__] = _real
