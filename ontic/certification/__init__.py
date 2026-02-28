"""Backward-compatibility shim — real module at ontic.sim.certification.

This shim exists so that legacy imports like::

    from ontic.certification import X
    from ontic.certification.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.sim.certification``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.sim.certification")
_sys.modules[__name__] = _real
