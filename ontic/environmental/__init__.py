"""Backward-compatibility shim — real module at ontic.energy_env.environmental.

This shim exists so that legacy imports like::

    from ontic.environmental import X
    from ontic.environmental.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.energy_env.environmental``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.energy_env.environmental")
_sys.modules[__name__] = _real
