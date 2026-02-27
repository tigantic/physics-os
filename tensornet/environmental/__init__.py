"""Backward-compatibility shim — real module at tensornet.energy_env.environmental.

This shim exists so that legacy imports like::

    from tensornet.environmental import X
    from tensornet.environmental.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.energy_env.environmental``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.energy_env.environmental")
_sys.modules[__name__] = _real
