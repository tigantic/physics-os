"""Backward-compatibility shim — real module at tensornet.energy_env.agri.

This shim exists so that legacy imports like::

    from tensornet.agri import X
    from tensornet.agri.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.energy_env.agri``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.energy_env.agri")
_sys.modules[__name__] = _real
