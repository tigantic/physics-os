"""Backward-compatibility shim — real module at tensornet.fluids.mesh_amr.

This shim exists so that legacy imports like::

    from tensornet.mesh_amr import X
    from tensornet.mesh_amr.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.fluids.mesh_amr``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.fluids.mesh_amr")
_sys.modules[__name__] = _real
