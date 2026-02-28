"""Backward-compatibility shim — real module at ontic.fluids.mesh_amr.

This shim exists so that legacy imports like::

    from ontic.mesh_amr import X
    from ontic.mesh_amr.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.fluids.mesh_amr``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.fluids.mesh_amr")
_sys.modules[__name__] = _real
