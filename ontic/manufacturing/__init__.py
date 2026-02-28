"""Backward-compatibility shim — real module at ontic.materials.manufacturing.

This shim exists so that legacy imports like::

    from ontic.manufacturing import X
    from ontic.manufacturing.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.materials.manufacturing``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.materials.manufacturing")
_sys.modules[__name__] = _real
