"""Backward-compatibility shim — real module at tensornet.materials.manufacturing.

This shim exists so that legacy imports like::

    from tensornet.manufacturing import X
    from tensornet.manufacturing.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.materials.manufacturing``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.materials.manufacturing")
_sys.modules[__name__] = _real
