"""Backward-compatibility shim — real module at tensornet.fluids.phase_field.

This shim exists so that legacy imports like::

    from tensornet.phase_field import X
    from tensornet.phase_field.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.fluids.phase_field``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.fluids.phase_field")
_sys.modules[__name__] = _real
