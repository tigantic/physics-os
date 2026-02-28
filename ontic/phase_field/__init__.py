"""Backward-compatibility shim — real module at ontic.fluids.phase_field.

This shim exists so that legacy imports like::

    from ontic.phase_field import X
    from ontic.phase_field.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.fluids.phase_field``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.fluids.phase_field")
_sys.modules[__name__] = _real
