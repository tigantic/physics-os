"""Backward-compatibility shim — real module at ontic.aerospace.defense.

This shim exists so that legacy imports like::

    from ontic.defense import X
    from ontic.defense.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.aerospace.defense``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.aerospace.defense")
_sys.modules[__name__] = _real
