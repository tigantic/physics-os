"""Backward-compatibility shim — real module at ontic.aerospace.autonomy.

This shim exists so that legacy imports like::

    from ontic.autonomy import X
    from ontic.autonomy.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.aerospace.autonomy``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.aerospace.autonomy")
_sys.modules[__name__] = _real
