"""Backward-compatibility shim — real module at ontic.plasma_nuclear.nuclear.

This shim exists so that legacy imports like::

    from ontic.nuclear import X
    from ontic.nuclear.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.plasma_nuclear.nuclear``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.plasma_nuclear.nuclear")
_sys.modules[__name__] = _real
