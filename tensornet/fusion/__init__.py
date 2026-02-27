"""Backward-compatibility shim — real module at tensornet.plasma_nuclear.fusion.

This shim exists so that legacy imports like::

    from tensornet.fusion import X
    from tensornet.fusion.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.plasma_nuclear.fusion``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.plasma_nuclear.fusion")
_sys.modules[__name__] = _real
