"""Backward-compatibility shim — real module at tensornet.plasma_nuclear.plasma.

This shim exists so that legacy imports like::

    from tensornet.plasma import X
    from tensornet.plasma.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.plasma_nuclear.plasma``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.plasma_nuclear.plasma")
_sys.modules[__name__] = _real
