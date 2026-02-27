"""Backward-compatibility shim — real module at tensornet.plasma_nuclear.nuclear.

This shim exists so that legacy imports like::

    from tensornet.nuclear import X
    from tensornet.nuclear.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.plasma_nuclear.nuclear``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.plasma_nuclear.nuclear")
_sys.modules[__name__] = _real
