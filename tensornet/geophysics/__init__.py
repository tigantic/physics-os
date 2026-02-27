"""Backward-compatibility shim — real module at tensornet.astro.geophysics.

This shim exists so that legacy imports like::

    from tensornet.geophysics import X
    from tensornet.geophysics.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.astro.geophysics``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.astro.geophysics")
_sys.modules[__name__] = _real
