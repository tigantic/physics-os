"""Backward-compatibility shim — real module at tensornet.fluids.porous_media.

This shim exists so that legacy imports like::

    from tensornet.porous_media import X
    from tensornet.porous_media.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.fluids.porous_media``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.fluids.porous_media")
_sys.modules[__name__] = _real
