"""Backward-compatibility shim — real module at ontic.fluids.porous_media.

This shim exists so that legacy imports like::

    from ontic.porous_media import X
    from ontic.porous_media.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.fluids.porous_media``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.fluids.porous_media")
_sys.modules[__name__] = _real
