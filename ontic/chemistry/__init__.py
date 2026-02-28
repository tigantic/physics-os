"""Backward-compatibility shim — real module at ontic.life_sci.chemistry.

This shim exists so that legacy imports like::

    from ontic.chemistry import X
    from ontic.chemistry.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.life_sci.chemistry``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.life_sci.chemistry")
_sys.modules[__name__] = _real
