"""Backward-compatibility shim — real module at ontic.life_sci.biology.

This shim exists so that legacy imports like::

    from ontic.biology import X
    from ontic.biology.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.life_sci.biology``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.life_sci.biology")
_sys.modules[__name__] = _real
