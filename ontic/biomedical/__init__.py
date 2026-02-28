"""Backward-compatibility shim — real module at ontic.life_sci.biomedical.

This shim exists so that legacy imports like::

    from ontic.biomedical import X
    from ontic.biomedical.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.life_sci.biomedical``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.life_sci.biomedical")
_sys.modules[__name__] = _real
