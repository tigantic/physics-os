"""Backward-compatibility shim — real module at tensornet.life_sci.biomedical.

This shim exists so that legacy imports like::

    from tensornet.biomedical import X
    from tensornet.biomedical.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.life_sci.biomedical``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.life_sci.biomedical")
_sys.modules[__name__] = _real
