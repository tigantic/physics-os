"""Backward-compatibility shim — real module at tensornet.life_sci.biology.

This shim exists so that legacy imports like::

    from tensornet.biology import X
    from tensornet.biology.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.life_sci.biology``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.life_sci.biology")
_sys.modules[__name__] = _real
