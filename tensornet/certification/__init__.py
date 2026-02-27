"""Backward-compatibility shim — real module at tensornet.sim.certification.

This shim exists so that legacy imports like::

    from tensornet.certification import X
    from tensornet.certification.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.sim.certification``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.sim.certification")
_sys.modules[__name__] = _real
