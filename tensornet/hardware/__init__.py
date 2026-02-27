"""Backward-compatibility shim — real module at tensornet.engine.hardware.

This shim exists so that legacy imports like::

    from tensornet.hardware import X
    from tensornet.hardware.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.engine.hardware``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.engine.hardware")
_sys.modules[__name__] = _real
