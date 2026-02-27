"""Backward-compatibility shim — real module at tensornet.engine.vm.

This shim exists so that legacy imports like::

    from tensornet.vm import X
    from tensornet.vm.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.engine.vm``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.engine.vm")
_sys.modules[__name__] = _real
