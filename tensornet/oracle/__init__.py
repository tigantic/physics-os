"""Backward-compatibility shim — real module at tensornet.infra.oracle.

This shim exists so that legacy imports like::

    from tensornet.oracle import X
    from tensornet.oracle.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.infra.oracle``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.infra.oracle")
_sys.modules[__name__] = _real
