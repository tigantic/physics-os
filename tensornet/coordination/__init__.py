"""Backward-compatibility shim — real module at tensornet.infra.coordination.

This shim exists so that legacy imports like::

    from tensornet.coordination import X
    from tensornet.coordination.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.infra.coordination``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.infra.coordination")
_sys.modules[__name__] = _real
