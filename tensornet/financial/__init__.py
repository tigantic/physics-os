"""Backward-compatibility shim — real module at tensornet.applied.financial.

This shim exists so that legacy imports like::

    from tensornet.financial import X
    from tensornet.financial.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.applied.financial``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.applied.financial")
_sys.modules[__name__] = _real
