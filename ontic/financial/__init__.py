"""Backward-compatibility shim — real module at ontic.applied.financial.

This shim exists so that legacy imports like::

    from ontic.financial import X
    from ontic.financial.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.applied.financial``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.applied.financial")
_sys.modules[__name__] = _real
