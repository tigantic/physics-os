"""Backward-compatibility shim — real module at ontic.infra.oracle.

This shim exists so that legacy imports like::

    from ontic.oracle import X
    from ontic.oracle.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.infra.oracle``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.infra.oracle")
_sys.modules[__name__] = _real
