"""Backward-compatibility shim — real module at ontic.infra.coordination.

This shim exists so that legacy imports like::

    from ontic.coordination import X
    from ontic.coordination.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.infra.coordination``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.infra.coordination")
_sys.modules[__name__] = _real
