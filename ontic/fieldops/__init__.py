"""Backward-compatibility shim — real module at ontic.infra.fieldops.

This shim exists so that legacy imports like::

    from ontic.fieldops import X
    from ontic.fieldops.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.infra.fieldops``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.infra.fieldops")
_sys.modules[__name__] = _real
