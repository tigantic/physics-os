"""Backward-compatibility shim — real module at ontic.infra.fieldos.

This shim exists so that legacy imports like::

    from ontic.fieldos import X
    from ontic.fieldos.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.infra.fieldos``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.infra.fieldos")
_sys.modules[__name__] = _real
