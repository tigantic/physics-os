"""Backward-compatibility shim — real module at ontic.infra.sdk.

This shim exists so that legacy imports like::

    from ontic.sdk import X
    from ontic.sdk.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.infra.sdk``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.infra.sdk")
_sys.modules[__name__] = _real
