"""Backward-compatibility shim — real module at ontic.infra.hypersim.

This shim exists so that legacy imports like::

    from ontic.hypersim import X
    from ontic.hypersim.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.infra.hypersim``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.infra.hypersim")
_sys.modules[__name__] = _real
