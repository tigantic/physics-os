"""Backward-compatibility shim — real module at ontic.infra.hyperenv.

This shim exists so that legacy imports like::

    from ontic.hyperenv import X
    from ontic.hyperenv.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.infra.hyperenv``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.infra.hyperenv")
_sys.modules[__name__] = _real
