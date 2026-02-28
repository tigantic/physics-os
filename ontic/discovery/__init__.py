"""Backward-compatibility shim — real module at ontic.ml.discovery.

This shim exists so that legacy imports like::

    from ontic.discovery import X
    from ontic.discovery.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.ml.discovery``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.ml.discovery")
_sys.modules[__name__] = _real
