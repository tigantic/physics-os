"""Backward-compatibility shim — real module at ontic.astro.relativity.

This shim exists so that legacy imports like::

    from ontic.relativity import X
    from ontic.relativity.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.astro.relativity``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.astro.relativity")
_sys.modules[__name__] = _real
