"""Backward-compatibility shim — real module at ontic.infra.site.

This shim exists so that legacy imports like::

    from ontic.site import X
    from ontic.site.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.infra.site``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.infra.site")
_sys.modules[__name__] = _real
