"""Backward-compatibility shim — real module at ontic.infra.integration.

This shim exists so that legacy imports like::

    from ontic.integration import X
    from ontic.integration.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.infra.integration``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.infra.integration")
_sys.modules[__name__] = _real
