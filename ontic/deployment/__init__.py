"""Backward-compatibility shim — real module at ontic.infra.deployment.

This shim exists so that legacy imports like::

    from ontic.deployment import X
    from ontic.deployment.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.infra.deployment``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.infra.deployment")
_sys.modules[__name__] = _real
