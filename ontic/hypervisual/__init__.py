"""Backward-compatibility shim — real module at ontic.infra.hypervisual.

This shim exists so that legacy imports like::

    from ontic.hypervisual import X
    from ontic.hypervisual.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.infra.hypervisual``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.infra.hypervisual")
_sys.modules[__name__] = _real
