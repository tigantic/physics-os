"""Backward-compatibility shim — real module at ontic.infra.sovereign.

This shim exists so that legacy imports like::

    from ontic.sovereign import X
    from ontic.sovereign.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.infra.sovereign``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.infra.sovereign")
_sys.modules[__name__] = _real
