"""Backward-compatibility shim — real module at ontic.infra.zk.

This shim exists so that legacy imports like::

    from ontic.zk import X
    from ontic.zk.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.infra.zk``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.infra.zk")
_sys.modules[__name__] = _real
