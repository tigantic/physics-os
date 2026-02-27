"""Backward-compatibility shim — real module at tensornet.infra.zk.

This shim exists so that legacy imports like::

    from tensornet.zk import X
    from tensornet.zk.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.infra.zk``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.infra.zk")
_sys.modules[__name__] = _real
