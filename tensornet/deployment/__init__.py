"""Backward-compatibility shim — real module at tensornet.infra.deployment.

This shim exists so that legacy imports like::

    from tensornet.deployment import X
    from tensornet.deployment.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.infra.deployment``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.infra.deployment")
_sys.modules[__name__] = _real
