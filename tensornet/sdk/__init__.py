"""Backward-compatibility shim — real module at tensornet.infra.sdk.

This shim exists so that legacy imports like::

    from tensornet.sdk import X
    from tensornet.sdk.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.infra.sdk``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.infra.sdk")
_sys.modules[__name__] = _real
