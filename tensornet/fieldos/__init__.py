"""Backward-compatibility shim — real module at tensornet.infra.fieldos.

This shim exists so that legacy imports like::

    from tensornet.fieldos import X
    from tensornet.fieldos.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.infra.fieldos``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.infra.fieldos")
_sys.modules[__name__] = _real
