"""Backward-compatibility shim — real module at tensornet.infra.hypervisual.

This shim exists so that legacy imports like::

    from tensornet.hypervisual import X
    from tensornet.hypervisual.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.infra.hypervisual``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.infra.hypervisual")
_sys.modules[__name__] = _real
