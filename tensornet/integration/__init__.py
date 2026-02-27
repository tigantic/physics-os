"""Backward-compatibility shim — real module at tensornet.infra.integration.

This shim exists so that legacy imports like::

    from tensornet.integration import X
    from tensornet.integration.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.infra.integration``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.infra.integration")
_sys.modules[__name__] = _real
