"""Backward-compatibility shim — real module at tensornet.infra.hyperenv.

This shim exists so that legacy imports like::

    from tensornet.hyperenv import X
    from tensornet.hyperenv.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.infra.hyperenv``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.infra.hyperenv")
_sys.modules[__name__] = _real
