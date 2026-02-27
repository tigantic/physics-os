"""Backward-compatibility shim — real module at tensornet.infra.sovereign.

This shim exists so that legacy imports like::

    from tensornet.sovereign import X
    from tensornet.sovereign.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.infra.sovereign``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.infra.sovereign")
_sys.modules[__name__] = _real
