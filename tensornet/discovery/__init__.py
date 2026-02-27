"""Backward-compatibility shim — real module at tensornet.ml.discovery.

This shim exists so that legacy imports like::

    from tensornet.discovery import X
    from tensornet.discovery.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.ml.discovery``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.ml.discovery")
_sys.modules[__name__] = _real
