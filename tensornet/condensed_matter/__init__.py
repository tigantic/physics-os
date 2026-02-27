"""Backward-compatibility shim — real module at tensornet.quantum.condensed_matter.

This shim exists so that legacy imports like::

    from tensornet.condensed_matter import X
    from tensornet.condensed_matter.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.quantum.condensed_matter``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.quantum.condensed_matter")
_sys.modules[__name__] = _real
