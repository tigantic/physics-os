"""Backward-compatibility shim — real module at tensornet.astro.relativity.

This shim exists so that legacy imports like::

    from tensornet.relativity import X
    from tensornet.relativity.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.astro.relativity``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.astro.relativity")
_sys.modules[__name__] = _real
