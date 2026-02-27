"""Backward-compatibility shim — real module at tensornet.infra.site.

This shim exists so that legacy imports like::

    from tensornet.site import X
    from tensornet.site.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.infra.site``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.infra.site")
_sys.modules[__name__] = _real
