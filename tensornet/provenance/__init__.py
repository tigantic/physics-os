"""Backward-compatibility shim — real module at tensornet.infra.provenance.

This shim exists so that legacy imports like::

    from tensornet.provenance import X
    from tensornet.provenance.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.infra.provenance``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.infra.provenance")
_sys.modules[__name__] = _real
