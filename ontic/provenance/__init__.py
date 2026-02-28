"""Backward-compatibility shim — real module at ontic.infra.provenance.

This shim exists so that legacy imports like::

    from ontic.provenance import X
    from ontic.provenance.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.infra.provenance``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.infra.provenance")
_sys.modules[__name__] = _real
