"""Backward-compatibility shim — real module at tensornet.infra.digital_twin.

This shim exists so that legacy imports like::

    from tensornet.digital_twin import X
    from tensornet.digital_twin.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.infra.digital_twin``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.infra.digital_twin")
_sys.modules[__name__] = _real
