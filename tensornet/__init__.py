"""Backward-compatibility shim: ``tensornet`` → ``ontic``.

.. deprecated:: 40.1.0
    The ``tensornet`` package has been renamed to ``ontic``.
    Update your imports::

        # Old
        from tensornet.core import TensorNetwork
        # New
        from ontic.core import TensorNetwork

    This shim will be removed in a future major release.
"""
from __future__ import annotations

import importlib
import sys
import warnings

warnings.warn(
    "The 'tensornet' package has been renamed to 'ontic'. "
    "Please update your imports: 'from ontic import ...' "
    "This compatibility shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

# Import the real package so ``import tensornet`` exposes the same API.
import ontic as _real  # noqa: E402

# Re-export everything from the canonical package.
from ontic import *  # noqa: F401, F403

# Dunder attributes are not re-exported by wildcard imports.
__version__ = _real.__version__
__author__ = _real.__author__


class _TensornetShimFinder:
    """Meta-path finder that redirects ``tensornet.*`` to ``ontic.*``."""

    def find_module(
        self, fullname: str, path: object = None
    ) -> "_TensornetShimFinder | None":
        if fullname == "tensornet" or fullname.startswith("tensornet."):
            canonical = "ontic" + fullname[len("tensornet"):]
            if canonical not in sys.modules:
                try:
                    importlib.import_module(canonical)
                except ImportError:
                    return None
            return self
        return None

    def load_module(self, fullname: str) -> object:
        canonical = "ontic" + fullname[len("tensornet"):]
        mod = importlib.import_module(canonical)
        sys.modules[fullname] = mod
        return mod


sys.meta_path.insert(0, _TensornetShimFinder())
