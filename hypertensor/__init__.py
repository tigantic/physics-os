"""Backward-compatibility shim: ``hypertensor`` → ``physics_os``.

.. deprecated:: 40.1.0
    The ``hypertensor`` package has been renamed to ``physics_os``.
    Update your imports::

        # Old
        from ontic.core.registry import DomainRegistry
        # New
        from physics_os.core.registry import DomainRegistry

    This shim will be removed in a future major release.
"""
from __future__ import annotations

import importlib
import sys
import warnings

warnings.warn(
    "The 'hypertensor' package has been renamed to 'physics_os'. "
    "Please update your imports: 'from physics_os import ...' "
    "This compatibility shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

# Import the real package so ``import hypertensor`` exposes the same API.
import physics_os as _real  # noqa: E402

# Re-export everything from the canonical package.
from physics_os import *  # noqa: F401, F403

# Dunder attributes are not re-exported by wildcard imports.
__version__ = _real.__version__


class _HypertensorShimFinder:
    """Meta-path finder that redirects ``hypertensor.*`` to ``physics_os.*``."""

    def find_module(
        self, fullname: str, path: object = None
    ) -> "_HypertensorShimFinder | None":
        if fullname == "hypertensor" or fullname.startswith("hypertensor."):
            canonical = "physics_os" + fullname[len("hypertensor"):]
            if canonical not in sys.modules:
                try:
                    importlib.import_module(canonical)
                except ImportError:
                    return None
            return self
        return None

    def load_module(self, fullname: str) -> object:
        canonical = "physics_os" + fullname[len("hypertensor"):]
        mod = importlib.import_module(canonical)
        sys.modules[fullname] = mod
        return mod


sys.meta_path.insert(0, _HypertensorShimFinder())
