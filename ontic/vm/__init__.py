"""Backward-compatibility shim — real module at ontic.engine.vm.

This shim exists so that legacy imports like::

    from ontic.vm import X
    from ontic.vm.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.engine.vm``.

The shim also registers submodule aliases so that relative imports
inside compilers loaded via ``ontic.vm.compilers.*`` resolve to the
same canonical ``ontic.engine.vm.*`` module objects, preventing
duplicate enum class identity issues.
"""
import importlib as _il
import pkgutil as _pk
import sys as _sys

_SHIM_PREFIX = "ontic.vm"
_REAL_PREFIX = "ontic.engine.vm"

_real = _il.import_module(_REAL_PREFIX)
_sys.modules[_SHIM_PREFIX] = _real


# Walk every submodule already loaded under the canonical prefix and
# create a matching alias under the shim prefix so that relative imports
# (e.g. ``from ..ir import OpCode`` inside a compiler loaded as
# ``ontic.vm.compilers.navier_stokes``) see the *same* module object.
def _alias_submodules() -> None:
    for key, mod in list(_sys.modules.items()):
        if key.startswith(_REAL_PREFIX + "."):
            alias = _SHIM_PREFIX + key[len(_REAL_PREFIX):]
            _sys.modules.setdefault(alias, mod)


_alias_submodules()


# Also install an import hook that aliases any *future* submodule loads
# under the shim prefix to the canonical prefix, keeping the enum
# identity invariant intact even for lazily-loaded subpackages.
class _ShimFinder:
    """Redirect ``ontic.vm.*`` imports to ``ontic.engine.vm.*``."""

    def find_module(self, fullname: str, path: object = None) -> "_ShimFinder | None":
        if fullname.startswith(_SHIM_PREFIX + "."):
            return self
        return None

    def load_module(self, fullname: str) -> object:  # type: ignore[override]
        canonical = _REAL_PREFIX + fullname[len(_SHIM_PREFIX):]
        mod = _il.import_module(canonical)
        _sys.modules[fullname] = mod
        return mod


_sys.meta_path.insert(0, _ShimFinder())
