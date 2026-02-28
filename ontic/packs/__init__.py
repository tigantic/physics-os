"""
HyperTensor Domain Packs — Physics Plugins (I–XX)
===================================================

Each domain pack is a cohesive bundle of equations, discretizations, solvers,
benchmarks, and documentation for one taxonomy category.

Auto-discovery
--------------
Call ``discover_all()`` to import every pack module and register them with the
global ``DomainRegistry``.  Individual packs can also be imported directly::

    from ontic.packs.pack_ii import FluidDynamicsPack

Phase 3 anchors (V0.4 Validated)
---------------------------------
- Pack II:   Fluid Dynamics       — 1-D viscous Burgers (Cole-Hopf benchmark)
- Pack III:  Electromagnetism     — 1-D FDTD Maxwell (Gaussian pulse)
- Pack V:    Thermo / Stat Mech   — 1-D advection-diffusion (exact)
- Pack VII:  Quantum Many-Body    — Heisenberg spin chain (TEBD + ED)
- Pack VIII: Electronic Structure — 1-D Kohn-Sham SCF (soft Coulomb)
- Pack XI:   Plasma Physics       — 1-D Vlasov-Poisson (Landau damping)
"""

from __future__ import annotations

import importlib
import logging
from typing import List

logger = logging.getLogger(__name__)

_PACK_MODULES: List[str] = [
    "ontic.packs.pack_i",
    "ontic.packs.pack_ii",
    "ontic.packs.pack_iii",
    "ontic.packs.pack_iv",
    "ontic.packs.pack_v",
    "ontic.packs.pack_vi",
    "ontic.packs.pack_vii",
    "ontic.packs.pack_viii",
    "ontic.packs.pack_ix",
    "ontic.packs.pack_x",
    "ontic.packs.pack_xi",
    "ontic.packs.pack_xii",
    "ontic.packs.pack_xiii",
    "ontic.packs.pack_xiv",
    "ontic.packs.pack_xv",
    "ontic.packs.pack_xvi",
    "ontic.packs.pack_xvii",
    "ontic.packs.pack_xviii",
    "ontic.packs.pack_xix",
    "ontic.packs.pack_xx",
]


def discover_all() -> int:
    """
    Import every domain-pack module to trigger ``@DomainRegistry.register``
    decorators.  Returns the number of packs successfully imported.

    Safe to call multiple times — already-registered packs are skipped.
    """
    loaded = 0
    for mod_path in _PACK_MODULES:
        try:
            importlib.import_module(mod_path)
            loaded += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load %s: %s", mod_path, exc)
    return loaded
