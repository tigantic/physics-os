"""
HyperTensor Domain Packs — Physics Plugins (I–XX)
===================================================

Each domain pack is a cohesive bundle of equations, discretizations, solvers,
benchmarks, and documentation for one taxonomy category.

Auto-discovery
--------------
Call ``discover_all()`` to import every pack module and register them with the
global ``DomainRegistry``.  Individual packs can also be imported directly::

    from tensornet.packs.pack_ii import FluidDynamicsPack

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
    "tensornet.packs.pack_i",
    "tensornet.packs.pack_ii",
    "tensornet.packs.pack_iii",
    "tensornet.packs.pack_iv",
    "tensornet.packs.pack_v",
    "tensornet.packs.pack_vi",
    "tensornet.packs.pack_vii",
    "tensornet.packs.pack_viii",
    "tensornet.packs.pack_ix",
    "tensornet.packs.pack_x",
    "tensornet.packs.pack_xi",
    "tensornet.packs.pack_xii",
    "tensornet.packs.pack_xiii",
    "tensornet.packs.pack_xiv",
    "tensornet.packs.pack_xv",
    "tensornet.packs.pack_xvi",
    "tensornet.packs.pack_xvii",
    "tensornet.packs.pack_xviii",
    "tensornet.packs.pack_xix",
    "tensornet.packs.pack_xx",
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
