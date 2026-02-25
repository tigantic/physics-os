"""QTT Physics VM — RF material library.

Provides catalogued RF substrates and conductors with physical properties
for antenna design automation.  Each material carries its permittivity,
loss tangent, conductivity, and (where applicable) frequency-dependent
Debye model parameters.

All values are in SI units.  The antenna compiler normalises internally.

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Material:
    """Radio-frequency material specification.

    Parameters
    ----------
    name : str
        Human-readable identifier (e.g. ``"FR-4"``, ``"Rogers RO4003C"``).
    eps_r : float
        Relative permittivity at the reference frequency.
    loss_tangent : float
        Dielectric loss tangent (tan δ) at the reference frequency.
    conductivity : float
        Bulk electrical conductivity σ (S/m).  0 for lossless dielectrics,
        5.8e7 for copper, etc.
    mu_r : float
        Relative permeability (typically 1.0 for RF substrates).
    thickness_m : float
        Standard thickness in metres (used as a default in design).
    ref_freq_hz : float
        Frequency at which eps_r and loss_tangent are specified.
    vendor : str
        Manufacturer or standards body.
    category : str
        ``"substrate"``, ``"conductor"``, or ``"superstrate"``.
    debye_eps_inf : float | None
        High-frequency permittivity for Debye model (optional).
    debye_eps_s : float | None
        Static permittivity for Debye model (optional).
    debye_tau : float | None
        Relaxation time constant (s) for Debye model (optional).
    min_feature_um : float
        Minimum etchable feature size in µm (manufacturability metric).
    cost_per_m2 : float
        Approximate material cost per m² (USD, for scoring).
    """

    name: str
    eps_r: float
    loss_tangent: float = 0.0
    conductivity: float = 0.0
    mu_r: float = 1.0
    thickness_m: float = 1.6e-3
    ref_freq_hz: float = 1.0e9
    vendor: str = ""
    category: str = "substrate"
    debye_eps_inf: float | None = None
    debye_eps_s: float | None = None
    debye_tau: float | None = None
    min_feature_um: float = 100.0
    cost_per_m2: float = 50.0

    def eps_r_at(self, freq_hz: float) -> complex:
        """Complex permittivity at a given frequency.

        If Debye model parameters are available, uses single-pole Debye:
            ε(f) = ε_∞ + (ε_s − ε_∞) / (1 + j 2π f τ)

        Otherwise returns the catalog value with loss tangent:
            ε(f) = ε_r (1 − j tan δ)
        """
        if (
            self.debye_eps_inf is not None
            and self.debye_eps_s is not None
            and self.debye_tau is not None
        ):
            omega = 2.0 * math.pi * freq_hz
            denom = 1.0 + 1j * omega * self.debye_tau
            return self.debye_eps_inf + (
                self.debye_eps_s - self.debye_eps_inf
            ) / denom

        return complex(self.eps_r, -self.eps_r * self.loss_tangent)

    def skin_depth_m(self, freq_hz: float) -> float:
        """Skin depth δ = 1 / √(π f μ σ) for conductors.

        Returns ``float('inf')`` for dielectrics (σ ≈ 0).
        """
        if self.conductivity < 1.0:
            return float("inf")
        mu_0 = 4.0e-7 * math.pi
        return 1.0 / math.sqrt(
            math.pi * freq_hz * mu_0 * self.mu_r * self.conductivity
        )

    def wavelength_m(self, freq_hz: float) -> float:
        """Wavelength in the material at a given frequency.

        λ = c₀ / (f √ε_r)
        """
        c0 = 299792458.0
        return c0 / (freq_hz * math.sqrt(self.eps_r))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON/YAML output."""
        return {
            "name": self.name,
            "eps_r": self.eps_r,
            "loss_tangent": self.loss_tangent,
            "conductivity": self.conductivity,
            "mu_r": self.mu_r,
            "thickness_m": self.thickness_m,
            "ref_freq_hz": self.ref_freq_hz,
            "vendor": self.vendor,
            "category": self.category,
            "min_feature_um": self.min_feature_um,
            "cost_per_m2": self.cost_per_m2,
        }


class MaterialLibrary:
    """Catalogued RF material database.

    Provides factory methods for standard substrates and conductors
    used in PCB antenna design.

    Usage::

        lib = MaterialLibrary()
        fr4 = lib.get("FR-4")
        ro4003c = lib.get("Rogers RO4003C")
        copper = lib.get("Copper")
    """

    def __init__(self) -> None:
        self._catalog: dict[str, Material] = {}
        self._populate_defaults()

    def _populate_defaults(self) -> None:
        """Load the built-in material catalog."""

        # ── Standard PCB substrates ──────────────────────────────────
        self._add(Material(
            name="FR-4",
            eps_r=4.4,
            loss_tangent=0.02,
            thickness_m=1.6e-3,
            ref_freq_hz=1.0e9,
            vendor="Generic",
            category="substrate",
            min_feature_um=150.0,
            cost_per_m2=15.0,
        ))

        self._add(Material(
            name="FR-4 Thin",
            eps_r=4.4,
            loss_tangent=0.02,
            thickness_m=0.8e-3,
            ref_freq_hz=1.0e9,
            vendor="Generic",
            category="substrate",
            min_feature_um=150.0,
            cost_per_m2=18.0,
        ))

        # ── Rogers high-frequency laminates ──────────────────────────
        self._add(Material(
            name="Rogers RO4003C",
            eps_r=3.55,
            loss_tangent=0.0027,
            thickness_m=0.508e-3,
            ref_freq_hz=10.0e9,
            vendor="Rogers Corporation",
            category="substrate",
            min_feature_um=75.0,
            cost_per_m2=120.0,
        ))

        self._add(Material(
            name="Rogers RO4350B",
            eps_r=3.48,
            loss_tangent=0.0037,
            thickness_m=0.508e-3,
            ref_freq_hz=10.0e9,
            vendor="Rogers Corporation",
            category="substrate",
            min_feature_um=75.0,
            cost_per_m2=130.0,
        ))

        self._add(Material(
            name="Rogers RT5880",
            eps_r=2.2,
            loss_tangent=0.0009,
            thickness_m=0.787e-3,
            ref_freq_hz=10.0e9,
            vendor="Rogers Corporation",
            category="substrate",
            min_feature_um=75.0,
            cost_per_m2=200.0,
        ))

        self._add(Material(
            name="Rogers RO3003",
            eps_r=3.0,
            loss_tangent=0.0013,
            thickness_m=0.508e-3,
            ref_freq_hz=10.0e9,
            vendor="Rogers Corporation",
            category="substrate",
            min_feature_um=75.0,
            cost_per_m2=180.0,
        ))

        self._add(Material(
            name="Rogers TMM10i",
            eps_r=9.8,
            loss_tangent=0.002,
            thickness_m=1.27e-3,
            ref_freq_hz=10.0e9,
            vendor="Rogers Corporation",
            category="substrate",
            min_feature_um=100.0,
            cost_per_m2=250.0,
        ))

        # ── Taconic substrates ───────────────────────────────────────
        self._add(Material(
            name="Taconic TLY-5",
            eps_r=2.2,
            loss_tangent=0.0009,
            thickness_m=0.508e-3,
            ref_freq_hz=10.0e9,
            vendor="Taconic",
            category="substrate",
            min_feature_um=75.0,
            cost_per_m2=190.0,
        ))

        # ── LTCC (Low-Temperature Co-fired Ceramic) ──────────────────
        self._add(Material(
            name="LTCC DuPont 951",
            eps_r=7.8,
            loss_tangent=0.006,
            thickness_m=0.1e-3,
            ref_freq_hz=10.0e9,
            vendor="DuPont",
            category="substrate",
            min_feature_um=50.0,
            cost_per_m2=500.0,
        ))

        # ── High-permittivity ceramics ───────────────────────────────
        self._add(Material(
            name="Alumina (Al2O3)",
            eps_r=9.9,
            loss_tangent=0.0001,
            thickness_m=0.635e-3,
            ref_freq_hz=10.0e9,
            vendor="Generic",
            category="substrate",
            min_feature_um=50.0,
            cost_per_m2=300.0,
        ))

        # ── mmWave substrates ────────────────────────────────────────
        self._add(Material(
            name="Rogers RO3006",
            eps_r=6.15,
            loss_tangent=0.002,
            thickness_m=0.254e-3,
            ref_freq_hz=28.0e9,
            vendor="Rogers Corporation",
            category="substrate",
            min_feature_um=50.0,
            cost_per_m2=220.0,
        ))

        self._add(Material(
            name="Isola Astra MT77",
            eps_r=3.0,
            loss_tangent=0.0017,
            thickness_m=0.254e-3,
            ref_freq_hz=28.0e9,
            vendor="Isola",
            category="substrate",
            min_feature_um=50.0,
            cost_per_m2=160.0,
        ))

        # ── Conductors ──────────────────────────────────────────────
        self._add(Material(
            name="Copper",
            eps_r=1.0,
            conductivity=5.8e7,
            thickness_m=35.0e-6,
            vendor="Generic",
            category="conductor",
            min_feature_um=75.0,
            cost_per_m2=5.0,
        ))

        self._add(Material(
            name="Silver",
            eps_r=1.0,
            conductivity=6.3e7,
            thickness_m=10.0e-6,
            vendor="Generic",
            category="conductor",
            min_feature_um=50.0,
            cost_per_m2=60.0,
        ))

        self._add(Material(
            name="Gold",
            eps_r=1.0,
            conductivity=4.1e7,
            thickness_m=5.0e-6,
            vendor="Generic",
            category="conductor",
            min_feature_um=25.0,
            cost_per_m2=200.0,
        ))

        self._add(Material(
            name="Aluminum",
            eps_r=1.0,
            conductivity=3.5e7,
            thickness_m=35.0e-6,
            vendor="Generic",
            category="conductor",
            min_feature_um=100.0,
            cost_per_m2=3.0,
        ))

        # ── Air / vacuum ─────────────────────────────────────────────
        self._add(Material(
            name="Air",
            eps_r=1.0,
            loss_tangent=0.0,
            conductivity=0.0,
            thickness_m=0.0,
            vendor="Physics",
            category="superstrate",
            min_feature_um=0.0,
            cost_per_m2=0.0,
        ))

    def _add(self, mat: Material) -> None:
        """Register a material in the catalog."""
        self._catalog[mat.name] = mat

    def get(self, name: str) -> Material:
        """Look up a material by name.

        Raises
        ------
        KeyError
            If the material is not in the catalog.
        """
        if name not in self._catalog:
            available = ", ".join(sorted(self._catalog.keys()))
            raise KeyError(
                f"Material '{name}' not in catalog. Available: {available}"
            )
        return self._catalog[name]

    def list_substrates(self) -> list[Material]:
        """Return all substrate materials, sorted by eps_r."""
        return sorted(
            [m for m in self._catalog.values() if m.category == "substrate"],
            key=lambda m: m.eps_r,
        )

    def list_conductors(self) -> list[Material]:
        """Return all conductor materials, sorted by conductivity."""
        return sorted(
            [m for m in self._catalog.values() if m.category == "conductor"],
            key=lambda m: m.conductivity,
            reverse=True,
        )

    def list_all(self) -> list[Material]:
        """Return all materials in the catalog."""
        return list(self._catalog.values())

    @property
    def names(self) -> list[str]:
        """All material names."""
        return sorted(self._catalog.keys())

    def add_custom(self, material: Material) -> None:
        """Add a user-defined material to the catalog."""
        self._add(material)

    def substrate_for_band(
        self,
        freq_hz: float,
        max_loss_tangent: float = 0.01,
    ) -> list[Material]:
        """Find substrates suitable for a given frequency band.

        Returns substrates with loss tangent ≤ max_loss_tangent,
        sorted by loss tangent (best first).
        """
        candidates = [
            m
            for m in self._catalog.values()
            if m.category == "substrate" and m.loss_tangent <= max_loss_tangent
        ]
        return sorted(candidates, key=lambda m: m.loss_tangent)
