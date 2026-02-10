"""Suture element mechanics for surgical plan simulation.

Sutures in facial plastics exert localized forces that reshape
tissue. This module models sutures as nonlinear spring elements
connecting pairs of mesh nodes/regions.

Suture types modeled:
  - Transdomal suture (compressive dome reshaping)
  - Interdomal suture (dome approximation)
  - Spanning sutures (lateral crural flattening)
  - Fixation sutures (tied to bone/cartilage)
  - Cinch sutures (alar base narrowing)

Each suture is a 2-node spring element with:
  - Nonlinear force-displacement curve
  - Pretension (initial shortening)
  - Creep relaxation over time
  - Material properties (PDS, nylon, Prolene)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Suture material properties ────────────────────────────────────

class SutureMaterial(str, Enum):
    """Suture material types with mechanical properties."""
    PDS = "pds"           # Polydioxanone (absorbable)
    NYLON = "nylon"       # Polyamide (permanent)
    PROLENE = "prolene"   # Polypropylene (permanent)
    VICRYL = "vicryl"     # Polyglactin 910 (absorbable)
    CHROMIC = "chromic"   # Chromic catgut (absorbable)


@dataclass(frozen=True)
class SutureMaterialProps:
    """Mechanical properties for suture material."""
    E_mpa: float          # Young's modulus (MPa)
    sigma_y_mpa: float    # Yield stress (MPa)
    diameter_mm: float    # Suture diameter (mm)
    absorption_days: float  # Days to 50% strength loss (0 = permanent)
    creep_rate: float     # Creep strain rate (1/s) under constant load
    knot_security: float  # Knot security factor (0-1)

    @property
    def cross_section_mm2(self) -> float:
        return np.pi * (self.diameter_mm / 2.0) ** 2

    @property
    def stiffness_n_per_mm(self) -> float:
        """Axial stiffness k = EA/L (using unit reference length)."""
        return self.E_mpa * self.cross_section_mm2

    @property
    def is_absorbable(self) -> bool:
        return self.absorption_days > 0


# Literature values for common suture materials
SUTURE_MATERIALS: Dict[SutureMaterial, Dict[str, SutureMaterialProps]] = {
    SutureMaterial.PDS: {
        "5-0": SutureMaterialProps(E_mpa=1400.0, sigma_y_mpa=60.0, diameter_mm=0.10, absorption_days=90, creep_rate=1e-8, knot_security=0.7),
        "4-0": SutureMaterialProps(E_mpa=1400.0, sigma_y_mpa=60.0, diameter_mm=0.15, absorption_days=90, creep_rate=1e-8, knot_security=0.7),
    },
    SutureMaterial.NYLON: {
        "5-0": SutureMaterialProps(E_mpa=3500.0, sigma_y_mpa=90.0, diameter_mm=0.10, absorption_days=0, creep_rate=5e-9, knot_security=0.6),
        "4-0": SutureMaterialProps(E_mpa=3500.0, sigma_y_mpa=90.0, diameter_mm=0.15, absorption_days=0, creep_rate=5e-9, knot_security=0.6),
        "3-0": SutureMaterialProps(E_mpa=3500.0, sigma_y_mpa=90.0, diameter_mm=0.20, absorption_days=0, creep_rate=5e-9, knot_security=0.6),
    },
    SutureMaterial.PROLENE: {
        "5-0": SutureMaterialProps(E_mpa=1500.0, sigma_y_mpa=50.0, diameter_mm=0.10, absorption_days=0, creep_rate=3e-9, knot_security=0.75),
        "4-0": SutureMaterialProps(E_mpa=1500.0, sigma_y_mpa=50.0, diameter_mm=0.15, absorption_days=0, creep_rate=3e-9, knot_security=0.75),
    },
    SutureMaterial.VICRYL: {
        "5-0": SutureMaterialProps(E_mpa=2000.0, sigma_y_mpa=70.0, diameter_mm=0.10, absorption_days=42, creep_rate=2e-8, knot_security=0.8),
        "4-0": SutureMaterialProps(E_mpa=2000.0, sigma_y_mpa=70.0, diameter_mm=0.15, absorption_days=42, creep_rate=2e-8, knot_security=0.8),
    },
    SutureMaterial.CHROMIC: {
        "4-0": SutureMaterialProps(E_mpa=500.0, sigma_y_mpa=30.0, diameter_mm=0.15, absorption_days=21, creep_rate=5e-8, knot_security=0.65),
    },
}


def get_suture_props(
    material: SutureMaterial,
    gauge: str = "5-0",
) -> SutureMaterialProps:
    """Look up suture material properties."""
    mat_dict = SUTURE_MATERIALS.get(material, {})
    props = mat_dict.get(gauge)
    if props is None:
        available = list(mat_dict.keys())
        if available:
            props = mat_dict[available[0]]
        else:
            raise ValueError(f"No properties for {material.value} {gauge}")
    return props


# ── Suture element ────────────────────────────────────────────────

@dataclass
class SutureElement:
    """A single suture element connecting two mesh node groups.

    Models a suture as a nonlinear spring with prestrain,
    wrapped around tissue nodes.
    """
    suture_id: str
    node_a: int           # Mesh node index (or representative node)
    node_b: int           # Mesh node index on opposing side
    material: SutureMaterial = SutureMaterial.PDS
    gauge: str = "5-0"
    pretension_mm: float = 0.0  # Initial shortening (tightening)
    wrap_nodes_a: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    wrap_nodes_b: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    reference_length_mm: float = 0.0

    def __post_init__(self) -> None:
        self._props = get_suture_props(self.material, self.gauge)

    @property
    def props(self) -> SutureMaterialProps:
        return self._props

    def compute_force(
        self,
        pos_a: np.ndarray,
        pos_b: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute suture force vectors on node A and node B.

        Returns (force_a, force_b, current_length).
        force_a is the force pulling node A toward B, and vice versa.
        """
        delta = pos_b - pos_a
        current_length = float(np.linalg.norm(delta))
        if current_length < 1e-12:
            return np.zeros(3), np.zeros(3), 0.0

        direction = delta / current_length

        ref_len = self.reference_length_mm
        if ref_len <= 0:
            ref_len = current_length + self.pretension_mm

        stretch = current_length - (ref_len - self.pretension_mm)

        if stretch <= 0:
            # Suture in compression — sutures don't resist compression
            return np.zeros(3), np.zeros(3), current_length

        # Nonlinear force-extension: bilinear model
        k = self._props.stiffness_n_per_mm / max(ref_len, 1e-6)
        max_force = self._props.sigma_y_mpa * self._props.cross_section_mm2

        force_mag = min(k * stretch, max_force)

        force_a = direction * force_mag
        force_b = -direction * force_mag

        return force_a, force_b, current_length

    def compute_stiffness(self) -> float:
        """Compute axial stiffness K = EA/L (N/mm)."""
        L = max(self.reference_length_mm, 1e-6)
        return self._props.stiffness_n_per_mm / L

    def strength_at_time(self, days: float) -> float:
        """Fractional remaining strength after given days.

        For absorbable sutures, strength decays exponentially.
        For permanent sutures, returns 1.0.
        """
        if not self._props.is_absorbable:
            return 1.0
        t_half = self._props.absorption_days
        if t_half <= 0:
            return 1.0
        # Exponential decay: strength = exp(-0.693 * t / t_half)
        return float(np.exp(-0.693 * days / t_half))

    def creep_stretch_mm(self, force_n: float, time_seconds: float) -> float:
        """Compute creep-induced elongation (mm).

        δ_creep = ε_creep * L₀ = (creep_rate * σ * t) * L₀
        """
        A = self._props.cross_section_mm2
        stress = force_n / max(A, 1e-12)
        creep_strain = self._props.creep_rate * stress * time_seconds
        return creep_strain * max(self.reference_length_mm, 1e-6)


# ── Suture system ─────────────────────────────────────────────────

class SutureSystem:
    """Collection of suture elements that apply forces to the FEM mesh.

    Manages all sutures in a surgical plan and computes their
    collective contribution to the global force vector.
    """

    def __init__(self) -> None:
        self._sutures: List[SutureElement] = []

    @property
    def n_sutures(self) -> int:
        return len(self._sutures)

    def add_suture(self, suture: SutureElement) -> None:
        """Add a suture element to the system."""
        self._sutures.append(suture)

    def create_transdomal(
        self,
        left_dome_nodes: np.ndarray,
        right_dome_nodes: np.ndarray,
        mesh_nodes: np.ndarray,
        *,
        material: SutureMaterial = SutureMaterial.PDS,
        gauge: str = "5-0",
        tension: float = 0.5,
    ) -> SutureElement:
        """Create a transdomal suture between left and right domes.

        Selects the most lateral node pair and creates a
        compressive suture to narrow the tip.
        """
        if len(left_dome_nodes) == 0 or len(right_dome_nodes) == 0:
            raise ValueError("Dome node arrays must not be empty")

        # Find the widest node pair
        left_pos = mesh_nodes[left_dome_nodes]
        right_pos = mesh_nodes[right_dome_nodes]

        left_centroid = left_pos.mean(axis=0)
        right_centroid = right_pos.mean(axis=0)

        # Select representative nodes closest to centroids
        left_dists = np.linalg.norm(left_pos - left_centroid, axis=1)
        right_dists = np.linalg.norm(right_pos - right_centroid, axis=1)
        node_a = int(left_dome_nodes[np.argmin(left_dists)])
        node_b = int(right_dome_nodes[np.argmin(right_dists)])

        ref_length = float(np.linalg.norm(
            mesh_nodes[node_a] - mesh_nodes[node_b]
        ))

        # Pretension: tension parameter maps to 0-3mm of shortening
        pretension = tension * 3.0

        suture = SutureElement(
            suture_id=f"transdomal_{self.n_sutures}",
            node_a=node_a,
            node_b=node_b,
            material=material,
            gauge=gauge,
            pretension_mm=pretension,
            wrap_nodes_a=left_dome_nodes,
            wrap_nodes_b=right_dome_nodes,
            reference_length_mm=ref_length,
        )
        self.add_suture(suture)
        return suture

    def create_interdomal(
        self,
        left_dome_nodes: np.ndarray,
        right_dome_nodes: np.ndarray,
        mesh_nodes: np.ndarray,
        *,
        material: SutureMaterial = SutureMaterial.PDS,
        gauge: str = "5-0",
        tension: float = 0.5,
    ) -> SutureElement:
        """Create an interdomal suture connecting dome peaks."""
        if len(left_dome_nodes) == 0 or len(right_dome_nodes) == 0:
            raise ValueError("Dome node arrays must not be empty")

        left_pos = mesh_nodes[left_dome_nodes]
        right_pos = mesh_nodes[right_dome_nodes]

        # Select most superior (highest Z) nodes
        node_a = int(left_dome_nodes[np.argmax(left_pos[:, 2])])
        node_b = int(right_dome_nodes[np.argmax(right_pos[:, 2])])

        ref_length = float(np.linalg.norm(
            mesh_nodes[node_a] - mesh_nodes[node_b]
        ))

        pretension = tension * 2.0

        suture = SutureElement(
            suture_id=f"interdomal_{self.n_sutures}",
            node_a=node_a,
            node_b=node_b,
            material=material,
            gauge=gauge,
            pretension_mm=pretension,
            wrap_nodes_a=left_dome_nodes,
            wrap_nodes_b=right_dome_nodes,
            reference_length_mm=ref_length,
        )
        self.add_suture(suture)
        return suture

    def compute_global_forces(
        self,
        mesh_nodes: np.ndarray,
        current_displacements: np.ndarray,
    ) -> np.ndarray:
        """Compute total suture force contribution to the global force vector.

        Parameters
        ----------
        mesh_nodes : (N,3) reference node positions
        current_displacements : (N,3) current displacements

        Returns
        -------
        f_suture : (N,3) suture force vector
        """
        n_nodes = mesh_nodes.shape[0]
        f_suture = np.zeros((n_nodes, 3), dtype=np.float64)
        current_pos = mesh_nodes + current_displacements

        for suture in self._sutures:
            pos_a = current_pos[suture.node_a]
            pos_b = current_pos[suture.node_b]

            force_a, force_b, _ = suture.compute_force(pos_a, pos_b)

            f_suture[suture.node_a] += force_a
            f_suture[suture.node_b] += force_b

            # Distribute to wrap nodes (if any) with inverse-distance weighting
            if len(suture.wrap_nodes_a) > 1:
                wrap_pos_a = current_pos[suture.wrap_nodes_a]
                dists = np.linalg.norm(wrap_pos_a - pos_a, axis=1)
                dists = np.maximum(dists, 1e-6)
                weights = 1.0 / dists
                weights /= weights.sum()
                for i, nid in enumerate(suture.wrap_nodes_a):
                    f_suture[nid] += force_a * weights[i] * 0.3  # 30% to wraps

            if len(suture.wrap_nodes_b) > 1:
                wrap_pos_b = current_pos[suture.wrap_nodes_b]
                dists = np.linalg.norm(wrap_pos_b - pos_b, axis=1)
                dists = np.maximum(dists, 1e-6)
                weights = 1.0 / dists
                weights /= weights.sum()
                for i, nid in enumerate(suture.wrap_nodes_b):
                    f_suture[nid] += force_b * weights[i] * 0.3

        return f_suture

    def compute_stiffness_contributions(self) -> List[Tuple[int, int, float]]:
        """Compute stiffness matrix contributions from all sutures.

        Returns list of (node_a, node_b, stiffness_n_per_mm).
        """
        contribs = []
        for suture in self._sutures:
            k = suture.compute_stiffness()
            contribs.append((suture.node_a, suture.node_b, k))
        return contribs

    def evolve_time(
        self,
        days: float,
        current_forces: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Compute time-evolved suture state (absorption, creep).

        Returns {suture_id: {"strength_fraction", "creep_mm"}}
        """
        state: Dict[str, Dict[str, float]] = {}

        for suture in self._sutures:
            strength = suture.strength_at_time(days)

            creep_mm = 0.0
            if current_forces is not None:
                # Estimate force on this suture
                f_mag = float(np.linalg.norm(current_forces[suture.node_a]))
                creep_mm = suture.creep_stretch_mm(f_mag, days * 86400.0)

            state[suture.suture_id] = {
                "strength_fraction": strength,
                "creep_mm": creep_mm,
                "effective_pretension_mm": max(
                    0.0, suture.pretension_mm * strength - creep_mm
                ),
            }

        return state

    def summary(self) -> str:
        """Summary of all sutures in the system."""
        lines = [f"SutureSystem: {self.n_sutures} sutures"]
        for s in self._sutures:
            lines.append(
                f"  {s.suture_id}: {s.material.value} {s.gauge}, "
                f"L={s.reference_length_mm:.1f}mm, "
                f"pretension={s.pretension_mm:.1f}mm"
            )
        return "\n".join(lines)
