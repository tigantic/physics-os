"""Cartilage-specific mechanics for nasal cartilage simulation.

Nasal cartilages have unique mechanical properties:
  - Anisotropic (fiber-reinforced via collagen)
  - Viscoelastic (time-dependent creep/relaxation)
  - Heterogeneous (stiffness varies with position)
  - Curved (shell-like behaviour with bending resistance)

This module provides:
  - Cartilage-specific constitutive model (transversely isotropic NeoHookean)
  - Bending resistance for thin cartilage plates
  - Scoring mechanics (controlled stiffness reduction)
  - Graft integration modeling (stiffness transition zones)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..core.types import (
    MaterialModel,
    MeshElementType,
    StructureType,
    Vec3,
    VolumeMesh,
)

logger = logging.getLogger(__name__)


# ── Cartilage material parameters (literature) ───────────────────

@dataclass(frozen=True)
class CartilageParams:
    """Material parameters for a specific cartilage type.

    Based on published nanoindentation and tensile testing data:
      - Richmon et al. 2005 (human septal cartilage)
      - Grellmann et al. 2006 (auricular cartilage)
      - Rotter et al. 2002 (nasal tip cartilage)
    """
    E_fiber: float      # Young's modulus along fiber direction (Pa)
    E_matrix: float     # Young's modulus of matrix (Pa)
    nu: float           # Poisson's ratio
    thickness_mm: float  # Typical thickness
    tau_1: float        # Prony series relaxation time (s)
    g_1: float          # Prony series relaxation coefficient
    tau_2: float = 0.0  # Second relaxation time (optional)
    g_2: float = 0.0    # Second coefficient (optional)
    density: float = 1100.0  # kg/m³
    fiber_fraction: float = 0.3  # volume fraction of collagen fibers

    @property
    def E_eff(self) -> float:
        """Effective Young's modulus (rule of mixtures)."""
        return self.fiber_fraction * self.E_fiber + (1.0 - self.fiber_fraction) * self.E_matrix

    @property
    def mu(self) -> float:
        """Effective shear modulus."""
        return self.E_eff / (2.0 * (1.0 + self.nu))

    @property
    def kappa(self) -> float:
        """Effective bulk modulus."""
        return self.E_eff / (3.0 * (1.0 - 2.0 * self.nu))


CARTILAGE_LIBRARY: Dict[StructureType, CartilageParams] = {
    StructureType.CARTILAGE_SEPTUM: CartilageParams(
        E_fiber=12.0e6,     # 12 MPa fiber direction
        E_matrix=2.5e6,     # 2.5 MPa matrix
        nu=0.3,
        thickness_mm=2.0,
        tau_1=5.0,          # 5s relaxation
        g_1=0.3,
        tau_2=50.0,
        g_2=0.15,
    ),
    StructureType.CARTILAGE_UPPER_LATERAL: CartilageParams(
        E_fiber=10.0e6,
        E_matrix=2.0e6,
        nu=0.3,
        thickness_mm=1.2,
        tau_1=4.0,
        g_1=0.25,
        tau_2=40.0,
        g_2=0.12,
    ),
    StructureType.CARTILAGE_LOWER_LATERAL: CartilageParams(
        E_fiber=8.0e6,
        E_matrix=1.5e6,
        nu=0.3,
        thickness_mm=0.8,
        tau_1=3.5,
        g_1=0.22,
        tau_2=35.0,
        g_2=0.10,
    ),
    StructureType.CARTILAGE_ALAR: CartilageParams(
        E_fiber=5.0e6,
        E_matrix=1.0e6,
        nu=0.3,
        thickness_mm=0.6,
        tau_1=3.0,
        g_1=0.20,
        tau_2=30.0,
        g_2=0.08,
    ),
    StructureType.CARTILAGE_EAR: CartilageParams(
        E_fiber=6.0e6,      # auricular: elastic cartilage
        E_matrix=1.2e6,
        nu=0.25,
        thickness_mm=1.0,
        tau_1=10.0,          # slower relaxation
        g_1=0.15,
    ),
}


# ── Cartilage scoring model ──────────────────────────────────────

@dataclass
class ScoreLine:
    """A single score line on a cartilage surface."""
    position_y: float   # position along cartilage length (mm)
    depth_fraction: float  # fraction of thickness (0-1)
    width_mm: float = 0.5

    @property
    def stiffness_factor(self) -> float:
        """Residual stiffness after scoring.

        Based on beam bending theory: I ∝ h³, so scoring to depth d
        reduces bending stiffness by (1 - d/h)³.
        """
        return (1.0 - self.depth_fraction) ** 3


@dataclass
class ScoredCartilage:
    """Cartilage element with score lines applied."""
    structure: StructureType
    base_params: CartilageParams
    score_lines: List[ScoreLine] = field(default_factory=list)

    def effective_stiffness_at(self, y_pos: float) -> float:
        """Compute effective stiffness factor at a given position.

        Returns a multiplier [0, 1] for the base stiffness.
        """
        factor = 1.0
        for score in self.score_lines:
            dist = abs(y_pos - score.position_y)
            if dist < score.width_mm:
                # Smooth transition using Gaussian
                w = score.width_mm / 2.0
                influence = np.exp(-0.5 * (dist / w) ** 2) if w > 0 else 1.0
                score_factor = score.stiffness_factor
                factor *= 1.0 - influence * (1.0 - score_factor)
        return float(max(0.01, factor))  # minimum 1% residual

    def effective_params_at(self, y_pos: float) -> Dict[str, float]:
        """Get position-dependent material parameters."""
        factor = self.effective_stiffness_at(y_pos)
        return {
            "mu": self.base_params.mu * factor,
            "kappa": self.base_params.kappa * factor,
            "density": self.base_params.density,
        }


# ── Bending spring model for thin cartilages ────────────────────

def compute_bending_stiffness(
    thickness_mm: float,
    E_eff_pa: float,
    nu: float,
) -> float:
    """Flexural rigidity D = E*h³ / (12*(1-ν²)) for a cartilage plate.

    Returns D in N·mm (flexural rigidity per unit width).
    """
    # Convert E from Pa to N/mm² (= MPa)
    E_mpa = E_eff_pa * 1e-6
    h = thickness_mm
    return E_mpa * h**3 / (12.0 * (1.0 - nu**2))


def compute_cartilage_spring_constants(
    mesh: VolumeMesh,
    structure: StructureType,
) -> np.ndarray:
    """Compute per-element bending spring constants for thin cartilage.

    Returns (n_elements,) array of spring constants (N/mm).
    """
    params = CARTILAGE_LIBRARY.get(structure)
    if params is None:
        raise ValueError(f"No cartilage parameters for {structure.value}")

    D = compute_bending_stiffness(params.thickness_mm, params.E_eff, params.nu)

    # Find elements belonging to this structure
    target_regions: List[int] = []
    for rid, props in mesh.region_materials.items():
        if props.structure_type == structure:
            target_regions.append(rid)

    n_elems = mesh.n_elements
    k_bend = np.zeros(n_elems, dtype=np.float64)

    for eid in range(n_elems):
        if int(mesh.region_ids[eid]) in target_regions:
            # Element size (approximate edge length)
            elem_conn = mesh.elements[eid]
            coords = mesh.nodes[elem_conn[:min(4, len(elem_conn))]]
            if len(coords) < 2:
                continue
            # Average edge length
            edges = np.linalg.norm(np.diff(coords, axis=0), axis=1)
            L_avg = float(np.mean(edges)) if len(edges) > 0 else 1.0
            # Spring constant: k = D / L² (approximate)
            k_bend[eid] = D / max(L_avg**2, 1e-6)

    return k_bend


# ── Graft integration model ──────────────────────────────────────

@dataclass
class GraftSpec:
    """Specification for a cartilage graft."""
    source: StructureType  # donor site
    target: StructureType  # insertion site
    length_mm: float
    width_mm: float
    thickness_mm: float
    position: Vec3  # center position in mesh coordinates
    orientation: np.ndarray  # (3,) primary axis direction

    @property
    def source_params(self) -> CartilageParams:
        params = CARTILAGE_LIBRARY.get(self.source)
        if params is None:
            raise ValueError(f"No cartilage parameters for {self.source.value}")
        return params

    @property
    def volume_mm3(self) -> float:
        return self.length_mm * self.width_mm * self.thickness_mm


def compute_graft_transition_zone(
    graft: GraftSpec,
    mesh: VolumeMesh,
    transition_width_mm: float = 2.0,
) -> Dict[int, float]:
    """Compute stiffness transition factors for graft-host interface.

    At the graft boundary, there's a transition zone where the
    effective stiffness blends between graft and host tissue.
    This prevents stress concentration at sharp interfaces.

    Returns {element_id: transition_factor} where factor blends
    from 1.0 (pure graft) to 0.0 (pure host) in the transition zone.
    """
    graft_center = graft.position.to_array()

    # Half-dimensions of the graft
    half_l = graft.length_mm / 2.0
    half_w = graft.width_mm / 2.0

    transition: Dict[int, float] = {}

    for eid in range(mesh.n_elements):
        elem_conn = mesh.elements[eid]
        coords = mesh.nodes[elem_conn[:min(4, len(elem_conn))]]
        centroid = coords.mean(axis=0)

        # Distance from graft center along and across graft axis
        delta = centroid - graft_center
        along = abs(np.dot(delta, graft.orientation))
        across = float(np.linalg.norm(
            delta - np.dot(delta, graft.orientation) * graft.orientation
        ))

        # Check if element is within graft OR transition zone
        in_graft = (along < half_l) and (across < half_w)
        in_transition = (
            (along < half_l + transition_width_mm) and
            (across < half_w + transition_width_mm)
        )

        if in_graft:
            transition[eid] = 1.0
        elif in_transition:
            # Smooth blend
            d_along = max(0.0, along - half_l) / transition_width_mm
            d_across = max(0.0, across - half_w) / transition_width_mm
            d = np.sqrt(d_along**2 + d_across**2)
            # Cosine blend
            factor = 0.5 * (1.0 + np.cos(np.pi * min(d, 1.0)))
            transition[eid] = float(factor)

    return transition


# ── Cartilage solver ──────────────────────────────────────────────

class CartilageSolver:
    """Specialized solver for nasal cartilage mechanics.

    Augments the global FEM solver with:
      - Cartilage-specific material parameters from literature
      - Bending resistance for thin plates
      - Score line mechanics
      - Graft integration transition zones
      - Viscoelastic relaxation (Prony series post-correction)
    """

    def __init__(self, mesh: VolumeMesh) -> None:
        self._mesh = mesh
        self._scored: Dict[StructureType, ScoredCartilage] = {}
        self._grafts: List[GraftSpec] = []
        self._bending_springs: Dict[StructureType, np.ndarray] = {}

    def apply_scoring(
        self,
        structure: StructureType,
        score_lines: List[ScoreLine],
    ) -> None:
        """Apply score lines to a cartilage structure."""
        params = CARTILAGE_LIBRARY.get(structure)
        if params is None:
            raise ValueError(f"No cartilage parameters for {structure.value}")

        self._scored[structure] = ScoredCartilage(
            structure=structure,
            base_params=params,
            score_lines=score_lines,
        )
        logger.info(
            "Applied %d score lines to %s", len(score_lines), structure.value,
        )

    def add_graft(self, graft: GraftSpec) -> None:
        """Register a graft for transition zone computation."""
        self._grafts.append(graft)

    def compute_material_map(self) -> Dict[int, Dict[str, float]]:
        """Compute position-dependent material parameters for all cartilage elements.

        Returns {element_id: {"mu": ..., "kappa": ..., "density": ...}}
        incorporating scoring and graft transition effects.
        """
        material_map: Dict[int, Dict[str, float]] = {}

        # Base cartilage parameters per element
        for eid in range(self._mesh.n_elements):
            rid = int(self._mesh.region_ids[eid])
            props = self._mesh.region_materials.get(rid)
            if props is None:
                continue

            structure = props.structure_type
            if structure not in CARTILAGE_LIBRARY:
                continue

            elem_conn = self._mesh.elements[eid]
            coords = self._mesh.nodes[elem_conn[:min(4, len(elem_conn))]]
            centroid = coords.mean(axis=0)

            # Check for scoring
            scored = self._scored.get(structure)
            if scored is not None:
                y_pos = float(centroid[1])
                material_map[eid] = scored.effective_params_at(y_pos)
            else:
                lib_params = CARTILAGE_LIBRARY[structure]
                material_map[eid] = {
                    "mu": lib_params.mu,
                    "kappa": lib_params.kappa,
                    "density": lib_params.density,
                }

        # Apply graft transition zones
        for graft in self._grafts:
            transition = compute_graft_transition_zone(graft, self._mesh)
            graft_params = graft.source_params

            for eid, factor in transition.items():
                if eid in material_map:
                    host_params = material_map[eid]
                    # Blend graft and host properties
                    material_map[eid] = {
                        "mu": factor * graft_params.mu + (1.0 - factor) * host_params["mu"],
                        "kappa": factor * graft_params.kappa + (1.0 - factor) * host_params["kappa"],
                        "density": factor * graft_params.density + (1.0 - factor) * host_params["density"],
                    }
                else:
                    material_map[eid] = {
                        "mu": graft_params.mu * factor,
                        "kappa": graft_params.kappa * factor,
                        "density": graft_params.density,
                    }

        return material_map

    def compute_bending_stiffness(self) -> Dict[StructureType, np.ndarray]:
        """Compute bending spring constants for all cartilage structures."""
        for structure in CARTILAGE_LIBRARY:
            try:
                k = compute_cartilage_spring_constants(self._mesh, structure)
                if np.any(k > 0):
                    self._bending_springs[structure] = k
            except ValueError:
                continue
        return self._bending_springs

    def apply_viscoelastic_correction(
        self,
        stresses: np.ndarray,
        time_seconds: float,
    ) -> np.ndarray:
        """Apply Prony series viscoelastic correction to static stresses.

        For a given hold time t, the stress relaxes according to:
          σ(t) = σ_0 * [1 - Σ g_i * (1 - exp(-t/τ_i))]

        This is a post-correction to the elastic solution.
        """
        corrected = stresses.copy()

        for eid in range(min(len(stresses), self._mesh.n_elements)):
            rid = int(self._mesh.region_ids[eid])
            props = self._mesh.region_materials.get(rid)
            if props is None:
                continue

            structure = props.structure_type
            cart_params = CARTILAGE_LIBRARY.get(structure)
            if cart_params is None:
                continue

            # Prony series relaxation
            relax_factor = 1.0
            if cart_params.tau_1 > 0 and cart_params.g_1 > 0:
                relax_factor -= cart_params.g_1 * (1.0 - np.exp(-time_seconds / cart_params.tau_1))
            if cart_params.tau_2 > 0 and cart_params.g_2 > 0:
                relax_factor -= cart_params.g_2 * (1.0 - np.exp(-time_seconds / cart_params.tau_2))

            relax_factor = max(relax_factor, 0.01)  # floor at 1%
            corrected[eid] *= relax_factor

        return corrected
