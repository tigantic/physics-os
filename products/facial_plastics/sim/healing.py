"""Time-domain healing models for post-operative tissue evolution.

Simulates the biological processes that occur after surgery:
  1. Acute phase (0–7 days): edema, inflammation
  2. Proliferative phase (7–21 days): collagen deposition, contraction
  3. Remodeling phase (21–365+ days): scar maturation, settling

Physics models:
  - Edema: poroelastic swelling (fluid influx)
  - Scar formation: stiffness increase via collagen remodeling
  - Tissue settling: gravity-driven creep of soft tissue envelope
  - Cartilage memory: elastic springback after scoring/bending
  - Graft integration: stiffness transition at host-graft interface
  - Suture absorption: time-dependent loss of constraining forces
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.types import StructureType, VolumeMesh

logger = logging.getLogger(__name__)


# ── Healing phase definitions ─────────────────────────────────────

@dataclass(frozen=True)
class HealingPhase:
    """Parameters for a single healing phase."""
    name: str
    onset_days: float
    duration_days: float
    # Edema
    edema_peak_fraction: float  # volumetric swelling at peak (0=none, 0.3=30%)
    edema_time_constant_days: float  # exponential decay constant
    # Scar
    scar_stiffness_multiplier: float  # final stiffness relative to native
    scar_formation_rate: float  # dimensionless (0-1) fraction per day
    # Settling
    settling_rate_mm_per_day: float  # gravitational creep rate


# Nominal healing timeline for nasal soft tissue
NASAL_HEALING_PHASES: List[HealingPhase] = [
    HealingPhase(
        name="acute",
        onset_days=0.0,
        duration_days=7.0,
        edema_peak_fraction=0.25,     # 25% swelling
        edema_time_constant_days=3.0,
        scar_stiffness_multiplier=1.0,  # no scar yet
        scar_formation_rate=0.0,
        settling_rate_mm_per_day=0.0,
    ),
    HealingPhase(
        name="proliferative",
        onset_days=7.0,
        duration_days=14.0,
        edema_peak_fraction=0.15,
        edema_time_constant_days=7.0,
        scar_stiffness_multiplier=2.0,  # scar is stiffer
        scar_formation_rate=0.05,       # 5% per day
        settling_rate_mm_per_day=0.02,
    ),
    HealingPhase(
        name="remodeling",
        onset_days=21.0,
        duration_days=344.0,           # up to 1 year
        edema_peak_fraction=0.05,
        edema_time_constant_days=30.0,
        scar_stiffness_multiplier=1.5,  # matures toward native
        scar_formation_rate=0.01,
        settling_rate_mm_per_day=0.005,
    ),
]


# ── Healing state ─────────────────────────────────────────────────

@dataclass
class HealingState:
    """Current state of the healing process at a given time point."""
    time_days: float

    # Per-element fields
    edema_fraction: np.ndarray      # (E,) volumetric swelling fraction
    scar_fraction: np.ndarray       # (E,) scar formation fraction [0,1]
    stiffness_multiplier: np.ndarray  # (E,) effective stiffness multiplier
    settling_displacement: np.ndarray  # (N,3) accumulated settling (mm)

    # Suture state
    suture_strength: Dict[str, float] = field(default_factory=dict)  # id → fraction

    # Summary
    phase_name: str = "acute"
    mean_edema_pct: float = 0.0
    mean_scar_pct: float = 0.0
    max_settling_mm: float = 0.0

    def update_summary(self) -> None:
        """Recompute summary statistics."""
        self.mean_edema_pct = float(np.mean(self.edema_fraction)) * 100.0
        self.mean_scar_pct = float(np.mean(self.scar_fraction)) * 100.0
        if self.settling_displacement.size > 0:
            self.max_settling_mm = float(np.max(
                np.linalg.norm(self.settling_displacement, axis=1)
            ))

    def summary(self) -> str:
        return (
            f"Healing @ day {self.time_days:.0f} ({self.phase_name}): "
            f"edema={self.mean_edema_pct:.1f}%, "
            f"scar={self.mean_scar_pct:.1f}%, "
            f"settling={self.max_settling_mm:.2f}mm"
        )


# ── Healing model ─────────────────────────────────────────────────

class HealingModel:
    """Time-domain healing model for post-operative tissue evolution.

    Computes the healing state at any given time point using
    the superposition of biological processes:

      edema(t) = Σ_phase peak * exp(-(t - onset)/τ)
      scar(t)  = min(1, Σ_phase rate * (t - onset))
      settling(t) = ∫₀ᵗ settling_rate(t') dt'
      stiffness(t) = (1 - scar) * native + scar * scar_multiplier
    """

    def __init__(
        self,
        mesh: VolumeMesh,
        *,
        phases: Optional[List[HealingPhase]] = None,
        gravity_direction: np.ndarray = np.array([0.0, 0.0, -1.0]),
    ) -> None:
        self._mesh = mesh
        self._phases = phases if phases is not None else NASAL_HEALING_PHASES
        self._gravity = gravity_direction / max(np.linalg.norm(gravity_direction), 1e-12)
        self._n_elems = mesh.n_elements
        self._n_nodes = mesh.n_nodes

        # Build structure map for tissue-specific healing rates
        self._tissue_healing_rates = self._compute_tissue_healing_rates()

        # Pre-build node → element adjacency for O(1) lookup
        self._node_elem_adj = self._build_node_element_adjacency()

    def _build_node_element_adjacency(self) -> List[np.ndarray]:
        """Pre-build per-node lists of connected element indices.

        Replaces the O(N*E) inner-loop scans in apply_healing_to_mesh
        with O(N * avg_degree) lookups.
        """
        from collections import defaultdict
        adj: Dict[int, List[int]] = defaultdict(list)
        for eid in range(self._n_elems):
            for nid in self._mesh.elements[eid]:
                adj[int(nid)].append(eid)
        return [
            np.array(adj.get(nid, []), dtype=np.int64)
            for nid in range(self._n_nodes)
        ]

    def _compute_tissue_healing_rates(self) -> np.ndarray:
        """Compute per-element healing rate multipliers based on tissue type.

        Different tissues heal at different rates:
          - Skin: normal (1.0)
          - Fat: slow (0.7)
          - Cartilage: very slow (0.3)
          - Bone: slow initially, fast remodeling (0.5)
          - Mucosa: fast (1.5)
        """
        rates = np.ones(self._n_elems, dtype=np.float64)
        tissue_rate_map = {
            StructureType.SKIN_ENVELOPE: 1.0,
            StructureType.SKIN_THICK: 1.0,
            StructureType.SKIN_THIN: 1.0,
            StructureType.FAT_SUBCUTANEOUS: 0.7,
            StructureType.FAT_MALAR: 0.7,
            StructureType.MUSCLE_MIMETIC: 0.9,
            StructureType.SMAS: 0.8,
            StructureType.CARTILAGE_SEPTUM: 0.3,
            StructureType.CARTILAGE_UPPER_LATERAL: 0.3,
            StructureType.CARTILAGE_LOWER_LATERAL: 0.3,
            StructureType.CARTILAGE_ALAR: 0.3,
            StructureType.BONE_NASAL: 0.5,
            StructureType.BONE_MAXILLA: 0.5,
            StructureType.MUCOSA_NASAL: 1.5,
            StructureType.TURBINATE_INFERIOR: 0.8,
            StructureType.PERIOSTEUM: 0.6,
        }

        for eid in range(self._n_elems):
            rid = int(self._mesh.region_ids[eid])
            props = self._mesh.region_materials.get(rid)
            if props is not None:
                rate = tissue_rate_map.get(props.structure_type, 1.0)
                rates[eid] = rate

        return rates

    def compute_state(
        self,
        time_days: float,
        *,
        surgical_displacements: Optional[np.ndarray] = None,
    ) -> HealingState:
        """Compute healing state at a given time point.

        Parameters
        ----------
        time_days : float
            Days post-surgery.
        surgical_displacements : optional (N,3)
            Initial surgical displacement field (from FEM result).
        """
        edema: np.ndarray = np.zeros(self._n_elems, dtype=np.float64)
        scar: np.ndarray = np.zeros(self._n_elems, dtype=np.float64)
        stiffness: np.ndarray = np.ones(self._n_elems, dtype=np.float64)
        settling: np.ndarray = np.zeros((self._n_nodes, 3), dtype=np.float64)

        current_phase = "acute"

        for phase in self._phases:
            if time_days < phase.onset_days:
                continue

            t_local = time_days - phase.onset_days
            t_in_phase = min(t_local, phase.duration_days)
            phase_progress = t_in_phase / max(phase.duration_days, 1e-6)

            current_phase = phase.name

            # Edema: exponential rise and decay
            if phase.edema_peak_fraction > 0:
                tau = phase.edema_time_constant_days
                if t_local < tau:
                    # Rising phase
                    edema_contribution = phase.edema_peak_fraction * (1.0 - np.exp(-t_local / tau))
                else:
                    # Decaying phase
                    edema_contribution = phase.edema_peak_fraction * np.exp(-(t_local - tau) / tau)

                edema += edema_contribution * self._tissue_healing_rates

            # Scar formation: linear accumulation with saturation
            if phase.scar_formation_rate > 0:
                scar_increment = phase.scar_formation_rate * t_in_phase * self._tissue_healing_rates
                scar += scar_increment

            # Stiffness: blend native and scar properties
            if phase.scar_stiffness_multiplier != 1.0:
                scar_stiff = phase.scar_stiffness_multiplier
                stiffness_blend = (1.0 - scar) * 1.0 + scar * scar_stiff
                stiffness = np.asarray(np.maximum(stiffness, stiffness_blend))

            # Settling: gravity-driven creep
            if phase.settling_rate_mm_per_day > 0:
                settling_mag = phase.settling_rate_mm_per_day * t_in_phase
                # Settling is larger for soft tissues (high healing rate = soft)
                for nid in range(self._n_nodes):
                    # Find nearby elements to estimate tissue softness
                    settle_factor = settling_mag
                    if surgical_displacements is not None:
                        # Settling proportional to surgical displacement magnitude
                        disp_mag = float(np.linalg.norm(surgical_displacements[nid]))
                        settle_factor *= min(disp_mag / 5.0, 1.0)

                    settling[nid] += self._gravity * settle_factor

        # Clamp scar fraction
        scar_clamped: np.ndarray = np.asarray(np.clip(scar, 0.0, 1.0))

        # Edema-adjusted stiffness (edemic tissue is softer)
        stiffness *= (1.0 - 0.3 * edema)  # 30% softening per unit edema
        stiffness_clamped: np.ndarray = np.asarray(np.maximum(stiffness, 0.1))

        state = HealingState(
            time_days=time_days,
            edema_fraction=edema,
            scar_fraction=scar_clamped,
            stiffness_multiplier=stiffness_clamped,
            settling_displacement=settling,
            phase_name=current_phase,
        )
        state.update_summary()

        return state

    def compute_timeline(
        self,
        time_points_days: List[float],
        *,
        surgical_displacements: Optional[np.ndarray] = None,
    ) -> List[HealingState]:
        """Compute healing states at multiple time points.

        Typical clinical follow-up: [1, 7, 14, 30, 90, 180, 365] days.
        """
        states = []
        for t in sorted(time_points_days):
            state = self.compute_state(
                t,
                surgical_displacements=surgical_displacements,
            )
            states.append(state)
            logger.info("Healing state: %s", state.summary())

        return states

    def apply_healing_to_mesh(
        self,
        state: HealingState,
        base_displacements: np.ndarray,
    ) -> np.ndarray:
        """Apply healing effects to the surgical displacement field.

        Returns modified displacements that include:
          - Edema swelling (volumetric expansion)
          - Gravitational settling
          - Scar contraction

        Parameters
        ----------
        state : HealingState
        base_displacements : (N,3) surgical displacement field

        Returns
        -------
        modified_displacements : (N,3) healed displacement field
        """
        healed = base_displacements.copy()

        # Add settling
        healed += state.settling_displacement

        # Edema: expand displacements outward from centroid
        centroid = self._mesh.nodes.mean(axis=0)
        for nid in range(self._n_nodes):
            # Average edema of elements connected to this node
            connected = self._node_elem_adj[nid]
            if len(connected) > 0:
                edema_avg = float(np.mean(state.edema_fraction[connected]))
            else:
                edema_avg = 0.0

            # Edema expands tissue outward
            direction = self._mesh.nodes[nid] - centroid
            dir_norm = np.linalg.norm(direction)
            if dir_norm > 1e-6:
                direction /= dir_norm
                healed[nid] += direction * edema_avg * 2.0  # 2mm per unit fraction

        # Scar contraction: reduce displacement magnitude over time
        # Scars pull tissue inward (contract), partially reversing surgical changes
        for nid in range(self._n_nodes):
            connected = self._node_elem_adj[nid]
            if len(connected) > 0:
                scar_avg = float(np.mean(state.scar_fraction[connected]))
            else:
                scar_avg = 0.0

            # Scar contracts: reduce displacement by scar fraction * contraction_factor
            contraction_factor = 0.1  # 10% of scar formation goes to contraction
            healed[nid] *= (1.0 - scar_avg * contraction_factor)

        return healed

    def predict_final_shape(
        self,
        base_displacements: np.ndarray,
        *,
        final_time_days: float = 365.0,
    ) -> Tuple[np.ndarray, HealingState]:
        """Predict the final post-healing shape at equilibrium.

        This integrates all healing effects to the final time point
        and returns the predicted displacement field and healing state.
        """
        state = self.compute_state(
            final_time_days,
            surgical_displacements=base_displacements,
        )
        final_displacements = self.apply_healing_to_mesh(state, base_displacements)
        return final_displacements, state
