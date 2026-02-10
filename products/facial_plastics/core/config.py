"""Platform-wide configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class SolverConfig:
    """Configuration for physics solvers."""
    max_iterations: int = 10_000
    convergence_tol: float = 1e-8
    newton_max_iter: int = 50
    newton_rel_tol: float = 1e-6
    newton_abs_tol: float = 1e-10
    time_step_s: float = 0.01
    max_time_s: float = 1.0
    n_load_steps: int = 20
    use_line_search: bool = True
    cg_preconditioner: str = "jacobi"
    n_threads: int = 0  # 0 = auto
    enable_gpu: bool = False


@dataclass
class MeshConfig:
    """Configuration for mesh generation."""
    target_edge_length_mm: float = 1.5
    min_edge_length_mm: float = 0.3
    max_edge_length_mm: float = 5.0
    refinement_zones: List[str] = field(default_factory=lambda: ["surgical_roi"])
    refinement_factor: float = 0.5
    quality_threshold: float = 0.3
    max_aspect_ratio: float = 10.0
    element_order: int = 1  # 1=linear, 2=quadratic
    surface_fidelity_mm: float = 0.2


@dataclass
class SegmentationConfig:
    """Configuration for multi-structure segmentation."""
    method: str = "threshold_morphological"
    bone_hu_threshold: float = 300.0
    cartilage_hu_range: tuple = (100.0, 300.0)
    airway_hu_threshold: float = -500.0
    skin_detection_mode: str = "largest_component"
    min_structure_volume_mm3: float = 10.0
    smoothing_sigma_mm: float = 0.5
    fill_holes: bool = True
    use_ml_segmentation: bool = False
    ml_model_path: Optional[str] = None


@dataclass
class CFDConfig:
    """Configuration for airway CFD."""
    inlet_velocity_m_s: float = 0.0  # 0 = compute from flow rate
    inlet_flow_rate_l_min: float = 15.0  # resting nasal breathing
    outlet_pressure_pa: float = 0.0
    air_density_kg_m3: float = 1.185
    air_viscosity_pa_s: float = 1.831e-5
    turbulence_model: str = "laminar"  # laminar, k-epsilon, k-omega-sst
    wall_temperature_c: float = 34.0
    max_cfl: float = 0.8
    n_steady_iterations: int = 2000
    convergence_residual: float = 1e-4


@dataclass
class UQConfig:
    """Uncertainty quantification settings."""
    n_samples: int = 100
    method: str = "latin_hypercube"  # latin_hypercube, sobol, monte_carlo
    confidence_level: float = 0.95
    parameter_cv: float = 0.15  # default coefficient of variation
    sensitivity_method: str = "sobol"  # sobol, morris, variance_decomposition
    seed: int = 42


@dataclass
class PlatformConfig:
    """Master configuration for the facial plastics platform."""
    # Paths
    data_root: Path = Path("data/facial_plastics")
    case_library_root: Path = Path("data/facial_plastics/case_library")
    output_root: Path = Path("outputs/facial_plastics")
    model_root: Path = Path("models/facial_plastics")
    report_root: Path = Path("outputs/facial_plastics/reports")

    # Sub-configs
    solver: SolverConfig = field(default_factory=SolverConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    cfd: CFDConfig = field(default_factory=CFDConfig)
    uq: UQConfig = field(default_factory=UQConfig)

    # Governance
    require_consent: bool = True
    audit_all_operations: bool = True
    max_case_retention_days: int = 365 * 7  # 7 years
    encryption_at_rest: bool = True

    # Compute
    max_memory_gb: float = 32.0
    n_parallel_cases: int = 1

    def ensure_dirs(self) -> None:
        """Create all required directories."""
        for p in [
            self.data_root,
            self.case_library_root,
            self.output_root,
            self.model_root,
            self.report_root,
        ]:
            p.mkdir(parents=True, exist_ok=True)


# ── Default tissue material library ──────────────────────────────

DEFAULT_TISSUE_LIBRARY: Dict[str, Dict[str, float]] = {
    "skin_forehead": {
        "mu": 10.0e3,       # Pa (Neo-Hookean shear modulus)
        "kappa": 100.0e3,   # Pa (bulk modulus)
        "density": 1100.0,  # kg/m³
        "thickness": 3.0,   # mm
    },
    "skin_nasal_dorsum": {
        "mu": 8.0e3,
        "kappa": 80.0e3,
        "density": 1100.0,
        "thickness": 2.5,
    },
    "skin_nasal_tip": {
        "mu": 15.0e3,
        "kappa": 150.0e3,
        "density": 1100.0,
        "thickness": 4.0,
    },
    "skin_alar": {
        "mu": 12.0e3,
        "kappa": 120.0e3,
        "density": 1100.0,
        "thickness": 3.0,
    },
    "skin_eyelid": {
        "mu": 3.0e3,
        "kappa": 30.0e3,
        "density": 1050.0,
        "thickness": 0.8,
    },
    "skin_cheek": {
        "mu": 6.0e3,
        "kappa": 60.0e3,
        "density": 1100.0,
        "thickness": 2.0,
    },
    "subcutaneous_fat": {
        "mu": 1.0e3,
        "kappa": 50.0e3,
        "density": 920.0,
    },
    "cartilage_septal": {
        "C1": 0.5e6,        # Pa (Mooney-Rivlin C1)
        "C2": 0.1e6,        # Pa (Mooney-Rivlin C2)
        "kappa": 10.0e6,    # Pa
        "density": 1100.0,
    },
    "cartilage_upper_lateral": {
        "C1": 0.4e6,
        "C2": 0.08e6,
        "kappa": 8.0e6,
        "density": 1100.0,
    },
    "cartilage_lower_lateral": {
        "C1": 0.3e6,
        "C2": 0.06e6,
        "kappa": 6.0e6,
        "density": 1100.0,
    },
    "cartilage_ear": {
        "C1": 0.6e6,
        "C2": 0.12e6,
        "kappa": 12.0e6,
        "density": 1050.0,
    },
    "bone_nasal": {
        "E": 2.0e9,          # Pa (Young's modulus)
        "nu": 0.3,
        "density": 1800.0,
    },
    "bone_maxilla": {
        "E": 10.0e9,
        "nu": 0.3,
        "density": 1900.0,
    },
    "mucosa_nasal": {
        "mu": 2.0e3,
        "kappa": 20.0e3,
        "density": 1050.0,
        "thickness": 2.0,
    },
    "muscle_mimetic": {
        "mu": 5.0e3,
        "kappa": 50.0e3,
        "density": 1050.0,
    },
    "smas": {
        "mu": 20.0e3,
        "kappa": 200.0e3,
        "density": 1050.0,
        "thickness": 2.5,
    },
    "periosteum": {
        "mu": 50.0e3,
        "kappa": 500.0e3,
        "density": 1100.0,
        "thickness": 0.5,
    },
    "filler_ha_soft": {
        "mu": 0.2e3,
        "kappa": 100.0e3,
        "density": 1010.0,
        "viscosity": 80.0,   # Pa·s
    },
    "filler_ha_firm": {
        "mu": 0.8e3,
        "kappa": 200.0e3,
        "density": 1020.0,
        "viscosity": 200.0,
    },
    "graft_costal_cartilage": {
        "C1": 0.8e6,
        "C2": 0.16e6,
        "kappa": 16.0e6,
        "density": 1100.0,
    },
}
