"""Safety metrics for surgical simulation outcomes.

Evaluates biomechanical safety to ensure simulated procedures
stay within physiologically safe limits:

  - Stress limits (von Mises, hydrostatic) per tissue type
  - Strain limits (principal, volumetric) → tissue damage thresholds
  - Vascularity preservation (distance to named vessels, perfusion risk)
  - Nerve proximity analysis
  - Cartilage viability (bending strain limits)
  - Skin tension analysis (flap viability, necrosis risk)
  - Osteotomy stability (bone fragment displacement)
  - Global safety index (weighted composite)

Threshold values from published biomechanics literature
(Richmon et al. 2005, Moriarity et al. 2014, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.types import (
    ClinicalMeasurement,
    StructureType,
    TissueProperties,
    Vec3,
    VolumeMesh,
)
from ..sim.fem_soft_tissue import FEMResult, _von_mises, _principal_strains

logger = logging.getLogger(__name__)


# ── Tissue-specific safety thresholds ────────────────────────────
# Units: stress in Pa, strain dimensionless

@dataclass(frozen=True)
class SafetyThreshold:
    """Per-tissue safety limits."""
    max_von_mises_pa: float
    max_principal_strain: float
    max_hydrostatic_tension_pa: float
    max_volumetric_strain: float
    description: str


SAFETY_THRESHOLDS: Dict[StructureType, SafetyThreshold] = {
    StructureType.SKIN_ENVELOPE: SafetyThreshold(
        max_von_mises_pa=150.0e3,       # skin yield ~150-300 kPa
        max_principal_strain=0.30,       # 30% stretch before damage
        max_hydrostatic_tension_pa=100.0e3,
        max_volumetric_strain=0.10,
        description="skin envelope",
    ),
    StructureType.SKIN_THICK: SafetyThreshold(
        max_von_mises_pa=200.0e3,
        max_principal_strain=0.25,
        max_hydrostatic_tension_pa=150.0e3,
        max_volumetric_strain=0.08,
        description="thick skin (nasal tip)",
    ),
    StructureType.SKIN_THIN: SafetyThreshold(
        max_von_mises_pa=120.0e3,
        max_principal_strain=0.35,
        max_hydrostatic_tension_pa=80.0e3,
        max_volumetric_strain=0.12,
        description="thin skin (dorsum)",
    ),
    StructureType.CARTILAGE_SEPTUM: SafetyThreshold(
        max_von_mises_pa=8.0e6,         # cartilage yield ~5-15 MPa
        max_principal_strain=0.10,       # 10% strain → fracture risk
        max_hydrostatic_tension_pa=5.0e6,
        max_volumetric_strain=0.05,
        description="septal cartilage",
    ),
    StructureType.CARTILAGE_UPPER_LATERAL: SafetyThreshold(
        max_von_mises_pa=6.0e6,
        max_principal_strain=0.12,
        max_hydrostatic_tension_pa=4.0e6,
        max_volumetric_strain=0.06,
        description="upper lateral cartilage",
    ),
    StructureType.CARTILAGE_LOWER_LATERAL: SafetyThreshold(
        max_von_mises_pa=5.0e6,
        max_principal_strain=0.15,
        max_hydrostatic_tension_pa=3.0e6,
        max_volumetric_strain=0.07,
        description="lower lateral cartilage (alar)",
    ),
    StructureType.CARTILAGE_ALAR: SafetyThreshold(
        max_von_mises_pa=4.0e6,
        max_principal_strain=0.15,
        max_hydrostatic_tension_pa=2.5e6,
        max_volumetric_strain=0.08,
        description="alar cartilage",
    ),
    StructureType.BONE_NASAL: SafetyThreshold(
        max_von_mises_pa=120.0e6,       # cortical bone yield ~100-200 MPa
        max_principal_strain=0.02,       # 2% = fracture threshold
        max_hydrostatic_tension_pa=80.0e6,
        max_volumetric_strain=0.01,
        description="nasal bone",
    ),
    StructureType.BONE_MAXILLA: SafetyThreshold(
        max_von_mises_pa=170.0e6,
        max_principal_strain=0.02,
        max_hydrostatic_tension_pa=100.0e6,
        max_volumetric_strain=0.01,
        description="maxillary bone",
    ),
    StructureType.MUCOSA_NASAL: SafetyThreshold(
        max_von_mises_pa=30.0e3,
        max_principal_strain=0.40,       # mucosal tissue is very distensible
        max_hydrostatic_tension_pa=20.0e3,
        max_volumetric_strain=0.15,
        description="nasal mucosa",
    ),
    StructureType.PERIOSTEUM: SafetyThreshold(
        max_von_mises_pa=500.0e3,
        max_principal_strain=0.15,
        max_hydrostatic_tension_pa=300.0e3,
        max_volumetric_strain=0.05,
        description="periosteum",
    ),
    StructureType.PERICHONDRIUM: SafetyThreshold(
        max_von_mises_pa=400.0e3,
        max_principal_strain=0.18,
        max_hydrostatic_tension_pa=250.0e3,
        max_volumetric_strain=0.06,
        description="perichondrium",
    ),
    StructureType.FAT_SUBCUTANEOUS: SafetyThreshold(
        max_von_mises_pa=10.0e3,        # fat is very soft
        max_principal_strain=0.60,
        max_hydrostatic_tension_pa=5.0e3,
        max_volumetric_strain=0.20,
        description="subcutaneous fat",
    ),
    StructureType.MUSCLE_MIMETIC: SafetyThreshold(
        max_von_mises_pa=50.0e3,
        max_principal_strain=0.40,
        max_hydrostatic_tension_pa=30.0e3,
        max_volumetric_strain=0.10,
        description="mimetic muscles",
    ),
    StructureType.SMAS: SafetyThreshold(
        max_von_mises_pa=100.0e3,
        max_principal_strain=0.25,
        max_hydrostatic_tension_pa=60.0e3,
        max_volumetric_strain=0.08,
        description="SMAS layer",
    ),
    StructureType.VESSEL_ARTERY: SafetyThreshold(
        max_von_mises_pa=200.0e3,       # arterial wall ultimate stress
        max_principal_strain=0.50,
        max_hydrostatic_tension_pa=150.0e3,
        max_volumetric_strain=0.15,
        description="arterial vessel wall",
    ),
}

# Default threshold for structures not in the table
DEFAULT_THRESHOLD = SafetyThreshold(
    max_von_mises_pa=100.0e3,
    max_principal_strain=0.30,
    max_hydrostatic_tension_pa=50.0e3,
    max_volumetric_strain=0.10,
    description="generic tissue",
)

# Named vessels to check proximity
CRITICAL_VESSELS = [
    ("lateral_nasal_artery", StructureType.VESSEL_ARTERY),
    ("angular_artery", StructureType.VESSEL_ARTERY),
    ("dorsal_nasal_artery", StructureType.VESSEL_ARTERY),
    ("columellar_artery", StructureType.VESSEL_ARTERY),
    ("infraorbital_artery", StructureType.VESSEL_ARTERY),
]

MINIMUM_VESSEL_CLEARANCE_MM = 2.0
MINIMUM_NERVE_CLEARANCE_MM = 3.0


# ── Safety analysis dataclasses ──────────────────────────────────

@dataclass
class StressViolation:
    """A single stress/strain safety violation."""
    element_id: int
    structure: StructureType
    metric_name: str
    actual_value: float
    threshold_value: float
    severity: str  # "warning", "critical"
    position_mm: Optional[np.ndarray] = None

    @property
    def ratio(self) -> float:
        if self.threshold_value > 0:
            return self.actual_value / self.threshold_value
        return float("inf")

    def description(self) -> str:
        return (
            f"{self.severity.upper()}: {self.metric_name} on {self.structure.value} "
            f"element {self.element_id}: {self.actual_value:.2e} "
            f"(limit {self.threshold_value:.2e}, ratio {self.ratio:.2f})"
        )


@dataclass
class VascularRisk:
    """Vascular proximity / perfusion risk assessment."""
    vessel_name: str
    min_distance_mm: float
    is_compromised: bool
    risk_level: str  # "safe", "caution", "danger"
    element_ids: List[int] = field(default_factory=list)


@dataclass
class NerveRisk:
    """Nerve proximity risk assessment."""
    min_distance_mm: float
    is_at_risk: bool
    risk_level: str
    element_ids: List[int] = field(default_factory=list)


@dataclass
class SkinTensionAnalysis:
    """Skin flap tension analysis for necrosis risk."""
    max_tension_pa: float = 0.0
    mean_tension_pa: float = 0.0
    area_above_threshold_pct: float = 0.0
    necrosis_risk: str = "low"  # "low", "moderate", "high"
    critical_region: str = ""

    def assess_risk(self) -> str:
        # Skin necrosis threshold: ~50 kPa sustained tension
        if self.max_tension_pa > 100.0e3:
            self.necrosis_risk = "high"
        elif self.max_tension_pa > 50.0e3:
            self.necrosis_risk = "moderate"
        else:
            self.necrosis_risk = "low"
        return self.necrosis_risk


@dataclass
class OsteotomyStability:
    """Bone fragment stability after osteotomy."""
    max_fragment_displacement_mm: float = 0.0
    mean_fragment_displacement_mm: float = 0.0
    max_gap_mm: float = 0.0
    is_stable: bool = True

    def assess_stability(self) -> bool:
        # Fragment displacement > 2mm = unstable
        self.is_stable = (
            self.max_fragment_displacement_mm < 2.0
            and self.max_gap_mm < 1.0
        )
        return self.is_stable


@dataclass
class SafetyReport:
    """Complete safety assessment."""
    stress_violations: List[StressViolation] = field(default_factory=list)
    vascular_risks: List[VascularRisk] = field(default_factory=list)
    nerve_risks: List[NerveRisk] = field(default_factory=list)
    skin_tension: SkinTensionAnalysis = field(default_factory=SkinTensionAnalysis)
    osteotomy: OsteotomyStability = field(default_factory=OsteotomyStability)

    overall_safety_index: float = 100.0  # 0–100 (100 = fully safe)
    critical_count: int = 0
    warning_count: int = 0
    is_safe: bool = True

    def compute_safety_index(self) -> float:
        """Compute weighted safety index."""
        # Start at 100, deduct for violations
        idx = 100.0

        # Stress violations
        for v in self.stress_violations:
            if v.severity == "critical":
                idx -= 15.0 * v.ratio
                self.critical_count += 1
            else:
                idx -= 5.0 * max(0, v.ratio - 0.8)
                self.warning_count += 1

        # Vascular risks
        for vr in self.vascular_risks:
            if vr.risk_level == "danger":
                idx -= 20.0
            elif vr.risk_level == "caution":
                idx -= 5.0

        # Nerve risks
        for nr in self.nerve_risks:
            if nr.risk_level == "danger":
                idx -= 15.0
            elif nr.risk_level == "caution":
                idx -= 3.0

        # Skin tension
        if self.skin_tension.necrosis_risk == "high":
            idx -= 25.0
        elif self.skin_tension.necrosis_risk == "moderate":
            idx -= 10.0

        # Osteotomy stability
        if not self.osteotomy.is_stable:
            idx -= 15.0

        self.overall_safety_index = max(0.0, min(100.0, idx))
        self.is_safe = self.overall_safety_index >= 60.0 and self.critical_count == 0
        return self.overall_safety_index

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_safety_index": self.overall_safety_index,
            "is_safe": self.is_safe,
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
            "stress_violations": [
                {
                    "element_id": v.element_id,
                    "structure": v.structure.value,
                    "metric": v.metric_name,
                    "actual": v.actual_value,
                    "threshold": v.threshold_value,
                    "ratio": v.ratio,
                    "severity": v.severity,
                }
                for v in self.stress_violations
            ],
            "vascular_risks": [
                {
                    "vessel": vr.vessel_name,
                    "min_distance_mm": vr.min_distance_mm,
                    "risk_level": vr.risk_level,
                }
                for vr in self.vascular_risks
            ],
            "skin_tension": {
                "max_tension_pa": self.skin_tension.max_tension_pa,
                "necrosis_risk": self.skin_tension.necrosis_risk,
            },
            "osteotomy": {
                "max_displacement_mm": self.osteotomy.max_fragment_displacement_mm,
                "is_stable": self.osteotomy.is_stable,
            },
        }

    def summary(self) -> str:
        return (
            f"Safety Index: {self.overall_safety_index:.1f}/100 "
            f"({'SAFE' if self.is_safe else 'UNSAFE'}), "
            f"critical={self.critical_count}, warnings={self.warning_count}, "
            f"skin_risk={self.skin_tension.necrosis_risk}, "
            f"osteotomy_stable={self.osteotomy.is_stable}"
        )


# ── Main safety metrics calculator ───────────────────────────────

class SafetyMetrics:
    """Evaluate biomechanical safety of a simulation result.

    Checks all tissue regions against their safety thresholds
    and identifies potential complications.
    """

    def __init__(self, mesh: VolumeMesh) -> None:
        self._mesh = mesh
        self._n_elems = mesh.n_elements
        self._n_nodes = mesh.n_nodes

        # Build element-to-structure mapping
        self._elem_struct: Dict[int, StructureType] = {}
        for eid in range(self._n_elems):
            rid = int(mesh.region_ids[eid])
            if rid in mesh.region_materials:
                self._elem_struct[eid] = mesh.region_materials[rid].structure_type

    def evaluate(self, fem_result: FEMResult) -> SafetyReport:
        """Run complete safety evaluation."""
        report = SafetyReport()

        # Stress/strain limit check
        self._check_stress_strain(fem_result, report)

        # Vascular proximity
        self._check_vascular_proximity(fem_result, report)

        # Nerve proximity
        self._check_nerve_proximity(fem_result, report)

        # Skin tension
        self._analyze_skin_tension(fem_result, report)

        # Osteotomy stability
        self._check_osteotomy_stability(fem_result, report)

        report.compute_safety_index()
        logger.info("Safety assessment: %s", report.summary())
        return report

    def _check_stress_strain(
        self,
        fem: FEMResult,
        report: SafetyReport,
    ) -> None:
        """Check per-element stress and strain against thresholds."""
        warning_ratio = 0.80  # 80% of threshold = warning

        for eid in range(min(self._n_elems, len(fem.stresses))):
            struct = self._elem_struct.get(eid)
            if struct is None:
                continue

            threshold = SAFETY_THRESHOLDS.get(struct, DEFAULT_THRESHOLD)
            stress = fem.stresses[eid]
            strain = fem.strains[eid]

            # Von Mises stress
            vm = _von_mises(stress)
            if vm > threshold.max_von_mises_pa:
                report.stress_violations.append(StressViolation(
                    element_id=eid,
                    structure=struct,
                    metric_name="von_mises_stress",
                    actual_value=vm,
                    threshold_value=threshold.max_von_mises_pa,
                    severity="critical",
                ))
            elif vm > warning_ratio * threshold.max_von_mises_pa:
                report.stress_violations.append(StressViolation(
                    element_id=eid,
                    structure=struct,
                    metric_name="von_mises_stress",
                    actual_value=vm,
                    threshold_value=threshold.max_von_mises_pa,
                    severity="warning",
                ))

            # Principal strain
            prin = _principal_strains(strain)
            max_prin = abs(prin[0])  # largest principal strain
            if max_prin > threshold.max_principal_strain:
                report.stress_violations.append(StressViolation(
                    element_id=eid,
                    structure=struct,
                    metric_name="principal_strain",
                    actual_value=max_prin,
                    threshold_value=threshold.max_principal_strain,
                    severity="critical",
                ))
            elif max_prin > warning_ratio * threshold.max_principal_strain:
                report.stress_violations.append(StressViolation(
                    element_id=eid,
                    structure=struct,
                    metric_name="principal_strain",
                    actual_value=max_prin,
                    threshold_value=threshold.max_principal_strain,
                    severity="warning",
                ))

            # Hydrostatic stress (tension: positive = bad for tissue adhesion)
            hydrostatic = (stress[0] + stress[1] + stress[2]) / 3.0
            if hydrostatic > threshold.max_hydrostatic_tension_pa:
                report.stress_violations.append(StressViolation(
                    element_id=eid,
                    structure=struct,
                    metric_name="hydrostatic_tension",
                    actual_value=hydrostatic,
                    threshold_value=threshold.max_hydrostatic_tension_pa,
                    severity="critical" if hydrostatic > 1.5 * threshold.max_hydrostatic_tension_pa else "warning",
                ))

            # Volumetric strain
            vol_strain = strain[0] + strain[1] + strain[2]
            if abs(vol_strain) > threshold.max_volumetric_strain:
                report.stress_violations.append(StressViolation(
                    element_id=eid,
                    structure=struct,
                    metric_name="volumetric_strain",
                    actual_value=abs(vol_strain),
                    threshold_value=threshold.max_volumetric_strain,
                    severity="warning",
                ))

    def _check_vascular_proximity(
        self,
        fem: FEMResult,
        report: SafetyReport,
    ) -> None:
        """Check distance of high-stress regions from named vessels."""
        # Find vessel elements
        vessel_elements: Dict[str, List[int]] = {}
        for eid, struct in self._elem_struct.items():
            if struct in (StructureType.VESSEL_ARTERY, StructureType.VESSEL_VEIN):
                name = struct.value
                vessel_elements.setdefault(name, []).append(eid)

        if not vessel_elements:
            return

        # Find centroids of vessel elements
        vessel_centroids: Dict[str, np.ndarray] = {}
        for name, eids in vessel_elements.items():
            centroids = []
            for eid in eids:
                if eid < self._n_elems:
                    conn = self._mesh.elements[eid]
                    centroid = self._mesh.nodes[conn].mean(axis=0)
                    centroids.append(centroid)
            if centroids:
                vessel_centroids[name] = np.array(centroids)

        # Find high-stress / high-displacement elements
        high_stress_eids = []
        for eid in range(min(self._n_elems, len(fem.stresses))):
            vm = _von_mises(fem.stresses[eid])
            elem_struct = self._elem_struct.get(eid)
            if elem_struct is None:
                threshold = DEFAULT_THRESHOLD
            else:
                threshold = SAFETY_THRESHOLDS.get(elem_struct, DEFAULT_THRESHOLD)
            if vm > 0.5 * threshold.max_von_mises_pa:
                high_stress_eids.append(eid)

        if not high_stress_eids:
            return

        # Compute min distances from high-stress regions to vessels
        stressed_centroids = []
        for eid in high_stress_eids:
            if eid < self._n_elems:
                conn = self._mesh.elements[eid]
                stressed_centroids.append(self._mesh.nodes[conn].mean(axis=0))

        if not stressed_centroids:
            return

        stressed_pts = np.array(stressed_centroids)

        for name, v_pts in vessel_centroids.items():
            # Min distance from any stressed element to any vessel element
            min_d = float("inf")
            for sp in stressed_pts:
                dists = np.linalg.norm(v_pts - sp, axis=1)
                min_d = min(min_d, float(dists.min()))

            if min_d < MINIMUM_VESSEL_CLEARANCE_MM:
                risk_level = "danger"
                is_comp = True
            elif min_d < 2 * MINIMUM_VESSEL_CLEARANCE_MM:
                risk_level = "caution"
                is_comp = False
            else:
                risk_level = "safe"
                is_comp = False

            report.vascular_risks.append(VascularRisk(
                vessel_name=name,
                min_distance_mm=min_d,
                is_compromised=is_comp,
                risk_level=risk_level,
            ))

    def _check_nerve_proximity(
        self,
        fem: FEMResult,
        report: SafetyReport,
    ) -> None:
        """Check nerve proximity to high-displacement regions."""
        nerve_eids = [
            eid for eid, s in self._elem_struct.items()
            if s == StructureType.NERVE
        ]
        if not nerve_eids:
            return

        nerve_centroids = []
        for eid in nerve_eids:
            if eid < self._n_elems:
                conn = self._mesh.elements[eid]
                nerve_centroids.append(self._mesh.nodes[conn].mean(axis=0))

        if not nerve_centroids:
            return
        nerve_pts = np.array(nerve_centroids)

        # Find high-displacement nodes
        disp_mag = np.linalg.norm(fem.displacements, axis=1)
        threshold_disp = max(1.0, float(np.percentile(disp_mag, 90)))  # mm
        high_disp_nodes = np.where(disp_mag > threshold_disp)[0]

        if len(high_disp_nodes) == 0:
            return

        high_disp_pts = self._mesh.nodes[high_disp_nodes]
        min_d = float("inf")
        for np_pt in nerve_pts:
            dists = np.linalg.norm(high_disp_pts - np_pt, axis=1)
            min_d = min(min_d, float(dists.min()))

        if min_d < MINIMUM_NERVE_CLEARANCE_MM:
            risk_level = "danger"
        elif min_d < 2 * MINIMUM_NERVE_CLEARANCE_MM:
            risk_level = "caution"
        else:
            risk_level = "safe"

        report.nerve_risks.append(NerveRisk(
            min_distance_mm=min_d,
            is_at_risk=(risk_level != "safe"),
            risk_level=risk_level,
        ))

    def _analyze_skin_tension(
        self,
        fem: FEMResult,
        report: SafetyReport,
    ) -> None:
        """Analyze skin flap tension for necrosis risk."""
        sta = SkinTensionAnalysis()

        skin_types = {
            StructureType.SKIN_ENVELOPE,
            StructureType.SKIN_THICK,
            StructureType.SKIN_THIN,
        }

        skin_stresses = []
        for eid, struct in self._elem_struct.items():
            if struct in skin_types and eid < len(fem.stresses):
                vm = _von_mises(fem.stresses[eid])
                skin_stresses.append(vm)

        if skin_stresses:
            arr = np.array(skin_stresses)
            sta.max_tension_pa = float(np.max(arr))
            sta.mean_tension_pa = float(np.mean(arr))
            sta.area_above_threshold_pct = float(
                np.sum(arr > 50.0e3) / len(arr) * 100.0
            )
            sta.assess_risk()

            if sta.necrosis_risk in ("moderate", "high"):
                # Find critical region
                max_eid = -1
                max_vm = 0.0
                for eid, struct in self._elem_struct.items():
                    if struct in skin_types and eid < len(fem.stresses):
                        vm = _von_mises(fem.stresses[eid])
                        if vm > max_vm:
                            max_vm = vm
                            max_eid = eid
                if max_eid >= 0:
                    struct_found = self._elem_struct.get(max_eid)
                    sta.critical_region = struct_found.value if struct_found else "unknown"

        report.skin_tension = sta

    def _check_osteotomy_stability(
        self,
        fem: FEMResult,
        report: SafetyReport,
    ) -> None:
        """Check post-osteotomy bone fragment stability."""
        ost = OsteotomyStability()

        bone_types = {
            StructureType.BONE_NASAL,
            StructureType.BONE_MAXILLA,
            StructureType.BONE_FRONTAL,
            StructureType.BONE_ZYGOMATIC,
        }

        bone_disps = []
        for eid, struct in self._elem_struct.items():
            if struct in bone_types and eid < self._n_elems:
                conn = self._mesh.elements[eid]
                valid = conn[conn < len(fem.displacements)]
                if len(valid) > 0:
                    elem_disps = np.linalg.norm(fem.displacements[valid], axis=1)
                    bone_disps.extend(elem_disps.tolist())

        if bone_disps:
            arr = np.array(bone_disps)
            ost.max_fragment_displacement_mm = float(np.max(arr))
            ost.mean_fragment_displacement_mm = float(np.mean(arr))
            # Gap: max difference between adjacent bone element displacements
            ost.max_gap_mm = float(np.max(arr) - np.min(arr)) if len(arr) > 1 else 0.0
            ost.assess_stability()

        report.osteotomy = ost
