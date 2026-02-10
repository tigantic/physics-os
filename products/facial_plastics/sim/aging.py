"""Long-horizon aging trajectory prediction for facial tissues.

Models the gradual tissue changes that occur over years to decades
after a surgical procedure, enabling long-term outcome prediction.

Biological processes modeled:

  1. **Collagen degradation** — progressive loss of collagen cross-links
     leading to reduced skin stiffness and elasticity.  Rate depends on
     UV exposure history, smoking status, and genetic factors.

  2. **Elastin fragmentation** — loss of elastic recoil in skin and
     connective tissue.  Primary driver of skin laxity.

  3. **Fat volume changes** — lipoatrophy (fat loss) and redistribution
     that alter facial contours over time.

  4. **Bone resorption** — gradual loss of bony support (orbital rim,
     maxilla, mandible, piriform aperture) that deepens midface hollowing.

  5. **Gravity-driven tissue descent** — creep deformation of skin and
     SMAS under gravitational loading.

  6. **Muscle atrophy** — age-related loss of mimetic muscle mass and
     tone affecting dynamic facial expression.

  7. **Graft resorption** — long-term cartilage/bone graft volume
     changes.  Autologous cartilage is relatively stable; rib grafts
     may warp over years.

The `AgingTrajectory` class computes tissue-property evolution at
arbitrary future time points (years), accounting for patient-specific
risk factors.  Results are returned as per-timepoint snapshots that
can be fed back into the FEM solver for long-term deformation
prediction.

References:
  - Mendelson & Wong (2013). Plast Reconstr Surg 131:516–530
  - Lambros (2007). Plast Reconstr Surg 120:1367–1376
  - Shaw et al (2010). Aesth Surg J 30:139–146
  - Kahn & Shaw (2019). Facial Plast Surg Clin N Am 27:87–101
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.types import StructureType

logger = logging.getLogger(__name__)


# ── Enumerations ──────────────────────────────────────────────────

class AgingFactor(Enum):
    """Individual factors that accelerate or decelerate aging."""
    UV_EXPOSURE = auto()            # photo-aging (Fitzpatrick zone)
    SMOKING = auto()                # nicotine-induced collagen degradation
    GENETICS = auto()               # family history of aging pattern
    BMI_CHANGE = auto()             # weight cycling affects fat pads
    SKIN_TYPE = auto()              # Fitzpatrick type (thicker skin ages slower)
    PREVIOUS_SURGERY = auto()       # scar tissue behaves differently
    HRT = auto()                    # hormone replacement affects collagen
    DIABETES = auto()               # altered wound healing and collagen quality


class GraftType(Enum):
    """Graft material types with different long-term stability."""
    AUTOLOGOUS_SEPTAL = auto()      # very stable
    AUTOLOGOUS_AURICULAR = auto()   # stable, some warping
    AUTOLOGOUS_RIB = auto()         # risk of warping
    IRRADIATED_RIB = auto()         # reduced warping risk, resorption risk
    POROUS_POLYETHYLENE = auto()    # permanent, no resorption
    EXPANDED_PTFE = auto()          # stable but infection risk
    FASCIA_LATA = auto()            # eventual resorption
    DICED_CARTILAGE_FASCIA = auto() # partial resorption


# ── Risk factor profile ──────────────────────────────────────────

@dataclass
class AgingRiskProfile:
    """Patient-specific aging risk factor quantification.

    Each factor is a dimensionless multiplier applied to the
    base aging rate:
      - 1.0 = average population
      - >1.0 = accelerated aging
      - <1.0 = decelerated aging
    """
    baseline_age_years: float = 45.0
    uv_multiplier: float = 1.0         # 0.5 = minimal sun; 2.0 = heavy sun
    smoking_multiplier: float = 1.0    # 1.0 = non-smoker; 2.0 = heavy smoker
    genetic_multiplier: float = 1.0
    bmi_change_per_year: float = 0.0   # kg/m² per year (+/-)
    skin_type_factor: float = 1.0      # 0.7 = Fitz V-VI; 1.3 = Fitz I-II
    previous_surgery_factor: float = 1.0
    hrt_factor: float = 1.0
    diabetes_factor: float = 1.0

    @property
    def composite_multiplier(self) -> float:
        """Combined aging rate multiplier (geometric mean of factors)."""
        factors = [
            self.uv_multiplier,
            self.smoking_multiplier,
            self.genetic_multiplier,
            self.skin_type_factor,
            self.previous_surgery_factor,
            self.hrt_factor,
            self.diabetes_factor,
        ]
        product = 1.0
        for f in factors:
            product *= max(f, 0.01)
        return product ** (1.0 / len(factors))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_age": self.baseline_age_years,
            "uv": self.uv_multiplier,
            "smoking": self.smoking_multiplier,
            "genetic": self.genetic_multiplier,
            "bmi_change_per_year": self.bmi_change_per_year,
            "skin_type": self.skin_type_factor,
            "previous_surgery": self.previous_surgery_factor,
            "hrt": self.hrt_factor,
            "diabetes": self.diabetes_factor,
            "composite": round(self.composite_multiplier, 3),
        }


# ── Tissue property evolution ─────────────────────────────────────

@dataclass
class TissueSnapshot:
    """Tissue properties at a single time point."""
    time_years: float
    patient_age: float
    collagen_fraction: float           # 0-1, fraction of original collagen
    elastin_fraction: float            # 0-1, fraction of original elastin
    skin_stiffness_multiplier: float   # relative to surgery-day stiffness
    skin_thickness_fraction: float     # relative to surgery-day thickness
    fat_volume_fraction: float         # relative to surgery-day volume
    bone_volume_fraction: float        # relative to surgery-day volume
    muscle_mass_fraction: float        # relative to surgery-day mass
    gravity_descent_mm: float          # cumulative gravitational sag
    graft_volume_fraction: float       # graft remaining volume

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_years": round(self.time_years, 2),
            "patient_age": round(self.patient_age, 1),
            "collagen": round(self.collagen_fraction, 4),
            "elastin": round(self.elastin_fraction, 4),
            "skin_stiffness": round(self.skin_stiffness_multiplier, 4),
            "skin_thickness": round(self.skin_thickness_fraction, 4),
            "fat_volume": round(self.fat_volume_fraction, 4),
            "bone_volume": round(self.bone_volume_fraction, 4),
            "muscle_mass": round(self.muscle_mass_fraction, 4),
            "gravity_descent_mm": round(self.gravity_descent_mm, 3),
            "graft_volume": round(self.graft_volume_fraction, 4),
        }


@dataclass
class AgingTrajectoryResult:
    """Full aging trajectory with multiple snapshots."""
    snapshots: List[TissueSnapshot] = field(default_factory=list)
    risk_profile: Optional[AgingRiskProfile] = None
    graft_type: Optional[GraftType] = None
    horizon_years: float = 0.0
    n_points: int = 0

    def at_year(self, year: float) -> Optional[TissueSnapshot]:
        """Interpolate tissue snapshot at a given year."""
        if not self.snapshots:
            return None
        if year <= self.snapshots[0].time_years:
            return self.snapshots[0]
        if year >= self.snapshots[-1].time_years:
            return self.snapshots[-1]

        # Linear interpolation between bracketing snapshots
        for i in range(len(self.snapshots) - 1):
            t0 = self.snapshots[i].time_years
            t1 = self.snapshots[i + 1].time_years
            if t0 <= year <= t1:
                alpha = (year - t0) / max(t1 - t0, 1e-12)
                s0 = self.snapshots[i]
                s1 = self.snapshots[i + 1]
                return TissueSnapshot(
                    time_years=year,
                    patient_age=s0.patient_age + alpha * (s1.patient_age - s0.patient_age),
                    collagen_fraction=s0.collagen_fraction + alpha * (s1.collagen_fraction - s0.collagen_fraction),
                    elastin_fraction=s0.elastin_fraction + alpha * (s1.elastin_fraction - s0.elastin_fraction),
                    skin_stiffness_multiplier=s0.skin_stiffness_multiplier + alpha * (s1.skin_stiffness_multiplier - s0.skin_stiffness_multiplier),
                    skin_thickness_fraction=s0.skin_thickness_fraction + alpha * (s1.skin_thickness_fraction - s0.skin_thickness_fraction),
                    fat_volume_fraction=s0.fat_volume_fraction + alpha * (s1.fat_volume_fraction - s0.fat_volume_fraction),
                    bone_volume_fraction=s0.bone_volume_fraction + alpha * (s1.bone_volume_fraction - s0.bone_volume_fraction),
                    muscle_mass_fraction=s0.muscle_mass_fraction + alpha * (s1.muscle_mass_fraction - s0.muscle_mass_fraction),
                    gravity_descent_mm=s0.gravity_descent_mm + alpha * (s1.gravity_descent_mm - s0.gravity_descent_mm),
                    graft_volume_fraction=s0.graft_volume_fraction + alpha * (s1.graft_volume_fraction - s0.graft_volume_fraction),
                )
        return self.snapshots[-1]

    def summary(self) -> str:
        if not self.snapshots:
            return "AgingTrajectory: no snapshots"
        last = self.snapshots[-1]
        return (
            f"AgingTrajectory: {self.horizon_years:.0f}yr horizon, "
            f"{self.n_points} points, "
            f"final collagen={last.collagen_fraction:.2f}, "
            f"descent={last.gravity_descent_mm:.1f}mm"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon_years": self.horizon_years,
            "n_points": self.n_points,
            "risk_profile": self.risk_profile.to_dict() if self.risk_profile else None,
            "graft_type": self.graft_type.name if self.graft_type else None,
            "snapshots": [s.to_dict() for s in self.snapshots],
        }


# ── Base aging rate functions ─────────────────────────────────────
#
# Each function returns a fraction remaining at a given age.
# All assume a starting fraction of 1.0 at the surgery age.

def _collagen_decay(
    years_post_op: float,
    age_at_surgery: float,
    rate_multiplier: float = 1.0,
) -> float:
    """Collagen content fraction remaining.

    After age 25, dermal collagen decreases ~1% per year.
    Photo-aging and smoking accelerate this by up to 2×.
    """
    # Base rate: ~1% per year after age 25
    base_rate_per_year = 0.01
    effective_rate = base_rate_per_year * rate_multiplier

    # Exponential decay from surgery time
    fraction = math.exp(-effective_rate * years_post_op)
    return max(fraction, 0.05)  # floor at 5%


def _elastin_decay(
    years_post_op: float,
    age_at_surgery: float,
    rate_multiplier: float = 1.0,
) -> float:
    """Elastin content fraction remaining.

    Elastin is not significantly regenerated after maturity.
    Degradation rate ~0.5% per year, heavily accelerated by UV.
    """
    base_rate = 0.005
    effective_rate = base_rate * rate_multiplier

    # Elastin loss accelerates with age
    age_factor = 1.0 + max((age_at_surgery + years_post_op) - 50.0, 0.0) * 0.01
    fraction = math.exp(-effective_rate * age_factor * years_post_op)
    return max(fraction, 0.1)


def _skin_stiffness_evolution(
    collagen_frac: float,
    elastin_frac: float,
) -> float:
    """Skin stiffness multiplier from collagen and elastin state.

    Stiffness is dominated by collagen network integrity.
    Elastin contributes to elastic recoil but not static stiffness.
    """
    # Stiffness scales roughly linearly with collagen for small losses,
    # but drops faster as the network becomes disconnected
    collagen_effect = collagen_frac ** 1.5  # nonlinear weakening
    elastin_effect = 0.8 + 0.2 * elastin_frac  # minor contribution

    return collagen_effect * elastin_effect


def _skin_thickness_evolution(
    years_post_op: float,
    age_at_surgery: float,
    rate_multiplier: float = 1.0,
) -> float:
    """Skin thickness fraction remaining.

    Skin thins ~6% per decade starting around age 40.
    """
    current_age = age_at_surgery + years_post_op
    effective_age = max(current_age - 40.0, 0.0)
    thinning_rate = 0.006 * rate_multiplier  # per year
    fraction = math.exp(-thinning_rate * effective_age)
    return max(fraction, 0.5)


def _fat_volume_evolution(
    years_post_op: float,
    age_at_surgery: float,
    bmi_change_per_year: float = 0.0,
) -> float:
    """Fat pad volume fraction remaining.

    Facial fat compartments undergo differential aging:
      - Deep medial cheek fat atrophies
      - Superficial fat may be preserved or increased
      - Net effect modeled as overall slow decline

    BMI changes modulate fat volume independently.
    """
    # Base lipoatrophy rate: ~0.5% per year after 35
    current_age = age_at_surgery + years_post_op
    active_years = max(current_age - 35.0, 0.0)
    base_fraction = math.exp(-0.005 * active_years)

    # BMI effect: ±1 BMI unit → ±3% fat volume (simplified)
    bmi_cumulative = bmi_change_per_year * years_post_op
    bmi_effect = 1.0 + 0.03 * bmi_cumulative

    fraction = base_fraction * max(bmi_effect, 0.5)
    return max(min(fraction, 1.5), 0.2)  # clamp


def _bone_resorption(
    years_post_op: float,
    age_at_surgery: float,
    rate_multiplier: float = 1.0,
) -> float:
    """Bony support volume fraction remaining.

    Key areas of facial bone resorption:
      - Pyriform aperture widens → nasal base broadens
      - Orbital rim recedes → increased eye show
      - Maxilla/mandible height decreases
      - Chin projection diminishes

    Rate: ~0.3% per year after age 40.
    """
    current_age = age_at_surgery + years_post_op
    active_years = max(current_age - 40.0, 0.0)
    rate = 0.003 * rate_multiplier
    fraction = math.exp(-rate * active_years)
    return max(fraction, 0.6)


def _muscle_atrophy(
    years_post_op: float,
    age_at_surgery: float,
    rate_multiplier: float = 1.0,
) -> float:
    """Mimetic muscle mass fraction remaining.

    Facial muscles undergo sarcopenia like skeletal muscles,
    but at a slower rate due to continuous use.
    Rate: ~0.2% per year after age 50.
    """
    current_age = age_at_surgery + years_post_op
    active_years = max(current_age - 50.0, 0.0)
    rate = 0.002 * rate_multiplier
    fraction = math.exp(-rate * active_years)
    return max(fraction, 0.7)


def _gravity_descent(
    years_post_op: float,
    skin_stiffness: float,
    elastin_fraction: float,
) -> float:
    """Cumulative gravitational tissue descent in mm.

    Soft tissue sag is a viscoelastic creep process driven by
    gravity on the skin–SMAS composite.  Rate increases as
    collagen/elastin support decreases.

    Typical values: 2-5 mm per decade for midface.
    """
    # Base creep rate: 0.3 mm/year (midface)
    base_rate = 0.3

    # Tissue weakening accelerates descent
    stiffness_effect = 1.0 / max(skin_stiffness, 0.1)
    elastin_effect = 1.0 / max(elastin_fraction, 0.1)

    # Combined effect (geometric mean)
    combined = math.sqrt(stiffness_effect * elastin_effect)

    # Time-dependent accumulation (integral of rate × combined)
    # Simple Euler integration approximate: rate * combined * t
    descent = base_rate * combined * years_post_op

    return descent


# ── Graft resorption models ──────────────────────────────────────

# Half-life of graft volume by type (in years)
# Null = no resorption (permanent implant)
GRAFT_HALF_LIVES: Dict[GraftType, Optional[float]] = {
    GraftType.AUTOLOGOUS_SEPTAL: None,          # stable
    GraftType.AUTOLOGOUS_AURICULAR: 50.0,       # very slow resorption
    GraftType.AUTOLOGOUS_RIB: 30.0,             # slow resorption + warping
    GraftType.IRRADIATED_RIB: 15.0,             # faster resorption
    GraftType.POROUS_POLYETHYLENE: None,         # permanent
    GraftType.EXPANDED_PTFE: None,               # permanent
    GraftType.FASCIA_LATA: 5.0,                 # significant resorption
    GraftType.DICED_CARTILAGE_FASCIA: 8.0,      # moderate resorption
}


def _graft_resorption(
    years_post_op: float,
    graft_type: GraftType,
) -> float:
    """Graft volume fraction remaining over time."""
    half_life = GRAFT_HALF_LIVES.get(graft_type)
    if half_life is None:
        return 1.0  # permanent
    if half_life <= 0.0:
        return 0.0

    # Exponential decay
    decay_rate = math.log(2.0) / half_life
    fraction = math.exp(-decay_rate * years_post_op)
    return max(fraction, 0.05)


# ── Aging trajectory engine ──────────────────────────────────────

class AgingTrajectory:
    """Long-horizon aging trajectory predictor.

    Computes tissue property evolution over a specified time horizon,
    accounting for patient-specific risk factors and graft characteristics.

    Usage::

        profile = AgingRiskProfile(
            baseline_age_years=50,
            uv_multiplier=1.5,
            smoking_multiplier=1.0,
        )
        predictor = AgingTrajectory(
            risk_profile=profile,
            graft_type=GraftType.AUTOLOGOUS_SEPTAL,
        )
        result = predictor.predict(horizon_years=20, n_points=50)
        snap_10yr = result.at_year(10)
    """

    def __init__(
        self,
        risk_profile: Optional[AgingRiskProfile] = None,
        graft_type: Optional[GraftType] = None,
    ) -> None:
        self._profile = risk_profile or AgingRiskProfile()
        self._graft_type = graft_type or GraftType.AUTOLOGOUS_SEPTAL

    @property
    def risk_profile(self) -> AgingRiskProfile:
        return self._profile

    @property
    def graft_type(self) -> GraftType:
        return self._graft_type

    def predict(
        self,
        horizon_years: float = 20.0,
        n_points: int = 50,
    ) -> AgingTrajectoryResult:
        """Generate the aging trajectory.

        Parameters
        ----------
        horizon_years : number of years to project forward
        n_points : number of time samples in the trajectory
        """
        if horizon_years <= 0.0:
            raise ValueError("horizon_years must be positive")
        if n_points < 2:
            raise ValueError("n_points must be >= 2")

        times = np.linspace(0.0, horizon_years, n_points)
        age0 = self._profile.baseline_age_years
        composite = self._profile.composite_multiplier

        snapshots: List[TissueSnapshot] = []

        for t in times:
            collagen = _collagen_decay(t, age0, composite)
            elastin = _elastin_decay(t, age0, composite * self._profile.uv_multiplier)
            stiffness = _skin_stiffness_evolution(collagen, elastin)
            thickness = _skin_thickness_evolution(t, age0, composite)
            fat = _fat_volume_evolution(t, age0, self._profile.bmi_change_per_year)
            bone = _bone_resorption(t, age0, self._profile.diabetes_factor)
            muscle = _muscle_atrophy(t, age0, composite)
            descent = _gravity_descent(t, stiffness, elastin)
            graft = _graft_resorption(t, self._graft_type)

            snapshots.append(TissueSnapshot(
                time_years=float(t),
                patient_age=age0 + float(t),
                collagen_fraction=collagen,
                elastin_fraction=elastin,
                skin_stiffness_multiplier=stiffness,
                skin_thickness_fraction=thickness,
                fat_volume_fraction=fat,
                bone_volume_fraction=bone,
                muscle_mass_fraction=muscle,
                gravity_descent_mm=descent,
                graft_volume_fraction=graft,
            ))

        return AgingTrajectoryResult(
            snapshots=snapshots,
            risk_profile=self._profile,
            graft_type=self._graft_type,
            horizon_years=horizon_years,
            n_points=n_points,
        )

    def predict_tissue_params(
        self,
        structure: StructureType,
        base_params: Dict[str, float],
        years_post_op: float,
    ) -> Dict[str, float]:
        """Modify tissue constitutive parameters for a given future time.

        Takes the surgery-day material parameters and returns modified
        parameters accounting for the aging process.

        This is used to feed into the FEM solver for long-term
        deformation prediction.
        """
        composite = self._profile.composite_multiplier
        age0 = self._profile.baseline_age_years

        collagen = _collagen_decay(years_post_op, age0, composite)
        elastin = _elastin_decay(years_post_op, age0, composite * self._profile.uv_multiplier)
        stiffness = _skin_stiffness_evolution(collagen, elastin)

        modified = dict(base_params)

        if structure in (
            StructureType.SKIN_ENVELOPE,
            StructureType.SKIN_THICK,
            StructureType.SKIN_THIN,
        ):
            # Skin softens as collagen degrades
            if "mu" in modified:
                modified["mu"] = base_params["mu"] * stiffness
            if "kappa" in modified:
                modified["kappa"] = base_params["kappa"] * max(stiffness, 0.3)

        elif structure == StructureType.FAT_SUBCUTANEOUS:
            fat = _fat_volume_evolution(years_post_op, age0, self._profile.bmi_change_per_year)
            # Fat stiffness changes slightly with volume
            if "mu" in modified:
                modified["mu"] = base_params["mu"] * (0.7 + 0.3 * fat)

        elif structure == StructureType.FAT_MALAR:
            fat = _fat_volume_evolution(years_post_op, age0, self._profile.bmi_change_per_year)
            if "mu" in modified:
                modified["mu"] = base_params["mu"] * (0.7 + 0.3 * fat)

        elif structure == StructureType.SMAS:
            if "mu" in modified:
                modified["mu"] = base_params["mu"] * (0.5 + 0.5 * collagen)

        elif structure == StructureType.MUSCLE_MIMETIC:
            muscle = _muscle_atrophy(years_post_op, age0, composite)
            if "mu" in modified:
                modified["mu"] = base_params["mu"] * muscle

        elif structure == StructureType.PERIOSTEUM:
            bone = _bone_resorption(years_post_op, age0, self._profile.diabetes_factor)
            if "mu" in modified:
                modified["mu"] = base_params["mu"] * bone

        return modified

    def compare_scenarios(
        self,
        base_profile: AgingRiskProfile,
        modified_profile: AgingRiskProfile,
        horizon_years: float = 20.0,
        n_points: int = 50,
    ) -> Dict[str, Any]:
        """Compare two aging scenarios (e.g., smoker vs. non-smoker).

        Returns both trajectories and the differential between them.
        """
        traj_base = AgingTrajectory(base_profile, self._graft_type)
        traj_mod = AgingTrajectory(modified_profile, self._graft_type)

        result_base = traj_base.predict(horizon_years, n_points)
        result_mod = traj_mod.predict(horizon_years, n_points)

        differentials: List[Dict[str, Any]] = []
        for s0, s1 in zip(result_base.snapshots, result_mod.snapshots):
            differentials.append({
                "time_years": s0.time_years,
                "delta_collagen": s1.collagen_fraction - s0.collagen_fraction,
                "delta_elastin": s1.elastin_fraction - s0.elastin_fraction,
                "delta_stiffness": s1.skin_stiffness_multiplier - s0.skin_stiffness_multiplier,
                "delta_fat": s1.fat_volume_fraction - s0.fat_volume_fraction,
                "delta_descent_mm": s1.gravity_descent_mm - s0.gravity_descent_mm,
            })

        return {
            "base": result_base.to_dict(),
            "modified": result_mod.to_dict(),
            "differentials": differentials,
        }
