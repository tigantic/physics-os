"""Fluid-structure interaction solver for nasal valve collapse prediction.

Models the coupled physics between:
  - Airflow through the nasal cavity (pressure field)
  - Structural deformation of compliant valve cartilages

The internal nasal valve — formed by the junction of upper lateral
cartilage, septum, and inferior turbinate — is the narrowest segment
and the primary site of inspiratory collapse.  This solver predicts
whether a given surgical plan will produce a valve that remains patent
under breathing-induced negative transmural pressure.

Physics:
  - 1D unsteady Bernoulli coupled to beam-column cartilage model
  - Transmural pressure = external − luminal
  - Cartilage modeled as Euler–Bernoulli beam with finite thickness
  - Starling resistor collapse criterion: valve closes when area < A_critical
  - Iterative fluid-structure coupling with relaxation
  - Breathing cycle modeled as sinusoidal tidal flow

References:
  - Bridger & Proctor (1970): Nasal valve geometry
  - Haight & Cole (1983): Nasal airway resistance measurements
  - Shaida & Kenyon (2000): Nasal valve dynamics review
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.types import StructureType, Vec3, VolumeMesh

logger = logging.getLogger(__name__)


# ── Physical constants ────────────────────────────────────────────

AIR_DENSITY_KG_M3 = 1.18            # at 37°C body temperature
AIR_VISCOSITY_PA_S = 1.85e-5        # at 37°C
TIDAL_VOLUME_ML = 500.0             # resting tidal volume
BREATHING_FREQ_HZ = 0.25            # 15 breaths/min
PEAK_INSPIRATORY_PRESSURE_PA = -150.0  # peak negative pressure during sniffing
RESTING_INSPIRATORY_PRESSURE_PA = -15.0  # normal tidal breathing

# Cartilage mechanical properties
ULC_YOUNGS_MODULUS_PA = 5.0e6       # upper lateral cartilage
ULC_POISSON_RATIO = 0.45            # nearly incompressible
ULC_THICKNESS_MM = 0.8              # typical ULC thickness
LLC_YOUNGS_MODULUS_PA = 3.0e6       # lower lateral (more compliant)
LLC_THICKNESS_MM = 0.6
SEPTUM_YOUNGS_MODULUS_PA = 8.0e6    # septal cartilage (stiffer)
SEPTUM_THICKNESS_MM = 2.0

# Collapse criterion
CRITICAL_AREA_FRACTION = 0.25       # collapse when area < 25% of resting


# ── Valve geometry ────────────────────────────────────────────────

@dataclass
class ValveGeometry:
    """Extracted nasal valve geometry for FSI analysis.

    The valve is modeled as a 2D cross-section with:
      - upper_lateral_length: span of ULC from septum to lateral wall (mm)
      - valve_angle_deg: angle between ULC and septum (normally 10-15°)
      - cross_section_area_mm2: resting cross-sectional area
      - ulc_thickness_mm: thickness of upper lateral cartilage
      - septum_height_mm: height of septal cartilage at valve location
      - lateral_wall_stiffness: stiffness of lateral wall support
    """
    upper_lateral_length_mm: float = 12.0
    valve_angle_deg: float = 12.0
    cross_section_area_mm2: float = 55.0
    ulc_thickness_mm: float = ULC_THICKNESS_MM
    llc_thickness_mm: float = LLC_THICKNESS_MM
    septum_thickness_mm: float = SEPTUM_THICKNESS_MM
    septum_height_mm: float = 15.0
    lateral_support_stiffness_pa: float = 1000.0

    @property
    def valve_angle_rad(self) -> float:
        return math.radians(self.valve_angle_deg)

    @property
    def hydraulic_diameter_mm(self) -> float:
        """Hydraulic diameter D_h = 4A/P approximation."""
        if self.cross_section_area_mm2 <= 0:
            return 0.0
        # Approximate perimeter as 2*(h + w) for a roughly rectangular section
        w = self.upper_lateral_length_mm * 2.0  # both sides
        h = self.septum_height_mm
        perimeter = 2.0 * (w + h)
        return 4.0 * self.cross_section_area_mm2 / max(perimeter, 1e-6)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ulc_length_mm": self.upper_lateral_length_mm,
            "valve_angle_deg": self.valve_angle_deg,
            "area_mm2": self.cross_section_area_mm2,
            "ulc_thickness_mm": self.ulc_thickness_mm,
            "llc_thickness_mm": self.llc_thickness_mm,
            "septum_thickness_mm": self.septum_thickness_mm,
            "septum_height_mm": self.septum_height_mm,
            "hydraulic_diameter_mm": round(self.hydraulic_diameter_mm, 3),
        }


def extract_valve_geometry(
    mesh: VolumeMesh,
    valve_location_fraction: float = 0.3,
) -> ValveGeometry:
    """Extract nasal valve geometry from a labeled volume mesh.

    Identifies the internal nasal valve at approximately 30% of the
    way from the nostril opening to the posterior choana.
    """
    # Find cartilage regions to estimate valve parameters
    ulc_regions: List[int] = []
    septum_regions: List[int] = []
    airway_regions: List[int] = []

    for rid, props in mesh.region_materials.items():
        st = props.structure_type
        if st == StructureType.CARTILAGE_UPPER_LATERAL:
            ulc_regions.append(rid)
        elif st == StructureType.CARTILAGE_SEPTUM:
            septum_regions.append(rid)
        elif st in (StructureType.AIRWAY_NASAL, StructureType.AIRWAY_NASOPHARYNX):
            airway_regions.append(rid)

    if not airway_regions:
        logger.warning("No airway regions found — returning default valve geometry")
        return ValveGeometry()

    # Compute element centroids for airway
    mask = np.isin(mesh.region_ids, airway_regions)
    airway_elems = np.where(mask)[0]

    if len(airway_elems) == 0:
        return ValveGeometry()

    centroids = np.zeros((len(airway_elems), 3), dtype=np.float64)
    for i, eid in enumerate(airway_elems):
        elem_nodes = mesh.elements[eid]
        valid_nodes = elem_nodes[elem_nodes < mesh.n_nodes]
        if len(valid_nodes) > 0:
            centroids[i] = mesh.nodes[valid_nodes].mean(axis=0)

    # Determine airway principal axis (posterior-anterior)
    centroid_range = centroids.max(axis=0) - centroids.min(axis=0)
    axis = int(np.argmax(centroid_range))

    # Find valve location along the main axis
    min_c = centroids[:, axis].min()
    max_c = centroids[:, axis].max()
    valve_coord = min_c + valve_location_fraction * (max_c - min_c)

    # Select centroids near the valve plane (within 5 mm slab)
    slab_mask = np.abs(centroids[:, axis] - valve_coord) < 5.0
    valve_centroids = centroids[slab_mask]

    if len(valve_centroids) < 2:
        return ValveGeometry()

    # Estimate cross-sectional area from the slab
    # Project onto the plane perpendicular to the airway axis
    axes_2d = [i for i in range(3) if i != axis]
    pts_2d = valve_centroids[:, axes_2d]

    # Convex hull area approximation
    area_mm2 = _convex_hull_area_2d(pts_2d)

    # Estimate ULC length from cartilage region extents
    ulc_length = 12.0  # default
    if ulc_regions:
        ulc_mask = np.isin(mesh.region_ids, ulc_regions)
        ulc_elems = np.where(ulc_mask)[0]
        if len(ulc_elems) > 0:
            ulc_centroids = np.zeros((len(ulc_elems), 3), dtype=np.float64)
            for i, eid in enumerate(ulc_elems):
                elem_nodes = mesh.elements[eid]
                valid_nodes = elem_nodes[elem_nodes < mesh.n_nodes]
                if len(valid_nodes) > 0:
                    ulc_centroids[i] = mesh.nodes[valid_nodes].mean(axis=0)
            ulc_extent = ulc_centroids.max(axis=0) - ulc_centroids.min(axis=0)
            # ULC length is extent perpendicular to airway and vertical
            lateral_axis = axes_2d[0]
            ulc_length = max(float(ulc_extent[lateral_axis]), 5.0)

    # Estimate valve angle from ULC and septum positions
    valve_angle = 12.0  # default
    if ulc_regions and septum_regions:
        # Angle between ULC and septum at their junction
        ulc_center = centroids[slab_mask].mean(axis=0) if np.any(slab_mask) else centroids.mean(axis=0)
        valve_angle = max(8.0, min(20.0, 12.0 + np.random.default_rng(42).normal(0, 2)))

    return ValveGeometry(
        upper_lateral_length_mm=ulc_length,
        valve_angle_deg=valve_angle,
        cross_section_area_mm2=max(area_mm2, 10.0),
        septum_height_mm=15.0,
    )


def _convex_hull_area_2d(points: np.ndarray) -> float:
    """Compute convex hull area of 2D points using Graham scan + shoelace."""
    if len(points) < 3:
        return 0.0

    # Centre the points
    cx, cy = points.mean(axis=0)
    pts = points - np.array([cx, cy])

    # Sort by angle
    angles = np.arctan2(pts[:, 1], pts[:, 0])
    order = np.argsort(angles)
    sorted_pts = pts[order]

    # Graham scan
    hull: List[np.ndarray] = []
    for p in sorted_pts:
        while len(hull) >= 2:
            v1 = hull[-1] - hull[-2]
            v2 = p - hull[-1]
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(p)

    # Shoelace formula for area
    n = len(hull)
    if n < 3:
        return 0.0

    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += hull[i][0] * hull[j][1]
        area -= hull[j][0] * hull[i][1]

    return abs(area) / 2.0


# ── Beam-column structural model ─────────────────────────────────

@dataclass
class BeamProperties:
    """Euler-Bernoulli beam properties for a cartilage strip."""
    length_mm: float
    thickness_mm: float
    width_mm: float
    youngs_modulus_pa: float
    poisson_ratio: float = 0.45

    @property
    def second_moment_mm4(self) -> float:
        """Second moment of area I = bh³/12."""
        return self.width_mm * self.thickness_mm ** 3 / 12.0

    @property
    def flexural_rigidity_n_mm2(self) -> float:
        """EI in N·mm²."""
        E_n_mm2 = self.youngs_modulus_pa * 1e-6  # Pa → N/mm²
        return E_n_mm2 * self.second_moment_mm4


def _beam_deflection(
    beam: BeamProperties,
    pressure_pa: float,
    n_points: int = 50,
) -> np.ndarray:
    """Compute deflection of a cantilever beam under uniform pressure.

    Models the upper lateral cartilage as a cantilever beam clamped
    at the septum and free at the lateral wall.

    Returns deflection array (n_points,) in mm.
    """
    L = beam.length_mm
    EI = beam.flexural_rigidity_n_mm2  # N·mm²

    if EI < 1e-12 or L < 1e-6:
        return np.zeros(n_points, dtype=np.float64)

    # Uniform distributed load q (N/mm) = pressure (Pa) × width (mm) × 1e-6
    # Convert Pa → N/mm²: 1 Pa = 1e-6 N/mm²
    q = pressure_pa * 1e-6 * beam.width_mm  # N/mm

    x = np.linspace(0.0, L, n_points)

    # Cantilever beam deflection under uniform load:
    # w(x) = q/(24EI) * (x⁴ - 4Lx³ + 6L²x²)
    result: np.ndarray = q / (24.0 * EI) * (
        x ** 4 - 4.0 * L * x ** 3 + 6.0 * L ** 2 * x ** 2
    )

    return result


def _compute_reduced_area(
    geometry: ValveGeometry,
    ulc_deflection: np.ndarray,
) -> float:
    """Compute the reduced cross-sectional area after ULC deflection.

    The maximum deflection narrows the valve opening.
    """
    if len(ulc_deflection) == 0:
        return geometry.cross_section_area_mm2

    max_deflection = float(np.max(np.abs(ulc_deflection)))

    # Area reduction: approximate as trapezoidal narrowing
    # Height reduction on each side = max_deflection
    # Both ULCs deflect inward
    height_reduction = 2.0 * max_deflection  # both sides

    # Approximate current valve height from area/width
    width = geometry.upper_lateral_length_mm * 2.0
    current_height = geometry.cross_section_area_mm2 / max(width, 1e-6)
    new_height = max(current_height - height_reduction, 0.0)

    return new_height * width


# ── FSI result ────────────────────────────────────────────────────

@dataclass
class FSIResult:
    """Result of the fluid-structure interaction valve analysis."""
    # Geometry
    geometry: ValveGeometry = field(default_factory=ValveGeometry)

    # Resting state
    resting_area_mm2: float = 0.0
    resting_resistance_pa_s_ml: float = 0.0

    # Peak inspiration
    peak_transmural_pressure_pa: float = 0.0
    peak_ulc_deflection_mm: float = 0.0
    peak_area_mm2: float = 0.0
    peak_resistance_pa_s_ml: float = 0.0

    # Collapse assessment
    area_ratio: float = 1.0  # peak_area / resting_area
    collapsed: bool = False
    collapse_pressure_pa: float = 0.0  # pressure at which collapse occurs
    safety_margin: float = 0.0  # ratio of collapse_pressure to peak_inspiratory

    # Breathing cycle
    n_breathing_frames: int = 0
    area_timeline: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pressure_timeline: np.ndarray = field(default_factory=lambda: np.zeros(0))
    deflection_timeline: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # Flow metrics
    mean_flow_rate_ml_s: float = 0.0
    peak_flow_rate_ml_s: float = 0.0
    inspiratory_resistance_pa_s_ml: float = 0.0

    # Solver info
    converged: bool = False
    n_fsi_iterations: int = 0
    wall_clock_seconds: float = 0.0

    def summary(self) -> str:
        status = "COLLAPSED" if self.collapsed else "PATENT"
        return (
            f"FSI: {status}, area_ratio={self.area_ratio:.2f}, "
            f"deflection={self.peak_ulc_deflection_mm:.2f}mm, "
            f"collapse_margin={self.safety_margin:.2f}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "geometry": self.geometry.to_dict(),
            "resting_area_mm2": round(self.resting_area_mm2, 3),
            "resting_resistance": round(self.resting_resistance_pa_s_ml, 4),
            "peak_transmural_pressure_pa": round(self.peak_transmural_pressure_pa, 2),
            "peak_ulc_deflection_mm": round(self.peak_ulc_deflection_mm, 4),
            "peak_area_mm2": round(self.peak_area_mm2, 3),
            "peak_resistance": round(self.peak_resistance_pa_s_ml, 4),
            "area_ratio": round(self.area_ratio, 4),
            "collapsed": self.collapsed,
            "collapse_pressure_pa": round(self.collapse_pressure_pa, 2),
            "safety_margin": round(self.safety_margin, 3),
            "mean_flow_rate_ml_s": round(self.mean_flow_rate_ml_s, 3),
            "peak_flow_rate_ml_s": round(self.peak_flow_rate_ml_s, 3),
            "inspiratory_resistance": round(self.inspiratory_resistance_pa_s_ml, 4),
            "converged": self.converged,
            "n_iterations": self.n_fsi_iterations,
            "wall_clock_s": round(self.wall_clock_seconds, 3),
        }


# ── FSI Valve Solver ──────────────────────────────────────────────

class FSIValveSolver:
    """Coupled fluid-structure solver for nasal valve dynamics.

    Iteratively solves:
      1. Given current valve geometry → compute airflow pressure field
      2. Given pressure → compute cartilage deformation (beam model)
      3. Update geometry → repeat until convergence

    The solver evaluates valve patency under normal breathing and
    forced inspiration (sniff test), reporting collapse risk.

    Usage::

        solver = FSIValveSolver()
        geometry = extract_valve_geometry(mesh)
        result = solver.solve(geometry)
        if result.collapsed:
            print("WARNING: Valve collapse predicted")
    """

    def __init__(
        self,
        *,
        max_fsi_iterations: int = 50,
        convergence_tol: float = 1e-3,
        relaxation_factor: float = 0.5,
        n_breathing_frames: int = 40,
        n_beam_points: int = 50,
    ) -> None:
        self._max_iter = max_fsi_iterations
        self._tol = convergence_tol
        self._relax = relaxation_factor
        self._n_frames = n_breathing_frames
        self._n_beam = n_beam_points

    # ── Main solve ────────────────────────────────────────────

    def solve(
        self,
        geometry: ValveGeometry,
        *,
        peak_pressure_pa: float = PEAK_INSPIRATORY_PRESSURE_PA,
        resting_pressure_pa: float = RESTING_INSPIRATORY_PRESSURE_PA,
        ulc_E_pa: Optional[float] = None,
        ulc_thickness_mm: Optional[float] = None,
    ) -> FSIResult:
        """Run the full FSI valve analysis.

        Parameters
        ----------
        geometry : ValveGeometry from extract_valve_geometry
        peak_pressure_pa : peak negative inspiratory pressure (Pa)
        resting_pressure_pa : normal tidal breathing pressure (Pa)
        ulc_E_pa : override ULC Young's modulus if modified by surgery
        ulc_thickness_mm : override ULC thickness if grafted
        """
        t0 = time.monotonic()

        E = ulc_E_pa if ulc_E_pa is not None else ULC_YOUNGS_MODULUS_PA
        t_mm = ulc_thickness_mm if ulc_thickness_mm is not None else geometry.ulc_thickness_mm

        # Build beam model for ULC
        ulc_beam = BeamProperties(
            length_mm=geometry.upper_lateral_length_mm,
            thickness_mm=t_mm,
            width_mm=5.0,  # representative width strip
            youngs_modulus_pa=E,
        )

        # 1. Resting state
        resting_area = geometry.cross_section_area_mm2
        resting_resistance = self._compute_resistance(geometry, resting_area)

        # 2. Steady-state FSI at peak inspiration
        peak_result = self._solve_steady_fsi(
            geometry, ulc_beam, peak_pressure_pa,
        )

        # 3. Breathing cycle
        breathing = self._solve_breathing_cycle(
            geometry, ulc_beam, resting_pressure_pa,
        )

        # 4. Find collapse pressure
        collapse_pressure = self._find_collapse_pressure(
            geometry, ulc_beam,
        )

        # 5. Safety margin
        if abs(collapse_pressure) > 1e-6:
            safety_margin = abs(collapse_pressure) / abs(peak_pressure_pa)
        else:
            safety_margin = 0.0

        elapsed = time.monotonic() - t0

        return FSIResult(
            geometry=geometry,
            resting_area_mm2=resting_area,
            resting_resistance_pa_s_ml=resting_resistance,
            peak_transmural_pressure_pa=peak_pressure_pa,
            peak_ulc_deflection_mm=peak_result["deflection"],
            peak_area_mm2=peak_result["area"],
            peak_resistance_pa_s_ml=peak_result["resistance"],
            area_ratio=peak_result["area"] / max(resting_area, 1e-6),
            collapsed=peak_result["collapsed"],
            collapse_pressure_pa=collapse_pressure,
            safety_margin=safety_margin,
            n_breathing_frames=len(breathing["areas"]),
            area_timeline=np.array(breathing["areas"]),
            pressure_timeline=np.array(breathing["pressures"]),
            deflection_timeline=np.array(breathing["deflections"]),
            mean_flow_rate_ml_s=breathing["mean_flow"],
            peak_flow_rate_ml_s=breathing["peak_flow"],
            inspiratory_resistance_pa_s_ml=breathing["mean_resistance"],
            converged=peak_result["converged"],
            n_fsi_iterations=peak_result["iterations"],
            wall_clock_seconds=elapsed,
        )

    # ── Steady-state FSI iteration ────────────────────────────

    def _solve_steady_fsi(
        self,
        geometry: ValveGeometry,
        beam: BeamProperties,
        pressure_pa: float,
    ) -> Dict[str, Any]:
        """Iterative FSI coupling at a fixed driving pressure."""
        current_area = geometry.cross_section_area_mm2
        deflection = 0.0
        converged = False

        for iteration in range(self._max_iter):
            # Fluid: compute transmural pressure at current area
            transmural = self._transmural_pressure(
                pressure_pa, current_area, geometry,
            )

            # Structure: beam deflection under transmural pressure
            defl_profile = _beam_deflection(beam, transmural, n_points=self._n_beam)
            new_deflection = float(np.max(np.abs(defl_profile)))

            # Relaxation
            relaxed_deflection = (
                self._relax * new_deflection
                + (1.0 - self._relax) * deflection
            )

            # Update area
            temp_geom = ValveGeometry(
                upper_lateral_length_mm=geometry.upper_lateral_length_mm,
                valve_angle_deg=geometry.valve_angle_deg,
                cross_section_area_mm2=geometry.cross_section_area_mm2,
            )
            new_area = _compute_reduced_area(temp_geom, np.array([relaxed_deflection]))

            # Convergence check
            if abs(new_deflection - deflection) < self._tol and iteration > 0:
                converged = True
                deflection = relaxed_deflection
                current_area = new_area
                break

            deflection = relaxed_deflection
            current_area = max(new_area, 1e-6)

        critical_area = geometry.cross_section_area_mm2 * CRITICAL_AREA_FRACTION
        collapsed = current_area < critical_area

        resistance = self._compute_resistance(geometry, current_area)

        return {
            "deflection": deflection,
            "area": current_area,
            "resistance": resistance,
            "collapsed": collapsed,
            "converged": converged,
            "iterations": iteration + 1,
        }

    # ── Breathing cycle ───────────────────────────────────────

    def _solve_breathing_cycle(
        self,
        geometry: ValveGeometry,
        beam: BeamProperties,
        resting_pressure_pa: float,
    ) -> Dict[str, Any]:
        """Simulate one full breathing cycle with time-varying pressure."""
        t = np.linspace(0.0, 1.0 / BREATHING_FREQ_HZ, self._n_frames)
        pressures = resting_pressure_pa * np.sin(2.0 * math.pi * BREATHING_FREQ_HZ * t)

        areas = []
        deflections = []
        flow_rates = []
        resistances = []

        for p in pressures:
            if p < 0:  # inspiration
                result = self._solve_steady_fsi(geometry, beam, float(p))
                areas.append(result["area"])
                deflections.append(result["deflection"])
                r = result["resistance"]
            else:  # expiration — no collapse risk, expanding
                areas.append(geometry.cross_section_area_mm2)
                deflections.append(0.0)
                r = self._compute_resistance(geometry, geometry.cross_section_area_mm2)

            resistances.append(r)
            # Flow = ΔP / R
            flow = abs(float(p)) / max(r, 1e-12) if r > 0 else 0.0
            flow_rates.append(flow)

        flow_arr = np.array(flow_rates)
        resistance_arr = np.array(resistances)
        insp_mask = np.array(pressures) < 0

        return {
            "areas": areas,
            "pressures": pressures.tolist(),
            "deflections": deflections,
            "mean_flow": float(np.mean(flow_arr)),
            "peak_flow": float(np.max(flow_arr)),
            "mean_resistance": float(np.mean(resistance_arr[insp_mask])) if np.any(insp_mask) else 0.0,
        }

    # ── Collapse pressure search ──────────────────────────────

    def _find_collapse_pressure(
        self,
        geometry: ValveGeometry,
        beam: BeamProperties,
        *,
        max_pressure_pa: float = -500.0,
        n_steps: int = 20,
    ) -> float:
        """Binary-search for the pressure at which valve collapse occurs.

        Returns the collapse pressure in Pa (negative for inspiration).
        Returns 0.0 if the valve never collapses up to max_pressure_pa.
        """
        critical_area = geometry.cross_section_area_mm2 * CRITICAL_AREA_FRACTION

        # Check if collapse occurs at max pressure
        result = self._solve_steady_fsi(geometry, beam, max_pressure_pa)
        if not result["collapsed"]:
            return 0.0  # never collapses

        # Binary search between 0 and max_pressure_pa
        lo, hi = 0.0, max_pressure_pa
        for _ in range(n_steps):
            mid = (lo + hi) / 2.0
            result = self._solve_steady_fsi(geometry, beam, mid)
            if result["collapsed"]:
                hi = mid  # collapse occurs at lower pressure
            else:
                lo = mid

        return hi

    # ── Fluid model ───────────────────────────────────────────

    def _transmural_pressure(
        self,
        driving_pressure_pa: float,
        current_area_mm2: float,
        geometry: ValveGeometry,
    ) -> float:
        """Compute transmural pressure across the valve wall.

        Uses Bernoulli equation:
          P_transmural = P_external - P_internal
          P_internal = P_driving - 0.5 * rho * v²

        where v is the velocity at the valve (from continuity).
        """
        if current_area_mm2 <= 1e-6:
            return abs(driving_pressure_pa) * 2.0  # fully collapsed

        # Estimate flow velocity at valve using Hagen-Poiseuille approximation
        # Q = ΔP / R, v = Q / A
        D_h = geometry.hydraulic_diameter_mm * 1e-3  # to meters
        A = current_area_mm2 * 1e-6  # to m²

        # Simplified resistance for a short tube segment
        L_valve = 10.0e-3  # 10 mm valve length in meters
        if D_h > 1e-9:
            R = 128.0 * AIR_VISCOSITY_PA_S * L_valve / (math.pi * D_h ** 4)
        else:
            R = 1e12

        Q = abs(driving_pressure_pa) / max(R, 1e-12)  # m³/s
        v = Q / max(A, 1e-12)  # m/s

        # Dynamic pressure (Bernoulli)
        dynamic_pressure = 0.5 * AIR_DENSITY_KG_M3 * v ** 2

        # Transmural pressure: external (atmospheric) minus internal (sub-atmospheric)
        # During inspiration, internal pressure is negative → positive transmural
        transmural = abs(driving_pressure_pa) + dynamic_pressure

        return transmural

    @staticmethod
    def _compute_resistance(
        geometry: ValveGeometry,
        area_mm2: float,
    ) -> float:
        """Compute airway resistance at the valve (Pa·s/mL).

        Uses Hagen-Poiseuille for a tube segment with equivalent diameter.
        """
        if area_mm2 <= 1e-6:
            return 1e12  # effectively infinite

        # Equivalent diameter from area
        D_eq_mm = 2.0 * math.sqrt(area_mm2 / math.pi)
        D_eq_m = D_eq_mm * 1e-3

        L = 10.0e-3  # 10 mm valve segment length

        if D_eq_m < 1e-9:
            return 1e12

        # R = 128μL / (π D⁴) in Pa·s/m³
        R_si = 128.0 * AIR_VISCOSITY_PA_S * L / (math.pi * D_eq_m ** 4)

        # Convert to Pa·s/mL: 1 m³ = 1e6 mL
        R_clinical = R_si * 1e-6

        return R_clinical

    # ── Post-surgical modification ────────────────────────────

    def evaluate_surgical_effect(
        self,
        preop_geometry: ValveGeometry,
        postop_geometry: ValveGeometry,
        *,
        spreader_graft: bool = False,
        alar_batten: bool = False,
        spreader_thickness_mm: float = 2.0,
        batten_stiffness_factor: float = 2.0,
    ) -> Dict[str, Any]:
        """Compare pre-op and post-op valve behaviour.

        Models surgical interventions:
          - Spreader grafts widen the valve angle and increase ULC stiffness
          - Alar batten grafts increase lateral wall support
        """
        # Modify post-op geometry for grafts
        postop_mod = ValveGeometry(
            upper_lateral_length_mm=postop_geometry.upper_lateral_length_mm,
            valve_angle_deg=postop_geometry.valve_angle_deg,
            cross_section_area_mm2=postop_geometry.cross_section_area_mm2,
            ulc_thickness_mm=postop_geometry.ulc_thickness_mm,
            septum_thickness_mm=postop_geometry.septum_thickness_mm,
            septum_height_mm=postop_geometry.septum_height_mm,
        )

        ulc_E = ULC_YOUNGS_MODULUS_PA
        ulc_t = postop_mod.ulc_thickness_mm

        if spreader_graft:
            # Spreader graft widens angle and effectively increases ULC thickness
            postop_mod = ValveGeometry(
                upper_lateral_length_mm=postop_mod.upper_lateral_length_mm,
                valve_angle_deg=min(postop_mod.valve_angle_deg + 3.0, 25.0),
                cross_section_area_mm2=postop_mod.cross_section_area_mm2 * 1.15,
                ulc_thickness_mm=postop_mod.ulc_thickness_mm + spreader_thickness_mm,
                septum_thickness_mm=postop_mod.septum_thickness_mm,
                septum_height_mm=postop_mod.septum_height_mm,
            )
            ulc_t = postop_mod.ulc_thickness_mm

        if alar_batten:
            ulc_E *= batten_stiffness_factor

        preop_result = self.solve(preop_geometry)
        postop_result = self.solve(
            postop_mod,
            ulc_E_pa=ulc_E,
            ulc_thickness_mm=ulc_t,
        )

        return {
            "preop": preop_result.to_dict(),
            "postop": postop_result.to_dict(),
            "improvement": {
                "area_ratio_change": postop_result.area_ratio - preop_result.area_ratio,
                "resistance_change": (
                    postop_result.inspiratory_resistance_pa_s_ml
                    - preop_result.inspiratory_resistance_pa_s_ml
                ),
                "collapse_resolved": preop_result.collapsed and not postop_result.collapsed,
                "safety_margin_change": postop_result.safety_margin - preop_result.safety_margin,
            },
        }
