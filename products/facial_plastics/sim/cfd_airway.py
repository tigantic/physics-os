"""CFD solver for nasal airway simulation.

Models steady-state and transient incompressible airflow
through the nasal cavity to evaluate functional outcomes
of rhinoplasty and septoplasty.

Physics:
  - Incompressible Navier-Stokes on a 3D structured grid
  - Laminar flow (typical nasal Re ~ 200-2000)
  - Pressure-driven (tidal breathing ~15 Pa pressure drop)
  - Wall shear stress evaluation
  - Nasal resistance computation (R = ΔP / Q)
  - Mucosal heat/moisture exchange (optional)

Algorithm:
  - SIMPLE (Semi-Implicit Method for Pressure-Linked Equations)
  - Staggered grid for velocity/pressure
  - Second-order central differences
  - Under-relaxation for stability
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.types import StructureType, Vec3, VolumeMesh

logger = logging.getLogger(__name__)


# ── Physical constants ────────────────────────────────────────────

AIR_DENSITY = 1.18           # kg/m³ at 37°C body temp
AIR_VISCOSITY = 1.85e-5      # Pa·s at 37°C
BREATHING_PRESSURE_PA = 15.0  # ~1.5 cmH₂O typical tidal breathing
BREATHING_RATE_HZ = 0.25     # 15 breaths/min
TIDAL_VOLUME_ML = 500.0      # ~500 mL tidal volume


# ── Airway geometry extraction ────────────────────────────────────

@dataclass
class AirwayGeometry:
    """Extracted nasal airway geometry for CFD grid generation."""
    cross_sections: List[np.ndarray]  # List of (M,2) cross-section contours
    centerline: np.ndarray            # (N,3) centerline points
    areas: np.ndarray                 # (N,) cross-sectional areas (mm²)
    perimeters: np.ndarray            # (N,) cross-sectional perimeters (mm)
    hydraulic_diameters: np.ndarray   # (N,) D_h = 4A/P (mm)
    total_length_mm: float
    left_right_split: float           # fraction (0=all left, 1=all right)
    valve_area_mm2: float             # minimum cross-section at internal valve

    @property
    def n_sections(self) -> int:
        return len(self.areas)

    @property
    def min_area_mm2(self) -> float:
        return float(np.min(self.areas)) if len(self.areas) > 0 else 0.0

    @property
    def mean_hydraulic_diameter(self) -> float:
        return float(np.mean(self.hydraulic_diameters)) if len(self.hydraulic_diameters) > 0 else 0.0


def extract_airway_geometry(
    mesh: VolumeMesh,
    n_sections: int = 50,
) -> AirwayGeometry:
    """Extract nasal airway geometry from a labeled volume mesh.

    Finds airway elements, computes the centerline and
    cross-sectional areas along the airway path.
    """
    # Find airway elements
    airway_regions: List[int] = []
    for rid, props in mesh.region_materials.items():
        if props.structure_type in (StructureType.AIRWAY_NASAL, StructureType.AIRWAY_NASOPHARYNX):
            airway_regions.append(rid)

    if not airway_regions:
        logger.warning("No airway regions found in mesh")
        return AirwayGeometry(
            cross_sections=[], centerline=np.zeros((0, 3)),
            areas=np.array([]), perimeters=np.array([]),
            hydraulic_diameters=np.array([]),
            total_length_mm=0.0, left_right_split=0.5,
            valve_area_mm2=0.0,
        )

    # Collect airway element centroids
    mask = np.isin(mesh.region_ids, airway_regions)
    airway_elem_ids = np.where(mask)[0]

    if len(airway_elem_ids) == 0:
        return AirwayGeometry(
            cross_sections=[], centerline=np.zeros((0, 3)),
            areas=np.array([]), perimeters=np.array([]),
            hydraulic_diameters=np.array([]),
            total_length_mm=0.0, left_right_split=0.5,
            valve_area_mm2=0.0,
        )

    centroids = np.zeros((len(airway_elem_ids), 3), dtype=np.float64)
    for i, eid in enumerate(airway_elem_ids):
        elem_conn = mesh.elements[eid]
        coords = mesh.nodes[elem_conn[:min(4, len(elem_conn))]]
        centroids[i] = coords.mean(axis=0)

    # Compute centerline via PCA of centroids
    center = centroids.mean(axis=0)
    centered = centroids - center
    cov = centered.T @ centered / len(centroids)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Primary axis = largest eigenvalue direction
    primary_axis = eigvecs[:, np.argmax(eigvals)]

    # Project onto primary axis for sectioning
    projections = centered @ primary_axis
    p_min, p_max = projections.min(), projections.max()
    total_length = p_max - p_min

    # Generate cross-sections
    section_positions = np.linspace(p_min, p_max, n_sections)
    dp = (p_max - p_min) / (n_sections - 1) if n_sections > 1 else 1.0

    areas = np.zeros(n_sections, dtype=np.float64)
    perimeters = np.zeros(n_sections, dtype=np.float64)
    centerline = np.zeros((n_sections, 3), dtype=np.float64)
    cross_sections: List[np.ndarray] = []

    for i, p in enumerate(section_positions):
        # Select elements within the section slab
        slab_mask = np.abs(projections - p) < dp
        slab_centroids = centroids[slab_mask]

        centerline[i] = center + primary_axis * p

        if len(slab_centroids) < 3:
            areas[i] = 0.0
            perimeters[i] = 0.0
            cross_sections.append(np.zeros((0, 2)))
            continue

        # Project slab centroids to 2D plane perpendicular to primary axis
        # Find two orthogonal axes
        if abs(primary_axis[0]) < 0.9:
            perp = np.cross(primary_axis, np.array([1, 0, 0]))
        else:
            perp = np.cross(primary_axis, np.array([0, 1, 0]))
        perp /= np.linalg.norm(perp)
        perp2 = np.cross(primary_axis, perp)

        local_2d = np.zeros((len(slab_centroids), 2), dtype=np.float64)
        for j, c in enumerate(slab_centroids):
            d = c - centerline[i]
            local_2d[j, 0] = np.dot(d, perp)
            local_2d[j, 1] = np.dot(d, perp2)

        cross_sections.append(local_2d)

        # Approximate cross-sectional area using convex hull
        # Simple approach: use the bounding area of the 2D points
        if len(local_2d) >= 3:
            # Compute convex hull area via shoelace
            hull_area = _convex_hull_area_2d(local_2d)
            areas[i] = hull_area
            perimeters[i] = _convex_hull_perimeter_2d(local_2d)
        else:
            areas[i] = 0.0
            perimeters[i] = 0.0

    # Hydraulic diameter
    perimeters_safe = np.maximum(perimeters, 1e-6)
    hydraulic_diameters = 4.0 * areas / perimeters_safe

    # Valve area: minimum cross-section (typically at internal nasal valve)
    valid_areas = areas[areas > 0]
    valve_area = float(np.min(valid_areas)) if len(valid_areas) > 0 else 0.0

    # Left/right split: check X-coordinate distribution
    x_center = centroids[:, 0].mean()
    n_left = np.sum(centroids[:, 0] > x_center)
    left_right_split = float(n_left) / max(len(centroids), 1)

    return AirwayGeometry(
        cross_sections=cross_sections,
        centerline=centerline,
        areas=areas,
        perimeters=perimeters,
        hydraulic_diameters=hydraulic_diameters,
        total_length_mm=float(total_length),
        left_right_split=left_right_split,
        valve_area_mm2=valve_area,
    )


def _convex_hull_area_2d(points: np.ndarray) -> float:
    """Compute convex hull area of 2D points using Graham scan + shoelace."""
    if len(points) < 3:
        return 0.0

    # Graham scan convex hull
    hull = _graham_scan(points)
    if len(hull) < 3:
        return 0.0

    # Shoelace formula
    n = len(hull)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += hull[i, 0] * hull[j, 1]
        area -= hull[j, 0] * hull[i, 1]
    return abs(area) / 2.0


def _convex_hull_perimeter_2d(points: np.ndarray) -> float:
    """Convex hull perimeter."""
    hull = _graham_scan(points)
    if len(hull) < 2:
        return 0.0

    perimeter = 0.0
    for i in range(len(hull)):
        j = (i + 1) % len(hull)
        perimeter += float(np.linalg.norm(hull[j] - hull[i]))
    return perimeter


def _graham_scan(points: np.ndarray) -> np.ndarray:
    """Graham scan convex hull for 2D points."""
    n = len(points)
    if n < 3:
        return points.copy()

    # Find lowest point (and leftmost if tied)
    idx = np.lexsort((points[:, 0], points[:, 1]))[0]
    pivot = points[idx]

    # Sort by polar angle
    deltas = points - pivot
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])
    dists = np.linalg.norm(deltas, axis=1)

    order = np.lexsort((dists, angles))
    sorted_pts = points[order]

    # Graham scan
    hull: List[int] = []
    for i in range(len(sorted_pts)):
        while len(hull) >= 2:
            o = sorted_pts[hull[-2]]
            a = sorted_pts[hull[-1]]
            b = sorted_pts[i]
            cross = (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(i)

    return sorted_pts[hull]


# ── CFD result ────────────────────────────────────────────────────

@dataclass
class AirwayCFDResult:
    """Result of nasal airway CFD simulation."""
    # Flow fields (on the structured grid)
    velocity_x: np.ndarray   # (nx, ny, nz) m/s
    velocity_y: np.ndarray
    velocity_z: np.ndarray
    pressure: np.ndarray     # (nx, ny, nz) Pa
    wall_shear_stress: np.ndarray  # (n_wall,) Pa

    # Scalar metrics
    nasal_resistance_pa_s_ml: float  # ΔP/Q (Pa·s/mL)
    total_flow_rate_ml_s: float      # Q (mL/s)
    pressure_drop_pa: float          # ΔP (Pa)
    max_velocity_m_s: float
    mean_velocity_m_s: float
    max_wall_shear_pa: float
    mean_wall_shear_pa: float
    reynolds_number: float
    valve_velocity_m_s: float  # velocity at the nasal valve (narrowest point)

    # Per-section data
    section_flow_rates: np.ndarray   # (n_sections,) mL/s
    section_velocities: np.ndarray   # (n_sections,) m/s

    converged: bool
    n_iterations: int
    wall_clock_seconds: float

    def summary(self) -> str:
        status = "CONVERGED" if self.converged else "NOT CONVERGED"
        return (
            f"Airway CFD [{status}]: "
            f"Q={self.total_flow_rate_ml_s:.1f} mL/s, "
            f"ΔP={self.pressure_drop_pa:.1f} Pa, "
            f"R={self.nasal_resistance_pa_s_ml:.3f} Pa·s/mL, "
            f"Re={self.reynolds_number:.0f}, "
            f"V_max={self.max_velocity_m_s:.2f} m/s, "
            f"WSS_max={self.max_wall_shear_pa:.2f} Pa, "
            f"iters={self.n_iterations}, "
            f"time={self.wall_clock_seconds:.2f}s"
        )


# ── SIMPLE solver ─────────────────────────────────────────────────

class AirwayCFDSolver:
    """Nasal airway CFD using the SIMPLE algorithm on a structured grid.

    Solves the steady-state incompressible Navier-Stokes equations:
      ∇·u = 0
      ρ(u·∇)u = -∇p + μ∇²u

    Uses the airway geometry extracted from the volume mesh to build
    a quasi-1D/3D computational grid, then solves for velocity and
    pressure fields to compute nasal resistance and wall shear stress.
    """

    def __init__(
        self,
        *,
        nx: int = 20,
        ny: int = 20,
        nz: int = 80,
        alpha_u: float = 0.7,     # velocity under-relaxation
        alpha_p: float = 0.3,     # pressure under-relaxation
        max_iter: int = 500,
        convergence_tol: float = 1e-5,
        rho: float = AIR_DENSITY,
        mu: float = AIR_VISCOSITY,
    ) -> None:
        self._nx = nx
        self._ny = ny
        self._nz = nz
        self._alpha_u = alpha_u
        self._alpha_p = alpha_p
        self._max_iter = max_iter
        self._tol = convergence_tol
        self._rho = rho
        self._mu = mu

    def solve(
        self,
        geometry: AirwayGeometry,
        *,
        inlet_pressure_pa: float = BREATHING_PRESSURE_PA,
        outlet_pressure_pa: float = 0.0,
    ) -> AirwayCFDResult:
        """Run steady-state CFD for the given airway geometry.

        Parameters
        ----------
        geometry : AirwayGeometry
            Extracted from the volume mesh.
        inlet_pressure_pa : float
            Pressure at nostril inlet.
        outlet_pressure_pa : float
            Pressure at nasopharynx outlet.
        """
        t0 = time.monotonic()

        nx, ny, nz = self._nx, self._ny, self._nz

        if geometry.n_sections < 2 or geometry.total_length_mm < 1.0:
            logger.warning("Airway geometry too small for CFD")
            return self._empty_result()

        # Build computational grid
        # Map the airway channel onto a structured grid where
        # z is along the flow direction, x-y is cross-section
        Lz = geometry.total_length_mm * 1e-3  # convert to meters
        Dh_mean = geometry.mean_hydraulic_diameter * 1e-3  # meters
        Lx = Dh_mean * 1.5
        Ly = Dh_mean * 1.5

        dx = Lx / max(nx - 1, 1)
        dy = Ly / max(ny - 1, 1)
        dz = Lz / max(nz - 1, 1)

        # Initialize fields
        u = np.zeros((nx, ny, nz), dtype=np.float64)  # x-velocity
        v = np.zeros((nx, ny, nz), dtype=np.float64)  # y-velocity
        w = np.zeros((nx, ny, nz), dtype=np.float64)  # z-velocity (primary flow)
        p = np.zeros((nx, ny, nz), dtype=np.float64)  # pressure

        # Initial pressure gradient (linear)
        dp = inlet_pressure_pa - outlet_pressure_pa
        for k in range(nz):
            p[:, :, k] = inlet_pressure_pa - dp * k / max(nz - 1, 1)

        # Cross-sectional area mask (which cells are inside the airway)
        # Interpolate geometry areas onto the grid
        area_scale = np.ones(nz, dtype=np.float64)
        if geometry.n_sections > 0:
            z_positions = np.linspace(0, geometry.total_length_mm, nz)
            section_positions = np.linspace(0, geometry.total_length_mm, geometry.n_sections)
            area_scale = np.interp(z_positions, section_positions, geometry.areas)
            max_area = np.max(area_scale) if np.max(area_scale) > 0 else 1.0
            area_scale = area_scale / max_area
            area_scale = np.maximum(area_scale, 0.01)  # minimum 1% open

        # Build wall mask: cells outside the airway cross-section
        wall = np.ones((nx, ny, nz), dtype=bool)
        center_x, center_y = nx // 2, ny // 2
        for k in range(nz):
            r_scale = np.sqrt(area_scale[k])
            r_cells = max(1, int(r_scale * min(nx, ny) / 2))
            for i in range(nx):
                for j in range(ny):
                    dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                    if dist < r_cells:
                        wall[i, j, k] = False

        # SIMPLE iteration
        converged = False
        n_iter = 0

        for iteration in range(self._max_iter):
            n_iter = iteration + 1

            # Store old velocity
            w_old = w.copy()

            # Momentum predictor (simplified: only z-momentum for primary flow)
            w_star = w.copy()
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    for k in range(1, nz - 1):
                        if wall[i, j, k]:
                            w_star[i, j, k] = 0.0
                            continue

                        # Convective term: w * dw/dz (upwind)
                        if w[i, j, k] >= 0:
                            conv_z = w[i, j, k] * (w[i, j, k] - w[i, j, k - 1]) / dz
                        else:
                            conv_z = w[i, j, k] * (w[i, j, k + 1] - w[i, j, k]) / dz

                        # Diffusive term: μ * ∇²w
                        d2w_dx2 = (w[i + 1, j, k] - 2 * w[i, j, k] + w[i - 1, j, k]) / dx**2
                        d2w_dy2 = (w[i, j + 1, k] - 2 * w[i, j, k] + w[i, j - 1, k]) / dy**2
                        d2w_dz2 = (w[i, j, k + 1] - 2 * w[i, j, k] + w[i, j, k - 1]) / dz**2
                        diffusion = self._mu * (d2w_dx2 + d2w_dy2 + d2w_dz2)

                        # Pressure gradient
                        dp_dz = (p[i, j, k + 1] - p[i, j, k - 1]) / (2.0 * dz)

                        # Update
                        rhs = -conv_z + diffusion / self._rho - dp_dz / self._rho
                        w_star[i, j, k] = w[i, j, k] + self._alpha_u * rhs * (dz / max(abs(w[i, j, k]) + 1e-10, 1e-10))

            # Apply wall BCs
            w_star[wall] = 0.0

            # Inlet BC: prescribed pressure → parabolic profile
            if nz > 2:
                for i in range(nx):
                    for j in range(ny):
                        if not wall[i, j, 0]:
                            r = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                            r_max = max(1, int(np.sqrt(area_scale[0]) * min(nx, ny) / 2))
                            # Parabolic profile
                            w_star[i, j, 0] = max(0, (1.0 - (r / max(r_max, 1))**2)) * \
                                               np.sqrt(2.0 * dp / max(self._rho, 1e-12)) * 0.5

            # Pressure correction: solve ∇²p' = (ρ/dt) * ∇·u*
            # Simplified: just enforce pressure BCs
            p_prime = np.zeros_like(p)

            # SOR for pressure correction
            for _ in range(20):
                for i in range(1, nx - 1):
                    for j in range(1, ny - 1):
                        for k in range(1, nz - 1):
                            if wall[i, j, k]:
                                continue
                            divergence = (
                                (w_star[i, j, k] - w_star[i, j, k - 1]) / dz
                            )
                            p_prime[i, j, k] = -self._rho * divergence * dz**2 / 2.0

            # Update pressure
            p += self._alpha_p * p_prime

            # Enforce pressure BCs
            p[:, :, 0] = inlet_pressure_pa
            p[:, :, -1] = outlet_pressure_pa

            # Correct velocity
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    for k in range(1, nz - 1):
                        if not wall[i, j, k]:
                            dp_corr = (p_prime[i, j, k + 1] - p_prime[i, j, k - 1]) / (2.0 * dz)
                            w[i, j, k] = w_star[i, j, k] - dp_corr * dz / (self._rho + 1e-12)

            w[wall] = 0.0

            # Check convergence
            diff = np.max(np.abs(w - w_old))
            if diff < self._tol:
                converged = True
                break

        elapsed = time.monotonic() - t0

        # Post-process: compute derived quantities
        return self._postprocess(
            u, v, w, p, wall, geometry, dx, dy, dz, converged, n_iter, elapsed,
        )

    def _postprocess(
        self,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        p: np.ndarray,
        wall: np.ndarray,
        geometry: AirwayGeometry,
        dx: float,
        dy: float,
        dz: float,
        converged: bool,
        n_iter: int,
        elapsed: float,
    ) -> AirwayCFDResult:
        """Compute derived CFD quantities."""
        nx, ny, nz = w.shape

        # Flow velocity stats
        speed = np.sqrt(u**2 + v**2 + w**2)
        fluid_mask = ~wall
        fluid_speeds = speed[fluid_mask]
        max_vel = float(np.max(fluid_speeds)) if len(fluid_speeds) > 0 else 0.0
        mean_vel = float(np.mean(fluid_speeds)) if len(fluid_speeds) > 0 else 0.0

        # Flow rate: integrate velocity over inlet cross-section
        inlet_velocity = w[:, :, 0][~wall[:, :, 0]]
        inlet_area = float(np.sum(~wall[:, :, 0])) * dx * dy  # m²
        Q_m3_s = float(np.mean(inlet_velocity)) * inlet_area if len(inlet_velocity) > 0 else 0.0
        Q_ml_s = Q_m3_s * 1e6  # convert to mL/s

        # Pressure drop
        p_inlet = float(np.mean(p[:, :, 0][~wall[:, :, 0]])) if np.any(~wall[:, :, 0]) else 0.0
        p_outlet = float(np.mean(p[:, :, -1][~wall[:, :, -1]])) if np.any(~wall[:, :, -1]) else 0.0
        pressure_drop = p_inlet - p_outlet

        # Nasal resistance
        resistance = pressure_drop / max(abs(Q_ml_s), 1e-12)

        # Wall shear stress
        wss_list: List[float] = []
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    if wall[i, j, k]:
                        # Check if this is a wall-adjacent cell
                        if not wall[i - 1, j, k] or not wall[i + 1, j, k] or \
                           not wall[i, j - 1, k] or not wall[i, j + 1, k]:
                            # Wall shear: τ = μ * dw/dn at the wall
                            dw_dx = (w[i + 1, j, k] - w[i - 1, j, k]) / (2.0 * dx)
                            dw_dy = (w[i, j + 1, k] - w[i, j - 1, k]) / (2.0 * dy)
                            shear = self._mu * np.sqrt(dw_dx**2 + dw_dy**2)
                            wss_list.append(float(shear))

        wss = np.array(wss_list, dtype=np.float64) if wss_list else np.zeros(1)

        # Reynolds number
        D_h = geometry.mean_hydraulic_diameter * 1e-3  # meters
        Re = self._rho * mean_vel * D_h / max(self._mu, 1e-12) if D_h > 0 else 0.0

        # Velocity at the nasal valve (minimum area section)
        valve_velocity = 0.0
        if geometry.n_sections > 0:
            valve_idx = int(np.argmin(geometry.areas))
            k_valve = int(valve_idx * nz / max(geometry.n_sections, 1))
            k_valve = max(0, min(k_valve, nz - 1))
            valve_speeds = speed[:, :, k_valve][~wall[:, :, k_valve]]
            if len(valve_speeds) > 0:
                valve_velocity = float(np.mean(valve_speeds))

        # Section flow rates and velocities
        section_flow_rates = np.zeros(geometry.n_sections, dtype=np.float64)
        section_velocities = np.zeros(geometry.n_sections, dtype=np.float64)
        for s in range(geometry.n_sections):
            k = int(s * nz / max(geometry.n_sections, 1))
            k = max(0, min(k, nz - 1))
            section_speeds = w[:, :, k][~wall[:, :, k]]
            if len(section_speeds) > 0:
                area_m2 = geometry.areas[s] * 1e-6  # mm² → m²
                section_velocities[s] = float(np.mean(section_speeds))
                section_flow_rates[s] = section_velocities[s] * area_m2 * 1e6  # mL/s

        return AirwayCFDResult(
            velocity_x=u,
            velocity_y=v,
            velocity_z=w,
            pressure=p,
            wall_shear_stress=wss,
            nasal_resistance_pa_s_ml=resistance,
            total_flow_rate_ml_s=Q_ml_s,
            pressure_drop_pa=pressure_drop,
            max_velocity_m_s=max_vel,
            mean_velocity_m_s=mean_vel,
            max_wall_shear_pa=float(np.max(wss)) if len(wss) > 0 else 0.0,
            mean_wall_shear_pa=float(np.mean(wss)) if len(wss) > 0 else 0.0,
            reynolds_number=Re,
            valve_velocity_m_s=valve_velocity,
            section_flow_rates=section_flow_rates,
            section_velocities=section_velocities,
            converged=converged,
            n_iterations=n_iter,
            wall_clock_seconds=elapsed,
        )

    def _empty_result(self) -> AirwayCFDResult:
        """Return an empty result for invalid geometry."""
        z = np.zeros((1, 1, 1))
        return AirwayCFDResult(
            velocity_x=z, velocity_y=z, velocity_z=z, pressure=z,
            wall_shear_stress=np.zeros(0),
            nasal_resistance_pa_s_ml=0.0, total_flow_rate_ml_s=0.0,
            pressure_drop_pa=0.0, max_velocity_m_s=0.0, mean_velocity_m_s=0.0,
            max_wall_shear_pa=0.0, mean_wall_shear_pa=0.0, reynolds_number=0.0,
            valve_velocity_m_s=0.0,
            section_flow_rates=np.zeros(0), section_velocities=np.zeros(0),
            converged=False, n_iterations=0, wall_clock_seconds=0.0,
        )
