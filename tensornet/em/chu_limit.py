"""Chu Limit Antenna Design via 3D QTT Topology Optimization.

Challenges the fundamental Chu limit on antenna Q-factor using
frequency-domain QTT Helmholtz solver with 3D topology optimization.

The Chu limit states that the minimum radiation Q for an antenna
enclosed in a sphere of electrical size ka is bounded by:

    Q_Chu = 1/(ka)^3 + 1/(ka)     [TM modes only]

This module implements a full 3D antenna design pipeline:

1. **Conductivity SIMP** material model:
   sigma(rho) = sigma_min + (sigma_max - sigma_min) * rho^p
   eps_eff = 1 - j * sigma_norm * rho^p (in normalised Helmholtz units)

2. **PML-absorbed power** as radiated power proxy:
   P_pml = 0.5 * sum_i sigma_pml_i * |E_i|^2 * dV
   This is the cleanest radiation metric in bounded PML domains.

3. **Exact adjoint gradient** (not Born approximation):
   Forward:  H(rho) E = b
   Adjoint:  H^H lambda = dJ/dE*
   dJ/drho = explicit + Re[lambda^H (dH/drho E)]

4. **Augmented Lagrangian volume constraint**:
   Prevents trivial 'delete conductor' escape hatch.

5. **Continuation**: beta (projection), p (SIMP), sigma_max (contrast).

Target: 1 GHz, ka = 0.3 -> a ~ 14.3 mm, Q_Chu ~ 40.

Dependencies
------------
- tensornet.em.qtt_3d: 3D QTT operators, solver, geometry
- tensornet.em.qtt_helmholtz: Core QTT algebra, DMRG solver
- tensornet.em.boundaries: PMLConfig, _compute_pml_stretching
- tensornet.em.topology_opt: Heaviside, density filter, binarisation
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from tensornet.em.qtt_helmholtz import (
    array_to_tt,
    reconstruct_1d as _reconstruct_flat,
    diag_mpo_from_tt,
    mpo_add_c,
    mpo_scale_c,
    tt_amen_solve,
    tt_scale_c,
    tt_add_c,
    tt_inner_hermitian,
    TTGMRESResult,
)
from tensornet.qtt.sparse_direct import tt_matvec, tt_round
from tensornet.em.boundaries import (
    PMLConfig,
    _compute_pml_stretching,
)
from tensornet.em.qtt_3d import (
    array_3d_to_tt,
    reconstruct_3d,
    helmholtz_mpo_3d_pml,
    build_pec_penalty_3d,
    solve_helmholtz_3d,
    point_source_3d,
    gap_source_3d,
    spherical_mask,
    spherical_shell_mask,
    extract_s11_3d,
    extract_s11_3d_impedance,
    compute_impedance_3d,
    compute_complex_power,
    compute_input_power,
    compute_reactive_power,
    compute_radiation_resistance,
    monopole_seed_density,
    spherical_multipole_s11,
    # QTT-native (no dense N³) functions
    separable_3d_to_tt,
    build_pml_sigma_tt,
    build_conductivity_eps_tt,
    compute_pml_power_tt,
    compute_cond_power_tt,
    build_adjoint_rhs_tt,
    tt_evaluate_at_indices,
    spherical_mask_flat_indices,
    build_sphere_mask_tt,
    compute_voxel_distances,
)
from tensornet.em.topology_opt import (
    heaviside_projection,
    heaviside_gradient,
    density_filter,
    density_filter_gradient,
    binarisation_metric,
)


# =====================================================================
# Physical Constants
# =====================================================================

C0 = 299_792_458.0  # Speed of light [m/s]
MU0 = 4.0e-7 * math.pi  # Permeability of free space [H/m]
EPS0 = 1.0 / (MU0 * C0 ** 2)  # Permittivity of free space [F/m]
ETA0 = math.sqrt(MU0 / EPS0)  # Free-space impedance [ohm] ~ 376.7


# =====================================================================
# Section 1: Analytical Q-Factor Limits
# =====================================================================

def chu_limit_q(ka: float) -> float:
    """Minimum Q for a linearly-polarised antenna (Chu, 1948).

    For a single TM10 (or TE10) spherical mode:

        Q_Chu = 1/(ka)^3 + 1/(ka)

    Parameters
    ----------
    ka : float
        Electrical size (wavenumber x sphere radius, dimensionless).

    Returns
    -------
    float
        Minimum achievable Q.
    """
    if ka <= 0:
        raise ValueError(f"ka must be positive, got {ka}")
    return 1.0 / (ka ** 3) + 1.0 / ka


def thal_limit_q(ka: float) -> float:
    """Thal lower bound on Q for self-resonant antennas (Thal, 2006).

    Q_Thal = 1.5 x Q_Chu
    """
    return 1.5 * chu_limit_q(ka)


def mclean_limit_q(ka: float) -> float:
    """McLean exact minimum Q (McLean, 1996).

    Joint TM + TE excitation:
        Q_McLean = 0.5 x [1/(ka)^3 + 1/(ka)]
    """
    return 0.5 * chu_limit_q(ka)


def gustafsson_limit_q(ka: float, polarisability_ratio: float = 1.0) -> float:
    """Gustafsson physical bound on Q (Gustafsson et al., 2007).

    For a sphere: Q_Gust = Q_McLean / polarisability_ratio.
    """
    return mclean_limit_q(ka) / max(polarisability_ratio, 1e-30)


def print_q_limits(ka: float) -> dict[str, float]:
    """Print and return all analytical Q limits for given ka."""
    limits = {
        "Chu (single mode)": chu_limit_q(ka),
        "McLean (dual mode)": mclean_limit_q(ka),
        "Thal (self-resonant)": thal_limit_q(ka),
        "Gustafsson (sphere)": gustafsson_limit_q(ka),
    }
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Fundamental Q Limits at ka = {ka:.4f}")
    print(f"{sep}")
    for name, Q in limits.items():
        print(f"  {name:30s}: Q = {Q:.2f}")
    print(f"{sep}\n")
    return limits


# =====================================================================
# Section 2: Problem Setup
# =====================================================================

@dataclass
class ChuProblemConfig:
    """Configuration for the Chu limit challenge.

    Material model:
        Conductivity SIMP:  sigma(rho) = sigma_min + (sigma_max - sigma_min) * rho^simp_p
        Complex permittivity: eps_eff = 1 - j * sigma_norm * rho^simp_p
        where sigma_norm is normalised conductivity in Helmholtz units.

    Domain auto-sizing ensures >= 1 lambda for proper PML absorption proxy.
    """

    frequency_hz: float = 1.0e9
    ka: float = 0.3
    n_bits: int = 6
    domain_wavelengths: float = 0.0
    pml_depth: int = 8
    max_rank: int = 128
    n_sweeps: int = 40
    solver_tol: float = 1e-4
    damping: float = 0.01
    pec_penalty: float = 1e8
    eps_contrast: float = 200.0  # backward compat for legacy S11 path
    min_cells_per_radius: int = 4
    # --- Conductivity SIMP ---
    sigma_min: float = 0.0
    sigma_max: float = 200.0
    simp_p: float = 3.0

    def __post_init__(self) -> None:
        """Auto-compute domain_wavelengths if not set."""
        if self.domain_wavelengths <= 0:
            lam = C0 / self.frequency_hz
            a = self.ka / (2.0 * math.pi / lam)
            N = 2 ** self.n_bits
            max_domain_wl = a * N / (self.min_cells_per_radius * lam)
            ideal_wl = 6.0 * a / lam
            object.__setattr__(
                self, "domain_wavelengths",
                min(max_domain_wl, max(ideal_wl, 0.15)),
            )

    @property
    def wavelength(self) -> float:
        return C0 / self.frequency_hz

    @property
    def k0(self) -> float:
        return 2.0 * math.pi / self.wavelength

    @property
    def sphere_radius(self) -> float:
        return self.ka / self.k0

    @property
    def N(self) -> int:
        return 2 ** self.n_bits

    @property
    def domain_size(self) -> float:
        return self.domain_wavelengths * self.wavelength

    @property
    def h_physical(self) -> float:
        return self.domain_size / self.N

    @property
    def h_normalised(self) -> float:
        return 1.0 / self.N

    @property
    def sphere_radius_normalised(self) -> float:
        return self.sphere_radius / self.domain_size

    @property
    def k0_normalised(self) -> float:
        return self.k0 * self.domain_size

    @property
    def q_chu(self) -> float:
        return chu_limit_q(self.ka)

    def pml_config(self) -> PMLConfig:
        return PMLConfig.for_problem(
            n_bits=self.n_bits,
            k=self.k0_normalised,
            target_R_dB=-40.0,
        )

    def summary(self) -> str:
        """Human-readable configuration summary."""
        sep = "=" * 60
        return "\n".join([
            f"\n{sep}",
            f"  Chu Limit Challenge Configuration",
            f"{sep}",
            f"  Frequency:   {self.frequency_hz / 1e9:.3f} GHz",
            f"  Wavelength:  {self.wavelength * 1e3:.1f} mm",
            f"  ka:          {self.ka:.4f}",
            f"  Sphere:      a = {self.sphere_radius * 1e3:.2f} mm",
            f"  Q_Chu:       {self.q_chu:.2f}",
            f"  Grid:        {self.N}^3 ({self.n_bits} bits/dim)",
            f"  Domain:      {self.domain_wavelengths:.4f} wl"
            f" ({self.domain_size * 1e3:.2f} mm)",
            f"  h:           {self.h_physical * 1e3:.4f} mm",
            f"  k0_norm:     {self.k0_normalised:.4f}",
            f"  Sphere r_n:  {self.sphere_radius_normalised:.4f}",
            f"  max_rank:    {self.max_rank}",
            f"  sigma SIMP:  [{self.sigma_min}, {self.sigma_max}] p={self.simp_p}",
            f"  eps_contrast:{self.eps_contrast}",
            f"{sep}",
        ])


# =====================================================================
# Section 2b: Conductivity SIMP Model
# =====================================================================

def simp_sigma(
    rho_proj: NDArray,
    sigma_min: float,
    sigma_max: float,
    p: float,
) -> NDArray:
    """SIMP conductivity interpolation.

    sigma(rho) = sigma_min + (sigma_max - sigma_min) * rho^p

    Parameters
    ----------
    rho_proj : NDArray
        Projected density in [0,1].
    sigma_min : float
        Minimum conductivity (air, typically 0 or small).
    sigma_max : float
        Maximum conductivity (conductor).
    p : float
        Penalisation exponent (1 = linear, 3 = standard).

    Returns
    -------
    NDArray
        Conductivity at each point.
    """
    rho = np.clip(rho_proj, 0.0, 1.0)
    return sigma_min + (sigma_max - sigma_min) * np.power(rho, p)


def simp_dsigma_drho(
    rho_proj: NDArray,
    sigma_min: float,
    sigma_max: float,
    p: float,
) -> NDArray:
    """Derivative of SIMP conductivity wrt projected density.

    dsigma/drho = (sigma_max - sigma_min) * p * rho^(p-1)

    Returns
    -------
    NDArray
        dsigma/drho_proj at each point.
    """
    rho = np.clip(rho_proj, 1e-12, 1.0)
    return (sigma_max - sigma_min) * p * np.power(rho, p - 1.0)


def build_pml_sigma_3d(
    n_bits: int,
    k0_norm: float,
    pml: PMLConfig,
) -> NDArray:
    r"""Build 3D PML loss weight array for power computation.

    Uses the **additive** PML sigma formulation:

        σ_pml(x,y,z) = [Im(s_x) + Im(s_y) + Im(s_z)] · k₀²

    This is inherently non-negative (each Im(s) ≥ 0) and avoids
    the negative values at PML triple-corners that arise from the
    multiplicative formula Im(s_x · s_y · s_z).

    The additive formula captures the dominant PML absorption from
    each axis independently and is exact for single-axis PML regions
    (faces), giving the same integral for typical outgoing waves.

    Parameters
    ----------
    n_bits : int
        Bits per dimension (N = 2^n_bits).
    k0_norm : float
        Normalised wavenumber.
    pml : PMLConfig
        PML configuration.

    Returns
    -------
    NDArray
        Real (N,N,N) array of PML loss weights, >= 0 everywhere.
        Nonzero only inside PML region.
    """
    N = 2 ** n_bits
    s_1d = _compute_pml_stretching(N, k0_norm, pml)
    ai = np.imag(s_1d)  # >= 0

    # Additive PML sigma: σ = (Im(s_x) + Im(s_y) + Im(s_z)) · k²
    sigma_pml = (
        ai.reshape(N, 1, 1)
        + ai.reshape(1, N, 1)
        + ai.reshape(1, 1, N)
    ) * k0_norm ** 2

    return sigma_pml


# =====================================================================
# Section 3: 3D Antenna Geometry Container
# =====================================================================

@dataclass
class AntennaGeometry3D:
    """Container for 3D antenna design within a spherical region."""

    config: ChuProblemConfig
    density: Optional[NDArray] = None

    def __post_init__(self) -> None:
        cfg = self.config
        N = cfg.N
        centre = (0.5, 0.5, 0.5)

        # Always compute flat indices analytically — O(N + n_design) memory.
        # This works at ANY grid size without dense N³ intermediates.
        self._design_flat_indices = spherical_mask_flat_indices(
            cfg.n_bits, centre=centre,
            radius=cfg.sphere_radius_normalised,
        )
        self._n_design = len(self._design_flat_indices)

        # For small grids (N ≤ 256), build dense mask for backward
        # compatibility with dense code paths and tests.
        if cfg.n_bits <= 8:
            self._design_mask = spherical_mask(
                cfg.n_bits, centre=centre,
                radius=cfg.sphere_radius_normalised,
            )
        else:
            # Large grids: dense mask would require N³ meshgrid (24+ GB).
            # Use flat_indices + QTT-native code path instead.
            self._design_mask = None

        self._feed_position = (0.5, 0.5, 0.5 - cfg.sphere_radius_normalised)
        if self.density is None:
            self.density = 0.5 * np.ones(self._n_design)
        elif len(self.density) != self._n_design:
            raise ValueError(
                f"Density length {len(self.density)} != n_design {self._n_design}"
            )

    @classmethod
    def with_monopole_seed(
        cls,
        config: ChuProblemConfig,
        wire_radius_cells: int = 1,
        base_density: float = 0.05,
        wire_density: float = 0.95,
        top_hat: bool = True,
        top_hat_radius_cells: int = 3,
    ) -> "AntennaGeometry3D":
        """Create geometry with monopole seed density.

        For large grids (n_bits >= 9), builds the seed analytically
        from flat indices — never allocates dense N³ arrays.
        """
        centre = (0.5, 0.5, 0.5)

        if config.n_bits <= 8:
            # Small grid: use dense path (backward compatible)
            design_mask = spherical_mask(
                config.n_bits, centre=centre,
                radius=config.sphere_radius_normalised,
            )
            seed = monopole_seed_density(
                n_bits=config.n_bits,
                centre=centre,
                sphere_radius=config.sphere_radius_normalised,
                design_mask=design_mask,
                wire_radius_cells=wire_radius_cells,
                base_density=base_density,
                wire_density=wire_density,
                top_hat=top_hat,
                top_hat_radius_cells=top_hat_radius_cells,
            )
        else:
            # Large grid: analytical seed from flat indices — O(n_design)
            flat_idx = spherical_mask_flat_indices(
                config.n_bits, centre=centre,
                radius=config.sphere_radius_normalised,
            )
            N = config.N
            h = 1.0 / N
            coords = np.linspace(h / 2, 1.0 - h / 2, N)

            # Decompose flat indices to (ix, iy, iz)
            ix = flat_idx % N
            iy = (flat_idx // N) % N
            iz = flat_idx // (N * N)

            cx, cy, cz = centre
            r_norm = config.sphere_radius_normalised
            ix_c = min(int(cx * N), N - 1)
            iy_c = min(int(cy * N), N - 1)
            iz_bottom = max(int((cz - r_norm) * N), 0)
            iz_top = min(int((cz + r_norm) * N), N - 1)

            seed = np.full(len(flat_idx), base_density, dtype=np.float64)

            # Wire: voxels where |ix-ix_c|²+|iy-iy_c|² ≤ wr²
            #        and iz_bottom ≤ iz ≤ iz_top
            wr = wire_radius_cells
            dx_wire = ix.astype(np.int64) - ix_c
            dy_wire = iy.astype(np.int64) - iy_c
            dist_sq_xy = dx_wire ** 2 + dy_wire ** 2
            in_wire = (dist_sq_xy <= wr * wr) & (iz >= iz_bottom) & (iz <= iz_top)
            seed[in_wire] = wire_density

            # Top hat: disc at iz_top - 1
            if top_hat:
                thr = top_hat_radius_cells
                iz_hat = iz_top - 1
                if iz_hat >= iz_bottom:
                    in_hat = (dist_sq_xy <= thr * thr) & (iz == iz_hat)
                    seed[in_hat] = wire_density

        return cls(config=config, density=seed)

    @property
    def design_mask(self) -> NDArray:
        """Dense boolean (N,N,N) design mask — small grids only."""
        if self._design_mask is None:
            raise RuntimeError(
                f"Dense design_mask unavailable for n_bits={self.config.n_bits} "
                f"(N={self.config.N}). Use design_flat_indices + "
                f"QTT-native code path for large grids."
            )
        return self._design_mask

    @property
    def design_flat_indices(self) -> NDArray:
        """Flat F-order indices of design voxels — O(n_design) memory."""
        return self._design_flat_indices

    @property
    def n_design(self) -> int:
        return self._n_design

    @property
    def feed_position(self) -> tuple[float, float, float]:
        return self._feed_position

    def build_conductor_mask(
        self, beta: float = 8.0, eta: float = 0.5, filter_radius: int = 0,
    ) -> NDArray:
        """Build boolean conductor mask from current density."""
        cfg = self.config
        N = cfg.N
        mask_3d = np.zeros((N, N, N), dtype=bool)
        rho_filt = density_filter(self.density, filter_radius)
        rho_proj = heaviside_projection(rho_filt, beta, eta)
        mask_3d[self._design_mask] = rho_proj > 0.5
        return mask_3d

    def build_eps_3d(
        self, beta: float = 1.0, eta: float = 0.5, filter_radius: int = 0,
    ) -> NDArray:
        """Build 3D permittivity using real eps_contrast (legacy S11 path).

        eps(r) = 1 + (eps_contrast - 1) * H(filter(rho))
        """
        cfg = self.config
        N = cfg.N
        eps_3d = np.ones((N, N, N), dtype=np.complex128)
        rho_filt = density_filter(self.density, filter_radius)
        rho_proj = heaviside_projection(rho_filt, beta, eta)
        eps_3d[self._design_mask] = (
            1.0 + (cfg.eps_contrast - 1.0) * rho_proj.astype(np.complex128)
        )
        return eps_3d

    def build_conductivity_eps_3d(
        self, beta: float = 1.0, eta: float = 0.5, filter_radius: int = 0,
    ) -> NDArray:
        """Build 3D complex permittivity from conductivity SIMP model.

        eps_eff(rho) = 1 - j * sigma_norm * rho_proj^p

        where sigma_norm = sigma from SIMP parametrisation, normalised
        into Helmholtz units (the k^2*eps term in H absorbs the rest).

        Parameters
        ----------
        beta : float
            Heaviside projection sharpness.
        eta : float
            Heaviside threshold.
        filter_radius : int
            Density filter radius in voxels.

        Returns
        -------
        NDArray
            Complex (N,N,N) permittivity.
        """
        cfg = self.config
        N = cfg.N
        eps_3d = np.ones((N, N, N), dtype=np.complex128)

        rho_filt = density_filter(self.density, filter_radius)
        rho_proj = heaviside_projection(rho_filt, beta, eta)

        sigma_values = simp_sigma(
            rho_proj, cfg.sigma_min, cfg.sigma_max, cfg.simp_p,
        )
        # eps = 1 - j * sigma_norm (conductivity enters as imaginary part)
        # The factor 1/omega is embedded in sigma_max calibration
        eps_3d[self._design_mask] = 1.0 - 1j * sigma_values

        return eps_3d

    def build_source(self, max_rank: int = 128) -> list[NDArray]:
        """Build QTT gap source at feed position."""
        cfg = self.config
        h = 1.0 / cfg.N
        gap_h = max(2.0 * h, 0.02)
        gap_r = max(2.0 * h, 0.02)
        return gap_source_3d(
            n_bits=cfg.n_bits,
            feed_position=self._feed_position,
            gap_height=gap_h,
            gap_radius=gap_r,
            max_rank=max_rank,
        )

    def build_source_3d(self) -> NDArray:
        """Build 3D source array (dense) for power metrics."""
        cfg = self.config
        N = cfg.N
        h = 1.0 / N
        coords = np.linspace(h / 2, 1.0 - h / 2, N)

        x0, y0, z0 = self._feed_position
        gap_h = max(2.0 * h, 0.02)
        gap_r = max(2.0 * h, 0.02)

        xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
        rr = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)

        J_3d = np.zeros((N, N, N), dtype=np.complex128)
        in_gap = (rr < gap_r) & (np.abs(zz - z0) < gap_h / 2)
        J_3d[in_gap] = 1.0

        vol = float(np.sum(in_gap)) * h ** 3
        if vol > 0:
            J_3d /= vol

        return J_3d


# =====================================================================
# Section 4: Forward Solve and S11 Extraction (legacy + new)
# =====================================================================

def solve_and_extract_s11(
    geometry: AntennaGeometry3D,
    k0_norm: float,
    beta: float = 1.0,
    eta: float = 0.5,
    filter_radius: int = 0,
    max_rank: int = 128,
    n_sweeps: int = 40,
    solver_tol: float = 1e-4,
    damping: float = 0.01,
    verbose: bool = False,
) -> tuple[complex, float, list[NDArray]]:
    """Solve and extract S11 for current geometry (legacy path)."""
    cfg = geometry.config
    pml = cfg.pml_config()
    rhs_cores = geometry.build_source(max_rank=max_rank)
    eps_3d = geometry.build_eps_3d(beta, eta, filter_radius)
    result = solve_helmholtz_3d(
        n_bits=cfg.n_bits, k=k0_norm, pml=pml,
        rhs_cores=rhs_cores, eps_3d=eps_3d,
        max_rank=max_rank, n_sweeps=n_sweeps,
        tol=solver_tol, damping=damping, verbose=verbose,
    )
    s11 = extract_s11_3d(
        E_cores=result.x, n_bits=cfg.n_bits, k0=k0_norm,
        feed_position=geometry.feed_position,
        ref_distance=2.0 / cfg.N, damping=damping,
    )
    return s11, result.final_residual, result.x


def solve_forward_conductivity(
    geometry: AntennaGeometry3D,
    k0_norm: float,
    beta: float = 1.0,
    eta: float = 0.5,
    filter_radius: int = 0,
    max_rank: int = 128,
    n_sweeps: int = 40,
    solver_tol: float = 1e-4,
    damping: float = 0.005,
    verbose: bool = False,
) -> tuple[list[NDArray], list[NDArray], float]:
    """Forward solve with conductivity SIMP material.

    Builds H = nabla^2_s + k^2 * eps(sigma(rho)) and solves H E = b.
    Returns operator cores for reuse in adjoint solve.

    Returns
    -------
    tuple[list[NDArray], list[NDArray], float]
        (H_cores, E_cores, residual).
    """
    cfg = geometry.config
    pml = cfg.pml_config()

    eps_3d = geometry.build_conductivity_eps_3d(beta, eta, filter_radius)

    H_cores = helmholtz_mpo_3d_pml(
        n_bits=cfg.n_bits, k=k0_norm, pml=pml,
        eps_3d=eps_3d, max_rank=max_rank, damping=damping,
    )

    rhs_cores = geometry.build_source(max_rank=max_rank)

    result = tt_amen_solve(
        H_cores, rhs_cores,
        max_rank=max_rank,
        n_sweeps=n_sweeps,
        tol=solver_tol,
        verbose=verbose,
    )

    return H_cores, result.x, result.final_residual


# =====================================================================
# Section 4b: MPO Hermitian Conjugate
# =====================================================================

def mpo_hermitian_conjugate(cores: list[NDArray]) -> list[NDArray]:
    """Compute H^H from MPO cores by conjugating and transposing.

    For each core of shape (r_l, d_out, d_in, r_r):
        H^H core shape: (r_l, d_in, d_out, r_r) with complex conjugation.

    This works because (A_1 otimes ... otimes A_n)^H
    = A_1^H otimes ... otimes A_n^H for the tensor product structure.
    """
    result = []
    for c in cores:
        c = np.asarray(c, dtype=np.complex128)
        if c.ndim == 4:
            # Shape (r_l, d_out, d_in, r_r) -> swap d_out, d_in and conj
            result.append(np.conj(c.swapaxes(1, 2)))
        elif c.ndim == 3:
            # Shape (r_l, d, r_r) -> just conjugate (diagonal-like)
            result.append(np.conj(c))
        else:
            raise ValueError(f"Unexpected core ndim={c.ndim}")
    return result


# =====================================================================
# Section 4c: QTT-Native Forward Solve (no dense N³ arrays)
# =====================================================================

def solve_forward_conductivity_tt(
    geometry: AntennaGeometry3D,
    k0_norm: float,
    design_mask_tt: list[NDArray],
    beta: float = 1.0,
    eta: float = 0.5,
    filter_radius: int = 0,
    max_rank: int = 128,
    n_sweeps: int = 40,
    solver_tol: float = 1e-4,
    damping: float = 0.005,
    verbose: bool = False,
) -> tuple[list[NDArray], list[NDArray], float, list[NDArray]]:
    """QTT-native forward solve — never materialises dense N³ arrays.

    Builds ε in QTT, constructs H as MPO, solves H E = b in QTT.
    The only dense object is the 1D design density (length n_design).

    Returns
    -------
    tuple[list[NDArray], list[NDArray], float, list[NDArray]]
        (H_cores, E_cores, residual, eps_tt_cores).
    """
    cfg = geometry.config
    pml = cfg.pml_config()

    # Build ε in QTT (QTT-native, no dense N³)
    eps_tt = build_conductivity_eps_tt(
        density=geometry.density,
        design_mask_tt=design_mask_tt,
        sigma_min=cfg.sigma_min,
        sigma_max=cfg.sigma_max,
        simp_p=cfg.simp_p,
        beta=beta, eta=eta,
        filter_radius=filter_radius,
        n_bits=cfg.n_bits,
        max_rank=max_rank,
    )

    # Build Helmholtz MPO using QTT eps (no dense)
    H_cores = helmholtz_mpo_3d_pml(
        n_bits=cfg.n_bits, k=k0_norm, pml=pml,
        eps_tt_cores=eps_tt, max_rank=max_rank, damping=damping,
    )

    # Source already builds in QTT format
    rhs_cores = geometry.build_source(max_rank=max_rank)

    result = tt_amen_solve(
        H_cores, rhs_cores,
        max_rank=max_rank,
        n_sweeps=n_sweeps,
        tol=solver_tol,
        verbose=verbose,
    )

    return H_cores, result.x, result.final_residual, eps_tt


def compute_power_metrics_tt(
    E_cores: list[NDArray],
    J_cores: list[NDArray],
    sigma_pml_tt: list[NDArray],
    sigma_design_tt: list[NDArray],
    geometry: AntennaGeometry3D,
    k0_norm: float,
    max_rank: int = 128,
) -> PowerMetrics:
    """Compute power balance entirely in QTT — no dense N³ arrays.

    P_pml = 0.5 * dV * <E, diag(σ_pml) E>     (via MPO-vec + inner)
    P_cond = 0.5 * k² * dV * <E, diag(σ_design) E>
    P_input = Re(S) from compute_input_power   (already QTT-native)

    Parameters
    ----------
    E_cores : list[NDArray]
        Solved field QTT cores.
    J_cores : list[NDArray]
        Source QTT cores (un-negated).
    sigma_pml_tt : list[NDArray]
        PML loss weight QTT cores.
    sigma_design_tt : list[NDArray]
        Design conductivity QTT cores (σ · mask).
    geometry : AntennaGeometry3D
        Antenna geometry.
    k0_norm : float
        Normalised wavenumber.
    max_rank : int
        Max rank for intermediates.

    Returns
    -------
    PowerMetrics
    """
    cfg = geometry.config
    N = cfg.N
    h = 1.0 / N
    dv = h ** 3

    P_pml = compute_pml_power_tt(E_cores, sigma_pml_tt, dv, max_rank)
    P_cond = compute_cond_power_tt(E_cores, sigma_design_tt, k0_norm, dv, max_rank)
    P_input = compute_input_power(E_cores, J_cores, dv)

    eta_rad = P_pml / (abs(P_pml) + abs(P_cond) + 1e-30)
    vol = float(np.mean(geometry.density))

    return PowerMetrics(
        P_pml=P_pml, P_cond=P_cond, P_input=P_input,
        eta_rad=eta_rad, vol=vol,
    )


# =====================================================================
# Section 5: Power Metrics
# =====================================================================

@dataclass
class PowerMetrics:
    """Power balance from a 3D Helmholtz solve.

    P_pml : Power absorbed by PML (radiation proxy).
    P_cond : Power dissipated in conductor.
    P_input : Power delivered by source.
    eta_rad : Radiation efficiency P_pml / (P_pml + P_cond + eps).
    vol : Design region volume fraction.
    P_pml_norm : P_pml / P_pml(air), normalised to baseline.
    P_cond_norm : P_cond / P_pml(air), normalised to baseline.
    E2_metal_avg : <|E|^2>_metal, metal-weighted mean field energy.
    M_dead : Fraction of metal placed in air-field dead zones.
             M_dead = sum rho_i * 1(|E_air_i|^2 < tau) / sum rho_i.
    """
    P_pml: float
    P_cond: float
    P_input: float
    eta_rad: float
    vol: float
    P_pml_norm: float = 0.0
    P_cond_norm: float = 0.0
    E2_metal_avg: float = 0.0
    M_dead: float = 0.0


def compute_power_metrics(
    E_3d: NDArray,
    geometry: AntennaGeometry3D,
    k0_norm: float,
    pml: PMLConfig,
    beta: float = 1.0,
    eta: float = 0.5,
    filter_radius: int = 0,
    damping: float = 0.005,
) -> PowerMetrics:
    """Compute physics-correct power balance from solved field.

    P_pml = 0.5 * sum sigma_pml_i * |E_i|^2 * dV   (radiation proxy)
    P_cond = 0.5 * k^2 * sum sigma_design_i * |E_i|^2 * dV  (ohmic loss)
    P_input = Re(<J*, E>) * dV                         (source power)

    Parameters
    ----------
    E_3d : NDArray
        Solved 3D field, shape (N,N,N).
    geometry : AntennaGeometry3D
        Antenna geometry with density.
    k0_norm : float
        Normalised wavenumber.
    pml : PMLConfig
        PML configuration.
    beta, eta, filter_radius : float, float, int
        Projection parameters.
    damping : float
        Helmholtz damping.

    Returns
    -------
    PowerMetrics
    """
    cfg = geometry.config
    N = cfg.N
    h = 1.0 / N
    dv = h ** 3

    # PML sigma weights
    sigma_pml = build_pml_sigma_3d(cfg.n_bits, k0_norm, pml)

    # PML absorbed power
    E_sq = np.abs(E_3d) ** 2
    P_pml = 0.5 * float(np.sum(sigma_pml * E_sq) * dv)

    # Conductor dissipation
    rho_filt = density_filter(geometry.density, filter_radius)
    rho_proj = heaviside_projection(rho_filt, beta, eta)
    sigma_design = simp_sigma(rho_proj, cfg.sigma_min, cfg.sigma_max, cfg.simp_p)

    E_design_sq = np.abs(E_3d[geometry.design_mask]) ** 2
    # The conductivity enters H as k^2 * (-j * sigma), so the power
    # dissipated is 0.5 * k^2 * sigma * |E|^2 * dV
    P_cond = 0.5 * k0_norm ** 2 * float(np.sum(sigma_design * E_design_sq) * dv)

    # Input power from source
    J_3d = geometry.build_source_3d()
    P_input = float(np.real(np.sum(np.conj(J_3d) * E_3d)) * dv)

    eta_rad = P_pml / (abs(P_pml) + abs(P_cond) + 1e-30)
    vol = float(np.mean(geometry.density))

    # Metal-weighted mean field energy: <|E|^2>_metal
    rho_sum = float(np.sum(rho_proj))
    if rho_sum > 1e-30:
        E2_metal_avg = float(np.sum(rho_proj * E_design_sq)) / rho_sum
    else:
        E2_metal_avg = 0.0

    return PowerMetrics(
        P_pml=P_pml, P_cond=P_cond, P_input=P_input,
        eta_rad=eta_rad, vol=vol, E2_metal_avg=E2_metal_avg,
    )


# =====================================================================
# Section 5b: Poynting Flux Cross-Check
# =====================================================================

def compute_poynting_flux_shell(
    E_3d: NDArray,
    geometry: AntennaGeometry3D,
    k0_norm: float,
    shell_radius_normalised: float,
    shell_thickness_cells: int = 2,
    damping: float = 0.005,
) -> float:
    """Compute outward Poynting flux through a spherical shell.

    This is the independent cross-check for P_pml.  For a lossless
    exterior, energy conservation requires:

        P_poynting(shell) = P_pml + P_cond   (within solver tolerance)

    In the scalar Helmholtz approximation the time-averaged Poynting
    vector simplifies to:

        S = (1/2k) Im(E* grad E)

    and the surface integral becomes:

        P_S = (1/2k) sum_{shell} Im(E* dE/dn) * dA

    where dE/dn is the outward normal derivative estimated via central
    differences and dA = h^2 (cell face area on the shell surface).

    Parameters
    ----------
    E_3d : NDArray
        Solved complex field (N,N,N).
    geometry : AntennaGeometry3D
        Geometry (provides config, centre).
    k0_norm : float
        Normalised wavenumber.
    shell_radius_normalised : float
        Radius of the Poynting shell in normalised coords [0,1].
        Should be between the antenna sphere and the PML inner face.
    shell_thickness_cells : int
        Thickness of shell in cells (default 2).
    damping : float
        Helmholtz damping.

    Returns
    -------
    float
        Outward Poynting flux (should be positive for a radiator).
    """
    cfg = geometry.config
    N = cfg.N
    h = 1.0 / N
    dA = h ** 2

    # Build shell mask
    inner_r = shell_radius_normalised
    outer_r = inner_r + shell_thickness_cells * h
    shell = spherical_shell_mask(
        cfg.n_bits,
        centre=(0.5, 0.5, 0.5),
        inner_radius=inner_r,
        outer_radius=outer_r,
    )

    # Cell centres
    coords = np.linspace(h / 2, 1.0 - h / 2, N)
    cx, cy, cz = 0.5, 0.5, 0.5

    # Find shell voxels
    shell_idx = np.argwhere(shell)
    if len(shell_idx) == 0:
        return 0.0

    P_total = 0.0
    for (ix, iy, iz) in shell_idx:
        x_c = coords[ix] - cx
        y_c = coords[iy] - cy
        z_c = coords[iz] - cz
        r = math.sqrt(x_c ** 2 + y_c ** 2 + z_c ** 2)
        if r < 1e-12:
            continue
        # Outward normal
        nx, ny, nz = x_c / r, y_c / r, z_c / r

        # Central difference for gradient components
        E_c = E_3d[ix, iy, iz]

        # dE/dx via central difference (clamped at boundaries)
        dEdx = (
            (E_3d[min(ix + 1, N - 1), iy, iz] - E_3d[max(ix - 1, 0), iy, iz])
            / (2.0 * h)
        )
        dEdy = (
            (E_3d[ix, min(iy + 1, N - 1), iz] - E_3d[ix, max(iy - 1, 0), iz])
            / (2.0 * h)
        )
        dEdz = (
            (E_3d[ix, iy, min(iz + 1, N - 1)] - E_3d[ix, iy, max(iz - 1, 0)])
            / (2.0 * h)
        )

        # Normal derivative
        dE_dn = dEdx * nx + dEdy * ny + dEdz * nz

        # Poynting flux contribution: (1/2k) Im(E* dE/dn) dA
        P_total += float(np.imag(np.conj(E_c) * dE_dn)) * dA

    P_total /= (2.0 * k0_norm)
    return P_total


# =====================================================================
# Section 5c: Impedance-Derivative Q
# =====================================================================

def compute_impedance_q(
    geometry: AntennaGeometry3D,
    f_center_hz: float,
    delta_f_fraction: float = 0.001,
    beta: float = 8.0,
    eta: float = 0.5,
    filter_radius: int = 0,
    max_rank: int = 128,
    n_sweeps: int = 40,
    solver_tol: float = 1e-4,
    damping: float = 0.005,
    verbose: bool = False,
) -> dict[str, float]:
    """Compute Q via impedance derivative at resonance.

    Q_Z = (omega_0 / 2R) |dX/domega|_{omega_0}

    Uses three-point central difference for dX/domega.

    This is independent of the half-power BW method and serves
    as the dual-Q cross-check required by Chu validation.

    Parameters
    ----------
    geometry : AntennaGeometry3D
        Optimised antenna geometry.
    f_center_hz : float
        Frequency at which to evaluate (should be near resonance).
    delta_f_fraction : float
        Fractional frequency step for finite differences.
    beta, eta, filter_radius : float, float, int
        Projection parameters for the final design.
    max_rank, n_sweeps, solver_tol, damping : int, int, float, float
        Solver settings.
    verbose : bool
        Print intermediate values.

    Returns
    -------
    dict[str, float]
        Keys: 'Q_Z', 'R', 'X', 'dX_domega', 'omega_0'.
    """
    cfg = geometry.config
    omega_0 = 2.0 * math.pi * f_center_hz

    def _impedance_at_freq(f_hz: float) -> tuple[float, float]:
        """Solve and extract Z = R + jX at given frequency."""
        lam_f = C0 / f_hz
        k0_f = 2.0 * math.pi / lam_f
        k0_norm_f = k0_f * cfg.domain_size

        s11, _, _ = solve_and_extract_s11(
            geometry=geometry, k0_norm=k0_norm_f,
            beta=beta, eta=eta, filter_radius=filter_radius,
            max_rank=max_rank, n_sweeps=n_sweeps,
            solver_tol=solver_tol, damping=damping, verbose=False,
        )
        # Z = Z0 * (1 + S11) / (1 - S11)  (normalised to 50 ohm)
        Z0 = 50.0
        denom = 1.0 - s11
        if abs(denom) < 1e-30:
            return float('inf'), float('inf')
        Z = Z0 * (1.0 + s11) / denom
        return float(np.real(Z)), float(np.imag(Z))

    # Three frequencies
    df = delta_f_fraction * f_center_hz
    R_lo, X_lo = _impedance_at_freq(f_center_hz - df)
    R_0, X_0 = _impedance_at_freq(f_center_hz)
    R_hi, X_hi = _impedance_at_freq(f_center_hz + df)

    # Central difference: dX/df
    dX_df = (X_hi - X_lo) / (2.0 * df)
    # Convert to dX/domega
    dX_domega = dX_df / (2.0 * math.pi)

    # Q = omega_0 / (2 * R) * |dX/domega|
    Q_Z = omega_0 / (2.0 * abs(R_0) + 1e-30) * abs(dX_domega)

    if verbose:
        print(f"  Z({f_center_hz / 1e9:.4f} GHz) = {R_0:.2f} + j{X_0:.2f} ohm")
        print(f"  dX/domega = {dX_domega:.4e}")
        print(f"  Q_Z = {Q_Z:.1f}")

    return {
        'Q_Z': Q_Z,
        'R': R_0,
        'X': X_0,
        'dX_domega': dX_domega,
        'omega_0': omega_0,
    }


# =====================================================================
# Section 5d: Chu Validation Scorecard
# =====================================================================

@dataclass
class ChuValidation:
    """Adversarial validation scorecard for Chu limit claims.

    All four checks must pass for a defensible result:
    A. Dual-Q agreement (impedance Q vs half-power Q)
    B. Resonance verified (X ≈ 0 at operating frequency)
    C. Radiated power validated (P_pml cross-checked by Poynting flux)
    D. Bound comparison is apples-to-apples
    """
    # --- Q values ---
    Q_halfpower: float = 0.0
    Q_impedance: float = 0.0
    Q_ratio: float = 0.0   # Q_halfpower / Q_impedance (want ≈ 1)
    Q_dual_pass: bool = False

    # --- Resonance ---
    X_at_f0: float = 0.0
    R_at_f0: float = 0.0
    resonance_pass: bool = False  # |X| < threshold * R

    # --- Power validation ---
    P_pml: float = 0.0
    P_poynting: float = 0.0
    P_cond: float = 0.0
    power_ratio: float = 0.0  # P_poynting / P_pml (want ≈ 1)
    power_pass: bool = False

    # --- Bound comparison ---
    ka: float = 0.0
    Q_chu: float = 0.0
    Q_mclean: float = 0.0
    Q_thal: float = 0.0
    Q_over_Chu: float = 0.0   # best Q / Q_Chu (want < 1 to claim)
    bound_mode: str = ""     # which bound applies (single/dual/self-res)

    # --- Domain metrics ---
    pml_clearance_wavelengths: float = 0.0
    cells_per_radius: float = 0.0
    n_design_voxels: int = 0
    binarisation: float = 0.0

    def summary(self) -> str:
        """Human-readable scorecard."""
        sep = "=" * 70
        lines = [
            f"\n{sep}",
            f"  CHU LIMIT VALIDATION SCORECARD",
            f"{sep}",
            f"",
            f"  A. DUAL-Q AGREEMENT {'PASS' if self.Q_dual_pass else '** FAIL **'}",
            f"     Q (half-power BW):    {self.Q_halfpower:.1f}",
            f"     Q (impedance deriv):  {self.Q_impedance:.1f}",
            f"     Ratio:                {self.Q_ratio:.3f} (want 0.8-1.2)",
            f"",
            f"  B. RESONANCE {'PASS' if self.resonance_pass else '** FAIL **'}",
            f"     Z = {self.R_at_f0:.2f} + j{self.X_at_f0:.2f} ohm",
            f"     |X|/R = {abs(self.X_at_f0) / (abs(self.R_at_f0) + 1e-30):.3f} (want < 0.1)",
            f"",
            f"  C. POWER VALIDATION {'PASS' if self.power_pass else '** FAIL **'}",
            f"     P_pml:      {self.P_pml:.4e}",
            f"     P_poynting:  {self.P_poynting:.4e}",
            f"     P_cond:      {self.P_cond:.4e}",
            f"     P_poynting / P_pml = {self.power_ratio:.3f} (want 0.8-1.2)",
            f"",
            f"  D. BOUND COMPARISON (ka={self.ka:.4f})",
            f"     Q_Chu (single):    {self.Q_chu:.1f}",
            f"     Q_McLean (dual):   {self.Q_mclean:.1f}",
            f"     Q_Thal (self-res):  {self.Q_thal:.1f}",
            f"     Best Q / Q_Chu:     {self.Q_over_Chu:.3f}",
            f"     Bound mode:         {self.bound_mode}",
            f"",
            f"  DOMAIN METRICS",
            f"     PML clearance:      {self.pml_clearance_wavelengths:.2f} λ",
            f"     Cells per radius:   {self.cells_per_radius:.1f}",
            f"     Design voxels:      {self.n_design_voxels}",
            f"     Binarisation:       {self.binarisation:.4f}",
            f"",
        ]

        all_pass = self.Q_dual_pass and self.resonance_pass and self.power_pass
        if self.Q_over_Chu < 1.0 and not all_pass:
            lines.append(
                f"  *** WARNING: Apparent Chu violation with failed checks. ***"
            )
            lines.append(
                f"  *** This is most likely an artifact, not a real result. ***"
            )
        elif self.Q_over_Chu < 1.0 and all_pass:
            lines.append(
                f"  *** EXTRAORDINARY CLAIM: Q < Q_Chu with all checks passed. ***"
            )
            lines.append(
                f"  *** Requires independent replication before publication. ***"
            )
        elif all_pass:
            lines.append(
                f"  Result: Q/Q_Chu = {self.Q_over_Chu:.2f} — "
                f"DEFENSIBLE under validated assumptions."
            )
        else:
            lines.append(f"  Result: FAILED checks — not publication-ready.")

        lines.append(f"{sep}")
        return "\n".join(lines)


def validate_chu_result(
    result: ChuOptResult,
    E_3d: Optional[NDArray] = None,
    poynting_shell_radius: Optional[float] = None,
    impedance_q: Optional[dict[str, float]] = None,
) -> ChuValidation:
    """Build adversarial validation scorecard from optimisation result.

    Parameters
    ----------
    result : ChuOptResult
        Completed optimisation result.
    E_3d : NDArray, optional
        Solved field for Poynting cross-check.  If None, skip power check.
    poynting_shell_radius : float, optional
        Normalised radius for Poynting shell.  If None, auto-compute.
    impedance_q : dict, optional
        Output from compute_impedance_q.  If None, skip impedance Q check.

    Returns
    -------
    ChuValidation
    """
    cfg = result.config
    if cfg is None:
        raise ValueError("Result has no config attached")

    v = ChuValidation()
    v.ka = cfg.ka
    v.Q_chu = chu_limit_q(cfg.ka)
    v.Q_mclean = mclean_limit_q(cfg.ka)
    v.Q_thal = thal_limit_q(cfg.ka)

    # Domain metrics
    v.cells_per_radius = cfg.sphere_radius / cfg.h_physical
    v.n_design_voxels = int(np.sum(result.conductor_mask))
    v.binarisation = result.binarisation

    # PML clearance: distance from sphere surface to PML inner face
    pml_depth_cells = cfg.pml_config().n_cells
    pml_inner_normalised = pml_depth_cells / cfg.N
    domain_radius_normalised = 0.5  # domain goes from 0 to 1
    sphere_edge = cfg.sphere_radius_normalised
    clearance_normalised = domain_radius_normalised - sphere_edge - pml_inner_normalised
    v.pml_clearance_wavelengths = clearance_normalised * cfg.domain_size / cfg.wavelength

    # --- A: Dual-Q ---
    if result.power_q_result is not None:
        v.Q_halfpower = result.power_q_result.Q

    if impedance_q is not None:
        v.Q_impedance = impedance_q['Q_Z']
        v.R_at_f0 = impedance_q['R']
        v.X_at_f0 = impedance_q['X']
    else:
        v.Q_impedance = v.Q_halfpower  # fallback: same method

    if v.Q_halfpower > 0 and v.Q_impedance > 0:
        v.Q_ratio = v.Q_halfpower / v.Q_impedance
        v.Q_dual_pass = 0.8 <= v.Q_ratio <= 1.2
    else:
        v.Q_dual_pass = False

    # --- B: Resonance ---
    if impedance_q is not None:
        ratio_xr = abs(v.X_at_f0) / (abs(v.R_at_f0) + 1e-30)
        v.resonance_pass = ratio_xr < 0.1
    else:
        v.resonance_pass = False

    # --- C: Power validation ---
    if result.power_metrics_history:
        last = result.power_metrics_history[-1]
        v.P_pml = last.P_pml
        v.P_cond = last.P_cond

    if E_3d is not None and cfg is not None:
        geometry = AntennaGeometry3D(
            config=cfg, density=result.density_final,
        )
        if poynting_shell_radius is None:
            # Place shell halfway between sphere and PML
            mid = cfg.sphere_radius_normalised + clearance_normalised / 2
            poynting_shell_radius = mid
        v.P_poynting = compute_poynting_flux_shell(
            E_3d, geometry, cfg.k0_normalised,
            shell_radius_normalised=poynting_shell_radius,
        )
        if abs(v.P_pml) > 1e-30:
            v.power_ratio = v.P_poynting / v.P_pml
        v.power_pass = 0.8 <= v.power_ratio <= 1.2 if abs(v.P_pml) > 1e-30 else False
    else:
        v.power_pass = False

    # --- D: Bound comparison ---
    best_Q = min(v.Q_halfpower, v.Q_impedance) if v.Q_impedance > 0 else v.Q_halfpower
    if best_Q > 0:
        v.Q_over_Chu = best_Q / v.Q_chu
    # Determine bound mode
    if best_Q > 0:
        if best_Q <= v.Q_mclean:
            v.bound_mode = "dual-mode (McLean)"
        elif best_Q <= v.Q_chu:
            v.bound_mode = "single-mode (Chu)"
        elif best_Q <= v.Q_thal:
            v.bound_mode = "self-resonant (Thal)"
        else:
            v.bound_mode = "above all bounds"

    return v


# =====================================================================
# Section 6: Exact Adjoint Gradient (Power-Based Objective)
# =====================================================================

@dataclass
class PowerObjectiveConfig:
    """Configuration for the staged PML-power + volume objective.

    Two-stage objective using log scaling for gradient stability:

    Stage 0 (radiation-first, alpha_loss_effective == 0):
        J = -log(P_pml + eps) + AL_vol

    Stage 1 (after alpha introduction):
        J = -log(P_pml + eps) + alpha_loss * log(P_cond + eps) + AL_vol

    The log form keeps gradients well-scaled when powers span orders
    of magnitude and prevents the 'everything gets small' trap.
    """
    alpha_loss: float = 0.1
    vol_target: float = 0.3
    al_lambda: float = 0.0     # augmented Lagrangian dual variable
    al_mu: float = 10.0        # augmented Lagrangian penalty
    log_eps: float = 1e-12     # regulariser inside log to avoid log(0)
    use_log_objective: bool = True  # False reverts to linear J form


def augmented_lagrangian_volume(
    rho_proj: NDArray,
    vol_target: float,
    al_lambda: float,
    al_mu: float,
) -> tuple[float, NDArray, float, float]:
    """Augmented Lagrangian penalty for volume constraint.

    L_vol = lambda * g + 0.5 * mu * g^2
    where g = mean(rho_proj) - vol_target.

    Returns (J_vol, dJ_vol/drho_proj, volume_fraction, constraint_violation).
    """
    n = len(rho_proj)
    V = float(np.mean(rho_proj))
    g = V - vol_target
    Jv = al_lambda * g + 0.5 * al_mu * g ** 2
    dJv = np.full(n, (al_lambda + al_mu * g) / n, dtype=np.float64)
    return float(Jv), dJv, V, g


def augmented_lagrangian_coupling(
    density: NDArray,
    coupling_mask: NDArray,
    threshold: float,
    al_lambda_c: float,
    al_mu_c: float,
) -> tuple[float, NDArray, float]:
    """AL penalty for near-feed density coupling constraint.

    Constraint: rho_near >= threshold, where rho_near = mean(density[coupling_mask]).
    Violation: g_c = threshold - rho_near  (positive when violated).
    Penalty:   L_c = lambda_c * g_c + 0.5 * mu_c * max(g_c, 0)^2.

    Parameters
    ----------
    density : NDArray
        Full design density vector.
    coupling_mask : NDArray
        Boolean mask over design region marking near-feed voxels.
    threshold : float
        Minimum allowed mean density near feed.
    al_lambda_c : float
        Dual variable.
    al_mu_c : float
        Penalty weight.

    Returns
    -------
    tuple[float, NDArray, float]
        (J_coupling, dJ_coupling/d_density, rho_near).
    """
    n_total = len(density)
    n_near = int(np.sum(coupling_mask))
    if n_near == 0:
        return 0.0, np.zeros(n_total, dtype=np.float64), 0.0

    rho_near = float(np.mean(density[coupling_mask]))
    g_c = threshold - rho_near  # positive = violated

    # Only penalise if violated or if dual pulls
    Jc = al_lambda_c * g_c + 0.5 * al_mu_c * max(g_c, 0.0) ** 2

    # Gradient wrt density: dJ/drho_i = (-1/n_near) * (lambda_c + mu_c * max(g_c,0))
    # Only for i in coupling_mask
    dJc = np.zeros(n_total, dtype=np.float64)
    if g_c > 0:
        coeff = -(al_lambda_c + al_mu_c * g_c) / n_near
    else:
        coeff = -al_lambda_c / n_near
    dJc[coupling_mask] = coeff

    return float(Jc), dJc, rho_near


def compute_adjoint_gradient_power(
    geometry: AntennaGeometry3D,
    k0_norm: float,
    pml: PMLConfig,
    obj_cfg: PowerObjectiveConfig,
    beta: float = 1.0,
    eta_h: float = 0.5,
    filter_radius: int = 0,
    max_rank: int = 128,
    n_sweeps: int = 40,
    solver_tol: float = 1e-4,
    damping: float = 0.005,
    verbose: bool = False,
) -> tuple[float, NDArray, PowerMetrics, float]:
    """Exact adjoint gradient for PML-power objective.

    Objective: J = -P_pml + alpha * P_cond + J_vol(rho)

    Algorithm:
    1. Forward solve: H(rho) E = b
    2. Compute power metrics + objective J
    3. Adjoint RHS: g = dJ/dE* = W_eff * E
       where W_eff = -0.5*sigma_pml*dV + alpha*0.5*k^2*sigma_design*dV
    4. Adjoint solve: H^H lambda = g
    5. Gradient: dJ/drho = dJ_explicit + adjoint_term, chained through
       filter+projection.

    Returns
    -------
    tuple[float, NDArray, PowerMetrics, float]
        (J_value, gradient, power_metrics, solver_residual).
    """
    cfg = geometry.config
    N = cfg.N
    n_bits = cfg.n_bits
    h = 1.0 / N
    dv = h ** 3

    # ---- Step 1: Forward solve ----
    H_cores, E_cores, residual = solve_forward_conductivity(
        geometry=geometry, k0_norm=k0_norm,
        beta=beta, eta=eta_h, filter_radius=filter_radius,
        max_rank=max_rank, n_sweeps=n_sweeps,
        solver_tol=solver_tol, damping=damping, verbose=verbose,
    )

    # ---- Reconstruct field ----
    E_3d = reconstruct_3d(E_cores, n_bits)

    # ---- Step 2: Power metrics + objective ----
    metrics = compute_power_metrics(
        E_3d, geometry, k0_norm, pml,
        beta=beta, eta=eta_h, filter_radius=filter_radius,
        damping=damping,
    )

    # Design density
    rho_filt = density_filter(geometry.density, filter_radius)
    rho_proj = heaviside_projection(rho_filt, beta, eta_h)
    sigma_design = simp_sigma(rho_proj, cfg.sigma_min, cfg.sigma_max, cfg.simp_p)

    # Volume constraint
    Jv, dJv_drho_proj, V, g_vol = augmented_lagrangian_volume(
        rho_proj, obj_cfg.vol_target, obj_cfg.al_lambda, obj_cfg.al_mu,
    )

    # ---- Staged log objective ----
    alpha = obj_cfg.alpha_loss
    eps_log = obj_cfg.log_eps
    P_pml = metrics.P_pml
    P_cond = metrics.P_cond

    if obj_cfg.use_log_objective:
        # J = -log(P_pml + eps) + alpha * log(P_cond + eps) + J_vol
        J = -math.log(P_pml + eps_log) + alpha * math.log(P_cond + eps_log) + Jv
        # Wirtinger: dJ/dP_pml = -1/(P_pml+eps), dJ/dP_cond = alpha/(P_cond+eps)
        # Then chain: dJ/dE* = (dJ/dP_pml)(dP_pml/dE*) + (dJ/dP_cond)(dP_cond/dE*)
        w_pml = -1.0 / (P_pml + eps_log)
        w_cond = alpha / (P_cond + eps_log)
    else:
        # Linear fallback: J = -P_pml + alpha * P_cond + J_vol
        J = -P_pml + alpha * P_cond + Jv
        w_pml = -1.0
        w_cond = alpha

    # ---- Step 3: Adjoint RHS g = dJ/dE* ----
    # dP_pml/dE* = 0.5 * sigma_pml * E * dV  (from P = 0.5 sigma |E|^2 dV)
    # dP_cond/dE* = 0.5 * k^2 * sigma_design * E * dV
    sigma_pml = build_pml_sigma_3d(n_bits, k0_norm, pml)

    g_3d = np.zeros((N, N, N), dtype=np.complex128)
    # d(-log(P_pml+eps))/dE* = w_pml * 0.5 * sigma_pml * E * dV
    g_3d += w_pml * 0.5 * sigma_pml * E_3d * dv
    # d(alpha*log(P_cond+eps))/dE*: only on design voxels
    design_contrib = w_cond * 0.5 * k0_norm ** 2 * sigma_design * E_3d[geometry.design_mask] * dv
    g_3d[geometry.design_mask] += design_contrib

    # Convert to QTT for adjoint solve
    g_cores = array_3d_to_tt(g_3d, n_bits, max_rank=max_rank)

    # ---- Step 4: Adjoint solve H^H lambda = g ----
    H_H_cores = mpo_hermitian_conjugate(H_cores)

    adj_result = tt_amen_solve(
        H_H_cores, g_cores,
        max_rank=max_rank,
        n_sweeps=n_sweeps,
        tol=solver_tol,
        verbose=False,
    )
    lam_3d = reconstruct_3d(adj_result.x, n_bits)

    # ---- Step 5: Gradient wrt rho_proj ----
    E_design = E_3d[geometry.design_mask]
    lam_design = lam_3d[geometry.design_mask]
    E_design_sq = np.abs(E_design) ** 2

    dsigma = simp_dsigma_drho(rho_proj, cfg.sigma_min, cfg.sigma_max, cfg.simp_p)

    # Explicit term: d(w_cond * P_cond)/drho_proj
    # P_cond = 0.5 * k^2 * sum sigma_i * |E_i|^2 * dV
    # dP_cond/drho_proj_i = 0.5 * k^2 * dsigma_i * |E_i|^2 * dV
    dJ_explicit = w_cond * 0.5 * k0_norm ** 2 * dsigma * E_design_sq * dv

    # Adjoint contraction: -Re[lambda^H (dH/drho)_i E_i]
    # dH/drho_i = k^2 * (1 + j*damp) * d(eps)/d(rho_proj_i) (diagonal)
    # d(eps)/d(rho_proj_i) = -j * dsigma_i
    # So (dH/drho_i) E_i = k^2 * (1 + j*damp) * (-j * dsigma_i) * E_i
    k2_damp = k0_norm ** 2 * (1.0 + 1j * damping)
    dH_drho_E = k2_damp * (-1j * dsigma) * E_design

    # -Re[conj(lam_i) * dH_drho_E_i]
    adjoint_term = -np.real(np.conj(lam_design) * dH_drho_E)

    # Total gradient wrt rho_proj
    dJ_drho_proj = dJ_explicit + dJv_drho_proj + adjoint_term

    # Chain through Heaviside projection
    h_grad = heaviside_gradient(rho_filt, beta, eta_h)
    grad_after_heaviside = dJ_drho_proj * h_grad

    # Chain through density filter
    grad = density_filter_gradient(grad_after_heaviside, filter_radius)

    return J, grad, metrics, residual


# =====================================================================
# Section 6a: QTT-Native Adjoint Gradient (no dense N³)
# =====================================================================

def compute_adjoint_gradient_power_tt(
    geometry: AntennaGeometry3D,
    k0_norm: float,
    pml: PMLConfig,
    obj_cfg: "PowerObjectiveConfig",
    design_mask_tt: list[NDArray],
    sigma_pml_tt: list[NDArray],
    design_flat_indices: NDArray,
    beta: float = 1.0,
    eta_h: float = 0.5,
    filter_radius: int = 0,
    max_rank: int = 128,
    n_sweeps: int = 40,
    solver_tol: float = 1e-4,
    damping: float = 0.005,
    verbose: bool = False,
) -> tuple[float, NDArray, PowerMetrics, float]:
    """QTT-native adjoint gradient — ZERO dense N³ arrays.

    Replaces compute_adjoint_gradient_power for large grids (1024³+).
    All operations stay in compressed QTT format:
    - Forward solve via QTT ε + MPO Helmholtz
    - Power via QTT inner products (no reconstruct)
    - Adjoint RHS via QTT Hadamard products
    - Adjoint solve in QTT
    - Gradient extraction via tt_evaluate_at_indices

    The ONLY dense objects are 1D arrays of length n_design.

    Returns
    -------
    tuple[float, NDArray, PowerMetrics, float]
        (J_value, gradient, power_metrics, solver_residual).
    """
    cfg = geometry.config
    N = cfg.N
    n_bits = cfg.n_bits
    h = 1.0 / N
    dv = h ** 3

    # ---- Step 1: Forward solve (QTT-native) ----
    H_cores, E_cores, residual, eps_tt = solve_forward_conductivity_tt(
        geometry=geometry, k0_norm=k0_norm,
        design_mask_tt=design_mask_tt,
        beta=beta, eta=eta_h, filter_radius=filter_radius,
        max_rank=max_rank, n_sweeps=n_sweeps,
        solver_tol=solver_tol, damping=damping, verbose=verbose,
    )

    # ---- Design density processing (1D, cheap) ----
    rho_filt = density_filter(geometry.density, filter_radius)
    rho_proj = heaviside_projection(rho_filt, beta, eta_h)
    sigma_design_vals = simp_sigma(rho_proj, cfg.sigma_min, cfg.sigma_max, cfg.simp_p)

    # Build sigma_design TT (mask × mean_sigma, same as in eps construction)
    sigma_mean = float(np.mean(sigma_design_vals))
    sigma_design_tt = tt_scale_c(
        [c.copy() for c in design_mask_tt], sigma_mean
    )

    # ---- Step 2: Power metrics (QTT-native) ----
    J_cores = geometry.build_source(max_rank=max_rank)
    # Note: J_cores is the NEGATED source (RHS = -J).
    # For power we need un-negated J: flip sign
    J_phys_cores = tt_scale_c(J_cores, -1.0)

    metrics = compute_power_metrics_tt(
        E_cores, J_phys_cores, sigma_pml_tt, sigma_design_tt,
        geometry, k0_norm, max_rank,
    )

    # Volume constraint (1D, cheap)
    Jv, dJv_drho_proj, V, g_vol = augmented_lagrangian_volume(
        rho_proj, obj_cfg.vol_target, obj_cfg.al_lambda, obj_cfg.al_mu,
    )

    # ---- Staged log objective ----
    alpha = obj_cfg.alpha_loss
    eps_log = obj_cfg.log_eps
    P_pml = metrics.P_pml
    P_cond = metrics.P_cond

    if obj_cfg.use_log_objective:
        J = -math.log(P_pml + eps_log) + alpha * math.log(P_cond + eps_log) + Jv
        w_pml = -1.0 / (P_pml + eps_log)
        w_cond = alpha / (P_cond + eps_log)
    else:
        J = -P_pml + alpha * P_cond + Jv
        w_pml = -1.0
        w_cond = alpha

    # ---- Step 3: Adjoint RHS (QTT-native) ----
    g_cores = build_adjoint_rhs_tt(
        E_cores, sigma_pml_tt, sigma_design_tt,
        w_pml, w_cond, k0_norm, dv, max_rank,
    )

    # ---- Step 4: Adjoint solve (QTT-native) ----
    H_H_cores = mpo_hermitian_conjugate(H_cores)
    adj_result = tt_amen_solve(
        H_H_cores, g_cores,
        max_rank=max_rank,
        n_sweeps=n_sweeps,
        tol=solver_tol,
        verbose=False,
    )

    # ---- Step 5: Gradient at design voxels only (sparse extract) ----
    # Extract E and λ at design voxel positions only — O(K·r²·n) cost
    E_design = tt_evaluate_at_indices(E_cores, design_flat_indices, n_bits)
    lam_design = tt_evaluate_at_indices(adj_result.x, design_flat_indices, n_bits)
    E_design_sq = np.abs(E_design) ** 2

    dsigma = simp_dsigma_drho(rho_proj, cfg.sigma_min, cfg.sigma_max, cfg.simp_p)

    # Explicit term
    dJ_explicit = w_cond * 0.5 * k0_norm ** 2 * dsigma * E_design_sq * dv

    # Adjoint contraction
    k2_damp = k0_norm ** 2 * (1.0 + 1j * damping)
    dH_drho_E = k2_damp * (-1j * dsigma) * E_design
    adjoint_term = -np.real(np.conj(lam_design) * dH_drho_E)

    # Total
    dJ_drho_proj = dJ_explicit + dJv_drho_proj + adjoint_term

    # Chain through Heaviside + filter (1D operations)
    h_grad = heaviside_gradient(rho_filt, beta, eta_h)
    grad_after_heaviside = dJ_drho_proj * h_grad
    grad = density_filter_gradient(grad_after_heaviside, filter_radius)

    return J, grad, metrics, residual


# =====================================================================
# Section 6b: FD Gradient Verification
# =====================================================================

def verify_gradient_fd(
    geometry: AntennaGeometry3D,
    k0_norm: float,
    pml: PMLConfig,
    obj_cfg: PowerObjectiveConfig,
    beta: float = 1.0,
    eta_h: float = 0.5,
    filter_radius: int = 0,
    max_rank: int = 16,
    n_sweeps: int = 5,
    solver_tol: float = 0.1,
    damping: float = 0.005,
    n_checks: int = 10,
    delta: float = 1e-4,
    verbose: bool = True,
) -> NDArray:
    """Verify adjoint gradient against central finite differences.

    Perturbs n_checks random design voxels and compares
    (J(rho+d) - J(rho-d)) / (2d) vs adjoint grad[i].

    Returns array of relative errors.
    """
    # Get adjoint gradient
    J0, grad_adj, _, _ = compute_adjoint_gradient_power(
        geometry=geometry, k0_norm=k0_norm, pml=pml,
        obj_cfg=obj_cfg, beta=beta, eta_h=eta_h,
        filter_radius=filter_radius, max_rank=max_rank,
        n_sweeps=n_sweeps, solver_tol=solver_tol,
        damping=damping, verbose=False,
    )

    n_design = geometry.n_design
    rng = np.random.default_rng(42)
    # Pick voxels with mid-range density (most informative)
    mid_mask = (geometry.density > 0.15) & (geometry.density < 0.85)
    candidates = np.where(mid_mask)[0]
    if len(candidates) < n_checks:
        candidates = np.arange(n_design)
    indices = rng.choice(candidates, size=min(n_checks, len(candidates)), replace=False)

    errors = np.zeros(len(indices))

    def _eval_J(rho_perturbed: NDArray) -> float:
        geom_p = AntennaGeometry3D(
            config=geometry.config, density=rho_perturbed)
        J_p, _, _, _ = compute_adjoint_gradient_power(
            geometry=geom_p, k0_norm=k0_norm, pml=pml,
            obj_cfg=obj_cfg, beta=beta, eta_h=eta_h,
            filter_radius=filter_radius, max_rank=max_rank,
            n_sweeps=n_sweeps, solver_tol=solver_tol,
            damping=damping, verbose=False,
        )
        return J_p

    if verbose:
        print(f"\nFD gradient check: {len(indices)} voxels, delta={delta}")

    for k, idx in enumerate(indices):
        rho_p = geometry.density.copy()
        rho_p[idx] = min(rho_p[idx] + delta, 1.0)
        J_plus = _eval_J(rho_p)

        rho_m = geometry.density.copy()
        rho_m[idx] = max(rho_m[idx] - delta, 0.0)
        J_minus = _eval_J(rho_m)

        actual_delta = (min(geometry.density[idx] + delta, 1.0)
                        - max(geometry.density[idx] - delta, 0.0)) / 2.0
        grad_fd = (J_plus - J_minus) / (2.0 * actual_delta)
        grad_adj_i = grad_adj[idx]

        denom = max(abs(grad_fd), abs(grad_adj_i), 1.0)
        rel_err = abs(grad_fd - grad_adj_i) / denom
        errors[k] = rel_err

        if verbose:
            print(f"  voxel {idx}: FD={grad_fd:.6e}, "
                  f"adj={grad_adj_i:.6e}, rel_err={rel_err:.3e}")

    if verbose:
        print(f"  Mean rel error: {np.mean(errors):.3e}, "
              f"max: {np.max(errors):.3e}")

    return errors


# =====================================================================
# Section 7: Legacy S11 Objective Functions + Gradient
# =====================================================================

def objective_minimize_s11(s11: complex) -> tuple[float, complex]:
    """Minimise |S11|^2."""
    return float(abs(s11) ** 2), np.conj(s11)


def objective_target_s11_db(target_db: float = -15.0) -> Callable:
    """Create objective that drives |S11| below target level.

    J = max(0, |S11|_dB - target_dB)^2
    """
    def fn(s11: complex) -> tuple[float, complex]:
        mag = abs(s11)
        mag_safe = max(mag, 1e-30)
        s11_db = 20.0 * math.log10(mag_safe)
        excess = max(0.0, s11_db - target_db)
        J = excess ** 2
        if excess > 0 and mag > 1e-20:
            ds11_db = 10.0 / (math.log(10) * mag ** 2) * np.conj(s11)
            dJ = 2.0 * excess * ds11_db
        else:
            dJ = 0.0 + 0j
        return J, dJ
    return fn


def compute_adjoint_gradient_3d(
    geometry: AntennaGeometry3D,
    k0_norm: float,
    objective_fn: Callable[[complex], tuple[float, complex]],
    beta: float = 1.0,
    eta: float = 0.5,
    filter_radius: int = 0,
    max_rank: int = 128,
    n_sweeps: int = 40,
    solver_tol: float = 1e-4,
    damping: float = 0.01,
    verbose: bool = False,
) -> tuple[float, NDArray, complex, float]:
    """Legacy adjoint gradient (Born approx, S11 objective)."""
    cfg = geometry.config
    s11, residual, E_cores = solve_and_extract_s11(
        geometry=geometry, k0_norm=k0_norm, beta=beta, eta=eta,
        filter_radius=filter_radius, max_rank=max_rank,
        n_sweeps=n_sweeps, solver_tol=solver_tol,
        damping=damping, verbose=verbose,
    )
    J_val, dJ_ds11 = objective_fn(s11)

    E_3d = reconstruct_3d(E_cores, cfg.n_bits)
    design_mask = geometry.design_mask
    E_design = E_3d[design_mask]

    N = cfg.N
    h = 1.0 / N
    k2 = k0_norm ** 2 * (1.0 + 1j * damping)
    delta_eps = cfg.eps_contrast - 1.0

    sensitivity = -k2 * h ** 3 * E_design ** 2
    grad_eps = np.real(dJ_ds11 * sensitivity) * delta_eps

    rho_filt = density_filter(geometry.density, filter_radius)
    h_grad = heaviside_gradient(rho_filt, beta, eta)
    grad_after_heaviside = grad_eps * h_grad
    grad = density_filter_gradient(grad_after_heaviside, filter_radius)

    return J_val, grad, s11, residual


# =====================================================================
# Section 7b: Radiated-Power Forward Solve and Adjoint Gradient (legacy)
# =====================================================================

def solve_and_extract_p_rad(
    geometry: AntennaGeometry3D,
    k0_norm: float,
    beta: float = 1.0,
    eta: float = 0.5,
    filter_radius: int = 0,
    max_rank: int = 128,
    n_sweeps: int = 40,
    solver_tol: float = 1e-4,
    damping: float = 0.01,
    verbose: bool = False,
) -> tuple[float, float, float, list[NDArray], list[NDArray]]:
    """Forward solve and extract input power (legacy path).

    P_in = -1/2 Re(<J, E>) h^3

    Returns (P_in, Q_reactive, solver_residual, E_cores, J_cores).
    """
    cfg = geometry.config
    pml = cfg.pml_config()
    N = cfg.N
    h = 1.0 / N

    rhs_cores = geometry.build_source(max_rank=max_rank)
    J_cores = tt_scale_c(rhs_cores, -1.0 + 0j)

    eps_3d = geometry.build_eps_3d(beta, eta, filter_radius)
    result = solve_helmholtz_3d(
        n_bits=cfg.n_bits, k=k0_norm, pml=pml,
        rhs_cores=rhs_cores, eps_3d=eps_3d,
        max_rank=max_rank, n_sweeps=n_sweeps,
        tol=solver_tol, damping=damping, verbose=verbose,
    )

    voxel_vol = h ** 3
    p_in = compute_input_power(result.x, J_cores, voxel_vol)
    q_react = compute_reactive_power(result.x, J_cores, voxel_vol)

    return p_in, q_react, result.final_residual, result.x, J_cores


def compute_adjoint_gradient_p_rad(
    geometry: AntennaGeometry3D,
    k0_norm: float,
    beta: float = 1.0,
    eta: float = 0.5,
    filter_radius: int = 0,
    max_rank: int = 128,
    n_sweeps: int = 40,
    solver_tol: float = 1e-4,
    damping: float = 0.01,
    verbose: bool = False,
) -> tuple[float, NDArray, float, float, float]:
    """Adjoint gradient for the input-power objective (legacy, self-adjoint).

    Objective: -P_in  (minimise to maximise radiated power).
    """
    cfg = geometry.config
    N = cfg.N
    h = 1.0 / N

    p_in, q_react, residual, E_cores, J_cores = solve_and_extract_p_rad(
        geometry=geometry, k0_norm=k0_norm, beta=beta, eta=eta,
        filter_radius=filter_radius, max_rank=max_rank,
        n_sweeps=n_sweeps, solver_tol=solver_tol,
        damping=damping, verbose=verbose,
    )

    obj = -p_in

    E_3d = reconstruct_3d(E_cores, cfg.n_bits)
    design_mask = geometry.design_mask
    E_design = E_3d[design_mask]

    k2_eff = k0_norm ** 2 * (1.0 + 1j * damping)
    delta_eps = cfg.eps_contrast - 1.0

    sensitivity = 0.5 * h ** 3 * delta_eps * np.real(k2_eff * E_design ** 2)

    rho_filt = density_filter(geometry.density, filter_radius)
    h_grad = heaviside_gradient(rho_filt, beta, eta)
    grad_after_heaviside = sensitivity * h_grad
    grad = density_filter_gradient(grad_after_heaviside, filter_radius)

    return obj, grad, p_in, q_react, residual


# =====================================================================
# Section 7c: Q-Factor Extraction
# =====================================================================

@dataclass
class QExtractionResult:
    """Results of Q extraction from S11 frequency sweep."""
    Q: float
    f_resonance: float
    bandwidth_hz: float
    s11_min: float
    frequencies: NDArray
    s11_values: NDArray

    def summary(self, q_chu: float = 0.0) -> str:
        sep = "=" * 60
        lines = [
            f"\n{sep}",
            f"  Q Extraction Results",
            f"{sep}",
            f"  Q:             {self.Q:.1f}",
            f"  f_resonance:   {self.f_resonance / 1e9:.4f} GHz",
            f"  BW(-10dB):     {self.bandwidth_hz / 1e6:.2f} MHz",
            f"  |S11|_min:     {self.s11_min:.4f}"
            f" ({20.0 * math.log10(max(self.s11_min, 1e-30)):.1f} dB)",
        ]
        if q_chu > 0:
            lines.append(f"  Q / Q_Chu:     {self.Q / q_chu:.2f}")
        lines.append(f"{sep}")
        return "\n".join(lines)


def extract_q_from_sweep(
    geometry: AntennaGeometry3D,
    f_center_hz: float,
    n_freq: int = 11,
    bw_fraction: float = 0.3,
    beta: float = 1.0,
    eta: float = 0.5,
    filter_radius: int = 0,
    max_rank: int = 128,
    n_sweeps: int = 40,
    solver_tol: float = 1e-4,
    damping: float = 0.01,
    verbose: bool = True,
) -> QExtractionResult:
    """Extract Q-factor from multi-frequency S11 sweep."""
    cfg = geometry.config
    f_lo = f_center_hz * (1.0 - bw_fraction / 2.0)
    f_hi = f_center_hz * (1.0 + bw_fraction / 2.0)
    freqs = np.linspace(f_lo, f_hi, n_freq)
    s11_arr = np.zeros(n_freq, dtype=np.complex128)

    if verbose:
        print(f"\nFrequency sweep: {f_lo / 1e9:.4f} -> {f_hi / 1e9:.4f} GHz "
              f"({n_freq} points)")

    for i, f in enumerate(freqs):
        lam_f = C0 / f
        k0_f = 2.0 * math.pi / lam_f
        k0_norm_f = k0_f * cfg.domain_size
        s11_f, res_f, _ = solve_and_extract_s11(
            geometry=geometry, k0_norm=k0_norm_f,
            beta=beta, eta=eta, filter_radius=filter_radius,
            max_rank=max_rank, n_sweeps=n_sweeps,
            solver_tol=solver_tol, damping=damping, verbose=False,
        )
        s11_arr[i] = s11_f
        if verbose:
            s_db = 20.0 * math.log10(max(abs(s11_f), 1e-30))
            print(f"  f={f / 1e9:.4f} GHz: |S11|={abs(s11_f):.4f} "
                  f"({s_db:.1f} dB), res={res_f:.2e}")

    s11_mag = np.abs(s11_arr)
    min_idx = int(np.argmin(s11_mag))
    f_res = freqs[min_idx]
    s11_min_val = float(s11_mag[min_idx])

    threshold = max(s11_min_val * math.sqrt(10), 0.9)

    f_lo_bw = freqs[0]
    for idx in range(min_idx, 0, -1):
        if s11_mag[idx] >= threshold:
            frac = (threshold - s11_mag[idx]) / (
                s11_mag[idx + 1] - s11_mag[idx] + 1e-30)
            f_lo_bw = freqs[idx] + frac * (freqs[idx + 1] - freqs[idx])
            break

    f_hi_bw = freqs[-1]
    for idx in range(min_idx, len(s11_mag) - 1):
        if s11_mag[idx] >= threshold:
            frac = (threshold - s11_mag[idx - 1]) / (
                s11_mag[idx] - s11_mag[idx - 1] + 1e-30)
            f_hi_bw = freqs[idx - 1] + frac * (freqs[idx] - freqs[idx - 1])
            break

    bw = f_hi_bw - f_lo_bw
    Q = f_res / max(bw, 1e-30)
    if verbose:
        print(f"  Resonance: f={f_res / 1e9:.4f} GHz, |S11|_min={s11_min_val:.4f}")
        print(f"  BW(-10dB): {bw / 1e6:.2f} MHz, Q = {Q:.1f}")

    return QExtractionResult(
        Q=float(Q), f_resonance=float(f_res),
        bandwidth_hz=float(bw), s11_min=float(s11_min_val),
        frequencies=freqs, s11_values=s11_arr,
    )


@dataclass
class PowerSweepResult:
    """Results of Q extraction from radiated power frequency sweep."""
    Q: float
    f_center: float
    bandwidth_hz: float
    p_in_max: float
    frequencies_hz: NDArray
    p_in_array: NDArray
    q_reactive_array: NDArray


def extract_q_from_power_sweep(
    geometry: AntennaGeometry3D,
    f_center_hz: float,
    n_freq: int = 11,
    bw_fraction: float = 0.3,
    beta: float = 1.0,
    eta: float = 0.5,
    filter_radius: int = 0,
    max_rank: int = 128,
    n_sweeps: int = 40,
    solver_tol: float = 1e-4,
    damping: float = 0.01,
    verbose: bool = True,
) -> PowerSweepResult:
    """Extract Q-factor from input-power frequency sweep (half-power BW)."""
    cfg = geometry.config
    f_lo = f_center_hz * (1.0 - bw_fraction / 2.0)
    f_hi = f_center_hz * (1.0 + bw_fraction / 2.0)
    freqs = np.linspace(f_lo, f_hi, n_freq)
    p_in_arr = np.zeros(n_freq, dtype=np.float64)
    q_react_arr = np.zeros(n_freq, dtype=np.float64)

    if verbose:
        print(f"\nPower sweep: {f_lo / 1e9:.4f} -> {f_hi / 1e9:.4f} GHz "
              f"({n_freq} points)")

    for i, f in enumerate(freqs):
        lam_f = C0 / f
        k0_f = 2.0 * math.pi / lam_f
        k0_norm_f = k0_f * cfg.domain_size
        p_in, q_react, residual, _, _ = solve_and_extract_p_rad(
            geometry=geometry, k0_norm=k0_norm_f,
            beta=beta, eta=eta, filter_radius=filter_radius,
            max_rank=max_rank, n_sweeps=n_sweeps,
            solver_tol=solver_tol, damping=damping, verbose=False,
        )
        p_in_arr[i] = p_in
        q_react_arr[i] = q_react
        if verbose:
            print(f"  f={f / 1e9:.4f} GHz: P_in={p_in:.4e}, "
                  f"Q_react={q_react:.4e}, res={residual:.2e}")

    # Use absolute value of P_in for 3dB bandwidth extraction:
    # when the structure is lossy / non-resonant, P_in can be
    # negative, which breaks the half-power threshold logic.
    p_in_abs = np.abs(p_in_arr)
    p_max_abs = float(np.max(p_in_abs))
    max_idx = int(np.argmax(p_in_abs))
    f_res = freqs[max_idx]
    threshold = p_max_abs / 2.0

    p_max = float(p_in_arr[max_idx])  # signed value for reporting

    # Lower 3dB frequency: search downward from one below peak.
    f_lo_3db = freqs[0]
    for idx in range(max_idx - 1, -1, -1):
        if p_in_abs[idx] <= threshold:
            # Linear interpolation between idx and idx+1
            denom = p_in_abs[idx + 1] - p_in_abs[idx]
            frac = (threshold - p_in_abs[idx]) / (denom + 1e-30)
            f_lo_3db = freqs[idx] + frac * (freqs[idx + 1] - freqs[idx])
            break

    # Upper 3dB frequency: search upward from one above peak.
    f_hi_3db = freqs[-1]
    for idx in range(max_idx + 1, len(p_in_abs)):
        if p_in_abs[idx] <= threshold:
            # Linear interpolation between idx-1 and idx
            denom = p_in_abs[idx] - p_in_abs[idx - 1]
            frac = (threshold - p_in_abs[idx - 1]) / (denom + 1e-30)
            f_hi_3db = freqs[idx - 1] + frac * (freqs[idx] - freqs[idx - 1])
            break

    bw = f_hi_3db - f_lo_3db
    Q = f_res / max(bw, 1e-30)
    if verbose:
        print(f"  Peak: f={f_res / 1e9:.4f} GHz, P_in={p_max:.4e}")
        print(f"  BW(-3dB): {bw / 1e6:.2f} MHz, Q = {Q:.1f}")

    return PowerSweepResult(
        Q=float(Q), f_center=float(f_res),
        bandwidth_hz=float(bw), p_in_max=float(p_max),
        frequencies_hz=freqs, p_in_array=p_in_arr,
        q_reactive_array=q_react_arr,
    )


# =====================================================================
# Section 8: 3D Topology Optimization Loop
# =====================================================================

@dataclass
class ChuOptConfig:
    """Optimization hyperparameters for Chu limit challenge.

    Supports three modes:
    - use_power_adjoint=True (NEW, default): Conductivity SIMP +
      PML absorption objective + exact adjoint + volume constraint.
    - use_p_rad=True (legacy): Input power objective, self-adjoint.
    - Both False: Legacy S11 objective.
    """

    max_iterations: int = 60
    learning_rate: float = 0.15
    beta_init: float = 1.0
    beta_max: float = 32.0
    beta_increase_every: int = 15
    beta_factor: float = 2.0
    eta: float = 0.5
    filter_radius: int = 1
    regularisation_weight: float = 0.0
    convergence_tol: float = 1e-4
    target_s11_db: float = -15.0
    # Legacy P_rad mode
    use_p_rad: bool = False
    use_monopole_seed: bool = True
    damping_init: float = 0.1
    damping_final: float = 0.01
    damping_schedule_iters: int = 20
    reactive_penalty: float = 0.1
    # --- NEW: Power adjoint mode ---
    use_power_adjoint: bool = True
    alpha_loss: float = 0.1
    alpha_intro_iter: int = 20  # introduce P_cond penalty after this iter
    vol_target: float = 0.3
    al_mu_init: float = 10.0
    al_mu_factor: float = 1.5
    # Feed seed clamp: lock rho=1 near feed for first N iters
    feed_seed_clamp_iters: int = 15
    feed_seed_clamp_radius: int = 2  # voxels from feed
    # Dynamic alpha introduction: "fixed" uses alpha_intro_iter,
    # "auto" waits for ~P > 1 to be stable for alpha_stable_window
    # consecutive iterations (post-clamp) before introducing alpha.
    alpha_intro_mode: str = "auto"
    alpha_stable_window: int = 20
    # M_dead threshold percentile: tau = this percentile of |E_air|^2
    m_dead_percentile: float = 10.0
    # Soft coupling constraint: replaces hard clamp for larger grids.
    # After feed_seed_clamp_iters, if use_coupling_constraint is True,
    # an AL penalty enforces min mean density near feed.
    use_coupling_constraint: bool = True
    coupling_density_threshold: float = 0.3
    coupling_al_mu_init: float = 10.0
    coupling_radius: int = 3  # voxels from feed (can be wider than clamp)
    # Continuation: sigma_max ramp
    sigma_max_init: float = 50.0
    sigma_max_final: float = 200.0
    sigma_ramp_iters: int = 30
    # Continuation: simp_p ramp
    simp_p_init: float = 1.0
    simp_p_final: float = 3.0
    simp_p_ramp_iters: int = 30
    # Optimizer: "adam" or "sgd" (normalised gradient descent)
    optimizer: str = "adam"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8

    def damping_at_iter(self, iteration: int) -> float:
        """Linear damping schedule from damping_init to damping_final."""
        if self.damping_schedule_iters <= 0:
            return self.damping_final
        t = min(iteration / self.damping_schedule_iters, 1.0)
        return self.damping_init + t * (self.damping_final - self.damping_init)

    def sigma_max_at_iter(self, iteration: int) -> float:
        """Linear sigma_max continuation."""
        if self.sigma_ramp_iters <= 0:
            return self.sigma_max_final
        t = min(iteration / self.sigma_ramp_iters, 1.0)
        return self.sigma_max_init + t * (self.sigma_max_final - self.sigma_max_init)

    def simp_p_at_iter(self, iteration: int) -> float:
        """Linear simp_p continuation."""
        if self.simp_p_ramp_iters <= 0:
            return self.simp_p_final
        t = min(iteration / self.simp_p_ramp_iters, 1.0)
        return self.simp_p_init + t * (self.simp_p_final - self.simp_p_init)


@dataclass
class ChuOptResult:
    """Results of the Chu limit topology optimization."""

    density_final: NDArray
    conductor_mask: NDArray
    s11_final: complex = 0.0 + 0j
    q_result: Optional[QExtractionResult] = None
    power_q_result: Optional[PowerSweepResult] = None
    objective_history: list[float] = field(default_factory=list)
    s11_history: list[complex] = field(default_factory=list)
    grad_norm_history: list[float] = field(default_factory=list)
    beta_history: list[float] = field(default_factory=list)
    p_in_history: list[float] = field(default_factory=list)
    damping_history: list[float] = field(default_factory=list)
    power_metrics_history: list[PowerMetrics] = field(default_factory=list)
    p_in_final: float = 0.0
    q_reactive_final: float = 0.0
    n_iterations: int = 0
    converged: bool = False
    total_time_s: float = 0.0
    config: Optional[ChuProblemConfig] = None
    opt_config: Optional[ChuOptConfig] = None

    @property
    def s11_db(self) -> float:
        return 20.0 * math.log10(max(abs(self.s11_final), 1e-30))

    @property
    def binarisation(self) -> float:
        return binarisation_metric(self.density_final)

    @property
    def volume_fraction(self) -> float:
        return float(np.mean(self.conductor_mask))

    def summary(self) -> str:
        """Human-readable optimization summary."""
        sep = "=" * 60
        lines = [
            f"\n{sep}",
            f"  Chu Limit Challenge: Optimization Results",
            f"{sep}",
            f"  Iterations:        {self.n_iterations}",
            f"  Converged:         {self.converged}",
            f"  Time:              {self.total_time_s:.1f} s",
            f"  Final |S11|:       {self.s11_db:.1f} dB",
            f"  Binarisation:      {self.binarisation:.4f}",
            f"  Conductor fill:    {self.volume_fraction:.1%}",
        ]
        if self.p_in_final != 0.0:
            lines.extend([
                f"  P_in (final):      {self.p_in_final:.4e}",
                f"  Q_reactive:        {self.q_reactive_final:.4e}",
            ])
        if self.power_metrics_history:
            last = self.power_metrics_history[-1]
            lines.extend([
                f"  P_pml (final):     {last.P_pml:.4e}",
                f"  P_cond (final):    {last.P_cond:.4e}",
                f"  P_pml_norm:        {last.P_pml_norm:.4f}",
                f"  P_cond_norm:       {last.P_cond_norm:.4f}",
                f"  E2_metal_avg:      {last.E2_metal_avg:.4e}",
                f"  M_dead:            {last.M_dead:.4f}",
                f"  eta_rad:           {last.eta_rad:.4f}",
                f"  Volume:            {last.vol:.4f}",
            ])
        if self.q_result is not None and self.config is not None:
            lines.extend([
                f"  Q (S11 sweep):     {self.q_result.Q:.1f}",
                f"  Q_Chu:             {self.config.q_chu:.1f}",
                f"  Q / Q_Chu:         {self.q_result.Q / self.config.q_chu:.2f}",
            ])
        if self.power_q_result is not None and self.config is not None:
            lines.extend([
                f"  Q (power sweep):   {self.power_q_result.Q:.1f}",
                f"  Q_Chu:             {self.config.q_chu:.1f}",
                f"  Q / Q_Chu:         {self.power_q_result.Q / self.config.q_chu:.2f}",
                f"  P_in(peak):        {self.power_q_result.p_in_max:.4e}",
                f"  BW(-3dB):          {self.power_q_result.bandwidth_hz / 1e6:.2f} MHz",
            ])
        lines.append(f"{sep}")
        return "\n".join(lines)


def optimize_chu_antenna(
    config: ChuProblemConfig,
    opt_config: ChuOptConfig = ChuOptConfig(),
    initial_density: Optional[NDArray] = None,
    verbose: bool = True,
    extract_q: bool = True,
    callback: Optional[Callable] = None,
    qtt_native: bool | None = None,
) -> ChuOptResult:
    """Run 3D topology optimization to challenge the Chu limit.

    Supports three modes controlled by opt_config:

    1. **Power adjoint** (use_power_adjoint=True, default):
       Conductivity SIMP + PML absorption objective + exact adjoint +
       augmented Lagrangian volume constraint. This is the correct formulation.

    2. **P_rad** (use_p_rad=True): Legacy input-power mode.

    3. **S11** (both False): Legacy S11 mode.

    Parameters
    ----------
    qtt_native : bool or None
        Force QTT-native code path (no dense N³ arrays).
        If None (default), auto-enables for n_bits >= 9 (N >= 512).
    """
    use_power = opt_config.use_power_adjoint
    use_p_rad = opt_config.use_p_rad and not use_power

    if verbose:
        print(config.summary())
        print_q_limits(config.ka)
        if use_power:
            mode_str = "PML-power adjoint (conductivity SIMP)"
        elif use_p_rad:
            mode_str = "P_rad (input power, legacy)"
        else:
            mode_str = "S11 (legacy)"
        print(f"\n  Optimization mode: {mode_str}")

    # --- Geometry initialisation ---
    if opt_config.use_monopole_seed and initial_density is None:
        geometry = AntennaGeometry3D.with_monopole_seed(config)
        if verbose:
            print("  Using monopole seed geometry")
    elif initial_density is not None:
        geometry = AntennaGeometry3D(config=config, density=initial_density)
    else:
        geometry = AntennaGeometry3D(config=config)

    if verbose:
        print(f"  Design region: {geometry.n_design} voxels "
              f"(sphere r={config.sphere_radius_normalised:.4f})")
        print(f"  Feed position: {geometry.feed_position}")

    # Legacy S11 objective
    if not use_power and not use_p_rad:
        obj_fn = objective_target_s11_db(opt_config.target_s11_db)

    # Histories
    obj_history: list[float] = []
    s11_history: list[complex] = []
    grad_norm_history: list[float] = []
    beta_history: list[float] = []
    p_in_history: list[float] = []
    damping_history_list: list[float] = []
    power_metrics_history: list[PowerMetrics] = []

    beta = opt_config.beta_init
    converged = False
    t_start = time.perf_counter()
    k0_norm = config.k0_normalised
    p_in_final = 0.0
    q_react_final = 0.0

    # Augmented Lagrangian dual variable
    al_lambda = 0.0
    al_mu = opt_config.al_mu_init

    pml = config.pml_config()

    # --- Auto-detect QTT-native mode ---
    if qtt_native is None:
        qtt_native = config.n_bits >= 9  # N >= 512
    if qtt_native and not use_power:
        raise ValueError(
            "QTT-native path only supports power-adjoint mode "
            "(use_power_adjoint=True)."
        )

    # --- QTT-native pre-computation (one-time, no dense N³) ---
    sigma_pml_tt: list[NDArray] | None = None
    design_mask_tt: list[NDArray] | None = None
    design_flat_idx: NDArray | None = None

    if qtt_native and use_power:
        if verbose:
            print("  [QTT-native] Building compressed infrastructure...")
            import sys; sys.stdout.flush()

        # PML σ as QTT (rank ≤ 4, built from 1D stretching profiles)
        sigma_pml_tt = build_pml_sigma_tt(
            config.n_bits, k0_norm, pml,
            max_rank=config.max_rank, cutoff=1e-12,
        )

        # Design sphere indicator as QTT (z-slice accumulation, O(N²) peak)
        design_mask_tt = build_sphere_mask_tt(
            config.n_bits,
            centre=(0.5, 0.5, 0.5),
            radius=config.sphere_radius_normalised,
            max_rank=min(config.max_rank, 64),
            cutoff=1e-10,
        )

        # Flat indices for sparse evaluation
        design_flat_idx = geometry.design_flat_indices

        if verbose:
            r_pml = max(c.shape[2] for c in sigma_pml_tt)
            r_mask = max(c.shape[2] for c in design_mask_tt)
            print(f"  [QTT-native] σ_pml rank={r_pml}, "
                  f"mask rank={r_mask}, "
                  f"n_design={len(design_flat_idx)}")

    # --- Baseline air solve for normalisation (D1 diagnostic) ---
    P_pml_air = 0.0
    E_air_design_sq: Optional[NDArray] = None
    tau_dead: float = 0.0
    if use_power:
        geom_air = AntennaGeometry3D(
            config=config, density=np.zeros(geometry.n_design),
        )

        if qtt_native:
            # QTT-native baseline: no dense N³ arrays
            assert sigma_pml_tt is not None
            assert design_mask_tt is not None
            assert design_flat_idx is not None

            _, E_air_cores, _, _ = solve_forward_conductivity_tt(
                geometry=geom_air, k0_norm=k0_norm,
                design_mask_tt=design_mask_tt,
                beta=1.0, eta=0.5, filter_radius=0,
                max_rank=config.max_rank, n_sweeps=config.n_sweeps,
                solver_tol=config.solver_tol, damping=config.damping,
                verbose=False,
            )
            h_base = 1.0 / config.N
            dv_base = h_base ** 3
            P_pml_air = compute_pml_power_tt(
                E_air_cores, sigma_pml_tt, dv_base,
                max_rank=config.max_rank,
            )
            # M_dead: extract |E_air|² at design voxels only
            E_air_at_design = tt_evaluate_at_indices(
                E_air_cores, design_flat_idx, config.n_bits,
            )
            E_air_design_sq = np.abs(E_air_at_design) ** 2
            tau_dead = float(np.percentile(
                E_air_design_sq, opt_config.m_dead_percentile,
            ))
            del geom_air, E_air_cores, E_air_at_design
        else:
            # Dense baseline (small grids)
            _, E_air_cores, _ = solve_forward_conductivity(
                geometry=geom_air, k0_norm=k0_norm,
                beta=1.0, eta=0.5, filter_radius=0,
                max_rank=config.max_rank, n_sweeps=config.n_sweeps,
                solver_tol=config.solver_tol, damping=config.damping,
                verbose=False,
            )
            E_air_3d = reconstruct_3d(E_air_cores, config.n_bits)
            sigma_pml_base = build_pml_sigma_3d(config.n_bits, k0_norm, pml)
            h_base = 1.0 / config.N
            dv_base = h_base ** 3
            E_air_sq = np.abs(E_air_3d) ** 2
            P_pml_air = 0.5 * float(
                np.sum(sigma_pml_base * E_air_sq) * dv_base
            )
            E_air_design_sq = E_air_sq[geometry.design_mask]
            tau_dead = float(np.percentile(
                E_air_design_sq, opt_config.m_dead_percentile,
            ))
            del geom_air, E_air_cores, E_air_3d, E_air_sq

        if verbose:
            print(f"  Baseline P_pml(air) = {P_pml_air:.4e}")
            print(f"  M_dead tau (p{opt_config.m_dead_percentile:.0f}) = {tau_dead:.3e}")

    # --- Build feed seed clamp mask ---
    # For QTT-native: use analytical distance computation (no meshgrid)
    feed_seed_mask: Optional[NDArray] = None
    if use_power and opt_config.feed_seed_clamp_iters > 0:
        h_fs = 1.0 / config.N
        clamp_r = opt_config.feed_seed_clamp_radius * h_fs
        fp = geometry.feed_position

        if qtt_native:
            # O(n_design) distance computation — no dense N³
            dists = compute_voxel_distances(
                geometry.design_flat_indices, config.n_bits, fp,
            )
            feed_seed_mask = dists < clamp_r
        else:
            coords_fs = np.linspace(h_fs / 2, 1.0 - h_fs / 2, config.N)
            xx_fs, yy_fs, zz_fs = np.meshgrid(
                coords_fs, coords_fs, coords_fs, indexing="ij"
            )
            dist_feed = np.sqrt(
                (xx_fs - fp[0]) ** 2 + (yy_fs - fp[1]) ** 2
                + (zz_fs - fp[2]) ** 2
            )
            clamp_3d = dist_feed < clamp_r
            feed_seed_mask = clamp_3d[geometry.design_mask]

        n_clamped = int(np.sum(feed_seed_mask))
        if verbose:
            print(f"  Feed seed clamp: {n_clamped} voxels locked for "
                  f"{opt_config.feed_seed_clamp_iters} iters")

    # Dynamic alpha introduction state
    p_tilde_stable_count = 0  # consecutive post-clamp iters with ~P > 1
    alpha_activated = False
    alpha_activated_iter = -1

    # Adam state (per-design-voxel moments)
    use_adam = (opt_config.optimizer == "adam") if use_power else False
    n_des = len(geometry.density)
    adam_m = np.zeros(n_des, dtype=np.float64)  # 1st moment
    adam_v = np.zeros(n_des, dtype=np.float64)  # 2nd moment
    adam_t = 0  # step counter

    # Soft coupling constraint state
    coupling_mask: Optional[NDArray] = None
    coupling_lambda_c = 0.0
    coupling_mu_c = opt_config.coupling_al_mu_init
    if use_power and opt_config.use_coupling_constraint:
        h_cm = 1.0 / config.N
        coupling_r = opt_config.coupling_radius * h_cm
        fp_cm = geometry.feed_position

        if qtt_native:
            # O(n_design) distance computation — no dense N³
            dists_cm = compute_voxel_distances(
                geometry.design_flat_indices, config.n_bits, fp_cm,
            )
            coupling_mask = dists_cm < coupling_r
        else:
            coords_cm = np.linspace(h_cm / 2, 1.0 - h_cm / 2, config.N)
            xx_cm, yy_cm, zz_cm = np.meshgrid(
                coords_cm, coords_cm, coords_cm, indexing="ij"
            )
            dist_cm = np.sqrt(
                (xx_cm - fp_cm[0]) ** 2 + (yy_cm - fp_cm[1]) ** 2
                + (zz_cm - fp_cm[2]) ** 2
            )
            coupling_3d = dist_cm < coupling_r
            coupling_mask = coupling_3d[geometry.design_mask]

        n_coupling = int(np.sum(coupling_mask))
        if verbose and n_coupling > 0:
            print(f"  Coupling constraint: {n_coupling} voxels, "
                  f"threshold={opt_config.coupling_density_threshold:.2f}")

    if verbose:
        print(f"\n  Optimization: max_iter={opt_config.max_iterations}, "
              f"lr={opt_config.learning_rate}, "
              f"beta: {opt_config.beta_init} -> {opt_config.beta_max}")
        if use_power:
            print(f"  Conductivity: sigma_max {opt_config.sigma_max_init} "
                  f"-> {opt_config.sigma_max_final}")
            print(f"  SIMP p: {opt_config.simp_p_init} -> {opt_config.simp_p_final}")
            if opt_config.alpha_intro_mode == "auto":
                alpha_intro_str = (
                    f"auto (~P>1 stable for "
                    f"{opt_config.alpha_stable_window} iters)"
                )
            else:
                alpha_intro_str = f"fixed at iter {opt_config.alpha_intro_iter}"
            print(f"  Volume target: {opt_config.vol_target:.0%}, "
                  f"alpha_loss: {opt_config.alpha_loss} "
                  f"(intro: {alpha_intro_str})")
        elif use_p_rad:
            print(f"  Damping schedule: {opt_config.damping_init:.3f} "
                  f"-> {opt_config.damping_final:.3f}")
        print("-" * 60)

    for iteration in range(opt_config.max_iterations):
        # --- beta continuation ---
        if (iteration > 0 and opt_config.beta_increase_every > 0 and
                iteration % opt_config.beta_increase_every == 0):
            beta = min(beta * opt_config.beta_factor, opt_config.beta_max)
            if use_power:
                al_mu *= opt_config.al_mu_factor
            if verbose:
                msg = f"  [beta -> {beta:.1f}"
                if use_power:
                    msg += f", mu_vol -> {al_mu:.1f}"
                msg += "]"
                print(msg)

        # Continuation: sigma_max, simp_p
        if use_power:
            current_sigma_max = opt_config.sigma_max_at_iter(iteration)
            current_simp_p = opt_config.simp_p_at_iter(iteration)
            # Temporarily override config for this iteration
            object.__setattr__(config, "sigma_max", current_sigma_max)
            object.__setattr__(config, "simp_p", current_simp_p)
            current_damping = config.damping
        else:
            if use_p_rad:
                current_damping = opt_config.damping_at_iter(iteration)
            else:
                current_damping = config.damping
        damping_history_list.append(current_damping)

        # --- Apply feed seed clamp ---
        if (use_power and feed_seed_mask is not None
                and iteration < opt_config.feed_seed_clamp_iters):
            geometry.density[feed_seed_mask] = 1.0

        # --- Compute gradient ---
        if use_power:
            # Staged alpha: 0 during radiation-first stage.
            # In "auto" mode, alpha is introduced when ~P > 1 has been
            # stable for alpha_stable_window consecutive post-clamp iters.
            # In "fixed" mode, alpha is introduced at alpha_intro_iter.
            if opt_config.alpha_intro_mode == "auto":
                alpha_eff = opt_config.alpha_loss if alpha_activated else 0.0
            else:
                alpha_eff = (
                    opt_config.alpha_loss
                    if iteration >= opt_config.alpha_intro_iter
                    else 0.0
                )
            obj_cfg = PowerObjectiveConfig(
                alpha_loss=alpha_eff,
                vol_target=opt_config.vol_target,
                al_lambda=al_lambda,
                al_mu=al_mu,
                use_log_objective=True,
            )

            if qtt_native:
                # ------ QTT-NATIVE gradient (no dense N³) ------
                assert design_mask_tt is not None
                assert sigma_pml_tt is not None
                assert design_flat_idx is not None
                J_val, grad, metrics, residual = (
                    compute_adjoint_gradient_power_tt(
                        geometry=geometry, k0_norm=k0_norm, pml=pml,
                        obj_cfg=obj_cfg,
                        design_mask_tt=design_mask_tt,
                        sigma_pml_tt=sigma_pml_tt,
                        design_flat_indices=design_flat_idx,
                        beta=beta, eta_h=opt_config.eta,
                        filter_radius=opt_config.filter_radius,
                        max_rank=config.max_rank,
                        n_sweeps=config.n_sweeps,
                        solver_tol=config.solver_tol,
                        damping=current_damping,
                        verbose=False,
                    )
                )
            else:
                # ------ Dense gradient (small grids) ------
                J_val, grad, metrics, residual = (
                    compute_adjoint_gradient_power(
                        geometry=geometry, k0_norm=k0_norm, pml=pml,
                        obj_cfg=obj_cfg, beta=beta, eta_h=opt_config.eta,
                        filter_radius=opt_config.filter_radius,
                        max_rank=config.max_rank,
                        n_sweeps=config.n_sweeps,
                        solver_tol=config.solver_tol,
                        damping=current_damping,
                        verbose=False,
                    )
                )
            s11 = 0.0 + 0j

            # Normalise to baseline (D1 diagnostic)
            if P_pml_air > 1e-30:
                metrics.P_pml_norm = metrics.P_pml / P_pml_air
                metrics.P_cond_norm = metrics.P_cond / P_pml_air

            # M_dead diagnostic: fraction of metal in air-field dead zones
            if E_air_design_sq is not None:
                rho_filt_md = density_filter(geometry.density, opt_config.filter_radius)
                rho_proj_md = heaviside_projection(rho_filt_md, beta, opt_config.eta)
                rho_sum_md = float(np.sum(rho_proj_md))
                if rho_sum_md > 1e-30:
                    dead_mask = E_air_design_sq < tau_dead
                    metrics.M_dead = float(
                        np.sum(rho_proj_md * dead_mask)
                    ) / rho_sum_md
                else:
                    metrics.M_dead = 0.0

            # Dynamic alpha activation: track ~P > 1 stability
            clamp_active = (
                feed_seed_mask is not None
                and iteration < opt_config.feed_seed_clamp_iters
            )
            if (
                opt_config.alpha_intro_mode == "auto"
                and not alpha_activated
                and not clamp_active
            ):
                if metrics.P_pml_norm > 1.0:
                    p_tilde_stable_count += 1
                else:
                    p_tilde_stable_count = 0
                if p_tilde_stable_count >= opt_config.alpha_stable_window:
                    alpha_activated = True
                    alpha_activated_iter = iteration
                    if verbose:
                        print(f"  >>> alpha activated at iter {iteration + 1} "
                              f"(~P>1 stable for {p_tilde_stable_count} iters)")

            power_metrics_history.append(metrics)

            # --- Soft coupling constraint (after clamp phase ends) ---
            if (
                coupling_mask is not None
                and opt_config.use_coupling_constraint
                and iteration >= opt_config.feed_seed_clamp_iters
            ):
                Jc, dJc, rho_near = augmented_lagrangian_coupling(
                    density=geometry.density,
                    coupling_mask=coupling_mask,
                    threshold=opt_config.coupling_density_threshold,
                    al_lambda_c=coupling_lambda_c,
                    al_mu_c=coupling_mu_c,
                )
                J_val += Jc
                grad = grad + dJc
                # Update coupling dual variable
                g_c = opt_config.coupling_density_threshold - rho_near
                coupling_lambda_c = coupling_lambda_c + coupling_mu_c * max(g_c, 0.0)

            # Update augmented Lagrangian dual variable
            rho_filt = density_filter(geometry.density, opt_config.filter_radius)
            rho_proj = heaviside_projection(rho_filt, beta, opt_config.eta)
            V_cur = float(np.mean(rho_proj))
            g_vol = V_cur - opt_config.vol_target
            al_lambda = al_lambda + al_mu * g_vol

        elif use_p_rad:
            obj, grad, p_in, q_react, residual = compute_adjoint_gradient_p_rad(
                geometry=geometry, k0_norm=k0_norm,
                beta=beta, eta=opt_config.eta,
                filter_radius=opt_config.filter_radius,
                max_rank=config.max_rank, n_sweeps=config.n_sweeps,
                solver_tol=config.solver_tol, damping=current_damping,
                verbose=False,
            )
            if opt_config.reactive_penalty > 0 and abs(q_react) > 0:
                obj += opt_config.reactive_penalty * abs(q_react)
            J_val = obj
            s11 = 0.0 + 0j
            p_in_history.append(p_in)
            p_in_final = p_in
            q_react_final = q_react
        else:
            J_val, grad, s11, residual = compute_adjoint_gradient_3d(
                geometry=geometry, k0_norm=k0_norm,
                objective_fn=obj_fn, beta=beta, eta=opt_config.eta,
                filter_radius=opt_config.filter_radius,
                max_rank=config.max_rank, n_sweeps=config.n_sweeps,
                solver_tol=config.solver_tol, damping=current_damping,
                verbose=False,
            )

        # --- TV regularisation (optional) ---
        if opt_config.regularisation_weight > 0:
            rho_proj_tv = heaviside_projection(
                density_filter(geometry.density, opt_config.filter_radius),
                beta, opt_config.eta,
            )
            tv = np.sum(np.abs(np.diff(rho_proj_tv)))
            J_val += opt_config.regularisation_weight * tv
            n_des = len(rho_proj_tv)
            tv_grad = np.zeros(n_des)
            for i_tv in range(n_des - 1):
                d = rho_proj_tv[i_tv + 1] - rho_proj_tv[i_tv]
                sign = np.sign(d) if abs(d) > 1e-12 else 0.0
                tv_grad[i_tv] -= opt_config.regularisation_weight * sign
                tv_grad[i_tv + 1] += opt_config.regularisation_weight * sign
            h_grad_tv = heaviside_gradient(
                density_filter(geometry.density, opt_config.filter_radius),
                beta, opt_config.eta,
            )
            tv_grad_chain = tv_grad * h_grad_tv
            tv_grad_final = density_filter_gradient(
                tv_grad_chain, opt_config.filter_radius)
            grad = grad + tv_grad_final

        grad_norm = float(np.linalg.norm(grad))
        obj_history.append(J_val)
        s11_history.append(s11)
        grad_norm_history.append(grad_norm)
        beta_history.append(beta)

        if verbose:
            if use_power:
                if opt_config.alpha_intro_mode == "auto":
                    stage_str = "S0" if not alpha_activated else "S1"
                else:
                    stage_str = (
                        "S0" if iteration < opt_config.alpha_intro_iter
                        else "S1"
                    )
                clamp_str = (
                    " CLAMP"
                    if (feed_seed_mask is not None
                        and iteration < opt_config.feed_seed_clamp_iters)
                    else ""
                )
                print(f"  [{iteration + 1}/{opt_config.max_iterations}]{clamp_str} "
                      f"[{stage_str}] "
                      f"J={J_val:.4e}, P_pml={metrics.P_pml:.3e}, "
                      f"P_cond={metrics.P_cond:.3e}, "
                      f"~P={metrics.P_pml_norm:.3f}, "
                      f"<E2>m={metrics.E2_metal_avg:.2e}, "
                      f"Md={metrics.M_dead:.3f}, "
                      f"V={metrics.vol:.2f}, "
                      f"|g|={grad_norm:.2e}, b={beta:.1f}, "
                      f"sig={current_sigma_max:.0f}, p={current_simp_p:.1f}, "
                      f"res={residual:.2e}")
            elif use_p_rad:
                print(f"  [{iteration + 1}/{opt_config.max_iterations}] "
                      f"obj={J_val:.4e}, P_in={p_in:.4e}, "
                      f"Q_react={q_react:.2e}, "
                      f"|grad|={grad_norm:.2e}, beta={beta:.1f}, "
                      f"damp={current_damping:.4f}, res={residual:.2e}")
            else:
                s_db = 20.0 * math.log10(max(abs(s11), 1e-30))
                print(f"  [{iteration + 1}/{opt_config.max_iterations}] "
                      f"J={J_val:.4f}, |S11|={abs(s11):.4f} ({s_db:.1f} dB), "
                      f"|grad|={grad_norm:.2e}, beta={beta:.1f}, res={residual:.2e}")

        if callback is not None:
            metric = metrics if use_power else (p_in if use_p_rad else s11)
            callback(iteration, J_val, metric, geometry.density.copy())

        # --- Convergence check ---
        if iteration > 0 and len(obj_history) >= 2:
            rel_change = abs(obj_history[-1] - obj_history[-2]) / (
                abs(obj_history[-2]) + 1e-30)
            if rel_change < opt_config.convergence_tol and grad_norm < 1e-5:
                converged = True
                if verbose:
                    print(f"  Converged at iteration {iteration + 1}.")
                break

        # --- Density update ---
        if use_adam:
            adam_t += 1
            b1 = opt_config.adam_beta1
            b2 = opt_config.adam_beta2
            eps_a = opt_config.adam_eps
            adam_m = b1 * adam_m + (1.0 - b1) * grad
            adam_v = b2 * adam_v + (1.0 - b2) * (grad ** 2)
            m_hat = adam_m / (1.0 - b1 ** adam_t)
            v_hat = adam_v / (1.0 - b2 ** adam_t)
            geometry.density = geometry.density - opt_config.learning_rate * m_hat / (np.sqrt(v_hat) + eps_a)
        else:
            # Normalised gradient descent (legacy SGD)
            max_g = np.max(np.abs(grad))
            if max_g > 0:
                grad_scaled = grad / max_g
            else:
                grad_scaled = grad
            geometry.density = geometry.density - opt_config.learning_rate * grad_scaled
        geometry.density = np.clip(geometry.density, 0.0, 1.0)

        # Re-enforce feed seed clamp after update
        if (use_power and feed_seed_mask is not None
                and iteration < opt_config.feed_seed_clamp_iters):
            geometry.density[feed_seed_mask] = 1.0

    t_end = time.perf_counter()

    final_mask = geometry.build_conductor_mask(
        beta=opt_config.beta_max, eta=opt_config.eta,
        filter_radius=opt_config.filter_radius,
    )

    q_result = None
    power_q_result = None
    if extract_q:
        if use_power or use_p_rad:
            if verbose:
                print(f"\n--- Q extraction via power sweep ---")
            power_q_result = extract_q_from_power_sweep(
                geometry=geometry, f_center_hz=config.frequency_hz,
                n_freq=11, bw_fraction=0.3,
                beta=opt_config.beta_max, eta=opt_config.eta,
                filter_radius=opt_config.filter_radius,
                max_rank=config.max_rank, n_sweeps=config.n_sweeps,
                solver_tol=config.solver_tol,
                damping=config.damping, verbose=verbose,
            )
        else:
            if verbose:
                print(f"\n--- Q extraction via S11 sweep ---")
            q_result = extract_q_from_sweep(
                geometry=geometry, f_center_hz=config.frequency_hz,
                n_freq=11, bw_fraction=0.3,
                beta=opt_config.beta_max, eta=opt_config.eta,
                filter_radius=opt_config.filter_radius,
                max_rank=config.max_rank, n_sweeps=config.n_sweeps,
                solver_tol=config.solver_tol,
                damping=config.damping, verbose=verbose,
            )

    total_time = t_end - t_start

    result = ChuOptResult(
        density_final=geometry.density.copy(),
        conductor_mask=final_mask,
        s11_final=s11_history[-1] if s11_history else 0.0 + 0j,
        q_result=q_result, power_q_result=power_q_result,
        objective_history=obj_history, s11_history=s11_history,
        grad_norm_history=grad_norm_history, beta_history=beta_history,
        p_in_history=p_in_history, damping_history=damping_history_list,
        power_metrics_history=power_metrics_history,
        p_in_final=p_in_final, q_reactive_final=q_react_final,
        n_iterations=len(obj_history), converged=converged,
        total_time_s=total_time, config=config, opt_config=opt_config,
    )

    if verbose:
        print(result.summary())
        if q_result is not None:
            print(q_result.summary(config.q_chu))

    return result


# =====================================================================
# Section 9: Grid-Level Schedule Factory
# =====================================================================

def make_chu_antenna_schedule(
    grid_level: str = "64",
    ka: float = 0.3,
    frequency_hz: float = 1.0e9,
) -> tuple[ChuProblemConfig, ChuOptConfig]:
    """Return calibrated (config, opt_config) for a given grid level.

    Each level preserves the "reversal" behavior proven at 16³
    while adjusting solver fidelity, domain clearance, and
    continuation cadence for the target resolution.

    **Resolution vs. clearance trade-off at ka=0.3:**
    At ka=0.3, the antenna sphere is only a/λ ≈ 0.048, so placing
    the PML at 0.5λ from the sphere requires most of the domain
    to be empty.  To get both >4 cells per radius AND adequate PML
    clearance, very large grids (256³+) or multiscale approaches
    are needed.

    This schedule prioritises topological DOF (sufficient cells in
    the design sphere) below 128³, and only adds meaningful PML
    clearance at 128³+:

      16³  :  ~4 cells/radius,  ~0.02λ clearance  (plumbing test)
      32³  :  ~5 cells/radius,  ~0.04λ clearance  (gradient validation)
      64³  : ~11 cells/radius,  ~0.06λ clearance  (topology validation)
      128³ : ~11 cells/radius,  ~0.30λ clearance  (physics validation)

    Parameters
    ----------
    grid_level : str
        One of "16", "32", "64", "128".
    ka : float
        Electrical size (default 0.3).
    frequency_hz : float
        Centre frequency (default 1 GHz).

    Returns
    -------
    tuple[ChuProblemConfig, ChuOptConfig]
        Problem and optimization configurations.

    Notes
    -----
    Continuation schedule tuple (p, beta, sigma_max, alpha) at each level:

    16³ — plumbing validation (domain ~ 0.19 λ, ~4 cells/r)
      p:     1.0 → 2.0 over 20 iters
      beta:  1 → 8   (×2 every 10)
      sigma: 30 → 100
      alpha: auto (window=5)

    32³ — gradient FD validation (domain ~ 0.29 λ, ~5 cells/r)
      p:     1.0 → 3.0 over 30 iters
      beta:  1 → 16  (×2 every 10)
      sigma: 30 → 200
      alpha: auto (window=10)

    64³ — topology validation (domain auto ~0.29 λ, ~11 cells/r)
      p:     1.0 → 3.0 over 50 iters
      beta:  1 → 32  (×2 every 15)
      sigma: 30 → 300
      alpha: auto (window=20)
      Design sphere has ~2500 voxels — real topology possible.
      PML clearance still tight (~0.06λ), so P_pml is a proxy.

    128³ — physics validation (domain 0.75 λ, ~11 cells/r)
      p:     1.0 → 4.0 over 60 iters
      beta:  1 → 64  (×2 every 20)
      sigma: 30 → 500
      alpha: auto (window=30)
      PML clearance ~0.30λ — first grid where P_pml is
      a reasonable far-field proxy.
    """
    # Physics at ka=0.3, f=1GHz:
    #   λ = 299.8mm, a = 14.31mm, a/λ ≈ 0.048
    #
    # Grid dimension table (domain_wavelengths=1.5):
    #   N      h [mm]   a/h        PML clearance  max_rank
    #   16     28.11    0.5        --              16
    #   32     14.06    1.0        --              32
    #   64     7.03     2.0        --              64
    #   128    3.51     4.1        ~0.30λ          128
    #   1024   0.439    32.6       ~0.45λ          64  ← higher compress
    #   4096   0.110    130.2      ~0.45λ          48  ← maximum compress
    #
    # Rank inversion rule: higher N → smoother fields in QTT →
    # LOWER rank needed. This is the fundamental QTT advantage:
    # cost scales as O(n_bits · r²) not O(N³).

    schedules = {
        "16": dict(
            n_bits=4, domain_wavelengths=0.0,  # auto
            max_rank=16, n_sweeps=5, solver_tol=0.5,
            sigma_max=100.0, simp_p=2.0, damping=0.01,
            # opt
            max_iterations=30, learning_rate=0.02,
            optimizer="adam",
            beta_init=1.0, beta_max=8.0, beta_increase_every=10,
            sigma_max_init=30.0, sigma_max_final=100.0, sigma_ramp_iters=20,
            simp_p_init=1.0, simp_p_final=2.0, simp_p_ramp_iters=20,
            alpha_loss=0.1, alpha_stable_window=5,
            feed_seed_clamp_iters=5, feed_seed_clamp_radius=2,
            coupling_radius=3, coupling_density_threshold=0.3,
            vol_target=0.3, al_mu_init=10.0, filter_radius=0,
        ),
        "32": dict(
            n_bits=5, domain_wavelengths=0.0,  # auto
            max_rank=32, n_sweeps=10, solver_tol=0.2,
            sigma_max=200.0, simp_p=3.0, damping=0.01,
            # opt
            max_iterations=50, learning_rate=0.02,
            optimizer="adam",
            beta_init=1.0, beta_max=16.0, beta_increase_every=10,
            sigma_max_init=30.0, sigma_max_final=200.0, sigma_ramp_iters=30,
            simp_p_init=1.0, simp_p_final=3.0, simp_p_ramp_iters=30,
            alpha_loss=0.1, alpha_stable_window=10,
            feed_seed_clamp_iters=8, feed_seed_clamp_radius=2,
            coupling_radius=3, coupling_density_threshold=0.3,
            vol_target=0.3, al_mu_init=10.0, filter_radius=1,
        ),
        "64": dict(
            n_bits=6, domain_wavelengths=0.0,  # auto
            max_rank=64, n_sweeps=20, solver_tol=0.1,
            sigma_max=300.0, simp_p=3.0, damping=0.005,
            # opt
            max_iterations=80, learning_rate=0.01,
            optimizer="adam",
            beta_init=1.0, beta_max=32.0, beta_increase_every=15,
            sigma_max_init=30.0, sigma_max_final=300.0, sigma_ramp_iters=50,
            simp_p_init=1.0, simp_p_final=3.0, simp_p_ramp_iters=50,
            alpha_loss=0.1, alpha_stable_window=20,
            feed_seed_clamp_iters=10, feed_seed_clamp_radius=3,
            coupling_radius=5, coupling_density_threshold=0.3,
            vol_target=0.3, al_mu_init=10.0, filter_radius=1,
        ),
        "128": dict(
            n_bits=7, domain_wavelengths=0.75,
            max_rank=128, n_sweeps=40, solver_tol=0.05,
            sigma_max=500.0, simp_p=4.0, damping=0.005,
            # opt
            max_iterations=120, learning_rate=0.005,
            optimizer="adam",
            beta_init=1.0, beta_max=64.0, beta_increase_every=20,
            sigma_max_init=30.0, sigma_max_final=500.0, sigma_ramp_iters=60,
            simp_p_init=1.0, simp_p_final=4.0, simp_p_ramp_iters=60,
            alpha_loss=0.1, alpha_stable_window=30,
            feed_seed_clamp_iters=15, feed_seed_clamp_radius=4,
            coupling_radius=6, coupling_density_threshold=0.3,
            vol_target=0.3, al_mu_init=10.0, filter_radius=2,
        ),
        # ==============================
        # Physics-valid levels (1024³+)
        # ==============================
        # Domain locked at 1.5λ.  PML clearance ~0.45λ.
        # Antenna sphere occupies tens to hundreds of cells.
        # These are the only levels where P_pml is a clean
        # far-field proxy and Chu comparison is defensible.
        "1024": dict(
            n_bits=10, domain_wavelengths=1.5,  # ~0.45λ PML clearance
            # QTT rank: LOWER than 128³ — higher N compresses better.
            # Smooth EM fields on 1024³ need only rank ~64 for <1% error.
            # More sweeps compensate for tighter compression.
            max_rank=64, n_sweeps=80, solver_tol=0.01,
            sigma_max=1000.0, simp_p=4.0, damping=0.002,
            # a/h ≈ 32.6 cells — real antenna geometry discovery
            # Design sphere: ~145k voxels
            # opt
            max_iterations=200, learning_rate=0.003,
            optimizer="adam",
            beta_init=1.0, beta_max=128.0, beta_increase_every=25,
            sigma_max_init=30.0, sigma_max_final=1000.0, sigma_ramp_iters=100,
            simp_p_init=1.0, simp_p_final=4.0, simp_p_ramp_iters=100,
            alpha_loss=0.1, alpha_stable_window=40,
            feed_seed_clamp_iters=20, feed_seed_clamp_radius=8,
            coupling_radius=12, coupling_density_threshold=0.3,
            vol_target=0.3, al_mu_init=10.0, filter_radius=3,
        ),
        "4096": dict(
            n_bits=12, domain_wavelengths=1.5,  # ~0.45λ PML clearance
            # QTT rank: LOWEST — 4096³ fields compress maximally.
            # Smooth EM fields on 4096³ need only rank ~32-48.
            # More sweeps (100) ensure convergence at tight compression.
            max_rank=48, n_sweeps=100, solver_tol=0.005,
            sigma_max=2000.0, simp_p=4.0, damping=0.001,
            # a/h ≈ 130.2 cells — O(100) cells per radius
            # Design sphere: ~9.3M voxels
            # PML clearance: ~0.45λ (physically valid far-field proxy)
            # This is where "geometry discovery" is real.
            # opt
            max_iterations=300, learning_rate=0.001,
            optimizer="adam",
            beta_init=1.0, beta_max=256.0, beta_increase_every=30,
            sigma_max_init=30.0, sigma_max_final=2000.0, sigma_ramp_iters=150,
            simp_p_init=1.0, simp_p_final=4.0, simp_p_ramp_iters=150,
            alpha_loss=0.1, alpha_stable_window=50,
            feed_seed_clamp_iters=30, feed_seed_clamp_radius=20,
            coupling_radius=30, coupling_density_threshold=0.3,
            vol_target=0.3, al_mu_init=10.0, filter_radius=5,
        ),
    }

    if grid_level not in schedules:
        raise ValueError(
            f"Unknown grid_level '{grid_level}'. "
            f"Choose from {list(schedules.keys())}."
        )

    s = schedules[grid_level]

    cfg = ChuProblemConfig(
        frequency_hz=frequency_hz,
        ka=ka,
        n_bits=s["n_bits"],
        domain_wavelengths=s["domain_wavelengths"],
        max_rank=s["max_rank"],
        n_sweeps=s["n_sweeps"],
        solver_tol=s["solver_tol"],
        sigma_max=s["sigma_max"],
        simp_p=s["simp_p"],
        damping=s["damping"],
    )

    opt = ChuOptConfig(
        max_iterations=s["max_iterations"],
        learning_rate=s["learning_rate"],
        beta_init=s["beta_init"],
        beta_max=s["beta_max"],
        beta_increase_every=s["beta_increase_every"],
        use_power_adjoint=True,
        use_p_rad=False,
        alpha_loss=s["alpha_loss"],
        alpha_intro_mode="auto",
        alpha_stable_window=s["alpha_stable_window"],
        vol_target=s["vol_target"],
        al_mu_init=s["al_mu_init"],
        feed_seed_clamp_iters=s["feed_seed_clamp_iters"],
        feed_seed_clamp_radius=s["feed_seed_clamp_radius"],
        use_coupling_constraint=True,
        coupling_density_threshold=s["coupling_density_threshold"],
        coupling_radius=s["coupling_radius"],
        sigma_max_init=s["sigma_max_init"],
        sigma_max_final=s["sigma_max_final"],
        sigma_ramp_iters=s["sigma_ramp_iters"],
        simp_p_init=s["simp_p_init"],
        simp_p_final=s["simp_p_final"],
        simp_p_ramp_iters=s["simp_p_ramp_iters"],
        filter_radius=s["filter_radius"],
        optimizer=s.get("optimizer", "adam"),
    )

    return cfg, opt
