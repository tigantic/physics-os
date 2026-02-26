"""QTT Boundary Conditions for Frequency-Domain Maxwell/Helmholtz.

Phase 3 of the QTT Frequency-Domain Maxwell program.

Implements:
  1. **PML (Perfectly Matched Layer)** — complex coordinate stretching
     that maps to a complex permittivity profile in QTT format.
     Polynomial grading (configurable order 1–4) ensures smooth
     impedance match at the PML interface.

  2. **PEC (Perfect Electric Conductor)** — enforces E = 0 at
     conductor locations via a diagonal mask MPO.  The conductor
     geometry is specified either by a boolean mask array or by
     geometric primitives (slabs, cylinders, patches).

  3. **Material Geometry Builder** — constructs spatially varying
     ε_r(x) profiles from layered stacks, dielectric objects, and
     waveguide structures.  Outputs a complex QTT representation
     that encodes both material properties and PML absorption.

All profiles are returned as QTT cores suitable for direct assembly
into the Helmholtz MPO via ``diag_mpo_from_tt → mpo_add_c``.

Dependencies
------------
- ``tensornet.em.qtt_helmholtz``: ``array_to_tt``, ``diag_mpo_from_tt``
- ``tensornet.vm.operators``: ``identity_mpo``
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from tensornet.em.qtt_helmholtz import (
    array_to_tt, diag_mpo_from_tt, mpo_add_c, mpo_scale_c,
)
from tensornet.vm.operators import (
    _shift_left_mpo, _shift_right_mpo, identity_mpo,
)


# =====================================================================
# Section 1: PML Configuration & Construction
# =====================================================================

@dataclass
class PMLConfig:
    """Configuration for Perfectly Matched Layer.

    The PML implements complex coordinate stretching:
      s(x) = 1 + σ_max · (d(x)/L_pml)^p · (1 - j·κ_ratio)

    where d(x) is distance into the PML layer, L_pml is the PML
    thickness, p is the polynomial grading order, and κ_ratio
    controls the real stretching component.

    The effective complex permittivity in the PML region is:
      ε_pml(x) = ε_r(x) · s(x)

    Parameters
    ----------
    n_cells : int
        Number of PML cells on each boundary.
    sigma_max : float
        Maximum PML conductivity (imaginary part of stretching).
        Larger values absorb more aggressively but cause more
        reflection at the PML interface if grading is insufficient.
    poly_order : int
        Polynomial grading order (1=linear, 2=quadratic, 3=cubic).
        Higher orders give smoother impedance transition but need
        more PML cells.  Default 3 is the standard choice.
    kappa_max : float
        Maximum real stretching factor.  Values > 1 compress
        evanescent waves in the PML.  Set to 1.0 (default) for
        purely absorbing PML; increase to ~5 for evanescent modes.
    damping : float
        Global imaginary shift added to ε everywhere.  Regularises
        the indefinite Helmholtz operator by ensuring no eigenvalue
        lies exactly on the real axis.  A value of 0.01 is usually
        sufficient.
    """

    n_cells: int = 20
    sigma_max: float = 10.0
    poly_order: int = 3
    kappa_max: float = 1.0
    damping: float = 0.01

    def optimal_sigma_max(self, k: float, h: float, target_R: float = 1e-6) -> float:
        """Compute optimal σ_max for a target reflection coefficient.

        From Berenger's analysis:
          R(θ) ≈ exp(-2 · σ_max · L_pml · cos(θ) / (p + 1))

        For normal incidence (θ=0):
          σ_max = -(p + 1) · ln(R) / (2 · L_pml)

        Parameters
        ----------
        k : float
            Wavenumber.
        h : float
            Grid spacing.
        target_R : float
            Target reflection coefficient (field amplitude).

        Returns
        -------
        float
            Optimal σ_max.
        """
        L_pml = self.n_cells * h
        p = self.poly_order
        return -(p + 1) * math.log(target_R) / (2.0 * L_pml)

    @classmethod
    def for_problem(
        cls,
        n_bits: int,
        k: float,
        n_cells: int = 30,
        target_R_dB: float = -60.0,
        poly_order: int = 3,
        kappa_max: float = 1.0,
        damping: float = 0.01,
    ) -> "PMLConfig":
        """Create PML auto-configured for specific problem parameters.

        Computes the optimal σ_max so that the theoretical normal-
        incidence reflection equals ``target_R_dB`` (in dB, field).

        Parameters
        ----------
        n_bits : int
            QTT resolution (N = 2^n_bits grid points).
        k : float
            Wavenumber (2πf/c).
        n_cells : int
            Number of PML cells on each boundary.
        target_R_dB : float
            Target reflection in dB (e.g. -60.0 for 10^{-3} field).
        poly_order : int
            Polynomial grading order (default 3 = cubic).
        kappa_max : float
            Maximum real stretching factor.
        damping : float
            Global damping regularisation.

        Returns
        -------
        PMLConfig
            Properly configured PML for the given problem.
        """
        N = 2 ** n_bits
        h = 1.0 / N
        L_pml = n_cells * h
        target_R = 10.0 ** (target_R_dB / 20.0)
        sigma = -(poly_order + 1) * math.log(max(target_R, 1e-30)) / (2.0 * L_pml)
        return cls(
            n_cells=n_cells,
            sigma_max=sigma,
            poly_order=poly_order,
            kappa_max=kappa_max,
            damping=damping,
        )


def build_pml_profile_1d(
    n_bits: int,
    pml: PMLConfig,
    k: float = 1.0,
    eps_r: NDArray | float = 1.0,
    max_rank: int = 64,
    cutoff: float = 1e-12,
) -> list[NDArray]:
    """Build 1D permittivity profile as QTT cores (damping only).

    The PML absorption is handled by the stretched-coordinate
    Laplacian (see ``stretched_laplacian_mpo_1d``).  This function
    applies only the global damping regularisation to ε_r.

    Parameters
    ----------
    n_bits : int
        Number of QTT sites (grid has N = 2^n_bits points).
    pml : PMLConfig
        PML configuration (only ``damping`` is used here).
    k : float
        Wavenumber (retained for API compatibility).
    eps_r : NDArray or float
        Base relative permittivity profile.
    max_rank : int
        Maximum QTT rank for compression.
    cutoff : float
        SVD truncation cutoff.

    Returns
    -------
    list[NDArray]
        Complex QTT cores for ε(x) with damping.
    """
    N = 2 ** n_bits

    # Base permittivity
    if isinstance(eps_r, (int, float)):
        eps = np.full(N, complex(eps_r), dtype=np.complex128)
    else:
        eps = np.asarray(eps_r, dtype=np.complex128).copy()

    # Global damping regularisation
    if pml.damping > 0:
        eps *= (1.0 + 1j * pml.damping)

    return array_to_tt(eps, max_rank=max_rank, cutoff=cutoff)


# =====================================================================
# Section 1b: UPML Stretched-Coordinate Laplacian
# =====================================================================

def _compute_pml_stretching(
    N: int,
    k: float,
    pml: PMLConfig,
) -> NDArray:
    """Compute the UPML complex stretching factor s(x).

    For exp(-jωt) convention:
      s(x) = κ(x) + j·σ(x)/ω

    where ω = k (normalised units, c = 1).  The +j sign ensures that
    outgoing waves decay in both left and right PML regions.

    Parameters
    ----------
    N : int
        Number of grid points.
    k : float
        Wavenumber / angular frequency.
    pml : PMLConfig
        PML parameters.

    Returns
    -------
    NDArray
        Complex array of shape (N,) with s(x) values.
    """
    s = np.ones(N, dtype=np.complex128)
    n_pml = pml.n_cells
    p = pml.poly_order
    omega = k

    for i in range(N):
        if i < n_pml:
            depth = (n_pml - i) / n_pml
            sigma = pml.sigma_max * depth ** p
            kappa = 1.0 + (pml.kappa_max - 1.0) * depth ** p
            s[i] = kappa + 1j * sigma / omega
        elif i >= N - n_pml:
            depth = (i - (N - n_pml - 1)) / n_pml
            sigma = pml.sigma_max * depth ** p
            kappa = 1.0 + (pml.kappa_max - 1.0) * depth ** p
            s[i] = kappa + 1j * sigma / omega

    return s


def diag_tt_times_mpo(
    diag_tt: list[NDArray],
    mpo: list[NDArray],
) -> list[NDArray]:
    """Multiply a diagonal MPO by a general MPO: diag(v) · M.

    Given a QTT vector v (representing the diagonal) and an MPO M,
    computes the MPO product D·M where D = diag(v).  The result
    core at site k is:

      P_k[(α,α'), i, j, (β,β')] = v_k[α,i,β] · M_k[α',i,j,β']

    Bond dimension of the result is rank(v) × rank(M).

    Parameters
    ----------
    diag_tt : list[NDArray]
        QTT cores of the diagonal vector v, shape (r1, 2, r2).
    mpo : list[NDArray]
        MPO cores M, shape (s1, 2, 2, s2).

    Returns
    -------
    list[NDArray]
        MPO cores of diag(v) · M, shape (r1*s1, 2, 2, r2*s2).
    """
    n = len(diag_tt)
    if len(mpo) != n:
        raise ValueError(
            f"diag_tt has {n} cores but mpo has {len(mpo)} cores"
        )

    result: list[NDArray] = []
    for v_k, M_k in zip(diag_tt, mpo):
        v_k = np.asarray(v_k, dtype=np.complex128)
        M_k = np.asarray(M_k, dtype=np.complex128)
        r1, d1, r2 = v_k.shape
        s1, d_in, d_out, s2 = M_k.shape
        # P[(a,a'), i, j, (b,b')] = v[a,i,b] * M[a',i,j,b']
        P = np.einsum("aib,cijd->acijbd", v_k, M_k)
        P = P.reshape(r1 * s1, d_in, d_out, r2 * s2)
        result.append(P)

    return result


def _dirichlet_correction_mpo(
    n_bits: int,
    alpha: complex,
    beta: complex,
) -> list[NDArray]:
    """Build rank-2 MPO for Dirichlet boundary correction.

    Produces the operator:
      Δ = α·|0⟩⟨N-1| + β·|N-1⟩⟨0|

    which cancels the periodic wrap-around terms in the shift MPOs.
    Each outer product is rank-1; their sum is rank-2.

    Parameters
    ----------
    n_bits : int
        Number of QTT sites (N = 2^n_bits).
    alpha : complex
        Coefficient for |0⟩⟨N-1| (cancels S₋ wrap: row 0, col N-1).
    beta : complex
        Coefficient for |N-1⟩⟨0| (cancels S₊ wrap: row N-1, col 0).

    Returns
    -------
    list[NDArray]
        MPO cores for the correction operator.
    """
    cores: list[NDArray] = []
    for site in range(n_bits):
        if site == 0:
            # First core: shape (1, 2, 2, 2) — two bond channels
            core = np.zeros((1, 2, 2, 2), dtype=np.complex128)
            # Channel 0: α · |0⟩⟨1| (MSB out=0, in=1)
            core[0, 0, 1, 0] = alpha
            # Channel 1: β · |1⟩⟨0| (MSB out=1, in=0)
            core[0, 1, 0, 1] = beta
        elif site == n_bits - 1:
            # Last core: shape (2, 2, 2, 1)
            core = np.zeros((2, 2, 2, 1), dtype=np.complex128)
            # Channel 0 from α: |0⟩⟨1| at LSB (out=0, in=1)
            core[0, 0, 1, 0] = 1.0
            # Channel 1 from β: |1⟩⟨0| at LSB (out=1, in=0)
            core[1, 1, 0, 0] = 1.0
        else:
            # Middle: shape (2, 2, 2, 2), propagate channel
            core = np.zeros((2, 2, 2, 2), dtype=np.complex128)
            # Channel 0 (α path): |0⟩⟨1| at each bit
            core[0, 0, 1, 0] = 1.0
            # Channel 1 (β path): |1⟩⟨0| at each bit
            core[1, 1, 0, 1] = 1.0
        cores.append(core)
    return cores


def stretched_laplacian_mpo_1d(
    n_bits: int,
    k: float,
    h: float,
    pml: PMLConfig,
    max_rank: int = 64,
    cutoff: float = 1e-12,
) -> list[NDArray]:
    r"""Build 1D UPML stretched-coordinate Laplacian as QTT MPO.

    Discretises the operator

    .. math:: \frac{1}{s}\frac{\mathrm{d}}{\mathrm{d}x}
              \left[\frac{1}{s}\frac{\mathrm{d}E}
              {\mathrm{d}x}\right]

    on a uniform grid with spacing *h* and Dirichlet BCs at both
    ends.  The complex stretching factor for exp(-jωt) convention is

    .. math:: s(x) = \kappa(x) + j\,\sigma(x)/\omega

    which ensures outgoing waves decay in the PML regions.

    The discretised operator at grid point *i* has the tridiagonal
    entries:

      L_s[i, i-1] = b_i / h²,   b_i = 1/(s_i · s_{i-½})
      L_s[i, i  ] = d_i / h²,   d_i = -(a_i + b_i)
      L_s[i, i+1] = a_i / h²,   a_i = 1/(s_i · s_{i+½})

    In the interior (no PML), s = 1 and this reduces to the standard
    second-order central-difference Laplacian.

    Implementation builds:
      L_s = (1/h²)·(D_b·S₋ + D_a·S₊ + D_d·I) + Dirichlet correction

    where S₊, S₋ are periodic shift MPOs (bond dim 2) and D_x are
    diagonal MPOs from QTT cores of the coefficient vectors.

    Parameters
    ----------
    n_bits : int
        Number of QTT sites (N = 2^n_bits grid points).
    k : float
        Wavenumber (ω = k in normalised units).
    h : float
        Grid spacing.
    pml : PMLConfig
        PML parameters (n_cells, sigma_max, poly_order, kappa_max).
    max_rank : int
        Maximum QTT rank for coefficient vectors.
    cutoff : float
        SVD truncation cutoff.

    Returns
    -------
    list[NDArray]
        Complex MPO cores for the stretched Laplacian.
    """
    N = 2 ** n_bits

    # --- Stretching factor s(x) at grid points ---
    s_grid = _compute_pml_stretching(N, k, pml)

    # --- Half-point stretching (averaged) ---
    s_half = np.empty(N + 1, dtype=np.complex128)
    s_half[0] = s_grid[0]
    s_half[N] = s_grid[N - 1]
    for i in range(N - 1):
        s_half[i + 1] = 0.5 * (s_grid[i] + s_grid[i + 1])

    # --- Tridiagonal coefficients ---
    #   a_i = 1/(s_i · s_{i+1/2})   (super-diagonal)
    #   b_i = 1/(s_i · s_{i-1/2})   (sub-diagonal)
    #   d_i = -(a_i + b_i)          (main diagonal)
    a_vec = np.empty(N, dtype=np.complex128)
    b_vec = np.empty(N, dtype=np.complex128)
    for i in range(N):
        a_vec[i] = 1.0 / (s_grid[i] * s_half[i + 1])
        b_vec[i] = 1.0 / (s_grid[i] * s_half[i])
    d_vec = -(a_vec + b_vec)

    # --- QTT decomposition of coefficient vectors ---
    a_tt = array_to_tt(a_vec, max_rank=max_rank, cutoff=cutoff)
    b_tt = array_to_tt(b_vec, max_rank=max_rank, cutoff=cutoff)
    d_tt = array_to_tt(d_vec, max_rank=max_rank, cutoff=cutoff)

    # --- Shift and identity MPOs (periodic, bond dim ≤ 2) ---
    S_plus = [c.astype(np.complex128) for c in _shift_right_mpo(n_bits)]
    S_minus = [c.astype(np.complex128) for c in _shift_left_mpo(n_bits)]
    I_mpo = [c.astype(np.complex128) for c in identity_mpo(n_bits)]

    # --- Weighted shift MPOs: D_a·S₊, D_b·S₋, D_d·I ---
    Da_Sp = diag_tt_times_mpo(a_tt, S_plus)
    Db_Sm = diag_tt_times_mpo(b_tt, S_minus)
    Dd_I = diag_tt_times_mpo(d_tt, I_mpo)

    # --- Sum: L_s = (1/h²)(D_b·S₋ + D_a·S₊ + D_d·I) ---
    L_s = mpo_add_c(Db_Sm, Da_Sp)
    L_s = mpo_add_c(L_s, Dd_I)
    inv_h2 = 1.0 / (h * h)
    L_s = mpo_scale_c(L_s, inv_h2)

    # --- Dirichlet correction (cancel periodic wrap-around) ---
    # Periodic S₋ has S₋[0, N-1] = 1 → contributes b_0/h² at [0, N-1]
    # Periodic S₊ has S₊[N-1, 0] = 1 → contributes a_{N-1}/h² at [N-1, 0]
    # Subtract these to enforce Dirichlet (E=0 at boundaries).
    alpha_corr = -b_vec[0] * inv_h2     # cancel [0, N-1] entry
    beta_corr = -a_vec[N - 1] * inv_h2  # cancel [N-1, 0] entry
    corr = _dirichlet_correction_mpo(n_bits, alpha_corr, beta_corr)
    L_s = mpo_add_c(L_s, corr)

    return L_s


# =====================================================================
# Section 2: PEC Boundary Conditions
# =====================================================================

def pec_mask_1d(
    n_bits: int,
    conductor_mask: NDArray,
    max_rank: int = 64,
    cutoff: float = 1e-12,
) -> list[NDArray]:
    """Build a diagonal mask MPO that enforces E = 0 at PEC locations.

    The mask m(x) = 0 where conductor, 1 elsewhere.  Applying
    ``diag(m) · E`` zeros out the field at conductor locations.

    For Helmholtz: the operator is modified to
      H_pec = diag(m) · H · diag(m) + diag(1-m) · (1/h²)·I

    The second term pins E = 0 at conductor points (acts like a
    very large diagonal penalty).

    Parameters
    ----------
    n_bits : int
        Number of QTT sites.
    conductor_mask : NDArray
        Boolean array of length N = 2^n_bits.  True = conductor.
    max_rank : int
        Maximum QTT rank.
    cutoff : float
        SVD cutoff.

    Returns
    -------
    list[NDArray]
        QTT cores for the mask m(x) (0 at conductor, 1 elsewhere).
    """
    N = 2 ** n_bits
    if len(conductor_mask) != N:
        raise ValueError(
            f"Conductor mask length {len(conductor_mask)} != grid size {N}"
        )
    mask = np.where(conductor_mask, 0.0, 1.0).astype(np.complex128)
    return array_to_tt(mask, max_rank=max_rank, cutoff=cutoff)


def build_pec_penalty_mpo(
    n_bits: int,
    conductor_mask: NDArray,
    penalty: float = 1e8,
    max_rank: int = 64,
    cutoff: float = 1e-12,
) -> list[NDArray]:
    """Build penalty MPO that enforces E = 0 at PEC locations.

    Adds a large diagonal penalty at conductor grid points:
      P = penalty · diag(1_{conductor})

    When added to the Helmholtz operator H, this drives the solution
    toward E = 0 at conductor locations via
      (H + P) E = b

    This is the standard volumetric penalty approach, numerically
    more stable than projecting out conductor DOFs.

    Parameters
    ----------
    n_bits : int
        Number of QTT sites.
    conductor_mask : NDArray
        Boolean array, True = conductor (PEC).
    penalty : float
        Diagonal penalty magnitude.  Must be >> k² for effective
        enforcement.
    max_rank : int
        Maximum QTT rank.
    cutoff : float
        SVD cutoff.

    Returns
    -------
    list[NDArray]
        MPO cores for the penalty operator.
    """
    N = 2 ** n_bits
    if len(conductor_mask) != N:
        raise ValueError(
            f"Conductor mask length {len(conductor_mask)} != grid size {N}"
        )

    # Penalty function: large value at conductor, zero elsewhere
    penalty_arr = np.where(conductor_mask, penalty, 0.0).astype(np.complex128)
    penalty_tt = array_to_tt(penalty_arr, max_rank=max_rank, cutoff=cutoff)
    return diag_mpo_from_tt(penalty_tt)


# =====================================================================
# Section 3: Geometry / Material Builder
# =====================================================================

@dataclass
class MaterialRegion:
    """A region of space with specified permittivity.

    Parameters
    ----------
    eps_r : complex
        Relative permittivity (real or complex for lossy materials).
    x_start : float
        Start of region in normalised coordinates [0, 1].
    x_end : float
        End of region in normalised coordinates [0, 1].
    label : str
        Human-readable label for the material.
    """

    eps_r: complex
    x_start: float
    x_end: float
    label: str = ""

    def contains(self, x: NDArray) -> NDArray:
        """Boolean mask: True where x is inside this region."""
        return (x >= self.x_start) & (x < self.x_end)


@dataclass
class Geometry1D:
    """1D geometry specification for EM simulation.

    Builds a spatially varying ε_r(x) profile from a list of material
    regions, then wraps it with PML absorbing boundaries.

    Parameters
    ----------
    n_bits : int
        QTT resolution (N = 2^n_bits grid points).
    background_eps : complex
        Background (default) permittivity.
    regions : list[MaterialRegion]
        Material regions (later regions overwrite earlier ones).
    pml : PMLConfig
        PML configuration for absorbing boundaries.
    conductors : list[tuple[float, float]]
        PEC conductor regions as (x_start, x_end) pairs.
    """

    n_bits: int = 14
    background_eps: complex = 1.0 + 0j
    regions: list[MaterialRegion] = field(default_factory=list)
    pml: PMLConfig = field(default_factory=PMLConfig)
    conductors: list[tuple[float, float]] = field(default_factory=list)

    def add_dielectric_slab(
        self, eps_r: complex, x_start: float, x_end: float, label: str = ""
    ) -> None:
        """Add a dielectric slab to the geometry."""
        self.regions.append(
            MaterialRegion(eps_r=eps_r, x_start=x_start, x_end=x_end, label=label)
        )

    def add_conductor(self, x_start: float, x_end: float) -> None:
        """Add a PEC conductor region."""
        self.conductors.append((x_start, x_end))

    def build_eps_profile(self) -> NDArray:
        """Build the dense ε_r(x) array (without PML).

        Returns
        -------
        NDArray
            Complex permittivity array of length N = 2^n_bits.
        """
        N = 2 ** self.n_bits
        x = np.linspace(0.5 / N, 1.0 - 0.5 / N, N)
        eps = np.full(N, complex(self.background_eps), dtype=np.complex128)

        # Apply material regions (later overwrites earlier)
        for region in self.regions:
            mask = region.contains(x)
            eps[mask] = region.eps_r

        return eps

    def build_conductor_mask(self) -> NDArray:
        """Build boolean conductor mask.

        Returns
        -------
        NDArray
            Boolean array of length N, True at conductor locations.
        """
        N = 2 ** self.n_bits
        x = np.linspace(0.5 / N, 1.0 - 0.5 / N, N)
        mask = np.zeros(N, dtype=bool)

        for x_start, x_end in self.conductors:
            mask |= (x >= x_start) & (x < x_end)

        return mask

    def build_eps_qtt(
        self, k: float = 1.0, max_rank: int = 64, cutoff: float = 1e-12
    ) -> list[NDArray]:
        """Build complete ε(x) as QTT cores (materials + damping).

        The PML absorption is handled by the stretched Laplacian
        (``stretched_laplacian_mpo_1d``), NOT by modifying ε here.
        This function only applies global damping regularisation.

        Parameters
        ----------
        k : float
            Wavenumber (retained for API compatibility).
        max_rank : int
            Maximum QTT rank for compression.
        cutoff : float
            SVD truncation cutoff.

        Returns
        -------
        list[NDArray]
            Complex QTT cores for the permittivity with damping.
        """
        eps_arr = self.build_eps_profile()
        return build_pml_profile_1d(
            self.n_bits, self.pml,
            k=k,
            eps_r=eps_arr,
            max_rank=max_rank,
            cutoff=cutoff,
        )

    def has_conductors(self) -> bool:
        """Check if geometry has any PEC conductors."""
        return len(self.conductors) > 0

    def build_penalty_mpo(
        self, penalty: float = 1e8, max_rank: int = 64
    ) -> list[NDArray]:
        """Build PEC penalty MPO for this geometry.

        Returns
        -------
        list[NDArray]
            MPO cores for the PEC penalty term.

        Raises
        ------
        ValueError
            If geometry has no conductors.
        """
        if not self.has_conductors():
            raise ValueError("No conductors in geometry")
        mask = self.build_conductor_mask()
        return build_pec_penalty_mpo(
            self.n_bits, mask, penalty=penalty, max_rank=max_rank
        )


# =====================================================================
# Section 4: Helmholtz MPO Assembly with Full BCs
# =====================================================================

def helmholtz_mpo_with_bc(
    geometry: Geometry1D,
    k: float,
    max_rank: int = 64,
    pec_penalty: float = 1e8,
) -> list[NDArray]:
    """Build Helmholtz operator with UPML boundary conditions.

    Assembles:
      H = L_s + k²·diag(ε_damped) [+ penalty·diag(conductor)]

    where L_s is the stretched-coordinate Laplacian implementing
    the UPML absorbing boundary.  The PML absorption is entirely
    in the Laplacian (coordinate stretching), not in ε.

    Parameters
    ----------
    geometry : Geometry1D
        Full geometry specification (materials + PML + conductors).
    k : float
        Wavenumber (2πf/c).
    max_rank : int
        Maximum QTT rank for intermediate representations.
    pec_penalty : float
        PEC enforcement penalty magnitude.

    Returns
    -------
    list[NDArray]
        Complex MPO cores for the full Helmholtz operator.
    """
    n = geometry.n_bits
    N = 2 ** n
    h = 1.0 / N

    # Stretched-coordinate Laplacian with PML (Dirichlet BCs)
    L_s = stretched_laplacian_mpo_1d(
        n, k, h, geometry.pml, max_rank=max_rank,
    )

    # ε(x) with damping only (PML is in the Laplacian)
    eps_qtt = geometry.build_eps_qtt(k=k, max_rank=max_rank)

    # k² · diag(ε) MPO
    eps_mpo = diag_mpo_from_tt(eps_qtt)
    k2_eps = mpo_scale_c(eps_mpo, k * k)

    # H = L_s + k²·diag(ε)
    H = mpo_add_c(L_s, k2_eps)

    # Add PEC penalty if conductors present
    if geometry.has_conductors():
        P = geometry.build_penalty_mpo(
            penalty=pec_penalty, max_rank=max_rank
        )
        P_complex = [c.astype(np.complex128) for c in P]
        H = mpo_add_c(H, P_complex)

    return H


# =====================================================================
# Section 5: Pre-built Geometries for Common Antenna/Waveguide Problems
# =====================================================================

def free_space_geometry(
    n_bits: int = 14,
    pml_cells: int = 30,
    sigma_max: float = 10.0,
) -> Geometry1D:
    """Free space with PML absorbing boundaries.

    Standard validation geometry: point source radiating outward,
    absorbed by PML at both ends.
    """
    return Geometry1D(
        n_bits=n_bits,
        background_eps=1.0,
        pml=PMLConfig(n_cells=pml_cells, sigma_max=sigma_max),
    )


def dielectric_slab_geometry(
    n_bits: int = 14,
    eps_slab: float = 4.0,
    slab_start: float = 0.3,
    slab_end: float = 0.7,
    pml_cells: int = 30,
    sigma_max: float = 10.0,
) -> Geometry1D:
    """Dielectric slab between PML boundaries.

    Canonical test case for reflection/transmission through a
    dielectric interface.  Exact Fresnel coefficients known.
    """
    geo = Geometry1D(
        n_bits=n_bits,
        background_eps=1.0,
        pml=PMLConfig(n_cells=pml_cells, sigma_max=sigma_max),
    )
    geo.add_dielectric_slab(eps_slab, slab_start, slab_end, "slab")
    return geo


def parallel_plate_geometry(
    n_bits: int = 14,
    plate_positions: tuple[float, float] = (0.3, 0.7),
    plate_width: float = 0.005,
    filling_eps: float = 1.0,
    pml_cells: int = 30,
    sigma_max: float = 10.0,
) -> Geometry1D:
    """Parallel-plate waveguide with PEC boundaries.

    Two PEC plates at specified positions with dielectric filling.
    Used for validating PEC enforcement and guided-wave physics.
    """
    geo = Geometry1D(
        n_bits=n_bits,
        background_eps=filling_eps,
        pml=PMLConfig(n_cells=pml_cells, sigma_max=sigma_max),
    )
    x1, x2 = plate_positions
    w = plate_width
    geo.add_conductor(x1, x1 + w)
    geo.add_conductor(x2, x2 + w)
    return geo


def microstrip_geometry(
    n_bits: int = 14,
    substrate_eps: float = 4.4,
    substrate_start: float = 0.35,
    substrate_end: float = 0.5,
    strip_position: float = 0.5,
    strip_width: float = 0.005,
    ground_position: float = 0.35,
    ground_width: float = 0.005,
    pml_cells: int = 30,
    sigma_max: float = 10.0,
) -> Geometry1D:
    """Simplified 1D microstrip cross-section.

    Ground plane → substrate (ε_r) → strip conductor → air.
    Used for characteristic impedance estimation.
    """
    geo = Geometry1D(
        n_bits=n_bits,
        background_eps=1.0,
        pml=PMLConfig(n_cells=pml_cells, sigma_max=sigma_max),
    )
    geo.add_dielectric_slab(substrate_eps, substrate_start, substrate_end, "substrate")
    geo.add_conductor(ground_position, ground_position + ground_width)
    geo.add_conductor(strip_position, strip_position + strip_width)
    return geo


# =====================================================================
# Section 6: Diagnostic Utilities
# =====================================================================

def analyze_pml_performance(
    n_bits: int,
    pml: PMLConfig,
    k: float,
    verbose: bool = True,
) -> dict:
    """Analyze PML absorption quality.

    Computes theoretical reflection coefficient and effective
    absorption per wavelength for the given PML configuration.

    Parameters
    ----------
    n_bits : int
        QTT resolution.
    pml : PMLConfig
        PML parameters.
    k : float
        Wavenumber.
    verbose : bool
        Print diagnostics.

    Returns
    -------
    dict
        PML performance metrics.
    """
    N = 2 ** n_bits
    h = 1.0 / N
    L_pml = pml.n_cells * h
    wavelength = 2.0 * math.pi / k
    pml_wavelengths = L_pml / wavelength
    p = pml.poly_order

    # Theoretical reflection at normal incidence
    # R = exp(-2 σ_max L_pml / (p + 1))
    R = math.exp(-2.0 * pml.sigma_max * L_pml / (p + 1))
    R_dB = 20.0 * math.log10(max(R, 1e-30))

    # QTT rank of ε and stretching profile
    eps_qtt = build_pml_profile_1d(n_bits, pml, k=k)
    ranks = [c.shape[2] for c in eps_qtt[:-1]]
    pml_rank = max(ranks) if ranks else 1

    # QTT rank of stretching factor 1/s (used by stretched Laplacian)
    s_vec = _compute_pml_stretching(N, k, pml)
    inv_s_tt = array_to_tt(1.0 / s_vec, max_rank=64, cutoff=1e-12)
    ranks_s = [c.shape[2] for c in inv_s_tt[:-1]]
    stretch_rank = max(ranks_s) if ranks_s else 1

    # Optimal σ_max for R = -60 dB
    sigma_opt = pml.optimal_sigma_max(k, h, target_R=1e-6)

    metrics = {
        "L_pml": L_pml,
        "pml_wavelengths": pml_wavelengths,
        "R_normal": R,
        "R_dB": R_dB,
        "sigma_max": pml.sigma_max,
        "sigma_optimal_R60dB": sigma_opt,
        "poly_order": pml.poly_order,
        "qtt_rank": pml_rank,
        "stretch_rank": stretch_rank,
    }

    if verbose:
        print(f"PML Analysis (N={N}, k={k:.2f}, λ={wavelength:.4f})")
        print(f"  Thickness:  {pml.n_cells} cells = {L_pml:.4f} = "
              f"{pml_wavelengths:.2f}λ")
        print(f"  σ_max:      {pml.sigma_max:.1f} "
              f"(optimal for R=-60dB: {sigma_opt:.1f})")
        print(f"  Grading:    polynomial order {p}")
        print(f"  Reflection: R = {R:.2e} ({R_dB:.1f} dB)")
        print(f"  QTT rank:   ε={pml_rank}, 1/s={stretch_rank}")

    return metrics


def visualize_geometry(
    geometry: Geometry1D,
    k: float = 0.0,
) -> dict:
    """Extract dense arrays for visualisation of geometry.

    Returns
    -------
    dict
        Keys: 'x', 'eps_r_real', 'eps_r_imag', 'conductor_mask',
        'eps_pml_real', 'eps_pml_imag'.
    """
    from tensornet.em.qtt_helmholtz import reconstruct_1d

    N = 2 ** geometry.n_bits
    x = np.linspace(0.5 / N, 1.0 - 0.5 / N, N)

    # Base material profile (no PML)
    eps_base = geometry.build_eps_profile()

    # Damped ε (no PML — PML is in the Laplacian)
    k_vis = k if k > 0.0 else 1.0
    eps_damped_qtt = geometry.build_eps_qtt(k=k_vis)
    eps_damped = reconstruct_1d(eps_damped_qtt)

    # PML stretching factor s(x)
    s_profile = _compute_pml_stretching(N, k_vis, geometry.pml)

    # Conductor mask
    cond_mask = geometry.build_conductor_mask()

    return {
        "x": x,
        "eps_r_real": eps_base.real,
        "eps_r_imag": eps_base.imag,
        "conductor_mask": cond_mask,
        "eps_pml_real": eps_damped.real,
        "eps_pml_imag": eps_damped.imag,
        "s_real": s_profile.real,
        "s_imag": s_profile.imag,
    }
