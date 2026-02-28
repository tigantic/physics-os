"""
Magnetotellurics / Geo-Electromagnetic Inversion
==================================================

Forward modelling and inversion of the magnetotelluric (MT) response
for layered and 2D earth conductivity structures.

Physics:
    MT uses natural electromagnetic (EM) fields from ionospheric
    currents to probe subsurface resistivity.  A plane wave
    ``E_x, H_y`` at frequency :math:`\\omega` penetrates to a
    skin depth:

    .. math::
        \\delta = \\sqrt{\\frac{2}{\\omega \\mu_0 \\sigma}}

    The surface impedance :math:`Z = E_x / H_y` encodes the
    1D conductivity profile.

1D Forward (Wait's recursion):
    For N layers with conductivities :math:`\\sigma_k` and thicknesses
    :math:`h_k`, the impedance at the top is computed recursively
    bottom-to-top:

    .. math::
        Z_k = Z_{k+1}^{\\text{int}} \\frac{Z_{k+1} + Z_{k+1}^{\\text{int}} \\tanh(\\gamma_k h_k)}
        {Z_{k+1}^{\\text{int}} + Z_{k+1} \\tanh(\\gamma_k h_k)}

Apparent resistivity & phase:
    .. math::
        \\rho_a = \\frac{1}{\\omega \\mu_0} |Z|^2, \\qquad
        \\phi = \\arg(Z)

References:
    [1] Wait, "On the Relation Between Telluric Currents and the
        Earth's Magnetic Field", Geophysics 19, 1954.
    [2] Vozoff, "The Magnetotelluric Method", in *Electromagnetic
        Methods in Applied Geophysics*, SEG 1991.
    [3] Constable, Parker & Constable, "Occam's inversion",
        Geophysics 52, 289 (1987).

Domain IX.17 — Geophysics / Magnetotellurics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


_MU_0 = 4e-7 * np.pi  # vacuum permeability [H/m]


# ---------------------------------------------------------------------------
# Earth model
# ---------------------------------------------------------------------------

@dataclass
class LayeredEarth:
    """
    1D layered-earth conductivity model.

    Attributes:
        sigma: Layer conductivities ``(n_layers,)`` [S/m].
        h: Layer thicknesses ``(n_layers - 1,)`` [m].
            (last layer is a half-space: infinite thickness.)
    """
    sigma: NDArray
    h: NDArray

    @property
    def n_layers(self) -> int:
        return self.sigma.shape[0]

    @property
    def rho(self) -> NDArray:
        """Layer resistivities [Ω·m]."""
        return 1.0 / (self.sigma + 1e-30)


# ---------------------------------------------------------------------------
# 1D forward: Wait's recursion
# ---------------------------------------------------------------------------

def mt_forward_1d(
    model: LayeredEarth,
    frequencies: NDArray,
) -> Tuple[NDArray, NDArray]:
    """
    Compute apparent resistivity and phase via Wait's recursion.

    Parameters:
        model: Layered-earth conductivity model.
        frequencies: Array of frequencies [Hz].

    Returns:
        (rho_a, phase) — apparent resistivity [Ω·m] and phase [degrees].
    """
    n_f = frequencies.shape[0]
    n_L = model.n_layers
    rho_a = np.zeros(n_f, dtype=np.float64)
    phase = np.zeros(n_f, dtype=np.float64)

    for fi, f in enumerate(frequencies):
        omega = 2.0 * np.pi * f

        # Propagation constants γ_k = sqrt(iωμ₀σ_k)
        gamma = np.sqrt(1j * omega * _MU_0 * model.sigma)

        # Intrinsic impedance of each layer: Z_int = iωμ₀ / γ
        Z_int = 1j * omega * _MU_0 / (gamma + 1e-30)

        # Bottom-up recursion (start from deepest half-space)
        Z = Z_int[n_L - 1]

        for k in range(n_L - 2, -1, -1):
            tanh_val = np.tanh(gamma[k] * model.h[k])
            Z = Z_int[k] * (Z + Z_int[k] * tanh_val) / (Z_int[k] + Z * tanh_val + 1e-30)

        rho_a[fi] = (1.0 / (omega * _MU_0)) * np.abs(Z) ** 2
        phase[fi] = np.degrees(np.angle(Z))

    return rho_a, phase


# ---------------------------------------------------------------------------
# Apparent resistivity utilities
# ---------------------------------------------------------------------------

def skin_depth(frequency: float, sigma: float) -> float:
    """Electromagnetic skin depth [m]."""
    omega = 2.0 * np.pi * frequency
    return np.sqrt(2.0 / (omega * _MU_0 * sigma + 1e-30))


def impedance_to_rho_a(Z: complex, frequency: float) -> float:
    """Convert impedance to apparent resistivity."""
    omega = 2.0 * np.pi * frequency
    return np.abs(Z) ** 2 / (omega * _MU_0)


# ---------------------------------------------------------------------------
# 1D Occam's inversion
# ---------------------------------------------------------------------------

class OccamInversion1D:
    r"""
    Occam's inversion (smoothness-constrained) for 1D MT data.

    Minimises:
    .. math::
        \Phi = \|\mathbf{W}_d (\mathbf{d}^{\text{obs}} - \mathbf{d}^{\text{pred}})\|^2
             + \lambda \|\mathbf{D} \log\sigma\|^2

    where :math:`\mathbf{D}` is a finite-difference roughness operator
    and :math:`\lambda` the trade-off (regularisation) parameter.

    Parameters:
        frequencies: Observation frequencies [Hz].
        layer_thicknesses: Fixed layer thicknesses (n_layers - 1,) [m].
        n_layers: Number of inversion layers.

    Example::

        inv = OccamInversion1D(freqs, h, n_layers=30)
        sigma_inv, rho_a_pred = inv.invert(rho_a_obs, phase_obs)
    """

    def __init__(
        self,
        frequencies: NDArray,
        layer_thicknesses: NDArray,
        n_layers: int = 30,
        lambda_reg: float = 1.0,
    ) -> None:
        self.frequencies = frequencies
        self.h = layer_thicknesses
        self.n_layers = n_layers
        self.lambda_reg = lambda_reg

    def _forward(self, sigma: NDArray) -> Tuple[NDArray, NDArray]:
        model = LayeredEarth(sigma=sigma, h=self.h)
        return mt_forward_1d(model, self.frequencies)

    def _roughness_matrix(self) -> NDArray:
        """First-difference roughness operator D."""
        n = self.n_layers
        D = np.zeros((n - 1, n), dtype=np.float64)
        for i in range(n - 1):
            D[i, i] = -1.0
            D[i, i + 1] = 1.0
        return D

    def _jacobian(self, sigma: NDArray, eps: float = 0.01) -> NDArray:
        """Numerical Jacobian ∂(log ρ_a) / ∂(log σ)."""
        rho_a0, _ = self._forward(sigma)
        n_f = self.frequencies.shape[0]
        n_L = self.n_layers
        J = np.zeros((n_f, n_L), dtype=np.float64)
        for k in range(n_L):
            sigma_pert = sigma.copy()
            sigma_pert[k] *= (1.0 + eps)
            rho_a_pert, _ = self._forward(sigma_pert)
            J[:, k] = (np.log(rho_a_pert) - np.log(rho_a0 + 1e-30)) / (np.log(1.0 + eps))
        return J

    def invert(
        self,
        rho_a_obs: NDArray,
        phase_obs: NDArray,
        sigma_0: Optional[NDArray] = None,
        n_iter: int = 20,
        tol: float = 1e-3,
    ) -> Tuple[NDArray, NDArray]:
        """
        Run Occam's inversion.

        Parameters:
            rho_a_obs: Observed apparent resistivity [Ω·m].
            phase_obs: Observed phase [degrees] (used for weighting only).
            sigma_0: Initial conductivity guess (default: 0.01 S/m uniform).
            n_iter: Maximum iterations.
            tol: Convergence tolerance on relative misfit change.

        Returns:
            (sigma_final, rho_a_predicted).
        """
        n_f = self.frequencies.shape[0]
        n_L = self.n_layers

        if sigma_0 is None:
            sigma = np.full(n_L, 0.01, dtype=np.float64)
        else:
            sigma = sigma_0.copy()

        D = self._roughness_matrix()
        lam = self.lambda_reg

        # Data weights (relative errors assumed 5%)
        W_d = np.diag(1.0 / (0.05 * rho_a_obs + 1e-30))

        prev_misfit = np.inf
        for it in range(n_iter):
            rho_a_pred, phase_pred = self._forward(sigma)
            residual = np.log(rho_a_obs + 1e-30) - np.log(rho_a_pred + 1e-30)
            misfit = np.linalg.norm(W_d @ residual)

            if abs(prev_misfit - misfit) / (prev_misfit + 1e-30) < tol:
                break
            prev_misfit = misfit

            J = self._jacobian(sigma)

            # Gauss-Newton with regularisation
            # (J^T W^T W J + λ D^T D) Δm = J^T W^T W r - λ D^T D m
            log_sigma = np.log(sigma + 1e-30)
            WJ = W_d @ J
            lhs = WJ.T @ WJ + lam * D.T @ D
            rhs = WJ.T @ W_d @ residual - lam * D.T @ D @ log_sigma

            dm = np.linalg.solve(lhs + 1e-10 * np.eye(n_L), rhs)
            log_sigma += 0.5 * dm  # damped step
            sigma = np.exp(log_sigma)
            sigma = np.clip(sigma, 1e-6, 1e3)

        rho_a_final, _ = self._forward(sigma)
        return sigma, rho_a_final


# ---------------------------------------------------------------------------
# 2D forward (simplified finite-difference, TE mode)
# ---------------------------------------------------------------------------

def mt_forward_2d_te(
    sigma_2d: NDArray,
    dx: float,
    dz: float,
    frequency: float,
) -> NDArray:
    """
    Simplified 2D MT forward modelling (TE mode) via finite differences.

    Solves the Helmholtz equation for :math:`E_y`:

    .. math::
        \\nabla^2 E_y + i \\omega \\mu_0 \\sigma E_y = 0

    Parameters:
        sigma_2d: Conductivity model ``(nz, nx)`` [S/m].
        dx, dz: Grid spacings [m].
        frequency: Frequency [Hz].

    Returns:
        E_y at the surface ``(nx,)`` (complex).
    """
    nz, nx = sigma_2d.shape
    omega = 2.0 * np.pi * frequency
    k2 = 1j * omega * _MU_0 * sigma_2d  # (nz, nx)

    N = nz * nx
    A = np.zeros((N, N), dtype=np.complex128)
    b = np.zeros(N, dtype=np.complex128)

    def _idx(iz: int, ix: int) -> int:
        return iz * nx + ix

    for iz in range(nz):
        for ix in range(nx):
            n = _idx(iz, ix)
            if iz == 0 or iz == nz - 1 or ix == 0 or ix == nx - 1:
                # Boundary: Dirichlet (plane-wave at top, zero at sides/bottom)
                A[n, n] = 1.0
                if iz == 0:
                    b[n] = 1.0  # unit incident field at surface
            else:
                # Interior: 5-point Laplacian + k² E = 0
                A[n, _idx(iz, ix - 1)] = 1.0 / dx ** 2
                A[n, _idx(iz, ix + 1)] = 1.0 / dx ** 2
                A[n, _idx(iz - 1, ix)] = 1.0 / dz ** 2
                A[n, _idx(iz + 1, ix)] = 1.0 / dz ** 2
                A[n, n] = -2.0 / dx ** 2 - 2.0 / dz ** 2 + k2[iz, ix]

    E = np.linalg.solve(A, b)
    E_surface = E[:nx]  # first row
    return E_surface
