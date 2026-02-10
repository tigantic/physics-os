"""
Non-Equilibrium Green's Function (NEGF) Method
================================================

Quantum transport through nano-scale devices, computing transmission
coefficients, current–voltage characteristics, and local density of
states from retarded and lesser Green's functions.

Formalism
---------
For a central device region coupled to left (L) and right (R) semi-
infinite leads:

.. math::
    G^R(E) = [(E + i\\eta) I - H_D - \\Sigma_L^R(E) - \\Sigma_R^R(E)]^{-1}

.. math::
    T(E) = \\text{Tr}[\\Gamma_L G^R \\Gamma_R G^{R\\dagger}]

.. math::
    I = \\frac{2e}{h} \\int T(E) [f_L(E) - f_R(E)]\\,dE

where:
    - :math:`\\Sigma_{L/R}^R` are the retarded self-energies of the leads.
    - :math:`\\Gamma_{L/R} = i (\\Sigma^R - \\Sigma^{R\\dagger})` are
      the broadening matrices.
    - :math:`f_{L/R}` are Fermi functions at chemical potentials
      :math:`\\mu_{L/R}`.

References:
    [1] Datta, *Electronic Transport in Mesoscopic Systems*, Cambridge (1995).
    [2] Datta, *Quantum Transport: Atom to Transistor*, Cambridge (2005).
    [3] Brandbyge et al., PRB 65, 165401 (2002) (TranSIESTA).

Domain V.2 — Condensed-Matter Physics / Quantum Transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_E_CHARGE = 1.602176634e-19    # C
_H_PLANCK = 6.62607015e-34     # J·s
_KB = 1.380649e-23             # J/K


# ---------------------------------------------------------------------------
# Utility: Fermi function
# ---------------------------------------------------------------------------

def fermi(E: NDArray, mu: float, kT: float) -> NDArray:
    """Fermi-Dirac distribution."""
    x = (E - mu) / (kT + 1e-30)
    return 1.0 / (1.0 + np.exp(np.clip(x, -500.0, 500.0)))


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class NEGFDevice:
    """
    NEGF device specification.

    Attributes:
        H_D: Device Hamiltonian ``(n, n)`` (Hermitian).
        H_DL: Coupling device → left lead ``(n, n_lead)``.
        H_DR: Coupling device → right lead ``(n, n_lead)``.
        H_L: Left lead unit-cell Hamiltonian ``(n_lead, n_lead)``.
        H_R: Right lead unit-cell Hamiltonian ``(n_lead, n_lead)``.
        t_L: Left lead hopping ``(n_lead, n_lead)``.
        t_R: Right lead hopping ``(n_lead, n_lead)``.
    """
    H_D: NDArray
    H_DL: NDArray
    H_DR: NDArray
    H_L: NDArray
    H_R: NDArray
    t_L: NDArray
    t_R: NDArray

    @property
    def n_device(self) -> int:
        return self.H_D.shape[0]

    @property
    def n_lead(self) -> int:
        return self.H_L.shape[0]


@dataclass
class NEGFResult:
    """NEGF transport results."""
    energies: NDArray           # (n_E,)
    transmission: NDArray       # (n_E,)
    dos: NDArray                # (n_E,)  local density of states
    current: float              # Total current [A]
    conductance: float          # Differential conductance [e²/h]


# ---------------------------------------------------------------------------
# Lead self-energy (iterative surface Green's function)
# ---------------------------------------------------------------------------

def surface_greens_function(
    E: complex,
    H_lead: NDArray,
    t_hop: NDArray,
    eta: float = 1e-6,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> NDArray:
    r"""
    Iterative decimation (Sancho-Rubio) to compute the surface
    Green's function :math:`g_s(E)` of a semi-infinite lead.

    The algorithm doubles the effective hopping range at each step
    until convergence.

    Parameters:
        E: Complex energy (with broadening).
        H_lead: Lead unit-cell Hamiltonian ``(m, m)``.
        t_hop: Inter-cell hopping ``(m, m)``.
        eta: Broadening.
        max_iter: Maximum decimation iterations.
        tol: Convergence tolerance.

    Returns:
        Surface Green's function ``(m, m)``.
    """
    m = H_lead.shape[0]
    z = (E + 1j * eta) * np.eye(m, dtype=np.complex128) - H_lead

    alpha = t_hop.copy().astype(np.complex128)
    beta = t_hop.conj().T.copy().astype(np.complex128)
    eps_s = H_lead.copy().astype(np.complex128)
    eps_bulk = H_lead.copy().astype(np.complex128)

    for _ in range(max_iter):
        g_inv = (E + 1j * eta) * np.eye(m, dtype=np.complex128) - eps_bulk
        g = np.linalg.inv(g_inv)

        eps_s_new = eps_s + alpha @ g @ beta
        eps_bulk_new = eps_bulk + alpha @ g @ beta + beta @ g @ alpha
        alpha_new = alpha @ g @ alpha
        beta_new = beta @ g @ beta

        if np.linalg.norm(alpha_new) < tol:
            eps_s = eps_s_new
            break

        eps_s = eps_s_new
        eps_bulk = eps_bulk_new
        alpha = alpha_new
        beta = beta_new

    g_s = np.linalg.inv(
        (E + 1j * eta) * np.eye(m, dtype=np.complex128) - eps_s,
    )
    return g_s


def lead_self_energy(
    E: complex,
    H_DX: NDArray,
    H_lead: NDArray,
    t_hop: NDArray,
    eta: float = 1e-6,
) -> NDArray:
    r"""
    Lead retarded self-energy:
    :math:`\Sigma_X^R(E) = H_{DX}^{\dagger}\, g_s(E)\, H_{DX}`.

    Parameters:
        E: Energy.
        H_DX: Device–lead coupling ``(n, n_lead)``.
        H_lead: Lead unit-cell Hamiltonian.
        t_hop: Lead hopping.
        eta: Broadening.

    Returns:
        Self-energy ``(n, n)`` matrix.
    """
    g_s = surface_greens_function(E, H_lead, t_hop, eta=eta)
    sigma = H_DX @ g_s @ H_DX.conj().T
    return sigma


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class NEGFSolver:
    r"""
    NEGF transport solver.

    Computes transmission, LDOS, and current for a nanoscale device
    coupled to two leads.

    Parameters:
        device: Device specification.
        eta: Broadening parameter (default 1e-4 eV).

    Example::

        # 1D tight-binding chain with on-site energy ε and hopping t
        n = 10
        t = -1.0
        eps = 0.0
        H_D = eps * np.eye(n) + t * (np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-1), -1))
        H_DL = np.zeros((n, 1)); H_DL[0, 0] = t
        H_DR = np.zeros((n, 1)); H_DR[-1, 0] = t
        H_lead = np.array([[eps]])
        t_lead = np.array([[t]])
        dev = NEGFDevice(H_D, H_DL, H_DR, H_lead, H_lead, t_lead, t_lead)
        solver = NEGFSolver(dev)
        result = solver.solve(E_range=(-3, 3), n_E=500, mu_L=0.5, mu_R=-0.5, kT=0.025)
    """

    def __init__(self, device: NEGFDevice, eta: float = 1e-4) -> None:
        self.device = device
        self.eta = eta

    def retarded_greens_function(self, E: float) -> NDArray:
        """Compute :math:`G^R(E)`."""
        d = self.device
        n = d.n_device
        sigma_L = lead_self_energy(E, d.H_DL, d.H_L, d.t_L, eta=self.eta)
        sigma_R = lead_self_energy(E, d.H_DR, d.H_R, d.t_R, eta=self.eta)
        G_inv = (E + 1j * self.eta) * np.eye(n, dtype=np.complex128) - d.H_D - sigma_L - sigma_R
        return np.linalg.inv(G_inv)

    def transmission(self, E: float) -> float:
        """Transmission coefficient :math:`T(E)`."""
        d = self.device
        sigma_L = lead_self_energy(E, d.H_DL, d.H_L, d.t_L, eta=self.eta)
        sigma_R = lead_self_energy(E, d.H_DR, d.H_R, d.t_R, eta=self.eta)

        gamma_L = 1j * (sigma_L - sigma_L.conj().T)
        gamma_R = 1j * (sigma_R - sigma_R.conj().T)

        G_R = self.retarded_greens_function(E)
        T_E = np.real(np.trace(gamma_L @ G_R @ gamma_R @ G_R.conj().T))
        return max(0.0, float(T_E))

    def local_dos(self, E: float) -> NDArray:
        """Local density of states at each device site."""
        G_R = self.retarded_greens_function(E)
        return -np.diag(G_R).imag / np.pi

    def solve(
        self,
        E_range: Tuple[float, float] = (-3.0, 3.0),
        n_E: int = 500,
        mu_L: float = 0.5,
        mu_R: float = -0.5,
        kT: float = 0.025,
    ) -> NEGFResult:
        """
        Full transport calculation.

        Parameters:
            E_range: Energy window.
            n_E: Number of energy points.
            mu_L, mu_R: Chemical potentials of left/right leads.
            kT: Thermal energy k_B T.

        Returns:
            NEGFResult with transmission, DOS, current, conductance.
        """
        energies = np.linspace(E_range[0], E_range[1], n_E)
        trans = np.zeros(n_E, dtype=np.float64)
        dos = np.zeros(n_E, dtype=np.float64)

        for i, E in enumerate(energies):
            trans[i] = self.transmission(E)
            ldos = self.local_dos(E)
            dos[i] = np.sum(ldos)

        # Landauer current: I = (2e/h) ∫ T(E) [f_L - f_R] dE
        dE = energies[1] - energies[0] if n_E > 1 else 1.0
        f_L = fermi(energies, mu_L, kT)
        f_R = fermi(energies, mu_R, kT)
        current = (2.0 * _E_CHARGE / _H_PLANCK) * np.trapz(trans * (f_L - f_R), energies)

        # Conductance quantum: G_0 = 2e²/h
        G_0 = 2.0 * _E_CHARGE ** 2 / _H_PLANCK
        V_bias = mu_L - mu_R
        conductance = float(current / (V_bias + 1e-30)) if abs(V_bias) > 1e-12 else 0.0

        return NEGFResult(
            energies=energies,
            transmission=trans,
            dos=dos,
            current=float(current),
            conductance=conductance,
        )
