"""
Electronic Band Structure — tight-binding, plane-wave expansion, k·p theory,
Wannier functions, density of states.

Domain IX.2 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

HBAR: float = 1.055e-34     # J·s
M_E: float = 9.109e-31      # electron mass (kg)
EV_J: float = 1.602e-19     # J/eV
BOHR: float = 5.292e-11     # m
ANGSTROM: float = 1e-10     # m


# ---------------------------------------------------------------------------
#  Tight-Binding Band Structure
# ---------------------------------------------------------------------------

class TightBindingBands:
    r"""
    Tight-binding band structure for crystalline solids.

    Hamiltonian:
    $$H(\mathbf{k}) = \sum_{\mathbf{R}} e^{i\mathbf{k}\cdot\mathbf{R}} H_{\mathbf{R}}$$

    1D chain: $E(k) = \epsilon_0 + 2t\cos(ka)$

    Graphene (2 sublattices):
    $$E_\pm(\mathbf{k}) = \pm|t||f(\mathbf{k})|$$
    $$f(\mathbf{k}) = 1 + e^{i\mathbf{k}\cdot\mathbf{a}_1} + e^{i\mathbf{k}\cdot\mathbf{a}_2}$$
    """

    def __init__(self, dim: int = 1, n_orbitals: int = 1) -> None:
        self.dim = dim
        self.n_orbitals = n_orbitals
        self.hoppings: List[Tuple[NDArray, NDArray]] = []

    def add_hopping(self, R: NDArray, H_R: NDArray) -> None:
        """Add hopping matrix H_R for lattice vector R.

        R: (dim,) lattice vector (in units of lattice constant).
        H_R: (n_orb, n_orb) hopping matrix.
        """
        self.hoppings.append((np.asarray(R, float), np.asarray(H_R, complex)))

    def hamiltonian(self, k: NDArray) -> NDArray:
        """Build H(k) = Σ exp(ik·R) H_R."""
        H = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
        for R, H_R in self.hoppings:
            phase = np.exp(1j * np.dot(k, R))
            H += phase * H_R
        return H

    def bands(self, k_path: NDArray) -> NDArray:
        """Compute eigenvalues along k-path.

        k_path: (n_k, dim)
        Returns: (n_k, n_orbitals) eigenvalues.
        """
        n_k = len(k_path)
        E = np.zeros((n_k, self.n_orbitals))
        for i in range(n_k):
            H = self.hamiltonian(k_path[i])
            E[i] = np.sort(np.real(np.linalg.eigvalsh(H)))
        return E

    @classmethod
    def chain_1d(cls, eps0: float = 0.0, t: float = -1.0,
                    a: float = 1.0) -> 'TightBindingBands':
        """1D monoatomic chain: E(k) = ε₀ + 2t cos(ka)."""
        model = cls(dim=1, n_orbitals=1)
        model.add_hopping(np.array([0.0]), np.array([[eps0]]))
        model.add_hopping(np.array([a]), np.array([[t]]))
        model.add_hopping(np.array([-a]), np.array([[t]]))
        return model

    @classmethod
    def graphene(cls, t: float = -2.7) -> 'TightBindingBands':
        """Graphene tight-binding (nearest-neighbour)."""
        model = cls(dim=2, n_orbitals=2)
        # On-site
        model.add_hopping(np.array([0.0, 0.0]),
                          np.array([[0, t], [t, 0]], complex))
        # Nearest neighbours
        a1 = np.array([1.0, 0.0])
        a2 = np.array([0.5, math.sqrt(3) / 2])
        model.add_hopping(a1, np.array([[0, t], [0, 0]], complex))
        model.add_hopping(-a1, np.array([[0, 0], [t, 0]], complex))
        model.add_hopping(a2, np.array([[0, t], [0, 0]], complex))
        model.add_hopping(-a2, np.array([[0, 0], [t, 0]], complex))
        return model


# ---------------------------------------------------------------------------
#  k·p Method
# ---------------------------------------------------------------------------

class KdotPMethod:
    r"""
    k·p perturbation theory for band structure near high-symmetry points.

    $$H(\mathbf{k}) = H_0 + \frac{\hbar}{m_0}\mathbf{k}\cdot\hat{\mathbf{p}}
      + \frac{\hbar^2 k^2}{2m_0}$$

    2-band Kane model:
    $$E_\pm(k) = \frac{E_g}{2} \pm \sqrt{\left(\frac{E_g}{2}\right)^2
      + \frac{\hbar^2 k^2 E_p}{2m_0}}$$

    where $E_p = 2|P|^2 m_0/\hbar^2$ is the Kane energy parameter.

    Effective mass: $\frac{1}{m^*} = \frac{1}{m_0}\left(1+\frac{E_p}{E_g}\right)$
    """

    def __init__(self, Eg: float = 1.42, Ep: float = 25.7) -> None:
        """
        Eg: band gap (eV).
        Ep: Kane energy (eV).
        """
        self.Eg = Eg * EV_J
        self.Ep = Ep * EV_J

    def dispersion(self, k: NDArray) -> Tuple[NDArray, NDArray]:
        """2-band Kane model E±(k).

        k: wavevector (1/m).
        Returns (E_c, E_v) in eV.
        """
        hk2 = HBAR**2 * k**2 / (2 * M_E)
        sqrt_term = np.sqrt((self.Eg / 2)**2 + hk2 * self.Ep)
        E_c = (self.Eg / 2 + sqrt_term) / EV_J
        E_v = (self.Eg / 2 - sqrt_term) / EV_J
        return E_c, E_v

    def effective_mass_cb(self) -> float:
        """Conduction band effective mass m*/m₀."""
        return 1 / (1 + self.Ep / self.Eg)

    def effective_mass_vb(self) -> float:
        """Valence band effective mass (negative)."""
        return -1 / (1 + self.Ep / self.Eg)

    def nonparabolicity(self, E: float) -> float:
        """Non-parabolicity parameter α = 1/Eg (eV⁻¹).

        k²ℏ²/(2m*) = E(1 + αE)
        """
        return 1.0 / (self.Eg / EV_J)

    def luttinger_parameters(self, gamma1: float = 6.85, gamma2: float = 2.1,
                                gamma3: float = 2.9) -> Dict[str, float]:
        """Luttinger parameters for valence band.

        Heavy hole: m_hh = m₀/(γ₁ − 2γ₂)
        Light hole: m_lh = m₀/(γ₁ + 2γ₂)
        Split-off: determined by ΔSO.
        """
        return {
            'm_hh/m0': 1 / (gamma1 - 2 * gamma2),
            'm_lh/m0': 1 / (gamma1 + 2 * gamma2),
            'gamma1': gamma1,
            'gamma2': gamma2,
            'gamma3': gamma3,
        }


# ---------------------------------------------------------------------------
#  Density of States
# ---------------------------------------------------------------------------

class DensityOfStates:
    r"""
    Electronic density of states from eigenvalue spectrum.

    Analytical DOS:
    - 1D: $g(E) = \frac{L}{\pi}\frac{1}{\hbar}\sqrt{\frac{m^*}{2(E-E_0)}}$
    - 2D: $g(E) = \frac{m^*}{\pi\hbar^2}$ (constant)
    - 3D: $g(E) = \frac{1}{2\pi^2}\left(\frac{2m^*}{\hbar^2}\right)^{3/2}\sqrt{E-E_0}$

    Numerical: histogram or kernel polynomial method.
    """

    @staticmethod
    def analytical_3d(E: NDArray, m_star: float = 0.067,
                         E0: float = 0.0) -> NDArray:
        """3D free-electron DOS g(E) (states/eV/m³).

        m_star: effective mass ratio m*/m₀.
        E0: band edge (eV).
        """
        m = m_star * M_E
        E_J = (E - E0) * EV_J
        g = np.where(E_J > 0,
                     1 / (2 * math.pi**2) * (2 * m / HBAR**2)**1.5 * np.sqrt(E_J),
                     0)
        return g / EV_J  # convert to per eV

    @staticmethod
    def analytical_2d(m_star: float = 0.067) -> float:
        """2D DOS g(E) = m*/(πℏ²) (states/eV/m² per spin)."""
        m = m_star * M_E
        return m / (math.pi * HBAR**2) / EV_J

    @staticmethod
    def analytical_1d(E: NDArray, m_star: float = 0.067,
                         E0: float = 0.0) -> NDArray:
        """1D DOS g(E) per unit length."""
        m = m_star * M_E
        E_J = (E - E0) * EV_J
        g = np.where(E_J > 0,
                     1 / (math.pi * HBAR) * np.sqrt(m / (2 * E_J + 1e-30)),
                     0)
        return g / EV_J

    @staticmethod
    def from_eigenvalues(eigenvalues: NDArray, E_range: Tuple[float, float],
                            sigma: float = 0.05,
                            n_pts: int = 500) -> Tuple[NDArray, NDArray]:
        """Numerical DOS via Gaussian broadening.

        eigenvalues: (n_k, n_bands) or flat array.
        sigma: broadening (eV).
        """
        evals = eigenvalues.flatten()
        E = np.linspace(E_range[0], E_range[1], n_pts)
        dos = np.zeros(n_pts)
        norm = 1 / (sigma * math.sqrt(2 * math.pi))

        for ev in evals:
            dos += norm * np.exp(-0.5 * ((E - ev) / sigma)**2)

        return E, dos / len(evals)


# ---------------------------------------------------------------------------
#  Wannier Functions
# ---------------------------------------------------------------------------

class WannierProjection:
    r"""
    Maximal localisation of Wannier functions from Bloch states.

    $$|w_{n\mathbf{R}}\rangle = \frac{V}{(2\pi)^3}\int_{\text{BZ}}
      e^{-i\mathbf{k}\cdot\mathbf{R}}|\psi_{n\mathbf{k}}\rangle\,d\mathbf{k}$$

    Spread functional:
    $$\Omega = \sum_n \left[\langle w_n|r^2|w_n\rangle - |\langle w_n|r|w_n\rangle|^2\right]$$

    Minimise Ω by gauge transformation: $|\tilde{u}_{n\mathbf{k}}\rangle = \sum_m U_{mn}^{(\mathbf{k})}|u_{m\mathbf{k}}\rangle$
    """

    def __init__(self, n_bands: int, n_k: int) -> None:
        self.n_bands = n_bands
        self.n_k = n_k
        self.U = np.eye(n_bands, dtype=complex)  # gauge matrix

    def overlap_matrix(self, states_k: NDArray,
                          states_kq: NDArray) -> NDArray:
        """M_{mn}(k, k+q) = ⟨u_mk|u_nk+q⟩.

        states_k, states_kq: (n_bands, n_basis) Bloch states.
        """
        return states_k.conj() @ states_kq.T

    def spread(self, M_matrices: List[NDArray]) -> float:
        """Total spread Ω from overlap matrices (simplified).

        Ω_I = N − Σ |M_nn|²
        """
        Omega = float(self.n_bands)
        for M in M_matrices:
            for n in range(self.n_bands):
                Omega -= abs(M[n, n])**2
        return Omega

    def steepest_descent_step(self, M_matrices: List[NDArray],
                                 alpha: float = 0.1) -> NDArray:
        """One step of spread minimisation.

        dΩ/dW ∝ Σ (M − M_diag)
        """
        G = np.zeros((self.n_bands, self.n_bands), dtype=complex)
        for M in M_matrices:
            A = M - np.diag(np.diag(M))
            G += A - A.conj().T

        # Anti-Hermitian part → unitary update
        dU = -alpha * G
        self.U = self.U @ (np.eye(self.n_bands) + dU)

        # Re-orthogonalise
        Q, _ = np.linalg.qr(self.U)
        self.U = Q
        return self.U
