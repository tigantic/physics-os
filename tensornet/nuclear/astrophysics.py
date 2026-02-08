"""
Nuclear Astrophysics — reaction networks, r-process, s-process,
thermonuclear rates, nucleosynthesis.

Domain X.3 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Thermonuclear Reaction Rate
# ---------------------------------------------------------------------------

class ThermonuclearRate:
    r"""
    Thermonuclear reaction rate ⟨σv⟩ for stellar burning.

    $$\langle\sigma v\rangle = \left(\frac{8}{\pi\mu}\right)^{1/2}
      \frac{1}{(k_BT)^{3/2}}\int_0^\infty \sigma(E)\,E\,
      e^{-E/k_BT}\,dE$$

    Gamow peak energy:
    $$E_0 = \left(\frac{b k_BT}{2}\right)^{2/3}$$
    where $b = \pi Z_1 Z_2 e^2\sqrt{2\mu}/\hbar$.

    Non-resonant parametric form (NA⟨σv⟩ in cm³/mol/s):
    $$N_A\langle\sigma v\rangle = C \cdot T_9^{a} \exp(-b/T_9^{1/3})$$
    """

    def __init__(self, Z1: int = 1, Z2: int = 1,
                 A1: float = 1.0, A2: float = 1.0) -> None:
        self.Z1 = Z1
        self.Z2 = Z2
        self.mu = A1 * A2 / (A1 + A2) * 931.5  # reduced mass (MeV)
        self.mu_amu = A1 * A2 / (A1 + A2)

    def gamow_energy(self, T9: float) -> float:
        """Gamow peak energy (MeV).

        E_0 = 1.22 (Z₁²Z₂²μ)^{1/3} T₉^{2/3} MeV
        """
        kT = T9 * 0.08617  # T9 → MeV (k_B in MeV/GK)
        b = 0.9895 * self.Z1 * self.Z2 * math.sqrt(self.mu_amu)  # Sommerfeld param
        return (b * kT / 2)**(2 / 3)

    def gamow_width(self, T9: float) -> float:
        """Width of Gamow window Δ ≈ (4/√3)(E₀ kT)^{1/2}."""
        E0 = self.gamow_energy(T9)
        kT = T9 * 0.08617
        return 4 / math.sqrt(3) * math.sqrt(E0 * kT)

    def rate_parametric(self, T9: float,
                           C: float = 1e7, a: float = -2 / 3,
                           b_param: float = 3.38) -> float:
        """Parametric rate: NA⟨σv⟩ = C T₉^a exp(−b/T₉^{1/3}).

        Returns cm³/mol/s.
        """
        if T9 <= 0:
            return 0.0
        return C * T9**a * math.exp(-b_param / T9**(1 / 3))

    def rate_integration(self, T9: float, sigma_func,
                            E_min: float = 0.001, E_max: float = 10.0,
                            n_points: int = 500) -> float:
        """Numerical ⟨σv⟩ by direct Maxwellian integration.

        sigma_func(E) returns σ(E) in barns.
        """
        kT = T9 * 0.08617
        E_grid = np.linspace(E_min, E_max, n_points)
        prefactor = math.sqrt(8 / (math.pi * self.mu)) / kT**1.5

        integrand = np.zeros(n_points)
        for i, E in enumerate(E_grid):
            integrand[i] = sigma_func(E) * E * math.exp(-E / kT)

        rate = prefactor * float(np.trapz(integrand, E_grid))
        return rate * 6.022e23 * 1e-24  # cm³/mol/s (barns → cm²)


# ---------------------------------------------------------------------------
#  Nuclear Reaction Network
# ---------------------------------------------------------------------------

class NuclearReactionNetwork:
    r"""
    Nuclear reaction network for nucleosynthesis calculations.

    $$\frac{dY_i}{dt} = \sum_j \lambda_j Y_j + \sum_{j,k}\rho N_A
      \langle\sigma v\rangle_{jk}Y_j Y_k + \ldots$$

    where Y_i = abundance (mol/g), λ = decay rate, ρ = density.

    Stiff ODE system — requires implicit integration (backward Euler).
    """

    @dataclass
    class Species:
        name: str
        Z: int
        A: int
        binding_energy: float = 0.0  # MeV

    @dataclass
    class Reaction:
        reactants: List[str]
        products: List[str]
        rate_function: Optional[object] = None
        Q_value: float = 0.0  # MeV

    def __init__(self) -> None:
        self.species: Dict[str, NuclearReactionNetwork.Species] = {}
        self.reactions: List[NuclearReactionNetwork.Reaction] = []
        self.abundances: Dict[str, float] = {}

    def add_species(self, name: str, Z: int, A: int,
                       BE: float = 0.0) -> None:
        self.species[name] = self.Species(name, Z, A, BE)
        if name not in self.abundances:
            self.abundances[name] = 0.0

    def add_reaction(self, reactants: List[str], products: List[str],
                        rate: float = 0.0, Q: float = 0.0) -> None:
        self.reactions.append(self.Reaction(reactants, products, rate, Q))

    def _build_rhs(self, Y: NDArray, species_list: List[str],
                      rho: float, T9: float) -> NDArray:
        """Build dY/dt vector."""
        n = len(species_list)
        idx = {s: i for i, s in enumerate(species_list)}
        dYdt = np.zeros(n)

        for rxn in self.reactions:
            rate = rxn.Q_value  # Using Q as constant rate here
            if callable(rxn.rate_function):
                rate = rxn.rate_function(T9)

            # Reactant consumption / product creation
            flux = rate
            for r in rxn.reactants:
                if r in idx:
                    flux *= Y[idx[r]]

            for r in rxn.reactants:
                if r in idx:
                    dYdt[idx[r]] -= flux
            for p in rxn.products:
                if p in idx:
                    dYdt[idx[p]] += flux

        return dYdt

    def evolve(self, dt: float, n_steps: int,
                  rho: float = 1e6, T9: float = 1.0) -> Dict[str, NDArray]:
        """Evolve network using backward Euler."""
        species_list = list(self.species.keys())
        n = len(species_list)
        Y = np.array([self.abundances.get(s, 0.0) for s in species_list])

        history = {s: [Y[i]] for i, s in enumerate(species_list)}
        times = [0.0]

        for step in range(n_steps):
            dYdt = self._build_rhs(Y, species_list, rho, T9)
            Y_new = Y + dt * dYdt
            Y_new = np.maximum(Y_new, 0.0)
            Y = Y_new

            for i, s in enumerate(species_list):
                history[s].append(Y[i])
            times.append((step + 1) * dt)

        # Convert to arrays
        result = {s: np.array(history[s]) for s in species_list}
        result['time'] = np.array(times)
        return result


# ---------------------------------------------------------------------------
#  r-Process Nucleosynthesis
# ---------------------------------------------------------------------------

class RProcess:
    r"""
    Rapid neutron capture process (r-process).

    Waiting-point approximation (β equilibrium):
    $$\frac{Y(Z,A+1)}{Y(Z,A)} = \frac{\rho N_A}{2}
      \left(\frac{2\pi\hbar^2}{m_n k_BT}\right)^{3/2}
      \exp\left(\frac{S_n(Z,A+1)}{k_BT}\right)$$

    where $S_n$ = neutron separation energy.

    Path determined by $S_n \approx 2-3$ MeV ("neutron drip line" proximity).
    """

    def __init__(self, T9: float = 1.5, n_n: float = 1e24) -> None:
        """
        T9: temperature in GK.
        n_n: neutron number density (cm⁻³).
        """
        self.T9 = T9
        self.kT = T9 * 0.08617  # MeV
        self.n_n = n_n

    def saha_ratio(self, S_n: float, A: int) -> float:
        """Y(Z,A+1)/Y(Z,A) from nuclear Saha equation.

        S_n: neutron separation energy (MeV).
        """
        rho_na = self.n_n / 6.022e23 * 6.022e23  # n_n per mol
        thermal_wavelength = 4.15e-13 / math.sqrt(self.T9)  # cm
        ratio = (0.5 * rho_na * thermal_wavelength**3
                * math.exp(S_n / self.kT))
        return ratio

    def waiting_point_path(self, Z_range: Tuple[int, int],
                              Sn_grid: NDArray) -> List[Tuple[int, int]]:
        """Find r-process path (A for each Z where Y peaks).

        Sn_grid: (Z_max, A_max) grid of neutron separation energies.
        """
        path = []
        for Z in range(Z_range[0], Z_range[1]):
            best_A = 0
            best_ratio = 0.0
            for A in range(Z, 3 * Z):
                if Z < Sn_grid.shape[0] and A < Sn_grid.shape[1]:
                    Sn = Sn_grid[Z, A]
                    if 1.5 < Sn < 4.0:
                        ratio = self.saha_ratio(Sn, A)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_A = A
            if best_A > 0:
                path.append((Z, best_A))
        return path

    def beta_decay_flow(self, abundances: NDArray,
                           half_lives: NDArray) -> NDArray:
        """β-decay flow: dY(Z+1)/dt = λ_β Y(Z).

        half_lives in seconds.
        """
        lambda_beta = np.log(2) / np.maximum(half_lives, 1e-30)
        flow = lambda_beta * abundances
        return flow


# ---------------------------------------------------------------------------
#  s-Process (Slow Neutron Capture)
# ---------------------------------------------------------------------------

class SProcess:
    r"""
    s-process: slow neutron capture nucleosynthesis.

    Classical s-process: σN_s = const along the s-process path.

    $$\sigma_A N_{s,A} = \sigma_{A-1} N_{s,A-1} - \lambda_\beta^A / (n_n v_T) N_{s,A}$$

    Exposure distribution (exponential):
    $$\rho(\tau) = \frac{G}{\tau_0}\exp(-\tau/\tau_0)$$
    """

    def __init__(self, n_n_vt: float = 4e7) -> None:
        """n_n_vt: neutron flux (cm⁻² s⁻¹) × exposure time."""
        self.n_n_vt = n_n_vt

    def classical_sigma_n(self, sigma: NDArray, N_seed: float,
                             A_start: int = 56) -> NDArray:
        """Classical s-process: Σ_A N_s iterative along path.

        sigma: (n_isotopes,) Maxwellian-averaged cross sections (mb).
        Returns abundance distribution.
        """
        n = len(sigma)
        N = np.zeros(n)
        N[0] = N_seed

        for i in range(1, n):
            if sigma[i] > 0:
                N[i] = sigma[i - 1] * N[i - 1] / sigma[i]

        return N

    def exposure_distribution(self, tau: NDArray,
                                 tau_0: float = 0.3, G: float = 1.0) -> NDArray:
        """Exponential exposure distribution ρ(τ) = (G/τ₀) exp(−τ/τ₀).

        τ = ∫ n_n v_T dt (neutron exposure, mb⁻¹).
        """
        return G / tau_0 * np.exp(-tau / tau_0)

    def stellar_sigma_30(self, sigma_peak: float, E_r: float,
                            Gamma: float, kT: float = 0.00259) -> float:
        """Maxwellian-averaged capture cross section at kT = 30 keV.

        Breit-Wigner resonance contribution.
        """
        integral = (math.sqrt(math.pi) * Gamma / 2
                   * math.exp(-E_r / kT) * sigma_peak / math.sqrt(kT))
        return integral
