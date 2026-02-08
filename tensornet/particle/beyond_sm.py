"""
Beyond the Standard Model (BSM) Physics — dark matter, neutrino oscillations,
leptogenesis, grand unification, effective field theory.

Domain X.6 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Neutrino Oscillations (PMNS Matrix)
# ---------------------------------------------------------------------------

class NeutrinoOscillations:
    r"""
    Neutrino flavour oscillations via the PMNS mixing matrix.

    $$P(\nu_\alpha\to\nu_\beta) = \left|\sum_i U_{\alpha i}^*
      U_{\beta i}\exp\left(-i\frac{m_i^2 L}{2E}\right)\right|^2$$

    Three-flavour vacuum oscillation probability.

    Default mixing parameters (NuFIT 5.2):
    θ₁₂ = 33.44°, θ₂₃ = 49.2°, θ₁₃ = 8.57°, δ_CP = 197°
    Δm²₂₁ = 7.42×10⁻⁵ eV², Δm²₃₁ = 2.515×10⁻³ eV² (NO)
    """

    def __init__(self, theta12: float = 33.44, theta23: float = 49.2,
                 theta13: float = 8.57, delta_cp: float = 197.0,
                 dm21_sq: float = 7.42e-5, dm31_sq: float = 2.515e-3) -> None:
        self.th12 = math.radians(theta12)
        self.th23 = math.radians(theta23)
        self.th13 = math.radians(theta13)
        self.delta = math.radians(delta_cp)
        self.dm21_sq = dm21_sq  # eV²
        self.dm31_sq = dm31_sq  # eV²

    def pmns_matrix(self) -> NDArray:
        """Construct 3×3 PMNS mixing matrix U."""
        c12 = math.cos(self.th12)
        s12 = math.sin(self.th12)
        c23 = math.cos(self.th23)
        s23 = math.sin(self.th23)
        c13 = math.cos(self.th13)
        s13 = math.sin(self.th13)
        d = self.delta

        U = np.array([
            [c12 * c13, s12 * c13, s13 * np.exp(-1j * d)],
            [-s12 * c23 - c12 * s23 * s13 * np.exp(1j * d),
             c12 * c23 - s12 * s23 * s13 * np.exp(1j * d),
             s23 * c13],
            [s12 * s23 - c12 * c23 * s13 * np.exp(1j * d),
             -c12 * s23 - s12 * c23 * s13 * np.exp(1j * d),
             c23 * c13],
        ], dtype=complex)
        return U

    def oscillation_probability(self, alpha: int, beta: int,
                                   L_km: float, E_GeV: float) -> float:
        """P(ν_α → ν_β) for given baseline L (km) and energy E (GeV).

        $P = |Σ_i U*_{αi} U_{βi} exp(−i Δm²ᵢ₁ L / (4E))|²$
        (natural units: L in km, E in GeV → phase = 1.267 Δm² L/E)
        """
        U = self.pmns_matrix()
        dm_sq = [0.0, self.dm21_sq, self.dm31_sq]

        amplitude = 0.0 + 0.0j
        for i in range(3):
            phase = 1.267 * dm_sq[i] * L_km / E_GeV
            amplitude += np.conj(U[alpha, i]) * U[beta, i] * np.exp(-1j * phase)

        return float(abs(amplitude)**2)

    def matter_potential(self, rho_e: float) -> float:
        """MSW matter potential: V = √2 G_F N_e.

        rho_e: electron density (mol/cm³).
        Returns V in eV.
        """
        GF = 1.166e-5  # GeV⁻²
        return math.sqrt(2) * GF * rho_e * 6.022e23 * 1e-9  # convert to eV


# ---------------------------------------------------------------------------
#  Dark Matter Relic Density
# ---------------------------------------------------------------------------

class DarkMatterRelic:
    r"""
    WIMP dark matter relic abundance calculation.

    Boltzmann equation:
    $$\frac{dn}{dt} + 3Hn = -\langle\sigma v\rangle(n^2 - n_{\text{eq}}^2)$$

    Freeze-out: $\Omega h^2 \approx \frac{3\times10^{-27}\,\text{cm}^3/\text{s}}
    {\langle\sigma v\rangle}$

    Lee-Weinberg bound: $m_\chi \gtrsim$ few GeV for thermal WIMPs.
    """

    def __init__(self, mass_chi: float = 100.0,
                 sigma_v: float = 3e-26) -> None:
        """
        mass_chi: WIMP mass (GeV).
        sigma_v: thermally averaged annihilation cross section (cm³/s).
        """
        self.m_chi = mass_chi
        self.sigma_v = sigma_v

    def equilibrium_density(self, T: float) -> float:
        """Equilibrium number density n_eq(T) = g(mT/2π)^{3/2} exp(−m/T)."""
        g = 2  # degrees of freedom
        x = self.m_chi / T
        if x > 500:
            return 0.0
        return g * (self.m_chi * T / (2 * math.pi))**1.5 * math.exp(-x)

    def freeze_out_temperature(self) -> float:
        """Approximate freeze-out: x_f = m/T_f ≈ 20−25."""
        x_f = 20 + math.log(self.sigma_v / 3e-26)
        return self.m_chi / x_f

    def relic_density(self) -> float:
        """Ωh² ≈ 3×10⁻²⁷ cm³/s / ⟨σv⟩."""
        return 3e-27 / self.sigma_v

    def direct_detection_rate(self, rho_dm: float = 0.3,
                                 sigma_SI: float = 1e-46,
                                 m_target: float = 131.0) -> float:
        """Direct detection event rate (events/kg/day).

        rho_dm: local DM density (GeV/cm³).
        sigma_SI: spin-independent cross section (cm²).
        m_target: target nucleus mass (GeV).
        """
        v0 = 220e5  # cm/s (typical WIMP velocity)
        mu = self.m_chi * m_target / (self.m_chi + m_target)
        N_t = 6.022e23 / m_target * 1e3  # nuclei per kg

        rate = N_t * rho_dm / self.m_chi * sigma_SI * v0
        return rate * 86400  # per day

    def boltzmann_evolution(self, T_start: float = 1000.0,
                               T_end: float = 0.01,
                               n_steps: int = 1000) -> Tuple[NDArray, NDArray]:
        """Evolve Boltzmann equation n(T) through freeze-out."""
        x_start = self.m_chi / T_start
        x_end = self.m_chi / T_end
        x = np.linspace(x_start, x_end, n_steps)
        dx = x[1] - x[0]

        Y = np.zeros(n_steps)  # Y = n/s (comoving density)
        Y[0] = self.equilibrium_density(T_start) / T_start**3

        for i in range(1, n_steps):
            T = self.m_chi / x[i]
            Y_eq = self.equilibrium_density(T) / T**3
            # Simplified: dY/dx = −λ/x² (Y² − Y_eq²)
            lam = self.sigma_v * self.m_chi * math.sqrt(math.pi / 45) * 1e10
            dYdx = -lam / x[i]**2 * (Y[i - 1]**2 - Y_eq**2)
            Y[i] = max(Y[i - 1] + dx * dYdx, Y_eq * 0.01)

        return x, Y


# ---------------------------------------------------------------------------
#  Grand Unification (GUT) Running Couplings
# ---------------------------------------------------------------------------

class GUTRunningCouplings:
    r"""
    Renormalisation group evolution of SM gauge couplings toward unification.

    One-loop RGE:
    $$\frac{1}{\alpha_i(\mu)} = \frac{1}{\alpha_i(M_Z)}
      - \frac{b_i}{2\pi}\ln\frac{\mu}{M_Z}$$

    SM coefficients: $b_1 = 41/10$, $b_2 = -19/6$, $b_3 = -7$.
    MSSM: $b_1 = 33/5$, $b_2 = 1$, $b_3 = -3$.
    """

    MZ = 91.1876  # GeV

    # SM values at M_Z
    ALPHA1_MZ = 1 / 59.0  # U(1)_Y (GUT normalised: 5/3 × α_em/cos²θ_W)
    ALPHA2_MZ = 1 / 29.6  # SU(2)_L
    ALPHA3_MZ = 0.118      # SU(3)_c

    SM_B = [41 / 10, -19 / 6, -7]
    MSSM_B = [33 / 5, 1, -3]

    def __init__(self, model: str = 'sm') -> None:
        self.b = self.MSSM_B if model.lower() == 'mssm' else self.SM_B

    def alpha_inverse(self, i: int, mu: float) -> float:
        """1/α_i(μ) at scale μ (GeV)."""
        alpha_mz = [self.ALPHA1_MZ, self.ALPHA2_MZ, self.ALPHA3_MZ][i]
        return (1 / alpha_mz - self.b[i] / (2 * math.pi)
                * math.log(mu / self.MZ))

    def running_couplings(self, mu_range: NDArray) -> NDArray:
        """Compute 1/α_i for array of scales.

        Returns (n_scales, 3) array.
        """
        result = np.zeros((len(mu_range), 3))
        for j, mu in enumerate(mu_range):
            for i in range(3):
                result[j, i] = self.alpha_inverse(i, mu)
        return result

    def unification_scale(self) -> float:
        """Estimate GUT scale where α₁ = α₂.

        1/α₁(M_GUT) = 1/α₂(M_GUT)
        → ln(M_GUT/M_Z) = 2π(1/α₁−1/α₂)/(b₂−b₁)
        """
        diff_alpha = 1 / self.ALPHA1_MZ - 1 / self.ALPHA2_MZ
        diff_b = self.b[1] - self.b[0]
        if abs(diff_b) < 1e-10:
            return 1e16
        log_ratio = 2 * math.pi * diff_alpha / diff_b
        return self.MZ * math.exp(log_ratio)

    def proton_lifetime_estimate(self, M_GUT: Optional[float] = None,
                                    alpha_GUT: float = 1 / 40) -> float:
        """Proton lifetime τ_p ∝ M_GUT⁴ / (α_GUT² m_p⁵).

        Returns log₁₀(τ/years).
        """
        if M_GUT is None:
            M_GUT = self.unification_scale()
        m_p = 0.938  # GeV
        tau_seconds = M_GUT**4 / (alpha_GUT**2 * m_p**5) * 1e-24
        tau_years = tau_seconds / 3.156e7
        return math.log10(max(tau_years, 1.0))


# ---------------------------------------------------------------------------
#  Effective Field Theory (EFT) for BSM
# ---------------------------------------------------------------------------

class SMEFTOperators:
    r"""
    Standard Model Effective Field Theory (SMEFT) dimension-6 operators.

    $$\mathcal{L}_{\text{SMEFT}} = \mathcal{L}_{\text{SM}}
      + \sum_i \frac{c_i}{\Lambda^2}\mathcal{O}_i^{(6)} + \ldots$$

    Warsaw basis: 59 independent operators (baryon-number conserving).

    Wilson coefficient running:
    $$\mu\frac{dc_i}{d\mu} = \gamma_{ij}c_j$$
    """

    def __init__(self, Lambda: float = 1000.0) -> None:
        """Lambda: new physics scale (GeV)."""
        self.Lambda = Lambda
        self.wilson_coefficients: Dict[str, float] = {}

    def set_coefficient(self, operator: str, value: float) -> None:
        """Set Wilson coefficient c_i for operator O_i."""
        self.wilson_coefficients[operator] = value

    def operator_contribution(self, operator: str, energy: float) -> float:
        """Contribution of operator O_i at energy scale E.

        σ_BSM/σ_SM ~ (c_i E² / Λ²)
        """
        c = self.wilson_coefficients.get(operator, 0.0)
        return c * energy**2 / self.Lambda**2

    def oblique_parameters(self, c_T: float = 0.0,
                              c_WB: float = 0.0) -> Dict[str, float]:
        """Peskin-Takeuchi S, T, U oblique parameters from dim-6 operators.

        Simplified mapping from Warsaw basis.
        """
        v = 246.0  # Higgs VEV (GeV)
        S = 16 * math.pi * v**2 / self.Lambda**2 * c_WB
        T = -8 * math.pi * v**2 / self.Lambda**2 * c_T / (137 * 0.231)
        return {'S': S, 'T': T, 'U': 0.0}

    def run_coefficients(self, mu_from: float, mu_to: float,
                            gamma_matrix: NDArray) -> Dict[str, float]:
        """RG-evolve Wilson coefficients: c(μ_to) = exp(γ ln(μ_to/μ_from)) c(μ_from).

        Simplified: one-loop leading-log.
        """
        ops = list(self.wilson_coefficients.keys())
        n = len(ops)
        c_vec = np.array([self.wilson_coefficients[op] for op in ops])

        log_ratio = math.log(mu_to / mu_from)
        c_new = c_vec + gamma_matrix[:n, :n] @ c_vec * log_ratio / (16 * math.pi**2)

        return {ops[i]: float(c_new[i]) for i in range(n)}
