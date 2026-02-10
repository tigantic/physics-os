"""
Perturbative QFT: Feynman diagram evaluation, dimensional regularisation,
MS-bar renormalization, running coupling.

Upgrades domain X.5 from stub to full 1-loop evaluator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

EULER_MASCHERONI: float = 0.5772156649015329


class LoopOrder(Enum):
    TREE = 0
    ONE_LOOP = 1
    TWO_LOOP = 2


# ---------------------------------------------------------------------------
#  Feynman Propagator / Vertex building blocks
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Propagator:
    """Feynman propagator in d dimensions.

    Scalar: 1 / (k² + m²)
    Fermion: (γ·k + m) / (k² + m²)  (Euclidean)
    Gauge (Feynman): δ_μν / k²
    """
    particle_type: str   # "scalar", "fermion", "gauge"
    mass: float = 0.0
    momentum_label: str = "k"


@dataclass(frozen=True)
class Vertex:
    """Interaction vertex."""
    vertex_type: str      # e.g. "QED", "phi4", "QCD"
    coupling: float = 1.0
    n_legs: int = 3


@dataclass
class FeynmanDiagram:
    """
    Representation of a Feynman diagram for perturbative evaluation.

    Stores topology (propagators, vertices, external momenta) and
    evaluates the corresponding loop integral via Feynman parameterisation
    + dimensional regularisation.
    """
    propagators: List[Propagator] = field(default_factory=list)
    vertices: List[Vertex] = field(default_factory=list)
    n_loops: int = 1
    external_momenta: List[NDArray[np.float64]] = field(default_factory=list)

    def symmetry_factor(self) -> float:
        """Estimate symmetry factor from topology."""
        n_prop = len(self.propagators)
        n_vert = len(self.vertices)
        # For φ⁴: 1-loop self-energy → S = 2
        # For QED vertex: S = 1
        if n_prop == 2 and n_vert == 2:
            return 2.0
        return 1.0

    def superficial_degree_of_divergence(self, d: float = 4.0) -> float:
        """ω = dL - 2I_B - I_F where L=loops, I_B=boson, I_F=fermion propagators."""
        n_boson = sum(1 for p in self.propagators if p.particle_type in ("scalar", "gauge"))
        n_fermion = sum(1 for p in self.propagators if p.particle_type == "fermion")
        return d * self.n_loops - 2 * n_boson - n_fermion

    def evaluate_scalar_bubble(self, p_sq: float, m1: float, m2: float,
                                  d: float = 4.0) -> complex:
        """
        1-loop scalar bubble integral B₀(p², m₁², m₂²) in d dimensions
        using Feynman parameterisation.

        B₀ = Γ(2-d/2) ∫₀¹ dx [m₁²x + m₂²(1-x) - p²x(1-x)]^{d/2-2}

        In d = 4-2ε:  B₀ = 1/ε - γ + ln(4π) - ∫₀¹ dx ln[Δ/μ²] + O(ε)
        where Δ = m₁²x + m₂²(1-x) - p²x(1-x).
        """
        eps = (4.0 - d) / 2.0
        n_x = 1000
        x_vals = np.linspace(1e-10, 1.0 - 1e-10, n_x)

        m1_sq = m1**2
        m2_sq = m2**2

        if abs(eps) < 1e-10:
            # d=4: return finite part (in MS-bar: subtract 1/ε pole)
            integral = 0.0
            dx = x_vals[1] - x_vals[0]
            for x in x_vals:
                delta = m1_sq * x + m2_sq * (1.0 - x) - p_sq * x * (1.0 - x)
                if delta > 0:
                    integral += math.log(max(delta, 1e-30)) * dx
                else:
                    integral += math.log(max(abs(delta), 1e-30)) * dx

            # Finite part: -γ + ln(4π) - integral
            finite = -EULER_MASCHERONI + math.log(4.0 * math.pi) - integral
            return complex(finite, 0)
        else:
            # General d
            gamma_factor = math.gamma(2.0 - d / 2.0)
            integral = 0.0
            dx = x_vals[1] - x_vals[0]
            for x in x_vals:
                delta = m1_sq * x + m2_sq * (1.0 - x) - p_sq * x * (1.0 - x)
                if delta > 0:
                    integral += delta**(d / 2.0 - 2.0) * dx
                else:
                    integral += abs(delta)**(d / 2.0 - 2.0) * dx

            return complex(gamma_factor * integral, 0)


# ---------------------------------------------------------------------------
#  Dimensional Regularisation
# ---------------------------------------------------------------------------

class DimensionalRegularisation:
    r"""
    Dimensional regularisation toolkit for loop integrals.

    Standard results in $d = 4 - 2\varepsilon$ dimensions:

    $$\int\frac{d^dk}{(2\pi)^d}\frac{1}{(k^2+\Delta)^n}
      = \frac{\Gamma(n-d/2)}{(4\pi)^{d/2}\Gamma(n)}\Delta^{d/2-n}$$
    """

    def __init__(self, d: float = 4.0, mu: float = 1.0) -> None:
        """
        Parameters
        ----------
        d : Space-time dimension (typically 4 - 2ε).
        mu : Renormalization scale μ.
        """
        self.d = d
        self.mu = mu

    @property
    def epsilon(self) -> float:
        return (4.0 - self.d) / 2.0

    def scalar_tadpole(self, m_sq: float) -> complex:
        """A₀(m²) = ∫ d^dk / (k² + m²)."""
        eps = self.epsilon
        d = self.d
        if abs(eps) < 1e-12:
            # d→4 limit: A₀ = m²(1/ε - γ + 1 + ln(4πμ²/m²))
            if m_sq < 1e-30:
                return complex(0, 0)
            return complex(m_sq * (1.0 + math.log(4.0 * math.pi * self.mu**2 / m_sq)
                                    - EULER_MASCHERONI), 0)
        prefactor = 1.0 / (4.0 * math.pi)**(d / 2.0)
        gamma_factor = math.gamma(1.0 - d / 2.0) / math.gamma(1.0)
        return complex(prefactor * gamma_factor * m_sq**(d / 2.0 - 1.0), 0)

    def scalar_bubble(self, p_sq: float, m1: float, m2: float) -> complex:
        """B₀(p², m₁², m₂²) via Feynman parameterisation."""
        diagram = FeynmanDiagram()
        return diagram.evaluate_scalar_bubble(p_sq, m1, m2, self.d)

    def scalar_triangle(self, p1_sq: float, p2_sq: float, p3_sq: float,
                          m1: float, m2: float, m3: float) -> complex:
        """
        C₀(p₁², p₂², p₃², m₁², m₂², m₃²) — scalar triangle.
        Feynman parameter integral with 2D integration.
        """
        m1s, m2s, m3s = m1**2, m2**2, m3**2
        n_pts = 200
        x_vals = np.linspace(1e-8, 1.0 - 1e-8, n_pts)
        dx = x_vals[1] - x_vals[0]

        integral = 0.0
        for x in x_vals:
            y_max = 1.0 - x
            if y_max < 1e-8:
                continue
            y_vals = np.linspace(1e-8, y_max - 1e-8, max(int(n_pts * y_max), 10))
            dy = y_vals[1] - y_vals[0] if len(y_vals) > 1 else 0
            for y in y_vals:
                z = 1.0 - x - y
                delta = (m1s * x + m2s * y + m3s * z
                         - p1_sq * x * y - p2_sq * y * z - p3_sq * x * z)
                if delta > 1e-30:
                    integral += (1.0 / delta) * dx * dy

        # Prefactor: Γ(3-d/2) / Γ(3) = Γ(1+ε) / 2 ≈ 1/2
        d = self.d
        return complex(math.gamma(3.0 - d / 2.0) / math.gamma(3.0) * integral, 0)


# ---------------------------------------------------------------------------
#  MS-bar Renormalization
# ---------------------------------------------------------------------------

class MSBarRenormalisation:
    r"""
    Modified Minimal Subtraction ($\overline{\text{MS}}$) scheme.

    In $d = 4 - 2\varepsilon$, subtract poles $1/\varepsilon^n$ plus
    $\ln(4\pi) - \gamma_E$ at each order.

    Counterterm: $\delta Z = -\frac{a}{\varepsilon} - \frac{b}{\varepsilon^2} + \ldots$
    """

    def __init__(self, mu: float = 91.187) -> None:
        """
        Parameters
        ----------
        mu : Renormalization scale in GeV (default: M_Z).
        """
        self.mu = mu
        self.counterterms: Dict[str, float] = {}

    def subtract_pole(self, coefficient: float, eps: float,
                        order: int = 1) -> float:
        """
        MS-bar subtraction: remove 1/ε^order pole.

        Returns finite remainder after subtracting
        coefficient × (1/ε^order + (ln4π - γ)/ε^{order-1} + ...).
        """
        if abs(eps) < 1e-15:
            raise ValueError("Cannot subtract pole at ε = 0")

        # The pole is coefficient / ε^order
        pole_value = coefficient / eps**order
        # MS-bar: also subtract ln(4π) - γ at each order
        ms_bar_constant = math.log(4.0 * math.pi) - EULER_MASCHERONI

        subtraction = pole_value
        if order >= 1:
            subtraction += coefficient * ms_bar_constant / eps**(order - 1) if order > 1 else 0

        self.counterterms[f"1/eps^{order}"] = self.counterterms.get(f"1/eps^{order}", 0) + coefficient

        return -subtraction  # counterterm is negative of the pole

    def qed_vertex_counterterm_z1(self, alpha: float) -> float:
        """QED vertex renormalization Z₁ at 1-loop.

        δZ₁ = -α/(4π) × 1/ε × (UV divergent part)
        Ward identity: Z₁ = Z₂ in QED.
        """
        return -alpha / (4.0 * math.pi)

    def qed_wavefunction_counterterm_z2(self, alpha: float) -> float:
        """QED electron self-energy renormalization Z₂ at 1-loop.

        δZ₂ = -α/(4πε) in Feynman gauge (ξ=1).
        """
        return -alpha / (4.0 * math.pi)

    def qed_mass_counterterm_zm(self, alpha: float) -> float:
        """QED mass renormalization δm/m = -3α/(4π) × 1/ε."""
        return -3.0 * alpha / (4.0 * math.pi)

    def qed_vacuum_polarisation_z3(self, alpha: float, n_f: int = 1) -> float:
        """QED photon self-energy (vacuum polarisation) Z₃.

        δZ₃ = -α/(3π) × N_f/ε where N_f = number of flavours.
        """
        return -alpha * n_f / (3.0 * math.pi)


# ---------------------------------------------------------------------------
#  Running Coupling
# ---------------------------------------------------------------------------

@dataclass
class BetaFunctionCoefficients:
    """β-function coefficients for gauge coupling evolution.

    β(g) = -b₀ g³/(16π²) - b₁ g⁵/(16π²)² - ...

    For QCD with N_f flavours, N_c = 3:
    b₀ = (11Nc - 2Nf) / 3
    b₁ = (34Nc² - 10NcNf - 3(Nc²-1)Nf/Nc) / 3
    """
    b0: float
    b1: float
    b2: float = 0.0


class RunningCoupling:
    r"""
    Perturbative running of gauge couplings.

    $$\alpha_s(\mu) = \frac{\alpha_s(\mu_0)}{1 + \frac{b_0 \alpha_s(\mu_0)}{2\pi}\ln(\mu/\mu_0)}$$

    Implements:
    - QED running coupling (vacuum polarisation)
    - QCD running coupling (asymptotic freedom)
    - Threshold matching at quark masses
    """

    def __init__(self, theory: str = "QCD", n_f: int = 5,
                 n_c: int = 3) -> None:
        """
        Parameters
        ----------
        theory : "QCD" or "QED".
        n_f : Number of active flavours.
        n_c : Number of colours (3 for QCD, N/A for QED).
        """
        self.theory = theory
        self.n_f = n_f
        self.n_c = n_c
        self.beta_coeffs = self._compute_beta_coefficients()

    def _compute_beta_coefficients(self) -> BetaFunctionCoefficients:
        if self.theory == "QCD":
            b0 = (11.0 * self.n_c - 2.0 * self.n_f) / 3.0
            cf = (self.n_c**2 - 1.0) / (2.0 * self.n_c)
            b1 = (34.0 * self.n_c**2 / 3.0
                   - 10.0 * self.n_c * self.n_f / 3.0
                   - 2.0 * cf * self.n_f)
            return BetaFunctionCoefficients(b0=b0, b1=b1)
        elif self.theory == "QED":
            # QED: b₀ = -4N_f/3 (note sign convention: β > 0 for QED)
            b0 = -4.0 * self.n_f / 3.0
            b1 = -4.0 * self.n_f  # 2-loop
            return BetaFunctionCoefficients(b0=b0, b1=b1)
        else:
            raise ValueError(f"Unknown theory: {self.theory}")

    def alpha_s_1loop(self, mu: float, mu0: float,
                        alpha_s_mu0: float) -> float:
        """1-loop running: α_s(μ)."""
        b0 = self.beta_coeffs.b0
        ratio = math.log(mu / mu0) if mu > 0 and mu0 > 0 else 0.0
        denom = 1.0 + b0 * alpha_s_mu0 / (2.0 * math.pi) * ratio
        if denom <= 0:
            raise ValueError(f"Landau pole encountered at μ = {mu}")
        return alpha_s_mu0 / denom

    def alpha_s_2loop(self, mu: float, mu0: float,
                        alpha_s_mu0: float) -> float:
        """2-loop running via iterative solution."""
        b0 = self.beta_coeffs.b0
        b1 = self.beta_coeffs.b1
        t = math.log(mu / mu0) if mu > 0 and mu0 > 0 else 0.0

        # Iterative: start from 1-loop
        alpha = self.alpha_s_1loop(mu, mu0, alpha_s_mu0)

        # Refine via implicit equation
        for _ in range(50):
            beta_val = -(b0 * alpha**2 / (2.0 * math.pi)
                          + b1 * alpha**3 / (8.0 * math.pi**2))
            alpha_new = alpha_s_mu0 + beta_val * t
            if alpha_new <= 0:
                break
            if abs(alpha_new - alpha) < 1e-12:
                break
            alpha = 0.5 * (alpha + alpha_new)

        return alpha

    def lambda_qcd(self, mu0: float, alpha_s_mu0: float) -> float:
        r"""
        Compute Λ_QCD from reference scale.

        $$\Lambda_{\overline{\text{MS}}} = \mu_0\exp\left(-\frac{2\pi}{b_0\alpha_s(\mu_0)}\right)$$
        """
        b0 = self.beta_coeffs.b0
        if b0 <= 0 or alpha_s_mu0 <= 0:
            raise ValueError("Λ_QCD only defined for asymptotically free theories")
        return mu0 * math.exp(-2.0 * math.pi / (b0 * alpha_s_mu0))

    def threshold_matching(self, alpha_below: float,
                             m_quark: float, mu: float) -> float:
        """
        Match coupling across flavour threshold at μ = m_quark.

        At 1-loop: α_s^{(nf)}(μ) = α_s^{(nf-1)}(μ) × [1 + O(α²)]
        Leading order: continuous matching.
        """
        # At leading order, α_s is continuous across threshold
        return alpha_below

    def run_qcd_with_thresholds(self, mu: float,
                                  alpha_mz: float = 0.1179,
                                  mz: float = 91.187) -> float:
        """
        Run QCD coupling from M_Z down to μ, crossing flavour thresholds.

        Standard thresholds: m_b ≈ 4.18 GeV, m_c ≈ 1.27 GeV.
        """
        m_b = 4.18
        m_c = 1.27

        if mu >= m_b:
            # n_f = 5
            runner = RunningCoupling("QCD", n_f=5)
            return runner.alpha_s_2loop(mu, mz, alpha_mz)
        elif mu >= m_c:
            # Run to m_b with n_f = 5, then below with n_f = 4
            runner5 = RunningCoupling("QCD", n_f=5)
            alpha_mb = runner5.alpha_s_2loop(m_b, mz, alpha_mz)
            alpha_mb_below = self.threshold_matching(alpha_mb, m_b, m_b)
            runner4 = RunningCoupling("QCD", n_f=4)
            return runner4.alpha_s_2loop(mu, m_b, alpha_mb_below)
        else:
            runner5 = RunningCoupling("QCD", n_f=5)
            alpha_mb = runner5.alpha_s_2loop(m_b, mz, alpha_mz)
            alpha_mb_below = self.threshold_matching(alpha_mb, m_b, m_b)
            runner4 = RunningCoupling("QCD", n_f=4)
            alpha_mc = runner4.alpha_s_2loop(m_c, m_b, alpha_mb_below)
            alpha_mc_below = self.threshold_matching(alpha_mc, m_c, m_c)
            runner3 = RunningCoupling("QCD", n_f=3)
            return runner3.alpha_s_2loop(mu, m_c, alpha_mc_below)


# ---------------------------------------------------------------------------
#  QCD Splitting Functions (LO)
# ---------------------------------------------------------------------------

class SplittingFunctions:
    r"""
    Leading-order Altarelli-Parisi splitting functions for parton showers.

    :math:`P_{qq}(z) = C_F \frac{1+z^2}{1-z}`

    :math:`P_{gq}(z) = C_F \frac{1+(1-z)^2}{z}`

    :math:`P_{qg}(z) = T_R [z^2 + (1-z)^2]`

    :math:`P_{gg}(z) = 2C_A \left[\frac{z}{1-z} + \frac{1-z}{z} + z(1-z)\right]`

    References:
        [1] Altarelli & Parisi, Nucl. Phys. B 126, 298 (1977).
    """
    C_F: float = 4.0 / 3.0
    C_A: float = 3.0
    T_R: float = 0.5
    N_F: int = 5

    @classmethod
    def P_qq(cls, z: float) -> float:
        """q → q g splitting."""
        if z >= 1.0 or z <= 0.0:
            return 0.0
        return cls.C_F * (1.0 + z ** 2) / (1.0 - z)

    @classmethod
    def P_gq(cls, z: float) -> float:
        """q → g q splitting."""
        if z >= 1.0 or z <= 0.0:
            return 0.0
        return cls.C_F * (1.0 + (1.0 - z) ** 2) / z

    @classmethod
    def P_qg(cls, z: float) -> float:
        """g → q q̄ splitting."""
        if z >= 1.0 or z <= 0.0:
            return 0.0
        return cls.T_R * (z ** 2 + (1.0 - z) ** 2)

    @classmethod
    def P_gg(cls, z: float) -> float:
        """g → g g splitting."""
        if z >= 1.0 or z <= 0.0:
            return 0.0
        return 2.0 * cls.C_A * (z / (1.0 - z) + (1.0 - z) / z + z * (1.0 - z))


# ---------------------------------------------------------------------------
#  Sudakov Form Factor
# ---------------------------------------------------------------------------

class SudakovFormFactor:
    r"""
    Sudakov form factor for parton-shower no-branching probability.

    .. math::
        \Delta(t, t_0) = \exp\left(-\int_{t_0}^{t} \frac{dt'}{t'}
            \int dz\,\frac{\alpha_s(t')}{2\pi} P(z)\right)

    Parameters:
        t_cut: IR cutoff scale [GeV²].
    """

    def __init__(self, t_cut: float = 1.0) -> None:
        self.t_cut = t_cut

    def no_branch_prob(
        self,
        t_start: float,
        t_end: float,
        alpha_s: float,
        split_fn: str = "qq",
        n_sample: int = 100,
    ) -> float:
        """
        Estimate Sudakov form factor Δ(t_start, t_end) by numerical quadrature.

        Parameters:
            t_start: Starting virtuality [GeV²].
            t_end: Ending virtuality [GeV²] (t_end < t_start).
            alpha_s: Fixed coupling value.
            split_fn: One of "qq", "gq", "qg", "gg".

        Returns:
            No-branching probability.
        """
        if t_end >= t_start:
            return 1.0

        P_func = {
            "qq": SplittingFunctions.P_qq,
            "gq": SplittingFunctions.P_gq,
            "qg": SplittingFunctions.P_qg,
            "gg": SplittingFunctions.P_gg,
        }[split_fn]

        integral = 0.0
        dt = (math.log(t_start) - math.log(max(t_end, self.t_cut))) / n_sample
        z_min, z_max = 0.01, 0.99
        dz = (z_max - z_min) / n_sample

        for i in range(n_sample):
            ln_t = math.log(t_start) - (i + 0.5) * dt
            z_integral = sum(
                P_func(z_min + (j + 0.5) * dz) * dz
                for j in range(n_sample)
            )
            integral += (alpha_s / (2.0 * math.pi)) * z_integral * dt

        return math.exp(-integral)


# ---------------------------------------------------------------------------
#  Parton Shower
# ---------------------------------------------------------------------------

@dataclass
class Parton:
    """A single parton in the shower."""
    pid: int          # PDG ID (1-6: quarks, 21: gluon)
    px: float = 0.0
    py: float = 0.0
    pz: float = 0.0
    E: float = 0.0
    t: float = 0.0    # virtuality scale

    @property
    def pt(self) -> float:
        return math.sqrt(self.px ** 2 + self.py ** 2)

    @property
    def mass_sq(self) -> float:
        return max(0.0, self.E ** 2 - self.px ** 2 - self.py ** 2 - self.pz ** 2)

    def is_gluon(self) -> bool:
        return self.pid == 21

    def is_quark(self) -> bool:
        return 1 <= abs(self.pid) <= 6


class PartonShower:
    r"""
    Leading-log parton shower with Sudakov veto algorithm.

    Implements the backward-evolution (final-state radiation) algorithm:
    1. Start at hard scale :math:`Q^2`.
    2. Evolve each parton downward in virtuality.
    3. At each step, generate trial emission via Sudakov veto.
    4. Split parton: q → qg, g → gg, g → qq̄.
    5. Stop at cutoff :math:`t_0 \sim \Lambda_{QCD}^2`.

    References:
        [1] Sjöstrand, Mrenna & Skands, JHEP 05, 026 (2006).
        [2] Buckley et al., Phys. Rep. 504, 145 (2011).
    """

    def __init__(
        self,
        Q_hard: float = 91.187,
        t_cut: float = 1.0,
        alpha_s_fixed: float = 0.118,
    ) -> None:
        self.Q_hard = Q_hard
        self.t_cut = t_cut
        self.alpha_s = alpha_s_fixed
        self.partons: List[Parton] = []
        self._sudakov = SudakovFormFactor(t_cut)

    def initialize(self, initial_partons: List[Parton]) -> None:
        """Set initial hard-process partons."""
        for p in initial_partons:
            p.t = self.Q_hard ** 2
        self.partons = list(initial_partons)

    def _select_z(self, split: str) -> float:
        """Sample splitting fraction z from overestimate distribution."""
        # Simple flat sampling in (z_min, z_max), reject via P(z)
        z_min, z_max = 0.01, 0.99
        P_func = {
            "qq": SplittingFunctions.P_qq,
            "gq": SplittingFunctions.P_gq,
            "qg": SplittingFunctions.P_qg,
            "gg": SplittingFunctions.P_gg,
        }[split]

        P_max = max(P_func(z_min + (z_max - z_min) * i / 100) for i in range(101))
        if P_max < 1e-30:
            return 0.5

        for _ in range(1000):
            z = np.random.uniform(z_min, z_max)
            if np.random.random() < P_func(z) / (P_max + 1e-30):
                return z
        return 0.5

    def _evolve_parton(self, parton: Parton) -> Optional[Tuple[Parton, Parton]]:
        """
        Attempt one emission from a parton via veto algorithm.

        Returns:
            (daughter1, daughter2) if emission, else None.
        """
        t = parton.t
        if t <= self.t_cut:
            return None

        # Trial emission: sample new scale
        for _ in range(100):
            # Generate trial t' < t using Sudakov
            R = np.random.random()
            if R < 1e-30:
                continue
            ln_ratio = math.log(R) * (2.0 * math.pi) / (self.alpha_s * 8.0)
            t_trial = t * math.exp(ln_ratio)

            if t_trial < self.t_cut:
                return None

            # Select splitting channel
            if parton.is_gluon():
                split = np.random.choice(["gg", "qg"], p=[0.7, 0.3])
            else:
                split = "qq"

            z = self._select_z(split)
            kt2 = z * (1.0 - z) * t_trial
            if kt2 < self.t_cut:
                continue

            # Create daughters
            kt = math.sqrt(kt2)
            phi = np.random.uniform(0, 2 * math.pi)

            if split == "qq":
                # q → q + g
                d1 = Parton(
                    pid=parton.pid,
                    E=z * parton.E,
                    pz=z * parton.pz,
                    px=z * parton.px + kt * math.cos(phi),
                    py=z * parton.py + kt * math.sin(phi),
                    t=t_trial,
                )
                d2 = Parton(
                    pid=21,
                    E=(1 - z) * parton.E,
                    pz=(1 - z) * parton.pz,
                    px=(1 - z) * parton.px - kt * math.cos(phi),
                    py=(1 - z) * parton.py - kt * math.sin(phi),
                    t=t_trial,
                )
            elif split == "gg":
                d1 = Parton(
                    pid=21,
                    E=z * parton.E,
                    pz=z * parton.pz,
                    px=z * parton.px + kt * math.cos(phi),
                    py=z * parton.py + kt * math.sin(phi),
                    t=t_trial,
                )
                d2 = Parton(
                    pid=21,
                    E=(1 - z) * parton.E,
                    pz=(1 - z) * parton.pz,
                    px=(1 - z) * parton.px - kt * math.cos(phi),
                    py=(1 - z) * parton.py - kt * math.sin(phi),
                    t=t_trial,
                )
            else:
                # g → q q̄
                q_flavour = np.random.randint(1, 4)
                d1 = Parton(
                    pid=q_flavour,
                    E=z * parton.E,
                    pz=z * parton.pz,
                    px=z * parton.px + kt * math.cos(phi),
                    py=z * parton.py + kt * math.sin(phi),
                    t=t_trial,
                )
                d2 = Parton(
                    pid=-q_flavour,
                    E=(1 - z) * parton.E,
                    pz=(1 - z) * parton.pz,
                    px=(1 - z) * parton.px - kt * math.cos(phi),
                    py=(1 - z) * parton.py - kt * math.sin(phi),
                    t=t_trial,
                )

            return d1, d2

        return None

    def shower(self, max_emissions: int = 1000) -> List[Parton]:
        """
        Run the parton shower until all partons are below cutoff.

        Returns:
            Final-state parton list.
        """
        active = list(self.partons)
        final: List[Parton] = []
        n_emissions = 0

        while active and n_emissions < max_emissions:
            new_active: List[Parton] = []
            for p in active:
                result = self._evolve_parton(p)
                if result is not None:
                    d1, d2 = result
                    new_active.append(d1)
                    new_active.append(d2)
                    n_emissions += 1
                else:
                    final.append(p)
            active = new_active

        final.extend(active)
        self.partons = final
        return final

    def multiplicity(self) -> int:
        """Number of final-state partons."""
        return len(self.partons)

    def total_energy(self) -> float:
        """Total energy of all partons."""
        return sum(p.E for p in self.partons)
