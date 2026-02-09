"""PWA compute engine core — physics, likelihood, fitting, scanning.

Implements the complete intensity construction from Eq. 5.48 of the
Badui dissertation:

    I(τ) = Σ_{ε,ε'} ρ_{ε,ε'}  A_ε(τ)  A*_{ε'}(τ)

where:
    A_ε(τ) = Σ_{b,k} V_{ε,b,k}  ψ_{ε,b,k}(τ)

Indices:
    ε ∈ {+1, −1}  — reflectivity
    b              — wave label (J^{PC} M^ε)
    k              — component index within a wave (decay channels)
    τ = (θ, φ)    — kinematic variables
    V_{ε,b,k}     — complex production amplitudes (fit parameters)
    ψ_{ε,b,k}(τ)  — Wigner-D basis functions
    ρ_{ε,ε'}      — spin density matrix (diagonal or full)

Acceleration:
    The acceptance-normalised likelihood N̄(V) = n_data · V† G V is
    evaluated via a precomputed Gram matrix G_{αβ} = (1/n_gen) Σ_j
    ψ_α(τ_j) ψ*_β(τ_j), turning O(n_MC × n_amp) per iteration into
    O(n_amp²).

References:
    Badui (2020), PhD Dissertation — Eq. 5.48
    Chung (2014), Partial Wave Analysis Formalism
    Adams (2026), QTT Compression for Turbulent Flow Simulation
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import curve_fit, minimize
from scipy.special import factorial, sph_harm
from torch import Tensor


# ════════════════════════════════════════════════════════════════════════════════
# WIGNER-D MATRIX COMPUTATION
# ════════════════════════════════════════════════════════════════════════════════

def wigner_small_d(
    j: float,
    m: float,
    mp: float,
    beta: np.ndarray,
) -> np.ndarray:
    """Wigner small-d matrix element d^j_{m',m}(β) via explicit summation.

    Uses the standard formula (Sakurai Eq. 3.8.33 / Rose convention):

        d^j_{m'm}(β) = Σ_k  (-1)^k √[(j+m)!(j-m)!(j+m')!(j-m')!]
                        / [(j+m-k)!(j-m'-k)! k! (k+m'-m)!]
                        × cos(β/2)^{2j-2k-m'+m} × sin(β/2)^{2k+m'-m}

    Parameters
    ----------
    j : float   Total angular momentum (half-integer or integer).
    m : float   Magnetic quantum number.
    mp : float  Helicity / second projection index.
    beta : ndarray  Polar angle(s) in radians.

    Returns
    -------
    ndarray  d^j_{m',m}(β) for each β.
    """
    k_min = int(max(0, m - mp))
    k_max = int(min(j + m, j - mp))

    prefactor = np.sqrt(
        factorial(j + m, exact=False)
        * factorial(j - m, exact=False)
        * factorial(j + mp, exact=False)
        * factorial(j - mp, exact=False)
    )

    half_beta = beta / 2.0
    cos_hb = np.cos(half_beta)
    sin_hb = np.sin(half_beta)
    result = np.zeros_like(beta, dtype=np.float64)

    for k in range(k_min, k_max + 1):
        sign = (-1.0) ** k
        denom = (
            factorial(j + m - k, exact=False)
            * factorial(j - mp - k, exact=False)
            * factorial(k, exact=False)
            * factorial(k + mp - m, exact=False)
        )
        cos_exp = int(2 * j - 2 * k - mp + m)
        sin_exp = int(2 * k + mp - m)
        cos_term = np.where(cos_exp == 0, 1.0, cos_hb**cos_exp)
        sin_term = np.where(sin_exp == 0, 1.0, sin_hb**sin_exp)
        result += sign * (prefactor / denom) * cos_term * sin_term

    return result


def wigner_D_element(
    j: float,
    m: float,
    mp: float,
    theta: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """Full Wigner-D matrix element D^j_{m,λ}(φ,θ,0).

    D^j_{m,λ}(α,β,γ) = e^{-imα} d^j_{m,λ}(β) e^{-iλγ}

    With γ = 0 (standard PWA convention).

    Returns
    -------
    ndarray  Complex D^j_{m,mp}(φ,θ,0).
    """
    d_val = wigner_small_d(j, m, mp, theta)
    return np.exp(-1j * m * phi) * d_val


# ════════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Wave:
    """Partial wave in the reflectivity basis.

    Attributes
    ----------
    j : float        Total angular momentum (half-integer for baryons).
    m : float        Magnetic quantum number.
    epsilon : int    Reflectivity (+1 or −1).
    n_components : int  Number of decay channels (k index).
    label : str      Human-readable spectroscopic label.
    """

    j: float
    m: float
    epsilon: int
    n_components: int = 1
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            eps_str = "+" if self.epsilon > 0 else "-"
            object.__setattr__(
                self,
                "label",
                f"J={self.j:.1f} M={self.m:+.1f} ε={eps_str}",
            )


class WaveSet:
    """Ordered collection of partial waves with flat amplitude indexing.

    The flat index α enumerates (ε, b, k) triples:
        α = 0, 1, ..., n_amplitudes - 1

    Each α maps to a unique (ε_b, b, k) triple where:
        ε_b = wave reflectivity
        b   = wave index in self.waves
        k   = component index within wave b
    """

    def __init__(self, waves: List[Wave]) -> None:
        self.waves = list(waves)
        self._build_index()

    def _build_index(self) -> None:
        self._alpha_to_ebk: Dict[int, Tuple[int, int, int]] = {}
        self._ebk_to_alpha: Dict[Tuple[int, int, int], int] = {}
        alpha = 0
        for b, w in enumerate(self.waves):
            for k in range(w.n_components):
                self._alpha_to_ebk[alpha] = (w.epsilon, b, k)
                self._ebk_to_alpha[(w.epsilon, b, k)] = alpha
                alpha += 1
        self._n_amp = alpha

        # Boolean masks per reflectivity
        eps_set = sorted(set(w.epsilon for w in self.waves))
        self._eps_list = eps_set
        self._masks: Dict[int, np.ndarray] = {}
        for eps in eps_set:
            mask = np.zeros(self._n_amp, dtype=bool)
            for a in range(self._n_amp):
                if self._alpha_to_ebk[a][0] == eps:
                    mask[a] = True
            self._masks[eps] = mask

    @property
    def n_amplitudes(self) -> int:
        return self._n_amp

    @property
    def reflectivities(self) -> List[int]:
        return list(self._eps_list)

    def epsilon_mask_np(self, eps: int) -> np.ndarray:
        return self._masks[eps]

    def epsilon_mask(self, eps: int, device: torch.device) -> Tensor:
        return torch.tensor(self._masks[eps], dtype=torch.bool, device=device)

    def wave_for_alpha(self, alpha: int) -> Tuple[int, int, int]:
        """Return (epsilon, wave_index, component_k) for flat index α."""
        return self._alpha_to_ebk[alpha]

    def alpha_for(self, epsilon: int, b: int, k: int = 0) -> int:
        return self._ebk_to_alpha[(epsilon, b, k)]


def build_wave_set(
    j_max: float,
    reflectivities: Tuple[int, ...] = (+1,),
    n_components: int = 1,
) -> WaveSet:
    """Build standard half-integer-J wave set up to j_max.

    Parameters
    ----------
    j_max : float       Maximum J (e.g. 3.5 for J up to 7/2).
    reflectivities : tuple  Which reflectivities to include.
    n_components : int  Decay components per wave.

    Returns
    -------
    WaveSet with all (j, m, ε) combinations.
    """
    waves: List[Wave] = []
    j = 0.5
    while j <= j_max + 0.01:
        for m_val in np.arange(-j, j + 0.5, 1.0):
            for eps in reflectivities:
                eps_str = "+" if eps > 0 else "-"
                label = f"J={j:.1f} M={m_val:+.1f} ε={eps_str}"
                waves.append(
                    Wave(j=j, m=m_val, epsilon=eps, n_components=n_components, label=label)
                )
        j += 1.0
    return WaveSet(waves)


# ════════════════════════════════════════════════════════════════════════════════
# BASIS AMPLITUDE PRECOMPUTATION
# ════════════════════════════════════════════════════════════════════════════════

class BasisAmplitudes:
    """Precomputed basis matrix ψ_α(τ_i) for all events and amplitude indices.

    Shape: (n_events, n_amplitudes) complex128.

    The matrix is computed once from the angular grid and stored. During
    fitting, only the production amplitudes V change; ψ stays fixed.
    """

    def __init__(
        self,
        wave_set: WaveSet,
        theta: np.ndarray,
        phi: np.ndarray,
        helicity: float = 0.5,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        n_events = len(theta)
        n_amp = wave_set.n_amplitudes
        psi_np = np.zeros((n_events, n_amp), dtype=np.complex128)

        for alpha in range(n_amp):
            eps, b, k = wave_set.wave_for_alpha(alpha)
            wave = wave_set.waves[b]
            # Component k can shift helicity for multi-component waves.
            # k=0 → base helicity; k>0 → higher helicity couplings.
            lambda_eff = helicity + k * 1.0
            if abs(lambda_eff) > wave.j:
                lambda_eff = helicity
            psi_np[:, alpha] = wigner_D_element(
                wave.j, wave.m, lambda_eff, theta, phi
            )

        self.psi = torch.tensor(psi_np, dtype=torch.complex128, device=device)
        self.n_events = n_events
        self.n_amplitudes = n_amp
        self.wave_set = wave_set
        self.device = device
        self._theta = theta
        self._phi = phi


# ════════════════════════════════════════════════════════════════════════════════
# INTENSITY MODEL (EQ. 5.48)
# ════════════════════════════════════════════════════════════════════════════════

class IntensityModel:
    """Full Eq. 5.48 intensity with reflectivity and density matrix.

    I(τ) = Σ_{ε,ε'} ρ_{ε,ε'} A_ε(τ) A*_{ε'}(τ)

    where A_ε(τ) = Σ_{α: ε_α=ε} V_α ψ_α(τ)

    Modes:
        diagonal_rho=True  → I = Σ_ε ρ_ε |A_ε|²  (incoherent reflectivity sum)
        diagonal_rho=False → I = Σ_{ε,ε'} ρ_{ε,ε'} A_ε A*_{ε'}  (full)

    Special case (single ε, ρ=1):
        I = |Σ_b V_b ψ_b|²  — reduces to simple coherent sum.
    """

    def __init__(
        self,
        basis: BasisAmplitudes,
        rho: Optional[Tensor] = None,
        diagonal_rho: bool = True,
    ) -> None:
        self.basis = basis
        self.wave_set = basis.wave_set
        self.diagonal_rho = diagonal_rho
        self.device = basis.device

        eps_list = self.wave_set.reflectivities
        n_eps = len(eps_list)
        self._eps_list = eps_list

        # Density matrix setup
        if rho is not None:
            if diagonal_rho:
                self.rho_diag = rho.to(dtype=torch.float64, device=self.device)
            else:
                self.rho_full = rho.to(dtype=torch.complex128, device=self.device)
        else:
            if n_eps == 1:
                # Single reflectivity → ρ = 1
                self.rho_diag = torch.ones(1, dtype=torch.float64, device=self.device)
            else:
                if diagonal_rho:
                    self.rho_diag = torch.ones(n_eps, dtype=torch.float64, device=self.device) / n_eps
                else:
                    self.rho_full = torch.eye(n_eps, dtype=torch.complex128, device=self.device) / n_eps

        # Torch boolean masks per reflectivity
        self._torch_masks: Dict[int, Tensor] = {}
        for eps in eps_list:
            self._torch_masks[eps] = self.wave_set.epsilon_mask(eps, self.device)

    def amplitude(self, V: Tensor, epsilon: int) -> Tensor:
        """Compute A_ε(τ_i) = Σ_{α: ε_α=ε} V_α ψ_α(τ_i).

        Returns: (n_events,) complex128
        """
        mask = self._torch_masks[epsilon]
        return self.basis.psi[:, mask] @ V[mask]

    def evaluate(self, V: Tensor) -> Tensor:
        """Compute intensity I(τ_i; V) for all events.

        Parameters
        ----------
        V : Tensor  (n_amplitudes,) complex128 production amplitudes.

        Returns
        -------
        Tensor  (n_events,) float64 intensity values.
        """
        n_events = self.basis.n_events
        I = torch.zeros(n_events, dtype=torch.float64, device=self.device)

        eps_list = self._eps_list
        if self.diagonal_rho:
            for i, eps in enumerate(eps_list):
                A_eps = self.amplitude(V, eps)
                I = I + self.rho_diag[i] * (A_eps * A_eps.conj()).real
        else:
            amps = {eps: self.amplitude(V, eps) for eps in eps_list}
            for i, eps_i in enumerate(eps_list):
                for j, eps_j in enumerate(eps_list):
                    cross = amps[eps_i] * amps[eps_j].conj()
                    I = I + (self.rho_full[i, j] * cross).real

        return I


# ════════════════════════════════════════════════════════════════════════════════
# GRAM MATRIX — FAST NORMALIZATION
# ════════════════════════════════════════════════════════════════════════════════

class GramMatrix:
    """Acceptance-weighted Gram matrix for fast normalization.

    G_{αβ} = (1 / n_generated) Σ_{j ∈ accepted} ψ_α(τ_j) ψ*_β(τ_j)

    The normalization integral becomes:
        N̄(V) = n_data · V† G V

    This replaces the O(n_MC × n_amp) per-iteration pointwise evaluation
    with O(n_amp²) — independent of event count.

    For diagonal ρ, the Gram matrix is block-diagonal by reflectivity.
    For full ρ, cross-reflectivity blocks contribute.
    """

    def __init__(
        self,
        basis_mc: BasisAmplitudes,
        n_generated: int,
    ) -> None:
        self.n_generated = n_generated
        self.n_accepted = basis_mc.n_events
        self.device = basis_mc.device
        n_amp = basis_mc.n_amplitudes

        # G = (1/n_gen) ψ†_mc ψ_mc
        psi = basis_mc.psi  # (n_accepted, n_amp)
        self.G = (psi.conj().T @ psi) / n_generated  # (n_amp, n_amp)

        # SVD for diagnostics
        self._svd_computed = False

    def normalization(self, V: Tensor, n_data: int) -> Tensor:
        """N̄(V) = n_data · V† G V.

        Returns scalar float64.
        """
        return n_data * (V.conj() @ self.G @ V).real

    def normalization_gradient(self, V: Tensor, n_data: int) -> Tensor:
        """∂N̄/∂V* = n_data · G V.

        This is the Wirtinger derivative w.r.t. V*.
        """
        return n_data * (self.G @ V)

    @property
    def svd_spectrum(self) -> np.ndarray:
        if not self._svd_computed:
            S = torch.linalg.svdvals(self.G)
            self._spectrum = S.detach().cpu().numpy()
            self._svd_computed = True
        return self._spectrum

    @property
    def condition_number(self) -> float:
        S = self.svd_spectrum
        return float(S[0] / S[np.nonzero(S > S[0] * 1e-15)[0][-1]]) if len(S) > 1 else 1.0


# ════════════════════════════════════════════════════════════════════════════════
# EXTENDED LIKELIHOOD
# ════════════════════════════════════════════════════════════════════════════════

class ExtendedLikelihood:
    """Extended negative log-likelihood with acceptance normalization.

    L(V) = -Σ_i ln I(τ_i; V) + N̄(V)

    Two backends:
        use_gram=True  → N̄(V) = n_data · V† G V             [O(n_amp²)]
        use_gram=False → N̄(V) = (n_data/n_gen) Σ_j I(τ_j;V) [O(n_MC × n_amp)]

    Both give identical results; Gram is faster for repeated evaluations.
    """

    def __init__(
        self,
        intensity_data: IntensityModel,
        intensity_mc: Optional[IntensityModel],
        gram: Optional[GramMatrix],
        n_data: int,
        n_generated: int,
        use_gram: bool = True,
        floor: float = 1e-300,
    ) -> None:
        self.intensity_data = intensity_data
        self.intensity_mc = intensity_mc
        self.gram = gram
        self.n_data = n_data
        self.n_generated = n_generated
        self.use_gram = use_gram and (gram is not None)
        self.floor = floor
        self.device = intensity_data.device
        self._eval_count = 0

    def __call__(self, V: Tensor) -> Tensor:
        """Evaluate NLL. V must have requires_grad=True for autodiff."""
        self._eval_count += 1

        # Data term: -Σ_i ln I(τ_i; V)
        I_data = self.intensity_data.evaluate(V)
        I_safe = torch.clamp(I_data, min=self.floor)
        data_term = -torch.sum(torch.log(I_safe))

        # Normalization term
        if self.use_gram:
            norm_term = self.gram.normalization(V, self.n_data)
        else:
            I_mc = self.intensity_mc.evaluate(V)
            norm_term = (self.n_data / self.n_generated) * torch.sum(I_mc)

        return data_term + norm_term

    @property
    def eval_count(self) -> int:
        return self._eval_count

    def reset_count(self) -> None:
        self._eval_count = 0


# ════════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATION
# ════════════════════════════════════════════════════════════════════════════════

class SyntheticDataGenerator:
    """Generate synthetic PWA data via accept/reject sampling.

    The acceptance function η(τ) models detector effects. Events are
    drawn uniformly in (θ, φ) and accepted with probability proportional
    to η(τ) I(τ; V_true).

    Also produces the MC sample (accepted generated events) needed for
    normalization.
    """

    def __init__(
        self,
        wave_set: WaveSet,
        V_true: np.ndarray,
        helicity: float = 0.5,
        acceptance_fn: Optional[Any] = None,
        seed: int = 42,
    ) -> None:
        self.wave_set = wave_set
        self.V_true = V_true
        self.helicity = helicity
        self.seed = seed

        if acceptance_fn is None:
            # Default: non-uniform acceptance with φ-dependence to break symmetry
            # and enable phase sensitivity in the likelihood.
            self.acceptance_fn = lambda theta, phi: (
                0.5 + 0.2 * np.cos(theta) + 0.15 * np.cos(phi) + 0.1 * np.sin(phi)
            )
        else:
            self.acceptance_fn = acceptance_fn

    def generate(
        self,
        n_data: int,
        n_generated: int,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, Any]:
        """Generate synthetic data and MC samples.

        Data events are drawn from I(τ;V_true) × η(τ) (physics × acceptance),
        while MC events are drawn from η(τ) only (acceptance without physics).
        This separation is essential for unbiased Gram matrix normalisation
        in the extended likelihood.

        Parameters
        ----------
        n_data : int       Target number of data events.
        n_generated : int  Total number of phase-space events generated.

        Returns
        -------
        dict with keys:
            'theta_data', 'phi_data'    — data event kinematics
            'theta_mc', 'phi_mc'        — accepted MC kinematics (η-only)
            'n_data', 'n_generated', 'n_mc_accepted'
            'V_true'                    — true parameters used
        """
        rng = np.random.default_rng(self.seed)

        # Generate flat phase space
        theta_gen = np.arccos(1.0 - 2.0 * rng.uniform(size=n_generated))
        phi_gen = rng.uniform(0.0, 2.0 * np.pi, size=n_generated)

        # Build basis and evaluate intensity at generated points
        basis_gen = BasisAmplitudes(
            self.wave_set, theta_gen, phi_gen, self.helicity, device
        )
        V_torch = torch.tensor(self.V_true, dtype=torch.complex128, device=device)
        model_gen = IntensityModel(basis_gen)
        I_gen = model_gen.evaluate(V_torch).detach().cpu().numpy()

        # Acceptance function
        eta_gen = self.acceptance_fn(theta_gen, phi_gen)

        # ── Data events: accept ∝ I(τ) × η(τ) ────────────────────────
        weights_data = I_gen * eta_gen
        w_max_data = weights_data.max() * 1.05
        u_data = rng.uniform(size=n_generated)
        data_mask = u_data < (weights_data / w_max_data)
        data_idx_all = np.nonzero(data_mask)[0]

        if len(data_idx_all) < n_data:
            data_idx = data_idx_all
        else:
            data_idx = rng.choice(data_idx_all, size=n_data, replace=False)

        theta_data = theta_gen[data_idx]
        phi_data = phi_gen[data_idx]

        # ── MC events: accept ∝ η(τ) only (no physics) ───────────────
        #   Separate phase-space draw so MC is statistically independent
        #   of data.  The Gram matrix G = (1/n_gen) Σ_{mc} ψ*ψ then
        #   correctly estimates ∫ ψ*ψ η(τ) dΩ / Ω_PS.
        theta_mc_gen = np.arccos(1.0 - 2.0 * rng.uniform(size=n_generated))
        phi_mc_gen = rng.uniform(0.0, 2.0 * np.pi, size=n_generated)
        eta_mc = self.acceptance_fn(theta_mc_gen, phi_mc_gen)
        eta_max = eta_mc.max() * 1.05
        u_mc = rng.uniform(size=n_generated)
        mc_mask = u_mc < (eta_mc / eta_max)
        theta_mc = theta_mc_gen[mc_mask]
        phi_mc = phi_mc_gen[mc_mask]

        return {
            "theta_data": theta_data,
            "phi_data": phi_data,
            "theta_mc": theta_mc,
            "phi_mc": phi_mc,
            "n_data": len(data_idx),
            "n_generated": n_generated,
            "n_mc_accepted": int(mc_mask.sum()),
            "V_true": self.V_true.copy(),
            "acceptance_rate": float(data_mask.mean()),
        }


# ════════════════════════════════════════════════════════════════════════════════
# L-BFGS OPTIMIZER
# ════════════════════════════════════════════════════════════════════════════════

class LBFGSFitter:
    """L-BFGS-B optimizer for extended likelihood minimization.

    Uses scipy.optimize.minimize with torch autograd for gradients.
    Complex parameters are handled via real parameterization:
        x = [Re(V_0), ..., Re(V_{n-1}), Im(V_0), ..., Im(V_{n-1})]

    Supports multi-start fitting for basin exploration.
    """

    def __init__(
        self,
        likelihood: ExtendedLikelihood,
        max_iter: int = 500,
        tolerance: float = 1e-10,
    ) -> None:
        self.likelihood = likelihood
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.n_amp = likelihood.intensity_data.basis.n_amplitudes
        self.device = likelihood.device

    def _x_to_V(self, x: np.ndarray) -> Tensor:
        """Convert real parameter vector to complex amplitude tensor."""
        n = self.n_amp
        re = torch.tensor(x[:n], dtype=torch.float64, device=self.device)
        im = torch.tensor(x[n:], dtype=torch.float64, device=self.device)
        return torch.complex(re, im)

    def _V_to_x(self, V: Tensor) -> np.ndarray:
        """Convert complex amplitude tensor to real parameter vector."""
        V_cpu = V.detach().cpu()
        return np.concatenate([V_cpu.real.numpy(), V_cpu.imag.numpy()])

    def _objective(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """Objective function with gradient for scipy."""
        n = self.n_amp
        re = torch.tensor(
            x[:n], dtype=torch.float64, device=self.device, requires_grad=True
        )
        im = torch.tensor(
            x[n:], dtype=torch.float64, device=self.device, requires_grad=True
        )
        V = torch.complex(re, im)

        nll = self.likelihood(V)
        nll.backward()

        grad_re = re.grad.detach().cpu().numpy()
        grad_im = im.grad.detach().cpu().numpy()
        grad = np.concatenate([grad_re, grad_im])

        return nll.item(), grad

    def fit(
        self,
        V_init: Optional[Tensor] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run single L-BFGS-B fit.

        Parameters
        ----------
        V_init : optional starting point. If None, random.

        Returns
        -------
        dict with 'V_best', 'nll', 'converged', 'n_evals', 'time_s'
        """
        if V_init is not None:
            x0 = self._V_to_x(V_init)
        else:
            rng = np.random.default_rng(seed)
            x0 = rng.standard_normal(2 * self.n_amp) * 0.5

        self.likelihood.reset_count()
        t0 = time.perf_counter()

        result = minimize(
            self._objective,
            x0,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.max_iter, "ftol": self.tolerance, "gtol": 1e-8},
        )

        elapsed = time.perf_counter() - t0
        V_best = self._x_to_V(result.x)

        return {
            "V_best": V_best,
            "nll": result.fun,
            "converged": result.success,
            "n_evals": self.likelihood.eval_count,
            "n_iter": result.nit,
            "grad_norm": float(np.linalg.norm(result.jac)),
            "time_s": elapsed,
            "message": result.message,
        }

    def multi_start_fit(
        self,
        n_starts: int = 20,
        seed_base: int = 0,
    ) -> Dict[str, Any]:
        """Run multiple fits from random starts.

        Returns
        -------
        dict with 'best_fit', 'all_fits', 'stability_metrics'
        """
        fits: List[Dict[str, Any]] = []
        for i in range(n_starts):
            fit = self.fit(seed=seed_base + i)
            fits.append(fit)

        # Find global minimum
        nlls = [f["nll"] for f in fits]
        best_idx = int(np.argmin(nlls))
        best_nll = nlls[best_idx]

        # Stability: how many converged near the best?
        tol_basin = 1.0  # NLL within 1 unit of min considered same basin
        near_best = sum(1 for x in nlls if abs(x - best_nll) < tol_basin)

        # Parameter spread at the best basin
        V_best = fits[best_idx]["V_best"]
        basin_fits = [
            f for f in fits if abs(f["nll"] - best_nll) < tol_basin
        ]
        if len(basin_fits) > 1:
            V_stack = torch.stack([f["V_best"] for f in basin_fits])
            param_std = V_stack.std(dim=0).abs().mean().item()
        else:
            param_std = 0.0

        return {
            "best_fit": fits[best_idx],
            "all_fits": fits,
            "n_starts": n_starts,
            "n_near_best": near_best,
            "basin_fraction": near_best / n_starts,
            "param_std": param_std,
            "nll_spread": float(np.std(nlls)),
            "best_nll": best_nll,
            "worst_nll": float(np.max(nlls)),
        }

    def hessian(self, V: Tensor) -> Tensor:
        """Compute Hessian of NLL via torch autograd (second-order).

        Uses real parameterization: H is (2n × 2n) real matrix.

        Returns
        -------
        Tensor (2*n_amp, 2*n_amp) float64.
        """
        n = self.n_amp
        x = self._V_to_x(V)
        x_torch = torch.tensor(x, dtype=torch.float64, device=self.device, requires_grad=True)

        re = x_torch[:n]
        im = x_torch[n:]
        V_param = torch.complex(re, im)
        nll = self.likelihood(V_param)

        grad = torch.autograd.grad(nll, x_torch, create_graph=True)[0]

        H = torch.zeros(2 * n, 2 * n, dtype=torch.float64, device=self.device)
        for i in range(2 * n):
            grad_i = torch.autograd.grad(grad[i], x_torch, retain_graph=True)[0]
            H[i] = grad_i

        return H

    def covariance(self, V: Tensor) -> Tensor:
        """Parameter covariance from inverse Hessian at minimum.

        Returns
        -------
        Tensor (2*n_amp, 2*n_amp) float64 covariance matrix.
        """
        H = self.hessian(V)
        return torch.linalg.inv(H)


# ════════════════════════════════════════════════════════════════════════════════
# CONVENTION REDUCTION TEST
# ════════════════════════════════════════════════════════════════════════════════

def convention_reduction_test(
    device: torch.device = torch.device("cpu"),
    n_events: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """Verify the full Eq. 5.48 model reduces to the simplified form.

    Tests three progressive simplifications:
        1. Full (ε ∈ {+1,-1}, full ρ)  →  set V_{ε=-1} = 0, ρ = diag(1,0)
        2. That reduces to single-ε diagonal form
        3. Single-ε with n_components=1 matches |Σ_b V_b ψ_b|²

    Returns dict with max pointwise errors for each reduction step.
    Machine-precision agreement (< 1e-12) proves the general model
    contains the simplified form as a special case.
    """
    rng = np.random.default_rng(seed)
    theta = np.pi * rng.uniform(size=n_events)
    phi = 2.0 * np.pi * rng.uniform(size=n_events)

    j_vals = [0.5, 1.5, 2.5]

    # ── Model A: Full form with both reflectivities ──────────────────────
    ws_full = build_wave_set(j_max=2.5, reflectivities=(+1, -1), n_components=1)
    basis_full = BasisAmplitudes(ws_full, theta, phi, helicity=0.5, device=device)

    # Density matrix: sorted reflectivities = [-1, +1], so index 0 is ε=-1.
    # Set ρ[-1]=0, ρ[+1]=1 → only ε=+1 contributes.
    rho_kill_minus = torch.tensor([0.0, 1.0], dtype=torch.float64, device=device)
    model_full = IntensityModel(basis_full, rho=rho_kill_minus, diagonal_rho=True)

    # V: zero out all ε=-1 amplitudes
    n_amp_full = ws_full.n_amplitudes
    V_full = torch.zeros(n_amp_full, dtype=torch.complex128, device=device)
    # Set ε=+1 amplitudes to known values
    V_plus_values = torch.tensor(
        rng.standard_normal(n_amp_full // 2) + 1j * rng.standard_normal(n_amp_full // 2),
        dtype=torch.complex128,
        device=device,
    )
    mask_plus = ws_full.epsilon_mask(+1, device)
    V_full[mask_plus] = V_plus_values

    I_full = model_full.evaluate(V_full)

    # ── Model B: Single-reflectivity form ────────────────────────────────
    ws_single = build_wave_set(j_max=2.5, reflectivities=(+1,), n_components=1)
    basis_single = BasisAmplitudes(ws_single, theta, phi, helicity=0.5, device=device)
    model_single = IntensityModel(basis_single, diagonal_rho=True)

    V_single = V_plus_values[: ws_single.n_amplitudes]
    I_single = model_single.evaluate(V_single)

    # Test 1: Full with killed ε=-1 ≡ Single-ε
    err_1 = torch.max(torch.abs(I_full - I_single)).item()

    # ── Model C: Direct coherent sum |Σ V_b ψ_b|² ───────────────────────
    psi_direct = basis_single.psi  # (n_events, n_amp)
    A_direct = psi_direct @ V_single
    I_direct = (A_direct * A_direct.conj()).real

    # Test 2: IntensityModel ≡ manual coherent sum
    err_2 = torch.max(torch.abs(I_single - I_direct)).item()

    # ── Model D: Full form with full (non-diagonal) ρ ────────────────────
    # ρ = [[0, 0], [0, 1]] as full matrix (eps sorted: [-1, +1])
    # ρ_{++}=1, all others 0 → should match diagonal [0, 1]
    rho_full_mat = torch.tensor(
        [[0.0 + 0j, 0.0 + 0j], [0.0 + 0j, 1.0 + 0j]],
        dtype=torch.complex128,
        device=device,
    )
    model_full_rho = IntensityModel(basis_full, rho=rho_full_mat, diagonal_rho=False)
    I_full_rho = model_full_rho.evaluate(V_full)

    # Test 3: Full ρ matrix with zero off-diag ≡ diagonal ρ
    err_3 = torch.max(torch.abs(I_full_rho - I_full)).item()

    return {
        "test_1_full_vs_single_eps": err_1,
        "test_2_model_vs_manual": err_2,
        "test_3_full_rho_vs_diagonal": err_3,
        "n_events": n_events,
        "n_amp_full": n_amp_full,
        "n_amp_single": ws_single.n_amplitudes,
        "all_pass": err_1 < 1e-12 and err_2 < 1e-12 and err_3 < 1e-12,
    }


# ════════════════════════════════════════════════════════════════════════════════
# WAVE-SET SCAN
# ════════════════════════════════════════════════════════════════════════════════

def wave_set_scan(
    j_max_values: List[float],
    V_true: np.ndarray,
    true_j_max: float,
    n_data: int = 5000,
    n_generated: int = 200_000,
    n_starts: int = 10,
    helicity: float = 0.5,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> Dict[str, Any]:
    """Systematic scan over wave-set complexity.

    Generates synthetic data from V_true at J_max = true_j_max, then fits
    with progressively larger wave sets. For each J_max in j_max_values:
        1. Build wave set
        2. Multi-start fit
        3. Record NLL, parameters, stability

    Returns
    -------
    dict with per-J_max results and robustness metrics.
    """
    # Generate data from true model
    true_ws = build_wave_set(true_j_max, reflectivities=(+1,))
    gen = SyntheticDataGenerator(true_ws, V_true, helicity=helicity, seed=seed)
    data = gen.generate(n_data, n_generated, device=device)

    scan_results: List[Dict[str, Any]] = []

    for j_max in j_max_values:
        ws = build_wave_set(j_max, reflectivities=(+1,))
        n_amp = ws.n_amplitudes

        # Build basis for data and MC
        basis_data = BasisAmplitudes(
            ws, data["theta_data"], data["phi_data"], helicity, device
        )
        basis_mc = BasisAmplitudes(
            ws, data["theta_mc"], data["phi_mc"], helicity, device
        )

        # Intensity models and Gram
        model_data = IntensityModel(basis_data)
        gram = GramMatrix(basis_mc, data["n_generated"])

        # Likelihood
        nll = ExtendedLikelihood(
            model_data, None, gram, data["n_data"], data["n_generated"]
        )

        # Multi-start fit
        fitter = LBFGSFitter(nll, max_iter=300)
        ms_result = fitter.multi_start_fit(n_starts=n_starts, seed_base=seed + int(j_max * 100))

        # Gram spectrum
        gram_spectrum = gram.svd_spectrum

        # Yields: |V_b|² for each amplitude
        V_best = ms_result["best_fit"]["V_best"]
        yields = (V_best * V_best.conj()).real.detach().cpu().numpy()

        scan_results.append({
            "j_max": j_max,
            "n_waves": len(ws.waves),
            "n_amplitudes": n_amp,
            "best_nll": ms_result["best_nll"],
            "basin_fraction": ms_result["basin_fraction"],
            "param_std": ms_result["param_std"],
            "nll_spread": ms_result["nll_spread"],
            "yields": yields,
            "V_best": V_best.detach().cpu().numpy(),
            "gram_spectrum": gram_spectrum,
            "gram_rank_1e8": int(np.sum(gram_spectrum > gram_spectrum[0] * 1e-8)),
            "fit_time_s": ms_result["best_fit"]["time_s"],
            "n_near_best": ms_result["n_near_best"],
            "converged": ms_result["best_fit"]["converged"],
        })

    return {
        "j_max_values": j_max_values,
        "true_j_max": true_j_max,
        "n_data": data["n_data"],
        "n_generated": data["n_generated"],
        "n_mc_accepted": data["n_mc_accepted"],
        "scan_results": scan_results,
    }


# ════════════════════════════════════════════════════════════════════════════════
# QTT COMPRESSION OF GRAM MATRIX
# ════════════════════════════════════════════════════════════════════════════════

def compress_gram_qtt(
    gram: GramMatrix,
    max_rank: int = 32,
    tol: float = 1e-10,
) -> Dict[str, Any]:
    """Compress the Gram matrix in QTT format via TT-SVD.

    For large wave sets, the Gram matrix G ∈ C^{n×n} may be
    compressible because the basis functions have angular momentum
    structure (selection rules limit coupling).

    Returns compression metrics and SVD spectra.
    """
    G = gram.G
    n = G.shape[0]

    # Pad to power of 2 if necessary
    n_bits = max(1, math.ceil(math.log2(max(n, 2))))
    N = 2**n_bits
    G_padded = torch.zeros(N, N, dtype=torch.complex128, device=G.device)
    G_padded[:n, :n] = G

    # Flatten and reshape to (2, 2, ..., 2) for QTT
    total_q = 2 * n_bits
    vec = G_padded.reshape([2] * n_bits + [2] * n_bits)

    # Morton interleave
    perm = []
    for i in range(n_bits):
        perm.extend([i, i + n_bits])
    morton = vec.permute(perm).reshape(2**total_q)

    # TT-SVD
    cores: List[Tensor] = []
    svd_spectra: Dict[int, np.ndarray] = {}
    bond_dims: List[int] = []
    current = morton.reshape(1, -1)

    for k in range(total_q - 1):
        r_left = current.shape[0]
        current = current.reshape(r_left * 2, -1)
        U, S, Vh = torch.linalg.svd(current, full_matrices=False)
        svd_spectra[k] = S.detach().cpu().numpy().copy()

        s_max = S[0].item()
        if s_max > 0:
            keep = int((S > s_max * tol).sum().item())
        else:
            keep = 1
        r = max(1, min(max_rank, keep))

        cores.append(U[:, :r].reshape(r_left, 2, r))
        bond_dims.append(r)
        current = S[:r].to(dtype=Vh.dtype).unsqueeze(1) * Vh[:r, :]

    cores.append(current.reshape(-1, 2, 1))
    bond_dims.append(1)

    chi_max = max(bond_dims)
    qtt_storage = sum(c.numel() for c in cores)
    dense_storage = N * N
    # For complex: multiply by 2
    compression_ratio = (2 * n * n) / (2 * qtt_storage)

    return {
        "n_original": n,
        "n_padded": N,
        "chi_max": chi_max,
        "bond_dims": bond_dims,
        "qtt_storage": qtt_storage,
        "dense_storage": n * n,
        "compression_ratio": compression_ratio,
        "svd_spectra": svd_spectra,
        "gram_svd": gram.svd_spectrum,
    }


# ════════════════════════════════════════════════════════════════════════════════
# SPEEDUP BENCHMARK
# ════════════════════════════════════════════════════════════════════════════════

def benchmark_normalization(
    wave_set: WaveSet,
    n_mc_events: int,
    n_generated: int,
    n_evals: int = 100,
    helicity: float = 0.5,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> Dict[str, Any]:
    """Benchmark baseline vs Gram-accelerated normalization.

    Measures wall time for n_evals evaluations of N̄(V) using:
        1. Baseline:  (n_data/n_gen) Σ_j I(τ_j; V)  — O(n_MC × n_amp)
        2. Gram:      n_data · V† G V                — O(n_amp²)

    Returns timing results and speedup.
    """
    rng = np.random.default_rng(seed)
    theta = np.pi * rng.uniform(size=n_mc_events)
    phi = 2.0 * np.pi * rng.uniform(size=n_mc_events)

    basis = BasisAmplitudes(wave_set, theta, phi, helicity, device)
    model = IntensityModel(basis)
    gram = GramMatrix(basis, n_generated)

    V = torch.tensor(
        rng.standard_normal(wave_set.n_amplitudes) + 1j * rng.standard_normal(wave_set.n_amplitudes),
        dtype=torch.complex128,
        device=device,
    )
    V = V / V.norm()
    n_data = 10000

    # Warmup
    _ = model.evaluate(V)
    _ = gram.normalization(V, n_data)

    # Baseline timing
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_evals):
        I_mc = model.evaluate(V)
        _ = (n_data / n_generated) * torch.sum(I_mc)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_baseline = time.perf_counter() - t0

    # Gram timing
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_evals):
        _ = gram.normalization(V, n_data)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_gram = time.perf_counter() - t0

    # Verify identical results
    N_baseline = ((n_data / n_generated) * torch.sum(model.evaluate(V))).item()
    N_gram = gram.normalization(V, n_data).item()
    agreement = abs(N_baseline - N_gram) / max(abs(N_baseline), 1e-30)

    return {
        "n_mc_events": n_mc_events,
        "n_amplitudes": wave_set.n_amplitudes,
        "n_evals": n_evals,
        "t_baseline_s": t_baseline,
        "t_gram_s": t_gram,
        "speedup": t_baseline / max(t_gram, 1e-30),
        "baseline_per_eval_ms": 1000 * t_baseline / n_evals,
        "gram_per_eval_ms": 1000 * t_gram / n_evals,
        "N_baseline": N_baseline,
        "N_gram": N_gram,
        "relative_agreement": agreement,
    }


# ════════════════════════════════════════════════════════════════════════════════
# ANGULAR MOMENTS — GOODNESS-OF-FIT DIAGNOSTIC
# ════════════════════════════════════════════════════════════════════════════════

def compute_angular_moments(
    theta: np.ndarray,
    phi: np.ndarray,
    L_max: int = 4,
    weights: Optional[np.ndarray] = None,
) -> Dict[Tuple[int, int], complex]:
    """Compute angular moments ⟨Y_L^M⟩ from event sample.

    Parameters
    ----------
    theta : ndarray   Polar angles (radians).
    phi : ndarray     Azimuthal angles (radians).
    L_max : int       Maximum angular momentum L to compute.
    weights : ndarray  Per-event weights (e.g. intensity). If None, uniform.

    Returns
    -------
    dict mapping (L, M) → complex moment value.
    """
    n = len(theta)
    if weights is None:
        weights = np.ones(n, dtype=np.float64) / n
    else:
        weights = weights / weights.sum()

    moments: Dict[Tuple[int, int], complex] = {}
    for L in range(L_max + 1):
        for M in range(-L, L + 1):
            # scipy sph_harm convention: sph_harm(M, L, phi, theta)
            Y_LM = sph_harm(M, L, phi, theta)
            moments[(L, M)] = complex(np.sum(weights * Y_LM))

    return moments


def moment_comparison(
    theta_data: np.ndarray,
    phi_data: np.ndarray,
    theta_mc: np.ndarray,
    phi_mc: np.ndarray,
    V_fit: Tensor,
    wave_set: WaveSet,
    n_generated: int,
    L_max: int = 4,
    helicity: float = 0.5,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Compare angular moments between data and fitted model.

    Computes ⟨Y_L^M⟩_data (uniform-weighted from data) and
    ⟨Y_L^M⟩_fit (intensity-weighted from MC using fitted V).

    The residual (data − fit) / σ is a model-independent goodness-of-fit
    diagnostic. Significant deviations in specific (L, M) channels
    reveal missing partial waves.

    Parameters
    ----------
    theta_data, phi_data : data event kinematics.
    theta_mc, phi_mc : accepted MC event kinematics.
    V_fit : fitted complex production amplitudes.
    wave_set : WaveSet used in the fit.
    n_generated : total number of generated MC events.
    L_max : maximum angular momentum.

    Returns
    -------
    dict with moments, residuals, and summary statistics.
    """
    # Data moments (uniform weights → weighted by acceptance × intensity already)
    moments_data = compute_angular_moments(theta_data, phi_data, L_max)

    # Model prediction: weight MC events by I(τ; V_fit)
    basis_mc = BasisAmplitudes(wave_set, theta_mc, phi_mc, helicity, device)
    model_mc = IntensityModel(basis_mc)
    I_mc = model_mc.evaluate(V_fit).detach().cpu().numpy()

    # Acceptance-corrected model prediction:
    # ⟨Y⟩_model = (1/N̄) Σ_{j∈accepted} I(τ_j; V) Y(τ_j)
    # With N̄ = (n_data/n_gen) Σ I_j
    I_sum = I_mc.sum()
    if I_sum > 0:
        model_weights = I_mc / I_sum
    else:
        model_weights = np.ones(len(I_mc)) / len(I_mc)

    moments_model = compute_angular_moments(theta_mc, phi_mc, L_max, model_weights)

    # Residuals and χ² diagnostic
    # Statistical uncertainty on data moments: σ ≈ 1/√N_data
    n_data = len(theta_data)
    sigma = 1.0 / np.sqrt(n_data)

    residuals: Dict[Tuple[int, int], float] = {}
    pulls: Dict[Tuple[int, int], float] = {}
    chi2_sum = 0.0
    n_moments = 0

    for (L, M), y_data in moments_data.items():
        y_model = moments_model.get((L, M), 0.0)
        res = abs(y_data - y_model)
        pull = res / max(sigma, 1e-30)
        residuals[(L, M)] = res
        pulls[(L, M)] = pull
        chi2_sum += pull**2
        n_moments += 1

    return {
        "moments_data": moments_data,
        "moments_model": moments_model,
        "residuals": residuals,
        "pulls": pulls,
        "chi2": chi2_sum,
        "ndf": n_moments,
        "chi2_per_ndf": chi2_sum / max(n_moments, 1),
        "L_max": L_max,
        "n_data": n_data,
        "n_mc": len(theta_mc),
        "sigma": sigma,
    }


# ════════════════════════════════════════════════════════════════════════════════
# BEAM ASYMMETRY — POLARIZATION OBSERVABLE
# ════════════════════════════════════════════════════════════════════════════════

class PolarizedIntensityModel:
    """Beam asymmetry Σ for linearly polarized photoproduction.

    For a linearly polarized photon beam with polarization angle Φ_γ,
    the intensity becomes:

        I(τ, Φ_γ) = I_0(τ) [1 − P_γ Σ(τ) cos(2Φ_γ)]

    where:
        I_0 = unpolarized intensity
        P_γ = degree of linear polarization
        Σ(τ) = beam asymmetry observable = [I(0°) − I(90°)] / [I(0°) + I(90°)]

    In terms of reflectivity amplitudes:
        I_0(τ) = |A_+(τ)|² + |A_−(τ)|²
        I_0(τ) Σ(τ) = |A_+(τ)|² − |A_−(τ)|²

    so:
        Σ(τ) = [|A_+|² − |A_−|²] / [|A_+|² + |A_−|²]
    """

    def __init__(
        self,
        basis: BasisAmplitudes,
    ) -> None:
        self.basis = basis
        self.wave_set = basis.wave_set
        self.device = basis.device
        self._eps_list = self.wave_set.reflectivities
        self._torch_masks: Dict[int, Tensor] = {}
        for eps in self._eps_list:
            self._torch_masks[eps] = self.wave_set.epsilon_mask(eps, self.device)

    def amplitude(self, V: Tensor, epsilon: int) -> Tensor:
        """A_ε(τ_i) = Σ_{α: ε_α=ε} V_α ψ_α(τ_i)."""
        mask = self._torch_masks[epsilon]
        return self.basis.psi[:, mask] @ V[mask]

    def unpolarized_intensity(self, V: Tensor) -> Tensor:
        """I_0(τ) = Σ_ε |A_ε(τ)|²."""
        I = torch.zeros(self.basis.n_events, dtype=torch.float64, device=self.device)
        for eps in self._eps_list:
            A = self.amplitude(V, eps)
            I = I + (A * A.conj()).real
        return I

    def polarized_intensity(
        self,
        V: Tensor,
        phi_gamma: float,
        P_gamma: float = 1.0,
    ) -> Tensor:
        """I(τ, Φ_γ) = I_0 [1 − P_γ Σ cos(2Φ_γ)].

        Parameters
        ----------
        V : production amplitudes.
        phi_gamma : beam polarization angle (radians).
        P_gamma : degree of linear polarization (0 to 1).
        """
        I_plus = torch.zeros(self.basis.n_events, dtype=torch.float64, device=self.device)
        I_minus = torch.zeros(self.basis.n_events, dtype=torch.float64, device=self.device)

        for eps in self._eps_list:
            A = self.amplitude(V, eps)
            Isq = (A * A.conj()).real
            if eps > 0:
                I_plus = I_plus + Isq
            else:
                I_minus = I_minus + Isq

        I_0 = I_plus + I_minus
        sigma = torch.where(
            I_0 > 1e-300,
            (I_plus - I_minus) / I_0,
            torch.zeros_like(I_0),
        )
        return I_0 * (1.0 - P_gamma * sigma * math.cos(2.0 * phi_gamma))

    def beam_asymmetry(self, V: Tensor) -> Tensor:
        """Σ(τ) = [|A_+|² − |A_−|²] / [|A_+|² + |A_−|²].

        Returns (n_events,) float64 in range [-1, 1].
        """
        I_plus = torch.zeros(self.basis.n_events, dtype=torch.float64, device=self.device)
        I_minus = torch.zeros(self.basis.n_events, dtype=torch.float64, device=self.device)

        for eps in self._eps_list:
            A = self.amplitude(V, eps)
            Isq = (A * A.conj()).real
            if eps > 0:
                I_plus = I_plus + Isq
            else:
                I_minus = I_minus + Isq

        I_0 = I_plus + I_minus
        return torch.where(
            I_0 > 1e-300,
            (I_plus - I_minus) / I_0,
            torch.zeros_like(I_0),
        )


def beam_asymmetry_sensitivity_test(
    n_events: int = 5000,
    n_generated: int = 200_000,
    n_starts: int = 20,
    helicity: float = 0.5,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> Dict[str, Any]:
    """Test how beam asymmetry Σ disambiguates phase ambiguities.

    Procedure:
        1. Create model with both reflectivities (ε=±1), J_max=1.5
        2. Generate data from known V_true with Σ ≠ 0
        3. Fit WITHOUT polarization (unpolarized likelihood)
        4. Fit WITH polarization (Σ-weighted likelihood)
        5. Compare phase recovery RMSE

    The polarized fit should have significantly better phase resolution
    because beam asymmetry constrains the relative sign between
    ε=+1 and ε=−1 amplitudes.
    """
    rng = np.random.default_rng(seed)

    # Two-reflectivity model
    ws = build_wave_set(j_max=1.5, reflectivities=(+1, -1))
    n_amp = ws.n_amplitudes

    # True amplitudes with significant interference
    V_true_np = rng.standard_normal(n_amp) + 1j * rng.standard_normal(n_amp)
    V_true_np /= np.linalg.norm(V_true_np)

    # Generate data
    gen = SyntheticDataGenerator(ws, V_true_np, helicity=helicity, seed=seed)
    data = gen.generate(n_events, n_generated, device=device)

    # Build basis
    basis_data = BasisAmplitudes(ws, data["theta_data"], data["phi_data"], helicity, device)
    basis_mc = BasisAmplitudes(ws, data["theta_mc"], data["phi_mc"], helicity, device)
    gram = GramMatrix(basis_mc, data["n_generated"])

    # ── Unpolarized fit ──────────────────────────────────────────────
    model_data = IntensityModel(basis_data)
    nll_unpol = ExtendedLikelihood(
        model_data, None, gram, data["n_data"], data["n_generated"]
    )
    fitter_unpol = LBFGSFitter(nll_unpol, max_iter=500)
    ms_unpol = fitter_unpol.multi_start_fit(n_starts=n_starts, seed_base=200)

    V_unpol = ms_unpol["best_fit"]["V_best"].detach().cpu().numpy()

    # ── Polarized fit ────────────────────────────────────────────────
    # Compute true Σ at data points
    pol_model_data = PolarizedIntensityModel(basis_data)
    V_true_torch = torch.tensor(V_true_np, dtype=torch.complex128, device=device)
    sigma_true = pol_model_data.beam_asymmetry(V_true_torch).detach().cpu().numpy()

    # Polarized likelihood = NLL + λ Σ_i (Σ_fit − Σ_data)²
    class _PolarizedLikelihood:
        """NLL + beam asymmetry penalty for polarization-sensitive fits."""

        def __init__(self, base_nll: ExtendedLikelihood, pol_model: PolarizedIntensityModel,
                     sigma_data: np.ndarray, lambda_pol: float) -> None:
            self.base_nll = base_nll
            self.pol_model = pol_model
            self.sigma_data_torch = torch.tensor(sigma_data, dtype=torch.float64, device=device)
            self.lambda_pol = lambda_pol
            self.intensity_data = base_nll.intensity_data
            self.device = base_nll.device
            self._eval_count = 0

        def __call__(self, V: Tensor) -> Tensor:
            self._eval_count += 1
            nll = self.base_nll(V)
            sigma_fit = self.pol_model.beam_asymmetry(V)
            penalty = self.lambda_pol * torch.sum(
                (sigma_fit - self.sigma_data_torch) ** 2
            )
            return nll + penalty

        @property
        def eval_count(self) -> int:
            return self._eval_count

        def reset_count(self) -> None:
            self._eval_count = 0

    lambda_pol = float(data["n_data"]) / 10.0
    pol_likelihood = _PolarizedLikelihood(nll_unpol, pol_model_data, sigma_true, lambda_pol)

    fitter_pol = LBFGSFitter.__new__(LBFGSFitter)
    fitter_pol.likelihood = pol_likelihood
    fitter_pol.max_iter = 500
    fitter_pol.tolerance = 1e-10
    fitter_pol.n_amp = n_amp
    fitter_pol.device = device

    ms_pol = fitter_pol.multi_start_fit(n_starts=n_starts, seed_base=300)
    V_pol = ms_pol["best_fit"]["V_best"].detach().cpu().numpy()

    # ── Compare phase recovery ───────────────────────────────────────
    def _phase_rmse(V_fit: np.ndarray, V_true: np.ndarray) -> float:
        y_true = np.abs(V_true) ** 2
        idx = int(np.argmax(y_true))
        shift = np.exp(1j * (np.angle(V_true[idx]) - np.angle(V_fit[idx])))
        V_al = V_fit * shift
        diffs = np.angle(V_true * V_al.conj())
        return float(np.sqrt(np.mean(diffs**2)))

    def _yield_rmse(V_fit: np.ndarray, V_true: np.ndarray) -> float:
        y_true = np.abs(V_true) ** 2
        y_fit = np.abs(V_fit) ** 2
        y_true = y_true / y_true.sum()
        y_fit = y_fit / y_fit.sum()
        return float(np.sqrt(np.mean((y_true - y_fit)**2)))

    rmse_phase_unpol = _phase_rmse(V_unpol, V_true_np)
    rmse_phase_pol = _phase_rmse(V_pol, V_true_np)
    rmse_yield_unpol = _yield_rmse(V_unpol, V_true_np)
    rmse_yield_pol = _yield_rmse(V_pol, V_true_np)

    # Beam asymmetry prediction from each fit
    V_unpol_torch = torch.tensor(V_unpol, dtype=torch.complex128, device=device)
    V_pol_torch = torch.tensor(V_pol, dtype=torch.complex128, device=device)

    sigma_unpol = pol_model_data.beam_asymmetry(V_unpol_torch).detach().cpu().numpy()
    sigma_pol = pol_model_data.beam_asymmetry(V_pol_torch).detach().cpu().numpy()

    sigma_rmse_unpol = float(np.sqrt(np.mean((sigma_true - sigma_unpol)**2)))
    sigma_rmse_pol = float(np.sqrt(np.mean((sigma_true - sigma_pol)**2)))

    return {
        "n_amp": n_amp,
        "n_data": data["n_data"],
        "n_generated": data["n_generated"],
        "n_starts": n_starts,
        # Unpolarized results
        "nll_unpol": ms_unpol["best_nll"],
        "basin_frac_unpol": ms_unpol["basin_fraction"],
        "phase_rmse_unpol": rmse_phase_unpol,
        "yield_rmse_unpol": rmse_yield_unpol,
        "sigma_rmse_unpol": sigma_rmse_unpol,
        # Polarized results
        "nll_pol": ms_pol["best_nll"],
        "basin_frac_pol": ms_pol["basin_fraction"],
        "phase_rmse_pol": rmse_phase_pol,
        "yield_rmse_pol": rmse_yield_pol,
        "sigma_rmse_pol": sigma_rmse_pol,
        # Improvement ratios
        "phase_improvement": rmse_phase_unpol / max(rmse_phase_pol, 1e-15),
        "sigma_improvement": sigma_rmse_unpol / max(sigma_rmse_pol, 1e-15),
        # Raw data for plotting
        "theta_data": data["theta_data"],
        "sigma_true": sigma_true,
        "sigma_unpol": sigma_unpol,
        "sigma_pol": sigma_pol,
        "V_true": V_true_np,
        "V_unpol": V_unpol,
        "V_pol": V_pol,
    }


# ════════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP UNCERTAINTY ESTIMATION
# ════════════════════════════════════════════════════════════════════════════════

def bootstrap_uncertainty(
    wave_set: WaveSet,
    theta_data: np.ndarray,
    phi_data: np.ndarray,
    theta_mc: np.ndarray,
    phi_mc: np.ndarray,
    n_generated: int,
    V_seed: Tensor,
    n_bootstrap: int = 200,
    max_iter: int = 300,
    helicity: float = 0.5,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> Dict[str, Any]:
    """Estimate parameter uncertainties via bootstrap resampling.

    Procedure:
        1. For each of n_bootstrap iterations:
            a. Resample data events WITH replacement
            b. Refit from V_seed (warm start)
            c. Store fitted parameters
        2. Compute standard deviation of fitted parameters across resamples

    This gives a non-parametric uncertainty estimate that accounts for
    correlations and non-Gaussianity in the likelihood surface.

    Parameters
    ----------
    wave_set : WaveSet for the fit.
    theta_data, phi_data : original data event kinematics.
    theta_mc, phi_mc : MC event kinematics (fixed across resamples).
    n_generated : total generated events.
    V_seed : starting point for each refit (typically best-fit V).
    n_bootstrap : number of bootstrap resamples.
    max_iter : max L-BFGS iterations per refit.

    Returns
    -------
    dict with per-parameter uncertainties, distributions, and diagnostics.
    """
    rng = np.random.default_rng(seed)
    n_data = len(theta_data)
    n_amp = wave_set.n_amplitudes

    # Pre-build MC basis and Gram (fixed across resamples)
    basis_mc = BasisAmplitudes(wave_set, theta_mc, phi_mc, helicity, device)
    gram = GramMatrix(basis_mc, n_generated)

    V_samples: List[np.ndarray] = []
    nll_samples: List[float] = []
    converged_count = 0
    t0 = time.perf_counter()

    for b in range(n_bootstrap):
        # Resample data with replacement
        idx = rng.choice(n_data, size=n_data, replace=True)
        theta_b = theta_data[idx]
        phi_b = phi_data[idx]

        # Build data basis for this resample
        basis_b = BasisAmplitudes(wave_set, theta_b, phi_b, helicity, device)
        model_b = IntensityModel(basis_b)
        nll_b = ExtendedLikelihood(model_b, None, gram, n_data, n_generated)

        # Warm-start fit from seed
        fitter_b = LBFGSFitter(nll_b, max_iter=max_iter)
        result_b = fitter_b.fit(V_init=V_seed, seed=None)

        V_b = result_b["V_best"].detach().cpu().numpy()
        V_samples.append(V_b)
        nll_samples.append(result_b["nll"])
        if result_b["converged"]:
            converged_count += 1

    elapsed = time.perf_counter() - t0
    V_stack = np.stack(V_samples)  # (n_bootstrap, n_amp) complex

    # Align phases across bootstrap samples (fix phase of reference amplitude)
    V_ref = V_seed.detach().cpu().numpy()
    yield_ref = np.abs(V_ref) ** 2
    idx_ref = int(np.argmax(yield_ref))

    for i in range(n_bootstrap):
        shift = np.exp(1j * (np.angle(V_ref[idx_ref]) - np.angle(V_stack[i, idx_ref])))
        V_stack[i] *= shift

    # Statistics
    yields_all = np.abs(V_stack) ** 2                       # (n_boot, n_amp)
    yields_rel = yields_all / yields_all.sum(axis=1, keepdims=True)  # relative
    phases_all = np.angle(V_stack)                           # (n_boot, n_amp)

    yield_mean = yields_rel.mean(axis=0)                     # (n_amp,)
    yield_std = yields_rel.std(axis=0)                       # (n_amp,)
    phase_mean = np.angle(np.mean(V_stack, axis=0))          # circular mean
    # Circular standard deviation
    R = np.abs(np.mean(np.exp(1j * phases_all), axis=0))
    R_clipped = np.clip(R, 1e-15, 1.0 - 1e-15)
    phase_std = np.sqrt(-2.0 * np.log(R_clipped))

    return {
        "n_bootstrap": n_bootstrap,
        "n_amp": n_amp,
        "n_data": n_data,
        "converged_fraction": converged_count / n_bootstrap,
        "time_s": elapsed,
        # Per-amplitude uncertainties
        "yield_mean": yield_mean,
        "yield_std": yield_std,
        "phase_mean": phase_mean,
        "phase_std": phase_std,
        # Raw distributions
        "yields_all": yields_rel,       # (n_bootstrap, n_amp)
        "phases_all": phases_all,       # (n_bootstrap, n_amp)
        "nll_samples": np.array(nll_samples),
        # Correlation matrix of yields
        "yield_corr": np.corrcoef(yields_rel.T) if n_amp > 1 else np.array([[1.0]]),
    }


# ════════════════════════════════════════════════════════════════════════════════
# COUPLED-CHANNEL PWA
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class ChannelConfig:
    """Configuration for a single decay channel in coupled-channel analysis.

    Each channel corresponds to a distinct final state (e.g. ηπ, η'π)
    observed through a different angular distribution but sharing the
    same resonance content (production amplitudes V).

    Attributes
    ----------
    name : str           Channel label (e.g. "ηπ₀", "η'π₀").
    wave_set : WaveSet   Partial waves accessible in this channel.
    helicity : float     Decay helicity (determines ψ basis functions).
    theta_data : ndarray Polar angles of data events.
    phi_data : ndarray   Azimuthal angles of data events.
    theta_mc : ndarray   Polar angles of accepted MC events.
    phi_mc : ndarray     Azimuthal angles of accepted MC events.
    n_generated : int    Total MC events generated before acceptance.
    """

    name: str
    wave_set: WaveSet
    helicity: float
    theta_data: np.ndarray
    phi_data: np.ndarray
    theta_mc: np.ndarray
    phi_mc: np.ndarray
    n_generated: int


class CoupledChannelSystem:
    """Multi-channel PWA system with shared production amplitudes.

    Multiple decay channels observe the same resonance content through
    different angular distributions (different ψ basis functions due to
    different helicities and/or different decay kinematics). The
    production amplitudes V are shared across channels for waves with
    the same quantum numbers (J, M, ε).

    The joint likelihood is:

        -ln L_total(V) = Σ_c [-ln L_c(V_c)]

    where V_c is the subset of the global V vector relevant to channel c,
    obtained by matching wave quantum numbers (ε, J, M).

    Parameters
    ----------
    channel_configs : list of ChannelConfig
        One per decay channel.
    device : torch.device
        Compute device.
    """

    def __init__(
        self,
        channel_configs: List[ChannelConfig],
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.device = device
        self.n_channels = len(channel_configs)

        # Build global wave roster: (ε, J, M) → global index
        self._global_labels: Dict[Tuple[int, float, float], int] = {}
        gidx = 0
        for cfg in channel_configs:
            for w in cfg.wave_set.waves:
                key = (w.epsilon, w.j, w.m)
                if key not in self._global_labels:
                    self._global_labels[key] = gidx
                    gidx += 1
        self.n_amp_global = gidx

        # Per-channel: build basis, model, gram, likelihood, and index map
        self.channels: List[Dict[str, Any]] = []
        for cfg in channel_configs:
            # Map local amplitude α → global V index
            # With n_components=1, local α coincides with wave index b
            local_to_global: List[int] = []
            for b, w in enumerate(cfg.wave_set.waves):
                for k in range(w.n_components):
                    key = (w.epsilon, w.j, w.m)
                    local_to_global.append(self._global_labels[key])
            local_to_global_arr = np.array(local_to_global, dtype=np.int64)
            local_to_global_t = torch.tensor(
                local_to_global_arr, dtype=torch.long, device=device
            )

            basis_data = BasisAmplitudes(
                cfg.wave_set, cfg.theta_data, cfg.phi_data, cfg.helicity, device
            )
            basis_mc = BasisAmplitudes(
                cfg.wave_set, cfg.theta_mc, cfg.phi_mc, cfg.helicity, device
            )
            model_data = IntensityModel(basis_data)
            gram = GramMatrix(basis_mc, cfg.n_generated)
            n_data = len(cfg.theta_data)
            likelihood = ExtendedLikelihood(
                model_data, None, gram, n_data, cfg.n_generated
            )

            self.channels.append({
                "name": cfg.name,
                "n_local": cfg.wave_set.n_amplitudes,
                "local_to_global": local_to_global_arr,
                "local_to_global_t": local_to_global_t,
                "likelihood": likelihood,
                "n_data": n_data,
                "wave_set": cfg.wave_set,
            })

    def joint_nll(self, V_global: Tensor) -> Tensor:
        """Compute total NLL across all channels.

        Parameters
        ----------
        V_global : (n_amp_global,) complex128 with requires_grad.

        Returns
        -------
        Scalar tensor — sum of per-channel negative log-likelihoods.
        """
        total = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        for ch in self.channels:
            V_local = V_global[ch["local_to_global_t"]]
            nll_ch = ch["likelihood"](V_local)
            total = total + nll_ch
        return total

    def fit_joint(
        self,
        n_starts: int = 20,
        max_iter: int = 500,
        seed_base: int = 0,
    ) -> Dict[str, Any]:
        """Fit shared V across all channels simultaneously.

        Uses LBFGSFitter with a joint likelihood that sums per-channel
        NLLs. The global V vector is optimised; each channel sees only
        the subset of amplitudes for waves it contains.

        Returns
        -------
        dict with best_fit, all_fits, stability metrics (same schema
        as LBFGSFitter.multi_start_fit).
        """
        system = self

        class _JointLikelihood:
            """Duck-typed likelihood for LBFGSFitter compatibility."""

            def __init__(self_inner) -> None:
                self_inner.device = system.device
                self_inner._eval_count = 0

            def __call__(self_inner, V: Tensor) -> Tensor:
                self_inner._eval_count += 1
                return system.joint_nll(V)

            @property
            def eval_count(self_inner) -> int:
                return self_inner._eval_count

            def reset_count(self_inner) -> None:
                self_inner._eval_count = 0

        joint_like = _JointLikelihood()
        fitter = LBFGSFitter.__new__(LBFGSFitter)
        fitter.likelihood = joint_like
        fitter.max_iter = max_iter
        fitter.tolerance = 1e-10
        fitter.n_amp = self.n_amp_global
        fitter.device = self.device

        return fitter.multi_start_fit(n_starts=n_starts, seed_base=seed_base)

    def fit_independent(
        self,
        n_starts: int = 20,
        max_iter: int = 500,
        seed_base: int = 0,
    ) -> List[Dict[str, Any]]:
        """Fit each channel independently (no parameter sharing).

        Returns
        -------
        List of multi_start_fit results, one per channel.
        """
        results: List[Dict[str, Any]] = []
        for i, ch in enumerate(self.channels):
            fitter = LBFGSFitter(ch["likelihood"], max_iter=max_iter)
            ms = fitter.multi_start_fit(
                n_starts=n_starts,
                seed_base=seed_base + i * 1000,
            )
            results.append(ms)
        return results


def coupled_channel_test(
    n_data_ch1: int = 5000,
    n_data_ch2: int = 3000,
    n_generated: int = 200_000,
    n_starts: int = 20,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> Dict[str, Any]:
    """Validate coupled-channel fitting vs independent channel fits.

    Setup:
        Channel 1 ("ηπ"):  J_max=2.5, helicity=0.5, 5000 events
        Channel 2 ("η'π"): J_max=1.5, helicity=1.5, 3000 events
        Shared production amplitudes for overlapping waves (J ≤ 1.5).

    The joint fit uses information from both channels simultaneously to
    constrain shared parameters, whereas independent fits each see only
    one channel. The joint fit should recover the shared-wave parameters
    at least as well as either independent fit.

    Returns
    -------
    dict with per-fit metrics, improvement ratios, raw amplitudes.
    """
    rng = np.random.default_rng(seed)

    # Channel 1: ηπ — full wave set up to J=2.5
    ws1 = build_wave_set(j_max=2.5, reflectivities=(+1,), n_components=1)
    # Channel 2: η'π — restricted to J≤1.5 (phase-space filter)
    ws2 = build_wave_set(j_max=1.5, reflectivities=(+1,), n_components=1)

    n_amp_global = ws1.n_amplitudes  # ws2 ⊂ ws1
    n_amp_ch2 = ws2.n_amplitudes

    # True production amplitudes (shared across channels)
    V_true_np = rng.standard_normal(n_amp_global) + 1j * rng.standard_normal(n_amp_global)
    V_true_np /= np.linalg.norm(V_true_np)

    # Channel 1 data: full wave set, helicity=0.5
    gen1 = SyntheticDataGenerator(ws1, V_true_np, helicity=0.5, seed=seed)
    data1 = gen1.generate(n_data_ch1, n_generated, device=device)

    # Channel 2 data: restricted wave set, helicity=1.5, different acceptance
    V_true_ch2 = V_true_np[:n_amp_ch2]
    gen2 = SyntheticDataGenerator(
        ws2, V_true_ch2, helicity=1.5,
        acceptance_fn=lambda t, p: 0.6 + 0.1 * np.cos(t) - 0.15 * np.sin(p),
        seed=seed + 100,
    )
    data2 = gen2.generate(n_data_ch2, n_generated, device=device)

    # Build coupled-channel system
    cfg1 = ChannelConfig(
        name="ηπ", wave_set=ws1, helicity=0.5,
        theta_data=data1["theta_data"], phi_data=data1["phi_data"],
        theta_mc=data1["theta_mc"], phi_mc=data1["phi_mc"],
        n_generated=data1["n_generated"],
    )
    cfg2 = ChannelConfig(
        name="η'π", wave_set=ws2, helicity=1.5,
        theta_data=data2["theta_data"], phi_data=data2["phi_data"],
        theta_mc=data2["theta_mc"], phi_mc=data2["phi_mc"],
        n_generated=data2["n_generated"],
    )
    system = CoupledChannelSystem([cfg1, cfg2], device=device)

    # ── Joint fit (both channels simultaneously) ─────────────────────
    joint_result = system.fit_joint(n_starts=n_starts, seed_base=500)
    V_joint = joint_result["best_fit"]["V_best"].detach().cpu().numpy()

    # ── Independent fits (each channel alone) ────────────────────────
    indep_results = system.fit_independent(n_starts=n_starts, seed_base=600)
    V_indep_ch1 = indep_results[0]["best_fit"]["V_best"].detach().cpu().numpy()
    V_indep_ch2 = indep_results[1]["best_fit"]["V_best"].detach().cpu().numpy()

    # ── Compare parameter recovery ───────────────────────────────────
    def _yield_rmse(V_fit: np.ndarray, V_true: np.ndarray) -> float:
        y_fit = np.abs(V_fit) ** 2
        y_true = np.abs(V_true) ** 2
        y_fit = y_fit / max(y_fit.sum(), 1e-30)
        y_true = y_true / max(y_true.sum(), 1e-30)
        return float(np.sqrt(np.mean((y_fit - y_true) ** 2)))

    def _phase_rmse(V_fit: np.ndarray, V_true: np.ndarray) -> float:
        y_true = np.abs(V_true) ** 2
        idx = int(np.argmax(y_true))
        shift = np.exp(1j * (np.angle(V_true[idx]) - np.angle(V_fit[idx])))
        V_al = V_fit * shift
        diffs = np.angle(V_true * V_al.conj())
        return float(np.sqrt(np.mean(diffs ** 2)))

    # Joint: all amplitudes
    yield_rmse_joint = _yield_rmse(V_joint, V_true_np)
    phase_rmse_joint = _phase_rmse(V_joint, V_true_np)

    # Joint: shared waves only (first n_amp_ch2)
    yield_rmse_joint_shared = _yield_rmse(V_joint[:n_amp_ch2], V_true_ch2)
    phase_rmse_joint_shared = _phase_rmse(V_joint[:n_amp_ch2], V_true_ch2)

    # Independent ch1: all amplitudes
    yield_rmse_ch1 = _yield_rmse(V_indep_ch1, V_true_np)
    phase_rmse_ch1 = _phase_rmse(V_indep_ch1, V_true_np)

    # Independent ch2: shared waves only
    yield_rmse_ch2 = _yield_rmse(V_indep_ch2, V_true_ch2)
    phase_rmse_ch2 = _phase_rmse(V_indep_ch2, V_true_ch2)

    return {
        "n_amp_global": n_amp_global,
        "n_amp_ch1": ws1.n_amplitudes,
        "n_amp_ch2": ws2.n_amplitudes,
        "n_data_ch1": data1["n_data"],
        "n_data_ch2": data2["n_data"],
        # Joint fit metrics
        "joint_nll": joint_result["best_nll"],
        "joint_basin_frac": joint_result["basin_fraction"],
        "joint_yield_rmse": yield_rmse_joint,
        "joint_phase_rmse": phase_rmse_joint,
        "joint_yield_rmse_shared": yield_rmse_joint_shared,
        "joint_phase_rmse_shared": phase_rmse_joint_shared,
        # Independent channel 1
        "ch1_nll": indep_results[0]["best_nll"],
        "ch1_basin_frac": indep_results[0]["basin_fraction"],
        "ch1_yield_rmse": yield_rmse_ch1,
        "ch1_phase_rmse": phase_rmse_ch1,
        # Independent channel 2
        "ch2_nll": indep_results[1]["best_nll"],
        "ch2_basin_frac": indep_results[1]["basin_fraction"],
        "ch2_yield_rmse": yield_rmse_ch2,
        "ch2_phase_rmse": phase_rmse_ch2,
        # Improvement ratios
        "yield_improvement_vs_ch1": yield_rmse_ch1 / max(yield_rmse_joint, 1e-15),
        "yield_improvement_vs_ch2": yield_rmse_ch2 / max(yield_rmse_joint_shared, 1e-15),
        # Raw data for plotting
        "V_true": V_true_np,
        "V_joint": V_joint,
        "V_indep_ch1": V_indep_ch1,
        "V_indep_ch2": V_indep_ch2,
        "n_starts": n_starts,
    }


# ════════════════════════════════════════════════════════════════════════════════
# BREIT-WIGNER AMPLITUDE
# ════════════════════════════════════════════════════════════════════════════════

class BreitWigner:
    """Non-relativistic Breit-Wigner amplitude.

    BW(m; m₀, Γ₀) = 1 / (m₀² − m² − i·m₀·Γ₀)

    Simplified form without barrier factors, suitable for demonstrating
    mass-dependent fitting methodology. The phase motion and peak
    position are physically correct; only the mass-dependent width
    is omitted (constant Γ approximation).

    Parameters
    ----------
    m0 : float      Resonance pole mass (GeV).
    gamma0 : float  Resonance total width (GeV).
    """

    def __init__(self, m0: float, gamma0: float) -> None:
        self.m0 = m0
        self.gamma0 = gamma0

    def amplitude(self, m: np.ndarray) -> np.ndarray:
        """Complex BW amplitude at mass values m."""
        return 1.0 / (self.m0 ** 2 - m ** 2 - 1j * self.m0 * self.gamma0)

    def intensity(self, m: np.ndarray) -> np.ndarray:
        """|BW(m)|² intensity spectrum."""
        return np.abs(self.amplitude(m)) ** 2


# ════════════════════════════════════════════════════════════════════════════════
# MASS-DEPENDENT FIT
# ════════════════════════════════════════════════════════════════════════════════

def mass_dependent_fit(
    n_mass_bins: int = 20,
    m_range: Tuple[float, float] = (0.8, 2.0),
    n_events_per_bin: int = 8000,
    n_generated_per_bin: int = 300_000,
    n_starts: int = 25,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> Dict[str, Any]:
    """Mass-dependent PWA: fit mass-binned amplitudes, extract BW resonances.

    Two-stage procedure:
        Stage 1 — Mass-binned fits:
            At each mass bin, generate synthetic data with mass-dependent
            V(m) = Σ_r c_{br} BW_r(m), then run standard PWA fit to
            extract V_fit(m).

        Stage 2 — Resonance extraction:
            Fit Breit-Wigner |BW(m)|² to the extracted per-wave-family
            intensity spectra |V(m)|² to recover resonance masses and
            widths.

    Uses a minimal 2-amplitude wave set (one S-wave, one D-wave) with
    distinct M projections to maximise angular sensitivity and eliminate
    the M-substate discrete ambiguity that plagues under-constrained fits.

    Model:
        S-wave (J=0.5, M=+0.5) ← R₁ at m₀=1.3 GeV, Γ₀=0.35 GeV (broad)
        D-wave (J=1.5, M=+0.5) ← R₂ at m₀=1.7 GeV, Γ₀=0.10 GeV (narrow)

    Returns
    -------
    dict with true/fitted resonance parameters, per-bin results, spectra.
    """
    rng = np.random.default_rng(seed)

    # True resonance parameters
    m0_R1, gamma_R1 = 1.3, 0.35   # broad S-wave resonance
    m0_R2, gamma_R2 = 1.7, 0.10   # narrow D-wave resonance
    bw_R1 = BreitWigner(m0_R1, gamma_R1)
    bw_R2 = BreitWigner(m0_R2, gamma_R2)

    # Minimal wave set: 1 S-wave + 1 D-wave  →  2 complex amplitudes
    # Avoids M-substate ambiguity inherent in full wave sets without
    # polarisation constraints.
    ws = WaveSet([
        Wave(j=0.5, m=0.5, epsilon=+1, n_components=1, label="S J=0.5 M=+0.5"),
        Wave(j=1.5, m=0.5, epsilon=+1, n_components=1, label="D J=1.5 M=+0.5"),
    ])
    n_amp = ws.n_amplitudes  # 2

    # Coupling matrix: c[wave_idx, resonance_idx]
    c_coupling_r1 = 1.0 + 0.3j   # S-wave coupling
    c_coupling_r2 = 0.8 - 0.2j   # D-wave coupling
    c_matrix = np.zeros((n_amp, 2), dtype=np.complex128)
    c_matrix[0, 0] = c_coupling_r1   # alpha=0 (S) ← R₁
    c_matrix[1, 1] = c_coupling_r2   # alpha=1 (D) ← R₂

    # Mass bin centers
    m_centers = np.linspace(m_range[0], m_range[1], n_mass_bins)

    # ── Stage 1: Per-bin PWA fits with warm-start chaining ──────────
    # Standard mass-dependent technique: fit the first bin with multi-start,
    # then use the previous bin's solution as warm-start for the next.
    # Additional random starts guard against chain-propagated local minima.
    bin_results: List[Dict[str, Any]] = []
    V_prev: Optional[Tensor] = None

    for i_bin, m_val in enumerate(m_centers):
        # True amplitudes at this mass
        bw_vals = np.array([
            bw_R1.amplitude(np.array([m_val]))[0],
            bw_R2.amplitude(np.array([m_val]))[0],
        ])
        V_true_m = c_matrix @ bw_vals   # (n_amp,) complex
        norm = np.linalg.norm(V_true_m)
        if norm > 1e-30:
            V_true_m /= norm

        # Generate data at this mass
        gen = SyntheticDataGenerator(
            ws, V_true_m, helicity=0.5, seed=seed + i_bin * 100
        )
        data = gen.generate(n_events_per_bin, n_generated_per_bin, device=device)

        # Build engine
        basis_data = BasisAmplitudes(
            ws, data["theta_data"], data["phi_data"], helicity=0.5, device=device
        )
        basis_mc = BasisAmplitudes(
            ws, data["theta_mc"], data["phi_mc"], helicity=0.5, device=device
        )
        model_data = IntensityModel(basis_data)
        gram = GramMatrix(basis_mc, data["n_generated"])
        nll_obj = ExtendedLikelihood(
            model_data, None, gram, data["n_data"], data["n_generated"]
        )
        fitter = LBFGSFitter(nll_obj, max_iter=300)

        if V_prev is None:
            # First bin: full multi-start exploration
            ms = fitter.multi_start_fit(
                n_starts=n_starts, seed_base=seed + i_bin * 1000
            )
            V_fit_t = ms["best_fit"]["V_best"]
            best_nll = ms["best_nll"]
            basin_frac = ms["basin_fraction"]
        else:
            # Subsequent bins: warm-start from previous + random guard starts
            fit_warm = fitter.fit(V_init=V_prev)
            random_fits = [
                fitter.fit(seed=seed + i_bin * 1000 + s)
                for s in range(max(3, n_starts // 3))
            ]
            all_fits = [fit_warm] + random_fits
            best = min(all_fits, key=lambda f: f["nll"])
            V_fit_t = best["V_best"]
            best_nll = best["nll"]
            n_near = sum(
                1 for f in all_fits if abs(f["nll"] - best_nll) < 1.0
            )
            basin_frac = n_near / len(all_fits)

        V_prev = V_fit_t
        V_fit = V_fit_t.detach().cpu().numpy()

        # Per-wave yields: alpha=0 is S-wave, alpha=1 is D-wave
        yield_s_fit = float(np.abs(V_fit[0]) ** 2)
        yield_d_fit = float(np.abs(V_fit[1]) ** 2)
        yield_s_true = float(np.abs(V_true_m[0]) ** 2)
        yield_d_true = float(np.abs(V_true_m[1]) ** 2)

        bin_results.append({
            "m": m_val,
            "V_true": V_true_m,
            "V_fit": V_fit,
            "nll": best_nll,
            "basin_frac": basin_frac,
            "yield_s_fit": yield_s_fit,
            "yield_d_fit": yield_d_fit,
            "yield_s_true": yield_s_true,
            "yield_d_true": yield_d_true,
            "n_data": data["n_data"],
        })

    # ── Stage 2: Fit BW parameters via wave-fraction model ─────────
    # The extended likelihood normalises each bin, so absolute yields
    # don't carry mass-dependent scale.  The correct observable is the
    # wave *fraction*  f_S(m) = yield_S / (yield_S + yield_D),  which
    # is independent of per-bin normalisation.
    m_vals = np.array([r["m"] for r in bin_results])
    yield_s_fit_arr = np.array([r["yield_s_fit"] for r in bin_results])
    yield_d_fit_arr = np.array([r["yield_d_fit"] for r in bin_results])
    yield_s_true_arr = np.array([r["yield_s_true"] for r in bin_results])
    yield_d_true_arr = np.array([r["yield_d_true"] for r in bin_results])

    total_fit = yield_s_fit_arr + yield_d_fit_arr
    frac_s_fit = yield_s_fit_arr / np.maximum(total_fit, 1e-30)
    total_true = yield_s_true_arr + yield_d_true_arr
    frac_s_true = yield_s_true_arr / np.maximum(total_true, 1e-30)

    def _frac_model(
        m: np.ndarray, m0_1: float, gamma_1: float,
        m0_2: float, gamma_2: float, log_r: float,
    ) -> np.ndarray:
        """S-wave fraction: |BW₁|² / (|BW₁|² + r·|BW₂|²).

        r = |c₂/c₁|² is the relative coupling strength, parameterised
        as exp(log_r) to guarantee positivity.
        """
        r = np.exp(log_r)
        bw1 = 1.0 / (m0_1 ** 2 - m ** 2 - 1j * m0_1 * gamma_1)
        bw2 = 1.0 / (m0_2 ** 2 - m ** 2 - 1j * m0_2 * gamma_2)
        I1 = np.abs(bw1) ** 2
        I2 = np.abs(bw2) ** 2
        return I1 / np.maximum(I1 + r * I2, 1e-30)

    # Simultaneous fit of both resonance parameters
    try:
        popt, pcov = curve_fit(
            _frac_model, m_vals, frac_s_fit,
            p0=[1.2, 0.30, 1.6, 0.15, 0.0],
            bounds=([0.5, 0.01, 0.5, 0.01, -5.0],
                    [3.0, 2.0, 3.0, 2.0, 5.0]),
            maxfev=20000,
        )
        m0_fit_s, gamma_fit_s, m0_fit_d, gamma_fit_d, log_r = popt
        perr = np.sqrt(np.diag(pcov))
        r_fit = float(np.exp(log_r))
    except (RuntimeError, ValueError):
        m0_fit_s = gamma_fit_s = m0_fit_d = gamma_fit_d = float("nan")
        perr = np.array([float("nan")] * 5)
        r_fit = float("nan")

    return {
        "n_mass_bins": n_mass_bins,
        "m_range": m_range,
        "n_events_per_bin": n_events_per_bin,
        "n_amp": n_amp,
        # True parameters
        "m0_true_s": m0_R1,
        "gamma_true_s": gamma_R1,
        "m0_true_d": m0_R2,
        "gamma_true_d": gamma_R2,
        # Fitted resonance parameters (simultaneous fraction fit)
        "m0_fit_s": float(m0_fit_s),
        "gamma_fit_s": float(gamma_fit_s),
        "m0_err_s": float(perr[0]),
        "gamma_err_s": float(perr[1]),
        "m0_fit_d": float(m0_fit_d),
        "gamma_fit_d": float(gamma_fit_d),
        "m0_err_d": float(perr[2]),
        "gamma_err_d": float(perr[3]),
        "r_fit": r_fit,
        # Mass spectra for plotting
        "m_vals": m_vals,
        "yield_s_fit": yield_s_fit_arr,
        "yield_d_fit": yield_d_fit_arr,
        "yield_s_true": yield_s_true_arr,
        "yield_d_true": yield_d_true_arr,
        "frac_s_fit": frac_s_fit,
        "frac_s_true": frac_s_true,
        # Per-bin details
        "bin_results": bin_results,
    }
