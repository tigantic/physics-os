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
from scipy.optimize import minimize
from scipy.special import factorial
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

        Parameters
        ----------
        n_data : int       Target number of data events.
        n_generated : int  Total number of phase-space events generated.

        Returns
        -------
        dict with keys:
            'theta_data', 'phi_data'    — data event kinematics
            'theta_mc', 'phi_mc'        — accepted MC kinematics
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

        # Acceptance
        eta_gen = self.acceptance_fn(theta_gen, phi_gen)
        weights = I_gen * eta_gen

        # Accept/reject for data
        w_max = weights.max() * 1.05
        u = rng.uniform(size=n_generated)
        accepted_mask = u < (weights / w_max)
        accepted_idx = np.nonzero(accepted_mask)[0]

        if len(accepted_idx) < n_data:
            # If not enough, use all accepted
            data_idx = accepted_idx
        else:
            data_idx = rng.choice(accepted_idx, size=n_data, replace=False)

        theta_data = theta_gen[data_idx]
        phi_data = phi_gen[data_idx]

        # MC accepted = all accepted generated events (for normalization)
        theta_mc = theta_gen[accepted_mask]
        phi_mc = phi_gen[accepted_mask]

        return {
            "theta_data": theta_data,
            "phi_data": phi_data,
            "theta_mc": theta_mc,
            "phi_mc": phi_mc,
            "n_data": len(data_idx),
            "n_generated": n_generated,
            "n_mc_accepted": int(accepted_mask.sum()),
            "V_true": self.V_true.copy(),
            "acceptance_rate": float(accepted_mask.mean()),
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
