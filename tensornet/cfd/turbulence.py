"""
RANS Turbulence Modeling
========================

Implements Reynolds-Averaged Navier-Stokes turbulence models
for high-Reynolds number hypersonic flows.

Models:
    1. k-ε (Standard, Realizable)
    2. k-ω SST (Menter's Shear Stress Transport)
    3. Spalart-Allmaras (one-equation)

Key Features:
    - Eddy viscosity hypothesis: τ_t = μ_t (∇u + ∇uᵀ - 2/3 k I)
    - Wall functions for near-wall treatment
    - Compressibility corrections for hypersonic flows
    - Low-Reynolds number damping functions

The RANS equations:
    ∂ρ/∂t + ∇·(ρũ) = 0
    ∂(ρũ)/∂t + ∇·(ρũ⊗ũ) = -∇p̄ + ∇·(τ + τ_t)
    ∂(ρẼ)/∂t + ∇·((ρẼ + p̄)ũ) = ∇·((τ + τ_t)·ũ - q̄ - q_t)

References:
    [1] Wilcox, "Turbulence Modeling for CFD", 3rd Ed., DCW Industries
    [2] Menter, "Two-Equation Eddy-Viscosity Turbulence Models for
        Engineering Applications", AIAA J. 32(8), 1994
    [3] Spalart & Allmaras, "A One-Equation Turbulence Model for
        Aerodynamic Flows", AIAA 92-0439
"""

from dataclasses import dataclass
from enum import Enum

import torch


class TurbulenceModel(Enum):
    """Available turbulence models."""

    LAMINAR = "laminar"
    K_EPSILON = "k-epsilon"
    K_EPSILON_REALIZABLE = "k-epsilon-realizable"
    K_OMEGA_SST = "k-omega-sst"
    SPALART_ALLMARAS = "spalart-allmaras"


# ============================================================================
# Model Constants
# ============================================================================

# k-ε model constants (Standard)
C_MU = 0.09
C_EPS1 = 1.44
C_EPS2 = 1.92
SIGMA_K = 1.0
SIGMA_EPS = 1.3

# k-ω SST model constants
ALPHA_1 = 5.0 / 9.0
ALPHA_2 = 0.44
BETA_1 = 3.0 / 40.0
BETA_2 = 0.0828
BETA_STAR = 0.09
SIGMA_K1 = 0.85
SIGMA_K2 = 1.0
SIGMA_W1 = 0.5
SIGMA_W2 = 0.856

# Spalart-Allmaras constants
CB1 = 0.1355
CB2 = 0.622
SIGMA_SA = 2.0 / 3.0
KAPPA_SA = 0.41
CW1 = CB1 / KAPPA_SA**2 + (1 + CB2) / SIGMA_SA
CW2 = 0.3
CW3 = 2.0
CV1 = 7.1
CT3 = 1.2
CT4 = 0.5


@dataclass
class TurbulentState:
    """
    State for turbulent flow quantities.

    For k-ε: k (TKE), epsilon (dissipation)
    For k-ω: k (TKE), omega (specific dissipation)
    For SA: nu_tilde (modified viscosity)
    """

    k: torch.Tensor | None = None  # Turbulent kinetic energy [m²/s²]
    epsilon: torch.Tensor | None = None  # Dissipation rate [m²/s³]
    omega: torch.Tensor | None = None  # Specific dissipation [1/s]
    nu_tilde: torch.Tensor | None = None  # SA modified viscosity [m²/s]
    mu_t: torch.Tensor | None = None  # Eddy viscosity [Pa·s]

    @property
    def shape(self) -> torch.Size:
        if self.k is not None:
            return self.k.shape
        if self.nu_tilde is not None:
            return self.nu_tilde.shape
        raise ValueError("No turbulent quantities defined")

    @classmethod
    def zeros(cls, shape: tuple[int, int], dtype=torch.float64) -> "TurbulentState":
        """Create zero-initialized turbulent state with all fields."""
        return cls(
            k=torch.zeros(shape, dtype=dtype),
            epsilon=torch.zeros(shape, dtype=dtype),
            omega=torch.zeros(shape, dtype=dtype),
            nu_tilde=torch.zeros(shape, dtype=dtype),
            mu_t=torch.zeros(shape, dtype=dtype),
        )


def wall_distance(Ny: int, Nx: int, dy: float, wall_j: int = 0) -> torch.Tensor:
    """
    Compute distance from wall for each grid point.

    Args:
        Ny, Nx: Grid dimensions
        dy: Grid spacing in y
        wall_j: Wall location index (default 0 = bottom)

    Returns:
        Wall distance field [m]
    """
    j_indices = torch.arange(Ny, dtype=torch.float64).unsqueeze(1).expand(Ny, Nx)
    d = (j_indices - wall_j + 0.5) * dy
    return d


def y_plus(
    rho: torch.Tensor, u_tau: torch.Tensor, y: torch.Tensor, mu: torch.Tensor
) -> torch.Tensor:
    """
    Compute dimensionless wall distance y⁺ = ρ u_τ y / μ.

    Args:
        rho: Density [kg/m³]
        u_tau: Friction velocity [m/s]
        y: Wall distance [m]
        mu: Dynamic viscosity [Pa·s]

    Returns:
        y⁺ dimensionless wall distance
    """
    return rho * u_tau * y / (mu + 1e-30)


def friction_velocity(tau_w: torch.Tensor, rho_w: torch.Tensor) -> torch.Tensor:
    """
    Compute friction velocity u_τ = √(τ_w / ρ).

    Args:
        tau_w: Wall shear stress [Pa]
        rho_w: Wall density [kg/m³]

    Returns:
        Friction velocity [m/s]
    """
    return torch.sqrt(torch.abs(tau_w) / (rho_w + 1e-30))


# ============================================================================
# k-ε Model
# ============================================================================


def k_epsilon_eddy_viscosity(
    rho: torch.Tensor, k: torch.Tensor, epsilon: torch.Tensor
) -> torch.Tensor:
    """
    Compute eddy viscosity for k-ε model.

    μ_t = ρ C_μ k² / ε

    Args:
        rho: Density [kg/m³]
        k: Turbulent kinetic energy [m²/s²]
        epsilon: Dissipation rate [m²/s³]

    Returns:
        Eddy viscosity [Pa·s]
    """
    return rho * C_MU * k**2 / (epsilon + 1e-30)


def k_epsilon_production(
    mu_t: torch.Tensor,
    du_dx: torch.Tensor,
    du_dy: torch.Tensor,
    dv_dx: torch.Tensor,
    dv_dy: torch.Tensor,
) -> torch.Tensor:
    """
    Compute turbulence production term P_k.

    P_k = μ_t * S² where S² = 2(S_ij S_ij)

    For 2D: S² = 2[(∂u/∂x)² + (∂v/∂y)² + 0.5(∂u/∂y + ∂v/∂x)²]

    Args:
        mu_t: Eddy viscosity [Pa·s]
        du_dx, du_dy, dv_dx, dv_dy: Velocity gradients [1/s]

    Returns:
        Production rate [kg/(m·s³)]
    """
    S_xx = du_dx
    S_yy = dv_dy
    S_xy = 0.5 * (du_dy + dv_dx)

    S_squared = 2.0 * (S_xx**2 + S_yy**2 + 2.0 * S_xy**2)

    return mu_t * S_squared


def k_epsilon_source(
    rho: torch.Tensor, k: torch.Tensor, epsilon: torch.Tensor, P_k: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute source terms for k and ε equations.

    S_k = P_k - ρε
    S_ε = C_ε1 (ε/k) P_k - C_ε2 ρ ε²/k

    Args:
        rho: Density [kg/m³]
        k: TKE [m²/s²]
        epsilon: Dissipation [m²/s³]
        P_k: Production [kg/(m·s³)]

    Returns:
        Tuple of (S_k, S_epsilon) source terms
    """
    # k equation source
    S_k = P_k - rho * epsilon

    # ε equation source
    eps_over_k = epsilon / (k + 1e-30)
    S_eps = C_EPS1 * eps_over_k * P_k - C_EPS2 * rho * epsilon * eps_over_k

    return S_k, S_eps


# ============================================================================
# k-ω SST Model
# ============================================================================


def k_omega_blending(
    d: torch.Tensor,
    k: torch.Tensor,
    omega: torch.Tensor,
    rho: torch.Tensor,
    mu: torch.Tensor,
    dk_dx: torch.Tensor,
    dk_dy: torch.Tensor,
    domega_dx: torch.Tensor,
    domega_dy: torch.Tensor,
) -> torch.Tensor:
    """
    Compute SST blending function F1.

    Blends between k-ω (near wall, F1→1) and k-ε (freestream, F1→0).

    Args:
        d: Wall distance [m]
        k, omega: Turbulent quantities
        rho: Density [kg/m³]
        mu: Molecular viscosity [Pa·s]
        dk_dx, dk_dy: k gradients
        domega_dx, domega_dy: ω gradients

    Returns:
        Blending function F1 ∈ [0, 1]
    """
    nu = mu / (rho + 1e-30)

    # Cross-diffusion term
    CD_kw = torch.maximum(
        2 * rho * SIGMA_W2 / omega * (dk_dx * domega_dx + dk_dy * domega_dy),
        torch.tensor(1e-10, dtype=rho.dtype, device=rho.device),
    )

    # Arguments for tanh
    arg1_a = torch.sqrt(k) / (BETA_STAR * omega * d + 1e-30)
    arg1_b = 500 * nu / (d**2 * omega + 1e-30)
    arg1_c = 4 * rho * SIGMA_W2 * k / (CD_kw * d**2 + 1e-30)

    arg1 = torch.minimum(torch.maximum(arg1_a, arg1_b), arg1_c)

    F1 = torch.tanh(arg1**4)

    return F1


def k_omega_sst_eddy_viscosity(
    rho: torch.Tensor,
    k: torch.Tensor,
    omega: torch.Tensor,
    F2: torch.Tensor | None = None,
    S: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute SST eddy viscosity with vorticity limiter.

    μ_t = ρ a₁ k / max(a₁ ω, S F₂)

    If F2 and S are not provided, uses simplified formula:
    μ_t = ρ k / ω

    Args:
        rho: Density [kg/m³]
        k: TKE [m²/s²]
        omega: Specific dissipation [1/s]
        F2: Second blending function (optional)
        S: Strain rate magnitude [1/s] (optional)

    Returns:
        Eddy viscosity [Pa·s]
    """
    a1 = 0.31

    if F2 is not None and S is not None:
        denominator = torch.maximum(a1 * omega, S * F2)
        mu_t = rho * a1 * k / (denominator + 1e-30)
    else:
        # Simplified formula
        mu_t = rho * k / (omega + 1e-30)

    return mu_t


def sst_blending_functions(
    k: torch.Tensor,
    omega: torch.Tensor,
    y: torch.Tensor,
    rho: torch.Tensor,
    mu: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute SST blending functions F1 and F2.

    Simplified version without gradient terms.

    Args:
        k: TKE [m²/s²]
        omega: Specific dissipation [1/s]
        y: Wall distance [m]
        rho: Density [kg/m³]
        mu: Molecular viscosity [Pa·s]

    Returns:
        Tuple of (F1, F2) blending functions
    """
    nu = mu / (rho + 1e-30)
    d = y + 1e-30  # Wall distance

    # F1 arguments (simplified)
    arg1_a = torch.sqrt(k) / (BETA_STAR * omega * d + 1e-30)
    arg1_b = 500 * nu / (d**2 * omega + 1e-30)

    arg1 = torch.minimum(arg1_a, arg1_b)
    F1 = torch.tanh(arg1**4)

    # F2 arguments
    arg2_a = 2 * torch.sqrt(k) / (BETA_STAR * omega * d + 1e-30)
    arg2_b = 500 * nu / (d**2 * omega + 1e-30)

    arg2 = torch.maximum(arg2_a, arg2_b)
    F2 = torch.tanh(arg2**2)

    return F1, F2


def k_omega_sst_source(
    rho: torch.Tensor,
    k: torch.Tensor,
    omega: torch.Tensor,
    P_k: torch.Tensor,
    F1: torch.Tensor,
    dk_dx: torch.Tensor,
    dk_dy: torch.Tensor,
    domega_dx: torch.Tensor,
    domega_dy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute source terms for k-ω SST model.

    Args:
        rho: Density [kg/m³]
        k, omega: Turbulent quantities
        P_k: Production term
        F1: Blending function
        Gradients: For cross-diffusion term

    Returns:
        Tuple of (S_k, S_omega) source terms
    """
    # Blended constants
    alpha = F1 * ALPHA_1 + (1 - F1) * ALPHA_2
    beta = F1 * BETA_1 + (1 - F1) * BETA_2

    # k equation: P_k - β* ρ k ω
    S_k = P_k - BETA_STAR * rho * k * omega

    # ω equation: α (ω/k) P_k - β ρ ω² + cross-diffusion
    S_omega = alpha * omega / (k + 1e-30) * P_k - beta * rho * omega**2

    # Cross-diffusion term (only in k-ε region, F1→0)
    cross_diff = (
        2 * (1 - F1) * rho * SIGMA_W2 / omega * (dk_dx * domega_dx + dk_dy * domega_dy)
    )
    S_omega = S_omega + cross_diff

    return S_k, S_omega


# ============================================================================
# Spalart-Allmaras Model
# ============================================================================


def spalart_allmaras_eddy_viscosity(
    rho: torch.Tensor, nu_tilde: torch.Tensor, nu: torch.Tensor
) -> torch.Tensor:
    """
    Compute eddy viscosity for SA model.

    μ_t = ρ ν̃ f_v1
    f_v1 = χ³ / (χ³ + c_v1³)
    χ = ν̃ / ν

    Args:
        rho: Density [kg/m³]
        nu_tilde: Modified viscosity [m²/s]
        nu: Molecular viscosity [m²/s]

    Returns:
        Eddy viscosity [Pa·s]
    """
    chi = nu_tilde / (nu + 1e-30)
    f_v1 = chi**3 / (chi**3 + CV1**3)

    return rho * nu_tilde * f_v1


def spalart_allmaras_source(
    rho: torch.Tensor,
    nu_tilde: torch.Tensor,
    nu: torch.Tensor,
    d: torch.Tensor,
    S: torch.Tensor,
) -> torch.Tensor:
    """
    Compute source term for SA model.

    Source = c_b1 S̃ ν̃ - c_w1 f_w (ν̃/d)²

    Args:
        rho: Density [kg/m³]
        nu_tilde: Modified viscosity [m²/s]
        nu: Molecular viscosity [m²/s]
        d: Wall distance [m]
        S: Strain rate magnitude [1/s]

    Returns:
        Source term for ν̃ equation
    """
    chi = nu_tilde / (nu + 1e-30)

    # f_v1, f_v2
    f_v1 = chi**3 / (chi**3 + CV1**3)
    f_v2 = 1 - chi / (1 + chi * f_v1)

    # Modified vorticity
    S_tilde = S + nu_tilde / (KAPPA_SA**2 * d**2 + 1e-30) * f_v2
    S_tilde = torch.maximum(S_tilde, torch.tensor(0.3 * S, dtype=S.dtype))

    # Production
    production = CB1 * S_tilde * nu_tilde

    # Destruction
    r = nu_tilde / (S_tilde * KAPPA_SA**2 * d**2 + 1e-30)
    r = torch.minimum(r, torch.tensor(10.0, dtype=r.dtype))
    g = r + CW2 * (r**6 - r)
    f_w = g * ((1 + CW3**6) / (g**6 + CW3**6)) ** (1.0 / 6.0)

    destruction = CW1 * f_w * (nu_tilde / d) ** 2

    return rho * (production - destruction)


# ============================================================================
# Wall Functions
# ============================================================================


def log_law_velocity(
    y_plus: torch.Tensor, kappa: float = 0.41, B: float = 5.2
) -> torch.Tensor:
    """
    Compute u⁺ from log-law.

    u⁺ = (1/κ) ln(y⁺) + B    for y⁺ > 11.6
    u⁺ = y⁺                  for y⁺ ≤ 11.6 (viscous sublayer)

    Args:
        y_plus: Dimensionless wall distance
        kappa: Von Karman constant
        B: Log-law intercept

    Returns:
        u⁺ dimensionless velocity
    """
    u_plus_visc = y_plus
    u_plus_log = (1.0 / kappa) * torch.log(y_plus + 1e-30) + B

    # Smooth blending at y⁺ ≈ 11.6
    blend = torch.sigmoid(2.0 * (y_plus - 11.6))

    return (1 - blend) * u_plus_visc + blend * u_plus_log


def wall_function_tau(
    rho: torch.Tensor,
    u_parallel: torch.Tensor,
    y: torch.Tensor,
    mu: torch.Tensor,
    kappa: float = 0.41,
    B: float = 5.2,
    max_iter: int = 10,
) -> torch.Tensor:
    """
    Compute wall shear stress using wall functions.

    Iteratively solves for u_τ using log-law.

    Args:
        rho: Near-wall density [kg/m³]
        u_parallel: Near-wall velocity magnitude [m/s]
        y: Wall distance [m]
        mu: Dynamic viscosity [Pa·s]
        kappa: Von Karman constant
        B: Log-law intercept
        max_iter: Newton iterations

    Returns:
        Wall shear stress [Pa]
    """
    nu = mu / (rho + 1e-30)

    # Initial guess: u_tau from viscous sublayer
    u_tau = torch.sqrt(nu * u_parallel / (y + 1e-30))

    for _ in range(max_iter):
        y_p = rho * u_tau * y / mu
        u_plus = log_law_velocity(y_p, kappa, B)

        # Newton update: F = u_parallel - u_tau * u_plus = 0
        u_tau_new = u_parallel / (u_plus + 1e-30)

        # Relaxation
        u_tau = 0.5 * u_tau + 0.5 * u_tau_new

    return rho * u_tau**2


# ============================================================================
# Turbulence Initialization
# ============================================================================


def initialize_turbulence(
    model: TurbulenceModel,
    rho: torch.Tensor,
    u: torch.Tensor,
    mu: torch.Tensor,
    turbulence_intensity: float = 0.01,
    viscosity_ratio: float = 10.0,
) -> TurbulentState:
    """
    Initialize turbulent quantities for a given model.

    Args:
        model: Turbulence model type
        rho: Density field [kg/m³]
        u: Velocity field [m/s]
        mu: Molecular viscosity [Pa·s]
        turbulence_intensity: Tu = u'/U
        viscosity_ratio: μ_t / μ

    Returns:
        TurbulentState with initialized fields
    """
    shape = rho.shape
    U_ref = u.abs().max().item() if u.abs().max().item() > 1 else 100.0
    nu_ref = (mu / rho).mean().item()

    if model == TurbulenceModel.LAMINAR:
        return TurbulentState()

    # TKE from turbulence intensity
    k_init = 1.5 * (turbulence_intensity * U_ref) ** 2

    if model in [TurbulenceModel.K_EPSILON, TurbulenceModel.K_EPSILON_REALIZABLE]:
        # ε from k and viscosity ratio
        epsilon_init = C_MU * k_init / (viscosity_ratio * nu_ref)

        return TurbulentState(
            k=torch.full(shape, k_init, dtype=rho.dtype),
            epsilon=torch.full(shape, epsilon_init, dtype=rho.dtype),
        )

    elif model == TurbulenceModel.K_OMEGA_SST:
        # ω from k and μ_t/μ ratio
        omega_init = k_init / (viscosity_ratio * nu_ref)

        return TurbulentState(
            k=torch.full(shape, k_init, dtype=rho.dtype),
            omega=torch.full(shape, omega_init, dtype=rho.dtype),
        )

    elif model == TurbulenceModel.SPALART_ALLMARAS:
        # ν̃ from viscosity ratio
        nu_tilde_init = viscosity_ratio * nu_ref

        return TurbulentState(
            nu_tilde=torch.full(shape, nu_tilde_init, dtype=rho.dtype)
        )

    else:
        raise ValueError(f"Unknown turbulence model: {model}")


# ============================================================================
# Compressibility Corrections
# ============================================================================


def sarkar_correction(
    k: torch.Tensor,
    T: torch.Tensor,
    epsilon: torch.Tensor | None = None,
    gamma: float = 1.4,
    R: float = 287.0,
) -> torch.Tensor:
    """
    Sarkar compressibility correction for high-Mach flows.

    Adds dilatation dissipation: ε_d = α_1 M_t² ε
    where M_t = √(2k) / a is turbulent Mach number.

    Args:
        k: TKE [m²/s²]
        T: Temperature [K]
        epsilon: Dissipation [m²/s³] (optional, returns M_t^2 if None)
        gamma: Specific heat ratio
        R: Gas constant [J/(kg·K)]

    Returns:
        Additional dissipation term (or M_t^2 if epsilon is None)
    """
    alpha_1 = 1.0  # Sarkar constant

    # Sound speed from temperature
    a = torch.sqrt(gamma * R * T)

    M_t = torch.sqrt(2 * k) / (a + 1e-30)

    if epsilon is not None:
        epsilon_d = alpha_1 * M_t**2 * epsilon
        return epsilon_d
    else:
        return M_t**2


def wilcox_compressibility(
    k: torch.Tensor,
    T: torch.Tensor,
    omega: torch.Tensor | None = None,
    gamma: float = 1.4,
    R: float = 287.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Wilcox compressibility correction for k-ω.

    Modifies β* based on turbulent Mach number.

    Args:
        k: TKE [m²/s²]
        T: Temperature [K]
        omega: Specific dissipation [1/s] (optional)
        gamma: Specific heat ratio
        R: Gas constant [J/(kg·K)]

    Returns:
        Tuple of (beta_star_modified, F_Mt)
    """
    # Sound speed from temperature
    a = torch.sqrt(gamma * R * T)

    M_t = torch.sqrt(2 * k) / (a + 1e-30)
    M_t0 = 0.25

    # Heaviside function
    H = torch.where(M_t > M_t0, torch.ones_like(M_t), torch.zeros_like(M_t))

    F_Mt = (M_t**2 - M_t0**2) * H
    beta_star_mod = BETA_STAR * (1 + 1.5 * F_Mt)

    return beta_star_mod, F_Mt


def validate_turbulence():
    """
    Run validation tests for turbulence module.
    """
    print("\n" + "=" * 70)
    print("TURBULENCE MODELING VALIDATION")
    print("=" * 70)

    # Test 1: k-ε eddy viscosity
    print("\n[Test 1] k-ε Eddy Viscosity")
    print("-" * 40)

    rho = torch.tensor([1.2], dtype=torch.float64)
    k = torch.tensor([10.0], dtype=torch.float64)  # m²/s²
    epsilon = torch.tensor([100.0], dtype=torch.float64)  # m²/s³

    mu_t = k_epsilon_eddy_viscosity(rho, k, epsilon)

    # μ_t = ρ C_μ k² / ε = 1.2 * 0.09 * 100 / 100 = 0.108
    expected = 1.2 * 0.09 * 100 / 100
    error = abs(mu_t.item() - expected) / expected

    print(f"μ_t = {mu_t.item():.6f} Pa·s")
    print(f"Expected: {expected:.6f} Pa·s")
    print(f"Error: {error*100:.4f}%")

    if error < 0.001:
        print("✓ PASS")
    else:
        print("✗ FAIL")

    # Test 2: Wall function log-law
    print("\n[Test 2] Log-Law Wall Function")
    print("-" * 40)

    y_p = torch.tensor([1.0, 11.6, 100.0, 1000.0], dtype=torch.float64)
    u_p = log_law_velocity(y_p)

    print("y⁺ → u⁺:")
    for yp, up in zip(y_p.tolist(), u_p.tolist()):
        print(f"  {yp:8.1f} → {up:.3f}")

    # At y⁺ = 1, should be ≈ 1 (viscous sublayer)
    # At y⁺ = 1000, log-law: u⁺ ≈ (1/0.41)*ln(1000) + 5.2 ≈ 22.0
    if abs(u_p[0].item() - 1.0) < 0.5:
        print("✓ PASS: Viscous sublayer correct")
    else:
        print("✗ FAIL: Viscous sublayer incorrect")

    # Test 3: SA eddy viscosity
    print("\n[Test 3] Spalart-Allmaras Eddy Viscosity")
    print("-" * 40)

    rho = torch.tensor([1.0], dtype=torch.float64)
    nu = torch.tensor([1.5e-5], dtype=torch.float64)
    nu_tilde = torch.tensor([1.5e-4], dtype=torch.float64)  # 10x molecular

    mu_t = spalart_allmaras_eddy_viscosity(rho, nu_tilde, nu)

    chi = nu_tilde / nu
    f_v1 = chi**3 / (chi**3 + CV1**3)
    expected = rho * nu_tilde * f_v1

    print(f"χ = ν̃/ν = {chi.item():.1f}")
    print(f"f_v1 = {f_v1.item():.4f}")
    print(f"μ_t = {mu_t.item():.2e} Pa·s")

    if torch.allclose(mu_t, expected, rtol=1e-10):
        print("✓ PASS")
    else:
        print("✗ FAIL")

    # Test 4: Turbulence initialization
    print("\n[Test 4] Turbulence Initialization")
    print("-" * 40)

    for model in [
        TurbulenceModel.K_EPSILON,
        TurbulenceModel.K_OMEGA_SST,
        TurbulenceModel.SPALART_ALLMARAS,
    ]:
        state = initialize_turbulence(model, (10, 10), U_inf=100.0)
        print(f"{model.value}: ", end="")

        if model == TurbulenceModel.K_EPSILON:
            print(f"k={state.k[0,0].item():.2e}, ε={state.epsilon[0,0].item():.2e}")
        elif model == TurbulenceModel.K_OMEGA_SST:
            print(f"k={state.k[0,0].item():.2e}, ω={state.omega[0,0].item():.2e}")
        else:
            print(f"ν̃={state.nu_tilde[0,0].item():.2e}")

    print("✓ PASS: All models initialize")

    print("\n" + "=" * 70)
    print("TURBULENCE VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    validate_turbulence()
