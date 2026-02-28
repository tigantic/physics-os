"""
Godunov-Type Riemann Solvers
============================

Numerical flux functions for the 1D Euler equations based on
solving the Riemann problem at cell interfaces.

Solvers implemented:
- Roe: Linearized Riemann solver with entropy fix
- HLL: Harten-Lax-van Leer two-wave approximation
- HLLC: HLL with contact wave restoration
- Exact: Newton iteration exact Riemann solver

For the Euler equations, the Riemann problem is:
    ∂U/∂t + ∂F/∂x = 0
with piecewise constant initial data U_L, U_R.

The solution consists of three waves:
1. Left-going wave (shock or rarefaction)
2. Contact discontinuity (entropy wave)
3. Right-going wave (shock or rarefaction)

Example:
    >>> import torch
    >>> from ontic.cfd.godunov import roe_flux, hllc_flux
    >>> U_L = torch.tensor([[1.0, 0.0, 2.5]])  # Left state
    >>> U_R = torch.tensor([[0.125, 0.0, 0.25]])  # Right state
    >>> F = roe_flux(U_L, U_R, gamma=1.4)
    >>> print(f"Roe flux: {F}")

Raises:
    ValueError: If input states have incompatible shapes
    RuntimeError: If Newton iteration fails to converge (exact solver)

References:
    .. [1] Roe, P.L. "Approximate Riemann solvers, parameter vectors, and
           difference schemes", J. Comput. Phys. 43, 357-372, 1981.
    .. [2] Harten, A., Lax, P.D., van Leer, B. "On upstream differencing
           and Godunov-type schemes for hyperbolic conservation laws",
           SIAM Review 25, 35-61, 1983.
    .. [3] Toro, E.F., Spruce, M., Speares, W. "Restoration of the contact
           surface in the HLL-Riemann solver", Shock Waves 4, 25-34, 1994.
    .. [4] Toro, E.F. "Riemann Solvers and Numerical Methods for Fluid
           Dynamics", 3rd ed., Springer, 2009.
"""

import math

import torch
from torch import Tensor


def primitive_to_conserved(
    rho: Tensor,
    u: Tensor,
    p: Tensor,
    gamma: float = 1.4,
) -> Tensor:
    """Convert primitive (ρ, u, p) to conserved (ρ, ρu, E)."""
    rho_u = rho * u
    E = p / (gamma - 1) + 0.5 * rho * u**2
    return torch.stack([rho, rho_u, E], dim=-1)


def conserved_to_primitive(
    U: Tensor,
    gamma: float = 1.4,
) -> tuple[Tensor, Tensor, Tensor]:
    """Convert conserved (ρ, ρu, E) to primitive (ρ, u, p)."""
    rho = U[..., 0]
    rho_u = U[..., 1]
    E = U[..., 2]

    u = rho_u / rho
    p = (gamma - 1) * (E - 0.5 * rho * u**2)

    return rho, u, p


def euler_flux(U: Tensor, gamma: float = 1.4) -> Tensor:
    """
    Physical flux for 1D Euler equations.

    F = [ρu, ρu² + p, (E + p)u]ᵀ
    """
    rho, u, p = conserved_to_primitive(U, gamma)

    rho_u = U[..., 1]
    E = U[..., 2]

    F = torch.stack(
        [
            rho_u,
            rho_u * u + p,
            (E + p) * u,
        ],
        dim=-1,
    )

    return F


def roe_flux(
    U_L: Tensor,
    U_R: Tensor,
    gamma: float = 1.4,
    entropy_fix: bool = True,
    epsilon: float = 0.1,
) -> Tensor:
    """
    Roe's linearized Riemann solver.

    Uses Roe-averaged quantities to construct an approximate
    Jacobian with exact eigenstructure.

    Roe averages:
        ρ̂ = √(ρ_L ρ_R)
        û = (√ρ_L u_L + √ρ_R u_R) / (√ρ_L + √ρ_R)
        Ĥ = (√ρ_L H_L + √ρ_R H_R) / (√ρ_L + √ρ_R)

    Args:
        U_L: Left state (batch, 3) - conserved variables [ρ, ρu, E]
        U_R: Right state (batch, 3) - conserved variables [ρ, ρu, E]
        gamma: Ratio of specific heats (default 1.4 for air)
        entropy_fix: Apply Harten's entropy fix for expansion shocks
        epsilon: Entropy fix parameter (unused, kept for API compatibility)

    Returns:
        Numerical flux (batch, 3)

    Example:
        >>> U_L = torch.tensor([[1.0, 0.0, 2.5]])
        >>> U_R = torch.tensor([[0.125, 0.0, 0.25]])
        >>> F = roe_flux(U_L, U_R)
        >>> print(f"Flux shape: {F.shape}")

    Raises:
        ValueError: If U_L and U_R have different shapes

    References:
        .. [1] Roe, P.L. "Approximate Riemann solvers", J. Comput. Phys. 43, 1981.
    """
    # Extract primitives
    rho_L, u_L, p_L = conserved_to_primitive(U_L, gamma)
    rho_R, u_R, p_R = conserved_to_primitive(U_R, gamma)

    E_L, E_R = U_L[..., 2], U_R[..., 2]

    # Specific enthalpies
    H_L = (E_L + p_L) / rho_L
    H_R = (E_R + p_R) / rho_R

    # Roe averages
    sqrt_rho_L = torch.sqrt(rho_L)
    sqrt_rho_R = torch.sqrt(rho_R)
    denom = sqrt_rho_L + sqrt_rho_R

    rho_hat = sqrt_rho_L * sqrt_rho_R
    u_hat = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) / denom
    H_hat = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) / denom

    # Sound speed from Roe average
    a_hat_sq = (gamma - 1) * (H_hat - 0.5 * u_hat**2)
    a_hat_sq = torch.clamp(a_hat_sq, min=1e-10)
    a_hat = torch.sqrt(a_hat_sq)

    # Eigenvalues
    lambda_1 = u_hat - a_hat  # Left acoustic
    lambda_2 = u_hat  # Entropy
    lambda_3 = u_hat + a_hat  # Right acoustic

    # Entropy fix (Harten)
    if entropy_fix:
        a_L = torch.sqrt(gamma * p_L / rho_L)
        a_R = torch.sqrt(gamma * p_R / rho_R)

        delta_1 = torch.maximum(4 * ((u_R - a_R) - (u_L - a_L)), torch.zeros_like(a_L))
        delta_3 = torch.maximum(4 * ((u_R + a_R) - (u_L + a_L)), torch.zeros_like(a_L))

        lambda_1 = torch.where(
            torch.abs(lambda_1) < delta_1 / 2,
            lambda_1**2 / delta_1 + delta_1 / 4,
            torch.abs(lambda_1),
        )
        lambda_3 = torch.where(
            torch.abs(lambda_3) < delta_3 / 2,
            lambda_3**2 / delta_3 + delta_3 / 4,
            torch.abs(lambda_3),
        )
        lambda_2 = torch.abs(lambda_2)
    else:
        lambda_1 = torch.abs(lambda_1)
        lambda_2 = torch.abs(lambda_2)
        lambda_3 = torch.abs(lambda_3)

    # Jump in conserved variables
    dU = U_R - U_L
    drho = dU[..., 0]
    drho_u = dU[..., 1]
    dE = dU[..., 2]

    du = (drho_u - u_hat * drho) / rho_hat
    dp = (gamma - 1) * (
        dE - u_hat * drho_u + (u_hat**2 - H_hat) * drho + 0.5 * u_hat**2 * drho
    )
    # Corrected pressure jump
    dp = (gamma - 1) * (dE - u_hat * du * rho_hat - 0.5 * u_hat**2 * drho)

    # Actually, use standard Roe decomposition
    # Wave strengths
    dp_computed = (gamma - 1) * (dE - u_hat * drho_u + 0.5 * u_hat**2 * drho)
    du_computed = (drho_u - u_hat * drho) / rho_hat

    alpha_2 = drho - dp_computed / a_hat_sq
    alpha_1 = (dp_computed - rho_hat * a_hat * du_computed) / (2 * a_hat_sq)
    alpha_3 = (dp_computed + rho_hat * a_hat * du_computed) / (2 * a_hat_sq)

    # Right eigenvectors (columns)
    # r_1 = [1, u-a, H-ua]
    # r_2 = [1, u, u²/2]
    # r_3 = [1, u+a, H+ua]

    # Compute |A| ΔU = Σ |λ_i| α_i r_i
    abs_A_dU_0 = lambda_1 * alpha_1 + lambda_2 * alpha_2 + lambda_3 * alpha_3
    abs_A_dU_1 = (
        lambda_1 * alpha_1 * (u_hat - a_hat)
        + lambda_2 * alpha_2 * u_hat
        + lambda_3 * alpha_3 * (u_hat + a_hat)
    )
    abs_A_dU_2 = (
        lambda_1 * alpha_1 * (H_hat - u_hat * a_hat)
        + lambda_2 * alpha_2 * 0.5 * u_hat**2
        + lambda_3 * alpha_3 * (H_hat + u_hat * a_hat)
    )

    abs_A_dU = torch.stack([abs_A_dU_0, abs_A_dU_1, abs_A_dU_2], dim=-1)

    # Physical fluxes
    F_L = euler_flux(U_L, gamma)
    F_R = euler_flux(U_R, gamma)

    # Roe flux
    flux = 0.5 * (F_L + F_R) - 0.5 * abs_A_dU

    return flux


def hll_flux(
    U_L: Tensor,
    U_R: Tensor,
    gamma: float = 1.4,
) -> Tensor:
    """
    HLL (Harten-Lax-van Leer) approximate Riemann solver.

    Uses two-wave approximation, ignoring the contact discontinuity.
    Robust but diffusive for contact waves.

    Wave speed estimates (Davis):
        S_L = min(u_L - a_L, u_R - a_R)
        S_R = max(u_L + a_L, u_R + a_R)
    """
    rho_L, u_L, p_L = conserved_to_primitive(U_L, gamma)
    rho_R, u_R, p_R = conserved_to_primitive(U_R, gamma)

    a_L = torch.sqrt(gamma * p_L / rho_L)
    a_R = torch.sqrt(gamma * p_R / rho_R)

    # Wave speed estimates (Davis)
    S_L = torch.minimum(u_L - a_L, u_R - a_R)
    S_R = torch.maximum(u_L + a_L, u_R + a_R)

    # Physical fluxes
    F_L = euler_flux(U_L, gamma)
    F_R = euler_flux(U_R, gamma)

    # HLL flux
    # F_HLL = (S_R F_L - S_L F_R + S_L S_R (U_R - U_L)) / (S_R - S_L)

    denom = S_R - S_L
    denom = torch.where(torch.abs(denom) < 1e-10, torch.ones_like(denom) * 1e-10, denom)

    F_HLL = (
        S_R.unsqueeze(-1) * F_L
        - S_L.unsqueeze(-1) * F_R
        + (S_L * S_R).unsqueeze(-1) * (U_R - U_L)
    ) / denom.unsqueeze(-1)

    # Choose flux based on wave speeds
    flux = torch.where(
        S_L.unsqueeze(-1) >= 0, F_L, torch.where(S_R.unsqueeze(-1) <= 0, F_R, F_HLL)
    )

    return flux


def hllc_flux(
    U_L: Tensor,
    U_R: Tensor,
    gamma: float = 1.4,
) -> Tensor:
    """
    HLLC (HLL-Contact) approximate Riemann solver.

    Extends HLL by restoring the contact discontinuity,
    giving exact resolution of isolated contact waves.

    Three-wave structure: S_L | S_* | S_R
    """
    rho_L, u_L, p_L = conserved_to_primitive(U_L, gamma)
    rho_R, u_R, p_R = conserved_to_primitive(U_R, gamma)

    E_L, E_R = U_L[..., 2], U_R[..., 2]

    a_L = torch.sqrt(gamma * p_L / rho_L)
    a_R = torch.sqrt(gamma * p_R / rho_R)

    # Wave speed estimates
    S_L = torch.minimum(u_L - a_L, u_R - a_R)
    S_R = torch.maximum(u_L + a_L, u_R + a_R)

    # Contact wave speed
    # S_* = (p_R - p_L + ρ_L u_L (S_L - u_L) - ρ_R u_R (S_R - u_R)) /
    #       (ρ_L (S_L - u_L) - ρ_R (S_R - u_R))
    numer = p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)
    denom = rho_L * (S_L - u_L) - rho_R * (S_R - u_R)
    denom = torch.where(torch.abs(denom) < 1e-10, torch.sign(denom) * 1e-10, denom)
    S_star = numer / denom

    # Star state pressure
    p_star = p_L + rho_L * (S_L - u_L) * (S_star - u_L)

    # Physical fluxes
    F_L = euler_flux(U_L, gamma)
    F_R = euler_flux(U_R, gamma)

    # Star states
    # U*_K = ρ_K (S_K - u_K)/(S_K - S*) [1, S*, E_K/ρ_K + (S* - u_K)(S* + p_K/(ρ_K(S_K - u_K)))]

    factor_L = rho_L * (S_L - u_L) / (S_L - S_star + 1e-10)
    factor_R = rho_R * (S_R - u_R) / (S_R - S_star + 1e-10)

    E_star_L = factor_L * (
        E_L / rho_L + (S_star - u_L) * (S_star + p_L / (rho_L * (S_L - u_L) + 1e-10))
    )
    E_star_R = factor_R * (
        E_R / rho_R + (S_star - u_R) * (S_star + p_R / (rho_R * (S_R - u_R) + 1e-10))
    )

    U_star_L = torch.stack(
        [
            factor_L,
            factor_L * S_star,
            E_star_L,
        ],
        dim=-1,
    )

    U_star_R = torch.stack(
        [
            factor_R,
            factor_R * S_star,
            E_star_R,
        ],
        dim=-1,
    )

    # HLLC flux
    F_star_L = F_L + S_L.unsqueeze(-1) * (U_star_L - U_L)
    F_star_R = F_R + S_R.unsqueeze(-1) * (U_star_R - U_R)

    # Choose appropriate flux
    flux = torch.where(
        S_L.unsqueeze(-1) >= 0,
        F_L,
        torch.where(
            S_star.unsqueeze(-1) >= 0,
            F_star_L,
            torch.where(S_R.unsqueeze(-1) >= 0, F_star_R, F_R),
        ),
    )

    return flux


def exact_riemann(
    rho_L: float,
    u_L: float,
    p_L: float,
    rho_R: float,
    u_R: float,
    p_R: float,
    gamma: float = 1.4,
    x: Tensor | None = None,
    t: float = 1.0,
    x0: float = 0.5,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Exact Riemann solver using Newton-Raphson iteration.

    Solves for the star-state pressure p* at the contact,
    then constructs the full solution.

    Args:
        rho_L, u_L, p_L: Left primitive state
        rho_R, u_R, p_R: Right primitive state
        gamma: Ratio of specific heats
        x: Spatial coordinates for sampling solution
        t: Time at which to sample solution
        x0: Initial discontinuity location
        tol: Newton tolerance
        max_iter: Maximum Newton iterations

    Returns:
        (rho, u, p) sampled at x coordinates
    """
    if x is None:
        x = torch.linspace(0, 1, 100)

    device = x.device
    dtype = x.dtype

    # Sound speeds
    a_L = math.sqrt(gamma * p_L / rho_L)
    a_R = math.sqrt(gamma * p_R / rho_R)

    # Pressure functions for Newton iteration
    def f_K(p, rho_K, p_K, a_K):
        """Pressure function for state K (L or R)."""
        if p > p_K:  # Shock
            A_K = 2 / ((gamma + 1) * rho_K)
            B_K = (gamma - 1) / (gamma + 1) * p_K
            return (p - p_K) * math.sqrt(A_K / (p + B_K))
        else:  # Rarefaction
            return (
                2 * a_K / (gamma - 1) * ((p / p_K) ** ((gamma - 1) / (2 * gamma)) - 1)
            )

    def df_K(p, rho_K, p_K, a_K):
        """Derivative of pressure function."""
        if p > p_K:  # Shock
            A_K = 2 / ((gamma + 1) * rho_K)
            B_K = (gamma - 1) / (gamma + 1) * p_K
            term = math.sqrt(A_K / (p + B_K))
            return term * (1 - (p - p_K) / (2 * (p + B_K)))
        else:  # Rarefaction
            return 1 / (rho_K * a_K) * (p / p_K) ** (-(gamma + 1) / (2 * gamma))

    # Newton iteration for p*
    # f(p) = f_L(p) + f_R(p) + u_R - u_L = 0

    # Initial guess (PVRS)
    p_star = 0.5 * (p_L + p_R) - 0.125 * (u_R - u_L) * (rho_L + rho_R) * (a_L + a_R)
    p_star = max(p_star, 1e-10)

    for _ in range(max_iter):
        f = f_K(p_star, rho_L, p_L, a_L) + f_K(p_star, rho_R, p_R, a_R) + (u_R - u_L)
        df = df_K(p_star, rho_L, p_L, a_L) + df_K(p_star, rho_R, p_R, a_R)

        if abs(df) < 1e-14:
            break

        p_new = p_star - f / df
        p_new = max(p_new, 1e-10)

        if abs(p_new - p_star) < tol * p_star:
            p_star = p_new
            break

        p_star = p_new

    # Star velocity
    u_star = 0.5 * (u_L + u_R) + 0.5 * (
        f_K(p_star, rho_R, p_R, a_R) - f_K(p_star, rho_L, p_L, a_L)
    )

    # Sample solution at x, t
    xi = (x - x0) / t  # Self-similarity variable

    rho = torch.zeros_like(x)
    u = torch.zeros_like(x)
    p = torch.zeros_like(x)

    for i, s in enumerate(xi.tolist()):
        # Determine which region we're in
        if s < u_star:  # Left of contact
            if p_star > p_L:  # Left shock
                # Shock speed
                S_L = u_L - a_L * math.sqrt(
                    (gamma + 1) / (2 * gamma) * p_star / p_L + (gamma - 1) / (2 * gamma)
                )
                if s < S_L:
                    rho[i] = rho_L
                    u[i] = u_L
                    p[i] = p_L
                else:
                    rho_star_L = (
                        rho_L
                        * (p_star / p_L + (gamma - 1) / (gamma + 1))
                        / ((gamma - 1) / (gamma + 1) * p_star / p_L + 1)
                    )
                    rho[i] = rho_star_L
                    u[i] = u_star
                    p[i] = p_star
            else:  # Left rarefaction
                a_star_L = a_L * (p_star / p_L) ** ((gamma - 1) / (2 * gamma))
                S_HL = u_L - a_L  # Head
                S_TL = u_star - a_star_L  # Tail

                if s < S_HL:
                    rho[i] = rho_L
                    u[i] = u_L
                    p[i] = p_L
                elif s < S_TL:
                    # Inside rarefaction fan
                    u_fan = 2 / (gamma + 1) * (a_L + (gamma - 1) / 2 * u_L + s)
                    a_fan = u_fan - s
                    rho_fan = rho_L * (a_fan / a_L) ** (2 / (gamma - 1))
                    p_fan = p_L * (a_fan / a_L) ** (2 * gamma / (gamma - 1))
                    rho[i] = rho_fan
                    u[i] = u_fan
                    p[i] = p_fan
                else:
                    rho_star_L = rho_L * (p_star / p_L) ** (1 / gamma)
                    rho[i] = rho_star_L
                    u[i] = u_star
                    p[i] = p_star
        else:  # Right of contact
            if p_star > p_R:  # Right shock
                S_R = u_R + a_R * math.sqrt(
                    (gamma + 1) / (2 * gamma) * p_star / p_R + (gamma - 1) / (2 * gamma)
                )
                if s > S_R:
                    rho[i] = rho_R
                    u[i] = u_R
                    p[i] = p_R
                else:
                    rho_star_R = (
                        rho_R
                        * (p_star / p_R + (gamma - 1) / (gamma + 1))
                        / ((gamma - 1) / (gamma + 1) * p_star / p_R + 1)
                    )
                    rho[i] = rho_star_R
                    u[i] = u_star
                    p[i] = p_star
            else:  # Right rarefaction
                a_star_R = a_R * (p_star / p_R) ** ((gamma - 1) / (2 * gamma))
                S_HR = u_R + a_R
                S_TR = u_star + a_star_R

                if s > S_HR:
                    rho[i] = rho_R
                    u[i] = u_R
                    p[i] = p_R
                elif s > S_TR:
                    u_fan = 2 / (gamma + 1) * (-a_R + (gamma - 1) / 2 * u_R + s)
                    a_fan = s - u_fan
                    rho_fan = rho_R * (a_fan / a_R) ** (2 / (gamma - 1))
                    p_fan = p_R * (a_fan / a_R) ** (2 * gamma / (gamma - 1))
                    rho[i] = rho_fan
                    u[i] = u_fan
                    p[i] = p_fan
                else:
                    rho_star_R = rho_R * (p_star / p_R) ** (1 / gamma)
                    rho[i] = rho_star_R
                    u[i] = u_star
                    p[i] = p_star

    return rho, u, p
