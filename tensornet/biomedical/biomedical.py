"""
Biomedical Engineering: Bidomain cardiac electrophysiology,
pharmacokinetic (PK) compartment models, tissue hyperelasticity.

Upgrades domain XX.6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

FARADAY: float = 96485.0       # C/mol
R_GAS: float = 8.314           # J/(mol·K)


# ---------------------------------------------------------------------------
#  Bidomain Cardiac Electrophysiology
# ---------------------------------------------------------------------------

class FitzHughNagumo:
    r"""
    FitzHugh-Nagumo cardiac cell model (simplified AP).

    $$\frac{dv}{dt} = v - v^3/3 - w + I_{ext}$$
    $$\frac{dw}{dt} = \epsilon(v + a - bw)$$

    where v = membrane voltage analogue, w = recovery variable.
    """

    def __init__(self, a: float = 0.7, b: float = 0.8,
                 epsilon: float = 0.08) -> None:
        self.a = a
        self.b = b
        self.epsilon = epsilon

    def rhs(self, v: float, w: float, I_ext: float = 0.0) -> Tuple[float, float]:
        dv = v - v**3 / 3.0 - w + I_ext
        dw = self.epsilon * (v + self.a - self.b * w)
        return dv, dw


class AlievPanfilov:
    r"""
    Aliev-Panfilov (1996) cardiac model: more realistic AP morphology.

    $$\frac{\partial V}{\partial t} = \nabla\cdot(D\nabla V) - kV(V-a)(V-1) - Vr$$
    $$\frac{\partial r}{\partial t} = (-\epsilon_0 - \frac{\mu_1 r}{\mu_2+V})(r + kV(V-a-1))$$
    """

    def __init__(self, k: float = 8.0, a: float = 0.15,
                 eps0: float = 0.002, mu1: float = 0.2,
                 mu2: float = 0.3) -> None:
        self.k = k
        self.a = a
        self.eps0 = eps0
        self.mu1 = mu1
        self.mu2 = mu2

    def reaction(self, V: NDArray, r: NDArray) -> Tuple[NDArray, NDArray]:
        """Reaction terms (no diffusion)."""
        dV = -self.k * V * (V - self.a) * (V - 1.0) - V * r
        eps = self.eps0 + self.mu1 * r / (self.mu2 + V + 1e-10)
        dr = -eps * (r + self.k * V * (V - self.a - 1.0))
        return dV, dr


class BidomainSolver:
    r"""
    Bidomain model for cardiac tissue electrophysiology.

    $$\nabla\cdot(\sigma_i\nabla V_m) + \nabla\cdot(\sigma_i\nabla\phi_e) = \beta(C_m\frac{\partial V_m}{\partial t} + I_{ion})$$
    $$\nabla\cdot((\sigma_i+\sigma_e)\nabla\phi_e) = -\nabla\cdot(\sigma_i\nabla V_m)$$

    where $V_m = \phi_i - \phi_e$, σ_i/σ_e = intra/extracellular conductivity.

    2D operator-splitting: diffusion (implicit) + reaction (explicit).
    """

    def __init__(self, nx: int, ny: int, dx: float,
                 sigma_i: float = 0.17, sigma_e: float = 0.62,
                 Cm: float = 1.0, beta: float = 140.0) -> None:
        """
        Parameters
        ----------
        nx, ny : Grid dimensions.
        dx : Grid spacing (mm).
        sigma_i : Intracellular conductivity (mS/mm).
        sigma_e : Extracellular conductivity (mS/mm).
        Cm : Membrane capacitance (µF/mm²).
        beta : Surface-to-volume ratio (mm⁻¹).
        """
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.sigma_i = sigma_i
        self.sigma_e = sigma_e
        self.Cm = Cm
        self.beta = beta

        self.Vm = np.zeros((nx, ny))       # Transmembrane potential
        self.phi_e = np.zeros((nx, ny))     # Extracellular potential
        self.r = np.zeros((nx, ny))         # Recovery variable

        # Use Aliev-Panfilov ionic model
        self.cell_model = AlievPanfilov()

    def _laplacian(self, f: NDArray) -> NDArray:
        """5-point Laplacian."""
        dx2 = self.dx**2
        return (np.roll(f, 1, 0) + np.roll(f, -1, 0)
                + np.roll(f, 1, 1) + np.roll(f, -1, 1)
                - 4.0 * f) / dx2

    def step(self, dt: float) -> None:
        """One time step via operator splitting."""
        # 1. Reaction step (explicit)
        dV_ion, dr = self.cell_model.reaction(self.Vm, self.r)
        self.Vm += dt * dV_ion / self.Cm
        self.r += dt * dr

        # 2. Diffusion step (explicit, small dt required for stability)
        D_eff = self.sigma_i * self.sigma_e / (self.sigma_i + self.sigma_e)
        lap_Vm = self._laplacian(self.Vm)
        self.Vm += dt * D_eff / (self.beta * self.Cm) * lap_Vm

        # 3. Solve for phi_e (Poisson: ∇·(σ_i+σ_e)∇φ_e = -∇·σ_i∇Vm)
        # Simplified: φ_e ≈ -σ_i/(σ_i+σ_e) Vm (equal anisotropy ratio)
        self.phi_e = -self.sigma_i / (self.sigma_i + self.sigma_e) * self.Vm

    def stimulate(self, region: Tuple[slice, slice],
                    amplitude: float = 1.0) -> None:
        """Apply stimulus current to a region."""
        self.Vm[region] = amplitude

    def run(self, dt: float, n_steps: int) -> List[NDArray]:
        """Run simulation, return snapshots."""
        snapshots = [self.Vm.copy()]
        save_interval = max(n_steps // 50, 1)
        for step in range(n_steps):
            self.step(dt)
            if (step + 1) % save_interval == 0:
                snapshots.append(self.Vm.copy())
        return snapshots


# ---------------------------------------------------------------------------
#  Pharmacokinetic Compartment Models
# ---------------------------------------------------------------------------

class CompartmentPK:
    r"""
    Multi-compartment pharmacokinetic model.

    $$\frac{dC_1}{dt} = -k_{10}C_1 - k_{12}C_1 + k_{21}C_2 + \frac{D(t)}{V_1}$$
    $$\frac{dC_2}{dt} = k_{12}C_1 - k_{21}C_2$$

    Supports:
    - 1-compartment (mono-exponential)
    - 2-compartment (bi-exponential)
    - 3-compartment
    - IV bolus, IV infusion, oral absorption
    """

    def __init__(self, n_compartments: int = 2) -> None:
        self.n = n_compartments
        # Rate constants matrix K[i,j] = transfer rate from j to i
        self.K = np.zeros((n_compartments, n_compartments))
        # Volume of each compartment (L)
        self.V = np.ones(n_compartments)
        # Elimination rate from compartment 0
        self.k_el = 0.1  # h⁻¹
        # Absorption rate (for oral)
        self.k_a = 0.0  # h⁻¹
        self.bioavailability = 1.0

    def set_two_compartment(self, k10: float, k12: float, k21: float,
                              V1: float, V2: float) -> None:
        """Standard 2-compartment parameters."""
        self.k_el = k10
        self.K[0, 1] = k21
        self.K[1, 0] = k12
        self.V[0] = V1
        self.V[1] = V2

    def rhs(self, C: NDArray, dose_rate: float = 0.0) -> NDArray:
        """dC/dt for all compartments."""
        dCdt = np.zeros(self.n)

        for i in range(self.n):
            # Inflow from other compartments
            for j in range(self.n):
                if i != j:
                    dCdt[i] += self.K[i, j] * C[j]
                    dCdt[i] -= self.K[j, i] * C[i]

        # Elimination from central compartment (0)
        dCdt[0] -= self.k_el * C[0]

        # Dose input to central compartment
        dCdt[0] += dose_rate / self.V[0]

        return dCdt

    def simulate_iv_bolus(self, dose: float, dt: float = 0.01,
                            t_max: float = 24.0) -> Tuple[NDArray, NDArray]:
        """IV bolus: instantaneous dose into compartment 1.

        Returns (time, concentrations shape (n_steps, n_compartments)).
        """
        n_steps = int(t_max / dt)
        C = np.zeros(self.n)
        C[0] = dose / self.V[0]

        t_arr = np.arange(n_steps + 1) * dt
        trajectory = np.zeros((n_steps + 1, self.n))
        trajectory[0] = C

        for step in range(n_steps):
            # RK4
            k1 = self.rhs(C)
            k2 = self.rhs(C + 0.5 * dt * k1)
            k3 = self.rhs(C + 0.5 * dt * k2)
            k4 = self.rhs(C + dt * k3)
            C = C + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            C = np.maximum(C, 0)
            trajectory[step + 1] = C

        return t_arr, trajectory

    def half_life(self) -> float:
        """Terminal half-life t₁/₂ = ln(2)/k_el."""
        return math.log(2) / self.k_el if self.k_el > 0 else float('inf')

    def auc(self, dose: float) -> float:
        """Area under curve: AUC = Dose / (CL) = Dose / (k_el V₁)."""
        CL = self.k_el * self.V[0]
        return dose / CL if CL > 0 else float('inf')


# ---------------------------------------------------------------------------
#  Tissue Hyperelasticity
# ---------------------------------------------------------------------------

class OgdenHyperelastic:
    r"""
    Ogden hyperelastic model for soft biological tissue.

    Strain energy:
    $$W = \sum_{p=1}^{N}\frac{2\mu_p}{\alpha_p^2}
          (\lambda_1^{\alpha_p} + \lambda_2^{\alpha_p} + \lambda_3^{\alpha_p} - 3)
          + \frac{1}{D}(J-1)^2$$

    where λ_i = principal stretches, J = det(F).

    Special cases:
    - N=1, α=2: neo-Hookean
    - N=2: Mooney-Rivlin
    """

    def __init__(self, mu: List[float], alpha: List[float],
                 D: float = 1e-6) -> None:
        """
        Parameters
        ----------
        mu : Shear moduli for each term (Pa).
        alpha : Exponents for each term.
        D : Compressibility parameter (1/Pa).
        """
        if len(mu) != len(alpha):
            raise ValueError("mu and alpha must have same length")
        self.mu = mu
        self.alpha = alpha
        self.D = D
        self.N = len(mu)

    def strain_energy(self, lambda1: float, lambda2: float,
                        lambda3: float) -> float:
        """W(λ₁, λ₂, λ₃)."""
        J = lambda1 * lambda2 * lambda3
        W = 0.0
        for p in range(self.N):
            a = self.alpha[p]
            m = self.mu[p]
            W += 2.0 * m / a**2 * (lambda1**a + lambda2**a + lambda3**a - 3.0)
        # Volumetric
        W += (J - 1.0)**2 / self.D
        return W

    def cauchy_stress_uniaxial(self, stretch: float) -> float:
        """Cauchy stress for uniaxial tension (σ₁₁).

        With incompressibility: λ₂ = λ₃ = 1/√λ₁.
        """
        lam = stretch
        lam_t = 1.0 / math.sqrt(lam)  # transverse stretch

        sigma = 0.0
        for p in range(self.N):
            a = self.alpha[p]
            m = self.mu[p]
            sigma += m / a * (a * lam**(a - 1) - a * lam_t**(a - 1) * (-0.5 / lam**1.5))

        # Simplified: σ = Σ μ_p (λ^(αp-1) - λ^(-αp/2 - 1))
        sigma = 0.0
        for p in range(self.N):
            a = self.alpha[p]
            sigma += self.mu[p] * (lam**(a - 1) - lam**(-a / 2.0 - 1.0))
        return sigma

    def tangent_modulus_uniaxial(self, stretch: float,
                                    d_stretch: float = 1e-6) -> float:
        """dσ/dλ numerically."""
        s1 = self.cauchy_stress_uniaxial(stretch + d_stretch)
        s0 = self.cauchy_stress_uniaxial(stretch - d_stretch)
        return (s1 - s0) / (2 * d_stretch)


class HolzapfelArtery:
    r"""
    Holzapfel-Gasser-Ogden (HGO) model for arterial tissue.

    $$W = \frac{\mu}{2}(I_1 - 3) + \sum_{i=1}^{2}\frac{k_1}{2k_2}
      \left[\exp(k_2\langle E_i \rangle^2) - 1\right]$$

    where $E_i = \kappa(I_1-3) + (1-3\kappa)(I_{4i}-1)$,
    $I_{4i} = \mathbf{a}_i\cdot\mathbf{C}\mathbf{a}_i$ (fibre invariants).
    """

    def __init__(self, mu: float = 6.0e3, k1: float = 2.6e3,
                 k2: float = 8.21, kappa: float = 0.226,
                 theta_fibre: float = math.radians(39.76)) -> None:
        """
        Parameters
        ----------
        mu : Ground substance shear modulus (Pa).
        k1 : Fibre stiffness parameter (Pa).
        k2 : Fibre exponential parameter (dimensionless).
        kappa : Dispersion parameter (0 = aligned, 1/3 = isotropic).
        theta_fibre : Fibre orientation angle from circumferential (rad).
        """
        self.mu = mu
        self.k1 = k1
        self.k2 = k2
        self.kappa = kappa
        self.theta = theta_fibre

    def _fibre_invariant(self, C11: float, C22: float, C33: float = 1.0,
                           family: int = 1) -> float:
        """I₄ for fibre family."""
        angle = self.theta if family == 1 else -self.theta
        a1 = math.cos(angle)
        a2 = math.sin(angle)
        return a1**2 * C11 + a2**2 * C22

    def strain_energy(self, lambda_theta: float,
                        lambda_z: float) -> float:
        """W for biaxial deformation (incompressible: λ_r = 1/(λ_θ λ_z))."""
        lam_r = 1.0 / (lambda_theta * lambda_z)
        I1 = lambda_theta**2 + lambda_z**2 + lam_r**2

        # Ground substance
        W = self.mu / 2.0 * (I1 - 3.0)

        # Fibre contributions (two families)
        for fam in [1, 2]:
            I4 = self._fibre_invariant(lambda_theta**2, lambda_z**2, lam_r**2, fam)
            E = self.kappa * (I1 - 3.0) + (1 - 3 * self.kappa) * (I4 - 1.0)
            if E > 0:
                W += self.k1 / (2.0 * self.k2) * (math.exp(self.k2 * E**2) - 1.0)

        return W

    def circumferential_stress(self, lambda_theta: float,
                                  lambda_z: float,
                                  d_lam: float = 1e-6) -> float:
        """σ_θ = λ_θ ∂W/∂λ_θ / J (Cauchy)."""
        W_plus = self.strain_energy(lambda_theta + d_lam, lambda_z)
        W_minus = self.strain_energy(lambda_theta - d_lam, lambda_z)
        dW = (W_plus - W_minus) / (2 * d_lam)
        return lambda_theta * dW  # incompressible: J=1

    def pressure_inflation(self, lambda_theta: float,
                              lambda_z: float = 1.0,
                              r_inner: float = 1.5e-3,
                              thickness: float = 0.5e-3) -> float:
        """Inflation pressure from Laplace law: P = σ_θ h / r."""
        sigma = self.circumferential_stress(lambda_theta, lambda_z)
        r = r_inner * lambda_theta
        h = thickness / lambda_theta  # incompressible thickness
        return sigma * h / r
