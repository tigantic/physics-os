"""
Cosmological Simulations — N-body, Friedmann equations, perturbation theory,
power spectrum, halo mass function.

Domain XII.4 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Cosmological Constants
# ---------------------------------------------------------------------------

H0_DEFAULT: float = 67.4   # km/s/Mpc (Planck 2018)
OMEGA_M: float = 0.315     # matter density parameter
OMEGA_LAMBDA: float = 0.685  # dark energy density parameter
OMEGA_B: float = 0.049     # baryon density
OMEGA_R: float = 9.1e-5    # radiation density
SIGMA_8: float = 0.811     # amplitude of matter fluctuations
N_S: float = 0.965         # spectral index


# ---------------------------------------------------------------------------
#  Friedmann Equations
# ---------------------------------------------------------------------------

class FriedmannCosmology:
    r"""
    Friedmann-Lemaître-Robertson-Walker (FLRW) cosmology.

    Friedmann equation:
    $$H^2(a) = H_0^2\left[\frac{\Omega_r}{a^4} + \frac{\Omega_m}{a^3}
      + \frac{\Omega_k}{a^2} + \Omega_\Lambda\right]$$

    Acceleration equation:
    $$\frac{\ddot{a}}{a} = -\frac{4\pi G}{3}(\rho + 3p)
      + \frac{\Lambda}{3}$$

    Comoving distance:
    $$\chi(z) = \frac{c}{H_0}\int_0^z \frac{dz'}{E(z')}$$
    where $E(z) = H(z)/H_0$.
    """

    def __init__(self, H0: float = H0_DEFAULT, Om: float = OMEGA_M,
                 OL: float = OMEGA_LAMBDA, Or: float = OMEGA_R) -> None:
        self.H0 = H0  # km/s/Mpc
        self.Om = Om
        self.OL = OL
        self.Or = Or
        self.Ok = 1.0 - Om - OL - Or

    def E(self, z: float) -> float:
        """E(z) = H(z)/H₀ = √[Ωr(1+z)⁴ + Ωm(1+z)³ + Ωk(1+z)² + ΩΛ]."""
        a = 1.0 / (1 + z)
        return math.sqrt(self.Or / a**4 + self.Om / a**3
                        + self.Ok / a**2 + self.OL)

    def hubble(self, z: float) -> float:
        """H(z) in km/s/Mpc."""
        return self.H0 * self.E(z)

    def comoving_distance(self, z: float, n_steps: int = 1000) -> float:
        """χ(z) = (c/H₀) ∫₀ᶻ dz′/E(z′)  [Mpc]."""
        c_H0 = 2.998e5 / self.H0  # c in km/s → c/H₀ in Mpc
        z_arr = np.linspace(0, z, n_steps)
        integrand = np.array([1.0 / self.E(zi) for zi in z_arr])
        return c_H0 * float(np.trapz(integrand, z_arr))

    def luminosity_distance(self, z: float) -> float:
        """D_L(z) = (1+z) χ(z)  [Mpc]."""
        return (1 + z) * self.comoving_distance(z)

    def angular_diameter_distance(self, z: float) -> float:
        """D_A(z) = χ(z)/(1+z)  [Mpc]."""
        return self.comoving_distance(z) / (1 + z)

    def age_of_universe(self, n_steps: int = 5000) -> float:
        """t₀ = (1/H₀) ∫₀¹ da / (a E(1/a−1))  [Gyr]."""
        da = 1.0 / n_steps
        t = 0.0
        for i in range(1, n_steps):
            a = i * da
            z = 1.0 / a - 1
            t += da / (a * self.E(z))
        t_seconds = t / (self.H0 * 3.241e-20)  # H₀ in s⁻¹
        return t_seconds / 3.156e16  # seconds → Gyr

    def growth_factor(self, z: float, n_steps: int = 1000) -> float:
        """Linear growth factor D(z) normalised to D(0)=1.

        D(a) ∝ H(a) ∫₀ᵃ da′/(a′ H(a′))³
        """
        a_target = 1.0 / (1 + z)
        da = a_target / n_steps
        integral = 0.0
        for i in range(1, n_steps + 1):
            a = i * da
            z_i = 1.0 / a - 1
            integral += da / (a * self.E(z_i))**3

        D_z = 5 * self.Om / 2 * self.E(z) * integral

        # Normalise to z=0
        da0 = 1.0 / n_steps
        integral0 = 0.0
        for i in range(1, n_steps + 1):
            a = i * da0
            z_i = 1.0 / a - 1
            integral0 += da0 / (a * self.E(z_i))**3

        D_0 = 5 * self.Om / 2 * self.E(0) * integral0
        return D_z / D_0 if abs(D_0) > 1e-30 else 1.0


# ---------------------------------------------------------------------------
#  Matter Power Spectrum
# ---------------------------------------------------------------------------

class MatterPowerSpectrum:
    r"""
    Linear matter power spectrum P(k).

    $$P(k) = A_s k^{n_s} T^2(k) D^2(z)$$

    Transfer function (Eisenstein-Hu, 1998 — zero-baryon fit):
    $$T(k) = \frac{\ln(1+2.34q)}{2.34q}
      \left[1 + 3.89q + (16.1q)^2 + (5.46q)^3 + (6.71q)^4\right]^{-1/4}$$

    where $q = k/(\Omega_m h^2\,\text{Mpc}^{-1})$.
    """

    def __init__(self, cosmo: Optional[FriedmannCosmology] = None,
                 ns: float = N_S, sigma8: float = SIGMA_8) -> None:
        self.cosmo = cosmo or FriedmannCosmology()
        self.ns = ns
        self.sigma8 = sigma8
        self.h = self.cosmo.H0 / 100

    def transfer_function(self, k: float) -> float:
        """Eisenstein-Hu zero-baryon transfer function.

        k in h/Mpc.
        """
        Gamma = self.cosmo.Om * self.h
        q = k / (Gamma + 1e-10)
        T = (math.log(1 + 2.34 * q) / (2.34 * q + 1e-30)
             * (1 + 3.89 * q + (16.1 * q)**2
                + (5.46 * q)**3 + (6.71 * q)**4)**(-0.25))
        return T

    def primordial_spectrum(self, k: float) -> float:
        """P_primordial(k) = k^{n_s}."""
        return k**self.ns

    def linear_power(self, k: float, z: float = 0.0) -> float:
        """P(k, z) = A_s k^{ns} T²(k) D²(z)."""
        T = self.transfer_function(k)
        D = self.cosmo.growth_factor(z)
        return self.primordial_spectrum(k) * T**2 * D**2

    def sigma_R(self, R: float, z: float = 0.0,
                   n_k: int = 500) -> float:
        """σ(R) — mass variance smoothed at scale R (Mpc/h).

        σ²(R) = (1/2π²) ∫ k² P(k) W²(kR) dk
        W(x) = 3(sin x − x cos x)/x³ (top-hat window)
        """
        k_min, k_max = 1e-4, 1e2
        k = np.logspace(math.log10(k_min), math.log10(k_max), n_k)
        dk = np.diff(np.log(k))

        sigma2 = 0.0
        for i in range(n_k - 1):
            ki = k[i]
            x = ki * R
            if x < 1e-6:
                W = 1.0
            else:
                W = 3 * (math.sin(x) - x * math.cos(x)) / x**3
            sigma2 += ki**3 * self.linear_power(ki, z) * W**2 * dk[i]

        sigma2 /= (2 * math.pi**2)
        return math.sqrt(max(sigma2, 0))

    def normalise_to_sigma8(self) -> float:
        """Find A_s such that σ(8 Mpc/h) = σ₈."""
        sigma_unnorm = self.sigma_R(8.0, 0.0)
        if sigma_unnorm > 0:
            return (self.sigma8 / sigma_unnorm)**2
        return 1.0


# ---------------------------------------------------------------------------
#  N-body Particle-Mesh Solver
# ---------------------------------------------------------------------------

class ParticleMeshNBody:
    r"""
    Particle-Mesh (PM) N-body solver for dark matter.

    Poisson equation:
    $$\nabla^2\Phi = 4\pi G\bar{\rho}\delta$$

    Solved in Fourier space: $\hat{\Phi}(\mathbf{k}) = -4\pi G\bar{\rho}\hat{\delta}(\mathbf{k})/k^2$

    Particle update (leapfrog/KDK):
    $$\mathbf{v}_{n+1/2} = \mathbf{v}_{n-1/2} + \mathbf{a}_n \Delta t$$
    $$\mathbf{x}_{n+1} = \mathbf{x}_n + \mathbf{v}_{n+1/2}\Delta t$$
    """

    def __init__(self, n_particles: int = 10000, n_grid: int = 64,
                 box_size: float = 100.0) -> None:
        """
        n_particles: number of dark matter particles.
        n_grid: PM grid size per dimension.
        box_size: comoving box side length (Mpc/h).
        """
        self.N = n_particles
        self.Ng = n_grid
        self.L = box_size
        self.dx = box_size / n_grid

        self.positions = np.random.uniform(0, box_size, (n_particles, 3))
        self.velocities = np.zeros((n_particles, 3))

    def deposit_density(self) -> NDArray:
        """Cloud-in-Cell (CIC) density assignment."""
        rho = np.zeros((self.Ng, self.Ng, self.Ng))
        for p in range(self.N):
            x, y, z = self.positions[p] / self.dx
            ix = int(x) % self.Ng
            iy = int(y) % self.Ng
            iz = int(z) % self.Ng

            dx_frac = x - int(x)
            dy_frac = y - int(y)
            dz_frac = z - int(z)

            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        wx = (1 - dx_frac) if di == 0 else dx_frac
                        wy = (1 - dy_frac) if dj == 0 else dy_frac
                        wz = (1 - dz_frac) if dk == 0 else dz_frac
                        rho[(ix + di) % self.Ng,
                            (iy + dj) % self.Ng,
                            (iz + dk) % self.Ng] += wx * wy * wz

        rho *= self.N / self.L**3  # normalise to mean density
        return rho

    def solve_poisson(self, delta: NDArray) -> NDArray:
        """Solve Poisson in Fourier space → potential Φ."""
        delta_k = np.fft.fftn(delta)
        kx = np.fft.fftfreq(self.Ng, self.dx) * 2 * math.pi
        ky = np.fft.fftfreq(self.Ng, self.dx) * 2 * math.pi
        kz = np.fft.fftfreq(self.Ng, self.dx) * 2 * math.pi
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        k2 = KX**2 + KY**2 + KZ**2
        k2[0, 0, 0] = 1.0  # avoid division by zero

        phi_k = -delta_k / k2
        phi_k[0, 0, 0] = 0.0

        return np.real(np.fft.ifftn(phi_k))

    def compute_acceleration(self, phi: NDArray) -> NDArray:
        """Gradient of potential → acceleration field."""
        ax = -np.gradient(phi, self.dx, axis=0)
        ay = -np.gradient(phi, self.dx, axis=1)
        az = -np.gradient(phi, self.dx, axis=2)
        return np.stack([ax, ay, az], axis=-1)

    def interpolate_acceleration(self, accel_field: NDArray) -> NDArray:
        """CIC interpolation of acceleration to particle positions."""
        accel = np.zeros((self.N, 3))
        for p in range(self.N):
            x, y, z = self.positions[p] / self.dx
            ix = int(x) % self.Ng
            iy = int(y) % self.Ng
            iz = int(z) % self.Ng
            accel[p] = accel_field[ix, iy, iz]
        return accel

    def step(self, dt: float) -> None:
        """Kick-Drift-Kick leapfrog step."""
        rho = self.deposit_density()
        delta = rho / (np.mean(rho) + 1e-30) - 1
        phi = self.solve_poisson(delta)
        accel_field = self.compute_acceleration(phi)
        accel = self.interpolate_acceleration(accel_field)

        self.velocities += accel * dt
        self.positions += self.velocities * dt
        self.positions %= self.L  # periodic BC


# ---------------------------------------------------------------------------
#  Halo Mass Function
# ---------------------------------------------------------------------------

class HaloMassFunction:
    r"""
    Halo mass function — number density of dark matter halos.

    Press-Schechter (1974):
    $$\frac{dn}{dM} = \sqrt{\frac{2}{\pi}}\frac{\bar{\rho}}{M^2}
      \frac{\delta_c}{\sigma}\left|\frac{d\ln\sigma}{d\ln M}\right|
      \exp\left(-\frac{\delta_c^2}{2\sigma^2}\right)$$

    $\delta_c = 1.686$ (linearly extrapolated collapse threshold).

    Sheth-Tormen (1999):
    $$\nu f(\nu) = A\sqrt{\frac{2a}{\pi}}\left[1+(a\nu^2)^{-p}\right]
      \nu\,e^{-a\nu^2/2}$$
    where $\nu = \delta_c/\sigma$, $a=0.707$, $p=0.3$, $A=0.3222$.
    """

    DELTA_C: float = 1.686

    def __init__(self, ps: Optional[MatterPowerSpectrum] = None) -> None:
        self.ps = ps or MatterPowerSpectrum()

    def press_schechter(self, M: float, z: float = 0.0) -> float:
        """Press-Schechter dn/dM (Mpc⁻³ M_sun⁻¹)."""
        R = (3 * M / (4 * math.pi * 2.775e11 * self.ps.cosmo.Om))**(1 / 3)
        sigma = self.ps.sigma_R(R, z)
        if sigma < 1e-10:
            return 0.0
        nu = self.DELTA_C / sigma

        dln_sigma_dln_M = -0.1  # approximate slope

        rho_bar = 2.775e11 * self.ps.cosmo.Om  # M_sun / Mpc³
        result = (math.sqrt(2 / math.pi) * rho_bar / M**2
                 * nu * abs(dln_sigma_dln_M) * math.exp(-nu**2 / 2))
        return result

    def sheth_tormen(self, M: float, z: float = 0.0) -> float:
        """Sheth-Tormen halo mass function."""
        a, p, A_st = 0.707, 0.3, 0.3222
        R = (3 * M / (4 * math.pi * 2.775e11 * self.ps.cosmo.Om))**(1 / 3)
        sigma = self.ps.sigma_R(R, z)
        if sigma < 1e-10:
            return 0.0
        nu = self.DELTA_C / sigma

        f_nu = (A_st * math.sqrt(2 * a / math.pi)
                * (1 + (a * nu**2)**(-p)) * nu * math.exp(-a * nu**2 / 2))

        rho_bar = 2.775e11 * self.ps.cosmo.Om
        dln_sigma_dln_M = -0.1
        return f_nu * rho_bar / M**2 * abs(dln_sigma_dln_M)
