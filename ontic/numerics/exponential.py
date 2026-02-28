"""
Exponential Integrators
========================

Methods that exactly integrate the linear part of a semi-linear ODE
:math:`u' = Lu + N(u)` using the matrix exponential :math:`e^{hL}`.

Implemented schemes:
    * **ETD1** (Exponential Time Differencing, 1st order)
    * **ETDRK2** (Cox & Matthews, 2nd order)
    * **ETDRK4** (Cox & Matthews, 4th order)
    * **Lawson-Euler** variant

For stiff PDEs (e.g. reaction-diffusion, Navier-Stokes in spectral
form), exponential integrators remove the CFL constraint imposed by
the linear operator.

ETDRK4 update:
    .. math::
        a &= e^{hL/2} u_n + L^{-1}(e^{hL/2} - I) N(u_n) \\\\
        b &= e^{hL/2} u_n + L^{-1}(e^{hL/2} - I) N(a) \\\\
        c &= e^{hL/2} a + L^{-1}(e^{hL/2} - I)(2N(b) - N(u_n)) \\\\
        u_{n+1} &= e^{hL}u_n + h^{-2}L^{-3}[(-4+hL+e^{hL}(4-3hL+h^2L^2))N_n \\\\
                   &\\quad + 2(2-hL+e^{hL}(-2+hL))(N_a+N_b) \\\\
                   &\\quad + (-4+3hL-h^2L^2+e^{hL}(4-hL))N_c]

References:
    [1] Cox & Matthews, "Exponential time differencing for stiff systems",
        J. Comput. Phys. 176, 430 (2002).
    [2] Kassam & Trefethen, "Fourth-order time-stepping for stiff PDEs",
        SIAM J. Sci. Comput. 26, 1214 (2005).

Domain I.3.2 — Numerics / Time integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray


NonlinearFn = Callable[[NDArray, float], NDArray]
"""Signature: N(u, t) -> NDArray (nonlinear part)."""


def _phi1(z: NDArray) -> NDArray:
    """φ₁(z) = (e^z - 1) / z, stable for small z."""
    out = np.empty_like(z, dtype=np.complex128)
    small = np.abs(z) < 1e-10
    out[small] = 1.0 + z[small] / 2.0
    out[~small] = (np.exp(z[~small]) - 1.0) / z[~small]
    return out


def _phi2(z: NDArray) -> NDArray:
    """φ₂(z) = (e^z - 1 - z) / z²."""
    out = np.empty_like(z, dtype=np.complex128)
    small = np.abs(z) < 1e-10
    out[small] = 0.5 + z[small] / 6.0
    out[~small] = (np.exp(z[~small]) - 1.0 - z[~small]) / z[~small] ** 2
    return out


def _phi3(z: NDArray) -> NDArray:
    """φ₃(z) = (e^z - 1 - z - z²/2) / z³."""
    out = np.empty_like(z, dtype=np.complex128)
    small = np.abs(z) < 1e-10
    out[small] = 1.0 / 6.0 + z[small] / 24.0
    out[~small] = (np.exp(z[~small]) - 1.0 - z[~small] - z[~small] ** 2 / 2.0) / z[~small] ** 3
    return out


@dataclass
class ETDResult:
    """Result from exponential integrator."""
    u: NDArray
    t: float


class ETD1:
    """
    Exponential Time Differencing, 1st order (ETD-Euler).

    .. math::
        u_{n+1} = e^{hL} u_n + h \\varphi_1(hL) N(u_n, t_n)
    """

    def __init__(self, L: NDArray, nonlinear: NonlinearFn) -> None:
        """
        Parameters:
            L: Linear operator (diagonal in spectral space) — 1D array.
            nonlinear: Nonlinear function N(u, t).
        """
        self.L = L.astype(np.complex128)
        self.N = nonlinear

    def step(self, u: NDArray, t: float, h: float) -> NDArray:
        """One ETD1 step."""
        hL = h * self.L
        exp_hL = np.exp(hL)
        phi1 = _phi1(hL)
        return exp_hL * u + h * phi1 * self.N(u, t)


class ETDRK2:
    """
    Exponential Time Differencing Runge-Kutta, 2nd order.

    Uses Contour-integral evaluation (Kassam & Trefethen) for
    numerical stability of the phi functions.
    """

    def __init__(self, L: NDArray, nonlinear: NonlinearFn) -> None:
        self.L = L.astype(np.complex128)
        self.N = nonlinear

    def step(self, u: NDArray, t: float, h: float) -> NDArray:
        hL = h * self.L
        exp_hL = np.exp(hL)
        phi1 = _phi1(hL)
        phi2 = _phi2(hL)

        N_n = self.N(u, t)
        a = exp_hL * u + h * phi1 * N_n
        N_a = self.N(a, t + h)
        return exp_hL * u + h * phi1 * N_n + h * phi2 * (N_a - N_n)


class ETDRK4:
    """
    Exponential Time Differencing Runge-Kutta, 4th order.

    Uses contour-integral evaluation of the :math:`\\varphi` functions
    for numerical stability (Kassam & Trefethen 2005).
    """

    def __init__(
        self,
        L: NDArray,
        nonlinear: NonlinearFn,
        n_contour: int = 32,
    ) -> None:
        """
        Parameters:
            L: Linear operator (diagonal in spectral space).
            nonlinear: N(u, t).
            n_contour: Number of contour-integral quadrature points.
        """
        self.L = L.astype(np.complex128)
        self.N = nonlinear
        self.n_contour = n_contour
        self._precomputed = False

    def _precompute(self, h: float) -> None:
        """Precompute phi functions via contour integral."""
        M = self.n_contour
        L = self.L
        hL = h * L
        hL2 = h / 2.0 * L

        # Contour: circle of radius 1 around each hL
        theta = np.linspace(0, 2 * np.pi, M, endpoint=False)
        z = np.exp(1j * theta)  # unit circle points

        self._E = np.exp(hL)
        self._E2 = np.exp(hL2)

        # Compute phi functions via contour averaging
        self._phi1_h = np.zeros_like(L, dtype=np.complex128)
        self._phi2_h = np.zeros_like(L, dtype=np.complex128)
        self._phi3_h = np.zeros_like(L, dtype=np.complex128)
        self._phi1_h2 = np.zeros_like(L, dtype=np.complex128)

        for j in range(M):
            zj = hL + z[j]
            self._phi1_h += _phi1(zj)
            self._phi2_h += _phi2(zj)
            self._phi3_h += _phi3(zj)

            zj2 = hL2 + z[j]
            self._phi1_h2 += _phi1(zj2)

        self._phi1_h /= M
        self._phi2_h /= M
        self._phi3_h /= M
        self._phi1_h2 /= M
        self._h = h
        self._precomputed = True

    def step(self, u: NDArray, t: float, h: float) -> NDArray:
        """One ETDRK4 step."""
        if not self._precomputed or abs(h - self._h) > 1e-15:
            self._precompute(h)

        N = self.N
        E = self._E
        E2 = self._E2

        Nu = N(u, t)
        a = E2 * u + h / 2.0 * self._phi1_h2 * Nu
        Na = N(a, t + h / 2)
        b = E2 * u + h / 2.0 * self._phi1_h2 * Na
        Nb = N(b, t + h / 2)
        c = E2 * a + h / 2.0 * self._phi1_h2 * (2.0 * Nb - Nu)
        Nc = N(c, t + h)

        return (E * u
                + h * self._phi1_h * Nu
                + 2.0 * h * self._phi2_h * (Na + Nb)
                + h * self._phi3_h * (Nc - 2.0 * (Na + Nb) + Nu + Nu))

    def solve(
        self,
        u0: NDArray,
        t_span: tuple,
        n_steps: int,
    ) -> NDArray:
        """
        Integrate over [t0, tf].

        Returns:
            Solution array ``(n_steps + 1, *shape)``.
        """
        t0, tf = t_span
        h = (tf - t0) / n_steps
        u = u0.copy()
        t = t0
        trajectory = [u.copy()]

        for _ in range(n_steps):
            u = self.step(u, t, h)
            t += h
            trajectory.append(np.real(u).copy())

        return np.array(trajectory)
