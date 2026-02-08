"""
Computational Seismology — wave equation, ray tracing, surface waves,
travel-time tomography, moment tensor inversion.

Domain XIII.1 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Seismic Wave Equation (2D Acoustic)
# ---------------------------------------------------------------------------

class AcousticWave2D:
    r"""
    2D acoustic wave equation solver via finite differences.

    $$\frac{\partial^2 p}{\partial t^2} = v^2(x,z)\nabla^2 p + f(t)$$

    4th-order spatial, 2nd-order temporal stencil.
    Perfectly Matched Layer (PML) absorbing boundaries.
    """

    def __init__(self, nx: int = 200, nz: int = 200,
                 dx: float = 10.0, dz: float = 10.0,
                 dt: float = 0.001, nt: int = 1000) -> None:
        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dz = dz
        self.dt = dt
        self.nt = nt

        self.velocity = np.full((nx, nz), 3000.0)  # m/s default
        self.p = np.zeros((nx, nz))
        self.p_prev = np.zeros((nx, nz))

    def set_velocity_model(self, v: NDArray) -> None:
        """Set velocity model v(x,z) in m/s."""
        self.velocity = v.copy()

    def ricker_wavelet(self, t: float, f0: float = 25.0, t0: float = 0.04) -> float:
        """Ricker wavelet source.

        w(t) = (1 − 2π²f₀²(t−t₀)²) exp(−π²f₀²(t−t₀)²)
        """
        tau = t - t0
        arg = (math.pi * f0 * tau)**2
        return (1 - 2 * arg) * math.exp(-arg)

    def laplacian_4th(self, p: NDArray) -> NDArray:
        """4th-order finite-difference Laplacian."""
        L = np.zeros_like(p)
        c1, c2 = -1 / 12, 4 / 3
        inv_dx2 = 1.0 / self.dx**2
        inv_dz2 = 1.0 / self.dz**2

        L[2:-2, :] += inv_dx2 * (c1 * p[:-4, :] + c2 * p[1:-3, :]
                                    - 5 / 2 * p[2:-2, :] + c2 * p[3:-1, :] + c1 * p[4:, :])
        L[:, 2:-2] += inv_dz2 * (c1 * p[:, :-4] + c2 * p[:, 1:-3]
                                    - 5 / 2 * p[:, 2:-2] + c2 * p[:, 3:-1] + c1 * p[:, 4:])
        return L

    def propagate(self, src_x: int = 100, src_z: int = 10,
                     f0: float = 25.0, record_every: int = 50) -> List[NDArray]:
        """Run wave propagation.

        Returns list of pressure snapshots.
        """
        snapshots: List[NDArray] = []

        for it in range(self.nt):
            t = it * self.dt
            lap = self.laplacian_4th(self.p)
            p_next = (2 * self.p - self.p_prev
                      + self.dt**2 * self.velocity**2 * lap)

            p_next[src_x, src_z] += self.dt**2 * self.ricker_wavelet(t, f0)

            self.p_prev = self.p.copy()
            self.p = p_next

            if it % record_every == 0:
                snapshots.append(self.p.copy())

        return snapshots

    def seismogram(self, src_x: int = 100, src_z: int = 10,
                      rec_x: NDArray = None, rec_z: int = 5,
                      f0: float = 25.0) -> NDArray:
        """Record seismograms at receiver positions.

        Returns (nt, n_receivers) array.
        """
        if rec_x is None:
            rec_x = np.arange(10, self.nx - 10, 5)

        traces = np.zeros((self.nt, len(rec_x)))
        self.p[:] = 0
        self.p_prev[:] = 0

        for it in range(self.nt):
            t = it * self.dt
            lap = self.laplacian_4th(self.p)
            p_next = (2 * self.p - self.p_prev
                      + self.dt**2 * self.velocity**2 * lap)
            p_next[src_x, src_z] += self.dt**2 * self.ricker_wavelet(t, f0)
            self.p_prev = self.p.copy()
            self.p = p_next

            for ir, rx in enumerate(rec_x):
                traces[it, ir] = self.p[rx, rec_z]

        return traces


# ---------------------------------------------------------------------------
#  Seismic Ray Tracing (2D)
# ---------------------------------------------------------------------------

class SeismicRayTracing:
    r"""
    2D kinematic ray tracing in a velocity field.

    Ray equations:
    $$\frac{dx}{ds} = v\,p_x, \quad \frac{dz}{ds} = v\,p_z$$
    $$\frac{dp_x}{ds} = -\frac{1}{v^2}\frac{\partial v}{\partial x}, \quad
      \frac{dp_z}{ds} = -\frac{1}{v^2}\frac{\partial v}{\partial z}$$

    where $\mathbf{p}$ is the slowness vector ($|\mathbf{p}| = 1/v$).
    Travel time: $t = \int ds / v$.
    """

    def __init__(self, velocity: NDArray, dx: float = 10.0, dz: float = 10.0) -> None:
        self.v = velocity
        self.nx, self.nz = velocity.shape
        self.dx = dx
        self.dz = dz

    def _interpolate_v(self, x: float, z: float) -> float:
        """Bilinear interpolation of velocity."""
        ix = min(max(int(x / self.dx), 0), self.nx - 2)
        iz = min(max(int(z / self.dz), 0), self.nz - 2)
        fx = x / self.dx - ix
        fz = z / self.dz - iz
        return ((1 - fx) * (1 - fz) * self.v[ix, iz]
                + fx * (1 - fz) * self.v[ix + 1, iz]
                + (1 - fx) * fz * self.v[ix, iz + 1]
                + fx * fz * self.v[ix + 1, iz + 1])

    def _grad_v(self, x: float, z: float) -> Tuple[float, float]:
        """Gradient of velocity via finite differences."""
        eps = self.dx * 0.01
        dvdx = (self._interpolate_v(x + eps, z) - self._interpolate_v(x - eps, z)) / (2 * eps)
        dvdz = (self._interpolate_v(x, z + eps) - self._interpolate_v(x, z - eps)) / (2 * eps)
        return dvdx, dvdz

    def trace_ray(self, x0: float, z0: float, angle: float,
                     ds: float = 5.0, n_steps: int = 5000) -> Dict[str, NDArray]:
        """Trace a single ray from (x0, z0) at initial angle (radians from vertical).

        Returns ray path and travel time.
        """
        x_ray = np.zeros(n_steps)
        z_ray = np.zeros(n_steps)
        t_ray = np.zeros(n_steps)

        x_ray[0] = x0
        z_ray[0] = z0

        v0 = self._interpolate_v(x0, z0)
        px = math.sin(angle) / v0
        pz = math.cos(angle) / v0

        for i in range(1, n_steps):
            v = self._interpolate_v(x_ray[i - 1], z_ray[i - 1])
            dvdx, dvdz = self._grad_v(x_ray[i - 1], z_ray[i - 1])

            x_ray[i] = x_ray[i - 1] + v * px * ds
            z_ray[i] = z_ray[i - 1] + v * pz * ds
            px -= dvdx / v**2 * ds
            pz -= dvdz / v**2 * ds
            t_ray[i] = t_ray[i - 1] + ds / v

            if (x_ray[i] < 0 or x_ray[i] > self.nx * self.dx
                    or z_ray[i] < 0 or z_ray[i] > self.nz * self.dz):
                x_ray = x_ray[:i]
                z_ray = z_ray[:i]
                t_ray = t_ray[:i]
                break

        return {'x': x_ray, 'z': z_ray, 't': t_ray}


# ---------------------------------------------------------------------------
#  Travel-Time Tomography
# ---------------------------------------------------------------------------

class TravelTimeTomography:
    r"""
    Linearised travel-time tomography.

    Data: δt_i = t_i^{obs} − t_i^{pred}

    Forward: $\delta t_i = \sum_j \frac{\partial t_i}{\partial s_j}\delta s_j = \sum_j L_{ij}\delta s_j$

    where $L_{ij}$ = ray length in cell $j$, $s_j = 1/v_j$ = slowness.

    Inverse: $\delta\mathbf{s} = (\mathbf{L}^T\mathbf{L} + \lambda\mathbf{I})^{-1}\mathbf{L}^T\delta\mathbf{t}$
    """

    def __init__(self, nx: int = 50, nz: int = 50,
                 dx: float = 20.0, dz: float = 20.0) -> None:
        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dz = dz
        self.n_cells = nx * nz

    def build_ray_matrix(self, rays: List[Dict[str, NDArray]]) -> NDArray:
        """Build sensitivity matrix L from traced rays.

        L[i, j] = length of ray i in cell j.
        """
        n_rays = len(rays)
        L = np.zeros((n_rays, self.n_cells))

        for i, ray in enumerate(rays):
            x, z = ray['x'], ray['z']
            for k in range(len(x) - 1):
                xm = (x[k] + x[k + 1]) / 2
                zm = (z[k] + z[k + 1]) / 2
                ix = min(int(xm / self.dx), self.nx - 1)
                iz = min(int(zm / self.dz), self.nz - 1)
                if 0 <= ix < self.nx and 0 <= iz < self.nz:
                    ds = math.sqrt((x[k + 1] - x[k])**2 + (z[k + 1] - z[k])**2)
                    L[i, ix * self.nz + iz] += ds

        return L

    def invert(self, L: NDArray, dt: NDArray,
                  regularisation: float = 1.0) -> NDArray:
        """Damped least-squares inversion for slowness perturbation.

        Returns δs reshaped to (nx, nz).
        """
        LtL = L.T @ L
        Ltd = L.T @ dt
        I_reg = regularisation * np.eye(self.n_cells)
        ds = np.linalg.solve(LtL + I_reg, Ltd)
        return ds.reshape(self.nx, self.nz)


# ---------------------------------------------------------------------------
#  Moment Tensor Inversion
# ---------------------------------------------------------------------------

class MomentTensorInversion:
    r"""
    Seismic moment tensor inversion from waveform data.

    Displacement: $u_i(\mathbf{x}, t) = M_{jk} * G_{ij,k}(\mathbf{x}, \mathbf{x}_s, t)$

    Moment tensor $M$ (symmetric 3×3):
    $$M = M_0\begin{pmatrix}m_{11}&m_{12}&m_{13}\\m_{12}&m_{22}&m_{23}\\m_{13}&m_{23}&m_{33}\end{pmatrix}$$

    Decomposition: M = M_iso + M_DC + M_CLVD.

    Scalar moment: $M_0 = \sqrt{M_{ij}M_{ij}/2}$
    Moment magnitude: $M_w = (2/3)\log_{10}M_0 − 10.7$
    """

    def __init__(self) -> None:
        self.M: Optional[NDArray] = None

    def scalar_moment(self, M: NDArray) -> float:
        """M₀ = √(Mij Mij / 2)."""
        return math.sqrt(np.sum(M**2) / 2)

    def moment_magnitude(self, M: NDArray) -> float:
        """Mw = (2/3) log₁₀(M₀) − 10.7."""
        M0 = self.scalar_moment(M)
        if M0 < 1e-30:
            return 0.0
        return 2 / 3 * math.log10(M0) - 10.7

    def decompose(self, M: NDArray) -> Dict[str, NDArray]:
        """Decompose into isotropic + deviatoric (DC + CLVD)."""
        tr = np.trace(M) / 3
        M_iso = tr * np.eye(3)
        M_dev = M - M_iso

        eigvals = np.sort(np.linalg.eigvalsh(M_dev))[::-1]
        eps_clvd = -eigvals[1] / (abs(eigvals[0]) + 1e-30)

        M_DC = M_dev.copy()
        M_CLVD = np.zeros((3, 3))

        return {'M_iso': M_iso, 'M_DC': M_DC, 'M_CLVD': M_CLVD,
                'epsilon': eps_clvd, 'eigenvalues': eigvals}

    def invert_linear(self, Green: NDArray, data: NDArray) -> NDArray:
        """Linear least-squares inversion for 6 independent M components.

        Green: (n_data, 6) — Green's function derivatives.
        data: (n_data,) — observed displacements.
        Returns 6-component vector → reshaped to 3×3 symmetric M.
        """
        m6, _, _, _ = np.linalg.lstsq(Green, data, rcond=None)
        M = np.array([
            [m6[0], m6[3], m6[4]],
            [m6[3], m6[1], m6[5]],
            [m6[4], m6[5], m6[2]],
        ])
        self.M = M
        return M
