"""
Heat transfer module — radiation, participating media, solidification, conjugate.

Upgrades domain V.5 from conduction/convection only (tensornet/cfd/thermal_qtt.py)
to full radiative + coupled heat transfer:
  - Monte Carlo view factor computation
  - Radiosity network method (diffuse enclosures)
  - Discrete Ordinates Method (SN) for participating media RTE
  - Stefan solidification (enthalpy method, moving front)
  - Conjugate conduction-convection-radiation CHT coupling

SI units: temperatures [K], heat fluxes [W/m²], lengths [m].
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
STEFAN_BOLTZMANN: float = 5.670374419e-8   # W/(m²·K⁴)
PI: float = math.pi


# ===================================================================
#  Data Structures
# ===================================================================

@dataclass
class ViewFactorResult:
    """Result of view factor computation."""
    matrix: NDArray[np.float64]         # (N, N) view factor matrix F_ij
    n_rays: int                         # Rays used per surface
    statistical_error: NDArray[np.float64]  # (N, N) ±1σ error estimate


@dataclass
class RadiosityResult:
    """Result of radiosity computation."""
    radiosities: NDArray[np.float64]    # (N,) [W/m²]
    irradiances: NDArray[np.float64]    # (N,) [W/m²]
    net_heat_fluxes: NDArray[np.float64]  # (N,) [W/m²]
    temperatures: NDArray[np.float64]   # (N,) [K]


# ===================================================================
#  View Factor (Monte Carlo Ray Tracing)
# ===================================================================

class ViewFactorMC:
    r"""
    Monte Carlo view factor computation by ray tracing.

    The view factor $F_{ij}$ is the fraction of radiation leaving surface $i$
    that arrives at surface $j$:

    $$F_{ij} = \frac{1}{A_i}\int_{A_i}\int_{A_j}
        \frac{\cos\theta_i\cos\theta_j}{\pi r^2} dA_j\, dA_i$$

    MC approach: emit $N$ rays from each surface $i$ with cosine-weighted
    direction, trace to first intersection → $F_{ij} \approx N_{ij}/N$.

    Supports arbitrary triangulated surfaces.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    def _cosine_hemisphere_sample(self, normal: NDArray) -> NDArray:
        """
        Sample direction from cosine-weighted hemisphere around normal.

        Uses Malley's method: sample unit disk, project to hemisphere.
        """
        # Build local frame
        if abs(normal[0]) < 0.9:
            tangent = np.cross(normal, np.array([1.0, 0.0, 0.0]))
        else:
            tangent = np.cross(normal, np.array([0.0, 1.0, 0.0]))
        tangent /= np.linalg.norm(tangent)
        bitangent = np.cross(normal, tangent)

        # Sample disk
        r = math.sqrt(self.rng.random())
        phi = 2.0 * PI * self.rng.random()
        x = r * math.cos(phi)
        y = r * math.sin(phi)
        z = math.sqrt(max(0.0, 1.0 - r * r))

        return x * tangent + y * bitangent + z * normal

    def _ray_triangle_intersect(self, origin: NDArray, direction: NDArray,
                                 v0: NDArray, v1: NDArray,
                                 v2: NDArray) -> float:
        """
        Möller-Trumbore ray-triangle intersection.

        Returns distance t > 0 if hit, else np.inf.
        """
        e1 = v1 - v0
        e2 = v2 - v0
        pvec = np.cross(direction, e2)
        det = np.dot(e1, pvec)

        if abs(det) < 1e-12:
            return np.inf

        inv_det = 1.0 / det
        tvec = origin - v0
        u = np.dot(tvec, pvec) * inv_det
        if u < 0.0 or u > 1.0:
            return np.inf

        qvec = np.cross(tvec, e1)
        v = np.dot(direction, qvec) * inv_det
        if v < 0.0 or u + v > 1.0:
            return np.inf

        t = np.dot(e2, qvec) * inv_det
        return t if t > 1e-8 else np.inf

    def compute(self, vertices: List[NDArray[np.float64]],
                triangles: NDArray[np.int64],
                normals: NDArray[np.float64],
                areas: NDArray[np.float64],
                n_rays: int = 100000) -> ViewFactorResult:
        """
        Compute view factor matrix by MC ray tracing.

        Parameters
        ----------
        vertices : List of (3,) vertex positions.
        triangles : (M, 3) triangle vertex indices, grouped by surface.
        normals : (N_surf,3) outward normals per surface.
        areas : (N_surf,) surface areas [m²].
        n_rays : Rays per surface.

        Returns
        -------
        ViewFactorResult with F_ij matrix and errors.
        """
        N_surf = len(normals)
        F = np.zeros((N_surf, N_surf))
        hits = np.zeros((N_surf, N_surf), dtype=int)

        verts = np.array(vertices)
        tris = np.array(triangles)
        norms = np.array(normals)
        n_tris = len(tris)

        # Map each triangle to its surface (assume surfaces listed sequentially)
        # For simplicity, assume one triangle per surface if not specified
        tri_to_surf: NDArray = np.zeros(n_tris, dtype=int)
        if n_tris == N_surf:
            tri_to_surf = np.arange(N_surf)
        else:
            # User must provide mapping; default: sequential grouping
            tris_per_surf = n_tris // N_surf
            for s in range(N_surf):
                start = s * tris_per_surf
                end = start + tris_per_surf if s < N_surf - 1 else n_tris
                tri_to_surf[start:end] = s

        for i in range(N_surf):
            # Get triangles belonging to surface i
            surf_tris = np.where(tri_to_surf == i)[0]
            if len(surf_tris) == 0:
                continue

            for _ in range(n_rays):
                # Random point on surface i (pick random triangle, random point on it)
                tri_idx = self.rng.choice(surf_tris)
                v0, v1, v2 = verts[tris[tri_idx]]
                # Uniform random point on triangle
                u1, u2 = self.rng.random(), self.rng.random()
                if u1 + u2 > 1.0:
                    u1 = 1.0 - u1
                    u2 = 1.0 - u2
                origin = v0 + u1 * (v1 - v0) + u2 * (v2 - v0)
                origin += norms[i] * 1e-6  # offset to avoid self-intersection

                # Cosine-weighted direction
                direction = self._cosine_hemisphere_sample(norms[i])

                # Trace ray against all other triangles
                t_min = np.inf
                hit_surf = -1
                for t in range(n_tris):
                    if tri_to_surf[t] == i:
                        continue
                    t_val = self._ray_triangle_intersect(
                        origin, direction,
                        verts[tris[t, 0]], verts[tris[t, 1]], verts[tris[t, 2]])
                    if t_val < t_min:
                        t_min = t_val
                        hit_surf = tri_to_surf[t]

                if hit_surf >= 0:
                    hits[i, hit_surf] += 1

            F[i, :] = hits[i, :] / n_rays

        # Statistical error: binomial σ = √(p(1-p)/N)
        errors = np.sqrt(F * (1.0 - F) / n_rays)

        return ViewFactorResult(matrix=F, n_rays=n_rays, statistical_error=errors)

    @staticmethod
    def parallel_rectangles(w: float, h: float, d: float) -> float:
        """
        Analytical view factor: two parallel equal rectangles of width w, height h,
        separated by distance d.

        Hottel's crossed-string formula for 2D analog; for 3D rectangles uses
        standard formula with double integrals reduced to:
        """
        X = w / d
        Y = h / d
        lnA = ((1 + X**2) * (1 + Y**2)) / (1 + X**2 + Y**2)
        t1 = math.log(lnA)
        t2 = X * math.sqrt(1 + Y**2) * math.atan(X / math.sqrt(1 + Y**2))
        t3 = Y * math.sqrt(1 + X**2) * math.atan(Y / math.sqrt(1 + X**2))
        t4 = X * math.atan(X) + Y * math.atan(Y)
        F = 2.0 / (PI * X * Y) * (t1 / 2.0 + t2 + t3 - t4)
        return max(0.0, min(1.0, F))

    @staticmethod
    def perpendicular_rectangles(w: float, h1: float, h2: float) -> float:
        """
        Analytical view factor: two perpendicular rectangles sharing a common edge
        of length w. Heights h1, h2.
        """
        H = h1 / w
        W = h2 / w
        A = (1 + H**2) * (1 + W**2) / (1 + H**2 + W**2)
        B = ((W**2 * (1 + H**2 + W**2)) /
             ((1 + W**2) * (H**2 + W**2)))
        C = ((H**2 * (1 + H**2 + W**2)) /
             ((1 + H**2) * (H**2 + W**2)))
        F = (1.0 / (PI * W)) * (
            W * math.atan(1.0 / W) + H * math.atan(1.0 / H)
            - math.sqrt(H**2 + W**2) * math.atan(1.0 / math.sqrt(H**2 + W**2))
            + 0.25 * math.log(A * B ** (W**2) * C ** (H**2)))
        return max(0.0, min(1.0, F))


# ===================================================================
#  Radiosity Network Method
# ===================================================================

class RadiosityNetwork:
    r"""
    Radiosity method for diffuse enclosures.

    For N surfaces with emissivities $\varepsilon_i$ and view factors $F_{ij}$:

    $$B_i = \varepsilon_i \sigma T_i^4 + (1 - \varepsilon_i) \sum_j F_{ij} B_j$$

    where $B_i$ is the radiosity [W/m²].

    In matrix form: $(I - \text{diag}(\rho) F) B = \varepsilon \sigma T^4$
    where $\rho_i = 1 - \varepsilon_i$ are reflectivities.
    """

    def __init__(self, emissivities: NDArray[np.float64],
                 view_factors: NDArray[np.float64],
                 areas: NDArray[np.float64]) -> None:
        """
        Parameters
        ----------
        emissivities : (N,) surface emissivities (0 < ε ≤ 1).
        view_factors : (N, N) view factor matrix.
        areas : (N,) surface areas [m²].
        """
        N = len(emissivities)
        if view_factors.shape != (N, N):
            raise ValueError(f"View factor matrix shape {view_factors.shape} ≠ ({N},{N})")
        if len(areas) != N:
            raise ValueError(f"Areas length {len(areas)} ≠ {N}")

        self.eps = np.array(emissivities, dtype=np.float64)
        self.F = np.array(view_factors, dtype=np.float64)
        self.A = np.array(areas, dtype=np.float64)
        self.N = N
        self.rho = 1.0 - self.eps  # reflectivity

    def solve(self, temperatures: NDArray[np.float64]) -> RadiosityResult:
        """
        Solve radiosity equations for given surface temperatures.

        Parameters
        ----------
        temperatures : (N,) surface temperatures [K].

        Returns
        -------
        RadiosityResult with radiosities, irradiances, and net heat fluxes.
        """
        T = np.array(temperatures, dtype=np.float64)
        eb = STEFAN_BOLTZMANN * T**4  # Blackbody emissive power

        # Build system: (I - diag(rho) @ F) @ B = eps * eb
        rho_diag = np.diag(self.rho)
        A_mat = np.eye(self.N) - rho_diag @ self.F
        rhs = self.eps * eb

        # Solve for radiosities
        B = np.linalg.solve(A_mat, rhs)

        # Irradiance: G_i = Σ_j F_ij B_j
        G = self.F @ B

        # Net heat flux: q_i = B_i - G_i
        q_net = B - G

        return RadiosityResult(
            radiosities=B,
            irradiances=G,
            net_heat_fluxes=q_net,
            temperatures=T.copy(),
        )

    def solve_mixed(self,
                    temperatures: NDArray[np.float64],
                    heat_fluxes: NDArray[np.float64],
                    known_T: NDArray[np.bool_]) -> RadiosityResult:
        """
        Solve with mixed boundary conditions: some surfaces have known T,
        others have known heat flux q.

        Parameters
        ----------
        temperatures : (N,) known temperatures [K] (ignored where known_T=False).
        heat_fluxes : (N,) known net fluxes [W/m²] (used where known_T=False).
        known_T : (N,) boolean mask, True = temperature specified.
        """
        N = self.N
        T = np.array(temperatures, dtype=np.float64)
        q = np.array(heat_fluxes, dtype=np.float64)

        # For known-T surfaces: same as before
        # For known-q surfaces: q_i = eps_i/(1-eps_i) * (eb_i - B_i)
        # So: B_i = eb_i - q_i*(1-eps_i)/eps_i

        A_mat = np.eye(N) - np.diag(self.rho) @ self.F
        rhs = np.zeros(N)

        for i in range(N):
            if known_T[i]:
                rhs[i] = self.eps[i] * STEFAN_BOLTZMANN * T[i]**4
            else:
                # Known flux: must reformulate
                # q_i = B_i - sum_j F_ij B_j
                # This adds a constraint; replace the ith equation
                A_mat[i, :] = -self.F[i, :]
                A_mat[i, i] = 1.0
                rhs[i] = q[i]

        B = np.linalg.solve(A_mat, rhs)
        G = self.F @ B
        q_net = B - G

        # Recover unknown temperatures from B_i = eps_i * sigma*T_i^4 + rho_i * G_i
        for i in range(N):
            if not known_T[i]:
                eb_i = (B[i] - self.rho[i] * G[i]) / self.eps[i] if self.eps[i] > 0 else 0
                T[i] = (eb_i / STEFAN_BOLTZMANN) ** 0.25 if eb_i > 0 else 0.0

        return RadiosityResult(
            radiosities=B, irradiances=G, net_heat_fluxes=q_net, temperatures=T)


# ===================================================================
#  Discrete Ordinates Method (RTE for Participating Media)
# ===================================================================

class DiscreteOrdinatesRTE:
    r"""
    Discrete Ordinates Method (S_N) for the radiative transfer equation
    in absorbing-emitting-scattering participating media.

    RTE along direction $\hat{s}_m$:

    $$\hat{s}_m \cdot \nabla I_m = \kappa I_b - (\kappa + \sigma_s) I_m
        + \frac{\sigma_s}{4\pi}\sum_{m'} w_{m'} \Phi_{m'm} I_{m'}$$

    where $\kappa$ = absorption coefficient [1/m], $\sigma_s$ = scattering
    coefficient, $I_b = \sigma T^4 / \pi$ is blackbody intensity,
    $\Phi$ is the scattering phase function.

    1D slab geometry implementation for production use.
    """

    def __init__(self, n_ordinates: int = 8) -> None:
        """
        Parameters
        ----------
        n_ordinates : Number of discrete directions (S_N order, must be even).
        """
        if n_ordinates % 2 != 0:
            raise ValueError("n_ordinates must be even")
        self.N = n_ordinates
        self.mu, self.w = self._gauss_legendre(n_ordinates)

    @staticmethod
    def _gauss_legendre(n: int) -> Tuple[NDArray, NDArray]:
        """Gauss-Legendre quadrature points and weights on [-1, 1]."""
        mu, w = np.polynomial.legendre.leggauss(n)
        return mu, w

    def solve_slab(self,
                   z: NDArray[np.float64],
                   temperature: NDArray[np.float64],
                   kappa: NDArray[np.float64],
                   sigma_s: NDArray[np.float64],
                   T_wall_left: float,
                   T_wall_right: float,
                   scattering_phase: Optional[NDArray] = None,
                   max_iter: int = 500,
                   tol: float = 1e-6) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Solve 1D slab RTE using discrete ordinates.

        Parameters
        ----------
        z : (Nz,) spatial grid [m].
        temperature : (Nz,) local temperature [K].
        kappa : (Nz,) absorption coefficient [1/m].
        sigma_s : (Nz,) scattering coefficient [1/m].
        T_wall_left : Left boundary temperature [K].
        T_wall_right : Right boundary temperature [K].
        scattering_phase : (N, N) phase function weights Φ_{m'm}. Default: isotropic.
        max_iter : Maximum iterations.
        tol : Convergence tolerance on incident radiation.

        Returns
        -------
        (G, q) where G = incident radiation [W/m²] and q = radiative heat flux [W/m²].
        """
        Nz = len(z)
        N = self.N
        mu = self.mu
        w = self.w

        # Blackbody intensity
        Ib = STEFAN_BOLTZMANN * temperature**4 / PI

        # Phase function (isotropic default)
        if scattering_phase is None:
            Phi = np.ones((N, N)) / (4.0 * PI)
        else:
            Phi = scattering_phase

        # Intensity: I[direction, spatial]
        I = np.zeros((N, Nz))

        # Boundary conditions
        Ib_left = STEFAN_BOLTZMANN * T_wall_left**4 / PI
        Ib_right = STEFAN_BOLTZMANN * T_wall_right**4 / PI

        beta = kappa + sigma_s  # extinction coefficient

        for iteration in range(max_iter):
            I_old = I.copy()

            # Scattering source
            S_scat = np.zeros((N, Nz))
            for m in range(N):
                for mp in range(N):
                    S_scat[m, :] += w[mp] * Phi[mp, m] * I[mp, :]
                S_scat[m, :] *= sigma_s

            for m in range(N):
                S = kappa * Ib + S_scat[m, :]

                if mu[m] > 0:
                    # Forward sweep (left to right)
                    I[m, 0] = Ib_left  # diffuse BC
                    for j in range(1, Nz):
                        dz = z[j] - z[j - 1]
                        tau = beta[j] * dz / mu[m]
                        if tau < 1e-10:
                            I[m, j] = I[m, j - 1]
                        else:
                            I[m, j] = (I[m, j - 1] * math.exp(-tau)
                                       + S[j] / beta[j] * (1.0 - math.exp(-tau)))
                else:
                    # Backward sweep (right to left)
                    I[m, Nz - 1] = Ib_right
                    for j in range(Nz - 2, -1, -1):
                        dz = z[j + 1] - z[j]
                        tau = beta[j] * dz / abs(mu[m])
                        if tau < 1e-10:
                            I[m, j] = I[m, j + 1]
                        else:
                            I[m, j] = (I[m, j + 1] * math.exp(-tau)
                                       + S[j] / beta[j] * (1.0 - math.exp(-tau)))

            # Convergence check on incident radiation G = Σ w_m I_m
            G = np.sum(w[:, None] * I, axis=0)
            G_old = np.sum(w[:, None] * I_old, axis=0)
            if np.max(np.abs(G - G_old)) < tol * np.max(np.abs(G) + 1e-30):
                break

        # Incident radiation
        G = np.sum(w[:, None] * I, axis=0)

        # Radiative heat flux: q = Σ w_m μ_m I_m
        q = np.sum((w * mu)[:, None] * I, axis=0)

        return G, q

    def divergence_of_flux(self,
                           z: NDArray[np.float64],
                           G: NDArray[np.float64],
                           temperature: NDArray[np.float64],
                           kappa: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute ∇·q_rad = κ(4σT⁴ - G) for energy equation coupling.
        """
        return kappa * (4.0 * STEFAN_BOLTZMANN * temperature**4 - G)


# ===================================================================
#  Stefan Solidification (Enthalpy Method)
# ===================================================================

class StefanSolidification:
    r"""
    Stefan problem solver for solidification/melting with latent heat.

    Enthalpy method — avoids explicit front tracking:

    $$\frac{\partial H}{\partial t} = \nabla\cdot(k\nabla T)$$

    where $H(T) = \rho c_p T + \rho L f_l(T)$ with liquid fraction:

    $$f_l(T) = \begin{cases} 0 & T < T_s \\
        (T - T_s)/(T_l - T_s) & T_s \leq T \leq T_l \\
        1 & T > T_l \end{cases}$$

    Supports mushy zone with solidus $T_s$ and liquidus $T_l$ temperatures.
    """

    def __init__(self,
                 rho: float,
                 cp_solid: float,
                 cp_liquid: float,
                 k_solid: float,
                 k_liquid: float,
                 latent_heat: float,
                 T_solidus: float,
                 T_liquidus: float) -> None:
        """
        Parameters
        ----------
        rho : Density [kg/m³].
        cp_solid : Specific heat, solid [J/(kg·K)].
        cp_liquid : Specific heat, liquid [J/(kg·K)].
        k_solid : Thermal conductivity, solid [W/(m·K)].
        k_liquid : Thermal conductivity, liquid [W/(m·K)].
        latent_heat : Latent heat of fusion [J/kg].
        T_solidus : Solidus temperature [K].
        T_liquidus : Liquidus temperature [K].
        """
        self.rho = rho
        self.cp_s = cp_solid
        self.cp_l = cp_liquid
        self.k_s = k_solid
        self.k_l = k_liquid
        self.L = latent_heat
        self.T_s = T_solidus
        self.T_l = T_liquidus

    def liquid_fraction(self, T: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute liquid fraction from temperature field."""
        fl = np.zeros_like(T)
        mushy = (T >= self.T_s) & (T <= self.T_l)
        liquid = T > self.T_l
        if self.T_l > self.T_s:
            fl[mushy] = (T[mushy] - self.T_s) / (self.T_l - self.T_s)
        fl[liquid] = 1.0
        return fl

    def effective_conductivity(self, fl: NDArray[np.float64]) -> NDArray[np.float64]:
        """Linear interpolation of conductivity in mushy zone."""
        return self.k_s * (1.0 - fl) + self.k_l * fl

    def enthalpy(self, T: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute enthalpy H(T) [J/m³]."""
        fl = self.liquid_fraction(T)
        cp_eff = self.cp_s * (1.0 - fl) + self.cp_l * fl
        return self.rho * cp_eff * T + self.rho * self.L * fl

    def _T_from_H(self, H: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Invert enthalpy to get temperature.
        Three regimes: solid, mushy, liquid.
        """
        T = np.zeros_like(H)
        # Enthalpy bounds
        H_s = self.rho * self.cp_s * self.T_s
        H_l = self.rho * self.cp_l * self.T_l + self.rho * self.L

        solid = H <= H_s
        liquid = H >= H_l
        mushy = ~solid & ~liquid

        # Solid: H = ρ cp_s T → T = H/(ρ cp_s)
        T[solid] = H[solid] / (self.rho * self.cp_s)

        # Liquid: H = ρ cp_l T + ρ L → T = (H - ρL)/(ρ cp_l)
        T[liquid] = (H[liquid] - self.rho * self.L) / (self.rho * self.cp_l)

        # Mushy: H = ρ cp_eff T + ρ L fl, where fl = (T - Ts)/(Tl - Ts)
        # Solve the linear system in T
        if np.any(mushy):
            dT = self.T_l - self.T_s if self.T_l > self.T_s else 1.0
            cp_eff = 0.5 * (self.cp_s + self.cp_l)
            # H = ρ cp_eff T + ρ L (T - Ts)/dT
            # H = T(ρ cp_eff + ρ L/dT) - ρ L Ts/dT
            coeff = self.rho * cp_eff + self.rho * self.L / dT
            offset = self.rho * self.L * self.T_s / dT
            T[mushy] = (H[mushy] + offset) / coeff

        return T

    def solve_1d(self,
                 x: NDArray[np.float64],
                 T_init: NDArray[np.float64],
                 T_left: float,
                 T_right: float,
                 dt: float,
                 n_steps: int) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Solve 1D Stefan problem with enthalpy method.

        Parameters
        ----------
        x : (Nx,) spatial grid [m].
        T_init : (Nx,) initial temperature [K].
        T_left : Left boundary T [K].
        T_right : Right boundary T [K].
        dt : Time step [s].
        n_steps : Number of time steps.

        Returns
        -------
        (T_final, fl_final, front_history)
        T_final : (Nx,) final temperature.
        fl_final : (Nx,) final liquid fraction.
        front_history : (n_steps,) solidification front position [m].
        """
        Nx = len(x)
        dx = x[1] - x[0]

        T = T_init.copy()
        H = self.enthalpy(T)

        front_history = np.zeros(n_steps)

        for step in range(n_steps):
            fl = self.liquid_fraction(T)
            k = self.effective_conductivity(fl)

            # Interface conductivity (harmonic mean)
            k_face = np.zeros(Nx - 1)
            for i in range(Nx - 1):
                if k[i] + k[i + 1] > 0:
                    k_face[i] = 2.0 * k[i] * k[i + 1] / (k[i] + k[i + 1])

            # Enthalpy update: dH/dt = d/dx(k dT/dx)
            dHdt = np.zeros(Nx)
            for i in range(1, Nx - 1):
                flux_right = k_face[i] * (T[i + 1] - T[i]) / dx
                flux_left = k_face[i - 1] * (T[i] - T[i - 1]) / dx
                dHdt[i] = (flux_right - flux_left) / dx

            H[1:-1] += dt * dHdt[1:-1]

            # Boundary conditions (Dirichlet)
            H[0] = self.enthalpy(np.array([T_left]))[0]
            H[-1] = self.enthalpy(np.array([T_right]))[0]

            # Invert to get temperature
            T = self._T_from_H(H)
            # Clamp BCs
            T[0] = T_left
            T[-1] = T_right

            # Track solidification front (fl = 0.5 isoline)
            fl = self.liquid_fraction(T)
            crossings = np.where(np.diff(np.sign(fl - 0.5)))[0]
            if len(crossings) > 0:
                j = crossings[0]
                if abs(fl[j + 1] - fl[j]) > 1e-12:
                    frac = (0.5 - fl[j]) / (fl[j + 1] - fl[j])
                    front_history[step] = x[j] + frac * dx
                else:
                    front_history[step] = x[j]
            else:
                front_history[step] = front_history[step - 1] if step > 0 else 0.0

        fl_final = self.liquid_fraction(T)
        return T, fl_final, front_history

    @staticmethod
    def neumann_solution(alpha_s: float, alpha_l: float,
                         T_wall: float, T_init: float, T_m: float,
                         k_s: float, k_l: float,
                         L: float, rho: float,
                         t: float) -> Tuple[float, float]:
        """
        Analytical Neumann solution for solidification front position.

        Returns (front_position, lambda_parameter) where x_f = 2λ√(α_s·t).

        Parameters
        ----------
        alpha_s : Solid thermal diffusivity [m²/s].
        alpha_l : Liquid thermal diffusivity  [m²/s].
        T_wall : Wall temperature [K].
        T_init : Initial liquid temperature [K].
        T_m : Melting temperature [K].
        k_s, k_l : Thermal conductivities [W/(m·K)].
        L : Latent heat [J/kg].
        rho : Density [kg/m³].
        t : Time [s].
        """
        from scipy.special import erf  # type: ignore

        # Transcendental equation for λ:
        # St_s exp(-λ²)/(erf(λ)) - St_l √(α_s/α_l) exp(-λ²α_s/α_l)/(1-erf(λ√(α_s/α_l)))
        # = λ√π
        St_s = (T_m - T_wall) * k_s / (L * math.sqrt(alpha_s))
        St_l = (T_init - T_m) * k_l / (L * math.sqrt(alpha_l))
        ratio = math.sqrt(alpha_s / alpha_l)

        # Newton's method to find λ
        lam = 0.5
        for _ in range(100):
            e1 = math.exp(-lam**2)
            e2 = math.exp(-lam**2 * alpha_s / alpha_l)
            erf1 = float(erf(lam))
            erf2 = float(erf(lam * ratio))

            f = (St_s * e1 / (erf1 + 1e-30)
                 - St_l * ratio * e2 / (1.0 - erf2 + 1e-30)
                 - lam * math.sqrt(PI))

            # Numerical derivative
            dl = 1e-6
            lam_p = lam + dl
            e1p = math.exp(-lam_p**2)
            e2p = math.exp(-lam_p**2 * alpha_s / alpha_l)
            erf1p = float(erf(lam_p))
            erf2p = float(erf(lam_p * ratio))
            fp = (St_s * e1p / (erf1p + 1e-30)
                  - St_l * ratio * e2p / (1.0 - erf2p + 1e-30)
                  - lam_p * math.sqrt(PI))
            df = (fp - f) / dl

            if abs(df) < 1e-15:
                break
            lam -= f / df

        x_f = 2.0 * lam * math.sqrt(alpha_s * t)
        return x_f, lam


# ===================================================================
#  Conjugate Heat Transfer (CHT)
# ===================================================================

class ConjugateCHT:
    r"""
    Conjugate conduction-convection-radiation coupling.

    Solves coupled energy equations in solid and fluid domains:

    Solid: $\rho_s c_s \frac{\partial T}{\partial t} = \nabla\cdot(k_s\nabla T) + \dot{q}''_{\text{rad}}$

    Fluid: $\rho_f c_f\left(\frac{\partial T}{\partial t} + \mathbf{u}\cdot\nabla T\right)
           = \nabla\cdot(k_f\nabla T) + \dot{q}''_{\text{rad}}$

    Interface: $k_s\frac{\partial T_s}{\partial n} = k_f\frac{\partial T_f}{\partial n}$
    with $T_s = T_f$ at the interface.

    1D conjugate solver (solid-fluid stack) with optional radiation source.
    """

    @dataclass
    class Region:
        """One solid or fluid region."""
        x_start: float          # Region start [m]
        x_end: float            # Region end [m]
        rho: float              # Density [kg/m³]
        cp: float               # Specific heat [J/(kg·K)]
        k: float                # Conductivity [W/(m·K)]
        is_fluid: bool = False
        velocity: float = 0.0   # 1D velocity [m/s] (for fluid)
        n_cells: int = 50       # Grid cells in this region

    def __init__(self, regions: List["ConjugateCHT.Region"],
                 radiation_source: Optional[Callable[[NDArray, NDArray], NDArray]] = None) -> None:
        """
        Parameters
        ----------
        regions : Ordered list of solid/fluid regions.
        radiation_source : Callable(x, T) → q_rad [W/m³] volumetric source.
        """
        self.regions = regions
        self.rad_source = radiation_source

        # Build composite grid
        x_parts = []
        region_map = []
        for idx, reg in enumerate(regions):
            xr = np.linspace(reg.x_start, reg.x_end, reg.n_cells + 1)
            if idx > 0:
                xr = xr[1:]  # avoid duplicate interface point
            x_parts.append(xr)
            region_map.extend([idx] * len(xr))

        self.x = np.concatenate(x_parts)
        self.region_map = np.array(region_map[:len(self.x)])
        self.Nx = len(self.x)

    def _get_properties(self, idx: int) -> Tuple[float, float, float, float]:
        """Return (rho, cp, k, vel) for grid point idx."""
        reg = self.regions[self.region_map[idx]]
        return reg.rho, reg.cp, reg.k, reg.velocity if reg.is_fluid else 0.0

    def solve(self,
              T_init: NDArray[np.float64],
              T_left: float,
              T_right: float,
              dt: float,
              n_steps: int) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Solve time-dependent conjugate heat transfer.

        Parameters
        ----------
        T_init : (Nx,) initial temperature [K].
        T_left : Left boundary temperature [K].
        T_right : Right boundary temperature [K].
        dt : Time step [s].
        n_steps : Number of time steps.

        Returns
        -------
        (x, T_final) spatial grid and final temperature distribution.
        """
        T = T_init.copy()
        x = self.x

        for step in range(n_steps):
            T_new = T.copy()

            for i in range(1, self.Nx - 1):
                rho, cp, k, vel = self._get_properties(i)
                _, _, k_l, _ = self._get_properties(i - 1)
                _, _, k_r, _ = self._get_properties(i + 1)

                dx_l = x[i] - x[i - 1]
                dx_r = x[i + 1] - x[i]
                dx = 0.5 * (dx_l + dx_r)

                # Interface conductivity (harmonic mean)
                k_face_l = 2.0 * k * k_l / (k + k_l) if (k + k_l) > 0 else 0.0
                k_face_r = 2.0 * k * k_r / (k + k_r) if (k + k_r) > 0 else 0.0

                # Diffusion
                cond = (k_face_r * (T[i + 1] - T[i]) / dx_r
                        - k_face_l * (T[i] - T[i - 1]) / dx_l) / dx

                # Advection (upwind for fluid)
                adv = 0.0
                if abs(vel) > 1e-15:
                    if vel > 0:
                        adv = vel * (T[i] - T[i - 1]) / dx_l
                    else:
                        adv = vel * (T[i + 1] - T[i]) / dx_r

                # Radiation source
                q_rad = 0.0
                if self.rad_source is not None:
                    q_rad = self.rad_source(
                        np.array([x[i]]), np.array([T[i]]))[0]

                T_new[i] = T[i] + dt / (rho * cp) * (cond - rho * cp * adv + q_rad)

            # BCs
            T_new[0] = T_left
            T_new[-1] = T_right
            T = T_new

        return self.x.copy(), T

    def steady_state(self,
                     T_left: float,
                     T_right: float,
                     T_guess: Optional[NDArray] = None,
                     max_iter: int = 10000,
                     tol: float = 1e-8) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Solve for steady-state temperature distribution.

        Uses pseudo-transient continuation: time-march with large dt until convergence.
        """
        if T_guess is None:
            T = np.linspace(T_left, T_right, self.Nx)
        else:
            T = T_guess.copy()

        # Estimate a safe large dt
        dx_min = np.min(np.diff(self.x))
        k_max = max(r.k for r in self.regions)
        rho_min = min(r.rho for r in self.regions)
        cp_min = min(r.cp for r in self.regions)
        alpha_max = k_max / (rho_min * cp_min)
        dt = 0.3 * dx_min**2 / alpha_max  # Stability limit (explicit Euler)

        for iteration in range(max_iter):
            T_old = T.copy()
            _, T = self.solve(T, T_left, T_right, dt, n_steps=1)
            residual = float(np.max(np.abs(T - T_old)))
            if residual < tol:
                break

        return self.x.copy(), T

    def interface_heat_flux(self, T: NDArray[np.float64]) -> List[Tuple[float, float]]:
        """
        Compute heat flux at each region interface.

        Returns list of (x_interface, q [W/m²]) for each internal interface.
        """
        fluxes = []
        x = self.x

        for i in range(1, self.Nx - 1):
            if self.region_map[i] != self.region_map[i - 1]:
                # Interface between region_map[i-1] and region_map[i]
                k_left = self.regions[self.region_map[i - 1]].k
                dx = x[i] - x[i - 1]
                q = -k_left * (T[i] - T[i - 1]) / dx
                fluxes.append((x[i], q))

        return fluxes
