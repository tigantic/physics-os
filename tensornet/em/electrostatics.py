"""
Electrostatics solvers — Poisson-Boltzmann, multipole expansion, capacitance.

Upgrades domain III.1 from generic Poisson (tensornet/cfd/tt_poisson.py) to
dedicated electrostatics with:
  - Nonlinear Poisson-Boltzmann (Picard iteration + linearised PB)
  - Debye-Hückel limiting case
  - Multipole expansion (Cartesian and spherical, arbitrary order)
  - Capacitance extraction from solved potential
  - Poisson-Nernst-Planck ion transport

Physical constants in SI throughout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------
EPSILON_0: float = 8.854187817e-12          # F/m  vacuum permittivity
Q_E: float = 1.602176634e-19               # C    elementary charge
K_B: float = 1.380649e-23                  # J/K  Boltzmann constant
N_A: float = 6.02214076e23                 # 1/mol Avogadro
BOHR_RADIUS: float = 5.29177210903e-11     # m

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

class BoundaryType(Enum):
    """Boundary condition type for electrostatics."""
    DIRICHLET = auto()
    NEUMANN = auto()
    PERIODIC = auto()


@dataclass
class ChargeDistribution:
    """
    Represents a collection of point charges or a continuous charge density.

    Attributes
    ----------
    positions : (N, dim) array of charge positions [m].
    charges : (N,) array of charge values [C].
    density_func : Optional callable ρ(x, y, z) → C/m³ for continuous distributions.
    """
    positions: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 3)))
    charges: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    density_func: Optional[Callable[..., float]] = None

    @staticmethod
    def point_charge(q: float, pos: Sequence[float]) -> "ChargeDistribution":
        """Single point charge."""
        return ChargeDistribution(
            positions=np.array([pos], dtype=np.float64),
            charges=np.array([q], dtype=np.float64),
        )

    @staticmethod
    def dipole(q: float, pos_plus: Sequence[float],
               pos_minus: Sequence[float]) -> "ChargeDistribution":
        """Electric dipole (+q at pos_plus, -q at pos_minus)."""
        return ChargeDistribution(
            positions=np.array([pos_plus, pos_minus], dtype=np.float64),
            charges=np.array([q, -q], dtype=np.float64),
        )

    @staticmethod
    def uniform_line(linear_density: float, start: Sequence[float],
                     end: Sequence[float], n_segments: int = 100) -> "ChargeDistribution":
        """Discretised uniform line charge λ [C/m]."""
        s = np.array(start, dtype=np.float64)
        e = np.array(end, dtype=np.float64)
        length = np.linalg.norm(e - s)
        dq = linear_density * length / n_segments
        t = np.linspace(0, 1, n_segments, endpoint=False) + 0.5 / n_segments
        positions = s[None, :] + t[:, None] * (e - s)[None, :]
        charges = np.full(n_segments, dq)
        return ChargeDistribution(positions=positions, charges=charges)

    @property
    def total_charge(self) -> float:
        return float(np.sum(self.charges))

    @property
    def dipole_moment(self) -> NDArray[np.float64]:
        """Electric dipole moment p = Σ q_i r_i [C·m]."""
        return np.einsum("i,ij->j", self.charges, self.positions)


@dataclass
class ElectrostaticResult:
    """Result container for electrostatics solvers."""
    potential: NDArray[np.float64]          # φ on grid [V]
    electric_field: Optional[NDArray[np.float64]] = None  # E = -∇φ [V/m]
    energy: Optional[float] = None          # Electrostatic energy [J]
    capacitance: Optional[float] = None     # Extracted capacitance [F]
    iterations: int = 0
    residual: float = 0.0


# ---------------------------------------------------------------------------
# Multipole Expansion
# ---------------------------------------------------------------------------

class MultipoleExpansion:
    r"""
    Cartesian and spherical multipole expansion up to arbitrary order L.

    Spherical expansion of potential:
    $$
    \phi(\mathbf{r}) = \frac{1}{4\pi\varepsilon_0} \sum_{l=0}^{L}
        \sum_{m=-l}^{l} \frac{q_{lm}}{r^{l+1}} Y_l^m(\theta,\varphi)
    $$

    Cartesian monopole + dipole + quadrupole:
    $$
    \phi \approx \frac{1}{4\pi\varepsilon_0}\left[
        \frac{Q}{r} + \frac{\mathbf{p}\cdot\hat{r}}{r^2}
        + \frac{1}{2}\sum_{ij}\frac{Q_{ij}\hat{r}_i\hat{r}_j}{r^3}
    \right]
    $$
    """

    def __init__(self, charges: ChargeDistribution,
                 origin: Optional[Sequence[float]] = None,
                 max_order: int = 4,
                 epsilon_r: float = 1.0) -> None:
        self.charges = charges
        self.origin = (np.array(origin, dtype=np.float64) if origin is not None
                       else np.mean(charges.positions, axis=0))
        self.max_order = max_order
        self.eps = EPSILON_0 * epsilon_r

        # Shifted positions
        self._dr = charges.positions - self.origin[None, :]

        # Precompute moments
        self.monopole = charges.total_charge
        self.dipole_vec = charges.dipole_moment - self.monopole * self.origin
        self.quadrupole = self._compute_quadrupole()
        self._spherical_moments = self._compute_spherical_moments()

    # -- Cartesian moments --------------------------------------------------

    def _compute_quadrupole(self) -> NDArray[np.float64]:
        r"""
        Traceless quadrupole tensor:
        $$Q_{ij} = \sum_k q_k (3 r_{k,i} r_{k,j} - |\mathbf{r}_k|^2 \delta_{ij})$$
        """
        q = self.charges.charges
        dr = self._dr
        r2 = np.sum(dr ** 2, axis=1)
        Q = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                Q[i, j] = np.sum(q * (3.0 * dr[:, i] * dr[:, j]
                                       - (r2 if i == j else 0.0)))
        return Q

    # -- Spherical moments --------------------------------------------------

    @staticmethod
    def _real_solid_harmonic_regular(l: int, m: int,
                                     x: NDArray, y: NDArray,
                                     z: NDArray) -> NDArray:
        """Real regular solid harmonic R_l^m(x,y,z) = r^l C_l^m."""
        r2 = x**2 + y**2 + z**2
        r = np.sqrt(r2 + 1e-300)
        ct = z / r
        st = np.sqrt(np.maximum(1.0 - ct**2, 0.0))
        phi = np.arctan2(y, x)

        # Associated Legendre via recursion (Schmidt semi-normalized)
        plm = _associated_legendre(l, abs(m), ct)
        if m > 0:
            return (r ** l) * plm * np.cos(m * phi)
        elif m < 0:
            return (r ** l) * plm * np.sin(abs(m) * phi)
        else:
            return (r ** l) * plm

    def _compute_spherical_moments(self) -> NDArray[np.complex128]:
        """Compute q_lm = Σ_k q_k r_k^l Y_l^m*(θ_k, φ_k)."""
        L = self.max_order
        q = self.charges.charges
        dr = self._dr
        x, y, z = dr[:, 0], dr[:, 1], dr[:, 2]

        # Store as (L+1)² flat array
        moments = np.zeros((L + 1) ** 2, dtype=np.complex128)
        idx = 0
        for l in range(L + 1):
            for m in range(-l, l + 1):
                Rlm = self._real_solid_harmonic_regular(l, m, x, y, z)
                moments[idx] = np.sum(q * Rlm)
                idx += 1
        return moments

    # -- Potential evaluation -----------------------------------------------

    def potential_cartesian(self, r: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluate potential at field points using Cartesian multipole expansion.

        Parameters
        ----------
        r : (M, 3) array of field points [m].

        Returns
        -------
        phi : (M,) potential [V].
        """
        dr = r - self.origin[None, :]
        dist = np.linalg.norm(dr, axis=1, keepdims=True)
        dist = np.maximum(dist, 1e-30)
        rhat = dr / dist
        dist = dist[:, 0]

        prefactor = 1.0 / (4.0 * np.pi * self.eps)

        # Monopole
        phi = prefactor * self.monopole / dist

        # Dipole
        phi += prefactor * np.dot(rhat, self.dipole_vec) / dist**2

        # Quadrupole
        quad_contrib = np.einsum("mi,ij,mj->m", rhat, self.quadrupole, rhat)
        phi += prefactor * quad_contrib / (2.0 * dist**3)

        return phi

    def potential_spherical(self, r: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluate potential using spherical multipole expansion to order L.

        Parameters
        ----------
        r : (M, 3) array of field points [m].

        Returns
        -------
        phi : (M,) potential [V].
        """
        dr = r - self.origin[None, :]
        x, y, z = dr[:, 0], dr[:, 1], dr[:, 2]
        dist = np.sqrt(x**2 + y**2 + z**2 + 1e-300)

        prefactor = 1.0 / (4.0 * np.pi * self.eps)
        phi = np.zeros(len(r))

        idx = 0
        for l in range(self.max_order + 1):
            norm_factor = 1.0 / (2 * l + 1)
            for m in range(-l, l + 1):
                Rlm = self._real_solid_harmonic_regular(l, m, x, y, z)
                Ilm = Rlm / (dist ** (2 * l + 1))
                phi += prefactor * norm_factor * np.real(
                    self._spherical_moments[idx]) * Ilm
                idx += 1
        return phi

    def field_at(self, r: NDArray[np.float64],
                 dx: float = 1e-8) -> NDArray[np.float64]:
        """E = -∇φ via central differences on the spherical expansion."""
        E = np.zeros_like(r)
        for dim in range(3):
            rp = r.copy()
            rm = r.copy()
            rp[:, dim] += dx
            rm[:, dim] -= dx
            E[:, dim] = -(self.potential_spherical(rp) -
                          self.potential_spherical(rm)) / (2.0 * dx)
        return E


def _associated_legendre(l: int, m: int, x: NDArray) -> NDArray:
    """
    Compute associated Legendre polynomial P_l^m(x) via upward recursion.

    Uses the standard (non-Condon-Shortley) convention with no (-1)^m phase.
    """
    if m > l:
        return np.zeros_like(x)
    # Start with P_m^m
    pmm = np.ones_like(x)
    if m > 0:
        somx2 = np.sqrt(np.maximum(1.0 - x * x, 0.0))
        fact = 1.0
        for i in range(1, m + 1):
            pmm *= fact * somx2
            fact += 2.0
    if l == m:
        return pmm

    # P_{m+1}^m
    pmm1 = x * (2 * m + 1) * pmm
    if l == m + 1:
        return pmm1

    # Upward recursion
    pll = np.zeros_like(x)
    for ll in range(m + 2, l + 1):
        pll = ((2 * ll - 1) * x * pmm1 - (ll + m - 1) * pmm) / (ll - m)
        pmm = pmm1
        pmm1 = pll
    return pll


# ---------------------------------------------------------------------------
# Poisson-Boltzmann Solver (1D/2D/3D finite difference)
# ---------------------------------------------------------------------------

class PoissonBoltzmannSolver:
    r"""
    Nonlinear Poisson-Boltzmann equation solver (finite difference).

    Full nonlinear PB:
    $$
    \nabla\cdot(\varepsilon\nabla\phi)
        = -\rho_f - \sum_i c_i^0 z_i e \exp\!\left(-\frac{z_i e\phi}{k_B T}\right)
    $$

    Linearised (Debye-Hückel) limit:
    $$
    \nabla^2\phi = \kappa^2 \phi, \quad
    \kappa^2 = \frac{2 c_0 z^2 e^2}{\varepsilon k_B T}
    $$

    Supports 1D (slab), 2D, 3D Cartesian grids.
    """

    def __init__(
        self,
        grid_shape: Tuple[int, ...],
        dx: float,
        epsilon_r: float = 78.5,
        temperature: float = 298.15,
        ion_concentrations: Optional[List[float]] = None,
        ion_valences: Optional[List[int]] = None,
    ) -> None:
        """
        Parameters
        ----------
        grid_shape : Grid dimensions (Nx,) or (Nx, Ny) or (Nx, Ny, Nz).
        dx : Grid spacing [m].
        epsilon_r : Relative permittivity (78.5 for water at 25°C).
        temperature : Temperature [K].
        ion_concentrations : Bulk ion concentrations [mol/m³] for each species.
        ion_valences : Integer valences (e.g. [+1, -1] for 1:1 salt).
        """
        self.shape = tuple(grid_shape)
        self.ndim = len(self.shape)
        self.dx = dx
        self.eps = EPSILON_0 * epsilon_r
        self.T = temperature
        self.beta = 1.0 / (K_B * temperature)

        # Default: symmetric 1:1 electrolyte at 100 mM
        if ion_concentrations is None:
            ion_concentrations = [100.0 * N_A, 100.0 * N_A]  # mol/m³ → ions/m³
        if ion_valences is None:
            ion_valences = [1, -1]

        self.c0 = np.array(ion_concentrations, dtype=np.float64)
        self.z = np.array(ion_valences, dtype=np.float64)

        # Debye length
        ionic_strength = 0.5 * np.sum(self.c0 * self.z ** 2)
        self.kappa2 = Q_E**2 * ionic_strength / (self.eps * K_B * self.T)
        self.debye_length = 1.0 / np.sqrt(self.kappa2) if self.kappa2 > 0 else np.inf

    def _laplacian(self, phi: NDArray) -> NDArray:
        """Discrete Laplacian via central differences (2nd order)."""
        lap = np.zeros_like(phi)
        h2 = self.dx ** 2
        for dim in range(self.ndim):
            lap += (np.roll(phi, 1, axis=dim)
                    + np.roll(phi, -1, axis=dim)
                    - 2.0 * phi) / h2
        return lap

    def _ionic_charge_density(self, phi: NDArray) -> NDArray:
        r"""
        Mobile ion charge density:
        $$\rho_{\text{ion}} = \sum_i c_i^0 z_i e \exp(-z_i e\phi / k_BT)$$
        """
        rho = np.zeros_like(phi)
        for ci, zi in zip(self.c0, self.z):
            exponent = np.clip(-zi * Q_E * phi * self.beta, -500.0, 500.0)
            rho += ci * zi * Q_E * np.exp(exponent)
        return rho

    def _ionic_charge_jacobian(self, phi: NDArray) -> NDArray:
        r"""
        Diagonal of ∂ρ_ion/∂φ for Newton linearisation:
        $$\frac{\partial\rho_{\text{ion}}}{\partial\phi}
            = -\beta\sum_i c_i^0 z_i^2 e^2 \exp(-z_i e\phi / k_BT)$$
        """
        jac = np.zeros_like(phi)
        for ci, zi in zip(self.c0, self.z):
            exponent = np.clip(-zi * Q_E * phi * self.beta, -500.0, 500.0)
            jac += -self.beta * ci * zi**2 * Q_E**2 * np.exp(exponent)
        return jac

    def solve(
        self,
        rho_fixed: NDArray[np.float64],
        phi_boundary: Optional[NDArray[np.float64]] = None,
        boundary_mask: Optional[NDArray[np.bool_]] = None,
        max_iter: int = 500,
        tol: float = 1e-8,
        omega: float = 0.6,
        linearised: bool = False,
    ) -> ElectrostaticResult:
        """
        Solve the (non)linear Poisson-Boltzmann equation.

        Parameters
        ----------
        rho_fixed : Fixed charge density on grid [C/m³].
        phi_boundary : Potential values at boundary nodes [V].
        boundary_mask : Boolean mask of Dirichlet boundary nodes.
        max_iter : Maximum Picard/Newton iterations.
        tol : Convergence tolerance (L∞ norm of update).
        omega : Under-relaxation factor (0.3–0.8 typical for nonlinear PB).
        linearised : If True, solve linearised (Debye-Hückel) equation.

        Returns
        -------
        ElectrostaticResult with potential, electric field, energy.
        """
        phi = np.zeros(self.shape, dtype=np.float64)
        if boundary_mask is None:
            boundary_mask = np.zeros(self.shape, dtype=bool)
        if phi_boundary is not None:
            phi[boundary_mask] = phi_boundary[boundary_mask]

        h2 = self.dx ** 2
        n_neighbours = 2 * self.ndim

        residual = np.inf
        for it in range(1, max_iter + 1):
            if linearised:
                # ∇²φ - κ²φ = -ρ_f/ε
                rhs = -rho_fixed / self.eps
                # Jacobi-style update
                lap = self._laplacian(phi)
                update = (lap - self.kappa2 * phi - rhs)
                phi_new = phi + omega * h2 * update / n_neighbours
            else:
                # Nonlinear Picard iteration
                rho_ion = self._ionic_charge_density(phi)
                rhs = -(rho_fixed + rho_ion) / self.eps
                lap = self._laplacian(phi)
                residual_field = lap - rhs
                jac_diag = self._ionic_charge_jacobian(phi) / self.eps
                denom = n_neighbours / h2 - jac_diag
                denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
                delta_phi = -residual_field / denom
                phi_new = phi + omega * delta_phi

            # Enforce boundary conditions
            phi_new[boundary_mask] = phi[boundary_mask]

            residual = float(np.max(np.abs(phi_new - phi)))
            phi = phi_new

            if residual < tol:
                break

        # Electric field: E = -∇φ
        E = self._gradient(phi)

        # Electrostatic energy: U = (ε/2) ∫ |E|² dV
        E_mag2 = sum(Ei**2 for Ei in E)
        dV = self.dx ** self.ndim
        energy = 0.5 * self.eps * float(np.sum(E_mag2)) * dV

        E_field = np.stack(E, axis=-1)

        return ElectrostaticResult(
            potential=phi,
            electric_field=E_field,
            energy=energy,
            iterations=it,
            residual=residual,
        )

    def _gradient(self, phi: NDArray) -> List[NDArray]:
        """Central-difference gradient."""
        grad = []
        for dim in range(self.ndim):
            g = (np.roll(phi, -1, axis=dim) - np.roll(phi, 1, axis=dim)) / (2 * self.dx)
            grad.append(g)
        return grad


# ---------------------------------------------------------------------------
# Debye-Hückel Solver (thin wrapper)
# ---------------------------------------------------------------------------

class DebyeHuckelSolver(PoissonBoltzmannSolver):
    r"""
    Linearised Poisson-Boltzmann (Debye-Hückel) solver.

    $$\nabla^2\phi = \kappa^2\phi - \rho_f/\varepsilon$$

    Exact solution for point charge in unbounded medium:
    $$\phi(r) = \frac{q}{4\pi\varepsilon r}\exp(-\kappa r)$$
    """

    def solve(self, rho_fixed: NDArray[np.float64], **kwargs) -> ElectrostaticResult:
        kwargs["linearised"] = True
        return super().solve(rho_fixed, **kwargs)

    @staticmethod
    def screened_potential(q: float, r: NDArray[np.float64],
                          kappa: float, epsilon_r: float = 78.5) -> NDArray[np.float64]:
        """
        Analytical Debye-Hückel screened Coulomb potential.

        Parameters
        ----------
        q : Charge [C].
        r : Distances from charge [m].
        kappa : Inverse Debye length [1/m].
        epsilon_r : Relative permittivity.
        """
        eps = EPSILON_0 * epsilon_r
        return q / (4.0 * np.pi * eps * r) * np.exp(-kappa * r)


# ---------------------------------------------------------------------------
# Capacitance Extractor
# ---------------------------------------------------------------------------

class CapacitanceExtractor:
    r"""
    Extract capacitance from electrostatic field solution.

    Parallel-plate:
    $$C = \varepsilon A / d$$

    General (from energy):
    $$C = 2U / V^2$$

    From Gauss's law (surface integral):
    $$Q = \oint \varepsilon \mathbf{E}\cdot d\mathbf{A} \implies C = Q/V$$
    """

    @staticmethod
    def from_energy(energy: float, voltage: float) -> float:
        """C = 2U / V² from total electrostatic energy."""
        if abs(voltage) < 1e-30:
            raise ValueError("Voltage must be nonzero for capacitance extraction")
        return 2.0 * energy / voltage**2

    @staticmethod
    def from_charge(charge: float, voltage: float) -> float:
        """C = Q / V from total induced charge."""
        if abs(voltage) < 1e-30:
            raise ValueError("Voltage must be nonzero for capacitance extraction")
        return charge / voltage

    @staticmethod
    def parallel_plate(area: float, separation: float,
                       epsilon_r: float = 1.0) -> float:
        """Analytical parallel-plate capacitance C = εA/d."""
        return EPSILON_0 * epsilon_r * area / separation

    @staticmethod
    def coaxial(length: float, r_inner: float, r_outer: float,
                epsilon_r: float = 1.0) -> float:
        """Coaxial capacitor C = 2πεL / ln(b/a)."""
        return 2.0 * np.pi * EPSILON_0 * epsilon_r * length / np.log(r_outer / r_inner)

    @staticmethod
    def spherical(r_inner: float, r_outer: float,
                  epsilon_r: float = 1.0) -> float:
        """Spherical capacitor C = 4πε·ab/(b-a)."""
        return (4.0 * np.pi * EPSILON_0 * epsilon_r *
                r_inner * r_outer / (r_outer - r_inner))

    @staticmethod
    def extract_from_field(
        E_field: NDArray[np.float64],
        dx: float,
        epsilon_r: float,
        conductor_mask: NDArray[np.bool_],
        voltage: float,
    ) -> float:
        """
        Extract capacitance via Gauss's law surface integral.

        Integrates ε E·n̂ over the conductor boundary surface to get Q,
        then C = Q / V.
        """
        ndim = E_field.ndim - 1
        eps = EPSILON_0 * epsilon_r
        total_charge = 0.0

        # Find boundary nodes of conductor (adjacent to non-conductor)
        for dim in range(ndim):
            # Forward face
            shifted = np.roll(conductor_mask, -1, axis=dim)
            face = conductor_mask & ~shifted
            area = dx ** (ndim - 1)
            total_charge += float(np.sum(
                E_field[..., dim][face])) * eps * area

            # Backward face
            shifted = np.roll(conductor_mask, 1, axis=dim)
            face = conductor_mask & ~shifted
            total_charge -= float(np.sum(
                E_field[..., dim][face])) * eps * area

        return abs(total_charge / voltage)


# ---------------------------------------------------------------------------
# Poisson-Nernst-Planck Ion Transport
# ---------------------------------------------------------------------------

@dataclass
class IonSpecies:
    """Ion species for PNP equations."""
    name: str
    valence: int
    diffusivity: float          # D [m²/s]
    bulk_concentration: float   # c_∞ [mol/m³]


class PoissonNernstPlanck:
    r"""
    Poisson-Nernst-Planck equations for electrolyte ion transport.

    Nernst-Planck flux:
    $$
    \mathbf{J}_i = -D_i\nabla c_i - \frac{D_i z_i e}{k_B T} c_i \nabla\phi
    $$

    Continuity:
    $$
    \frac{\partial c_i}{\partial t} = -\nabla\cdot\mathbf{J}_i
    $$

    Coupled Poisson:
    $$
    \nabla\cdot(\varepsilon\nabla\phi) = -\sum_i z_i e c_i - \rho_f
    $$

    Solved via operator-splitting: diffusion-migration step + Poisson step.
    """

    def __init__(
        self,
        grid_shape: Tuple[int, ...],
        dx: float,
        species: List[IonSpecies],
        epsilon_r: float = 78.5,
        temperature: float = 298.15,
    ) -> None:
        self.shape = tuple(grid_shape)
        self.ndim = len(self.shape)
        self.dx = dx
        self.species = species
        self.eps = EPSILON_0 * epsilon_r
        self.T = temperature
        self.beta = Q_E / (K_B * temperature)

        # Concentration fields: one per species
        self.concentrations: List[NDArray[np.float64]] = [
            np.full(self.shape, sp.bulk_concentration, dtype=np.float64)
            for sp in species
        ]
        self.potential = np.zeros(self.shape, dtype=np.float64)

    def _laplacian(self, f: NDArray) -> NDArray:
        lap = np.zeros_like(f)
        h2 = self.dx ** 2
        for dim in range(self.ndim):
            lap += (np.roll(f, 1, axis=dim) + np.roll(f, -1, axis=dim)
                    - 2.0 * f) / h2
        return lap

    def _gradient(self, f: NDArray) -> List[NDArray]:
        grads = []
        for dim in range(self.ndim):
            g = (np.roll(f, -1, axis=dim) - np.roll(f, 1, axis=dim)) / (2 * self.dx)
            grads.append(g)
        return grads

    def _divergence(self, flux_components: List[NDArray]) -> NDArray:
        div = np.zeros(self.shape, dtype=np.float64)
        for dim, fc in enumerate(flux_components):
            div += (np.roll(fc, -1, axis=dim) - np.roll(fc, 1, axis=dim)) / (2 * self.dx)
        return div

    def _solve_poisson(self, rho_fixed: NDArray, max_iter: int = 200,
                       tol: float = 1e-8) -> None:
        """Solve Poisson equation for current ion concentrations."""
        # Total charge density from ions
        rho_ion = np.zeros(self.shape, dtype=np.float64)
        for sp, c in zip(self.species, self.concentrations):
            rho_ion += sp.valence * Q_E * c

        rho_total = rho_fixed + rho_ion
        h2 = self.dx ** 2
        n_nb = 2 * self.ndim

        for _ in range(max_iter):
            lap = self._laplacian(self.potential)
            residual = lap + rho_total / self.eps
            update = h2 * residual / n_nb
            self.potential += 0.8 * update
            if float(np.max(np.abs(update))) < tol:
                break

    def step(self, dt: float, rho_fixed: Optional[NDArray] = None) -> None:
        """
        One time step of the PNP system via operator splitting.

        Parameters
        ----------
        dt : Time step [s].
        rho_fixed : External fixed charge density [C/m³].
        """
        if rho_fixed is None:
            rho_fixed = np.zeros(self.shape, dtype=np.float64)

        # 1. Solve Poisson for current concentrations
        self._solve_poisson(rho_fixed)

        # 2. Update concentrations (Nernst-Planck)
        grad_phi = self._gradient(self.potential)

        for idx, sp in enumerate(self.species):
            c = self.concentrations[idx]
            D = sp.diffusivity
            z = sp.valence

            # Diffusion flux: -D ∇c
            grad_c = self._gradient(c)
            flux_diff = [-D * gc for gc in grad_c]

            # Migration flux: -(D z e / kT) c ∇φ
            flux_mig = [-(D * z * self.beta) * c * gp for gp in grad_phi]

            # Total flux
            flux = [fd + fm for fd, fm in zip(flux_diff, flux_mig)]

            # ∂c/∂t = -∇·J
            dc_dt = -self._divergence(flux)

            # Forward Euler step (explicit)
            c_new = c + dt * dc_dt

            # Enforce positivity
            c_new = np.maximum(c_new, 0.0)

            self.concentrations[idx] = c_new

    def run(self, n_steps: int, dt: float,
            rho_fixed: Optional[NDArray] = None) -> List[NDArray]:
        """
        Run PNP simulation for n_steps.

        Returns list of concentration snapshots (one per species, last step).
        """
        if rho_fixed is None:
            rho_fixed = np.zeros(self.shape, dtype=np.float64)

        for _ in range(n_steps):
            self.step(dt, rho_fixed)

        return [c.copy() for c in self.concentrations]

    def current_density(self) -> NDArray[np.float64]:
        """
        Compute total ionic current density J = Σ z_i e J_i [A/m²].
        """
        grad_phi = self._gradient(self.potential)
        J = [np.zeros(self.shape, dtype=np.float64) for _ in range(self.ndim)]

        for sp, c in zip(self.species, self.concentrations):
            D = sp.diffusivity
            z = sp.valence
            grad_c = self._gradient(c)

            for dim in range(self.ndim):
                flux = -D * grad_c[dim] - D * z * self.beta * c * grad_phi[dim]
                J[dim] += z * Q_E * flux

        return np.stack(J, axis=-1)
