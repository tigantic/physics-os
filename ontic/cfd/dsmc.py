"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          R A R E F I E D   G A S   D Y N A M I C S  (DSMC)                ║
║                                                                            ║
║  Direct Simulation Monte Carlo with BGK relaxation.                        ║
║  Covers domain II.6 of the 140-domain taxonomy.                            ║
║                                                                            ║
║  Solvers:                                                                  ║
║    - DSMC with NTC (No Time Counter) collision kernel                      ║
║    - Variable Hard Sphere (VHS) cross-section                              ║
║    - BGK relaxation operator                                               ║
║    - Knudsen-regime switching logic (Kn → continuum/slip/transition/free)  ║
║    - Maxwellian sampling & diffuse/specular wall boundaries                ║
║                                                                            ║
║  Physical regimes:                                                         ║
║    Kn < 0.001:   Continuum (Navier-Stokes valid)                          ║
║    0.001 < Kn < 0.1: Slip flow (NS + slip BC)                             ║
║    0.1 < Kn < 10: Transition (DSMC required)                              ║
║    Kn > 10:       Free molecular flow                                      ║
║                                                                            ║
║  References:                                                               ║
║    [1] Bird, G.A. (1994). Molecular Gas Dynamics and DSMC.                ║
║    [2] Bird, G.A. (2013). The DSMC Method.                                ║
║    [3] Bhatnagar, Gross, Krook (1954). Phys. Rev. 94, 511.               ║
║    [4] Koura & Matsumoto (1991). Variable soft sphere model.              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

import torch
from torch import Tensor
import math


# ═══════════════════════════════════════════════════════════════════════════════
#  GAS SPECIES & KNUDSEN REGIME
# ═══════════════════════════════════════════════════════════════════════════════

class KnudsenRegime(Enum):
    """Flow regime based on Knudsen number Kn = λ/L."""
    CONTINUUM = "continuum"         # Kn < 0.001
    SLIP = "slip"                   # 0.001 ≤ Kn < 0.1
    TRANSITION = "transition"       # 0.1 ≤ Kn < 10
    FREE_MOLECULAR = "free_molecular"  # Kn ≥ 10


def classify_knudsen(Kn: float) -> KnudsenRegime:
    """Classify flow regime from Knudsen number."""
    if Kn < 0.001:
        return KnudsenRegime.CONTINUUM
    elif Kn < 0.1:
        return KnudsenRegime.SLIP
    elif Kn < 10.0:
        return KnudsenRegime.TRANSITION
    else:
        return KnudsenRegime.FREE_MOLECULAR


@dataclass
class GasSpecies:
    """
    Molecular gas species for DSMC.

    VHS cross-section model:
        σ = σ_ref (c_r / c_ref)^(2(1-ω))

    where ω is the viscosity-temperature exponent (mu ∝ T^ω).

    Attributes:
        name: Gas name
        mass: Molecular mass (kg)
        diam: Reference diameter (m) at T_ref
        omega: VHS viscosity exponent (0.5 = hard sphere, ~0.74 for N₂)
        T_ref: Reference temperature (K) for VHS parameters
        dof_internal: Internal degrees of freedom (0=mono, 2=diatomic rot)
        alpha: VHS alpha exponent (= 1 for simple VHS)
    """

    name: str = "N2"
    mass: float = 4.65e-26       # N₂ mass (kg)
    diam: float = 4.17e-10       # N₂ VHS diameter at 273 K (m)
    omega: float = 0.74          # VHS exponent for N₂
    T_ref: float = 273.0         # Reference temperature
    dof_internal: int = 2        # Rotational DOF (diatomic)
    alpha: float = 1.0           # VHS alpha

    @classmethod
    def argon(cls) -> "GasSpecies":
        return cls(name="Ar", mass=6.63e-26, diam=4.17e-10, omega=0.81,
                   T_ref=273.0, dof_internal=0)

    @classmethod
    def nitrogen(cls) -> "GasSpecies":
        return cls(name="N2", mass=4.65e-26, diam=4.17e-10, omega=0.74,
                   T_ref=273.0, dof_internal=2)

    @classmethod
    def oxygen(cls) -> "GasSpecies":
        return cls(name="O2", mass=5.31e-26, diam=4.07e-10, omega=0.77,
                   T_ref=273.0, dof_internal=2)

    @classmethod
    def helium(cls) -> "GasSpecies":
        return cls(name="He", mass=6.65e-27, diam=2.33e-10, omega=0.66,
                   T_ref=273.0, dof_internal=0)

    @property
    def gamma(self) -> float:
        """Specific heat ratio γ = (5 + dof_internal) / (3 + dof_internal)."""
        f = 3 + self.dof_internal
        return (f + 2.0) / f

    @property
    def k_boltzmann(self) -> float:
        return 1.380649e-23

    def mean_free_path(self, n_density: float, T: float) -> float:
        """
        Mean free path for VHS model.

        λ = 1 / (√2 π d² n)

        where d = d_ref (T/T_ref)^(ω - 0.5) for VHS.
        """
        d_eff = self.diam * (T / self.T_ref) ** (self.omega - 0.5)
        return 1.0 / (math.sqrt(2.0) * math.pi * d_eff ** 2 * n_density)

    def thermal_speed(self, T: float) -> float:
        """Most probable thermal speed c_mp = √(2kT/m)."""
        return math.sqrt(2.0 * self.k_boltzmann * T / self.mass)

    def mean_speed(self, T: float) -> float:
        """Mean molecular speed c̄ = √(8kT/(πm))."""
        return math.sqrt(8.0 * self.k_boltzmann * T / (math.pi * self.mass))

    def sound_speed(self, T: float) -> float:
        """Sound speed a = √(γkT/m)."""
        return math.sqrt(self.gamma * self.k_boltzmann * T / self.mass)

    def vhs_cross_section(self, c_r: float, T: float) -> float:
        """
        VHS collision cross-section.

        σ_T = π d_ref² (c_r_ref / c_r)^(2(ω-0.5))

        where c_r_ref corresponds to T_ref.
        """
        c_ref = self.thermal_speed(self.T_ref) * math.sqrt(2.0)
        if c_r < 1e-10:
            return math.pi * self.diam ** 2
        ratio = c_ref / c_r
        exponent = 2.0 * (self.omega - 0.5)
        return math.pi * self.diam ** 2 * ratio ** exponent


# ═══════════════════════════════════════════════════════════════════════════════
#  DSMC GRID & PARTICLES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DSMCParticles:
    """
    Particle ensemble for DSMC.

    Each row i represents a simulation particle with weight W_n
    (number of real molecules per simulation particle).

    Attributes:
        positions: [N, dim] particle positions
        velocities: [N, 3] particle velocities (always 3 components)
        species_ids: [N] integer species index
        weights: [N] statistical weights (F_num)
        internal_energy: [N] rotational/vibrational energy per particle
    """

    positions: Tensor
    velocities: Tensor
    species_ids: Tensor
    weights: Tensor
    internal_energy: Tensor

    @property
    def n_particles(self) -> int:
        return self.positions.shape[0]

    @property
    def dim(self) -> int:
        return self.positions.shape[1]


@dataclass
class DSMCCell:
    """
    Single cell in the DSMC spatial grid.

    Macroscopic quantities are sampled/accumulated here.
    """

    # Indices of particles currently in this cell
    particle_indices: List[int] = field(default_factory=list)

    # Sampled macroscopic quantities (accumulated over sample steps)
    n_samples: int = 0
    sum_n_density: float = 0.0
    sum_velocity: Optional[Tensor] = None
    sum_temperature: float = 0.0
    sum_pressure: float = 0.0

    # NTC collision remainder
    remainder: float = 0.0

    @property
    def avg_n_density(self) -> float:
        if self.n_samples == 0:
            return 0.0
        return self.sum_n_density / self.n_samples

    @property
    def avg_temperature(self) -> float:
        if self.n_samples == 0:
            return 0.0
        return self.sum_temperature / self.n_samples


# ═══════════════════════════════════════════════════════════════════════════════
#  MAXWELLIAN SAMPLING
# ═══════════════════════════════════════════════════════════════════════════════

def sample_maxwellian(
    n_particles: int,
    T: float,
    u_bulk: Tensor,
    species: GasSpecies,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """
    Sample velocities from a Maxwellian distribution.

    f(v) = (m / 2πkT)^(3/2) exp(-m|v - u|² / 2kT)

    Each component is Gaussian: v_i ~ N(u_i, kT/m).

    Args:
        n_particles: Number of particles to sample
        T: Temperature (K)
        u_bulk: [3] bulk velocity
        species: Gas species

    Returns:
        [n_particles, 3] velocity samples
    """
    std = math.sqrt(species.k_boltzmann * T / species.mass)
    v = torch.randn(n_particles, 3, dtype=torch.float64, device=device) * std
    v += u_bulk.unsqueeze(0)
    return v


def sample_maxwellian_flux(
    n_particles: int,
    T: float,
    u_bulk: Tensor,
    normal: Tensor,
    species: GasSpecies,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """
    Sample velocities from the Maxwellian flux distribution for inlet BC.

    The normal component follows a Rayleigh distribution (biased toward inflow):
        f(v_n) ∝ v_n exp(-m v_n² / 2kT)  for v_n > 0

    Tangential components are Gaussian.

    Args:
        n_particles: Number of particles
        T: Temperature
        u_bulk: [3] bulk velocity
        normal: [3] inward normal direction
        species: Gas species

    Returns:
        [n_particles, 3] velocity samples
    """
    std = math.sqrt(species.k_boltzmann * T / species.mass)

    # Build local frame: normal, tangent1, tangent2
    n = normal / (torch.norm(normal) + 1e-30)

    # Find tangent vectors
    if abs(n[0].item()) < 0.9:
        t1 = torch.cross(n, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64, device=device))
    else:
        t1 = torch.cross(n, torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64, device=device))
    t1 = t1 / (torch.norm(t1) + 1e-30)
    t2 = torch.cross(n, t1)

    # Normal component: Rayleigh distribution
    # v_n = std * sqrt(-2 ln(U))  where U ~ Uniform(0,1)
    U = torch.rand(n_particles, dtype=torch.float64, device=device)
    v_n = std * torch.sqrt(-2.0 * torch.log(U.clamp(min=1e-30)))

    # Tangential components: Gaussian
    v_t1 = torch.randn(n_particles, dtype=torch.float64, device=device) * std
    v_t2 = torch.randn(n_particles, dtype=torch.float64, device=device) * std

    # Assemble in global frame
    v = (
        v_n.unsqueeze(-1) * n.unsqueeze(0)
        + v_t1.unsqueeze(-1) * t1.unsqueeze(0)
        + v_t2.unsqueeze(-1) * t2.unsqueeze(0)
    )

    # Add bulk velocity
    v += u_bulk.unsqueeze(0)

    return v


# ═══════════════════════════════════════════════════════════════════════════════
#  DSMC COLLISION KERNEL (NTC)
# ═══════════════════════════════════════════════════════════════════════════════

class NTCCollisionKernel:
    """
    No Time Counter (NTC) collision algorithm from Bird (1994).

    The number of candidate collision pairs in each cell per timestep:

        N_cand = ½ N(N-1) F_N (σ_T c_r)_max Δt / V_cell

    For each candidate pair, accept with probability:
        P_accept = (σ_T c_r) / (σ_T c_r)_max

    VHS scattering: isotropic in center-of-mass frame.

    Attributes:
        species: Gas species
        sigma_cr_max: Running maximum of σ_T c_r per cell
    """

    def __init__(self, species: GasSpecies) -> None:
        self.species = species
        self._sigma_cr_max: Dict[int, float] = {}  # cell_id → max(σ_T c_r)

    def perform_collisions(
        self,
        particles: DSMCParticles,
        cell: DSMCCell,
        cell_id: int,
        cell_volume: float,
        dt: float,
        F_num: float,
    ) -> int:
        """
        NTC collision algorithm for one cell.

        Args:
            particles: Global particle data
            cell: Cell with particle indices
            cell_id: Cell identifier
            cell_volume: Cell volume (m³ or m² for 2D)
            dt: Timestep (s)
            F_num: Statistical weight (real molecules per sim particle)

        Returns:
            Number of collisions performed
        """
        indices = cell.particle_indices
        N_p = len(indices)

        if N_p < 2:
            return 0

        # Get or initialize (σ_T c_r)_max
        sigma_cr_max = self._sigma_cr_max.get(cell_id, 0.0)
        if sigma_cr_max < 1e-30:
            # Initialize with thermal estimate
            T_est = 300.0  # Will be updated
            c_r_est = self.species.thermal_speed(T_est) * 2.0
            sigma_cr_max = self.species.vhs_cross_section(c_r_est, T_est) * c_r_est

        # Number of candidate pairs (with remainder for fractional collisions)
        N_cand_float = (
            0.5 * N_p * (N_p - 1) * F_num * sigma_cr_max * dt / cell_volume
            + cell.remainder
        )
        N_cand = int(N_cand_float)
        cell.remainder = N_cand_float - N_cand

        n_collisions = 0

        for _ in range(N_cand):
            # Select random pair
            i_local = torch.randint(0, N_p, (1,)).item()
            j_local = torch.randint(0, N_p - 1, (1,)).item()
            if j_local >= i_local:
                j_local += 1

            i_global = indices[i_local]
            j_global = indices[j_local]

            # Relative velocity
            v_i = particles.velocities[i_global]
            v_j = particles.velocities[j_global]
            c_r_vec = v_i - v_j
            c_r = torch.norm(c_r_vec).item()

            if c_r < 1e-30:
                continue

            # VHS cross-section
            sigma = self.species.vhs_cross_section(c_r, 300.0)
            sigma_cr = sigma * c_r

            # Update maximum
            if sigma_cr > sigma_cr_max:
                sigma_cr_max = sigma_cr

            # Acceptance probability
            P_accept = sigma_cr / sigma_cr_max if sigma_cr_max > 0 else 1.0
            if torch.rand(1).item() > P_accept:
                continue

            # Perform VHS collision (isotropic scattering in CoM frame)
            v_cm = 0.5 * (v_i + v_j)  # Equal mass assumed

            # Random post-collision relative velocity direction (uniform on sphere)
            cos_theta = 2.0 * torch.rand(1, dtype=torch.float64).item() - 1.0
            sin_theta = math.sqrt(max(1.0 - cos_theta ** 2, 0.0))
            phi = 2.0 * math.pi * torch.rand(1, dtype=torch.float64).item()

            c_r_new = torch.tensor([
                c_r * sin_theta * math.cos(phi),
                c_r * sin_theta * math.sin(phi),
                c_r * cos_theta,
            ], dtype=torch.float64, device=particles.velocities.device)

            # Post-collision velocities (equal mass)
            particles.velocities[i_global] = v_cm + 0.5 * c_r_new
            particles.velocities[j_global] = v_cm - 0.5 * c_r_new

            n_collisions += 1

        # Store updated maximum
        self._sigma_cr_max[cell_id] = sigma_cr_max

        return n_collisions


# ═══════════════════════════════════════════════════════════════════════════════
#  BGK RELAXATION OPERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class BGKRelaxation:
    """
    Bhatnagar-Gross-Krook (BGK) collision operator.

    The BGK model replaces the Boltzmann collision integral with a
    relaxation toward local Maxwellian:

        (∂f/∂t)_coll = ν (f_eq - f)

    where ν = collision frequency = n σ c̄.

    In DSMC particle representation, each particle is independently
    replaced with a Maxwellian sample with probability:
        P_relax = 1 - exp(-ν Δt)

    BGK is ~10-100× faster than DSMC for near-equilibrium flows.

    Attributes:
        species: Gas species
    """

    def __init__(self, species: GasSpecies) -> None:
        self.species = species

    def collision_frequency(self, n_density: float, T: float) -> float:
        """
        BGK collision frequency ν = n σ c̄.

        Uses VHS reference cross-section at temperature T.
        """
        c_mean = self.species.mean_speed(T)
        sigma = math.pi * self.species.diam ** 2 * (T / self.species.T_ref) ** (1.0 - self.species.omega)
        return n_density * sigma * c_mean

    def relax(
        self,
        particles: DSMCParticles,
        cell: DSMCCell,
        n_density: float,
        T_cell: float,
        u_cell: Tensor,
        dt: float,
    ) -> int:
        """
        Apply BGK relaxation to particles in a cell.

        Each particle is replaced with a Maxwellian sample
        with probability P = 1 - exp(-νΔt).

        Args:
            particles: Global particle data
            cell: Cell with particle indices
            n_density: Number density in cell
            T_cell: Cell temperature
            u_cell: [3] bulk velocity in cell
            dt: Timestep

        Returns:
            Number of particles relaxed
        """
        indices = cell.particle_indices
        if len(indices) == 0:
            return 0

        nu = self.collision_frequency(n_density, T_cell)
        P_relax = 1.0 - math.exp(-nu * dt)

        n_relaxed = 0
        for idx in indices:
            if torch.rand(1).item() < P_relax:
                # Replace velocity with Maxwellian sample
                new_v = sample_maxwellian(
                    1, T_cell, u_cell, self.species,
                    device=particles.velocities.device,
                )
                particles.velocities[idx] = new_v.squeeze(0)
                n_relaxed += 1

        return n_relaxed


# ═══════════════════════════════════════════════════════════════════════════════
#  WALL BOUNDARY CONDITIONS
# ═══════════════════════════════════════════════════════════════════════════════

class WallBoundaryType(Enum):
    SPECULAR = "specular"
    DIFFUSE = "diffuse"
    MAXWELL = "maxwell"  # Mixed specular + diffuse


@dataclass
class WallBoundary:
    """
    Wall boundary condition for DSMC.

    Specular: v_n → -v_n, v_t unchanged (elastic mirror).
    Diffuse: re-emit from wall Maxwellian at T_wall.
    Maxwell: fraction α_c diffuse + (1-α_c) specular.

    α_c is the accommodation coefficient:
        α_c = (E_i - E_r) / (E_i - E_w)

    Attributes:
        wall_type: Boundary type
        T_wall: Wall temperature (K)
        u_wall: [3] wall velocity
        accommodation: Accommodation coefficient for Maxwell model
        normal: [3] inward normal (pointing into domain)
        species: Gas species for Maxwellian sampling
    """

    wall_type: WallBoundaryType = WallBoundaryType.DIFFUSE
    T_wall: float = 300.0
    u_wall: Tensor = field(default_factory=lambda: torch.zeros(3, dtype=torch.float64))
    accommodation: float = 1.0  # Fully diffuse
    normal: Tensor = field(default_factory=lambda: torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64))
    species: GasSpecies = field(default_factory=GasSpecies)

    def reflect(self, velocity: Tensor) -> Tensor:
        """
        Apply wall reflection to a particle velocity.

        Args:
            velocity: [3] incoming velocity

        Returns:
            [3] outgoing velocity
        """
        n = self.normal / (torch.norm(self.normal) + 1e-30)

        if self.wall_type == WallBoundaryType.SPECULAR:
            # v_out = v_in - 2(v_in · n)n
            v_n = torch.dot(velocity - self.u_wall, n)
            return velocity - 2.0 * v_n * n

        elif self.wall_type == WallBoundaryType.DIFFUSE:
            return self._diffuse_reflect(n)

        else:
            # Maxwell: mix specular and diffuse
            if torch.rand(1).item() < self.accommodation:
                return self._diffuse_reflect(n)
            else:
                v_n = torch.dot(velocity - self.u_wall, n)
                return velocity - 2.0 * v_n * n

    def _diffuse_reflect(self, n: Tensor) -> Tensor:
        """
        Diffuse reflection: emit from wall Maxwellian.

        Normal component: Rayleigh distribution (toward domain).
        Tangential components: Gaussian at T_wall.
        """
        v = sample_maxwellian_flux(
            1, self.T_wall, self.u_wall, n, self.species,
            device=n.device,
        )
        return v.squeeze(0)


# ═══════════════════════════════════════════════════════════════════════════════
#  DSMC SOLVER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DSMCSolver:
    """
    Direct Simulation Monte Carlo solver.

    The DSMC algorithm splits each timestep into:
        1. Move particles (ballistic advection)
        2. Sort particles into cells
        3. Perform collisions (NTC or BGK)
        4. Sample macroscopic quantities

    Grid: uniform Cartesian (Δx should be < λ/3 for accuracy).
    Timestep: Δt < Δx / c_max (CFL-like stability).

    Attributes:
        species: Gas species
        domain_min: [dim] lower corner of computational domain
        domain_max: [dim] upper corner
        n_cells: [dim] number of cells per direction
        use_bgk: If True, use BGK instead of NTC collisions
        F_num: Statistical weight (real molecules per sim particle)
    """

    species: GasSpecies
    domain_min: Tensor
    domain_max: Tensor
    n_cells: Tuple[int, ...]
    use_bgk: bool = False
    F_num: float = 1.0e12

    def __post_init__(self):
        self.dim = len(self.n_cells)
        self.dx = (self.domain_max - self.domain_min) / torch.tensor(
            self.n_cells, dtype=torch.float64
        )
        self.total_cells = 1
        for nc in self.n_cells:
            self.total_cells *= nc

        self.cells: List[DSMCCell] = [DSMCCell() for _ in range(self.total_cells)]
        self.collision_kernel = NTCCollisionKernel(self.species)
        self.bgk = BGKRelaxation(self.species)

        # Wall boundaries: dict from face_id → WallBoundary
        self.walls: Dict[str, WallBoundary] = {}

    def cell_volume(self) -> float:
        """Volume (area in 2D) of a single cell."""
        vol = 1.0
        for d in range(self.dim):
            vol *= self.dx[d].item()
        return vol

    def _cell_index(self, position: Tensor) -> int:
        """Map a position to a flat cell index."""
        idx_vec = ((position - self.domain_min) / self.dx).long()
        for d in range(self.dim):
            idx_vec[d] = idx_vec[d].clamp(0, self.n_cells[d] - 1)

        flat = 0
        stride = 1
        for d in reversed(range(self.dim)):
            flat += idx_vec[d].item() * stride
            stride *= self.n_cells[d]
        return int(flat)

    def initialize_uniform(
        self,
        n_particles: int,
        T: float,
        u_bulk: Tensor,
        n_density: float,
    ) -> DSMCParticles:
        """
        Initialize particles uniformly in the domain.

        Args:
            n_particles: Total number of simulation particles
            T: Initial temperature (K)
            u_bulk: [3] initial bulk velocity
            n_density: Number density (1/m³ or 1/m²)

        Returns:
            Initialized DSMCParticles
        """
        device = self.domain_min.device

        # Uniform random positions
        positions = torch.rand(n_particles, self.dim, dtype=torch.float64, device=device)
        for d in range(self.dim):
            positions[:, d] = (
                self.domain_min[d]
                + positions[:, d] * (self.domain_max[d] - self.domain_min[d])
            )

        # Maxwellian velocities
        velocities = sample_maxwellian(n_particles, T, u_bulk, self.species, device)

        # All same species
        species_ids = torch.zeros(n_particles, dtype=torch.long, device=device)

        # Compute F_num from density
        domain_vol = self.cell_volume() * self.total_cells
        self.F_num = n_density * domain_vol / n_particles

        weights = torch.full((n_particles,), self.F_num, dtype=torch.float64, device=device)

        internal_energy = torch.zeros(n_particles, dtype=torch.float64, device=device)
        if self.species.dof_internal > 0:
            # Equi-partition: E_int = (dof_int/2) kT per particle
            E_int_mean = 0.5 * self.species.dof_internal * self.species.k_boltzmann * T
            # Gamma distribution for internal energy
            shape_param = self.species.dof_internal / 2.0
            # PyTorch Gamma: shape=alpha, rate=1 → scale by kT
            gamma_dist = torch.distributions.Gamma(
                torch.tensor(shape_param), torch.tensor(1.0)
            )
            internal_energy = gamma_dist.sample((n_particles,)).to(
                dtype=torch.float64, device=device
            ) * self.species.k_boltzmann * T

        return DSMCParticles(
            positions=positions,
            velocities=velocities,
            species_ids=species_ids,
            weights=weights,
            internal_energy=internal_energy,
        )

    def sort_particles(self, particles: DSMCParticles) -> None:
        """Assign particles to cells based on current positions."""
        for cell in self.cells:
            cell.particle_indices.clear()

        for i in range(particles.n_particles):
            cell_idx = self._cell_index(particles.positions[i])
            self.cells[cell_idx].particle_indices.append(i)

    def move_particles(self, particles: DSMCParticles, dt: float) -> None:
        """
        Ballistic advection: x_new = x + v * dt.

        After moving, apply wall boundary conditions for particles
        that have exited the domain.
        """
        # Move
        for d in range(self.dim):
            particles.positions[:, d] += particles.velocities[:, d] * dt

        # Apply wall boundaries
        for i in range(particles.n_particles):
            for d in range(self.dim):
                pos_d = particles.positions[i, d].item()
                lo = self.domain_min[d].item()
                hi = self.domain_max[d].item()

                if pos_d < lo:
                    face_key = f"lo_{d}"
                    if face_key in self.walls:
                        particles.velocities[i] = self.walls[face_key].reflect(
                            particles.velocities[i]
                        )
                    else:
                        # Default: specular reflection
                        particles.velocities[i, d] = abs(particles.velocities[i, d])
                    particles.positions[i, d] = lo + (lo - pos_d)

                elif pos_d > hi:
                    face_key = f"hi_{d}"
                    if face_key in self.walls:
                        particles.velocities[i] = self.walls[face_key].reflect(
                            particles.velocities[i]
                        )
                    else:
                        particles.velocities[i, d] = -abs(particles.velocities[i, d])
                    particles.positions[i, d] = hi - (pos_d - hi)

    def sample_macroscopic(self, particles: DSMCParticles) -> None:
        """
        Sample macroscopic quantities (density, velocity, temperature)
        in each cell from particle data.
        """
        k_B = self.species.k_boltzmann
        m = self.species.mass
        V_cell = self.cell_volume()

        for cell_id, cell in enumerate(self.cells):
            idx = cell.particle_indices
            N_p = len(idx)

            if N_p == 0:
                continue

            # Number density: n = N_p * F_num / V_cell
            n_density = N_p * self.F_num / V_cell

            # Bulk velocity: u = <v>
            v_sum = torch.zeros(3, dtype=torch.float64, device=particles.velocities.device)
            for i in idx:
                v_sum += particles.velocities[i]
            u_bulk = v_sum / N_p

            # Temperature from kinetic energy: (3/2)kT = (m/2)<|v-u|²>
            ke_sum = 0.0
            for i in idx:
                dv = particles.velocities[i] - u_bulk
                ke_sum += torch.dot(dv, dv).item()
            T_cell = m * ke_sum / (3.0 * k_B * N_p) if N_p > 0 else 0.0

            # Accumulate
            cell.n_samples += 1
            cell.sum_n_density += n_density
            cell.sum_temperature += T_cell
            if cell.sum_velocity is None:
                cell.sum_velocity = u_bulk.clone()
            else:
                cell.sum_velocity += u_bulk
            cell.sum_pressure += n_density * k_B * T_cell

    def step(self, particles: DSMCParticles, dt: float) -> Dict[str, int]:
        """
        Perform one DSMC timestep.

        Steps:
            1. Move particles (ballistic)
            2. Sort into cells
            3. Collisions (NTC or BGK)
            4. Sample macroscopic

        Args:
            particles: Particle ensemble
            dt: Timestep

        Returns:
            Dict with statistics: 'n_collisions', 'n_particles'
        """
        # 1. Move
        self.move_particles(particles, dt)

        # 2. Sort
        self.sort_particles(particles)

        # 3. Collisions
        total_collisions = 0
        V_cell = self.cell_volume()

        for cell_id, cell in enumerate(self.cells):
            if len(cell.particle_indices) < 2:
                continue

            if self.use_bgk:
                # Need cell-local macroscopic quantities
                idx = cell.particle_indices
                N_p = len(idx)
                n_density = N_p * self.F_num / V_cell

                v_sum = torch.zeros(3, dtype=torch.float64, device=particles.velocities.device)
                for i in idx:
                    v_sum += particles.velocities[i]
                u_cell = v_sum / N_p

                ke_sum = 0.0
                for i in idx:
                    dv = particles.velocities[i] - u_cell
                    ke_sum += torch.dot(dv, dv).item()
                T_cell = max(self.species.mass * ke_sum / (3.0 * self.species.k_boltzmann * N_p), 1.0)

                n_coll = self.bgk.relax(
                    particles, cell, n_density, T_cell, u_cell, dt
                )
            else:
                n_coll = self.collision_kernel.perform_collisions(
                    particles, cell, cell_id, V_cell, dt, self.F_num
                )

            total_collisions += n_coll

        # 4. Sample
        self.sample_macroscopic(particles)

        return {
            "n_collisions": total_collisions,
            "n_particles": particles.n_particles,
        }

    def estimate_timestep(self, T: float) -> float:
        """
        Estimate stable timestep from CFL criterion.

        Δt < Δx / (c_max)  where c_max ≈ 3 × thermal speed.
        """
        c_max = 3.0 * self.species.thermal_speed(T)
        dx_min = min(self.dx[d].item() for d in range(self.dim))
        return 0.2 * dx_min / c_max

    def knudsen_number(self, n_density: float, T: float, L: float) -> float:
        """
        Compute Knudsen number Kn = λ/L.

        Args:
            n_density: Number density
            T: Temperature
            L: Characteristic length scale

        Returns:
            Knudsen number
        """
        mfp = self.species.mean_free_path(n_density, T)
        return mfp / L


# ═══════════════════════════════════════════════════════════════════════════════
#  KNUDSEN REGIME ADAPTIVE SOLVER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class KnudsenAdaptiveSolver:
    """
    Hybrid solver that switches between methods based on local Knudsen number.

    Continuum (Kn < 0.001): Skip DSMC, use NS-based correction.
    Slip (0.001-0.1): DSMC with enhanced sampling, or NS + slip BC.
    Transition (0.1-10): Full DSMC (NTC).
    Free molecular (Kn > 10): Collisionless DSMC (skip NTC kernel).

    This solver runs DSMC everywhere but adjusts the collision model
    per-cell based on the local Knudsen number.
    """

    dsmc: DSMCSolver
    L_char: float = 1.0  # Characteristic length

    def adaptive_step(self, particles: DSMCParticles, dt: float) -> Dict[str, int]:
        """
        Perform one timestep with regime-adaptive collisions.

        In cells where Kn > 10, skip collisions entirely.
        In cells where Kn < 0.001, use BGK for efficiency.
        Otherwise, use full NTC.
        """
        # Move & sort
        self.dsmc.move_particles(particles, dt)
        self.dsmc.sort_particles(particles)

        total_collisions = 0
        V_cell = self.dsmc.cell_volume()
        regime_counts = {r: 0 for r in KnudsenRegime}

        for cell_id, cell in enumerate(self.dsmc.cells):
            idx = cell.particle_indices
            N_p = len(idx)

            if N_p < 2:
                continue

            # Compute cell-local properties
            n_density = N_p * self.dsmc.F_num / V_cell
            v_sum = torch.zeros(3, dtype=torch.float64, device=particles.velocities.device)
            for i in idx:
                v_sum += particles.velocities[i]
            u_cell = v_sum / N_p

            ke_sum = 0.0
            for i in idx:
                dv = particles.velocities[i] - u_cell
                ke_sum += torch.dot(dv, dv).item()
            T_cell = max(
                self.dsmc.species.mass * ke_sum
                / (3.0 * self.dsmc.species.k_boltzmann * N_p),
                1.0,
            )

            # Local Knudsen
            Kn_local = self.dsmc.knudsen_number(n_density, T_cell, self.L_char)
            regime = classify_knudsen(Kn_local)
            regime_counts[regime] += 1

            if regime == KnudsenRegime.FREE_MOLECULAR:
                # Skip collisions
                continue
            elif regime == KnudsenRegime.CONTINUUM:
                # Use BGK for efficiency
                n_coll = self.dsmc.bgk.relax(
                    particles, cell, n_density, T_cell, u_cell, dt
                )
            else:
                # Transition or slip: full NTC
                n_coll = self.dsmc.collision_kernel.perform_collisions(
                    particles, cell, cell_id, V_cell, dt, self.dsmc.F_num
                )

            total_collisions += n_coll

        # Sample macroscopic
        self.dsmc.sample_macroscopic(particles)

        return {
            "n_collisions": total_collisions,
            "n_particles": particles.n_particles,
            "regime_continuum": regime_counts[KnudsenRegime.CONTINUUM],
            "regime_slip": regime_counts[KnudsenRegime.SLIP],
            "regime_transition": regime_counts[KnudsenRegime.TRANSITION],
            "regime_free_molecular": regime_counts[KnudsenRegime.FREE_MOLECULAR],
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Regime classification
    "KnudsenRegime",
    "classify_knudsen",
    # Gas species
    "GasSpecies",
    # Sampling
    "sample_maxwellian",
    "sample_maxwellian_flux",
    # Collision kernels
    "NTCCollisionKernel",
    "BGKRelaxation",
    # Boundaries
    "WallBoundaryType",
    "WallBoundary",
    # Data structures
    "DSMCParticles",
    "DSMCCell",
    # Core solver
    "DSMCSolver",
    # Adaptive
    "KnudsenAdaptiveSolver",
]
