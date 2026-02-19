"""
Hybrid 1D Reacting-Flow DNS Solver
====================================

Premixed H2-air flame propagation.  Dense torch.Tensor computation
for the solver loop (standard central-difference FD + vectorised
Arrhenius chemistry).  Thin QTT-compatible wrapper on the state so
the Campaign IV caller can access ``.max_rank`` / ``.memory_bytes()``.

At nx=256 the full state is ~18 KB of dense floats — QTT would add
overhead with zero compression benefit and lossy TCI reconstruction
that kills chemistry accuracy.

Architecture:
    Transport :  Central-difference FD on dense arrays (vectorised).
    Chemistry :  Subcycled forward-Euler, enthalpy-conserving, on the
                 full grid simultaneously.  Vectorised — no per-cell loop.
    BCs       :  Direct Dirichlet writes at boundaries.
    Strang    :  Chem(dt/2) → Transport(dt) → Chem(dt/2).

Physical target:  S_L ≈ 2–3 m/s for stoichiometric H2-air at 1 atm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor


# =====================================================================
# Chemical Mechanism Data
# =====================================================================

_H2 = 0
_O2 = 1
_H2O = 2
_H = 3
_O = 4
_OH = 5
_N_SPECIES = 6

_MW = torch.tensor(
    [2.016e-3, 32.0e-3, 18.015e-3, 1.008e-3, 16.0e-3, 17.008e-3],
    dtype=torch.float64,
)

_NASA_HIGH: Dict[int, Tuple[float, ...]] = {
    _H2: (2.99142, 7.0006e-4, -5.633e-8, -9.231e-12, 1.582e-15, -835.0, -1.355),
    _O2: (3.66096, 6.5637e-4, -1.411e-7, 2.058e-11, -1.299e-15, -1216.0, 3.416),
    _H2O: (2.67704, 2.9735e-3, -7.737e-7, 9.443e-11, -4.269e-15, -29886.0, 6.883),
    _H: (2.50000, 0.0, 0.0, 0.0, 0.0, 25474.0, -0.460),
    _O: (2.54206, -2.755e-5, -3.102e-9, 4.551e-12, -4.368e-16, 29230.0, 4.921),
    _OH: (2.88273, 1.0139e-3, -2.276e-7, 2.175e-11, -5.126e-16, 3886.0, 5.595),
}

_NASA_LOW: Dict[int, Tuple[float, ...]] = {
    _H2: (3.29812, 8.249e-4, -8.143e-7, -9.475e-11, 4.134e-13, -1013.0, -3.294),
    _O2: (3.21294, 1.1275e-3, -5.756e-7, 1.314e-10, -8.768e-15, -1005.0, 6.034),
    _H2O: (3.38684, 3.4749e-3, -6.355e-6, 6.969e-9, -2.507e-12, -30208.0, 2.590),
    _H: (2.50000, 0.0, 0.0, 0.0, 0.0, 25474.0, -0.460),
    _O: (2.94643, -1.6382e-3, 2.421e-6, -1.602e-9, 3.891e-13, 29148.0, 2.964),
    _OH: (3.63727, -1.8510e-4, 3.426e-6, -3.286e-9, 1.121e-12, 3615.0, 1.359),
}

_HF298 = torch.tensor(
    [0.0, 0.0, -241826.0, 217999.0, 249175.0, 38987.0],
    dtype=torch.float64,
)

_R_UNIV = 8.31446
_W_N2 = 28.014e-3

# N₂ NASA-7 thermodynamic coefficients (not a tracked species but needed for cp)
_N2_NASA_HIGH = (2.95258, 1.3969e-3, -4.926e-7, 7.860e-11, -4.608e-15, -923.9, 5.872)
_N2_NASA_LOW = (3.53101, -1.2366e-4, -5.030e-7, 2.435e-9, -1.409e-12, -1047.0, 2.967)


# =====================================================================
# Mechanism dataclass
# =====================================================================

@dataclass
class ReactionMechanism:
    """Reduced H2-air mechanism with correct thermodynamics."""
    n_species: int = _N_SPECIES
    n_reactions: int = 3
    species_names: Tuple[str, ...] = ("H2", "O2", "H2O", "H", "O", "OH")
    mw: Tensor = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.mw is None:
            self.mw = _MW.clone()


def hydrogen_air_9species() -> ReactionMechanism:
    """Return reduced H2-air mechanism (backward-compat name)."""
    return ReactionMechanism()


# =====================================================================
# Thermodynamic Properties
# =====================================================================

def _cp_species_R(T: Tensor) -> Tensor:
    """cp_k / R for each species.  (N, n_species)."""
    N = T.shape[0]
    cp = torch.zeros(N, _N_SPECIES, dtype=T.dtype, device=T.device)
    high_mask = T >= 1000.0
    for k in range(_N_SPECIES):
        ah = _NASA_HIGH[k]
        al = _NASA_LOW[k]
        cp_high = ah[0] + ah[1] * T + ah[2] * T**2 + ah[3] * T**3 + ah[4] * T**4
        cp_low = al[0] + al[1] * T + al[2] * T**2 + al[3] * T**3 + al[4] * T**4
        cp[:, k] = torch.where(high_mask, cp_high, cp_low)
    return cp


def mixture_cp(T: Tensor, Y: Tensor, mw: Tensor) -> Tensor:
    """Mixture cp [J/(kg*K)].  Includes N₂ dilutant (Y_N2 = 1 - sum(Y_k))."""
    cp_over_R = _cp_species_R(T)  # (N, n_species)
    mw_dev = mw.to(device=T.device, dtype=T.dtype)
    cp_mass = cp_over_R * _R_UNIV / mw_dev.unsqueeze(0)  # (N, n_species)
    cp_tracked = (Y * cp_mass).sum(dim=1)  # J/(kg·K) from tracked species

    # N₂ contribution: Y_N2 = 1 - sum(Y_k), cp_N2 from NASA polynomials
    Y_N2 = (1.0 - Y.sum(dim=1)).clamp(min=0.0)
    high_mask = T >= 1000.0
    ah = _N2_NASA_HIGH
    al = _N2_NASA_LOW
    cp_N2_R_high = ah[0] + ah[1] * T + ah[2] * T**2 + ah[3] * T**3 + ah[4] * T**4
    cp_N2_R_low = al[0] + al[1] * T + al[2] * T**2 + al[3] * T**3 + al[4] * T**4
    cp_N2_R = torch.where(high_mask, cp_N2_R_high, cp_N2_R_low)
    cp_N2_mass = cp_N2_R * _R_UNIV / _W_N2  # J/(kg·K)

    return cp_tracked + Y_N2 * cp_N2_mass


# =====================================================================
# Transport Properties
# =====================================================================

def mixture_thermal_conductivity(T: Tensor) -> Tensor:
    """lambda [W/(m*K)].  Eucken for N2-dominated."""
    return 2.58e-2 * (T.clamp(min=200.0) / 300.0) ** 0.7


def species_diffusivity(T: Tensor, p: float) -> Tensor:
    """D_k [m2/s] into N2.  Chapman-Enskog.  (N, n_sp)."""
    D_ref = torch.tensor(
        [7.7e-5, 2.1e-5, 2.5e-5, 8.5e-5, 2.2e-5, 2.3e-5],
        dtype=T.dtype, device=T.device,
    )
    return D_ref.unsqueeze(0) * (T.clamp(min=200.0) / 300.0).unsqueeze(1) ** 1.75 * (101325.0 / p)


def thermal_diffusivity(T: Tensor, Y: Tensor, rho: Tensor, mw: Tensor) -> Tensor:
    """alpha = lambda / (rho * cp) [m2/s]."""
    lam = mixture_thermal_conductivity(T)
    cp = mixture_cp(T, Y, mw)
    return lam / (rho * cp)


# =====================================================================
# Chemical Kinetics
# =====================================================================

@dataclass
class _ArrheniusRate:
    A: float
    beta: float
    Ea: float

_REACTIONS = [
    _ArrheniusRate(A=1.2e8, beta=0.0, Ea=80000.0),      # 2H2 + O2 → 2H2O  (global, calibrated S_L ≈ 2 m/s)
    _ArrheniusRate(A=5.06e4, beta=2.67, Ea=26300.0),    # O + H2 → H + OH  (chain branching)
    _ArrheniusRate(A=3.55e15, beta=-0.41, Ea=69500.0),  # H + O2 → O + OH  (chain branching)
]


def _rate_constants(T: Tensor, thickening_factor: float = 1.0) -> Tuple[Tensor, Tensor, Tensor]:
    """Arrhenius rate constants, optionally divided by thickening factor F."""
    T_safe = T.clamp(min=200.0, max=6000.0)
    inv_RT = 1.0 / (_R_UNIV * T_safe)
    inv_F = 1.0 / thickening_factor
    k1 = _REACTIONS[0].A * inv_F * torch.exp(-_REACTIONS[0].Ea * inv_RT)
    k2 = _REACTIONS[1].A * inv_F * T_safe ** _REACTIONS[1].beta * torch.exp(-_REACTIONS[1].Ea * inv_RT)
    k3 = _REACTIONS[2].A * inv_F * T_safe ** _REACTIONS[2].beta * torch.exp(-_REACTIONS[2].Ea * inv_RT)
    return k1, k2, k3


def species_production_rates(
    T: Tensor, Y: Tensor, rho: Tensor, mw: Tensor,
    thickening_factor: float = 1.0,
) -> Tensor:
    """omega_k [kg/(m3*s)].  (N, n_sp).  Optionally thinned by 1/F."""
    mw_dev = mw.to(device=T.device, dtype=T.dtype)
    C = (rho.unsqueeze(1) * Y / mw_dev.unsqueeze(0)).clamp(min=0.0)
    k1, k2, k3 = _rate_constants(T, thickening_factor)
    r1 = k1 * C[:, _H2] * C[:, _O2]
    r2 = k2 * C[:, _H2] * C[:, _O]
    r3 = k3 * C[:, _H] * C[:, _O2]
    omega = torch.zeros_like(C)
    omega[:, _H2] += -2.0 * r1 - 1.0 * r2
    omega[:, _O2] += -1.0 * r1 - 1.0 * r3
    omega[:, _H2O] += 2.0 * r1
    omega[:, _H] += 1.0 * r2 - 1.0 * r3
    omega[:, _O] += -1.0 * r2 + 1.0 * r3
    omega[:, _OH] += 1.0 * r2 + 1.0 * r3
    return omega * mw_dev.unsqueeze(0)


def heat_release_rate(
    T: Tensor, Y: Tensor, rho: Tensor, mw: Tensor,
) -> Tensor:
    """q [W/m3] = -sum(omega_k * hf_k / W_k)."""
    omega = species_production_rates(T, Y, rho, mw)
    mw_dev = mw.to(device=T.device, dtype=T.dtype)
    hf = _HF298.to(device=T.device, dtype=T.dtype)
    return -(omega / mw_dev.unsqueeze(0) * hf.unsqueeze(0)).sum(dim=1)


# =====================================================================
# Dense Field Wrapper (PackedQTT-compatible interface)
# =====================================================================

class DenseField:
    """
    Wraps a 1-D torch.Tensor but exposes the PackedQTT interface
    (.max_rank, .memory_bytes(), .rr, .num_sites, .clone())
    so the Campaign IV caller works without changes.
    """

    def __init__(self, data: Tensor) -> None:
        self.data = data

    @property
    def max_rank(self) -> int:
        return 1

    @property
    def num_sites(self) -> int:
        n = self.data.numel()
        return max(1, int(math.log2(n))) if n > 0 else 0

    @property
    def rr(self) -> Tensor:
        return torch.ones(max(1, self.num_sites - 1), dtype=torch.long, device=self.data.device)

    def memory_bytes(self) -> int:
        return self.data.nelement() * self.data.element_size()

    def clone(self) -> "DenseField":
        return DenseField(self.data.clone())

    def __repr__(self) -> str:
        return f"DenseField(shape={self.data.shape}, dtype={self.data.dtype})"


# =====================================================================
# Solver State
# =====================================================================

@dataclass
class CombustionState:
    """All fields as DenseField (PackedQTT-compatible wrapper)."""
    T: DenseField
    Y: List[DenseField]
    rho: DenseField
    u: DenseField
    t: float


# =====================================================================
# Hybrid Combustion DNS Solver
# =====================================================================

class CombustionDNSSolver:
    """
    1D premixed flame DNS — dense torch.Tensor computation, QTT-compatible API.

    Strang splitting: Chem(dt/2) → Transport(dt) → Chem(dt/2)
    Transport: standard central-difference FD on dense arrays (vectorised)
    Chemistry: subcycled forward-Euler, enthalpy-conserving, fully vectorised
    BCs: direct Dirichlet writes at boundary cells
    """

    def __init__(
        self,
        mechanism: ReactionMechanism,
        nx: int = 256,
        L: float = 0.02,
        cfl: float = 0.3,
        tol: float = 1e-6,
        rank_cap: int = 64,
        n_chem_sub: int = 10,
        p_atm: float = 101325.0,
    ) -> None:
        self.mech = mechanism
        self.p = p_atm
        self.n_bits = int(math.log2(nx))
        if 2 ** self.n_bits != nx:
            raise ValueError(f"nx must be power of 2, got {nx}")
        self.nx = nx
        self.L = L
        self.dx = L / nx
        self.tol = tol
        self.rank_cap = rank_cap
        self.cfl = cfl
        self.n_chem_sub = n_chem_sub
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float64
        self.mw = mechanism.mw.to(device=self.device, dtype=torch.float64)

        # Thickened Flame Model (TFM) factor.
        # Blint-corrected laminar flame thickness for stoich H₂-air:
        #   δ_B = (T_ad/T_u)^0.7 * λ_u / (ρ_u * cp_u * S_L)
        #       ≈ (2230/300)^0.7 * 0.026 / (0.85 * 1389 * 2) ≈ 50 μm
        _delta_flame = 5.0e-5
        _n_cells_min = 4
        self.thickening_factor: float = max(1.0, _n_cells_min * self.dx / _delta_flame)

        # BC storage (set during init)
        self._bc_T_left: float = 300.0
        self._bc_T_right: float = 2200.0
        self._bc_rho_left: float = 1.16
        self._bc_rho_right: float = 0.11
        self._bc_Y_left: List[float] = [0.0] * _N_SPECIES
        self._bc_Y_right: List[float] = [0.0] * _N_SPECIES

    # -----------------------------------------------------------------
    # Finite-difference operators (dense, vectorised)
    # -----------------------------------------------------------------

    def _laplacian(self, f: Tensor) -> Tensor:
        """d²f/dx² via central difference.  Neumann at boundaries."""
        lap = torch.zeros_like(f)
        lap[1:-1] = (f[2:] - 2.0 * f[1:-1] + f[:-2]) / self.dx ** 2
        lap[0] = lap[1]
        lap[-1] = lap[-2]
        return lap

    def _gradient(self, f: Tensor) -> Tensor:
        """df/dx via central difference.  One-sided at boundaries."""
        grad = torch.zeros_like(f)
        grad[1:-1] = (f[2:] - f[:-2]) / (2.0 * self.dx)
        grad[0] = (f[1] - f[0]) / self.dx
        grad[-1] = (f[-1] - f[-2]) / self.dx
        return grad

    # -----------------------------------------------------------------
    # Boundary Conditions
    # -----------------------------------------------------------------

    def _apply_bcs(
        self, T: Tensor, Y: Tensor, rho: Tensor, u: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Dirichlet BCs at left (unburned) and right (burned)."""
        n_buf = 3

        T[:n_buf] = self._bc_T_left
        rho[:n_buf] = self._bc_rho_left
        u[:n_buf] = 0.0
        for k in range(_N_SPECIES):
            Y[:n_buf, k] = self._bc_Y_left[k]

        T[-n_buf:] = self._bc_T_right
        rho[-n_buf:] = self._bc_rho_right
        u[-n_buf:] = 0.0
        for k in range(_N_SPECIES):
            Y[-n_buf:, k] = self._bc_Y_right[k]

        return T, Y, rho, u

    # -----------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------

    def premixed_flame_init(
        self,
        T_u: float = 300.0,
        T_b: float = 2200.0,
        phi: float = 1.0,
        p: float = 101325.0,
    ) -> CombustionState:
        """
        Initialize 1D premixed flame.  Flame at x = L/2.

        Independent tanh profiles for T (broad) and species (sharp)
        create a preheat zone: T has risen but fuel hasn't burned yet
        → disequilibrium drives the flame.
        T_ad from enthalpy balance.
        """
        self.p = p
        nx = self.nx
        device = self.device
        mw = self.mw
        hf = _HF298.to(device=device, dtype=torch.float64)

        W_H2, W_O2, W_H2O = 2.016e-3, 32.0e-3, 18.015e-3
        denom_u = phi * 2.0 * W_H2 + W_O2 + 3.76 * _W_N2
        Y_H2_u = phi * 2.0 * W_H2 / denom_u
        Y_O2_u = W_O2 / denom_u

        if phi <= 1.0:
            Y_H2_b, Y_O2_b = 0.0, (1.0 - phi) * W_O2 / denom_u
            Y_H2O_b = phi * 2.0 * W_H2O / denom_u
        else:
            Y_O2_b = 0.0
            Y_H2O_b = 2.0 * W_H2O / denom_u
            Y_H2_b = (phi - 1.0) * 2.0 * W_H2 / denom_u

        Y_u = torch.tensor([Y_H2_u, Y_O2_u, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64, device=device)
        Y_b = torch.tensor([Y_H2_b, Y_O2_b, Y_H2O_b, 0.0, 0.0, 0.0], dtype=torch.float64, device=device)

        # Adiabatic flame temperature from enthalpy balance
        cp_u = mixture_cp(
            torch.tensor([T_u], dtype=torch.float64, device=device),
            Y_u.unsqueeze(0), mw,
        ).item()
        h_chem_u = (Y_u * hf / mw).sum().item()
        h_total_u = cp_u * T_u + h_chem_u

        T_ad = float(T_b)
        for _ in range(80):
            cp_b = mixture_cp(
                torch.tensor([T_ad], dtype=torch.float64, device=device),
                Y_b.unsqueeze(0), mw,
            ).item()
            h_chem_b = (Y_b * hf / mw).sum().item()
            T_ad_new = (h_total_u - h_chem_b) / max(cp_b, 100.0)
            if abs(T_ad_new - T_ad) < 0.05:
                break
            T_ad = 0.7 * T_ad + 0.3 * T_ad_new
        T_ad = max(T_ad, T_u + 100.0)

        def _mean_mw(Yk: Tensor) -> float:
            Y_N2_val = max(0.0, 1.0 - Yk.sum().item())
            return 1.0 / ((Yk / mw).sum().item() + Y_N2_val / _W_N2)

        rho_u = p / (_R_UNIV / _mean_mw(Y_u) * T_u)
        rho_b = p / (_R_UNIV / _mean_mw(Y_b) * T_ad)

        self._bc_T_left = T_u
        self._bc_T_right = T_ad
        self._bc_rho_left = rho_u
        self._bc_rho_right = rho_b
        self._bc_Y_left = Y_u.tolist()
        self._bc_Y_right = Y_b.tolist()

        # Build profiles on grid
        x = torch.linspace(0.0, self.L, nx, dtype=torch.float64, device=device)
        x_flame = self.L / 2.0

        delta_T = max(5e-4, 4.0 * self.dx)   # thermal profile half-width
        delta_Y = delta_T * 0.4               # species profile half-width (sharper)

        xi_T = 0.5 * (1.0 + torch.tanh((x - x_flame) / delta_T))
        xi_Y = 0.5 * (1.0 + torch.tanh((x - x_flame) / delta_Y))

        T = T_u + (T_ad - T_u) * xi_T

        Y = torch.zeros(nx, _N_SPECIES, dtype=torch.float64, device=device)
        for k in range(_N_SPECIES):
            Y[:, k] = Y_u[k].item() + (Y_b[k].item() - Y_u[k].item()) * xi_Y

        Y_N2 = (1.0 - Y.sum(dim=1)).clamp(min=0.0)
        inv_W = (Y / mw.unsqueeze(0)).sum(dim=1) + Y_N2 / _W_N2
        rho = p / (_R_UNIV * inv_W * T)

        u = torch.zeros(nx, dtype=torch.float64, device=device)

        T, Y, rho, u = self._apply_bcs(T, Y, rho, u)

        return CombustionState(
            T=DenseField(T),
            Y=[DenseField(Y[:, k].clone()) for k in range(_N_SPECIES)],
            rho=DenseField(rho),
            u=DenseField(u),
            t=0.0,
        )

    # -----------------------------------------------------------------
    # Transport Step (central-difference FD, vectorised)
    # -----------------------------------------------------------------

    def _transport_step(self, state: CombustionState, dt: float) -> CombustionState:
        """
        dT/dt  = alpha(x) * d²T/dx² - u * dT/dx
        dYk/dt = Dk(x)    * d²Yk/dx² - u * dYk/dx
        """
        mw, p_atm = self.mw, self.p
        F = self.thickening_factor

        T = state.T.data.clone()
        Y = torch.stack([state.Y[k].data for k in range(_N_SPECIES)], dim=1).clone()
        rho = state.rho.data.clone()
        u = state.u.data.clone()

        alpha = F * thermal_diffusivity(T, Y, rho, mw)
        Dk = F * species_diffusivity(T, p_atm)

        # Temperature
        lap_T = self._laplacian(T)
        grad_T = self._gradient(T)
        T = T + dt * (alpha * lap_T - u * grad_T)
        T = T.clamp(min=200.0, max=5000.0)

        # Species
        for k in range(_N_SPECIES):
            Yk = Y[:, k]
            lap_Y = self._laplacian(Yk)
            grad_Y = self._gradient(Yk)
            Y[:, k] = (Yk + dt * (Dk[:, k] * lap_Y - u * grad_Y)).clamp(min=0.0)

        # Density from ideal gas
        Y_N2 = (1.0 - Y.sum(dim=1)).clamp(min=0.0)
        inv_W = (Y / mw.unsqueeze(0)).sum(dim=1) + Y_N2 / _W_N2
        rho = p_atm / (_R_UNIV * inv_W.clamp(min=1e-10) * T.clamp(min=200.0))

        T, Y, rho, u = self._apply_bcs(T, Y, rho, u)

        return CombustionState(
            T=DenseField(T),
            Y=[DenseField(Y[:, k].clone()) for k in range(_N_SPECIES)],
            rho=DenseField(rho),
            u=DenseField(u),
            t=state.t,
        )

    # -----------------------------------------------------------------
    # Chemistry Step (subcycled forward-Euler, enthalpy-conserving)
    # -----------------------------------------------------------------

    def _chemistry_step(self, state: CombustionState, dt: float) -> CombustionState:
        """
        Integrate dY/dt = omega/rho  with T from enthalpy conservation.
        Subcycled for stiffness; fully vectorised over all grid points.
        """
        mw = self.mw
        hf = _HF298.to(device=self.device, dtype=torch.float64)
        n_sub_base = self.n_chem_sub
        p_atm = self.p
        F = self.thickening_factor

        T = state.T.data.clone()
        Y = torch.stack([state.Y[k].data for k in range(_N_SPECIES)], dim=1).clone()
        rho = state.rho.data.clone()

        # Total enthalpy (conserved)
        cp_init = mixture_cp(T, Y, mw)
        h_chem_init = (Y * (hf / mw).unsqueeze(0)).sum(dim=1)
        h_total = cp_init * T + h_chem_init

        # Adaptive subcycling
        omega_init = species_production_rates(T, Y, rho, mw, F)
        active_mask = (Y > 1e-8) & (omega_init.abs() > 1e-6)
        n_sub = n_sub_base
        if active_mask.any():
            tau_chem = (
                Y[active_mask] * rho.unsqueeze(1).expand_as(Y)[active_mask]
            ).abs() / (omega_init[active_mask].abs() + 1e-30)
            tau_min = tau_chem.min().item()
            if tau_min > 0:
                n_sub = max(n_sub_base, int(math.ceil(dt / (0.5 * tau_min))))
        n_sub = min(n_sub, 5000)
        dt_sub = dt / n_sub

        for _ in range(n_sub):
            omega = species_production_rates(T, Y, rho, mw, F)
            Y = (Y + dt_sub * omega / rho.clamp(min=0.01).unsqueeze(1)).clamp(min=0.0)

            # Enforce sum(Y_k) ≤ 1
            Y_sum = Y.sum(dim=1, keepdim=True)
            excess = (Y_sum - 1.0).clamp(min=0.0)
            Y = Y - excess * Y / (Y_sum + 1e-30)

            # T from enthalpy conservation
            h_chem = (Y * (hf / mw).unsqueeze(0)).sum(dim=1)
            cp = mixture_cp(T, Y, mw)
            T = ((h_total - h_chem) / cp.clamp(min=100.0)).clamp(200.0, 5000.0)

            # Density from ideal gas
            Y_N2 = (1.0 - Y.sum(dim=1)).clamp(min=0.0)
            inv_W = (Y / mw.unsqueeze(0)).sum(dim=1) + Y_N2 / _W_N2
            rho = p_atm / (_R_UNIV * inv_W.clamp(min=1e-10) * T)

        u = state.u.data.clone()

        return CombustionState(
            T=DenseField(T),
            Y=[DenseField(Y[:, k].clone()) for k in range(_N_SPECIES)],
            rho=DenseField(rho),
            u=DenseField(u),
            t=state.t,
        )

    # -----------------------------------------------------------------
    # Time Step
    # -----------------------------------------------------------------

    def _compute_dt(self, state: CombustionState) -> float:
        """CFL + diffusion stability constraint (thermal AND species)."""
        T = state.T.data
        u = state.u.data
        rho = state.rho.data
        Y = torch.stack([state.Y[k].data for k in range(_N_SPECIES)], dim=1)
        F = self.thickening_factor

        u_eff = max(u.abs().max().item(), 3.0)
        dt_cfl = self.cfl * self.dx / u_eff

        # Thermal diffusion stability
        alpha_max = (F * thermal_diffusivity(T, Y, rho, self.mw)).max().item()

        # Species diffusion stability — D_H2, D_H can be 5-10× larger than alpha
        Dk = F * species_diffusivity(T, self.p)  # (nx, n_sp)
        D_max = Dk.max().item()

        diff_max = max(alpha_max, D_max)
        dt_diff = 0.25 * self.dx ** 2 / max(diff_max, 1e-20)

        return max(min(dt_cfl, dt_diff), 1e-12)

    # -----------------------------------------------------------------
    # Time Integration
    # -----------------------------------------------------------------

    def evolve(self, state: CombustionState, t_end: float) -> CombustionState:
        """Strang splitting: Chem(dt/2) → Transport(dt) → Chem(dt/2)."""
        current = CombustionState(
            T=state.T.clone(),
            Y=[state.Y[k].clone() for k in range(_N_SPECIES)],
            rho=state.rho.clone(),
            u=state.u.clone(),
            t=state.t,
        )
        t_rem = t_end

        while t_rem > 1e-15:
            dt = min(self._compute_dt(current), t_rem)
            current = self._chemistry_step(current, dt / 2.0)
            current = self._transport_step(current, dt)
            current = self._chemistry_step(current, dt / 2.0)
            current.t += dt
            t_rem -= dt

        return current

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    def max_heat_release(self, state: CombustionState) -> float:
        """Peak heat release [W/m³]."""
        T = state.T.data
        rho = state.rho.data
        Y = torch.stack([state.Y[k].data for k in range(_N_SPECIES)], dim=1)
        return float(heat_release_rate(T, Y, rho, self.mw).max().item())

    def flame_speed(self, state: CombustionState) -> float:
        """Consumption-based S_L = integral(|omega_fuel_TFM| dx) / (rho_u * Y_fu).

        Uses TFM-consistent omega (÷F) so the integral over the thickened
        flame gives the PHYSICAL burning velocity, not F × S_L.
        """
        T = state.T.data
        rho = state.rho.data
        Y = torch.stack([state.Y[k].data for k in range(_N_SPECIES)], dim=1)
        omega = species_production_rates(
            T, Y, rho, self.mw, self.thickening_factor,
        )

        n_u = max(1, self.nx // 10)
        rho_u = rho[:n_u].mean().item()
        Y_fu = Y[:n_u, _H2].mean().item()
        return omega[:, _H2].abs().sum().item() * self.dx / max(rho_u * Y_fu, 1e-10)

    def get_temperature_profile(self, state: CombustionState) -> Tensor:
        """T at all points."""
        return state.T.data.clone()

    def get_species_profiles(self, state: CombustionState) -> Tensor:
        """Y_k at all points (N, n_sp)."""
        return torch.stack([state.Y[k].data for k in range(_N_SPECIES)], dim=1)

    def get_max_ranks(self, state: CombustionState) -> Dict[str, int]:
        """Max rank per field (always 1 for dense storage)."""
        r: Dict[str, int] = {
            "T": state.T.max_rank,
            "rho": state.rho.max_rank,
            "u": state.u.max_rank,
        }
        for k in range(_N_SPECIES):
            r[f"Y_{k}"] = state.Y[k].max_rank
        return r
