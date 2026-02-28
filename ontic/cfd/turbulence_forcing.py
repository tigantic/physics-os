"""
Turbulence Forcing for QTT Navier-Stokes Solver
================================================

Implements large-scale stochastic forcing to maintain stationary turbulence.

The forcing injects energy at low wavenumbers (large scales), which then
cascades down to small scales following the Kolmogorov cascade:

    Large scales (forcing) → Inertial range (k^-5/3) → Dissipation (viscous)

Forcing types:
1. SPECTRAL FORCING: Energy injection at specific wavenumber band k_f
2. ORNSTEIN-UHLENBECK: Time-correlated stochastic forcing
3. TAYLOR-GREEN BOOST: Periodic re-energization of TG mode

The forcing is constructed in QTT format to avoid dense operations.

Author: TiganticLabz
Date: 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import math

import torch
from torch import Tensor

from ontic.cfd.qtt_turbo import (
    turbo_add_cores,
    turbo_scale,
    turbo_truncate,
    turbo_linear_combination,
    turbo_hadamard_cores,
)


@dataclass
class ForcingConfig:
    """Configuration for turbulence forcing."""
    forcing_type: str = 'spectral'  # 'spectral', 'ornstein_uhlenbeck', 'taylor_green'
    
    # Spectral forcing parameters
    k_forcing: int = 2              # Wavenumber band for energy injection
    k_band_width: int = 2           # Width of forcing band [k_f - w, k_f + w]
    epsilon_target: float = 0.1     # Target energy injection rate
    
    # Ornstein-Uhlenbeck parameters
    correlation_time: float = 0.1   # Time correlation for OU process
    forcing_amplitude: float = 1.0  # Amplitude of OU forcing
    
    # Taylor-Green boost
    tg_amplitude: float = 0.1       # Amplitude of TG re-injection
    tg_period: int = 50             # Re-inject every N steps
    
    # Common
    seed: int = 42                  # Random seed for reproducibility


@dataclass
class ForcingState:
    """State for time-correlated forcing."""
    ou_state: Optional[List[List[Tensor]]] = None  # OU process state for each component
    step_count: int = 0
    rng: torch.Generator = field(default_factory=torch.Generator)


class TurbulenceForcing:
    """
    Large-scale forcing to maintain stationary turbulence.
    
    Energy is injected at low wavenumbers and cascades down via
    the nonlinear advection term, creating the Kolmogorov cascade.
    """
    
    def __init__(
        self,
        n_bits: int,
        config: ForcingConfig,
        device: torch.device,
    ):
        self.n_bits = n_bits
        self.config = config
        self.device = device
        self.n_cores = 3 * n_bits  # 3D QTT
        self.N = 2 ** n_bits
        
        # Initialize state
        self.state = ForcingState()
        self.state.rng = torch.Generator(device='cpu')
        self.state.rng.manual_seed(config.seed)
        
        # Pre-build forcing modes (QTT representation of low-k modes)
        self._build_forcing_modes()
    
    def _build_forcing_modes(self):
        """
        Build QTT representations of low-wavenumber forcing modes.
        
        For turbulence, we force at large scales (low k) which then
        cascade to small scales via the energy cascade.
        
        The modes are sines/cosines at wavenumbers k_f ± band_width.
        """
        k_f = self.config.k_forcing
        k_width = self.config.k_band_width
        N = self.N
        n_bits = self.n_bits
        
        # We'll create forcing in dense format then compress to QTT
        # This is done once at initialization, not every step
        
        # Build sin(k*x), cos(k*x), etc. for forcing wavenumbers
        x = torch.linspace(0, 2 * math.pi, N + 1, device=self.device)[:-1]
        
        self.forcing_modes = []
        
        # Create modes for each wavenumber in forcing band
        for k in range(max(1, k_f - k_width), k_f + k_width + 1):
            # Mode in x-direction: sin(k*x)
            mode_1d = torch.sin(k * x)
            
            # Extend to 3D: sin(k*x) * cos(k*y) * cos(k*z) (divergence-free pattern)
            X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')
            
            # Divergence-free forcing (like Taylor-Green)
            fx = torch.sin(k * X) * torch.cos(k * Y) * torch.cos(k * Z)
            fy = -torch.cos(k * X) * torch.sin(k * Y) * torch.cos(k * Z)
            fz = torch.zeros_like(fx)
            
            # Compress to QTT
            fx_qtt = self._dense_to_qtt(fx)
            fy_qtt = self._dense_to_qtt(fy)
            fz_qtt = self._dense_to_qtt(fz)
            
            self.forcing_modes.append((k, [fx_qtt, fy_qtt, fz_qtt]))
    
    def _dense_to_qtt(self, field: Tensor, max_rank: int = 32) -> List[Tensor]:
        """Convert dense 3D field to QTT format."""
        N = field.shape[0]
        n_bits = self.n_bits
        
        # Morton reorder
        flat = torch.zeros(N**3, device=field.device, dtype=field.dtype)
        for ix in range(N):
            for iy in range(N):
                for iz in range(N):
                    morton_idx = 0
                    for bit in range(n_bits):
                        morton_idx |= ((ix >> bit) & 1) << (3 * bit)
                        morton_idx |= ((iy >> bit) & 1) << (3 * bit + 1)
                        morton_idx |= ((iz >> bit) & 1) << (3 * bit + 2)
                    flat[morton_idx] = field[ix, iy, iz]
        
        # TT-SVD decomposition
        cores = []
        work = flat.unsqueeze(0)  # (1, N³)
        n_cores = 3 * n_bits
        
        for i in range(n_cores - 1):
            r_left = work.shape[0]
            remaining = work.shape[1]
            d = 2  # QTT mode size
            r_right_max = remaining // d
            
            work = work.reshape(r_left * d, r_right_max)
            
            # SVD
            U, S, Vh = torch.linalg.svd(work, full_matrices=False)
            k = min(max_rank, len(S))
            
            # Truncate
            U_k = U[:, :k]
            S_k = S[:k]
            Vh_k = Vh[:k, :]
            
            # Core
            cores.append(U_k.reshape(r_left, d, k))
            work = torch.diag(S_k) @ Vh_k
        
        # Last core
        cores.append(work.reshape(work.shape[0], 2, 1))
        
        return cores
    
    def get_forcing(self) -> List[List[Tensor]]:
        """
        Get forcing term for current timestep.
        
        Returns [f_x, f_y, f_z] where each is a list of QTT cores.
        """
        config = self.config
        
        if config.forcing_type == 'spectral':
            return self._spectral_forcing()
        elif config.forcing_type == 'ornstein_uhlenbeck':
            return self._ou_forcing()
        elif config.forcing_type == 'taylor_green':
            return self._taylor_green_forcing()
        else:
            raise ValueError(f"Unknown forcing type: {config.forcing_type}")
    
    def _spectral_forcing(self) -> List[List[Tensor]]:
        """
        Spectral forcing: Random phases on low-k modes.
        
        f(x,t) = Σ_k A_k * sin(k·x + φ_k(t))
        
        where φ_k are random phases that change each timestep.
        """
        # Random amplitudes for each mode
        n_modes = len(self.forcing_modes)
        
        # Generate random phases
        phases = torch.rand(n_modes, generator=self.state.rng) * 2 * math.pi
        amplitudes = torch.randn(n_modes, generator=self.state.rng)
        amplitudes = amplitudes / math.sqrt(n_modes)  # Normalize
        amplitudes *= math.sqrt(self.config.epsilon_target)
        
        # Combine modes
        fx_total = None
        fy_total = None
        fz_total = None
        
        for i, (k, (fx, fy, fz)) in enumerate(self.forcing_modes):
            amp = amplitudes[i].item()
            
            if fx_total is None:
                fx_total = turbo_scale(fx, amp)
                fy_total = turbo_scale(fy, amp)
                fz_total = turbo_scale(fz, amp)
            else:
                fx_total = turbo_add_cores(fx_total, fx, alpha=1.0, beta=amp)
                fy_total = turbo_add_cores(fy_total, fy, alpha=1.0, beta=amp)
                fz_total = turbo_add_cores(fz_total, fz, alpha=1.0, beta=amp)
        
        # Truncate combined forcing
        max_rank = 32
        fx_total = turbo_truncate(fx_total, max_rank)
        fy_total = turbo_truncate(fy_total, max_rank)
        fz_total = turbo_truncate(fz_total, max_rank)
        
        self.state.step_count += 1
        
        return [fx_total, fy_total, fz_total]
    
    def _ou_forcing(self) -> List[List[Tensor]]:
        """
        Ornstein-Uhlenbeck forcing: Time-correlated stochastic process.
        
        dF = -F/τ dt + σ dW
        
        where τ is correlation time, σ is amplitude.
        """
        tau = self.config.correlation_time
        sigma = self.config.forcing_amplitude
        dt = 0.01  # Assumed timestep
        
        # Decay factor
        decay = math.exp(-dt / tau)
        noise_std = sigma * math.sqrt(1 - decay**2)
        
        if self.state.ou_state is None:
            # Initialize with spectral forcing
            self.state.ou_state = self._spectral_forcing()
        
        # Update: F_new = decay * F_old + noise
        new_state = []
        noise_forcing = self._spectral_forcing()
        
        for i in range(3):
            updated = turbo_linear_combination([
                (decay, self.state.ou_state[i]),
                (noise_std, noise_forcing[i]),
            ], max_rank=32)
            new_state.append(updated)
        
        self.state.ou_state = new_state
        self.state.step_count += 1
        
        return new_state
    
    def _taylor_green_forcing(self) -> List[List[Tensor]]:
        """
        Taylor-Green re-injection: Periodically boost the TG mode.
        """
        step = self.state.step_count
        period = self.config.tg_period
        
        if len(self.forcing_modes) == 0:
            return [self._zero_field(), self._zero_field(), self._zero_field()]
        
        # Only force every `period` steps
        if step % period == 0:
            amp = self.config.tg_amplitude
            # Use the k=1 mode (closest to TG)
            _, (fx, fy, fz) = self.forcing_modes[0]
            forcing = [
                turbo_scale(fx, amp),
                turbo_scale(fy, amp),
                turbo_scale(fz, amp),
            ]
        else:
            forcing = [self._zero_field(), self._zero_field(), self._zero_field()]
        
        self.state.step_count += 1
        return forcing
    
    def _zero_field(self) -> List[Tensor]:
        """Create a zero QTT field."""
        cores = []
        for i in range(self.n_cores):
            r_l = 1 if i == 0 else 1
            r_r = 1
            core = torch.zeros(r_l, 2, r_r, device=self.device)
            cores.append(core)
        return cores


def estimate_dissipation_rate(
    omega: List[List[Tensor]],
    nu: float,
) -> float:
    """
    Estimate energy dissipation rate from enstrophy.
    
    ε = 2ν * (enstrophy) for incompressible flow
    
    where enstrophy = (1/2) ∫ |ω|² dV
    """
    from ontic.cfd.qtt_turbo import turbo_inner
    
    enstrophy = sum(turbo_inner(w, w).item() for w in omega)
    epsilon = 2 * nu * enstrophy
    
    return epsilon


def compute_taylor_reynolds(
    kinetic_energy: float,
    dissipation_rate: float,
    nu: float,
) -> float:
    """
    Compute Taylor microscale Reynolds number.
    
    Re_λ = u' * λ / ν
    
    where u' = sqrt(2K/3) is RMS velocity
    and λ = sqrt(15ν u'² / ε) is Taylor microscale
    
    Simplifies to: Re_λ = sqrt(20 K² / (3 ν ε))
    """
    if dissipation_rate < 1e-30:
        return 0.0
    
    u_rms = math.sqrt(2 * kinetic_energy / 3)
    lambda_taylor = math.sqrt(15 * nu * u_rms**2 / dissipation_rate)
    Re_lambda = u_rms * lambda_taylor / nu
    
    return Re_lambda


def compute_kolmogorov_scales(
    dissipation_rate: float,
    nu: float,
) -> Tuple[float, float, float]:
    """
    Compute Kolmogorov scales.
    
    Returns (η, τ_η, u_η) where:
    - η = (ν³/ε)^(1/4) is Kolmogorov length
    - τ_η = (ν/ε)^(1/2) is Kolmogorov time
    - u_η = (νε)^(1/4) is Kolmogorov velocity
    """
    if dissipation_rate < 1e-30:
        return 1.0, 1.0, 0.0
    
    eta = (nu**3 / dissipation_rate) ** 0.25
    tau_eta = math.sqrt(nu / dissipation_rate)
    u_eta = (nu * dissipation_rate) ** 0.25
    
    return eta, tau_eta, u_eta


@dataclass
class TurbulenceStats:
    """Turbulence statistics at a given time."""
    time: float
    kinetic_energy: float
    enstrophy: float
    dissipation_rate: float
    taylor_reynolds: float
    kolmogorov_length: float
    kolmogorov_time: float
    integral_length: float
    max_rank: int
    step_time_ms: float


def compute_turbulence_stats(
    omega: List[List[Tensor]],
    u: List[List[Tensor]],
    nu: float,
    t: float,
    step_time_ms: float,
    domain_size: float = 2 * math.pi,
) -> TurbulenceStats:
    """
    Compute comprehensive turbulence statistics.
    """
    from ontic.cfd.qtt_turbo import turbo_inner
    
    # Enstrophy = (1/2) |ω|²
    enstrophy = sum(turbo_inner(w, w).item() for w in omega)
    
    # Kinetic energy = (1/2) |u|²
    kinetic_energy = sum(turbo_inner(v, v).item() for v in u)
    
    # Dissipation rate
    epsilon = estimate_dissipation_rate(omega, nu)
    
    # Taylor Reynolds
    Re_lambda = compute_taylor_reynolds(kinetic_energy, epsilon, nu)
    
    # Kolmogorov scales
    eta, tau_eta, _ = compute_kolmogorov_scales(epsilon, nu)
    
    # Integral length (estimate from energy and dissipation)
    if epsilon > 1e-30:
        u_rms = math.sqrt(2 * kinetic_energy / 3)
        L_int = u_rms**3 / epsilon if epsilon > 0 else domain_size
        L_int = min(L_int, domain_size)  # Can't exceed domain
    else:
        L_int = domain_size
    
    # Max rank
    all_ranks = [c.shape[2] for w in omega for c in w[:-1]]
    max_rank = max(all_ranks) if all_ranks else 0
    
    return TurbulenceStats(
        time=t,
        kinetic_energy=kinetic_energy,
        enstrophy=enstrophy,
        dissipation_rate=epsilon,
        taylor_reynolds=Re_lambda,
        kolmogorov_length=eta,
        kolmogorov_time=tau_eta,
        integral_length=L_int,
        max_rank=max_rank,
        step_time_ms=step_time_ms,
    )
