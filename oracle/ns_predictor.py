#!/usr/bin/env python3
"""
NS PREDICTOR - Navier-Stokes Market Prediction Engine
======================================================

Treats the market as a turbulent fluid and uses physics-based forward 
integration to predict entropy evolution, regime transitions, and 
price distribution dynamics.

PHYSICAL MODEL:
    The market is modeled as a turbulent flow in a 2D phase space:
    - x-axis: Price momentum (returns)
    - y-axis: Volume momentum (participation)
    
    State Variables:
    - ω (vorticity): Market rotation/churn
    - H (entropy): Turbulent kinetic energy
    - ψ (stream function): Price potential field
    - T (temperature): Volatility field
    
    Governing Equations (Navier-Stokes analogy):
    
    1. Vorticity Transport:
       ∂ω/∂t + (u·∇)ω = ν∇²ω + F_liq + F_funding
       
       where:
       - u = velocity from stream function (∂ψ/∂y, -∂ψ/∂x)
       - ν = market viscosity (mean-reversion strength)
       - F_liq = liquidation forcing (energy injection)
       - F_funding = funding pressure gradient
    
    2. Entropy Evolution:
       ∂H/∂t = P - ε + D
       
       where:
       - P = production (from mean flow via regime transition)
       - ε = dissipation (natural decay towards equilibrium)
       - D = diffusion (entropy spreading)
    
    3. Kolmogorov Cascade:
       Energy transfers from large scales (market-wide moves) to small 
       scales (individual liquidations) following:
       
       E(k) ~ k^(-5/3)  (in the inertial subrange)
       
       This gives predictable statistics for price fluctuations.

PREDICTION OUTPUTS:
    1. Entropy trajectory: H(t) for t = [0, 5, 15, 30, 60] minutes
    2. Regime transition probability: P(regime_change | current_state)
    3. Price distribution: μ ± σ cone evolution
    4. Cascade probability: P(liquidation_cascade | H, funding)

Author: Genesis Stack / HyperTensor VM
"""

import asyncio
import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, List, Dict
import os

os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "0")

import numpy as np
import torch
import triton
import triton.language as tl

# =============================================================================
# CONFIGURATION
# =============================================================================

# Physics parameters
VISCOSITY = 0.15              # Market viscosity (mean-reversion strength)
DIFFUSIVITY = 0.08            # Entropy diffusion rate
DISSIPATION_RATE = 0.05       # Natural entropy decay
FORCING_COEFFICIENT = 2.0     # Liquidation forcing strength
PRESSURE_COEFFICIENT = 1.5    # Funding pressure coefficient

# Kolmogorov cascade parameters
KOLMOGOROV_EXPONENT = -5/3    # Energy spectrum exponent
INTEGRAL_SCALE = 1.0          # Large-scale motion (hours)
KOLMOGOROV_SCALE = 0.001      # Smallest scale (seconds)

# Prediction horizons (seconds)
HORIZONS = [60, 300, 900, 1800, 3600]  # 1m, 5m, 15m, 30m, 1h

# Grid resolution for PDE solver
GRID_SIZE = 64                # 64x64 phase space grid
DT = 0.1                      # Time step (normalized)

# Regime thresholds (from galaxy_feed_v3)
REGIME_QUIET = 3.0
REGIME_BUILDING = 5.0
REGIME_VOLATILE = 7.0
REGIME_CHAOTIC = 9.0

# EMA smoothing
EMA_ALPHA = 0.1

# WebSocket symbols (top perpetuals)
TOP_SYMBOLS = [
    "btcusdt", "ethusdt", "solusdt", "xrpusdt", "dogeusdt",
    "bnbusdt", "adausdt", "avaxusdt", "linkusdt", "dotusdt",
    "maticusdt", "arbusdt", "opusdt", "ltcusdt", "bchusdt",
    "aptusdt", "suiusdt", "nearusdt", "atomusdt", "ftmusdt",
]

# =============================================================================
# ENUMS
# =============================================================================

class Regime(Enum):
    QUIET = "QUIET"
    BUILDING = "BUILDING"
    VOLATILE = "VOLATILE"
    CHAOTIC = "CHAOTIC"
    PLASMA = "PLASMA"

class Prediction(Enum):
    ENTROPY_UP = "↑ ENTROPY RISING"
    ENTROPY_DOWN = "↓ ENTROPY FALLING"
    ENTROPY_STABLE = "→ ENTROPY STABLE"
    REGIME_TRANSITION = "⚡ REGIME TRANSITION"
    CASCADE_LIKELY = "🌊 CASCADE LIKELY"


class TradingSignal(Enum):
    """Trading signals generated from physics predictions."""
    NONE = "NONE"
    
    # Momentum signals (entropy rising = trending)
    LONG_MOMENTUM = "🟢 LONG MOMENTUM"
    SHORT_MOMENTUM = "🔴 SHORT MOMENTUM"
    
    # Mean-reversion signals (entropy falling after peak)
    LONG_REVERSION = "🔵 LONG REVERSION"
    SHORT_REVERSION = "🔵 SHORT REVERSION"
    
    # Volatility signals
    VOL_EXPANSION = "📈 VOL EXPANSION"
    VOL_CONTRACTION = "📉 VOL CONTRACTION"
    
    # Risk signals
    REDUCE_EXPOSURE = "⚠️ REDUCE EXPOSURE"
    HEDGE = "🛡️ HEDGE"

# =============================================================================
# TRITON KERNELS - GPU-accelerated PDE solver
# =============================================================================

@triton.jit
def laplacian_kernel(
    field_ptr, output_ptr,
    N: tl.constexpr, dx: tl.constexpr
):
    """Compute ∇²f using 5-point stencil."""
    pid = tl.program_id(0)
    i = pid // N
    j = pid % N
    
    # Load neighbors with periodic BC
    idx = i * N + j
    idx_left = i * N + ((j - 1) % N)
    idx_right = i * N + ((j + 1) % N)
    idx_up = ((i - 1) % N) * N + j
    idx_down = ((i + 1) % N) * N + j
    
    center = tl.load(field_ptr + idx)
    left = tl.load(field_ptr + idx_left)
    right = tl.load(field_ptr + idx_right)
    up = tl.load(field_ptr + idx_up)
    down = tl.load(field_ptr + idx_down)
    
    # 5-point Laplacian
    dx_sq = dx * dx
    laplacian = (left + right + up + down - 4.0 * center) / dx_sq
    
    tl.store(output_ptr + idx, laplacian)


@triton.jit
def advection_kernel(
    field_ptr, u_ptr, v_ptr, output_ptr,
    N: tl.constexpr, dx: tl.constexpr
):
    """Compute (u·∇)f using upwind scheme for stability."""
    pid = tl.program_id(0)
    i = pid // N
    j = pid % N
    idx = i * N + j
    
    # Load velocity at this point
    u = tl.load(u_ptr + idx)
    v = tl.load(v_ptr + idx)
    
    # Load field neighbors
    idx_left = i * N + ((j - 1) % N)
    idx_right = i * N + ((j + 1) % N)
    idx_up = ((i - 1) % N) * N + j
    idx_down = ((i + 1) % N) * N + j
    
    center = tl.load(field_ptr + idx)
    left = tl.load(field_ptr + idx_left)
    right = tl.load(field_ptr + idx_right)
    up = tl.load(field_ptr + idx_up)
    down = tl.load(field_ptr + idx_down)
    
    # Upwind scheme: use backward difference if velocity positive
    # Forward x (j direction)
    df_dx_forward = (right - center) / dx
    df_dx_backward = (center - left) / dx
    df_dx = tl.where(u > 0, df_dx_backward, df_dx_forward)
    
    # Forward y (i direction) 
    df_dy_forward = (down - center) / dx
    df_dy_backward = (center - up) / dx
    df_dy = tl.where(v > 0, df_dy_backward, df_dy_forward)
    
    advection = u * df_dx + v * df_dy
    tl.store(output_ptr + idx, advection)


@triton.jit  
def stream_to_velocity_kernel(
    psi_ptr, u_ptr, v_ptr,
    N: tl.constexpr, dx: tl.constexpr
):
    """Compute velocity from stream function: u = ∂ψ/∂y, v = -∂ψ/∂x."""
    pid = tl.program_id(0)
    i = pid // N
    j = pid % N
    idx = i * N + j
    
    # Load neighbors
    idx_left = i * N + ((j - 1) % N)
    idx_right = i * N + ((j + 1) % N)
    idx_up = ((i - 1) % N) * N + j
    idx_down = ((i + 1) % N) * N + j
    
    psi_left = tl.load(psi_ptr + idx_left)
    psi_right = tl.load(psi_ptr + idx_right)
    psi_up = tl.load(psi_ptr + idx_up)
    psi_down = tl.load(psi_ptr + idx_down)
    
    # Central differences
    u = (psi_down - psi_up) / (2.0 * dx)    # ∂ψ/∂y
    v = -(psi_right - psi_left) / (2.0 * dx)  # -∂ψ/∂x
    
    tl.store(u_ptr + idx, u)
    tl.store(v_ptr + idx, v)


@triton.jit
def poisson_jacobi_kernel(
    omega_ptr, psi_ptr, psi_new_ptr,
    N: tl.constexpr, dx: tl.constexpr
):
    """One Jacobi iteration for Poisson equation: ∇²ψ = -ω."""
    pid = tl.program_id(0)
    i = pid // N
    j = pid % N
    idx = i * N + j
    
    # Load neighbors
    idx_left = i * N + ((j - 1) % N)
    idx_right = i * N + ((j + 1) % N)
    idx_up = ((i - 1) % N) * N + j
    idx_down = ((i + 1) % N) * N + j
    
    psi_left = tl.load(psi_ptr + idx_left)
    psi_right = tl.load(psi_ptr + idx_right)
    psi_up = tl.load(psi_ptr + idx_up)
    psi_down = tl.load(psi_ptr + idx_down)
    omega = tl.load(omega_ptr + idx)
    
    dx_sq = dx * dx
    psi_new = 0.25 * (psi_left + psi_right + psi_up + psi_down + dx_sq * omega)
    
    tl.store(psi_new_ptr + idx, psi_new)


# =============================================================================
# NAVIER-STOKES SOLVER
# =============================================================================

class NavierStokesSolver:
    """
    2D Vorticity-Stream Function formulation of Navier-Stokes.
    
    Solves:
        ∂ω/∂t + (u·∇)ω = ν∇²ω + F
        ∇²ψ = -ω
        u = ∂ψ/∂y, v = -∂ψ/∂x
    """
    
    def __init__(
        self,
        N: int = GRID_SIZE,
        viscosity: float = VISCOSITY,
        dt: float = DT,
        device: torch.device = None
    ):
        self.N = N
        self.viscosity = viscosity
        self.dt = dt
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Grid spacing (normalized to [0, 2π])
        self.dx = 2 * math.pi / N
        
        # State fields
        self.omega = torch.zeros(N * N, dtype=torch.float32, device=self.device)  # Vorticity
        self.psi = torch.zeros(N * N, dtype=torch.float32, device=self.device)    # Stream function
        self.u = torch.zeros(N * N, dtype=torch.float32, device=self.device)      # x-velocity
        self.v = torch.zeros(N * N, dtype=torch.float32, device=self.device)      # y-velocity
        
        # Work arrays
        self.laplacian = torch.zeros_like(self.omega)
        self.advection = torch.zeros_like(self.omega)
        self.psi_new = torch.zeros_like(self.psi)
        self.forcing = torch.zeros_like(self.omega)
        
        # History for prediction
        self.omega_history = deque(maxlen=100)
        self.energy_history = deque(maxlen=100)
        
    def set_initial_condition(self, omega_init: torch.Tensor) -> None:
        """Set initial vorticity field."""
        self.omega = omega_init.flatten().to(self.device)
        self._solve_poisson()
        self._compute_velocity()
    
    def _solve_poisson(self, n_iters: int = 50) -> None:
        """Solve ∇²ψ = -ω using Jacobi iteration."""
        N = self.N
        dx = self.dx
        
        for _ in range(n_iters):
            poisson_jacobi_kernel[(N * N,)](
                self.omega, self.psi, self.psi_new,
                N=N, dx=dx
            )
            self.psi, self.psi_new = self.psi_new, self.psi
    
    def _compute_velocity(self) -> None:
        """Compute velocity from stream function."""
        N = self.N
        dx = self.dx
        
        stream_to_velocity_kernel[(N * N,)](
            self.psi, self.u, self.v,
            N=N, dx=dx
        )
    
    def _compute_laplacian(self) -> None:
        """Compute ∇²ω."""
        N = self.N
        dx = self.dx
        
        laplacian_kernel[(N * N,)](
            self.omega, self.laplacian,
            N=N, dx=dx
        )
    
    def _compute_advection(self) -> None:
        """Compute (u·∇)ω."""
        N = self.N
        dx = self.dx
        
        advection_kernel[(N * N,)](
            self.omega, self.u, self.v, self.advection,
            N=N, dx=dx
        )
    
    def step(self, forcing: Optional[torch.Tensor] = None) -> float:
        """
        Advance one time step using RK2 (midpoint method).
        
        Returns:
            Total kinetic energy (enstrophy)
        """
        # Store forcing
        if forcing is not None:
            self.forcing = forcing.flatten().to(self.device)
        else:
            self.forcing.zero_()
        
        # RK2 Step 1: Compute k1 = f(ω_n)
        self._compute_velocity()
        self._compute_laplacian()
        self._compute_advection()
        
        k1 = (
            -self.advection 
            + self.viscosity * self.laplacian 
            + self.forcing
        )
        
        # Midpoint: ω_mid = ω_n + 0.5 * dt * k1
        omega_mid = self.omega + 0.5 * self.dt * k1
        omega_backup = self.omega.clone()
        self.omega = omega_mid
        
        # RK2 Step 2: Compute k2 = f(ω_mid)
        self._solve_poisson()
        self._compute_velocity()
        self._compute_laplacian()
        self._compute_advection()
        
        k2 = (
            -self.advection 
            + self.viscosity * self.laplacian 
            + self.forcing
        )
        
        # Full step: ω_{n+1} = ω_n + dt * k2
        self.omega = omega_backup + self.dt * k2
        
        # Update stream function and velocity
        self._solve_poisson()
        self._compute_velocity()
        
        # Compute enstrophy (total vorticity squared)
        enstrophy = 0.5 * (self.omega ** 2).sum().item() * self.dx ** 2
        
        # Store history
        self.omega_history.append(self.omega.clone())
        self.energy_history.append(enstrophy)
        
        return enstrophy
    
    def get_kinetic_energy(self) -> float:
        """Compute total kinetic energy."""
        return 0.5 * ((self.u ** 2 + self.v ** 2).sum().item()) * self.dx ** 2
    
    def get_entropy(self) -> float:
        """
        Compute flow entropy (turbulent kinetic energy proxy).
        
        Uses velocity gradient tensor to estimate local dissipation.
        """
        # Reshape to 2D
        N = self.N
        u_2d = self.u.view(N, N)
        v_2d = self.v.view(N, N)
        
        # Compute strain rate tensor components
        # S_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)
        du_dx = torch.roll(u_2d, -1, dims=1) - torch.roll(u_2d, 1, dims=1)
        du_dy = torch.roll(u_2d, -1, dims=0) - torch.roll(u_2d, 1, dims=0)
        dv_dx = torch.roll(v_2d, -1, dims=1) - torch.roll(v_2d, 1, dims=1)
        dv_dy = torch.roll(v_2d, -1, dims=0) - torch.roll(v_2d, 1, dims=0)
        
        du_dx = du_dx / (2 * self.dx)
        du_dy = du_dy / (2 * self.dx)
        dv_dx = dv_dx / (2 * self.dx)
        dv_dy = dv_dy / (2 * self.dx)
        
        # Strain rate magnitude: S = sqrt(2 * S_ij * S_ij)
        S_xx = du_dx
        S_yy = dv_dy
        S_xy = 0.5 * (du_dy + dv_dx)
        
        strain_sq = 2 * (S_xx**2 + S_yy**2 + 2 * S_xy**2)
        
        # Entropy ~ log of mean strain rate (normalized to 0-10 range)
        mean_strain = torch.sqrt(strain_sq.mean() + 1e-9)
        entropy = 3.0 + 2.0 * torch.log10(mean_strain + 0.01)
        
        return max(0.0, min(10.0, entropy.item()))


# =============================================================================
# MARKET STATE ENCODER
# =============================================================================

@dataclass
class MarketState:
    """Encodes market data into phase space for NS solver."""
    
    # Price dynamics per symbol
    returns: Dict[str, deque] = field(default_factory=lambda: {})
    volumes: Dict[str, deque] = field(default_factory=lambda: {})
    
    # Aggregate metrics
    entropy_raw: float = 0.0
    entropy_smooth: float = 0.0
    entropy_derivative: float = 0.0
    
    funding_rate: float = 0.0
    long_liqs: float = 0.0
    short_liqs: float = 0.0
    
    # Reference prices
    ref_prices: Dict[str, float] = field(default_factory=dict)
    
    # History
    entropy_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    funding_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    liq_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    buffer_size: int = 256
    
    def ingest_trade(self, symbol: str, price: float, qty: float, is_sell: bool) -> None:
        """Add trade and compute returns/volume."""
        if symbol not in self.returns:
            self.returns[symbol] = deque(maxlen=self.buffer_size)
            self.volumes[symbol] = deque(maxlen=self.buffer_size)
        
        # Initialize reference
        if symbol not in self.ref_prices:
            self.ref_prices[symbol] = price
        
        # Compute return (basis points)
        ref = self.ref_prices[symbol]
        ret = ((price / ref) - 1.0) * 10000  # basis points
        
        # Signed volume
        sign = -1.0 if is_sell else 1.0
        vol = sign * qty * price / 10000  # Scale to reasonable range
        
        self.returns[symbol].append(ret)
        self.volumes[symbol].append(vol)
        
        # Update reference slowly
        self.ref_prices[symbol] = ref * 0.99 + price * 0.01
    
    def update_funding(self, rate: float) -> None:
        """Update aggregate funding rate."""
        self.funding_rate = rate
        self.funding_history.append((time.time(), rate))
    
    def update_liquidations(self, long_total: float, short_total: float) -> None:
        """Update liquidation totals."""
        self.long_liqs = long_total
        self.short_liqs = short_total
        self.liq_history.append((time.time(), long_total, short_total))
    
    def update_entropy(self, raw: float, smooth: float, derivative: float) -> None:
        """Update entropy metrics."""
        self.entropy_raw = raw
        self.entropy_smooth = smooth
        self.entropy_derivative = derivative
        self.entropy_history.append((time.time(), raw, smooth, derivative))
    
    def encode_phase_space(self, N: int = GRID_SIZE) -> torch.Tensor:
        """
        Encode market state into 2D vorticity field for NS solver.
        
        Phase space axes:
        - x: Price momentum (returns distribution)
        - y: Volume momentum (participation distribution)
        
        Returns:
            Vorticity field (N×N)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        omega = torch.zeros(N, N, dtype=torch.float32, device=device)
        
        # Gather all returns and volumes
        all_returns = []
        all_volumes = []
        for symbol in self.returns:
            all_returns.extend(list(self.returns[symbol]))
            all_volumes.extend(list(self.volumes[symbol]))
        
        if len(all_returns) < 10:
            return omega
        
        returns = torch.tensor(all_returns[-1000:], dtype=torch.float32, device=device)
        volumes = torch.tensor(all_volumes[-1000:], dtype=torch.float32, device=device)
        
        # Normalize to grid coordinates
        ret_min, ret_max = returns.min(), returns.max()
        vol_min, vol_max = volumes.min(), volumes.max()
        
        # Avoid division by zero
        ret_range = max(ret_max - ret_min, 1.0)
        vol_range = max(vol_max - vol_min, 1.0)
        
        ret_norm = ((returns - ret_min) / ret_range * (N - 1)).long().clamp(0, N - 1)
        vol_norm = ((volumes - vol_min) / vol_range * (N - 1)).long().clamp(0, N - 1)
        
        # Build vorticity from trade distribution
        # Each trade adds positive or negative vorticity based on direction
        for i in range(len(returns)):
            x = ret_norm[i].item()
            y = vol_norm[i].item()
            sign = 1.0 if volumes[i] > 0 else -1.0
            omega[y, x] += sign * abs(volumes[i]) / 100
        
        # Apply Gaussian smoothing
        omega = omega.unsqueeze(0).unsqueeze(0)
        kernel_size = 5
        sigma = 1.0
        kernel = self._gaussian_kernel(kernel_size, sigma, device)
        omega = torch.nn.functional.conv2d(omega, kernel, padding=kernel_size // 2)
        omega = omega.squeeze()
        
        # Add forcing from funding and liquidations
        cx, cy = N // 2, N // 2
        
        # Funding creates a dipole (pressure gradient)
        funding_strength = self.funding_rate * PRESSURE_COEFFICIENT * 1000
        omega[cy - 5:cy, cx - 10:cx + 10] += funding_strength
        omega[cy:cy + 5, cx - 10:cx + 10] -= funding_strength
        
        # Liquidations inject energy at the edges
        liq_imbalance = (self.short_liqs - self.long_liqs) / max(self.short_liqs + self.long_liqs, 1e6)
        liq_strength = liq_imbalance * FORCING_COEFFICIENT
        omega[:10, :] += liq_strength
        omega[-10:, :] -= liq_strength
        
        return omega
    
    @staticmethod
    def _gaussian_kernel(size: int, sigma: float, device: torch.device) -> torch.Tensor:
        """Create 2D Gaussian kernel for smoothing."""
        x = torch.arange(size, device=device) - size // 2
        x = x.float()
        gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
        gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
        gauss_2d = gauss_2d / gauss_2d.sum()
        return gauss_2d.unsqueeze(0).unsqueeze(0)
    
    def get_regime(self) -> Regime:
        """Classify current regime."""
        H = self.entropy_smooth
        if H < REGIME_QUIET:
            return Regime.QUIET
        elif H < REGIME_BUILDING:
            return Regime.BUILDING
        elif H < REGIME_VOLATILE:
            return Regime.VOLATILE
        elif H < REGIME_CHAOTIC:
            return Regime.CHAOTIC
        else:
            return Regime.PLASMA


# =============================================================================
# PREDICTION ENGINE
# =============================================================================

@dataclass
class PredictionResult:
    """Container for prediction outputs."""
    
    horizon_seconds: int
    
    # Entropy prediction
    entropy_current: float
    entropy_predicted: float
    entropy_confidence: float
    
    # Regime prediction
    regime_current: Regime
    regime_predicted: Regime
    regime_transition_prob: float
    
    # Price distribution (relative to current)
    price_mean_pct: float       # Expected price change %
    price_std_pct: float        # 1σ price range %
    
    # Risk metrics
    cascade_probability: float  # P(liquidation cascade)
    volatility_multiplier: float  # Expected vol vs current
    
    def __repr__(self) -> str:
        direction = "↑" if self.entropy_predicted > self.entropy_current else "↓"
        return (
            f"[{self.horizon_seconds}s] H: {self.entropy_current:.2f} → {self.entropy_predicted:.2f} {direction} "
            f"({self.entropy_confidence:.0%} conf) | "
            f"Regime: {self.regime_current.value}→{self.regime_predicted.value} "
            f"(P_trans={self.regime_transition_prob:.1%}) | "
            f"Price: {self.price_mean_pct:+.2f}% ± {self.price_std_pct:.2f}% | "
            f"Cascade: {self.cascade_probability:.1%}"
        )


class NSPredictor:
    """
    Physics-based market prediction using Navier-Stokes dynamics.
    
    Core idea: Market entropy follows turbulent flow physics.
    We can integrate forward in time to predict where entropy is going.
    """
    
    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        horizons: List[int] = None
    ):
        self.grid_size = grid_size
        self.horizons = horizons or HORIZONS
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # NS solver for forward integration
        self.solver = NavierStokesSolver(
            N=grid_size,
            viscosity=VISCOSITY,
            dt=DT,
            device=self.device
        )
        
        # Market state encoder
        self.state = MarketState()
        
        # Calibration (learned from history)
        self.entropy_baseline = 4.0
        self.entropy_scale = 2.0
        self.volatility_baseline = 0.01  # 1% daily
        
        # Prediction tracking for validation
        # Format: {prediction_id: (timestamp_made, horizon_seconds, predicted_entropy, expires_at)}
        self.pending_predictions: Dict[str, Tuple[float, int, float, float]] = {}
        self.validated_predictions: List[Dict] = []  # Completed predictions with actuals
        self.prediction_counter = 0
        
        # Accuracy tracking per horizon
        self.accuracy_stats: Dict[int, Dict] = {
            h: {"count": 0, "total_error": 0.0, "total_sq_error": 0.0, "correct_direction": 0}
            for h in (horizons or HORIZONS)
        }
        
        print(f"[NSPredictor] Initialized on {self.device}")
        print(f"[NSPredictor] Grid: {grid_size}x{grid_size}, Horizons: {horizons}")
    
    def ingest_trade(self, symbol: str, price: float, qty: float, is_sell: bool) -> None:
        """Forward trade to market state encoder."""
        self.state.ingest_trade(symbol, price, qty, is_sell)
    
    def update_funding(self, rate: float) -> None:
        """Update funding rate."""
        self.state.update_funding(rate)
    
    def update_liquidations(self, long_total: float, short_total: float) -> None:
        """Update liquidation totals."""
        self.state.update_liquidations(long_total, short_total)
    
    def update_entropy(self, raw: float, smooth: float, derivative: float) -> None:
        """Update entropy from external computation."""
        self.state.update_entropy(raw, smooth, derivative)
        
        # Recalibrate baseline slowly
        self.entropy_baseline = 0.99 * self.entropy_baseline + 0.01 * smooth
    
    def predict(self) -> List[PredictionResult]:
        """
        Generate predictions for all horizons.
        
        Uses NS forward integration + statistical modeling.
        Physics-calibrated to produce realistic entropy forecasts.
        """
        results = []
        
        # Encode current state to vorticity field
        omega_init = self.state.encode_phase_space(self.grid_size)
        self.solver.set_initial_condition(omega_init)
        
        current_entropy = self.state.entropy_smooth
        current_regime = self.state.get_regime()
        current_derivative = self.state.entropy_derivative
        
        # Forward integrate for each horizon
        for horizon in self.horizons:
            # Number of NS steps - scale appropriately
            # Shorter horizons = fewer steps, more weight on current state
            normalized_time = horizon / 3600 * 2 * math.pi
            n_steps = max(1, min(100, int(normalized_time / self.solver.dt)))
            
            # Run forward integration
            energies = []
            for _ in range(n_steps):
                energy = self.solver.step()
                energies.append(energy)
            
            # Get NS-derived entropy change (scaled appropriately)
            ns_entropy = self.solver.get_entropy()
            
            # The key insight: NS gives us the DIRECTION and MAGNITUDE of change
            # but we anchor to current entropy
            ns_delta = (ns_entropy - self.entropy_baseline) * 0.3  # Dampen NS contribution
            
            # Ornstein-Uhlenbeck mean-reversion model
            # dH = θ(μ - H)dt + σdW
            theta = 0.1  # Mean-reversion speed
            mu = self.entropy_baseline  # Long-term mean
            sigma = 0.5  # Volatility
            
            # Expected value under OU process
            time_hours = horizon / 3600
            ou_factor = math.exp(-theta * time_hours)
            ou_mean = current_entropy * ou_factor + mu * (1 - ou_factor)
            
            # Add momentum from current derivative
            momentum_contrib = current_derivative * min(horizon / 60, 5.0) * 0.3
            
            # Add NS forcing (small contribution)
            ns_contrib = ns_delta * (1 - ou_factor) * 0.2
            
            # Combine: OU mean-reversion + momentum + NS forcing
            predicted_entropy = ou_mean + momentum_contrib + ns_contrib
            
            # Clamp to realistic range
            predicted_entropy = max(0.0, min(10.0, predicted_entropy))
            
            # Confidence (decreases with horizon)
            confidence = math.exp(-horizon / 1800)
            
            # Regime prediction
            predicted_regime = self._entropy_to_regime(predicted_entropy)
            transition_prob = self._regime_transition_probability(
                current_regime, predicted_regime, current_entropy, predicted_entropy
            )
            
            # Price distribution (Kolmogorov scaling)
            # Price variance ~ t^(2/3) in turbulent regime
            time_factor = (horizon / 60) ** (2/3)
            entropy_factor = predicted_entropy / self.entropy_baseline
            
            price_std = self.volatility_baseline * time_factor * entropy_factor
            price_mean = self._expected_drift(current_regime, predicted_regime)
            
            # Cascade probability
            cascade_prob = self._cascade_probability(
                predicted_entropy, self.state.funding_rate,
                self.state.long_liqs, self.state.short_liqs
            )
            
            # Volatility multiplier
            vol_mult = entropy_factor * (1 + 0.5 * transition_prob)
            
            result = PredictionResult(
                horizon_seconds=horizon,
                entropy_current=current_entropy,
                entropy_predicted=predicted_entropy,
                entropy_confidence=confidence,
                regime_current=current_regime,
                regime_predicted=predicted_regime,
                regime_transition_prob=transition_prob,
                price_mean_pct=price_mean * 100,
                price_std_pct=price_std * 100,
                cascade_probability=cascade_prob,
                volatility_multiplier=vol_mult
            )
            
            results.append(result)
            
            # Track this prediction for validation
            self.prediction_counter += 1
            pred_id = f"pred_{self.prediction_counter}_{horizon}"
            now = time.time()
            self.pending_predictions[pred_id] = (
                now,                          # when prediction was made
                horizon,                      # horizon in seconds
                predicted_entropy,            # what we predicted
                now + horizon,                # when it expires
                current_entropy,              # entropy at prediction time
                current_derivative            # derivative at prediction time
            )
            
            # Reset solver for next horizon (fresh integration)
            self.solver.set_initial_condition(omega_init)
        
        return results
    
    def validate_predictions(self) -> List[Dict]:
        """
        Check all pending predictions that have expired and validate against actual entropy.
        
        Returns list of newly validated predictions.
        """
        now = time.time()
        current_entropy = self.state.entropy_smooth
        newly_validated = []
        expired_ids = []
        
        for pred_id, (made_at, horizon, predicted, expires_at, initial_entropy, initial_derivative) in self.pending_predictions.items():
            if now >= expires_at:
                # This prediction has expired - validate it
                actual_entropy = current_entropy
                error = predicted - actual_entropy
                abs_error = abs(error)
                
                # Direction accuracy: did we correctly predict up/down/stable?
                predicted_direction = "up" if predicted > initial_entropy + 0.1 else ("down" if predicted < initial_entropy - 0.1 else "stable")
                actual_direction = "up" if actual_entropy > initial_entropy + 0.1 else ("down" if actual_entropy < initial_entropy - 0.1 else "stable")
                direction_correct = predicted_direction == actual_direction
                
                validation = {
                    "pred_id": pred_id,
                    "horizon": horizon,
                    "made_at": made_at,
                    "expires_at": expires_at,
                    "initial_entropy": initial_entropy,
                    "predicted_entropy": predicted,
                    "actual_entropy": actual_entropy,
                    "error": error,
                    "abs_error": abs_error,
                    "predicted_direction": predicted_direction,
                    "actual_direction": actual_direction,
                    "direction_correct": direction_correct,
                }
                
                self.validated_predictions.append(validation)
                newly_validated.append(validation)
                expired_ids.append(pred_id)
                
                # Update accuracy stats
                stats = self.accuracy_stats[horizon]
                stats["count"] += 1
                stats["total_error"] += abs_error
                stats["total_sq_error"] += abs_error ** 2
                if direction_correct:
                    stats["correct_direction"] += 1
        
        # Remove expired predictions
        for pred_id in expired_ids:
            del self.pending_predictions[pred_id]
        
        return newly_validated
    
    def get_accuracy_report(self) -> Dict:
        """
        Get accuracy statistics per horizon.
        
        Returns dict with MAE, RMSE, direction accuracy for each horizon.
        """
        report = {}
        for horizon, stats in self.accuracy_stats.items():
            if stats["count"] > 0:
                mae = stats["total_error"] / stats["count"]
                rmse = math.sqrt(stats["total_sq_error"] / stats["count"])
                direction_acc = stats["correct_direction"] / stats["count"]
                report[horizon] = {
                    "count": stats["count"],
                    "mae": mae,
                    "rmse": rmse,
                    "direction_accuracy": direction_acc
                }
            else:
                report[horizon] = {"count": 0, "mae": None, "rmse": None, "direction_accuracy": None}
        return report
        
        return results
    
    def _entropy_to_regime(self, entropy: float) -> Regime:
        """Convert entropy value to regime."""
        if entropy < REGIME_QUIET:
            return Regime.QUIET
        elif entropy < REGIME_BUILDING:
            return Regime.BUILDING
        elif entropy < REGIME_VOLATILE:
            return Regime.VOLATILE
        elif entropy < REGIME_CHAOTIC:
            return Regime.CHAOTIC
        else:
            return Regime.PLASMA
    
    def _regime_transition_probability(
        self,
        current: Regime,
        predicted: Regime,
        current_entropy: float,
        predicted_entropy: float
    ) -> float:
        """Estimate probability of regime transition."""
        if current == predicted:
            return 0.0
        
        # Distance from threshold
        thresholds = [REGIME_QUIET, REGIME_BUILDING, REGIME_VOLATILE, REGIME_CHAOTIC]
        
        min_distance = float('inf')
        for thresh in thresholds:
            dist = abs(current_entropy - thresh)
            min_distance = min(min_distance, dist)
        
        # Probability increases as we approach threshold
        prob = math.exp(-min_distance / 0.5)
        
        # Increase if entropy is changing rapidly
        delta = abs(predicted_entropy - current_entropy)
        prob = min(1.0, prob * (1 + delta / 2))
        
        return prob
    
    def _expected_drift(self, current_regime: Regime, predicted_regime: Regime) -> float:
        """Estimate expected price drift."""
        # In volatile/chaotic regimes, expect larger moves
        regime_drift = {
            Regime.QUIET: 0.0,
            Regime.BUILDING: 0.0001,
            Regime.VOLATILE: 0.0002,
            Regime.CHAOTIC: 0.0005,
            Regime.PLASMA: 0.001
        }
        
        # Use predicted regime
        base_drift = regime_drift.get(predicted_regime, 0.0)
        
        # Add funding-based bias
        funding_drift = -self.state.funding_rate * 100  # Counter-trade
        
        return base_drift + funding_drift
    
    def _cascade_probability(
        self,
        entropy: float,
        funding: float,
        long_liqs: float,
        short_liqs: float
    ) -> float:
        """Estimate probability of liquidation cascade."""
        # Base probability from entropy
        if entropy < REGIME_VOLATILE:
            base_prob = 0.01
        elif entropy < REGIME_CHAOTIC:
            base_prob = 0.05
        else:
            base_prob = 0.15
        
        # Increase for extreme funding
        funding_factor = 1 + abs(funding) * 1000
        
        # Increase for existing liquidation activity
        liq_total = long_liqs + short_liqs
        liq_factor = 1 + liq_total / 10_000_000  # Normalize by $10M
        
        prob = base_prob * funding_factor * liq_factor
        return min(0.95, prob)
    
    def generate_signal(self, predictions: List[PredictionResult]) -> Tuple[TradingSignal, dict]:
        """
        Generate trading signal from predictions.
        
        Uses multi-horizon analysis:
        - Short horizon (1-5m): Momentum detection
        - Medium horizon (15-30m): Trend confirmation
        - Long horizon (60m): Mean-reversion/regime check
        
        Returns:
            (signal, metadata)
        """
        if not predictions:
            return TradingSignal.NONE, {}
        
        # Get predictions by horizon
        preds_by_horizon = {p.horizon_seconds: p for p in predictions}
        
        # Current state
        current_entropy = self.state.entropy_smooth
        current_derivative = self.state.entropy_derivative
        funding_rate = self.state.funding_rate
        long_liqs, short_liqs = self.state.long_liqs, self.state.short_liqs
        current_regime = self.state.get_regime()
        
        signal = TradingSignal.NONE
        confidence = 0.0
        reason = ""
        
        # Get key predictions
        pred_1m = preds_by_horizon.get(60)
        pred_5m = preds_by_horizon.get(300)
        pred_15m = preds_by_horizon.get(900)
        pred_30m = preds_by_horizon.get(1800)
        pred_60m = preds_by_horizon.get(3600)
        
        # =====================================================================
        # RISK SIGNALS (highest priority)
        # =====================================================================
        
        # Check for cascade risk
        max_cascade_prob = max(p.cascade_probability for p in predictions)
        if max_cascade_prob > 0.3:
            signal = TradingSignal.REDUCE_EXPOSURE
            confidence = max_cascade_prob
            reason = f"Cascade probability {max_cascade_prob:.0%}"
        
        # Check for extreme volatility expansion
        if pred_5m and pred_5m.volatility_multiplier > 2.0:
            if current_regime in (Regime.CHAOTIC, Regime.PLASMA):
                signal = TradingSignal.HEDGE
                confidence = pred_5m.volatility_multiplier / 3
                reason = f"Vol multiplier {pred_5m.volatility_multiplier:.1f}x in {current_regime.value}"
        
        # =====================================================================
        # VOLATILITY SIGNALS
        # =====================================================================
        
        if signal == TradingSignal.NONE:
            # Volatility expansion: entropy rising through regimes
            if pred_5m and pred_15m:
                if (pred_5m.entropy_predicted > current_entropy + 0.5 and
                    pred_15m.entropy_predicted > pred_5m.entropy_predicted):
                    signal = TradingSignal.VOL_EXPANSION
                    confidence = (pred_15m.entropy_predicted - current_entropy) / 3
                    reason = f"H rising: {current_entropy:.1f}→{pred_15m.entropy_predicted:.1f}"
            
            # Volatility contraction: entropy falling
            if pred_15m and pred_30m:
                if (pred_15m.entropy_predicted < current_entropy - 0.5 and
                    pred_30m.entropy_predicted < pred_15m.entropy_predicted):
                    signal = TradingSignal.VOL_CONTRACTION
                    confidence = (current_entropy - pred_30m.entropy_predicted) / 3
                    reason = f"H falling: {current_entropy:.1f}→{pred_30m.entropy_predicted:.1f}"
        
        # =====================================================================
        # MOMENTUM SIGNALS
        # =====================================================================
        
        if signal == TradingSignal.NONE:
            # Strong momentum when entropy rising AND flow imbalanced
            if current_derivative > 0.2 and current_regime in (Regime.BUILDING, Regime.VOLATILE):
                # Funding tells us direction
                if funding_rate < -0.0001:  # Shorts paying → likely short squeeze
                    signal = TradingSignal.LONG_MOMENTUM
                    confidence = min(1.0, abs(funding_rate) * 5000 + current_derivative)
                    reason = f"Short funding {funding_rate*100:.3f}%, rising entropy"
                elif funding_rate > 0.0001:  # Longs paying → likely long squeeze
                    signal = TradingSignal.SHORT_MOMENTUM
                    confidence = min(1.0, abs(funding_rate) * 5000 + current_derivative)
                    reason = f"Long funding {funding_rate*100:.3f}%, rising entropy"
        
        # =====================================================================
        # MEAN-REVERSION SIGNALS
        # =====================================================================
        
        if signal == TradingSignal.NONE:
            # Mean reversion after volatility spike
            if (current_regime in (Regime.VOLATILE, Regime.CHAOTIC) and
                current_derivative < -0.2 and
                pred_30m and pred_30m.entropy_predicted < REGIME_VOLATILE):
                
                # Fading the move direction
                if funding_rate > 0.0002:  # Longs overextended
                    signal = TradingSignal.SHORT_REVERSION
                    confidence = abs(current_derivative) * abs(funding_rate) * 1000
                    reason = f"Vol fading, longs overextended"
                elif funding_rate < -0.0002:  # Shorts overextended
                    signal = TradingSignal.LONG_REVERSION
                    confidence = abs(current_derivative) * abs(funding_rate) * 1000
                    reason = f"Vol fading, shorts overextended"
        
        # Clamp confidence
        confidence = min(1.0, max(0.0, confidence))
        
        metadata = {
            "signal": signal.value,
            "confidence": confidence,
            "reason": reason,
            "current_entropy": current_entropy,
            "current_derivative": current_derivative,
            "current_regime": current_regime.value,
            "funding_rate": funding_rate,
            "predictions": {p.horizon_seconds: {
                "entropy": p.entropy_predicted,
                "regime": p.regime_predicted.value,
                "cascade_prob": p.cascade_probability
            } for p in predictions}
        }
        
        return signal, metadata


# =============================================================================
# INTEGRATED PREDICTOR (connects to galaxy_feed streams)
# =============================================================================

class GalaxyPredictor:
    """
    Full predictor system integrating with live Binance streams.
    
    Combines:
    - Galaxy Feed entropy calculation
    - NS Predictor forward integration
    - Real-time prediction output
    """
    
    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or TOP_SYMBOLS
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize predictor
        self.predictor = NSPredictor(
            grid_size=GRID_SIZE,
            horizons=HORIZONS
        )
        
        # Entropy calculation (from galaxy_feed_v3)
        self.prices = {}
        self.volumes = {}
        self.sides = {}
        self.ref_prices = {}
        
        self.smoothed_entropy = 0.0
        self.prev_entropy = 0.0
        
        # Funding/liquidation tracking
        self.funding_rates = {}
        self.long_liqs = []
        self.short_liqs = []
        
        # Stats
        self.trade_count = 0
        self.start_time = time.time()
        self.latest_prices = {}
        
        # URLs
        trade_streams = "/".join([f"{s}@aggTrade" for s in self.symbols])
        self.trade_url = f"wss://fstream.binance.com/stream?streams={trade_streams}"
        self.liq_url = "wss://fstream.binance.com/ws/!forceOrder@arr"
        self.mark_url = "wss://fstream.binance.com/ws/!markPrice@arr"
    
    def _compute_entropy(self) -> Tuple[float, float, float]:
        """Compute entropy (mirroring galaxy_feed_v3 logic)."""
        all_prices = []
        all_volumes = []
        all_sides = []
        
        for symbol in self.prices:
            all_prices.extend(list(self.prices[symbol])[-256:])
            all_volumes.extend(list(self.volumes[symbol])[-256:])
            all_sides.extend(list(self.sides[symbol])[-256:])
        
        if len(all_prices) < 32:
            return 0.0, self.smoothed_entropy, 0.0
        
        prices_t = torch.tensor(all_prices[-4096:], dtype=torch.float32, device=self.device)
        volumes_t = torch.tensor(all_volumes[-4096:], dtype=torch.float32, device=self.device)
        sides_t = torch.tensor(all_sides[-4096:], dtype=torch.float32, device=self.device)
        
        # Compute entropy components
        # Volatility (price std in basis points)
        price_std = prices_t.std().item()
        vol_component = price_std * 0.2
        
        # Volume distribution entropy
        vol_sum = volumes_t.sum()
        if vol_sum > 0:
            vol_norm = volumes_t / vol_sum
            vol_entropy = -(vol_norm * torch.log(vol_norm + 1e-9)).sum().item()
            max_entropy = math.log(len(volumes_t))
            norm_vol_entropy = vol_entropy / max(max_entropy, 1.0)
        else:
            norm_vol_entropy = 0.0
        
        # Flow imbalance
        buy_vol = (volumes_t * (1.0 - sides_t)).sum()
        sell_vol = (volumes_t * sides_t).sum()
        total = buy_vol + sell_vol + 1e-9
        imbalance = abs(buy_vol - sell_vol) / total
        
        raw_entropy = vol_component + norm_vol_entropy * 5.0 + imbalance.item() * 3.0
        
        # EMA smoothing
        self.prev_entropy = self.smoothed_entropy
        self.smoothed_entropy = EMA_ALPHA * raw_entropy + (1 - EMA_ALPHA) * self.smoothed_entropy
        
        derivative = self.smoothed_entropy - self.prev_entropy
        
        return raw_entropy, self.smoothed_entropy, derivative
    
    def _get_weighted_funding(self) -> float:
        """Get weighted funding rate."""
        weights = {
            "BTCUSDT": 0.35, "ETHUSDT": 0.25, "SOLUSDT": 0.15,
            "XRPUSDT": 0.10, "DOGEUSDT": 0.05
        }
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for symbol, weight in weights.items():
            if symbol in self.funding_rates:
                weighted_sum += self.funding_rates[symbol] * weight
                total_weight += weight
        
        return weighted_sum / max(total_weight, 1e-9)
    
    def _get_liq_totals(self) -> Tuple[float, float]:
        """Get liquidation totals in rolling window."""
        cutoff = time.time() - 60
        
        self.long_liqs = [(t, v) for t, v in self.long_liqs if t > cutoff]
        self.short_liqs = [(t, v) for t, v in self.short_liqs if t > cutoff]
        
        long_total = sum(v for _, v in self.long_liqs)
        short_total = sum(v for _, v in self.short_liqs)
        
        return long_total, short_total
    
    async def _trade_handler(self):
        """Handle aggTrade stream."""
        import websockets
        
        while True:
            try:
                async with websockets.connect(self.trade_url) as ws:
                    while True:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        
                        if "data" in data:
                            trade = data["data"]
                        else:
                            trade = data
                        
                        if trade.get("e") != "aggTrade":
                            continue
                        
                        symbol = trade["s"]
                        price = float(trade["p"])
                        qty = float(trade["q"])
                        is_sell = trade["m"]
                        
                        # Local entropy calculation
                        if symbol not in self.ref_prices:
                            self.ref_prices[symbol] = price
                            self.prices[symbol] = deque(maxlen=4096)
                            self.volumes[symbol] = deque(maxlen=4096)
                            self.sides[symbol] = deque(maxlen=4096)
                        
                        ref = self.ref_prices[symbol]
                        norm_price = ((price / ref) - 1.0) * 10000
                        log_vol = min(qty * price, 1_000_000) / 1000
                        
                        self.prices[symbol].append(norm_price)
                        self.volumes[symbol].append(log_vol)
                        self.sides[symbol].append(1.0 if is_sell else 0.0)
                        
                        self.ref_prices[symbol] = ref * 0.99 + price * 0.01
                        
                        # Forward to predictor
                        self.predictor.ingest_trade(symbol, price, qty, is_sell)
                        
                        self.trade_count += 1
                        self.latest_prices[symbol] = price
                        
            except Exception as e:
                print(f"Trade stream error: {e}")
                await asyncio.sleep(1)
    
    async def _liquidation_handler(self):
        """Handle forceOrder stream."""
        import websockets
        
        while True:
            try:
                async with websockets.connect(self.liq_url) as ws:
                    while True:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        
                        if data.get("e") != "forceOrder":
                            continue
                        
                        order = data["o"]
                        side = order["S"]
                        qty = float(order["q"])
                        price = float(order["ap"])
                        notional = qty * price
                        
                        now = time.time()
                        if side == "BUY":
                            self.short_liqs.append((now, notional))
                        else:
                            self.long_liqs.append((now, notional))
                        
            except Exception as e:
                print(f"Liquidation stream error: {e}")
                await asyncio.sleep(1)
    
    async def _markprice_handler(self):
        """Handle markPrice stream."""
        import websockets
        
        while True:
            try:
                async with websockets.connect(self.mark_url) as ws:
                    while True:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        
                        if not isinstance(data, list):
                            continue
                        
                        for item in data:
                            if item.get("e") != "markPriceUpdate":
                                continue
                            
                            symbol = item["s"]
                            rate = float(item.get("r", "0"))
                            self.funding_rates[symbol] = rate
                        
            except Exception as e:
                print(f"MarkPrice stream error: {e}")
                await asyncio.sleep(1)
    
    async def _prediction_loop(self):
        """Main prediction and display loop."""
        last_predict = 0
        
        while True:
            await asyncio.sleep(1.0)
            
            now = time.time()
            
            # Update entropy
            raw_H, smooth_H, dH = self._compute_entropy()
            funding = self._get_weighted_funding()
            long_liqs, short_liqs = self._get_liq_totals()
            
            self.predictor.update_entropy(raw_H, smooth_H, dH)
            self.predictor.update_funding(funding)
            self.predictor.update_liquidations(long_liqs, short_liqs)
            
            # Predict every 5 seconds
            if now - last_predict < 5.0:
                # Quick status line
                regime = self.predictor.state.get_regime()
                btc = self.latest_prices.get("BTCUSDT", 0)
                tps = self.trade_count / max(now - self.start_time, 1)
                
                line = (
                    f"\r[{time.strftime('%H:%M:%S')}] "
                    f"BTC:${btc:,.0f} | "
                    f"{regime.value:8s} H={smooth_H:.2f} Δ={dH:+.2f} | "
                    f"F:{funding*100:+.3f}% | "
                    f"{tps:.0f}/s    "
                )
                print(line, end="", flush=True)
                continue
            
            last_predict = now
            
            # Generate predictions
            predictions = self.predictor.predict()
            
            # Display
            print("\n" + "=" * 100)
            print(f"  NS PREDICTOR - {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 100)
            
            btc = self.latest_prices.get("BTCUSDT", 0)
            eth = self.latest_prices.get("ETHUSDT", 0)
            sol = self.latest_prices.get("SOLUSDT", 0)
            
            print(f"  BTC: ${btc:,.0f}  ETH: ${eth:,.0f}  SOL: ${sol:.2f}")
            print(f"  Entropy: {smooth_H:.2f} (raw: {raw_H:.2f}, Δ: {dH:+.3f})")
            print(f"  Funding: {funding*100:+.4f}%")
            print(f"  Liquidations: Long ${long_liqs/1e6:.2f}M, Short ${short_liqs/1e6:.2f}M")
            print("-" * 100)
            print("  PREDICTIONS:")
            print("-" * 100)
            
            for pred in predictions:
                horizon_label = f"{pred.horizon_seconds // 60}m" if pred.horizon_seconds >= 60 else f"{pred.horizon_seconds}s"
                
                # Direction indicator
                if pred.entropy_predicted > pred.entropy_current + 0.3:
                    h_dir = "↑"
                    h_color = "\033[91m"  # Red
                elif pred.entropy_predicted < pred.entropy_current - 0.3:
                    h_dir = "↓"
                    h_color = "\033[92m"  # Green
                else:
                    h_dir = "→"
                    h_color = "\033[93m"  # Yellow
                
                # Regime transition
                if pred.regime_transition_prob > 0.3:
                    regime_str = f"{pred.regime_current.value} ⚡→ {pred.regime_predicted.value}"
                else:
                    regime_str = pred.regime_current.value
                
                # Cascade warning
                cascade_str = ""
                if pred.cascade_probability > 0.2:
                    cascade_str = f" 🌊 CASCADE {pred.cascade_probability:.0%}"
                
                print(
                    f"  {horizon_label:>4s}: "
                    f"H={pred.entropy_current:.2f}→{pred.entropy_predicted:.2f} {h_dir} "
                    f"({pred.entropy_confidence:.0%}) | "
                    f"{regime_str:16s} | "
                    f"Price: {pred.price_mean_pct:+.2f}% ± {pred.price_std_pct:.2f}%"
                    f"{cascade_str}"
                )
            
            # Generate and display trading signal
            signal, signal_meta = self.predictor.generate_signal(predictions)
            
            print("-" * 100)
            if signal != TradingSignal.NONE:
                conf_bar = "█" * int(signal_meta["confidence"] * 10) + "░" * (10 - int(signal_meta["confidence"] * 10))
                print(f"  SIGNAL: {signal.value}")
                print(f"  Confidence: [{conf_bar}] {signal_meta['confidence']:.0%}")
                print(f"  Reason: {signal_meta['reason']}")
            else:
                print("  SIGNAL: No actionable signal")
            
            # Validate expired predictions and show accuracy
            validated = self.predictor.validate_predictions()
            accuracy_report = self.predictor.get_accuracy_report()
            
            # Show validation results
            print("-" * 100)
            print("  VALIDATION (measured accuracy):")
            print("-" * 100)
            
            total_validated = sum(r["count"] for r in accuracy_report.values())
            pending = len(self.predictor.pending_predictions)
            
            if total_validated == 0:
                print(f"  Pending: {pending} predictions waiting to expire...")
                print("  (Need to run for at least 1 minute to validate 1m predictions)")
            else:
                for horizon in sorted(accuracy_report.keys()):
                    stats = accuracy_report[horizon]
                    if stats["count"] > 0:
                        horizon_label = f"{horizon // 60}m" if horizon >= 60 else f"{horizon}s"
                        mae = stats["mae"]
                        dir_acc = stats["direction_accuracy"]
                        print(
                            f"  {horizon_label:>4s}: "
                            f"n={stats['count']:4d} | "
                            f"MAE={mae:.3f} | "
                            f"Direction={dir_acc:.1%}"
                        )
                
                print(f"  Total validated: {total_validated} | Pending: {pending}")
            
            # Show recent validations
            if validated:
                print("-" * 100)
                print("  JUST VALIDATED:")
                for v in validated[-3:]:  # Show last 3
                    horizon_label = f"{v['horizon'] // 60}m"
                    err_sign = "+" if v["error"] > 0 else ""
                    dir_icon = "✓" if v["direction_correct"] else "✗"
                    print(
                        f"    {horizon_label}: predicted={v['predicted_entropy']:.2f}, "
                        f"actual={v['actual_entropy']:.2f}, "
                        f"error={err_sign}{v['error']:.2f} {dir_icon}"
                    )
            
            print("=" * 100)
            print()
    
    async def run(self):
        """Run all handlers."""
        print("=" * 90)
        print("  NS PREDICTOR - Navier-Stokes Market Prediction Engine")
        print("=" * 90)
        print(f"  GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
        print(f"  Symbols: {len(self.symbols)}")
        print(f"  Grid: {GRID_SIZE}x{GRID_SIZE}")
        print(f"  Horizons: {[f'{h//60}m' if h >= 60 else f'{h}s' for h in HORIZONS]}")
        print("=" * 90)
        print()
        print("  Physics Model:")
        print("    - Vorticity transport: ∂ω/∂t + (u·∇)ω = ν∇²ω + F")
        print("    - Stream function:     ∇²ψ = -ω")
        print("    - Energy cascade:      E(k) ~ k^(-5/3)")
        print()
        print("  Prediction Outputs:")
        print("    - Entropy trajectory at each horizon")
        print("    - Regime transition probability")
        print("    - Price distribution evolution")
        print("    - Cascade probability")
        print("=" * 90)
        print()
        
        await asyncio.gather(
            self._trade_handler(),
            self._liquidation_handler(),
            self._markprice_handler(),
            self._prediction_loop(),
        )


# =============================================================================
# MAIN
# =============================================================================

async def main():
    predictor = GalaxyPredictor()
    await predictor.run()


if __name__ == "__main__":
    asyncio.run(main())
