"""
FDTD Acoustic Wave Equation Solver

Solves the Full Wave Equation on GPU - NOT ray tracing approximations.
We see diffraction, leakage around obstacles, and perfect acoustic shadows.

The Wave Equation:
    ∂²p/∂t² = c² ∇²p

Where:
    p = Pressure field
    c = Sound speed (varies with depth via Munk profile)
    ∇² = Laplacian operator

Discretization (FDTD - Finite Difference Time Domain):
    p(t+dt) = 2p(t) - p(t-dt) + (c·dt/dx)² · ∇²p(t)

Boundary Conditions:
    - Surface (z=0): Pressure release (p=0) - sound reflects with phase inversion
    - Bottom: Rigid reflection (∂p/∂n=0) or absorbing
    - Terrain: Rigid reflection

CFL Stability Condition:
    dt ≤ dx / (c_max · √2)

Reference: Taflove, A. (2005). "Computational Electrodynamics:
The Finite-Difference Time-Domain Method."
(Same math applies to acoustics!)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .ocean import OceanDomain


@dataclass
class AcousticField:
    """
    Result of acoustic wave propagation simulation.

    Contains the pressure field and derived quantities for stealth analysis.
    """

    # Raw pressure field (nz, nx)
    pressure: torch.Tensor

    # Acoustic intensity (|p|² averaged)
    intensity: torch.Tensor

    # Maximum pressure reached at each point (for shadow detection)
    max_pressure: torch.Tensor

    # Simulation parameters
    dt: float
    steps: int
    source_depth: float
    source_freq: float

    def get_intensity_at(self, depth_m: float, range_km: float, res: float) -> float:
        """Get acoustic intensity at a specific location."""
        z_idx = int(depth_m / res)
        x_idx = int((range_km * 1000) / res)

        z_idx = max(0, min(z_idx, self.intensity.shape[0] - 1))
        x_idx = max(0, min(x_idx, self.intensity.shape[1] - 1))

        return self.intensity[z_idx, x_idx].item()

    def get_transmission_loss(
        self, depth_m: float, range_km: float, res: float
    ) -> float:
        """
        Get transmission loss in dB at a location.

        TL = -20 log₁₀(p/p₀)
        Higher TL = weaker signal = better for hiding
        """
        intensity = self.get_intensity_at(depth_m, range_km, res)
        if intensity <= 0:
            return float("inf")  # Perfect shadow

        # Reference intensity (normalized to max)
        ref_intensity = self.intensity.max().item()
        if ref_intensity <= 0:
            return 0.0

        # TL in dB
        return -10 * np.log10(intensity / ref_intensity + 1e-10)


@dataclass
class StealthReport:
    """
    Tactical stealth assessment for a submarine position.
    """

    # Position
    range_km: float
    depth_m: float

    # Acoustic exposure
    signal_strength: float  # Received intensity (linear)
    transmission_loss_db: float  # Signal attenuation in dB

    # Classification
    is_detected: bool
    detection_margin_db: float  # How far above/below threshold

    # Tactical recommendation
    recommendation: str

    def __str__(self) -> str:
        status = "⛔ DETECTED" if self.is_detected else "✅ UNDETECTED"
        return (
            f"[STEALTH REPORT]\n"
            f"  Position: Range {self.range_km}km, Depth {self.depth_m}m\n"
            f"  Signal Strength: {self.signal_strength:.6f}\n"
            f"  Transmission Loss: {self.transmission_loss_db:.1f} dB\n"
            f"  Status: {status}\n"
            f"  Detection Margin: {self.detection_margin_db:+.1f} dB\n"
            f"  Recommendation: {self.recommendation}"
        )


def solve_sonar_ping(
    ocean: OceanDomain,
    source_depth_m: float = 100.0,
    source_range_km: float = 0.0,
    frequency_hz: float = 50.0,
    steps: int = 2500,
    source_duration_steps: int = 100,
    absorbing_boundaries: bool = True,
    callback: Callable | None = None,
) -> AcousticField:
    """
    Simulate a sonar ping propagating through the ocean.

    Uses Finite Difference Time Domain (FDTD) to solve the wave equation.
    This captures the FULL wave physics including:
    - Refraction (bending) due to sound speed gradients
    - Diffraction around obstacles
    - Reflection from surface, bottom, and terrain
    - Acoustic shadow zones

    Args:
        ocean: The ocean domain with sound speed profile
        source_depth_m: Depth of the sonar source
        source_range_km: Range of source (0 = left edge)
        frequency_hz: Sonar frequency (lower = longer range)
        steps: Number of time steps to simulate
        source_duration_steps: How long the ping lasts
        absorbing_boundaries: Use absorbing BC at edges (prevents reflections)
        callback: Optional function(step, pressure) for visualization

    Returns:
        AcousticField with pressure, intensity, and analysis data
    """
    nz, nx = ocean.nz, ocean.nx
    device = ocean.device
    dx = ocean.res

    # 1. Initialize pressure fields (current, previous)
    p = torch.zeros((nz, nx), device=device)
    p_prev = torch.zeros_like(p)

    # Track maximum pressure for shadow detection
    max_pressure = torch.zeros_like(p)

    # Running intensity average
    intensity_sum = torch.zeros_like(p)

    # 2. CFL Stability Condition
    # dt ≤ dx / (c_max · √2)
    c_max = ocean.c.max().item()
    dt = (dx / (c_max * 1.414)) * 0.9  # 90% safety margin

    # Precompute coefficient: (c · dt / dx)²
    coeff = (ocean.c * dt / dx) ** 2

    # 3. Source location
    sz = int(source_depth_m / dx)
    sx = int((source_range_km * 1000) / dx) + 10  # Slight offset from boundary
    sz = max(1, min(sz, nz - 2))
    sx = max(1, min(sx, nx - 2))

    # Angular frequency for source
    omega = 2 * np.pi * frequency_hz

    print("[SONAR] Ping simulation starting...")
    print(f"[SONAR] Source: depth={source_depth_m}m, freq={frequency_hz}Hz")
    print(f"[SONAR] Grid: {nz}×{nx}, dt={dt*1000:.3f}ms, {steps} steps")
    print(
        f"[SONAR] Simulation time: {steps*dt:.2f}s, Range: {steps*dt*c_max/1000:.1f}km"
    )

    # 4. Time stepping loop
    for t in range(steps):
        # === Laplacian (5-point stencil) ===
        # ∇²p ≈ (p[z+1] + p[z-1] + p[x+1] + p[x-1] - 4p) / dx²
        # Using torch.roll for efficient neighbor access
        p_up = torch.roll(p, shifts=-1, dims=0)
        p_down = torch.roll(p, shifts=1, dims=0)
        p_left = torch.roll(p, shifts=-1, dims=1)
        p_right = torch.roll(p, shifts=1, dims=1)

        laplacian = p_up + p_down + p_left + p_right - 4.0 * p

        # === Wave Equation Update ===
        # p(t+dt) = 2p(t) - p(t-dt) + coeff · ∇²p
        p_next = 2.0 * p - p_prev + coeff * laplacian

        # === Boundary Conditions ===

        # Surface (z=0): Pressure release (p=0)
        # Sound reflects with phase inversion (like open end of pipe)
        p_next[0, :] = 0.0

        # Bottom: Rigid reflection (Neumann BC)
        # ∂p/∂z = 0 → p[nz-1] = p[nz-2]
        p_next[-1, :] = p_next[-2, :]

        # Terrain: Rigid reflection
        if ocean.terrain_mask.any():
            p_next[ocean.terrain_mask > 0.5] = 0.0

        # Absorbing boundaries (left/right edges)
        if absorbing_boundaries:
            # Simple absorbing BC: p_next = p (no reflection)
            # More sophisticated: Perfectly Matched Layers (PML)
            # For now, use simple taper
            taper_width = 20
            taper = torch.linspace(0, 1, taper_width, device=device)

            # Left edge
            p_next[:, :taper_width] *= taper.view(1, -1)
            # Right edge
            p_next[:, -taper_width:] *= taper.flip(0).view(1, -1)

        # === Inject Source ===
        if t < source_duration_steps:
            # Ricker wavelet (Mexican hat) - common for seismic/sonar
            # More physically realistic than pure sine
            t_norm = (t - source_duration_steps / 2) / (source_duration_steps / 4)
            ricker = (1 - 2 * (np.pi * t_norm) ** 2) * np.exp(-((np.pi * t_norm) ** 2))

            # Also add sine component for frequency content
            sine = np.sin(omega * t * dt)

            # Combined source
            source_amplitude = 1.0
            p_next[sz, sx] += source_amplitude * (0.5 * ricker + 0.5 * sine)

        # === Update tracking fields ===
        max_pressure = torch.maximum(max_pressure, torch.abs(p_next))
        intensity_sum += p_next**2

        # === Cycle buffers ===
        p_prev = p
        p = p_next

        # Progress callback
        if callback is not None and t % 100 == 0:
            callback(t, p)

        # Progress reporting
        if t > 0 and t % 500 == 0:
            max_p = torch.abs(p).max().item()
            print(f"[SONAR] Step {t}/{steps}, max|p|={max_p:.4f}")

    # Compute final intensity (time-averaged)
    intensity = intensity_sum / steps

    print(f"[SONAR] Simulation complete. Max intensity: {intensity.max():.6f}")

    return AcousticField(
        pressure=p,
        intensity=intensity,
        max_pressure=max_pressure,
        dt=dt,
        steps=steps,
        source_depth=source_depth_m,
        source_freq=frequency_hz,
    )


def analyze_stealth(
    ocean: OceanDomain,
    field: AcousticField,
    sub_range_km: float,
    sub_depth_m: float,
    detection_threshold_db: float = 60.0,
) -> StealthReport:
    """
    Analyze whether a submarine at a given position would be detected.

    Args:
        ocean: The ocean domain
        field: Computed acoustic field
        sub_range_km: Submarine range in km
        sub_depth_m: Submarine depth in meters
        detection_threshold_db: Minimum TL for stealth (higher = harder to detect)

    Returns:
        StealthReport with detection assessment
    """
    # Check if position is valid (not in terrain)
    if ocean.is_solid(sub_depth_m, sub_range_km):
        return StealthReport(
            range_km=sub_range_km,
            depth_m=sub_depth_m,
            signal_strength=0.0,
            transmission_loss_db=float("inf"),
            is_detected=False,
            detection_margin_db=float("inf"),
            recommendation="Position is inside solid terrain. Invalid.",
        )

    # Get signal strength at submarine position
    signal = field.get_intensity_at(sub_depth_m, sub_range_km, ocean.res)
    tl_db = field.get_transmission_loss(sub_depth_m, sub_range_km, ocean.res)

    # Detection assessment
    is_detected = tl_db < detection_threshold_db
    margin = tl_db - detection_threshold_db  # Positive = hidden, negative = detected

    # Generate tactical recommendation
    if margin > 20:
        recommendation = "SHADOW ZONE. Excellent stealth position. Hold station."
    elif margin > 10:
        recommendation = "LOW SIGNATURE. Good concealment. Reduce noise."
    elif margin > 0:
        recommendation = "MARGINAL. Risk of detection. Seek deeper cover."
    elif margin > -10:
        recommendation = "EXPOSED. High detection risk. Evade immediately."
    else:
        recommendation = "COMPROMISED. Active tracking likely. Go deep, go silent."

    return StealthReport(
        range_km=sub_range_km,
        depth_m=sub_depth_m,
        signal_strength=signal,
        transmission_loss_db=tl_db,
        is_detected=is_detected,
        detection_margin_db=margin,
        recommendation=recommendation,
    )


def find_shadow_zones(
    ocean: OceanDomain,
    field: AcousticField,
    threshold_db: float = 60.0,
) -> torch.Tensor:
    """
    Find all shadow zones in the acoustic field.

    Returns a mask where 1.0 = shadow zone (safe), 0.0 = exposed
    """
    # Get reference intensity (near source)
    ref_intensity = field.intensity.max().item()

    if ref_intensity <= 0:
        return torch.ones_like(field.intensity)

    # Compute transmission loss field
    with torch.no_grad():
        tl_field = -10 * torch.log10(field.intensity / ref_intensity + 1e-10)

    # Shadow zones are where TL > threshold
    shadow_mask = (tl_field > threshold_db).float()

    # Exclude terrain
    shadow_mask[ocean.terrain_mask > 0.5] = 0.0

    return shadow_mask


def compute_detection_probability(
    ocean: OceanDomain,
    field: AcousticField,
    sub_range_km: float,
    sub_depth_m: float,
    sonar_sensitivity_db: float = 60.0,
    noise_floor_db: float = 10.0,
) -> float:
    """
    Compute probability of detection using a simple model.

    Uses sigmoid function:
    P(detect) = 1 / (1 + exp(-(signal_db - threshold_db) / noise_db))

    Returns:
        Detection probability [0, 1]
    """
    signal = field.get_intensity_at(sub_depth_m, sub_range_km, ocean.res)

    if signal <= 0:
        return 0.0  # Perfect shadow

    # Convert to dB relative to reference
    ref = field.intensity.max().item()
    signal_db = 10 * np.log10(signal / ref + 1e-10)

    # Sigmoid detection model
    # signal_db is negative (TL), threshold is negative (sensitivity limit)
    threshold = -sonar_sensitivity_db
    prob = 1.0 / (1.0 + np.exp(-(signal_db - threshold) / noise_floor_db))

    return prob


def scan_for_optimal_hiding_spot(
    ocean: OceanDomain,
    field: AcousticField,
    min_range_km: float = 5.0,
    max_range_km: float = None,
    min_depth_m: float = 50.0,
    max_depth_m: float = None,
    grid_resolution_km: float = 1.0,
) -> list[tuple[float, float, float]]:
    """
    Scan the domain for optimal hiding positions.

    Returns:
        List of (range_km, depth_m, transmission_loss_db) sorted by TL (best first)
    """
    if max_range_km is None:
        max_range_km = ocean.range_km - 1
    if max_depth_m is None:
        max_depth_m = ocean.depth - 100

    candidates = []

    range_step = grid_resolution_km
    depth_step = grid_resolution_km * 100  # Same resolution in depth

    range_km = min_range_km
    while range_km <= max_range_km:
        depth_m = min_depth_m
        while depth_m <= max_depth_m:
            # Skip terrain
            if not ocean.is_solid(depth_m, range_km):
                tl = field.get_transmission_loss(depth_m, range_km, ocean.res)
                candidates.append((range_km, depth_m, tl))
            depth_m += depth_step
        range_km += range_step

    # Sort by transmission loss (highest = best hiding)
    candidates.sort(key=lambda x: x[2], reverse=True)

    return candidates
