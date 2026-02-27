"""
Two-Stream Instability Demo

Classic plasma physics benchmark: counter-propagating electron beams
become unstable and develop phase-space vortices.

This is a standard test problem for Vlasov codes and demonstrates
the ability to capture kinetic plasma effects.

Example:
    >>> from qtenet.demos import two_stream_instability
    >>> 
    >>> results = two_stream_instability(dims=5, n_steps=500)
    >>> print(f"Final rank: {results.final_max_rank}")

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch


@dataclass
class TwoStreamResult:
    """Results from two-stream instability simulation."""
    dims: int
    n_steps: int
    dt: float
    final_time: float
    initial_max_rank: int
    final_max_rank: int
    compression_ratio: float
    total_time_s: float
    growth_detected: bool


def two_stream_instability(
    dims: int = 5,
    qubits_per_dim: int = 5,
    max_rank: int = 64,
    n_steps: int = 500,
    dt: float = 0.01,
    device: str = "cpu",
    verbose: bool = True,
) -> TwoStreamResult:
    """
    Run two-stream instability simulation.
    
    The two-stream instability is a classic kinetic plasma phenomenon
    where counter-propagating beams become unstable due to wave-particle
    interactions.
    
    Args:
        dims: 5 or 6 dimensional phase space
        qubits_per_dim: Qubits per dimension
        max_rank: Maximum QTT rank
        n_steps: Number of timesteps
        dt: Time step
        device: Torch device
        verbose: Print progress
    
    Returns:
        TwoStreamResult with simulation data
    """
    if dims == 5:
        from qtenet.solvers import Vlasov5D, Vlasov5DConfig
        config = Vlasov5DConfig(
            qubits_per_dim=qubits_per_dim,
            max_rank=max_rank,
            device=device,
        )
        solver = Vlasov5D(config)
    elif dims == 6:
        from qtenet.solvers import Vlasov6D, Vlasov6DConfig
        config = Vlasov6DConfig(
            qubits_per_dim=qubits_per_dim,
            max_rank=max_rank,
            device=device,
        )
        solver = Vlasov6D(config)
    else:
        raise ValueError(f"dims must be 5 or 6, got {dims}")
    
    if verbose:
        print(f"\nTwo-Stream Instability ({dims}D)")
        print("=" * 50)
    
    state = solver.two_stream_ic()
    initial_rank = state.max_rank
    initial_params = sum(c.numel() for c in state.cores)
    total_points = config.grid_size ** dims
    
    if verbose:
        print(f"Initial: rank={initial_rank}, params={initial_params:,}")
        print(f"Compression: {total_points / initial_params:,.0f}×")
        print(f"\nRunning {n_steps} steps...")
    
    t0 = time.perf_counter()
    for step in range(n_steps):
        state = solver.step(state, dt=dt)
        
        if verbose and (step + 1) % 100 == 0:
            print(f"  Step {step + 1}: rank={state.max_rank}")
    
    total_time = time.perf_counter() - t0
    
    final_params = sum(c.numel() for c in state.cores)
    
    result = TwoStreamResult(
        dims=dims,
        n_steps=n_steps,
        dt=dt,
        final_time=state.time,
        initial_max_rank=initial_rank,
        final_max_rank=state.max_rank,
        compression_ratio=total_points / final_params,
        total_time_s=total_time,
        growth_detected=state.max_rank > initial_rank,
    )
    
    if verbose:
        print(f"\nFinal: rank={result.final_max_rank}")
        print(f"Compression: {result.compression_ratio:,.0f}×")
        print(f"Time: {result.total_time_s:.1f}s")
        if result.growth_detected:
            print("Instability growth detected (rank increased)!")
    
    return result


__all__ = ["two_stream_instability", "TwoStreamResult"]
