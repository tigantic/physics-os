#!/usr/bin/env python3
"""
BLACK SWAN #945 FORENSIC REPRODUCTION
=====================================

Target signature from Dec 28, 2025 detection:
- Grid: 512³ (134M points)
- IC: random_turb
- Steps to blowup: 945
- Blowup time: t = 1.5463
- dt ≈ 0.00164 (derived: t/steps)
- Initial rank: 81
- Final rank: 403 (threshold: 400)
- Late growth rate: 7.3 ranks/step
- Acceleration factor: 730x
- Dominant component: v

Key insight: The original ran 945 STABLE steps before rank explosion.
This is NOT numerical instability - it's gradual complexity growth.
"""

import torch
import numpy as np
import time
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional

# Import QTT infrastructure
from tensornet.cfd.nd_shift_mpo import make_nd_shift_mpo, apply_nd_shift_mpo, truncate_cores
from tensornet.cfd.pure_qtt_ops import QTTState, qtt_add, dense_to_qtt


@dataclass
class QTT3DState:
    cores: List[torch.Tensor]
    n_qubits: int
    
    @property
    def max_rank(self) -> int:
        return max(c.shape[0] for c in self.cores)


@dataclass
class EulerState3D:
    u: QTT3DState
    v: QTT3DState
    w: QTT3DState
    
    def max_rank(self) -> int:
        return max(self.u.max_rank, self.v.max_rank, self.w.max_rank)


def build_rank1_3d_qtt_kronecker(fx, fy, fz, n_qubits, max_rank=16):
    """Build 3D QTT from separable 1D functions using Kronecker."""
    N = len(fx)
    dtype = fx.dtype
    device = fx.device
    
    qtt_x = dense_to_qtt(fx, max_bond=4)
    qtt_y = dense_to_qtt(fy, max_bond=4)
    qtt_z = dense_to_qtt(fz, max_bond=4)
    
    cores_3d = []
    for bit in range(n_qubits):
        cores_3d.append(qtt_x.cores[bit].clone())
        cores_3d.append(qtt_y.cores[bit].clone())
        cores_3d.append(qtt_z.cores[bit].clone())
    
    cores_3d = truncate_cores(cores_3d, max_rank, tol=1e-8)
    return QTT3DState(cores_3d, n_qubits)


class BlackSwan945Reproducer:
    """
    Reproduce Black Swan #945 with exact parameter matching.
    """
    
    def __init__(self, qubits_per_dim=9, max_rank=500, device=None):
        self.n_qubits = qubits_per_dim
        self.N = 2 ** qubits_per_dim
        self.total_qubits = 3 * qubits_per_dim
        self.max_rank = max_rank
        self.device = device or torch.device('cpu')
        self.dtype = torch.float32
        
        self.L = 2 * np.pi
        self.dx = self.L / self.N
        
        # Original Black Swan parameters
        self.original_dt = 0.001636  # t_blowup / steps = 1.5463 / 945
        
        print(f"Building shift MPOs for {self.N}³ grid...")
        self.shift_mpos = []
        for axis in range(3):
            mpo = make_nd_shift_mpo(
                self.total_qubits, num_dims=3, axis_idx=axis,
                direction=+1, device=self.device, dtype=self.dtype
            )
            self.shift_mpos.append(mpo)
        print("  MPOs ready.")
    
    def _shift(self, qtt: QTT3DState, axis: int) -> QTT3DState:
        cores = apply_nd_shift_mpo(qtt.cores, self.shift_mpos[axis], 
                                    max_rank=self.max_rank)
        return QTT3DState(cores, self.n_qubits)
    
    def _add(self, a: QTT3DState, b: QTT3DState) -> QTT3DState:
        a_qtt = QTTState(cores=a.cores, num_qubits=len(a.cores))
        b_qtt = QTTState(cores=b.cores, num_qubits=len(b.cores))
        result = qtt_add(a_qtt, b_qtt, max_bond=self.max_rank)
        return QTT3DState(result.cores, self.n_qubits)
    
    def _scale(self, a: QTT3DState, s: float) -> QTT3DState:
        cores = [c.clone() for c in a.cores]
        cores[0] = cores[0] * s
        return QTT3DState(cores, self.n_qubits)
    
    def _truncate(self, qtt: QTT3DState, tol=1e-8) -> QTT3DState:
        cores = truncate_cores(qtt.cores, self.max_rank, tol=tol)
        return QTT3DState(cores, self.n_qubits)
    
    def create_random_turb_ic(self, seed=42, target_rank=81):
        """
        Create random turbulent IC matching original rank profile.
        """
        print(f"Creating random_turb IC (seed={seed}, target_rank≈{target_rank})...")
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        x = torch.linspace(0, self.L, self.N, dtype=self.dtype, device=self.device)
        
        # Calibrate n_modes to achieve target_rank ≈ 81
        # 5 modes gave rank 80, which matches original
        n_modes = 5
        
        def build_field():
            result = None
            for m in range(n_modes):
                kx = np.random.randint(1, 4)
                ky = np.random.randint(1, 4)
                kz = np.random.randint(1, 4)
                phi_x = np.random.uniform(0, 2*np.pi)
                phi_y = np.random.uniform(0, 2*np.pi)
                phi_z = np.random.uniform(0, 2*np.pi)
                k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
                amp = 1.0 / (k_mag ** (5/6))
                
                fx = torch.cos(kx * x + phi_x) * amp
                fy = torch.cos(ky * x + phi_y)
                fz = torch.cos(kz * x + phi_z)
                
                mode = build_rank1_3d_qtt_kronecker(fx, fy, fz, self.n_qubits, max_rank=16)
                
                if result is None:
                    result = mode
                else:
                    result = self._add(result, mode)
            
            return self._truncate(result, tol=1e-8)
        
        u = build_field()
        v = build_field()
        w = build_field()
        
        state = EulerState3D(u, v, w)
        print(f"  Initial ranks: u={u.max_rank}, v={v.max_rank}, w={w.max_rank}")
        return state
    
    def euler_step_conservative(self, state: EulerState3D, dt: float) -> EulerState3D:
        """
        Conservative Euler step with aggressive truncation.
        
        Key: Truncate AFTER each operation to control rank growth.
        """
        u, v, w = state.u, state.v, state.w
        
        # Compute derivatives via shifts: df/dx ≈ (f_{i+1} - f_{i-1}) / (2*dx)
        def derivative(f, axis):
            f_plus = self._shift(f, axis)
            f_minus_neg = self._shift(self._scale(f, -1.0), axis)
            diff = self._add(f_plus, f_minus_neg)
            diff = self._scale(diff, 1.0 / (2.0 * self.dx))
            return self._truncate(diff, tol=1e-6)
        
        # Simple advection: u_new = u - dt * du/dx (self-advection only)
        du_dx = derivative(u, 0)
        dv_dy = derivative(v, 1)
        dw_dz = derivative(w, 2)
        
        u_new = self._add(u, self._scale(du_dx, -dt))
        v_new = self._add(v, self._scale(dv_dy, -dt))
        w_new = self._add(w, self._scale(dw_dz, -dt))
        
        # Aggressive truncation to control rank
        u_new = self._truncate(u_new, tol=1e-6)
        v_new = self._truncate(v_new, tol=1e-6)
        w_new = self._truncate(w_new, tol=1e-6)
        
        return EulerState3D(u_new, v_new, w_new)
    
    def run(self, state: EulerState3D, max_steps=1000, threshold=400):
        """Run simulation tracking rank evolution."""
        print(f"\n{'='*60}")
        print("BLACK SWAN #945 REPRODUCTION")
        print(f"{'='*60}")
        print(f"Grid: {self.N}³ | Threshold: {threshold} | dt: {self.original_dt:.6f}")
        print()
        
        dt = self.original_dt
        t = 0.0
        step = 0
        
        rank_history = []
        t0 = time.perf_counter()
        
        initial_rank = state.max_rank()
        
        print(f"{'Step':>6} {'t':>8} {'rank':>6} {'u':>4} {'v':>4} {'w':>4} {'Δrank':>6}")
        print("-" * 55)
        
        while step < max_steps:
            rank = state.max_rank()
            rank_history.append(rank)
            
            delta = rank - rank_history[-2] if len(rank_history) > 1 else 0
            
            if step % 50 == 0 or rank >= threshold - 50:
                print(f"{step:6d} {t:8.4f} {rank:6d} {state.u.max_rank:4d} {state.v.max_rank:4d} {state.w.max_rank:4d} {delta:+6d}")
            
            if rank >= threshold:
                elapsed = time.perf_counter() - t0
                print(f"\n{'*'*60}")
                print(f"*** BLACK SWAN DETECTED at step {step}! ***")
                print(f"{'*'*60}")
                print(f"  Time: t = {t:.4f}")
                print(f"  Rank: {rank} (threshold: {threshold})")
                print(f"  Initial rank: {initial_rank}")
                print(f"  Growth factor: {rank/initial_rank:.2f}x")
                print(f"  Wall time: {elapsed:.1f}s")
                
                if len(rank_history) >= 10:
                    print(f"  Last 10 ranks: {rank_history[-10:]}")
                    
                    # Calculate late growth rate
                    late_growth = np.mean(np.diff(rank_history[-10:]))
                    early_growth = np.mean(np.diff(rank_history[:min(50, len(rank_history))]))
                    accel = late_growth / (early_growth + 1e-10)
                    print(f"  Late growth rate: {late_growth:.1f} ranks/step")
                    print(f"  Acceleration factor: {accel:.1f}x")
                
                # Save evidence
                evidence = {
                    "event": "BLACK_SWAN_DETECTED",
                    "ic_type": "random_turb",
                    "t": t,
                    "step": step,
                    "rank": rank,
                    "threshold": threshold,
                    "grid": f"{self.N}³",
                    "initial_rank": initial_rank,
                    "rank_growth_factor": rank / initial_rank,
                    "u_rank": state.u.max_rank,
                    "v_rank": state.v.max_rank,
                    "w_rank": state.w.max_rank,
                    "dominant_component": "v" if state.v.max_rank == rank else ("u" if state.u.max_rank == rank else "w"),
                    "rank_history_last_10": rank_history[-10:],
                    "timestamp_utc": datetime.utcnow().isoformat()
                }
                
                fname = f"BLACK_SWAN_reproduction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(fname, 'w') as f:
                    json.dump(evidence, f, indent=2)
                print(f"\n  Evidence saved to: {fname}")
                
                return evidence
            
            state = self.euler_step_conservative(state, dt)
            t += dt
            step += 1
        
        elapsed = time.perf_counter() - t0
        print(f"\nNo blowup in {max_steps} steps. Final rank: {state.max_rank()}")
        return {"event": "BOUNDED", "final_rank": state.max_rank(), "steps": step}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--qubits', type=int, default=9, help='9=512³')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--threshold', type=int, default=400)
    args = parser.parse_args()
    
    print("="*60)
    print("BLACK SWAN #945 FORENSIC REPRODUCTION")
    print("="*60)
    print()
    print("Original event signature:")
    print("  Steps: 945 | t: 1.5463 | rank: 81→403")
    print("  Late growth: 7.3 ranks/step | Accel: 730x")
    print()
    
    solver = BlackSwan945Reproducer(qubits_per_dim=args.qubits)
    state = solver.create_random_turb_ic(seed=args.seed)
    result = solver.run(state, max_steps=args.max_steps, threshold=args.threshold)
    
    print()
    if result["event"] == "BLACK_SWAN_DETECTED":
        print("🦢 BLACK SWAN REPRODUCTION SUCCESSFUL")
    else:
        print("Solution remained bounded")


if __name__ == "__main__":
    main()
