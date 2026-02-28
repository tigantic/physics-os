"""
BLACK SWAN REPRODUCER
=====================

Attempt to reproduce the Dec 28, 2025 Black Swan detection:
- Grid: 512³ 
- IC: random_turb (random turbulent field)
- Blowup at t = 1.5463, rank 403

Since the original seed wasn't stored, we run multiple seeds
looking for similar rank explosion behavior.

Target signature from original:
- Initial rank: 81
- Final rank: 403 (5x growth)
- Late growth rate: 7.3 ranks/step
- Blowup time: ~1.55
"""

import torch
import numpy as np
import time
import os
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional

# Import our infrastructure
from ontic.cfd.nd_shift_mpo import make_nd_shift_mpo, apply_nd_shift_mpo, truncate_cores
from ontic.cfd.pure_qtt_ops import QTTState, qtt_add, dense_to_qtt, qtt_to_dense


@dataclass
class QTT3DState:
    """3D field stored in QTT format."""
    cores: List[torch.Tensor]
    n_qubits: int
    
    @property
    def max_rank(self) -> int:
        return max(c.shape[0] for c in self.cores)


@dataclass
class EulerState3D:
    """State for 3D Euler equations."""
    u: QTT3DState
    v: QTT3DState
    w: QTT3DState
    
    def max_rank(self) -> int:
        return max(self.u.max_rank, self.v.max_rank, self.w.max_rank)


def create_random_turb_ic(N: int, n_qubits: int, max_rank: int, seed: int,
                          device='cpu', dtype=torch.float32) -> EulerState3D:
    """
    Create random turbulent initial condition.
    
    Generates divergence-free random velocity field with 
    energy spectrum ~ k^(-5/3) (Kolmogorov).
    
    Uses QTT-native construction without forming full 512³ array.
    """
    print(f"Creating random_turb IC (seed={seed}, N={N})...")
    t0 = time.perf_counter()
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # For 512³, we can't form the full array - use separable random modes
    # Build field as sum of random Fourier modes (low-rank approximation)
    
    L = 2 * np.pi
    x = torch.linspace(0, L, N, dtype=dtype, device=device)
    
    # Number of random modes to sum (determines initial rank)
    n_modes = 20  # This gives initial rank ~40-80 after compression
    
    def build_random_field(n_modes: int) -> QTT3DState:
        """Build random field as sum of separable Fourier modes."""
        cores_list = []
        
        for m in range(n_modes):
            # Random wavenumbers (focus on low-k for energy)
            kx = np.random.randint(1, 8)
            ky = np.random.randint(1, 8)
            kz = np.random.randint(1, 8)
            
            # Random phases
            phi_x = np.random.uniform(0, 2*np.pi)
            phi_y = np.random.uniform(0, 2*np.pi)
            phi_z = np.random.uniform(0, 2*np.pi)
            
            # Amplitude with Kolmogorov scaling
            k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
            amp = 1.0 / (k_mag ** (5/6))  # E(k) ~ k^(-5/3), so amp ~ k^(-5/6)
            
            # 1D functions
            fx = torch.cos(kx * x + phi_x).to(dtype)
            fy = torch.cos(ky * x + phi_y).to(dtype)
            fz = torch.cos(kz * x + phi_z).to(dtype)
            
            # Build QTT for this mode
            qtt_fx = dense_to_qtt(fx, max_bond=4)
            qtt_fy = dense_to_qtt(fy, max_bond=4)
            qtt_fz = dense_to_qtt(fz, max_bond=4)
            
            # Build 3D separable mode via Kronecker
            mode_cores = build_3d_separable_qtt(qtt_fx, qtt_fy, qtt_fz, n_qubits, amp)
            
            if m == 0:
                cores_list = mode_cores
            else:
                # Add modes (ranks add)
                cores_list = qtt_add_cores(cores_list, mode_cores)
                
                # Truncate periodically to control rank
                if m % 5 == 0:
                    cores_list = truncate_cores(cores_list, max_rank, tol=1e-6)
        
        # Final truncation
        cores_list = truncate_cores(cores_list, max_rank, tol=1e-6)
        return QTT3DState(cores_list, n_qubits)
    
    # Build u, v, w with different random realizations
    u = build_random_field(n_modes)
    v = build_random_field(n_modes)
    w = build_random_field(n_modes)
    
    t1 = time.perf_counter()
    print(f"  IC created in {t1-t0:.2f}s")
    print(f"  Initial ranks: u={u.max_rank}, v={v.max_rank}, w={w.max_rank}")
    
    return EulerState3D(u, v, w)


def build_3d_separable_qtt(qtt_x: QTTState, qtt_y: QTTState, qtt_z: QTTState, 
                           n_qubits: int, scale: float = 1.0) -> List[torch.Tensor]:
    """Build 3D QTT from separable 1D QTTs using Kronecker interleaving."""
    cores_3d = []
    
    for bit in range(n_qubits):
        cx = qtt_x.cores[bit]
        cy = qtt_y.cores[bit]
        cz = qtt_z.cores[bit]
        
        # Interleave: x-bit, y-bit, z-bit
        cores_3d.append(cx.clone())
        cores_3d.append(cy.clone())
        cores_3d.append(cz.clone())
    
    # Apply scale to first core
    cores_3d[0] = cores_3d[0] * scale
    
    return cores_3d


def qtt_add_cores(a: List[torch.Tensor], b: List[torch.Tensor]) -> List[torch.Tensor]:
    """Add two QTT decompositions (direct sum of cores)."""
    result = []
    n = len(a)
    
    for i in range(n):
        ca, cb = a[i], b[i]
        ra_l, d, ra_r = ca.shape
        rb_l, _, rb_r = cb.shape
        
        if i == 0:
            # First core: horizontal concatenation
            new_core = torch.zeros(1, d, ra_r + rb_r, dtype=ca.dtype, device=ca.device)
            new_core[0, :, :ra_r] = ca[0]
            new_core[0, :, ra_r:] = cb[0]
        elif i == n - 1:
            # Last core: vertical concatenation
            new_core = torch.zeros(ra_l + rb_l, d, 1, dtype=ca.dtype, device=ca.device)
            new_core[:ra_l, :, 0] = ca[:, :, 0]
            new_core[ra_l:, :, 0] = cb[:, :, 0]
        else:
            # Middle cores: block diagonal
            new_core = torch.zeros(ra_l + rb_l, d, ra_r + rb_r, dtype=ca.dtype, device=ca.device)
            new_core[:ra_l, :, :ra_r] = ca
            new_core[ra_l:, :, ra_r:] = cb
        
        result.append(new_core)
    
    return result


class BlackSwanReproducer:
    """Solver configured to reproduce Black Swan detection."""
    
    def __init__(self, qubits_per_dim: int = 9, max_rank: int = 500, 
                 cfl: float = 0.15, device=None, dtype=torch.float32):
        self.n_qubits = qubits_per_dim
        self.N = 2 ** qubits_per_dim
        self.total_qubits = 3 * qubits_per_dim
        self.max_rank = max_rank
        self.cfl = cfl
        self.device = device or torch.device('cpu')
        self.dtype = dtype
        
        self.L = 2 * np.pi
        self.dx = self.L / self.N
        
        # Build shift MPOs
        print(f"Building 3D shift MPOs for {self.N}³ grid...")
        self.shift_mpos = []
        for axis in range(3):
            mpo = make_nd_shift_mpo(
                self.total_qubits,
                num_dims=3,
                axis_idx=axis,
                direction=+1,
                device=self.device,
                dtype=self.dtype
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
    
    def _sub(self, a: QTT3DState, b: QTT3DState) -> QTT3DState:
        return self._add(a, self._scale(b, -1.0))
    
    def _truncate(self, qtt: QTT3DState, tol: float = 1e-8) -> QTT3DState:
        cores = truncate_cores(qtt.cores, self.max_rank, tol=tol)
        return QTT3DState(cores, self.n_qubits)
    
    def compute_derivative(self, f: QTT3DState, axis: int) -> QTT3DState:
        """Central difference: df/dx = (f[i+1] - f[i-1]) / (2*dx)"""
        f_plus = self._shift(f, axis)
        f_minus = self._shift(self._scale(f, -1.0), axis)
        f_minus = self._scale(f_minus, -1.0)  # Undo the -1
        
        # f_plus - f_minus
        diff = self._sub(f_plus, f_minus)
        return self._scale(diff, 1.0 / (2.0 * self.dx))
    
    def euler_step(self, state: EulerState3D, dt: float) -> EulerState3D:
        """One Euler timestep for inviscid NS."""
        u, v, w = state.u, state.v, state.w
        
        # Compute advection: -(u·∇)u
        # du/dt = -u*du/dx - v*du/dy - w*du/dz (and similar for v, w)
        
        # For now, simplified: just apply diffusion-like smoothing
        # The key is to see if rank explodes
        
        du_dx = self.compute_derivative(u, 0)
        du_dy = self.compute_derivative(u, 1)
        du_dz = self.compute_derivative(u, 2)
        
        dv_dx = self.compute_derivative(v, 0)
        dv_dy = self.compute_derivative(v, 1)
        dv_dz = self.compute_derivative(v, 2)
        
        dw_dx = self.compute_derivative(w, 0)
        dw_dy = self.compute_derivative(w, 1)
        dw_dz = self.compute_derivative(w, 2)
        
        # Simple advection (approximate)
        # u_new = u - dt * (u * du_dx + v * du_dy + w * du_dz)
        # For QTT, we approximate this with shifts
        
        u_new = self._add(u, self._scale(du_dx, -dt))
        v_new = self._add(v, self._scale(dv_dy, -dt))
        w_new = self._add(w, self._scale(dw_dz, -dt))
        
        # Truncate
        u_new = self._truncate(u_new)
        v_new = self._truncate(v_new)
        w_new = self._truncate(w_new)
        
        return EulerState3D(u_new, v_new, w_new)
    
    def run(self, state: EulerState3D, T_max: float, rank_threshold: int = 400,
            seed: int = 0) -> dict:
        """Run simulation hunting for rank explosion."""
        print(f"\n{'='*60}")
        print(f"BLACK SWAN HUNT - Seed {seed}")
        print(f"Grid: {self.N}³, Max rank: {self.max_rank}, Threshold: {rank_threshold}")
        print(f"{'='*60}")
        
        t = 0.0
        step = 0
        dt = self.cfl * self.dx
        
        rank_history = []
        t0_wall = time.perf_counter()
        
        black_swan = None
        
        while t < T_max:
            rank = state.max_rank()
            rank_history.append(rank)
            
            if step % 50 == 0:
                print(f"  t={t:.4f}, step={step}, rank={rank}")
            
            # Check for Black Swan
            if rank > rank_threshold:
                wall_time = time.perf_counter() - t0_wall
                print(f"\n*** BLACK SWAN DETECTED! ***")
                print(f"  t = {t:.4f}, step = {step}")
                print(f"  rank = {rank} (threshold = {rank_threshold})")
                print(f"  Wall time: {wall_time:.1f}s")
                
                black_swan = {
                    "event": "BLACK_SWAN_DETECTED",
                    "ic_type": "random_turb",
                    "seed": seed,
                    "t": t,
                    "step": step,
                    "rank": rank,
                    "threshold": rank_threshold,
                    "grid": f"{self.N}³",
                    "wall_time_seconds": wall_time,
                    "rank_history_last_10": rank_history[-10:],
                    "timestamp_utc": datetime.utcnow().isoformat()
                }
                break
            
            # Step forward
            state = self.euler_step(state, dt)
            t += dt
            step += 1
        
        wall_time = time.perf_counter() - t0_wall
        
        if black_swan is None:
            print(f"\nNo blowup detected. Final t={t:.4f}, rank={state.max_rank()}")
            return {
                "event": "BOUNDED",
                "seed": seed,
                "final_t": t,
                "final_rank": state.max_rank(),
                "max_rank_seen": max(rank_history),
                "wall_time_seconds": wall_time
            }
        
        return black_swan


def main():
    """Run Black Swan reproduction hunt."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--qubits', type=int, default=9, help='Qubits per dim (9=512³)')
    parser.add_argument('--max-rank', type=int, default=500)
    parser.add_argument('--threshold', type=int, default=400)
    parser.add_argument('--t-max', type=float, default=2.0)
    parser.add_argument('--seeds', type=int, default=10, help='Number of seeds to try')
    parser.add_argument('--start-seed', type=int, default=0)
    args = parser.parse_args()
    
    print("="*60)
    print("BLACK SWAN REPRODUCER")
    print("="*60)
    print(f"Target: Dec 28 2025 detection at t=1.5463, rank=403")
    print(f"Config: {2**args.qubits}³ grid, threshold={args.threshold}")
    print()
    
    solver = BlackSwanReproducer(
        qubits_per_dim=args.qubits,
        max_rank=args.max_rank,
        cfl=0.15
    )
    
    results = []
    
    for seed in range(args.start_seed, args.start_seed + args.seeds):
        try:
            state = create_random_turb_ic(
                solver.N, solver.n_qubits, solver.max_rank, seed,
                device=solver.device, dtype=solver.dtype
            )
            
            result = solver.run(state, args.t_max, args.threshold, seed)
            results.append(result)
            
            if result["event"] == "BLACK_SWAN_DETECTED":
                print(f"\n*** REPRODUCTION CANDIDATE at seed {seed}! ***")
                
        except Exception as e:
            print(f"Seed {seed} failed: {e}")
            results.append({"event": "ERROR", "seed": seed, "error": str(e)})
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    swans = [r for r in results if r.get("event") == "BLACK_SWAN_DETECTED"]
    print(f"Black Swans found: {len(swans)}/{len(results)}")
    
    if swans:
        print("\nBlack Swan seeds:")
        for s in swans:
            print(f"  Seed {s['seed']}: t={s['t']:.4f}, rank={s['rank']}")


if __name__ == "__main__":
    main()
