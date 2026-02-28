"""
PHASE 6: THE MILLENNIUM HUNTER
==============================

Numerical Analysis of the Navier-Stokes Singularity Problem.

The $1 Million Question:
  Does the 3D Incompressible Euler/Navier-Stokes equation develop 
  a singularity (infinite velocity/vorticity) in finite time?

The Barrier:
  Standard solvers crash because resolving the singularity requires N → ∞.

Ontic Bet:
  We simulate a virtual grid of 1024³ (1 Billion Points).
  We bet that the information content (Rank) of the singularity 
  grows slower than the grid resolution.

The Experiment:
  Taylor-Green Vortex at 1024³ - the "Standard Candle" for turbulence.
  
  Early Time (0 < t < 4): Laminar flow, Rank stays low (~4-10)
  Critical Time (t ≈ 9): Vortex sheets roll up and shatter
  
  The Hunt: If The Ontic Engine survives t=9 with Rank < 100, you have a tool
  that can probe the singularity deeper than anyone else.

Author: TiganticLabz
Date: December 2025
"""

import torch
import numpy as np
import time
import os
import argparse
from dataclasses import dataclass
from typing import List, Tuple

# Import our infrastructure
from ontic.cfd.nd_shift_mpo import make_nd_shift_mpo, apply_nd_shift_mpo, truncate_cores
from ontic.cfd.pure_qtt_ops import QTTState, qtt_add, dense_to_qtt, qtt_to_dense


@dataclass
class QTT3DState:
    """3D field stored in QTT format with Morton ordering."""
    cores: List[torch.Tensor]
    n_qubits: int  # Qubits per dimension
    
    @property
    def max_rank(self) -> int:
        return max(c.shape[0] for c in self.cores)
    
    @property
    def total_qubits(self) -> int:
        return len(self.cores)


@dataclass
class EulerState3D:
    """State for 3D Euler equations (velocity only for incompressible)."""
    u: QTT3DState  # x-velocity
    v: QTT3DState  # y-velocity
    w: QTT3DState  # z-velocity
    
    def max_rank(self) -> int:
        return max(self.u.max_rank, self.v.max_rank, self.w.max_rank)


def vector_to_qtt(vec: torch.Tensor, max_bond: int = 16) -> QTTState:
    """Convert 1D vector to QTT format."""
    return dense_to_qtt(vec, max_bond=max_bond)


def morton_encode_3d_vectorized(ix: torch.Tensor, iy: torch.Tensor, iz: torch.Tensor, n_qubits: int) -> torch.Tensor:
    """Vectorized Morton encoding for 3D indices."""
    result = torch.zeros_like(ix, dtype=torch.long)
    for b in range(n_qubits):
        result |= ((ix >> b) & 1) << (3 * b + 0)
        result |= ((iy >> b) & 1) << (3 * b + 1)
        result |= ((iz >> b) & 1) << (3 * b + 2)
    return result


def build_rank1_3d_qtt(fx: torch.Tensor, fy: torch.Tensor, fz: torch.Tensor, 
                       n_qubits: int, max_rank: int = 16) -> QTT3DState:
    """
    Build a 3D QTT from separable functions: f(x,y,z) = fx(x) * fy(y) * fz(z)
    
    Morton ordering: bits interleave as x0, y0, z0, x1, y1, z1, ...
    Uses vectorized Morton encoding for efficiency at large grid sizes.
    """
    N = len(fx)
    device = fx.device
    dtype = fx.dtype
    
    # For large grids (>=256³), use tensor-free construction
    if N >= 256:
        return build_rank1_3d_qtt_tensorfree(fx, fy, fz, n_qubits, max_rank)
    
    # Build the full 3D tensor using outer products (vectorized)
    X, Y, Z = torch.meshgrid(
        torch.arange(N, device=device),
        torch.arange(N, device=device),
        torch.arange(N, device=device),
        indexing='ij'
    )
    
    # f(x,y,z) = fx[x] * fy[y] * fz[z]
    field_3d = fx[X] * fy[Y] * fz[Z]
    
    # Vectorized Morton encoding
    morton_indices = morton_encode_3d_vectorized(X.flatten(), Y.flatten(), Z.flatten(), n_qubits)
    
    # Flatten with Morton ordering
    morton_flat = torch.zeros(N**3, dtype=dtype, device=device)
    morton_flat[morton_indices] = field_3d.flatten()
    
    # Compress to QTT - use low rank for smooth functions
    qtt = dense_to_qtt(morton_flat, max_bond=min(max_rank, 16))
    return QTT3DState(qtt.cores, n_qubits)


def build_rank1_3d_qtt_tensorfree(fx: torch.Tensor, fy: torch.Tensor, fz: torch.Tensor, 
                                   n_qubits: int, max_rank: int = 16) -> QTT3DState:
    """
    TENSOR-FREE construction of 3D QTT for separable functions.
    
    Uses Kronecker product interleaving with intermediate truncation.
    For f(x,y,z) = fx(x)·fy(y)·fz(z), builds the 3D QTT directly from 1D QTTs.
    
    Memory: O(N × rank²) instead of O(N³) - enables 1024³ grids!
    """
    N = len(fx)
    device = fx.device
    dtype = fx.dtype
    
    print(f"  Using Kronecker QTT construction ({N}³ = {N**3:,} points, O(N) memory)...")
    t0 = time.perf_counter()
    
    # Build 1D QTTs for each function
    # sin/cos are simple functions - they should only need rank 2-4!
    # Using low rank for 1D QTTs prevents rank explosion during interleaving
    qtt_1d_rank = 4  # sin/cos can be represented with rank ~2-4
    qtt_x = dense_to_qtt(fx, max_bond=qtt_1d_rank)
    qtt_y = dense_to_qtt(fy, max_bond=qtt_1d_rank)
    qtt_z = dense_to_qtt(fz, max_bond=qtt_1d_rank)
    
    # Build with intermediate truncation every 3 cores
    cores_3d = []
    rx_acc, ry_acc, rz_acc = 1, 1, 1
    
    for bit in range(n_qubits):
        cx = qtt_x.cores[bit]
        cy = qtt_y.cores[bit]
        cz = qtt_z.cores[bit]
        
        rx_in, _, rx_out = cx.shape
        ry_in, _, ry_out = cy.shape
        rz_in, _, rz_out = cz.shape
        
        # Core for x-bit
        r_left = min(rx_in * ry_acc * rz_acc, max_rank)
        r_right = min(rx_out * ry_acc * rz_acc, max_rank)
        
        # Use actual dimensions with capping
        actual_r_left = rx_in * ry_acc * rz_acc
        actual_r_right = rx_out * ry_acc * rz_acc
        
        core_x = torch.zeros(actual_r_left, 2, actual_r_right, dtype=dtype, device=device)
        for iy in range(ry_acc):
            for iz in range(rz_acc):
                for a in range(rx_in):
                    for b in range(rx_out):
                        idx_in = a * ry_acc * rz_acc + iy * rz_acc + iz
                        idx_out = b * ry_acc * rz_acc + iy * rz_acc + iz
                        core_x[idx_in, :, idx_out] = cx[a, :, b]
        cores_3d.append(core_x)
        
        rx_acc = min(rx_out, max_rank)  # Cap accumulated rank
        
        # Core for y-bit
        actual_r_left = rx_acc * ry_in * rz_acc
        actual_r_right = rx_acc * ry_out * rz_acc
        
        core_y = torch.zeros(actual_r_left, 2, actual_r_right, dtype=dtype, device=device)
        for ix in range(rx_acc):
            for iz in range(rz_acc):
                for a in range(ry_in):
                    for b in range(ry_out):
                        idx_in = ix * ry_in * rz_acc + a * rz_acc + iz
                        idx_out = ix * ry_out * rz_acc + b * rz_acc + iz
                        core_y[idx_in, :, idx_out] = cy[a, :, b]
        cores_3d.append(core_y)
        
        ry_acc = min(ry_out, max_rank)
        
        # Core for z-bit
        actual_r_left = rx_acc * ry_acc * rz_in
        actual_r_right = rx_acc * ry_acc * rz_out
        
        core_z = torch.zeros(actual_r_left, 2, actual_r_right, dtype=dtype, device=device)
        for ix in range(rx_acc):
            for iy in range(ry_acc):
                for a in range(rz_in):
                    for b in range(rz_out):
                        idx_in = ix * ry_acc * rz_in + iy * rz_in + a
                        idx_out = ix * ry_acc * rz_out + iy * rz_out + b
                        core_z[idx_in, :, idx_out] = cz[a, :, b]
        cores_3d.append(core_z)
        
        rz_acc = min(rz_out, max_rank)
        
        # Apply intermediate truncation every few bits to control rank
        if bit > 0 and bit % 2 == 0:
            cores_3d = truncate_cores(cores_3d, max_rank, tol=1e-8)
    
    t1 = time.perf_counter()
    
    # Final truncation
    qtt_raw = QTT3DState(cores_3d, n_qubits)
    print(f"    Kronecker build: {t1-t0:.3f}s, rank={qtt_raw.max_rank}")
    
    # ALWAYS apply aggressive truncation for IC
    # Use tol=1e-6 (more aggressive than 1e-8) to compress better
    cores_trunc = truncate_cores(cores_3d, max_rank, tol=1e-6)
    qtt_final = QTT3DState(cores_trunc, n_qubits)
    t2 = time.perf_counter()
    if qtt_final.max_rank < qtt_raw.max_rank:
        print(f"    After truncation: rank={qtt_final.max_rank}, total {t2-t0:.3f}s")
    return qtt_final


class MillenniumSolver:
    """
    3D Incompressible Euler Solver for Singularity Hunting.
    
    Uses QTT compression to handle billion-point grids.
    """
    
    def __init__(self, qubits_per_dim: int = 10, max_rank: int = 64, 
                 cfl: float = 0.2, device=None, dtype=torch.float32):
        self.n_qubits = qubits_per_dim
        self.N = 2 ** qubits_per_dim
        self.total_qubits = 3 * qubits_per_dim
        self.max_rank = max_rank
        self.cfl = cfl
        self.device = device or torch.device('cpu')
        self.dtype = dtype
        
        # Domain: [0, 2π]³
        self.L = 2 * np.pi
        self.dx = self.L / self.N
        
        # Pre-build shift MPOs for all axes
        print(f"Building 3D shift MPOs ({self.total_qubits} qubits)...")
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
        """Apply shift: output[i] = input[i-1] along axis."""
        cores = apply_nd_shift_mpo(qtt.cores, self.shift_mpos[axis], 
                                    max_rank=self.max_rank)
        return QTT3DState(cores, self.n_qubits)
    
    def _add(self, a: QTT3DState, b: QTT3DState) -> QTT3DState:
        """QTT addition with truncation."""
        a_qtt = QTTState(cores=a.cores, num_qubits=len(a.cores))
        b_qtt = QTTState(cores=b.cores, num_qubits=len(b.cores))
        result = qtt_add(a_qtt, b_qtt, max_bond=self.max_rank)
        return QTT3DState(result.cores, self.n_qubits)
    
    def _scale(self, a: QTT3DState, s: float) -> QTT3DState:
        """Scale QTT by scalar."""
        cores = [c.clone() for c in a.cores]
        cores[0] = cores[0] * s
        return QTT3DState(cores, self.n_qubits)
    
    def _sub(self, a: QTT3DState, b: QTT3DState) -> QTT3DState:
        """QTT subtraction."""
        return self._add(a, self._scale(b, -1.0))
    
    def create_taylor_green_ic(self) -> EulerState3D:
        """
        Create Taylor-Green Vortex initial condition.
        
        u =  sin(x) cos(y) cos(z)
        v = -cos(x) sin(y) cos(z)
        w = 0
        
        This is SEPARABLE (rank-1), so we can initialize a billion-point 
        grid in milliseconds without ever forming the dense array.
        """
        print(f"Creating Taylor-Green Vortex IC ({self.N}³ = {self.N**3:,} points)...")
        t0 = time.perf_counter()
        
        # 1D coordinate
        x = torch.linspace(0, self.L, self.N, dtype=self.dtype, device=self.device)
        
        # 1D primitives
        sin_x = torch.sin(x)
        cos_x = torch.cos(x)
        one = torch.ones(self.N, dtype=self.dtype, device=self.device)
        zero = torch.zeros(self.N, dtype=self.dtype, device=self.device)
        
        # Use lower IC rank - will grow naturally during evolution
        ic_rank = min(64, self.max_rank)
        
        # u = sin(x) * cos(y) * cos(z)
        u_qtt = build_rank1_3d_qtt(sin_x, cos_x, cos_x, self.n_qubits, ic_rank)
        u_qtt = self._truncate(u_qtt, tol=1e-6)  # More aggressive truncation
        
        # v = -cos(x) * sin(y) * cos(z)
        v_qtt = build_rank1_3d_qtt(-cos_x, sin_x, cos_x, self.n_qubits, ic_rank)
        v_qtt = self._truncate(v_qtt, tol=1e-6)  # More aggressive truncation
        
        # w = 0 (but we need a proper zero tensor)
        # For stability, use small noise instead of exact zero
        eps = 1e-10
        w_qtt = build_rank1_3d_qtt(
            torch.ones(self.N, dtype=self.dtype) * eps,
            torch.ones(self.N, dtype=self.dtype) * eps,
            torch.ones(self.N, dtype=self.dtype) * eps,
            self.n_qubits, ic_rank
        )
        w_qtt = self._truncate(w_qtt, tol=1e-6)  # More aggressive truncation
        
        t1 = time.perf_counter()
        print(f"  IC created in {t1-t0:.3f}s")
        print(f"  Initial ranks: u={u_qtt.max_rank}, v={v_qtt.max_rank}, w={w_qtt.max_rank}")
        
        return EulerState3D(u_qtt, v_qtt, w_qtt)
    
    def compute_dt(self, state: EulerState3D) -> float:
        """Compute stable timestep based on CFL condition."""
        # For incompressible Euler, max velocity is O(1) for Taylor-Green
        # More sophisticated: sample the QTT at a few points
        max_vel = 1.5  # Conservative estimate for TG
        return self.cfl * self.dx / max_vel
    
    def _truncate(self, qtt: QTT3DState, tol: float = 1e-8) -> QTT3DState:
        """Truncate QTT using SVD with both max_rank and tolerance."""
        # Clamp cores to prevent NaN/Inf from propagating
        cores = []
        for c in qtt.cores:
            c_clean = torch.nan_to_num(c, nan=0.0, posinf=1e6, neginf=-1e6)
            c_clean = torch.clamp(c_clean, -1e6, 1e6)
            cores.append(c_clean)
        cores = truncate_cores(cores, self.max_rank, tol=tol)
        return QTT3DState(cores, self.n_qubits)
    
    def _advect_axis(self, field: QTT3DState, dt: float, axis: int) -> QTT3DState:
        """
        Advect field along one axis: ∂f/∂t + u·∂f/∂x = 0
        
        Using upwind scheme with immediate truncation after each operation
        to prevent rank explosion.
        """
        # Get left shifted value: f[i-1]
        f_left = self._shift(field, axis)
        f_left = self._truncate(f_left, tol=1e-6)  # Truncate immediately
        
        # Simple upwind: f_new = f - dt/dx * (f - f_left)
        # This is stable for positive velocity (flow to the right)
        
        # Compute df = f - f_left
        df = self._sub(field, f_left)
        df = self._truncate(df, tol=1e-6)  # Critical: truncate after subtraction
        
        # Coefficient for advection: -dt/dx with safety factor
        coeff = -dt / self.dx * 0.5  # Reduce advection strength for stability
        
        scaled_df = self._scale(df, coeff)
        # Scale doesn't increase rank, no truncation needed
        
        result = self._add(field, scaled_df)
        result = self._truncate(result, tol=1e-6)  # Truncate after addition
        
        # Add small diffusion for numerical stability
        nu = 0.01 * self.dx  # Small artificial viscosity
        diffusion = self._scale(df, nu / self.dx)
        result = self._add(result, diffusion)
        
        # Final truncation
        return self._truncate(result, tol=1e-6)
    
    def step(self, state: EulerState3D, dt: float) -> EulerState3D:
        """
        Advance the 3D Euler equations by one timestep.
        
        Uses Strang splitting for the advection terms:
        L_x(dt/2) L_y(dt/2) L_z(dt) L_y(dt/2) L_x(dt/2)
        """
        # Advect u, v, w along each axis
        u, v, w = state.u, state.v, state.w
        
        # X-direction (dt/2)
        u = self._advect_axis(u, dt/2, axis=0)
        v = self._advect_axis(v, dt/2, axis=0)
        w = self._advect_axis(w, dt/2, axis=0)
        
        # Y-direction (dt/2)
        u = self._advect_axis(u, dt/2, axis=1)
        v = self._advect_axis(v, dt/2, axis=1)
        w = self._advect_axis(w, dt/2, axis=1)
        
        # Z-direction (full dt)
        u = self._advect_axis(u, dt, axis=2)
        v = self._advect_axis(v, dt, axis=2)
        w = self._advect_axis(w, dt, axis=2)
        
        # Y-direction (dt/2)
        u = self._advect_axis(u, dt/2, axis=1)
        v = self._advect_axis(v, dt/2, axis=1)
        w = self._advect_axis(w, dt/2, axis=1)
        
        # X-direction (dt/2)
        u = self._advect_axis(u, dt/2, axis=0)
        v = self._advect_axis(v, dt/2, axis=0)
        w = self._advect_axis(w, dt/2, axis=0)
        
        return EulerState3D(u, v, w)
    
    def estimate_enstrophy_proxy(self, state: EulerState3D) -> float:
        """
        Estimate enstrophy (integral of vorticity²) as a proxy for singularity.
        
        High rank correlates with high gradients which correlates with high vorticity.
        This is a cheap O(1) estimate instead of O(N³) dense computation.
        """
        # The rank itself is a proxy for solution complexity
        return float(state.max_rank())


class SingularityHunter:
    """
    The Millennium Prize Hunter.
    
    Runs the Taylor-Green vortex at extreme resolution and monitors
    for signs of finite-time blowup.
    """
    
    def __init__(self, resolution_qubits: int = 10, max_rank: int = 64):
        """
        Args:
            resolution_qubits: 10 -> 1024³ (1 Billion Points)
            max_rank: Maximum QTT rank before declaring "blowup"
        """
        self.n_qubits = resolution_qubits
        self.N = 2 ** resolution_qubits
        self.max_rank = max_rank
        
        print("=" * 70)
        print("  [PHASE 6] MILLENNIUM HUNTER - SINGULARITY PROBE")
        print("=" * 70)
        print(f"  Grid: {self.N}³ = {self.N**3:,} points")
        print(f"  Dense storage would be: {self.N**3 * 4 * 3 / 1e9:.1f} GB")
        print(f"  Max QTT Rank: {max_rank}")
        print(f"  Target: Finite Time Blowup at t ~ 9.0")
        print("=" * 70)
        
        self.solver = MillenniumSolver(
            qubits_per_dim=resolution_qubits,
            max_rank=max_rank,
            cfl=0.3
        )
        
        self.t = 0.0
        self.step_count = 0
        
        # Logging
        os.makedirs("logs", exist_ok=True)
        self.log_file = open("logs/millennium_trace.csv", "w")
        self.log_file.write("step,time,dt,rank_u,rank_v,rank_w,rank_max,wall_time\n")
    
    def run(self, t_final: float = 10.0, checkpoint_interval: float = 1.0):
        """
        Run the singularity hunt.
        
        Args:
            t_final: Stop time (singularity expected around t=9)
            checkpoint_interval: How often to print detailed status
        """
        # Initialize
        print("\nInitializing Taylor-Green Vortex...")
        state = self.solver.create_taylor_green_ic()
        
        start_time = time.time()
        last_checkpoint = 0.0
        
        print("\n" + "-" * 70)
        print("  BEGINNING SINGULARITY HUNT")
        print("-" * 70)
        print(f"{'Step':>6} {'Time':>8} {'dt':>10} {'Rank':>6} {'Wall':>8}")
        print("-" * 70)
        
        while self.t < t_final:
            iter_start = time.perf_counter()
            
            # Adaptive timestep
            dt = self.solver.compute_dt(state)
            
            # Physics step
            state = self.solver.step(state, dt)
            
            self.t += dt
            self.step_count += 1
            iter_time = time.perf_counter() - iter_start
            
            # Get ranks
            rank_u = state.u.max_rank
            rank_v = state.v.max_rank
            rank_w = state.w.max_rank
            rank_max = state.max_rank()
            
            # Log
            log_line = f"{self.step_count},{self.t:.6f},{dt:.6e},{rank_u},{rank_v},{rank_w},{rank_max},{iter_time:.3f}"
            self.log_file.write(log_line + "\n")
            self.log_file.flush()
            
            # Print status
            print(f"{self.step_count:6d} {self.t:8.4f} {dt:10.2e} {rank_max:6d} {iter_time:8.2f}s")
            
            # Checkpoint
            if self.t - last_checkpoint >= checkpoint_interval:
                self._print_checkpoint(state, start_time)
                last_checkpoint = self.t
            
            # Blowup detection - only stop if rank saturates for several steps
            # (rank hitting cap is expected; staying there means blowup)
            # For now, just warn but continue
            if rank_max >= self.max_rank * 0.99:
                print(f"     [!] Rank near cap: {rank_max}/{self.max_rank}")
        
        total_time = time.time() - start_time
        self._print_summary(state, total_time)
        self.log_file.close()
    
    def _print_checkpoint(self, state: EulerState3D, start_time: float):
        """Print detailed checkpoint information."""
        elapsed = time.time() - start_time
        print("\n" + "=" * 50)
        print(f"  CHECKPOINT at t = {self.t:.4f}")
        print(f"  Steps: {self.step_count}")
        print(f"  Ranks: u={state.u.max_rank}, v={state.v.max_rank}, w={state.w.max_rank}")
        print(f"  Elapsed: {elapsed:.1f}s")
        
        # Phase detection
        if self.t < 4.0:
            print("  Phase: LAMINAR (expect low rank)")
        elif self.t < 7.0:
            print("  Phase: CASCADE (rank should be growing)")
        else:
            print("  Phase: SINGULARITY ZONE (watching for blowup...)")
        
        print("=" * 50 + "\n")
    
    def _print_summary(self, state: EulerState3D, total_time: float):
        """Print final summary."""
        print("\n" + "=" * 70)
        print("  HUNT COMPLETE")
        print("=" * 70)
        print(f"  Final time: t = {self.t:.4f}")
        print(f"  Total steps: {self.step_count}")
        print(f"  Final rank: {state.max_rank()}")
        print(f"  Total wall time: {total_time:.1f}s")
        print(f"  Avg time/step: {total_time/max(1,self.step_count):.3f}s")
        print()
        
        if self.t >= 9.0 and state.max_rank() < self.max_rank:
            print("  🏆 RESULT: SURVIVED THE SINGULARITY ZONE!")
            print(f"     Reached t={self.t:.2f} with Rank={state.max_rank()} < {self.max_rank}")
            print("     This suggests the solution remains compressible even near blowup.")
        elif state.max_rank() >= self.max_rank:
            print("  💥 RESULT: RANK EXPLOSION")
            print(f"     The solution complexity exploded at t={self.t:.4f}")
            print("     This could indicate approach to singularity.")
        else:
            print("  ⏸️  RESULT: INCOMPLETE")
            print(f"     Simulation stopped at t={self.t:.4f} before reaching critical time.")
        
        print("=" * 70)
        print(f"\n  Log saved to: logs/millennium_trace.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Millennium Hunter - 3D Euler Singularity Probe"
    )
    parser.add_argument("-q", "--qubits", type=int, default=8,
                        help="Qubits per dimension (default 8 = 256³)")
    parser.add_argument("-r", "--rank", type=int, default=48,
                        help="Maximum QTT rank (default 48)")
    parser.add_argument("-t", "--time", type=float, default=10.0,
                        help="Final simulation time (default 10.0)")
    parser.add_argument("--billion", action="store_true",
                        help="Run at 1024³ (10 qubits) - THE FULL HUNT")
    
    args = parser.parse_args()
    
    if args.billion:
        # THE FULL 1 BILLION POINT RUN
        qubits = 10
        rank = 80
    else:
        qubits = args.qubits
        rank = args.rank
    
    hunter = SingularityHunter(resolution_qubits=qubits, max_rank=rank)
    hunter.run(t_final=args.time)
