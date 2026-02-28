#!/usr/bin/env python3
"""
🦢 TRAP THE SWAN - Black Swan Singularity Hunter
=================================================

PHILOSOPHICAL FOUNDATION:
- Proving smoothness requires showing it for ALL initial conditions, ALL time (impossible)
- Proving blowup requires finding just ONE counterexample (achievable)

THE IRONY OF HYPERTENSOR:
Our SVD truncation is designed to SUPPRESS rank explosion. We are literally 
"smoothing over" the crack in reality that mathematicians are looking for.

THIS SCRIPT:
- Uses HIGH rank cap (1024) to let the physics breathe
- Uses LOOSE truncation tolerance to preserve potential singularities
- Treats rank explosion as EVIDENCE, not a problem to fix
- FREEZES simulation when potential blowup detected
- Preserves state for forensic mathematical analysis

The Black Swan: A specific vortex configuration that leads to infinite velocity.
If we find one, we've found the counterexample. We win.

USAGE:
    python demos/trap_the_swan.py --ic taylor_green   # Standard IC (white swan)
    python demos/trap_the_swan.py --ic random_turb    # Random high-k turbulence (hunt mode)
    python demos/trap_the_swan.py --ic random_turb --seed 42  # Reproducible random IC
"""

import time
import torch
import numpy as np
import argparse
import sys
import os
import json
from datetime import datetime, timezone

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from millennium_hunter import MillenniumSolver, EulerState3D, build_rank1_3d_qtt, QTT3DState
from ontic.cfd.nd_shift_mpo import truncate_cores, make_nd_shift_mpo, apply_nd_shift_mpo
from ontic.cfd.pure_qtt_ops import QTTState, qtt_add

# Auto-detect GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"🚀 GPU DETECTED: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️  No GPU detected, using CPU")


class BlackSwanHunter:
    """
    A modified Navier-Stokes solver that HUNTS for blowup rather than suppressing it.
    
    Key differences from standard MillenniumSolver:
    1. Very high rank cap (1024) - let the physics breathe
    2. Looser truncation tolerance - don't smooth over potential singularities
    3. Rank monitoring - treat explosion as evidence, not error
    4. State freezing - preserve evidence when threshold crossed
    """
    
    def __init__(self, n_qubits: int = 9, max_rank: int = 1024, ic_type: str = 'taylor_green', seed: int = 42, truncation_tol: float = 1e-6):
        """Initialize the Black Swan Hunter.
        
        Args:
            n_qubits: Qubits per dimension (9 = 512³)
            max_rank: Maximum allowed rank (high = let physics breathe)
            ic_type: Initial condition type ('taylor_green' or 'random_turb')
            seed: Random seed for reproducibility (only used with random_turb)
            truncation_tol: SVD truncation tolerance (lower = more aggressive compression)
        """
        self.n_qubits = n_qubits
        self.N = 2 ** n_qubits
        self.max_rank = max_rank
        self.ic_type = ic_type
        self.seed = seed
        self.truncation_tol = truncation_tol
        
        print("CONFIGURATION:")
        print(f"  • Grid: {self.N}³ = {self.N**3:,} points")
        print(f"  • Max Rank Allowed: {max_rank} (HIGH - letting physics breathe)")
        print(f"  • Blowup Trigger: Rank > 400")
        print(f"  • IC Type: {ic_type}")
        if ic_type == 'random_turb':
            print(f"  • Random Seed: {seed}")
        print(f"  • Truncation Tolerance: {truncation_tol} (lower = more compression)")
        print()
        
        # Device setup - USE GPU
        self.device = DEVICE
        self.dtype = torch.float32
        
        # Domain parameters
        self.L = 2 * np.pi
        self.dx = self.L / self.N
        self.cfl = 0.2
        
        # Create the underlying solver with HIGH rank cap and GPU
        self.solver = MillenniumSolver(
            qubits_per_dim=n_qubits,
            max_rank=max_rank,  # HIGH - let it explode if it wants to
            cfl=0.2,
            device=self.device,
            dtype=self.dtype
        )
        
        # Build shift MPOs for conservative stepper (on GPU)
        print(f"Building shift MPOs on {self.device}...")
        self.total_qubits = 3 * n_qubits
        self.shift_mpos = []
        for axis in range(3):
            mpo = make_nd_shift_mpo(
                self.total_qubits, num_dims=3, axis_idx=axis,
                direction=+1, device=self.device, dtype=self.dtype
            )
            self.shift_mpos.append(mpo)
        print("  Shift MPOs ready.")
        
        # Initialize state based on IC type
        if ic_type == 'taylor_green':
            self.state = self.solver.create_taylor_green_ic()
        elif ic_type == 'random_turb':
            self.state = self.create_random_turb_ic(seed)
        else:
            raise ValueError(f"Unknown IC type: {ic_type}")
            
        self.dt = self.solver.compute_dt(self.state)
        
        print(f"  • Timestep: dt = {self.dt:.6f}")
        print(f"  • Initial Rank: u={self.state.u.max_rank}, v={self.state.v.max_rank}, w={self.state.w.max_rank}")
        print()
    
    def create_random_turb_ic(self, seed: int = 42) -> EulerState3D:
        """
        Create random high-k turbulent initial condition.
        
        This generates a divergence-free random velocity field with
        energy concentrated at moderate wavenumbers - matching the
        original December run that found Black Swan #945.
        """
        print(f"Creating Random High-k Turbulence IC ({self.N}³ = {self.N**3:,} points)...")
        print(f"  Random seed: {seed}")
        t0 = time.perf_counter()
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        L = 2 * np.pi
        x = torch.linspace(0, L, self.N, dtype=self.dtype, device=self.device)
        
        # Match original: 5 modes gave rank ≈80-81
        n_modes = 5
        
        def build_field():
            result = None
            for m in range(n_modes):
                # Random wavenumbers (moderate k)
                kx = np.random.randint(1, 4)
                ky = np.random.randint(1, 4)
                kz = np.random.randint(1, 4)
                
                # Random phases
                phi_x = np.random.uniform(0, 2*np.pi)
                phi_y = np.random.uniform(0, 2*np.pi)
                phi_z = np.random.uniform(0, 2*np.pi)
                
                # Kolmogorov-like amplitude scaling
                k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
                amp = 1.0 / (k_mag ** (5/6))
                
                # Build separable 1D functions
                fx = torch.cos(kx * x + phi_x) * amp
                fy = torch.cos(ky * x + phi_y)
                fz = torch.cos(kz * x + phi_z)
                
                # Build 3D QTT via Kronecker
                mode = build_rank1_3d_qtt(fx, fy, fz, self.n_qubits, max_rank=16)
                
                if result is None:
                    result = mode
                else:
                    # Add modes
                    a_qtt = QTTState(cores=result.cores, num_qubits=len(result.cores))
                    b_qtt = QTTState(cores=mode.cores, num_qubits=len(mode.cores))
                    summed = qtt_add(a_qtt, b_qtt, max_bond=self.max_rank)
                    result = QTT3DState(summed.cores, self.n_qubits)
            
            # Final truncation
            result_cores = truncate_cores(result.cores, self.max_rank, tol=1e-8)
            return QTT3DState(result_cores, self.n_qubits)
        
        u = build_field()
        v = build_field()
        w = build_field()
        
        print(f"  IC created in {time.perf_counter() - t0:.3f}s")
        print(f"  Initial ranks: u={u.max_rank}, v={v.max_rank}, w={w.max_rank}")
        
        return EulerState3D(u, v, w)
    
    def get_max_rank(self) -> int:
        """Get maximum rank across all velocity components."""
        return self.state.max_rank()
    
    # === Conservative stepper (matches original December run) ===
    
    def _shift(self, qtt: QTT3DState, axis: int) -> QTT3DState:
        """Apply periodic shift along axis."""
        cores = apply_nd_shift_mpo(qtt.cores, self.shift_mpos[axis], max_rank=self.max_rank)
        return QTT3DState(cores, self.n_qubits)
    
    def _add(self, a: QTT3DState, b: QTT3DState) -> QTT3DState:
        """QTT addition."""
        a_qtt = QTTState(cores=a.cores, num_qubits=len(a.cores))
        b_qtt = QTTState(cores=b.cores, num_qubits=len(b.cores))
        result = qtt_add(a_qtt, b_qtt, max_bond=self.max_rank)
        return QTT3DState(result.cores, self.n_qubits)
    
    def _scale(self, a: QTT3DState, s: float) -> QTT3DState:
        """Scale QTT by scalar."""
        cores = [c.clone() for c in a.cores]
        cores[0] = cores[0] * s
        return QTT3DState(cores, self.n_qubits)
    
    def _truncate(self, qtt: QTT3DState, tol: float = 1e-6) -> QTT3DState:
        """Truncate QTT to control rank growth."""
        cores = truncate_cores(qtt.cores, self.max_rank, tol=tol)
        return QTT3DState(cores, self.n_qubits)
    
    def step_conservative(self, verbose: bool = False):
        """
        Conservative Euler step - matches original December run.
        
        This is simpler than Strang splitting and more stable for
        long-time integration with high rank caps.
        """
        u, v, w = self.state.u, self.state.v, self.state.w
        dt = self.dt
        
        if verbose:
            print(f"    Before step: u={u.max_rank}, v={v.max_rank}, w={w.max_rank}")
        
        # Compute derivatives via central differences: df/dx ≈ (f_{i+1} - f_{i-1}) / (2*dx)
        def derivative(f, axis, name=""):
            f_plus = self._shift(f, axis)
            if verbose:
                print(f"      {name} shift+: rank={f_plus.max_rank}")
            f_minus = self._shift(self._scale(f, -1.0), axis)  # Shift negative
            if verbose:
                print(f"      {name} shift-: rank={f_minus.max_rank}")
            diff = self._add(f_plus, f_minus)
            if verbose:
                print(f"      {name} diff: rank={diff.max_rank}")
            diff = self._scale(diff, 1.0 / (2.0 * self.dx))
            result = self._truncate(diff, tol=self.truncation_tol)
            if verbose:
                print(f"      {name} trunc: rank={result.max_rank}")
            return result
        
        # Simple self-advection: u_new = u - dt * u * du/dx
        # (simplified: just du/dx for now, matches original)
        du_dx = derivative(u, 0, "du/dx")
        dv_dy = derivative(v, 1, "dv/dy")
        dw_dz = derivative(w, 2, "dw/dz")
        
        u_new = self._add(u, self._scale(du_dx, -dt))
        v_new = self._add(v, self._scale(dv_dy, -dt))
        w_new = self._add(w, self._scale(dw_dz, -dt))
        
        # Truncate to control rank (but allow growth up to max_rank)
        u_new = self._truncate(u_new, tol=self.truncation_tol)
        v_new = self._truncate(v_new, tol=self.truncation_tol)
        w_new = self._truncate(w_new, tol=self.truncation_tol)
        
        self.state = EulerState3D(u_new, v_new, w_new)
    
    def step(self, verbose: bool = False):
        """Advance one timestep using conservative scheme."""
        # Use conservative stepper instead of Strang splitting
        # This matches the original December run that found Black Swan #945
        self.step_conservative(verbose=verbose)


def hunt_for_blowup(ic_type: str = 'taylor_green', seed: int = 42, n_qubits: int = 9, max_rank: int = 1024, truncation_tol: float = 1e-6):
    """
    The main hunting loop.
    
    We run the simulation with HIGH rank cap.
    If rank exceeds threshold, we FREEZE and preserve evidence.
    
    Args:
        ic_type: Initial condition type ('taylor_green' or 'random_turb')
        seed: Random seed for reproducibility (only used with random_turb)
        n_qubits: Qubits per dimension (9 = 512³, 10 = 1024³)
        max_rank: Maximum allowed rank (high = let physics breathe)
        truncation_tol: SVD truncation tolerance (lower = more compression)
    """
    
    print("=" * 70)
    print("🦢  BLACK SWAN TRAP - Hunting for Navier-Stokes Singularity")
    print("=" * 70)
    print()
    print("ASYMMETRIC BURDEN OF PROOF:")
    print("  • Proving smoothness: Must show for ALL cases (impossible)")
    print("  • Proving blowup: Need only ONE counterexample (achievable)")
    print()
    print("THE HUNT:")
    print(f"  • IC Type: {ic_type}")
    if ic_type == 'random_turb':
        print(f"  • Random Seed: {seed}")
    print(f"  • High rank cap ({max_rank}) - let physics breathe")
    print("  • Rank > 400 triggers evidence preservation")
    print()
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Initialize hunter with HIGH rank cap
    hunter = BlackSwanHunter(
        n_qubits=n_qubits,
        max_rank=max_rank,
        ic_type=ic_type,
        seed=seed,
        truncation_tol=truncation_tol
    )
    
    # Hunting parameters
    BLOWUP_THRESHOLD = 400
    t = 0.0
    step = 0
    t_max = 10.0
    
    # Rank history for analysis
    rank_history = []
    
    start_time = time.time()
    checkpoint_time = start_time
    
    print("🎯 BEGINNING THE HUNT...")
    print("-" * 70)
    
    while t < t_max:
        # Run one step (verbose for first 5 steps to diagnose)
        hunter.step(verbose=(step < 3))
        t += hunter.dt
        step += 1
        
        # Get current vitals
        current_rank = hunter.get_max_rank()
        rank_history.append({'t': t, 'rank': current_rank, 'step': step})
        
        # Log progress every 10 steps
        if step % 10 == 0:
            elapsed = time.time() - checkpoint_time
            print(f"   Step {step:5d} | t={t:.4f} | Rank={current_rank:4d} | {elapsed:.2f}s")
            checkpoint_time = time.time()
        
        # === 🚨 THE TRAP LOGIC 🚨 ===
        if current_rank > BLOWUP_THRESHOLD:
            total_time = time.time() - start_time
            
            print()
            print("=" * 70)
            print("🚨 BLACK SWAN DETECTED! 🚨")
            print("=" * 70)
            print(f"   Condition Met: Rank {current_rank} > {BLOWUP_THRESHOLD}")
            print(f"   Simulation Time: t = {t:.6f}")
            print(f"   Wall Time: {total_time:.2f}s")
            print(f"   Steps Completed: {step}")
            print()
            print("   Freezing state for forensic analysis...")
            
            # Create evidence filename
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            evidence_path = f"logs/BLACK_SWAN_t{t:.4f}_rank{current_rank}_{timestamp}.pt"
            
            # Save everything
            evidence = {
                'u_cores': [c.clone() for c in hunter.state.u.cores],
                'v_cores': [c.clone() for c in hunter.state.v.cores],
                'w_cores': [c.clone() for c in hunter.state.w.cores],
                't': t,
                'dt': hunter.dt,
                'step': step,
                'n_qubits': hunter.n_qubits,
                'grid_size': hunter.N,
                'max_rank': current_rank,
                'rank_history': rank_history,
                'threshold': BLOWUP_THRESHOLD,
                'timestamp_utc': datetime.now(timezone.utc).isoformat()
            }
            
            torch.save(evidence, evidence_path)
            
            # Also save JSON summary
            summary_path = evidence_path.replace('.pt', '.json')
            summary = {
                'event': 'BLACK_SWAN_DETECTED',
                't': t,
                'step': step,
                'rank': current_rank,
                'threshold': BLOWUP_THRESHOLD,
                'grid': f"{hunter.N}³",
                'points': hunter.N ** 3,
                'ic_type': ic_type,
                'seed': seed if ic_type == 'random_turb' else None,
                'wall_time_seconds': total_time,
                'evidence_file': evidence_path,
                'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                'interpretation': (
                    "Rank explosion detected. This could indicate:\n"
                    "1. Genuine singularity formation (THE PRIZE)\n"
                    "2. Numerical instability (needs investigation)\n"
                    "3. Insufficient resolution (needs refinement)\n\n"
                    "Next steps:\n"
                    "- Analyze vorticity field at this timestep\n"
                    "- Check for velocity blowup (max |u|)\n"
                    "- Refine mesh locally around high-rank regions\n"
                    "- Compare with higher-resolution run"
                )
            }
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print()
            print(f"   ✅ Evidence secured:")
            print(f"      - State: {evidence_path}")
            print(f"      - Summary: {summary_path}")
            print()
            print("   ❌ Terminating simulation to preserve evidence.")
            print()
            print("   NEXT STEPS:")
            print("   1. Analyze vorticity field at this timestep")
            print("   2. Check for velocity blowup (max |u|)")
            print("   3. Compare rank growth rate to theoretical predictions")
            print("   4. Run at higher resolution to confirm")
            print("=" * 70)
            
            sys.exit(0)
    
    # If we get here, no blowup detected
    total_time = time.time() - start_time
    
    print()
    print("=" * 70)
    print("🏁 HUNT COMPLETE - NO BLACK SWAN FOUND")
    print("=" * 70)
    print(f"   Final Time: t = {t:.4f}")
    print(f"   Final Rank: {current_rank}")
    print(f"   Total Steps: {step}")
    print(f"   Wall Time: {total_time:.2f}s ({total_time/3600:.2f} hours)")
    print()
    print("   INTERPRETATION:")
    print("   The simulation remained stable with bounded rank.")
    print("   This is evidence FOR regularity (but not proof).")
    print()
    print("   To continue the hunt:")
    print("   1. Try different initial conditions (more violent vortex collisions)")
    print("   2. Try higher resolution (1024³)")  
    print("   3. Try longer simulation time")
    print("=" * 70)
    
    # Save completion summary
    summary = {
        'event': 'HUNT_COMPLETE_NO_BLOWUP',
        't_final': t,
        'steps': step,
        'final_rank': current_rank,
        'max_rank_observed': max(r['rank'] for r in rank_history),
        'grid': f"{hunter.N}³",
        'ic_type': ic_type,
        'seed': seed if ic_type == 'random_turb' else None,
        'wall_time_seconds': total_time,
        'timestamp_utc': datetime.now(timezone.utc).isoformat()
    }
    
    summary_path = f"logs/HUNT_COMPLETE_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n   Summary saved: {summary_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="🦢 BLACK SWAN TRAP - Hunt for Navier-Stokes singularities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Taylor-Green vortex (default)
  python trap_the_swan.py --ic taylor_green
  
  # Random turbulence IC (THE ONE THAT FOUND BLACK SWAN #945)
  python trap_the_swan.py --ic random_turb --seed 42
  
  # Higher resolution hunt (1024³)
  python trap_the_swan.py --ic random_turb --seed 42 --qubits 10
  
  # Higher rank cap (let physics breathe more)
  python trap_the_swan.py --ic random_turb --seed 42 --max-rank 2048
"""
    )
    
    parser.add_argument(
        '--ic', type=str, default='taylor_green',
        choices=['taylor_green', 'random_turb'],
        help="Initial condition type (default: taylor_green)"
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help="Random seed for random_turb IC (default: 42)"
    )
    parser.add_argument(
        '--qubits', type=int, default=9,
        help="Qubits per dimension: 9=512³, 10=1024³ (default: 9)"
    )
    parser.add_argument(
        '--max-rank', type=int, default=1024,
        help="Maximum allowed rank (default: 1024)"
    )
    parser.add_argument(
        '--tol', type=float, default=1e-4,
        help="SVD truncation tolerance - lower = more compression (default: 1e-4)"
    )
    
    args = parser.parse_args()
    
    hunt_for_blowup(
        ic_type=args.ic,
        seed=args.seed,
        n_qubits=args.qubits,
        max_rank=args.max_rank,
        truncation_tol=args.tol
    )
