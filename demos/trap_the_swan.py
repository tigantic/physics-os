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
"""

import time
import torch
import sys
import os
import json
from datetime import datetime, timezone

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from millennium_hunter import MillenniumSolver, EulerState3D


class BlackSwanHunter:
    """
    A modified Navier-Stokes solver that HUNTS for blowup rather than suppressing it.
    
    Key differences from standard MillenniumSolver:
    1. Very high rank cap (1024) - let the physics breathe
    2. Looser truncation tolerance - don't smooth over potential singularities
    3. Rank monitoring - treat explosion as evidence, not error
    4. State freezing - preserve evidence when threshold crossed
    """
    
    def __init__(self, n_qubits: int = 9, max_rank: int = 1024):
        """Initialize the Black Swan Hunter.
        
        Args:
            n_qubits: Qubits per dimension (9 = 512³)
            max_rank: Maximum allowed rank (high = let physics breathe)
        """
        self.n_qubits = n_qubits
        self.N = 2 ** n_qubits
        self.max_rank = max_rank
        
        print("CONFIGURATION:")
        print(f"  • Grid: {self.N}³ = {self.N**3:,} points")
        print(f"  • Max Rank Allowed: {max_rank} (HIGH - letting physics breathe)")
        print(f"  • Blowup Trigger: Rank > 400")
        print()
        
        # Create the underlying solver with HIGH rank cap
        self.solver = MillenniumSolver(
            qubits_per_dim=n_qubits,
            max_rank=max_rank,  # HIGH - let it explode if it wants to
            cfl=0.2
        )
        
        # Initialize state
        self.state = self.solver.create_taylor_green_ic()
        self.dt = self.solver.compute_dt(self.state)
        
        print(f"  • Timestep: dt = {self.dt:.6f}")
        print()
    
    def get_max_rank(self) -> int:
        """Get maximum rank across all velocity components."""
        return self.state.max_rank()
    
    def step(self):
        """Advance one timestep."""
        self.state = self.solver.step(self.state, self.dt)


def hunt_for_blowup():
    """
    The main hunting loop.
    
    We run the simulation with HIGH rank cap.
    If rank exceeds threshold, we FREEZE and preserve evidence.
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
    print("  • High rank cap (1024) - let physics breathe")
    print("  • Rank > 400 triggers evidence preservation")
    print()
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Initialize hunter with HIGH rank cap
    hunter = BlackSwanHunter(
        n_qubits=9,      # 512³
        max_rank=1024    # HIGH - let it explode if it wants to
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
        # Run one step
        hunter.step()
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
        'wall_time_seconds': total_time,
        'timestamp_utc': datetime.now(timezone.utc).isoformat()
    }
    
    summary_path = f"logs/HUNT_COMPLETE_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n   Summary saved: {summary_path}")


if __name__ == "__main__":
    hunt_for_blowup()
