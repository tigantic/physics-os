#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║                    NAVIER-STOKES BLACK SWAN HUNTER                                   ║
║                                                                                      ║
║                         Singularity Search System                                    ║
║                                                                                      ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                      ║
║  MISSION: Find a Black Swan - an initial condition that causes NS blowup            ║
║                                                                                      ║
║  STRATEGY:                                                                           ║
║  ─────────                                                                           ║
║  1. Hou-Luo Geometry: Counter-rotating vortex rings (best candidate)                ║
║  2. Gradient Ascent: Optimize IC to maximize enstrophy growth                       ║
║  3. Reynolds Scaling: Test from Re=1000 to Re=1,000,000                             ║
║  4. BKM Tracking: Monitor ∫|ω|_∞ dt - if diverges, BLOWUP!                          ║
║  5. Multi-Resolution: 32³ → 64³ → 128³ convergence study                            ║
║                                                                                      ║
║  SUCCESS CRITERIA:                                                                   ║
║  ─────────────────                                                                   ║
║  • Find IC where BKM integral diverges (singularity)                                ║
║  • OR prove BKM bounded for all tested ICs (regularity evidence)                    ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import torch
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import json
import hashlib

sys.path.insert(0, str(Path(__file__).parent))

# Import existing infrastructure
from tensornet.cfd.ns_3d import NS3DSolver
from tensornet.cfd.hou_luo_ansatz import HouLuoConfig, create_hou_luo_profile


# ═══════════════════════════════════════════════════════════════════════════════════════
# BKM CRITERION TRACKER
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class BKMResult:
    """Beale-Kato-Majda tracking result."""
    time_points: List[float]
    omega_max: List[float]  # max|ω(t)|
    bkm_integral: float     # ∫₀^T max|ω| dt
    enstrophy: List[float]  # (1/2)∫|ω|² dx
    blowup_detected: bool
    blowup_time: Optional[float] = None


class BKMTracker:
    """
    Track the Beale-Kato-Majda criterion.
    
    THEOREM (BKM 1984):
    A smooth solution u(t) of 3D Navier-Stokes exists on [0, T] if and only if
        ∫₀^T ‖ω(t)‖_∞ dt < ∞
    
    where ω = ∇ × u is the vorticity.
    
    Equivalently, blowup at time T* occurs if and only if:
        lim_{t→T*} ∫₀^t ‖ω(s)‖_∞ ds = ∞
    """
    
    def __init__(self, threshold: float = 1e6):
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        self.times = []
        self.omega_max_values = []
        self.enstrophy_values = []
        self.bkm_integral = 0.0
    
    def update(self, t: float, omega_max: float, enstrophy: float, dt: float):
        """Update BKM integral."""
        self.times.append(t)
        self.omega_max_values.append(omega_max)
        self.enstrophy_values.append(enstrophy)
        self.bkm_integral += omega_max * dt
    
    def check_blowup(self) -> Tuple[bool, Optional[float]]:
        """Check if blowup is detected."""
        if self.bkm_integral > self.threshold:
            return True, self.times[-1] if self.times else None
        
        # Also check for runaway growth rate
        if len(self.omega_max_values) > 5:
            recent = self.omega_max_values[-5:]
            growth_rate = recent[-1] / (recent[0] + 1e-10)
            if growth_rate > 100:  # 100x growth in 5 steps
                return True, self.times[-1]
        
        return False, None
    
    def get_result(self) -> BKMResult:
        blowup, blowup_time = self.check_blowup()
        return BKMResult(
            time_points=self.times.copy(),
            omega_max=self.omega_max_values.copy(),
            enstrophy=self.enstrophy_values.copy(),
            bkm_integral=self.bkm_integral,
            blowup_detected=blowup,
            blowup_time=blowup_time
        )


# ═══════════════════════════════════════════════════════════════════════════════════════
# INITIAL CONDITION GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════════════

def create_taylor_green_ic(N: int, L: float = 2*np.pi) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Taylor-Green vortex - classic test case, known to be regular."""
    x = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    
    u = np.sin(X) * np.cos(Y) * np.cos(Z)
    v = -np.cos(X) * np.sin(Y) * np.cos(Z)
    w = np.zeros_like(u)
    
    return u, v, w


def create_abc_flow_ic(N: int, L: float = 2*np.pi, 
                        A: float = 1.0, B: float = 1.0, C: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Arnold-Beltrami-Childress flow - exact Euler solution, chaotic."""
    x = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    
    u = A * np.sin(Z) + C * np.cos(Y)
    v = B * np.sin(X) + A * np.cos(Z)
    w = C * np.sin(Y) + B * np.cos(X)
    
    return u, v, w


def create_kida_vortex_ic(N: int, L: float = 2*np.pi) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kida vortex - high symmetry, proposed blowup candidate."""
    x = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    
    # Symmetric initial condition
    u = np.sin(X) * (np.cos(3*Y) * np.cos(Z) - np.cos(Y) * np.cos(3*Z))
    v = np.sin(Y) * (np.cos(3*Z) * np.cos(X) - np.cos(Z) * np.cos(3*X))
    w = np.sin(Z) * (np.cos(3*X) * np.cos(Y) - np.cos(X) * np.cos(3*Y))
    
    # Normalize
    energy = (u**2 + v**2 + w**2).mean()
    scale = 1.0 / np.sqrt(energy + 1e-10)
    
    return u * scale, v * scale, w * scale


def create_hou_luo_ic(N: int, L: float = 2*np.pi,
                       ring_radius: float = 0.4,
                       separation: float = 0.3,
                       swirl: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hou-Luo geometry - counter-rotating vortex rings.
    
    This is THE best candidate for blowup according to Hou & Luo (2014).
    """
    config = HouLuoConfig(
        N=N,
        L=L,
        ring_radius=ring_radius,
        core_thickness=0.1,
        ring_separation=separation,
        circulation=1.0,
        swirl_amplitude=swirl
    )
    
    U = create_hou_luo_profile(config)
    U_np = U.numpy()
    
    return U_np[..., 0], U_np[..., 1], U_np[..., 2]


def create_anti_parallel_vortex_tubes(N: int, L: float = 2*np.pi) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Anti-parallel vortex tubes - another blowup candidate."""
    x = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    
    # Two parallel tubes in z-direction, with opposite circulation
    tube1_center = L/3
    tube2_center = 2*L/3
    tube_radius = L/10
    
    # Distance from tube axes
    r1 = np.sqrt((X - tube1_center)**2 + (Y - L/2)**2)
    r2 = np.sqrt((X - tube2_center)**2 + (Y - L/2)**2)
    
    # Gaussian vortex cores
    omega1 = np.exp(-r1**2 / (2*tube_radius**2))
    omega2 = -np.exp(-r2**2 / (2*tube_radius**2))  # Opposite sign
    
    # Induced velocity (approximate)
    u = -(Y - L/2) * omega1 / (r1 + 0.1) + (Y - L/2) * omega2 / (r2 + 0.1)
    v = (X - tube1_center) * omega1 / (r1 + 0.1) - (X - tube2_center) * omega2 / (r2 + 0.1)
    w = np.zeros_like(u)
    
    # Project to divergence-free
    u, v, w = project_divergence_free_np(u, v, w, L)
    
    # Normalize
    energy = (u**2 + v**2 + w**2).mean()
    scale = 1.0 / np.sqrt(energy + 1e-10)
    
    return u * scale, v * scale, w * scale


def project_divergence_free_np(u: np.ndarray, v: np.ndarray, w: np.ndarray, 
                                L: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project velocity to divergence-free space."""
    N = u.shape[0]
    dx = L / N
    
    k = np.fft.fftfreq(N, dx) * 2 * np.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    k_sq = kx**2 + ky**2 + kz**2
    k_sq[0, 0, 0] = 1.0
    
    u_hat = np.fft.fftn(u)
    v_hat = np.fft.fftn(v)
    w_hat = np.fft.fftn(w)
    
    div_hat = 1j * (kx * u_hat + ky * v_hat + kz * w_hat)
    P_hat = div_hat / k_sq
    P_hat[0, 0, 0] = 0
    
    u_hat -= 1j * kx * P_hat
    v_hat -= 1j * ky * P_hat
    w_hat -= 1j * kz * P_hat
    
    return np.fft.ifftn(u_hat).real, np.fft.ifftn(v_hat).real, np.fft.ifftn(w_hat).real


# ═══════════════════════════════════════════════════════════════════════════════════════
# BLACK SWAN HUNTER
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class HuntConfig:
    """Configuration for Black Swan hunt."""
    N: int = 48                    # Grid resolution
    L: float = 2 * np.pi          # Domain size
    Re: float = 10000             # Reynolds number
    T_final: float = 2.0          # Integration time
    dt: float = 0.005             # Time step
    bkm_threshold: float = 1e6    # BKM blowup threshold


@dataclass
class HuntResult:
    """Result from single hunt."""
    ic_name: str
    Re: float
    N: int
    T_final: float
    bkm_integral: float
    max_omega: float
    max_enstrophy: float
    blowup_detected: bool
    verdict: str
    runtime_sec: float


class BlackSwanHunter:
    """
    The Black Swan Hunter.
    
    Systematically search for initial conditions that cause Navier-Stokes
    solutions to blow up (develop singularities).
    """
    
    def __init__(self, config: HuntConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self.results: List[HuntResult] = []
    
    def log(self, msg: str):
        if self.verbose:
            print(msg)
    
    def hunt_single_ic(self, u0: np.ndarray, v0: np.ndarray, w0: np.ndarray,
                        ic_name: str) -> HuntResult:
        """Hunt for singularity with a single initial condition."""
        import time
        start_time = time.time()
        
        N = self.config.N
        L = self.config.L
        nu = 1.0 / self.config.Re
        dt = self.config.dt
        
        # Create solver
        solver = NS3DSolver(
            Nx=N, Ny=N, Nz=N,
            Lx=L, Ly=L, Lz=L,
            nu=nu
        )
        
        # Import NSState3D
        from tensornet.cfd.ns_3d import NSState3D
        
        # Initialize state
        state = NSState3D(
            u=torch.from_numpy(u0).double(),
            v=torch.from_numpy(v0).double(),
            w=torch.from_numpy(w0).double(),
            t=0.0,
            step=0
        )
        
        # BKM tracker
        bkm = BKMTracker(threshold=self.config.bkm_threshold)
        
        n_steps = int(self.config.T_final / dt)
        
        self.log(f"\n  Running {ic_name}: Re={self.config.Re:.0f}, N={N}, T={self.config.T_final}")
        
        for step in range(n_steps):
            t = state.t
            
            # Compute vorticity
            u_np = state.u.numpy()
            v_np = state.v.numpy()
            w_np = state.w.numpy()
            omega_x, omega_y, omega_z = self._compute_vorticity(u_np, v_np, w_np, L, N)
            omega_mag = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
            
            # Track BKM
            omega_max = omega_mag.max()
            enstrophy = 0.5 * (omega_mag**2).mean() * L**3
            bkm.update(t, omega_max, enstrophy, dt)
            
            # Check for blowup
            blowup, blowup_time = bkm.check_blowup()
            if blowup:
                self.log(f"    ⚠️  BLOWUP DETECTED at t={blowup_time:.4f}!")
                break
            
            # Step solver using RK4
            state, _ = solver.step_rk4(state, dt)
            
            # Progress
            if step > 0 and step % (n_steps // 5) == 0:
                self.log(f"    t={t:.3f}: max|ω|={omega_max:.4f}, BKM={bkm.bkm_integral:.4f}")
        
        runtime = time.time() - start_time
        bkm_result = bkm.get_result()
        
        # Verdict
        if bkm_result.blowup_detected:
            verdict = "⚠️  SINGULARITY CANDIDATE"
        else:
            verdict = "✓ BOUNDED"
        
        result = HuntResult(
            ic_name=ic_name,
            Re=self.config.Re,
            N=self.config.N,
            T_final=self.config.T_final,
            bkm_integral=bkm_result.bkm_integral,
            max_omega=max(bkm_result.omega_max) if bkm_result.omega_max else 0,
            max_enstrophy=max(bkm_result.enstrophy) if bkm_result.enstrophy else 0,
            blowup_detected=bkm_result.blowup_detected,
            verdict=verdict,
            runtime_sec=runtime
        )
        
        self.results.append(result)
        self.log(f"    Verdict: {verdict} (BKM={bkm_result.bkm_integral:.4f}, max|ω|={result.max_omega:.4f})")
        
        return result
    
    def _compute_vorticity(self, u: np.ndarray, v: np.ndarray, w: np.ndarray,
                            L: float, N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute vorticity ω = ∇ × u using spectral derivatives."""
        dx = L / N
        k = np.fft.fftfreq(N, dx) * 2 * np.pi
        kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
        
        u_hat = np.fft.fftn(u)
        v_hat = np.fft.fftn(v)
        w_hat = np.fft.fftn(w)
        
        # ω = ∇ × u = (∂w/∂y - ∂v/∂z, ∂u/∂z - ∂w/∂x, ∂v/∂x - ∂u/∂y)
        omega_x = np.fft.ifftn(1j * ky * w_hat - 1j * kz * v_hat).real
        omega_y = np.fft.ifftn(1j * kz * u_hat - 1j * kx * w_hat).real
        omega_z = np.fft.ifftn(1j * kx * v_hat - 1j * ky * u_hat).real
        
        return omega_x, omega_y, omega_z
    
    def hunt_all_candidates(self) -> Dict:
        """Hunt across all IC candidates."""
        N = self.config.N
        L = self.config.L
        
        self.log("\n" + "═" * 70)
        self.log("BLACK SWAN HUNT: Searching for singularity candidates")
        self.log("═" * 70)
        
        # List of candidate ICs
        candidates = [
            ("Taylor-Green", create_taylor_green_ic),
            ("ABC Flow", create_abc_flow_ic),
            ("Kida Vortex", create_kida_vortex_ic),
            ("Anti-Parallel Tubes", create_anti_parallel_vortex_tubes),
            ("Hou-Luo (standard)", lambda n, l: create_hou_luo_ic(n, l, 0.4, 0.3, 0.5)),
            ("Hou-Luo (tight)", lambda n, l: create_hou_luo_ic(n, l, 0.3, 0.2, 0.8)),
            ("Hou-Luo (wide)", lambda n, l: create_hou_luo_ic(n, l, 0.5, 0.4, 0.3)),
        ]
        
        for name, ic_func in candidates:
            u0, v0, w0 = ic_func(N, L)
            self.hunt_single_ic(u0, v0, w0, name)
        
        # Summary
        n_blowup = sum(1 for r in self.results if r.blowup_detected)
        n_bounded = len(self.results) - n_blowup
        
        return {
            "total": len(self.results),
            "blowup": n_blowup,
            "bounded": n_bounded,
            "results": self.results
        }


# ═══════════════════════════════════════════════════════════════════════════════════════
# REYNOLDS SCALING ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════════════

def reynolds_scaling_hunt(Re_values: List[float] = None, N: int = 48) -> Dict:
    """
    Test scaling of BKM integral with Reynolds number.
    
    If NS is regular: BKM should stay bounded as Re → ∞
    If NS can blow up: BKM might diverge at some critical Re
    """
    if Re_values is None:
        Re_values = [1000, 5000, 10000, 50000, 100000]
    
    print("\n" + "═" * 70)
    print("REYNOLDS SCALING ANALYSIS")
    print("═" * 70)
    print(f"\nTesting Re = {Re_values}")
    print("Using Hou-Luo geometry (best blowup candidate)")
    
    results = []
    
    for Re in Re_values:
        config = HuntConfig(
            N=N,
            Re=Re,
            T_final=1.0,
            dt=0.002
        )
        
        hunter = BlackSwanHunter(config, verbose=True)
        u0, v0, w0 = create_hou_luo_ic(N, config.L)
        result = hunter.hunt_single_ic(u0, v0, w0, f"Hou-Luo Re={Re:.0f}")
        results.append(result)
    
    # Analyze scaling
    print("\n" + "═" * 70)
    print("SCALING ANALYSIS")
    print("═" * 70)
    
    Re_arr = np.array([r.Re for r in results])
    bkm_arr = np.array([r.bkm_integral for r in results])
    
    # Fit power law: BKM ~ Re^α
    log_Re = np.log(Re_arr)
    log_bkm = np.log(bkm_arr + 1e-10)
    
    # Linear regression
    from numpy.polynomial import polynomial as P
    coeffs = P.polyfit(log_Re, log_bkm, 1)
    alpha = coeffs[1]  # Power law exponent
    
    print(f"\n  Power law fit: BKM ~ Re^{alpha:.4f}")
    
    if alpha < 0.1:
        print("  → BKM nearly constant with Re → EVIDENCE FOR REGULARITY")
    elif alpha < 0.5:
        print("  → Weak growth with Re → INCONCLUSIVE")
    else:
        print("  → Strong growth with Re → POTENTIAL SINGULARITY FORMATION")
    
    return {
        "Re_values": Re_values,
        "results": results,
        "scaling_exponent": alpha,
        "all_bounded": all(not r.blowup_detected for r in results)
    }


# ═══════════════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════════════

def run_black_swan_hunt():
    """Execute the complete Black Swan hunt."""
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "NAVIER-STOKES BLACK SWAN HUNTER" + " " * 30 + "║")
    print("║" + " " * 78 + "║")
    print("║  Searching for initial conditions that cause singularities..." + " " * 13 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # Hunt configuration
    config = HuntConfig(
        N=48,           # Grid resolution
        Re=10000,       # Reynolds number
        T_final=2.0,    # Integration time
        dt=0.005        # Time step
    )
    
    # Run hunt
    hunter = BlackSwanHunter(config, verbose=True)
    hunt_results = hunter.hunt_all_candidates()
    
    # Reynolds scaling
    print("\n")
    scaling_results = reynolds_scaling_hunt(
        Re_values=[1000, 5000, 10000, 50000],
        N=48
    )
    
    # Final verdict
    print("\n" + "═" * 80)
    print("FINAL VERDICT")
    print("═" * 80)
    
    total_tests = hunt_results["total"] + len(scaling_results["results"])
    total_blowups = hunt_results["blowup"] + sum(1 for r in scaling_results["results"] if r.blowup_detected)
    
    if total_blowups > 0:
        print()
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 78 + "║")
        print("║" + " " * 15 + "⚠️  BLACK SWAN CANDIDATE FOUND! ⚠️" + " " * 25 + "║")
        print("║" + " " * 78 + "║")
        print(f"║  {total_blowups} out of {total_tests} tests showed potential blowup" + " " * 30 + "║")
        print("║  Further investigation required with higher resolution" + " " * 22 + "║")
        print("╚" + "═" * 78 + "╝")
    else:
        print()
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 78 + "║")
        print("║" + " " * 15 + "NO BLACK SWAN FOUND" + " " * 42 + "║")
        print("║" + " " * 78 + "║")
        print(f"║  All {total_tests} tests showed bounded behavior" + " " * 36 + "║")
        print(f"║  BKM scaling: ~ Re^{scaling_results['scaling_exponent']:.4f}" + " " * 43 + "║")
        print("║  This is consistent with global regularity of Navier-Stokes" + " " * 17 + "║")
        print("║" + " " * 78 + "║")
        print("║  NOTE: Absence of evidence ≠ evidence of absence" + " " * 28 + "║")
        print("║  Higher resolution and longer times may reveal singularities" + " " * 16 + "║")
        print("╚" + "═" * 78 + "╝")
    
    return {
        "hunt_results": hunt_results,
        "scaling_results": scaling_results,
        "black_swan_found": total_blowups > 0
    }


if __name__ == "__main__":
    results = run_black_swan_hunt()
    
    # Export results
    output = {
        "timestamp": datetime.now().isoformat(),
        "hunt_summary": {
            "total_ics": results["hunt_results"]["total"],
            "blowups": results["hunt_results"]["blowup"],
            "bounded": results["hunt_results"]["bounded"],
        },
        "scaling": {
            "exponent": results["scaling_results"]["scaling_exponent"],
            "all_bounded": results["scaling_results"]["all_bounded"]
        },
        "verdict": "BLACK_SWAN_FOUND" if results["black_swan_found"] else "ALL_BOUNDED"
    }
    
    with open("black_swan_hunt_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to black_swan_hunt_results.json")
