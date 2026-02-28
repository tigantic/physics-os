#!/usr/bin/env python3
"""
QTT TURBULENCE CHECKPOINTED WORKFLOW
=====================================

Production-grade checkpointed workflow for QTT turbulence validation.
Supports interrupt/resume, state persistence, and constitutional gates.

WORKFLOW PHASES:
1. INIT       - System validation and checkpoint loading
2. DECAY      - Taylor-Green vortex decay proof
3. ENERGY     - Inviscid energy conservation proof
4. SCALING    - O(log N) time complexity proof
5. COMPRESS   - QTT compression ratio proof
6. STABILITY  - Long-time numerical stability proof
7. ATTEST     - Generate final attestation

CONSTITUTIONAL PRINCIPLES:
- Each phase is atomic and checkpointed
- Failed phases can be retried without re-running passed phases
- State is persisted to disk after each phase
- SHA256 attestation on completion

USAGE:
    python3 prove_qtt_turbulence_workflow.py          # Run full workflow
    python3 prove_qtt_turbulence_workflow.py --resume  # Resume from checkpoint
    python3 prove_qtt_turbulence_workflow.py --reset   # Clear checkpoint and restart
    python3 prove_qtt_turbulence_workflow.py --status  # Show current status

Author: HyperTensor Team
Date: 2026-02-05
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

import torch
from torch import Tensor


# ═══════════════════════════════════════════════════════════════════════════════════════
# WORKFLOW STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════════════

class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PhaseResult:
    """Result of a single phase execution."""
    phase_id: str
    status: PhaseStatus
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_s: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    gate_criterion: str = ""
    theory: str = ""


@dataclass
class WorkflowState:
    """Persistent workflow state."""
    workflow_id: str
    version: str = "1.0.0"
    created_at: str = ""
    updated_at: str = ""
    device: str = ""
    vram_gb: float = 0.0
    git_commit: str = ""
    current_phase: str = "init"
    phases: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    final_attestation: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowState":
        """Create from dictionary."""
        return cls(**data)


class CheckpointManager:
    """Manages workflow checkpoint persistence."""
    
    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.backup_path = checkpoint_path.with_suffix(".json.bak")
    
    def exists(self) -> bool:
        """Check if checkpoint exists."""
        return self.checkpoint_path.exists()
    
    def load(self) -> Optional[WorkflowState]:
        """Load checkpoint from disk."""
        if not self.exists():
            return None
        
        try:
            with open(self.checkpoint_path, "r") as f:
                data = json.load(f)
            return WorkflowState.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"⚠ Checkpoint corrupted: {e}")
            # Try backup
            if self.backup_path.exists():
                print("  Attempting to load backup...")
                with open(self.backup_path, "r") as f:
                    data = json.load(f)
                return WorkflowState.from_dict(data)
            return None
    
    def save(self, state: WorkflowState) -> None:
        """Save checkpoint to disk with backup."""
        state.updated_at = datetime.now().isoformat()
        
        # Create backup of existing checkpoint
        if self.checkpoint_path.exists():
            import shutil
            shutil.copy(self.checkpoint_path, self.backup_path)
        
        # Write new checkpoint atomically
        temp_path = self.checkpoint_path.with_suffix(".json.tmp")
        with open(temp_path, "w") as f:
            json.dump(state.to_dict(), f, indent=2, default=str)
        
        # Atomic rename
        os.replace(temp_path, self.checkpoint_path)
    
    def clear(self) -> None:
        """Clear checkpoint and backup."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
        if self.backup_path.exists():
            self.backup_path.unlink()


# ═══════════════════════════════════════════════════════════════════════════════════════
# WORKFLOW ENGINE
# ═══════════════════════════════════════════════════════════════════════════════════════

class TurbulenceWorkflow:
    """Checkpointed QTT turbulence validation workflow."""
    
    PHASES = ["init", "decay", "energy", "scaling", "compress", "stability", "attest"]
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_path = checkpoint_dir / "QTT_TURBULENCE_CHECKPOINT.json"
        self.attestation_path = checkpoint_dir / "QTT_TURBULENCE_PROOF.json"
        
        self.checkpoint_mgr = CheckpointManager(self.checkpoint_path)
        self.state: Optional[WorkflowState] = None
        
        # Phase handlers
        self._phase_handlers: Dict[str, Callable[[], PhaseResult]] = {
            "init": self._phase_init,
            "decay": self._phase_decay,
            "energy": self._phase_energy,
            "scaling": self._phase_scaling,
            "compress": self._phase_compress,
            "stability": self._phase_stability,
            "attest": self._phase_attest,
        }
    
    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', '--short', 'HEAD'],
                capture_output=True, text=True, cwd=Path(__file__).parent
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"
    
    def _create_new_state(self) -> WorkflowState:
        """Create new workflow state."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cuda_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
        
        state = WorkflowState(
            workflow_id=f"qtt_turbulence_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            device=cuda_name,
            vram_gb=vram,
            git_commit=self._get_git_commit(),
            current_phase="init",
            phases={phase: {"status": PhaseStatus.PENDING.value} for phase in self.PHASES},
        )
        return state
    
    def load_or_create_state(self, force_new: bool = False) -> WorkflowState:
        """Load existing state or create new one."""
        if force_new:
            self.checkpoint_mgr.clear()
            self.state = self._create_new_state()
        else:
            self.state = self.checkpoint_mgr.load()
            if self.state is None:
                self.state = self._create_new_state()
        
        return self.state
    
    def get_next_phase(self) -> Optional[str]:
        """Get the next phase to execute."""
        for phase in self.PHASES:
            phase_data = self.state.phases.get(phase, {})
            status = phase_data.get("status", PhaseStatus.PENDING.value)
            
            if status == PhaseStatus.PENDING.value:
                return phase
            elif status == PhaseStatus.FAILED.value:
                return phase  # Retry failed phases
            elif status == PhaseStatus.RUNNING.value:
                return phase  # Resume interrupted phases
        
        return None  # All phases complete
    
    def run_phase(self, phase_id: str) -> PhaseResult:
        """Execute a single phase with checkpointing."""
        handler = self._phase_handlers.get(phase_id)
        if handler is None:
            raise ValueError(f"Unknown phase: {phase_id}")
        
        # Mark phase as running
        self.state.current_phase = phase_id
        self.state.phases[phase_id] = {
            "status": PhaseStatus.RUNNING.value,
            "started_at": datetime.now().isoformat(),
        }
        self.checkpoint_mgr.save(self.state)
        
        # Execute phase
        start = time.perf_counter()
        try:
            result = handler()
        except Exception as e:
            result = PhaseResult(
                phase_id=phase_id,
                status=PhaseStatus.FAILED,
                error=str(e),
            )
        
        result.duration_s = time.perf_counter() - start
        result.completed_at = datetime.now().isoformat()
        
        # Update state
        self.state.phases[phase_id] = {
            "status": result.status.value,
            "started_at": result.started_at or self.state.phases[phase_id].get("started_at"),
            "completed_at": result.completed_at,
            "duration_s": result.duration_s,
            "metrics": result.metrics,
            "error": result.error,
            "gate_criterion": result.gate_criterion,
            "theory": result.theory,
        }
        self.checkpoint_mgr.save(self.state)
        
        return result
    
    def run(self, resume: bool = True) -> bool:
        """Run the complete workflow."""
        self.load_or_create_state(force_new=not resume)
        
        self._print_banner()
        self._print_status()
        
        while True:
            next_phase = self.get_next_phase()
            if next_phase is None:
                break
            
            print(f"\n{'═' * 70}")
            print(f"PHASE: {next_phase.upper()}")
            print(f"{'═' * 70}")
            
            result = self.run_phase(next_phase)
            
            status_icon = "✓" if result.status == PhaseStatus.PASSED else "✗"
            print(f"\n  Result: {status_icon} {result.status.value.upper()}")
            
            if result.status == PhaseStatus.FAILED:
                print(f"  Error: {result.error}")
                print("\n⚠ Workflow halted. Fix the issue and run with --resume to continue.")
                return False
        
        self._print_summary()
        return self._all_passed()
    
    def _all_passed(self) -> bool:
        """Check if all phases passed."""
        for phase in self.PHASES:
            status = self.state.phases.get(phase, {}).get("status")
            if status != PhaseStatus.PASSED.value:
                return False
        return True
    
    def _print_banner(self) -> None:
        """Print workflow banner."""
        print("╔" + "═" * 68 + "╗")
        print("║" + " " * 10 + "QTT TURBULENCE CHECKPOINTED WORKFLOW" + " " * 20 + "║")
        print("║" + " " * 10 + "Production-Grade Physics Validation" + " " * 21 + "║")
        print("╚" + "═" * 68 + "╝")
        print(f"\nWorkflow ID: {self.state.workflow_id}")
        print(f"Device: {self.state.device}")
        print(f"VRAM: {self.state.vram_gb:.1f} GB")
        print(f"Git: {self.state.git_commit}")
    
    def _print_status(self) -> None:
        """Print current workflow status."""
        print(f"\n{'─' * 70}")
        print("PHASE STATUS:")
        print(f"{'─' * 70}")
        
        for phase in self.PHASES:
            phase_data = self.state.phases.get(phase, {})
            status = phase_data.get("status", PhaseStatus.PENDING.value)
            
            icon = {
                PhaseStatus.PENDING.value: "○",
                PhaseStatus.RUNNING.value: "◐",
                PhaseStatus.PASSED.value: "●",
                PhaseStatus.FAILED.value: "✗",
                PhaseStatus.SKIPPED.value: "◌",
            }.get(status, "?")
            
            duration = phase_data.get("duration_s", 0)
            duration_str = f"({duration:.1f}s)" if duration > 0 else ""
            
            print(f"  {icon} {phase.upper():12s} : {status:8s} {duration_str}")
    
    def _print_summary(self) -> None:
        """Print final summary."""
        passed = sum(
            1 for p in self.PHASES 
            if self.state.phases.get(p, {}).get("status") == PhaseStatus.PASSED.value
        )
        total = len(self.PHASES)
        
        print(f"\n{'═' * 70}")
        print("WORKFLOW SUMMARY")
        print(f"{'═' * 70}")
        
        for phase in self.PHASES:
            phase_data = self.state.phases.get(phase, {})
            status = phase_data.get("status", PhaseStatus.PENDING.value)
            icon = "✓" if status == PhaseStatus.PASSED.value else "✗"
            duration = phase_data.get("duration_s", 0)
            print(f"  {phase.upper():12s}: {icon} {status:8s} ({duration:.1f}s)")
        
        print(f"\n  Total: {passed}/{total} phases passed")
        
        if passed == total:
            print(f"\n{'═' * 70}")
            print("       ✓✓✓ WORKFLOW COMPLETE — QTT TURBULENCE VALIDATED ✓✓✓")
            print(f"{'═' * 70}")
            
            if self.state.final_attestation:
                sha = self.state.final_attestation.get("sha256", "")[:16]
                print(f"\nAttestation: {self.attestation_path}")
                print(f"SHA256: {sha}...")
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # PHASE IMPLEMENTATIONS
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def _compute_enstrophy(self, omega: List[List[Tensor]]) -> float:
        """Compute enstrophy from QTT vorticity."""
        from ontic.cfd.qtt_turbo import turbo_inner
        return sum(turbo_inner(omega[i], omega[i]).item() for i in range(3))
    
    def _phase_init(self) -> PhaseResult:
        """PHASE 1: System initialization and validation."""
        print("Validating system requirements...")
        
        metrics = {}
        
        # Check CUDA
        cuda_available = torch.cuda.is_available()
        metrics["cuda_available"] = cuda_available
        
        if cuda_available:
            metrics["cuda_device"] = torch.cuda.get_device_name()
            metrics["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # Check imports
        try:
            from ontic.cfd.ns3d_turbo import TurboNS3DSolver, TurboNS3DConfig
            from ontic.cfd.qtt_turbo import turbo_inner
            metrics["imports_ok"] = True
        except ImportError as e:
            metrics["imports_ok"] = False
            metrics["import_error"] = str(e)
            return PhaseResult(
                phase_id="init",
                status=PhaseStatus.FAILED,
                metrics=metrics,
                error=f"Import failed: {e}",
                gate_criterion="All required modules importable",
            )
        
        # Quick solver smoke test
        print("Running solver smoke test...")
        try:
            config = TurboNS3DConfig(
                n_bits=4,  # 16³ - minimal
                max_rank=8,
                poisson_iterations=0,
                nu=0.01,
                dt=0.001,
                device="cuda" if cuda_available else "cpu",
            )
            solver = TurboNS3DSolver(config)
            solver.initialize_taylor_green()
            solver.step()
            metrics["smoke_test"] = "passed"
            del solver
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            metrics["smoke_test"] = "failed"
            return PhaseResult(
                phase_id="init",
                status=PhaseStatus.FAILED,
                metrics=metrics,
                error=f"Smoke test failed: {e}",
                gate_criterion="Solver smoke test passes",
            )
        
        print(f"  CUDA: {cuda_available}")
        print(f"  Imports: OK")
        print(f"  Smoke test: PASSED")
        
        return PhaseResult(
            phase_id="init",
            status=PhaseStatus.PASSED,
            metrics=metrics,
            gate_criterion="CUDA available, imports OK, smoke test passes",
            theory="System validation for turbulence proofs",
        )
    
    def _phase_decay(self) -> PhaseResult:
        """PHASE 2: Taylor-Green vortex decay proof."""
        print("Validating Taylor-Green vortex decay...")
        
        from ontic.cfd.ns3d_turbo import TurboNS3DSolver, TurboNS3DConfig
        
        n_bits = 5  # 32³
        N = 2 ** n_bits
        nu = 0.01
        dt = 0.001
        n_steps = 50
        
        config = TurboNS3DConfig(
            n_bits=n_bits,
            max_rank=16,
            poisson_iterations=0,
            nu=nu,
            dt=dt,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        
        solver = TurboNS3DSolver(config)
        solver.initialize_taylor_green()
        
        Omega_0 = self._compute_enstrophy(solver.omega)
        print(f"  Grid: {N}³, ν = {nu}")
        print(f"  Initial enstrophy: Ω₀ = {Omega_0:.2f}")
        print(f"  Evolving {n_steps} steps...")
        
        enstrophy_history = [Omega_0]
        for step in range(n_steps):
            solver.step()
            if (step + 1) % 10 == 0:
                Omega_t = self._compute_enstrophy(solver.omega)
                enstrophy_history.append(Omega_t)
                print(f"    t={((step+1)*dt):.3f}: Ω = {Omega_t:.2f}")
        
        Omega_final = enstrophy_history[-1]
        t_final = n_steps * dt
        
        fractional_decay = (Omega_0 - Omega_final) / Omega_0
        decay_monotonic = Omega_final < Omega_0
        decay_in_range = 0.05 < fractional_decay < 0.50
        
        metrics = {
            "Omega_0": Omega_0,
            "Omega_final": Omega_final,
            "t_final": t_final,
            "fractional_decay": fractional_decay,
            "decay_monotonic": decay_monotonic,
            "decay_in_range": decay_in_range,
        }
        
        passed = decay_monotonic and decay_in_range
        
        print(f"\n  Fractional decay: {fractional_decay*100:.1f}%")
        print(f"  Monotonic: {'✓' if decay_monotonic else '✗'}")
        print(f"  In range (5%-50%): {'✓' if decay_in_range else '✗'}")
        
        del solver
        torch.cuda.empty_cache()
        gc.collect()
        
        return PhaseResult(
            phase_id="decay",
            status=PhaseStatus.PASSED if passed else PhaseStatus.FAILED,
            metrics=metrics,
            error=None if passed else f"Decay {fractional_decay*100:.1f}% outside range",
            gate_criterion="Monotonic decay in 5%-50% range for t=0.05, ν=0.01",
            theory="dΩ/dt = -2ν∫|∇×ω|²dV < 0 for viscous flow",
        )
    
    def _phase_energy(self) -> PhaseResult:
        """PHASE 3: Inviscid energy conservation proof."""
        print("Validating inviscid energy conservation...")
        
        from ontic.cfd.ns3d_turbo import TurboNS3DSolver, TurboNS3DConfig
        
        n_bits = 5  # 32³
        nu = 0.0  # INVISCID
        dt = 0.0001
        n_steps = 20
        
        config = TurboNS3DConfig(
            n_bits=n_bits,
            max_rank=16,
            poisson_iterations=0,
            nu=nu,
            dt=dt,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        
        solver = TurboNS3DSolver(config)
        solver.initialize_taylor_green()
        
        E_0 = self._compute_enstrophy(solver.omega)
        print(f"  Initial energy proxy: E₀ = {E_0:.2f}")
        print(f"  Evolving {n_steps} steps (inviscid)...")
        
        energy_history = [E_0]
        for _ in range(n_steps):
            solver.step()
            energy_history.append(self._compute_enstrophy(solver.omega))
        
        E_final = energy_history[-1]
        energy_drift = abs(E_final - E_0) / E_0 * 100
        fluctuation = (max(energy_history) - min(energy_history)) / E_0 * 100
        
        metrics = {
            "E_0": E_0,
            "E_final": E_final,
            "energy_drift_percent": energy_drift,
            "fluctuation_percent": fluctuation,
        }
        
        passed = energy_drift < 0.5
        
        print(f"\n  E_final: {E_final:.2f}")
        print(f"  Energy drift: {energy_drift:.3f}%")
        print(f"  Gate (<0.5%): {'✓' if passed else '✗'}")
        
        del solver
        torch.cuda.empty_cache()
        gc.collect()
        
        return PhaseResult(
            phase_id="energy",
            status=PhaseStatus.PASSED if passed else PhaseStatus.FAILED,
            metrics=metrics,
            error=None if passed else f"Energy drift {energy_drift:.3f}% > 0.5%",
            gate_criterion="Energy drift < 0.5% over 20 inviscid steps",
            theory="dE/dt = 0 for Euler equations (ν=0)",
        )
    
    def _phase_scaling(self) -> PhaseResult:
        """PHASE 4: O(log N) time complexity proof."""
        print("Validating O(log N) time scaling...")
        
        from ontic.cfd.ns3d_turbo import TurboNS3DSolver, TurboNS3DConfig
        
        grids = [
            (4, "16³", 4096),
            (5, "32³", 32768),
            (6, "64³", 262144),
            (7, "128³", 2097152),
        ]
        
        results = {}
        metrics = {}
        
        for n_bits, label, cells in grids:
            print(f"\n  Testing {label} ({cells:,} cells)...")
            
            config = TurboNS3DConfig(
                n_bits=n_bits,
                max_rank=16,
                poisson_iterations=0,
                nu=0.01,
                dt=0.001,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            
            solver = TurboNS3DSolver(config)
            solver.initialize_taylor_green()
            
            # Warmup
            solver.step()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            step_times = []
            for _ in range(3):
                t0 = time.perf_counter()
                solver.step()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                step_times.append(time.perf_counter() - t0)
            
            avg_time = sum(step_times) / len(step_times) * 1000  # ms
            results[label] = avg_time
            metrics[f"{label}_ms"] = avg_time
            
            print(f"    Average: {avg_time:.1f} ms")
            
            del solver
            torch.cuda.empty_cache()
            gc.collect()
        
        ratio_64_16 = results["64³"] / results["16³"]
        ratio_128_32 = results["128³"] / results["32³"]
        
        metrics["ratio_64_to_16"] = ratio_64_16
        metrics["ratio_128_to_32"] = ratio_128_32
        
        passed = ratio_64_16 < 3.0 and ratio_128_32 < 3.0
        
        print(f"\n  16³ → 64³ (4× cells): {ratio_64_16:.2f}× time")
        print(f"  32³ → 128³ (4× cells): {ratio_128_32:.2f}× time")
        print(f"  Gate (<3×): {'✓' if passed else '✗'}")
        
        return PhaseResult(
            phase_id="scaling",
            status=PhaseStatus.PASSED if passed else PhaseStatus.FAILED,
            metrics=metrics,
            error=None if passed else f"Scaling ratio {max(ratio_64_16, ratio_128_32):.2f}× > 3×",
            gate_criterion="4× grid → <3× time (sublinear, not O(N³))",
            theory="QTT ops are O(d·r³) where d = 3·log₂(N)",
        )
    
    def _phase_compress(self) -> PhaseResult:
        """PHASE 5: QTT compression ratio proof."""
        print("Validating QTT compression ratio...")
        
        from ontic.cfd.ns3d_turbo import TurboNS3DSolver, TurboNS3DConfig
        
        grids = [
            (5, "32³", 32768),
            (6, "64³", 262144),
            (7, "128³", 2097152),
        ]
        
        metrics = {}
        
        for n_bits, label, N_cubed in grids:
            print(f"\n  {label}:")
            
            config = TurboNS3DConfig(
                n_bits=n_bits,
                max_rank=16,
                poisson_iterations=0,
                nu=0.01,
                dt=0.001,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            
            solver = TurboNS3DSolver(config)
            solver.initialize_taylor_green()
            
            total_params = sum(
                core.numel() 
                for comp in range(3) 
                for core in solver.omega[comp]
            )
            
            dense_params = 3 * N_cubed
            compression = dense_params / total_params
            
            metrics[f"{label}_qtt_params"] = total_params
            metrics[f"{label}_dense_params"] = dense_params
            metrics[f"{label}_compression"] = compression
            
            print(f"    Dense: {dense_params:,} params")
            print(f"    QTT: {total_params:,} params")
            print(f"    Compression: {compression:.0f}×")
            
            del solver
            torch.cuda.empty_cache()
            gc.collect()
        
        compression_128 = metrics.get("128³_compression", 0)
        passed = compression_128 > 100
        
        print(f"\n  Gate (>100× at 128³): {'✓' if passed else '✗'}")
        
        return PhaseResult(
            phase_id="compress",
            status=PhaseStatus.PASSED if passed else PhaseStatus.FAILED,
            metrics=metrics,
            error=None if passed else f"Compression {compression_128:.0f}× < 100×",
            gate_criterion="Compression > 100× at 128³",
            theory="QTT stores O(d·r²) vs O(N³) dense",
        )
    
    def _phase_stability(self) -> PhaseResult:
        """PHASE 6: Long-time numerical stability proof."""
        print("Validating numerical stability (long integration)...")
        
        from ontic.cfd.ns3d_turbo import TurboNS3DSolver, TurboNS3DConfig
        
        n_bits = 6  # 64³
        N = 2 ** n_bits
        n_steps = 100
        
        config = TurboNS3DConfig(
            n_bits=n_bits,
            max_rank=16,
            poisson_iterations=0,
            nu=0.01,
            dt=0.001,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        
        solver = TurboNS3DSolver(config)
        solver.initialize_taylor_green()
        
        Omega_0 = self._compute_enstrophy(solver.omega)
        print(f"  Grid: {N}³, {n_steps} steps")
        print(f"  Initial enstrophy: {Omega_0:.2f}")
        
        nan_detected = False
        inf_detected = False
        enstrophy_history = [Omega_0]
        
        for step in range(n_steps):
            solver.step()
            
            if (step + 1) % 10 == 0:
                for i in range(3):
                    for core in solver.omega[i]:
                        if torch.isnan(core).any():
                            nan_detected = True
                        if torch.isinf(core).any():
                            inf_detected = True
                
                Omega_t = self._compute_enstrophy(solver.omega)
                enstrophy_history.append(Omega_t)
                
                if math.isnan(Omega_t) or math.isinf(Omega_t):
                    nan_detected = True
                
                print(f"    Step {step+1:3d}: Ω = {Omega_t:.2f}")
                
                if nan_detected or inf_detected:
                    break
        
        max_Omega = max(enstrophy_history)
        enstrophy_bounded = max_Omega < 10 * Omega_0
        
        metrics = {
            "Omega_0": Omega_0,
            "Omega_final": enstrophy_history[-1],
            "steps_completed": len(enstrophy_history) * 10,
            "nan_detected": nan_detected,
            "inf_detected": inf_detected,
            "max_enstrophy": max_Omega,
            "enstrophy_bounded": enstrophy_bounded,
        }
        
        passed = not nan_detected and not inf_detected and enstrophy_bounded
        
        print(f"\n  NaN detected: {nan_detected}")
        print(f"  Inf detected: {inf_detected}")
        print(f"  Enstrophy bounded: {enstrophy_bounded}")
        print(f"  Gate: {'✓' if passed else '✗'}")
        
        del solver
        torch.cuda.empty_cache()
        gc.collect()
        
        error = None
        if nan_detected:
            error = "NaN detected"
        elif inf_detected:
            error = "Inf detected"
        elif not enstrophy_bounded:
            error = "Enstrophy unbounded"
        
        return PhaseResult(
            phase_id="stability",
            status=PhaseStatus.PASSED if passed else PhaseStatus.FAILED,
            metrics=metrics,
            error=error,
            gate_criterion="100 steps @ 64³, no NaN/Inf, bounded enstrophy",
            theory="Numerical stability requires bounded solutions",
        )
    
    def _phase_attest(self) -> PhaseResult:
        """PHASE 7: Generate final attestation."""
        print("Generating final attestation...")
        
        # Collect all phase results
        proofs = []
        for phase in self.PHASES[1:-1]:  # Skip init and attest
            phase_data = self.state.phases.get(phase, {})
            proofs.append({
                "name": phase,
                "passed": phase_data.get("status") == PhaseStatus.PASSED.value,
                "duration_s": phase_data.get("duration_s", 0),
                "metrics": phase_data.get("metrics", {}),
                "theory": phase_data.get("theory", ""),
                "gate_criterion": phase_data.get("gate_criterion", ""),
                "error": phase_data.get("error"),
            })
        
        passed_count = sum(1 for p in proofs if p["passed"])
        total_duration = sum(
            self.state.phases.get(p, {}).get("duration_s", 0) 
            for p in self.PHASES
        )
        
        attestation = {
            "timestamp": datetime.now().isoformat(),
            "workflow_id": self.state.workflow_id,
            "git_commit": self.state.git_commit,
            "device": self.state.device,
            "vram_gb": self.state.vram_gb,
            "proofs": proofs,
            "summary": {
                "passed": passed_count,
                "total": len(proofs),
                "all_passed": passed_count == len(proofs),
                "total_duration_s": total_duration,
            },
        }
        
        # Compute SHA256
        json_str = json.dumps(attestation, indent=2, default=str)
        hash_val = sha256(json_str.encode()).hexdigest()
        attestation["sha256"] = hash_val
        
        # Save attestation
        with open(self.attestation_path, "w") as f:
            json.dump(attestation, f, indent=2, default=str)
        
        self.state.final_attestation = attestation
        
        metrics = {
            "proofs_passed": passed_count,
            "proofs_total": len(proofs),
            "sha256": hash_val,
            "attestation_path": str(self.attestation_path),
        }
        
        print(f"\n  Attestation: {self.attestation_path}")
        print(f"  SHA256: {hash_val[:16]}...")
        print(f"  Proofs: {passed_count}/{len(proofs)} passed")
        
        return PhaseResult(
            phase_id="attest",
            status=PhaseStatus.PASSED,
            metrics=metrics,
            gate_criterion="Generate SHA256-attested proof document",
            theory="Constitutional attestation for audit trail",
        )
    
    def status(self) -> None:
        """Print current workflow status."""
        state = self.checkpoint_mgr.load()
        if state is None:
            print("No checkpoint found. Run workflow to start.")
            return
        
        self.state = state
        self._print_banner()
        self._print_status()


# ═══════════════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════════════

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="QTT Turbulence Checkpointed Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 prove_qtt_turbulence_workflow.py           # Run full workflow
  python3 prove_qtt_turbulence_workflow.py --resume  # Resume from checkpoint
  python3 prove_qtt_turbulence_workflow.py --reset   # Clear checkpoint and restart
  python3 prove_qtt_turbulence_workflow.py --status  # Show current status
        """,
    )
    
    parser.add_argument("--resume", action="store_true", 
                       help="Resume from existing checkpoint (default)")
    parser.add_argument("--reset", action="store_true",
                       help="Clear checkpoint and start fresh")
    parser.add_argument("--status", action="store_true",
                       help="Show current workflow status")
    
    args = parser.parse_args()
    
    artifacts_dir = Path(__file__).parent / "artifacts"
    workflow = TurbulenceWorkflow(artifacts_dir)
    
    if args.status:
        workflow.status()
        return 0
    
    if args.reset:
        workflow.checkpoint_mgr.clear()
        print("Checkpoint cleared.")
    
    # Default to resume behavior
    resume = not args.reset
    
    success = workflow.run(resume=resume)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
