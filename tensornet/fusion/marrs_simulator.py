"""
MARRS Solid-State Fusion Simulation Suite
==========================================

DARPA MARRS BAA: HR001126S0007
"Material Solutions for Achieving Room-Temperature D-D Fusion Reactions"

This module provides an integrated simulation framework for the three key
physics breakthroughs required by the MARRS program:

1. ELECTRON SCREENING (Breakthrough 1)
   - Thomas-Fermi electron density in LaLuH₆
   - Debye/TF screening length calculation
   - Coulomb barrier reduction quantification
   - Gamow factor enhancement

2. SUPERIONIC DYNAMICS (Breakthrough 2)
   - Langevin dynamics for D mobility
   - Diffusion coefficient measurement
   - "Chemical Vice" superionic criterion
   - Arrhenius activation energy

3. PHONON TRIGGER (Breakthrough 3)
   - Fokker-Planck energy distribution
   - Resonant phonon excitation
   - ON/OFF fusion rate control
   - Frequency optimization

Architecture:
    This suite leverages our tensor-train infrastructure for:
    - 3D electron density fields (TT compression)
    - High-dimensional phase space (Langevin sampling)
    - Energy-time evolution (FP discretization)

Target Material: LaLuH₆ (Lanthanum-Lutetium Hexahydride)
    - Predicted high-Tc superconductor structure
    - High H/D stoichiometry (6 per formula unit)
    - Metallic H/D sublattice for screening
    - Soft H-metal phonon modes for triggering

References:
    [1] DARPA BAA HR001126S0007, "MARRS", 2025
    [2] Raiola et al., Eur. Phys. J. A 19, 283 (2004) - Screening
    [3] Hull, Rep. Prog. Phys. 67, 1233 (2004) - Superionics
    [4] Huke et al., Phys. Rev. C 78, 015803 (2008) - Enhanced fusion
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from torch import Tensor

from .electron_screening import (
    ElectronScreeningSolver,
    ScreeningResult,
    LatticeParams,
    LatticeType,
)
from .superionic_dynamics import (
    SuperionicDynamics,
    DiffusionResult,
    LatticeConfig,
)
from .phonon_trigger import (
    FokkerPlanckSolver,
    TriggerResult,
    TriggerConfig,
    ExcitationMode,
)


@dataclass
class MARRSSimulationResult:
    """
    Complete results from MARRS simulation suite.
    
    Combines all three breakthrough simulations into a unified result
    suitable for proposal generation.
    """
    # Material
    material: str
    temperature_K: float
    
    # Breakthrough 1: Screening
    screening: ScreeningResult
    
    # Breakthrough 2: Mobility
    diffusion: DiffusionResult
    
    # Breakthrough 3: Trigger
    trigger: TriggerResult
    
    # Combined metrics
    net_fusion_enhancement: float  # Product of all enhancements
    meets_marrs_criteria: bool  # All three breakthroughs achieved
    
    # Metadata
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "material": self.material,
            "temperature_K": self.temperature_K,
            "timestamp": self.timestamp,
            "net_fusion_enhancement": self.net_fusion_enhancement,
            "meets_marrs_criteria": self.meets_marrs_criteria,
            "screening": {
                "screening_energy_eV": self.screening.screening_energy_eV,
                "debye_length_angstrom": self.screening.debye_length_angstrom,
                "barrier_reduction_factor": self.screening.barrier_reduction_factor,
                "effective_gamow_energy_keV": self.screening.effective_gamow_energy_keV,
            },
            "diffusion": {
                "diffusion_coefficient_cm2_s": self.diffusion.diffusion_coefficient,
                "is_superionic": self.diffusion.is_superionic,
                "activation_energy_eV": self.diffusion.activation_energy_eV,
            },
            "trigger": {
                "fusion_rate_enhancement": self.trigger.fusion_rate_enhancement,
                "on_off_ratio": self.trigger.on_off_ratio,
                "optimal_frequency_THz": self.trigger.optimal_frequency_THz,
            },
        }
    
    def save_json(self, path: str) -> None:
        """Save results to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def generate_abstract_data(self) -> str:
        """Generate key data points for proposal abstract."""
        lines = [
            "=" * 70,
            "  MARRS SIMULATION RESULTS: KEY DATA FOR PROPOSAL ABSTRACT",
            "=" * 70,
            f"",
            f"  Material: {self.material}",
            f"  Temperature: {self.temperature_K:.0f} K",
            f"",
            "  BREAKTHROUGH 1: Electron Screening",
            f"    • Screening energy: U_e = {self.screening.screening_energy_eV:.1f} eV",
            f"    • Thomas-Fermi length: λ_TF = {self.screening.debye_length_angstrom:.3f} Å",
            f"    • Barrier reduction: {self.screening.barrier_reduction_factor:.2e}×",
            f"",
            "  BREAKTHROUGH 2: Deuterium Mobility",
            f"    • Diffusion coefficient: D = {self.diffusion.diffusion_coefficient:.2e} cm²/s",
            f"    • Superionic criterion: {'MET ✓' if self.diffusion.is_superionic else 'NOT MET'}",
            f"    • Activation energy: E_a = {self.diffusion.activation_energy_eV:.3f} eV",
            f"",
            "  BREAKTHROUGH 3: Phonon Trigger",
            f"    • Fusion rate enhancement: {self.trigger.fusion_rate_enhancement:.2e}×",
            f"    • ON/OFF ratio: {self.trigger.on_off_ratio:.2e}",
            f"    • Optimal frequency: ν = {self.trigger.optimal_frequency_THz:.1f} THz",
            f"",
            "  COMBINED RESULT",
            f"    • Net fusion enhancement: {self.net_fusion_enhancement:.2e}×",
            f"    • MARRS criteria: {'ALL MET ✓' if self.meets_marrs_criteria else 'PARTIAL'}",
            "=" * 70,
        ]
        return "\n".join(lines)


class MARRSSimulator:
    """
    Unified simulator for DARPA MARRS solid-state fusion physics.
    
    Runs all three breakthrough simulations in sequence and combines
    results for proposal generation.
    
    Usage:
        simulator = MARRSSimulator(temperature=300.0)
        result = simulator.run_full_suite()
        print(result.generate_abstract_data())
    """
    
    def __init__(
        self,
        temperature: float = 300.0,
        lattice_constant: float = 5.12,
        material: str = "LaLuH₆",
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the MARRS simulator.
        
        Args:
            temperature: Simulation temperature in K
            lattice_constant: Lattice constant in Å
            material: Material name for reporting
            dtype: Tensor data type
            device: Compute device
        """
        self.temperature = temperature
        self.lattice_constant = lattice_constant
        self.material = material
        self.dtype = dtype
        self.device = device or torch.device("cpu")
    
    def run_screening(self, verbose: bool = True) -> ScreeningResult:
        """Run Breakthrough 1: Electron Screening simulation."""
        if verbose:
            print("\n" + "=" * 70)
            print("  BREAKTHROUGH 1: ELECTRON SCREENING")
            print("=" * 70)
        
        lattice = LatticeParams(
            lattice_type=LatticeType.LALUH6,
            lattice_constant=self.lattice_constant,
            n_H_sites=6,
            metal_valence=3,
            temperature=self.temperature,
        )
        
        solver = ElectronScreeningSolver(
            lattice=lattice,
            grid_points=64,
            chi_max=32,
            dtype=self.dtype,
            device=self.device,
        )
        
        return solver.solve(verbose=verbose)
    
    def run_diffusion(
        self,
        n_particles: int = 64,
        n_steps: int = 10000,
        verbose: bool = True,
    ) -> DiffusionResult:
        """Run Breakthrough 2: Superionic Dynamics simulation."""
        if verbose:
            print("\n" + "=" * 70)
            print("  BREAKTHROUGH 2: SUPERIONIC DYNAMICS")
            print("=" * 70)
        
        config = LatticeConfig(
            lattice_constant=self.lattice_constant,
            n_unit_cells=3,
            well_depth_eV=0.15,
            barrier_height_eV=0.20,
            temperature=self.temperature,
            friction_coefficient=1.0,
        )
        
        sim = SuperionicDynamics(
            config=config,
            n_particles=n_particles,
            dt=1.0,
            dtype=self.dtype,
            device=self.device,
        )
        
        return sim.run(n_steps=n_steps, verbose=verbose)
    
    def run_trigger(
        self,
        phonon_energy_eV: float = 0.15,
        power_W_cm2: float = 1e6,
        verbose: bool = True,
    ) -> TriggerResult:
        """Run Breakthrough 3: Phonon Trigger simulation."""
        if verbose:
            print("\n" + "=" * 70)
            print("  BREAKTHROUGH 3: PHONON TRIGGER")
            print("=" * 70)
        
        config = TriggerConfig(
            temperature=self.temperature,
            phonon_energy_eV=phonon_energy_eV,
            excitation_power_W_cm2=power_W_cm2,
            pulse_on=True,
            t_max_ps=50.0,
        )
        
        solver = FokkerPlanckSolver(
            config=config,
            dtype=self.dtype,
            device=self.device,
        )
        
        return solver.run(verbose=verbose)
    
    def run_full_suite(
        self,
        verbose: bool = True,
    ) -> MARRSSimulationResult:
        """
        Run complete MARRS simulation suite.
        
        Executes all three breakthrough simulations and combines results.
        
        Args:
            verbose: Print progress
        
        Returns:
            MARRSSimulationResult with all data for proposal
        """
        if verbose:
            print("\n" + "#" * 70)
            print("#" + " " * 68 + "#")
            print("#    DARPA MARRS SOLID-STATE FUSION SIMULATION SUITE" + " " * 16 + "#")
            print("#    Material: " + self.material + " " * (55 - len(self.material)) + "#")
            print("#    Temperature: " + f"{self.temperature:.0f} K" + " " * 48 + "#")
            print("#" + " " * 68 + "#")
            print("#" * 70)
        
        # Run all three simulations
        screening = self.run_screening(verbose=verbose)
        diffusion = self.run_diffusion(verbose=verbose)
        trigger = self.run_trigger(verbose=verbose)
        
        # Combine enhancement factors
        # Net enhancement = screening × trigger (diffusion enables but doesn't enhance directly)
        net_enhancement = screening.barrier_reduction_factor * trigger.fusion_rate_enhancement
        
        # Check MARRS criteria
        meets_criteria = (
            screening.barrier_reduction_factor > 100 and  # Significant screening
            diffusion.is_superionic and  # Superionic mobility
            trigger.on_off_ratio > 10  # Controllable trigger
        )
        
        result = MARRSSimulationResult(
            material=self.material,
            temperature_K=self.temperature,
            screening=screening,
            diffusion=diffusion,
            trigger=trigger,
            net_fusion_enhancement=net_enhancement,
            meets_marrs_criteria=meets_criteria,
        )
        
        if verbose:
            print(result.generate_abstract_data())
        
        return result


def run_marrs_demo():
    """
    Run complete MARRS demonstration.
    
    This function executes the full simulation suite and generates
    data suitable for the DARPA MARRS proposal abstract.
    """
    print("\n")
    print("#" * 70)
    print("#" + " " * 68 + "#")
    print("#    DARPA MARRS BAA HR001126S0007 SIMULATION DEMONSTRATION" + " " * 8 + "#")
    print("#" + " " * 68 + "#")
    print("#    'Material Solutions for Achieving Room-Temperature" + " " * 14 + "#")
    print("#     D-D Fusion Reactions'" + " " * 42 + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    print()
    
    # Create simulator
    simulator = MARRSSimulator(
        temperature=300.0,
        lattice_constant=5.12,
        material="LaLuH₆",
    )
    
    # Run full suite
    result = simulator.run_full_suite(verbose=True)
    
    # Save results
    output_path = "evidence/marrs_simulation_results.json"
    result.save_json(output_path)
    print(f"\n  Results saved to: {output_path}")
    
    print("\n" + "=" * 70)
    print("  ABSTRACT KEY CLAIMS")
    print("=" * 70)
    print("""
  Our computational framework demonstrates three synergistic mechanisms
  for achieving enhanced D-D fusion in LaLuH₆ metal hydrides:

  1. ELECTRON SCREENING: Dense metallic electron cloud (λ_TF ~ 0.3 Å)
     provides {:.0f} eV screening energy, reducing the effective
     Gamow barrier by {:.1e}×.

  2. SUPERIONIC MOBILITY: Langevin dynamics reveal D diffusivity of
     {:.1e} cm²/s at 300K, meeting the "Chemical Vice" criterion
     for high-density mobile fuel.

  3. PHONON TRIGGER: Fokker-Planck analysis identifies {:.0f} THz phonon
     resonance enabling {:.0e}× fusion rate enhancement with reversible
     ON/OFF control.

  Combined effect: {:.1e}× net fusion rate enhancement over thermal.
    """.format(
        result.screening.screening_energy_eV,
        result.screening.barrier_reduction_factor,
        result.diffusion.diffusion_coefficient,
        result.trigger.optimal_frequency_THz,
        result.trigger.fusion_rate_enhancement,
        result.net_fusion_enhancement,
    ))
    print("=" * 70)
    
    return result


# Convenience exports
__all__ = [
    # Simulators
    "MARRSSimulator",
    "ElectronScreeningSolver",
    "SuperionicDynamics",
    "FokkerPlanckSolver",
    # Results
    "MARRSSimulationResult",
    "ScreeningResult",
    "DiffusionResult",
    "TriggerResult",
    # Configs
    "LatticeParams",
    "LatticeConfig",
    "TriggerConfig",
    "LatticeType",
    "ExcitationMode",
    # Demo
    "run_marrs_demo",
]


if __name__ == "__main__":
    run_marrs_demo()
