"""
TensorNet Fusion Module
========================

Phase 9: Magnetically Confined Plasma (Tokamak)
- Toroidal magnetic geometry (the "donut")
- Boris particle pusher for ion dynamics
- Magnetic bottle confinement verification

Phase 21: DARPA MARRS Solid-State Fusion
- Electron screening in metal hydrides
- Superionic Langevin dynamics for D mobility
- Fokker-Planck phonon trigger mechanism
- QTT-compressed electron density fields

BAA Alignment: HR001126S0007
"Material Solutions for Achieving Room-Temperature D-D Fusion Reactions"

F = q(E + v × B) - The Lorentz Force that confines stars.
"""

from .tokamak import ConfinementReport, PlasmaState, TokamakReactor

# MARRS Solid-State Fusion Components
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
from .marrs_simulator import (
    MARRSSimulator,
    MARRSSimulationResult,
    run_marrs_demo,
)

# QTT-Compressed MARRS Solvers (Phase 21 + TensorTrain)
from .qtt_screening import (
    QTTElectronScreeningSolver,
    QTTScreeningResult,
    compare_qtt_vs_dense,
    demo_qtt_screening,
)
from .qtt_superionic import (
    QTTSuperionicDynamics,
    QTTDiffusionResult,
    demo_qtt_superionic,
)

__all__ = [
    # Tokamak (Phase 9)
    "TokamakReactor",
    "PlasmaState",
    "ConfinementReport",
    # MARRS Electron Screening (Phase 21)
    "ElectronScreeningSolver",
    "ScreeningResult",
    "LatticeParams",
    "LatticeType",
    # MARRS Superionic Dynamics (Phase 21)
    "SuperionicDynamics",
    "DiffusionResult",
    "LatticeConfig",
    # MARRS Phonon Trigger (Phase 21)
    "FokkerPlanckSolver",
    "TriggerResult",
    "TriggerConfig",
    "ExcitationMode",
    # MARRS Unified Simulator (Phase 21)
    "MARRSSimulator",
    "MARRSSimulationResult",
    "run_marrs_demo",
    # QTT-Compressed MARRS (Phase 21 + TensorTrain)
    "QTTElectronScreeningSolver",
    "QTTScreeningResult",
    "compare_qtt_vs_dense",
    "demo_qtt_screening",
    "QTTSuperionicDynamics",
    "QTTDiffusionResult",
    "demo_qtt_superionic",
]
