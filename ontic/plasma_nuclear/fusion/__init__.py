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

Phase 22: Resonant Catalysis (Hummingbird)
- Phonon-assisted bond activation
- Frequency-matched catalyst design
- Ambient-temperature nitrogen fixation
- "Opera Singer" mechanism for N≡N rupture

BAA Alignment: HR001126S0007
"Material Solutions for Achieving Room-Temperature D-D Fusion Reactions"

F = q(E + v × B) - The Lorentz Force that confines stars.
ω_catalyst = ω_bond - The Resonance that breaks them.
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

# Resonant Catalysis (Phase 22 - Hummingbird)
from .resonant_catalysis import (
    ResonantCatalysisSolver,
    ResonantActivationResult,
    ResonanceMatch,
    BondState,
    CatalystParams,
    CatalystType,
    TargetBond,
    N2_TRIPLE_BOND,
    CO_TRIPLE_BOND,
    O2_DOUBLE_BOND,
    create_hummingbird_catalyst,
    create_fe4s4_cube,
    create_femoco,
    create_fe3_graphene,
    screen_catalysts,
    run_hummingbird_demo,
    generate_hummingbird_attestation,
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
    # Resonant Catalysis (Phase 22 - Hummingbird)
    "ResonantCatalysisSolver",
    "ResonantActivationResult",
    "ResonanceMatch",
    "BondState",
    "CatalystParams",
    "CatalystType",
    "TargetBond",
    "N2_TRIPLE_BOND",
    "CO_TRIPLE_BOND",
    "O2_DOUBLE_BOND",
    "create_hummingbird_catalyst",
    "create_fe4s4_cube",
    "create_femoco",
    "create_fe3_graphene",
    "screen_catalysts",
    "run_hummingbird_demo",
    "generate_hummingbird_attestation",
]
