"""
QTT-Aging: Biological Aging as Tensor Rank Dynamics

TENSOR GENESIS Protocol — Layer 27
Phase 21 of the Civilization Stack

Core Thesis:
    The biological state of a cell is a tensor. Aging is rank growth.
    Reversal is rank reduction. Yamanaka showed four transcription factors
    can reset cellular age — that's a rank-4 intervention collapsing a
    million-dimensional state back to its initial condition.

    This module provides the mathematical framework to understand WHY
    it works, and to engineer new interventions.

Modules:
    cell_state      — CellStateTensor: biological state in QTT format
    dynamics        — AgingOperator: time evolution of biological aging
    epigenetic_clock — Horvath/GrimAge clocks in QTT basis
    interventions   — Yamanaka, partial reprogramming, senolytics, CR
    topology        — Persistent homology of aging trajectories

Mathematical Framework:
    Cell state:     ψ ∈ ℝ^{d₁ × d₂ × ... × d_N}  (TT format)
    Young cell:     max_rank(ψ_young) ≈ 4
    Aged cell:      max_rank(ψ_aged) ≈ 50-200
    Yamanaka:       max_rank(OSKM · ψ_aged) ≈ 4
    Aging operator: ψ(t+Δt) = normalize(ψ(t) + Σ_k ε_k · Δ_k(ψ(t)))
    Reversal:       find P of min rank s.t. ||P · ψ_aged - ψ_young|| < ε

Integration Points:
    - QTT-OT (Layer 20): Optimal transport for rejuvenation paths
    - QTT-RKHS (Layer 24): Gaussian process regression on aging data
    - QTT-PH (Layer 25): Persistent homology of aging trajectories
    - Proteome Compiler: Age-dependent protein folding landscape
    - Epigenomics Module: CpG methylation tensor

Usage:
    >>> from ontic.genesis.aging import young_cell, aged_cell, AgingOperator
    >>> from ontic.genesis.aging import YamanakaOperator, HorvathClock

    # Create cells
    >>> young = young_cell(seed=42)
    >>> print(f"Young: rank={young.max_rank}, age={young.chronological_age}")

    # Age a cell
    >>> operator = AgingOperator(seed=42)
    >>> aged, trajectory = operator.evolve(young, target_age=70.0, dt_years=5.0)
    >>> print(f"Aged: rank={aged.max_rank}")

    # Measure epigenetic age
    >>> clock = HorvathClock()
    >>> bio_age = clock.predict_age_from_cell(aged)

    # Reverse aging
    >>> yamanaka = YamanakaOperator(target_rank=4)
    >>> result = yamanaka.apply(aged)
    >>> print(f"Reversed: rank {result.rank_before} → {result.rank_after}")
    >>> print(f"Years reversed: {result.years_reversed:.1f}")

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

# Cell state tensor
from ontic.genesis.aging.cell_state import (
    # Core data structures
    CellStateTensor,
    ModeSpec,
    AgingSignature,
    # Enums
    CellType,
    AgingHallmark,
    BiologicalMode,
    # Factory functions
    young_cell,
    aged_cell,
    embryonic_stem_cell,
    # Constants
    YAMANAKA_FACTORS,
    THOMSON_FACTORS,
    NUM_PROTEIN_CODING_GENES,
    NUM_PROTEINS,
    NUM_CPG_SITES,
)

# Aging dynamics
from ontic.genesis.aging.dynamics import (
    AgingOperator,
    AgingRateModel,
    AgingTrajectory,
    ModePerturbation,
)

# Epigenetic clocks
from ontic.genesis.aging.epigenetic_clock import (
    HorvathClock,
    GrimAgeClock,
    MethylationState,
    extract_methylation,
    young_methylation,
    aged_methylation,
    HORVATH_TOP_SITES,
)

# Interventions
from ontic.genesis.aging.interventions import (
    InterventionResult,
    YamanakaOperator,
    PartialReprogrammingOperator,
    SenolyticOperator,
    CalorieRestrictionOperator,
    CombinationIntervention,
    find_optimal_intervention,
)

# Topology
from ontic.genesis.aging.topology import (
    AgingTopologyAnalyzer,
    AgingTopology,
    AgingPhase,
    TopologicalBarrier,
    RejuvenationPath,
    compute_rejuvenation_path,
)

__all__ = [
    # Cell state
    "CellStateTensor",
    "ModeSpec",
    "AgingSignature",
    "CellType",
    "AgingHallmark",
    "BiologicalMode",
    "young_cell",
    "aged_cell",
    "embryonic_stem_cell",
    "YAMANAKA_FACTORS",
    "THOMSON_FACTORS",
    "NUM_PROTEIN_CODING_GENES",
    "NUM_PROTEINS",
    "NUM_CPG_SITES",
    # Dynamics
    "AgingOperator",
    "AgingRateModel",
    "AgingTrajectory",
    "ModePerturbation",
    # Clocks
    "HorvathClock",
    "GrimAgeClock",
    "MethylationState",
    "extract_methylation",
    "young_methylation",
    "aged_methylation",
    "HORVATH_TOP_SITES",
    # Interventions
    "InterventionResult",
    "YamanakaOperator",
    "PartialReprogrammingOperator",
    "SenolyticOperator",
    "CalorieRestrictionOperator",
    "CombinationIntervention",
    "find_optimal_intervention",
    # Topology
    "AgingTopologyAnalyzer",
    "AgingTopology",
    "AgingPhase",
    "TopologicalBarrier",
    "RejuvenationPath",
    "compute_rejuvenation_path",
]
