"""
Genesis Fusion — Cross-Primitive Composition Module

This module provides utilities for combining all 7 TENSOR GENESIS primitives
into unified pipelines that achieve capabilities impossible with any single
primitive alone.

The power of Genesis is not just in individual primitives, but in their
COMPOSITION. When combined:

    OT + PH: Transport-aware topology
    SGW + RKHS: Spectral kernel methods  
    RMT + GA: Uncertainty-quantified transformations
    TG + OT: Discrete optimal transport
    
    ALL 7: Complete geometric-topological-statistical framework
"""

from .genesis_fusion_demo import (
    GenesisFusionPipeline,
    GenesisFusionResult,
    demonstrate_scaling,
    demonstrate_composition,
)

__all__ = [
    "GenesisFusionPipeline",
    "GenesisFusionResult", 
    "demonstrate_scaling",
    "demonstrate_composition",
]
