"""
HyperTensor Physics Engine
==========================

Universal physics solver with TT-compressed state evolution.

The Patent: Everyone else runs out of RAM. We compress the universe every millisecond.

Quick Start:
    from hypertensor import tt_round, TTTensor
    from hypertensor.integrators import LangevinDynamics
    from hypertensor.pde import ResistiveMHD, FokkerPlanck
"""

from hypertensor.core.tensor_train import TTTensor, tt_round, tt_to_full
from hypertensor.core.constants import k_B, e, m_p, mu_0, epsilon_0, c, hbar

__version__ = "0.1.0"
__all__ = [
    "TTTensor",
    "tt_round", 
    "tt_to_full",
    "k_B", "e", "m_p", "mu_0", "epsilon_0", "c", "hbar",
]
