"""
Coupled physics package: MHD-coupled flows, magnetoconvection, EM pumps.

Domains: XVIII.4.
"""

from .coupled_mhd import (
    HartmannFlow,
    CzochralskiMHD,
    EMPump,
    Magnetoconvection,
)

__all__ = [
    "HartmannFlow",
    "CzochralskiMHD",
    "EMPump",
    "Magnetoconvection",
]
