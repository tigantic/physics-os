"""
Manufacturing physics package: welding, solidification, AM melt pool, machining.

Domains: XX.9.
"""

from .manufacturing import (
    GoldakWeldingSource,
    WeldingHeatTransfer1D,
    ScheilSolidification,
    MarangoniMeltPool,
    MerchantMachining,
)

__all__ = [
    "GoldakWeldingSource",
    "WeldingHeatTransfer1D",
    "ScheilSolidification",
    "MarangoniMeltPool",
    "MerchantMachining",
]
