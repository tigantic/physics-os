"""Assumptions extraction module."""

from tensornet.oracle.assumptions.economic_extractor import EconomicExtractor
from tensornet.oracle.assumptions.explicit_extractor import ExplicitExtractor
from tensornet.oracle.assumptions.implicit_extractor import ImplicitExtractor

# Aliases for compatibility
ExplicitAssumptionExtractor = ExplicitExtractor
ImplicitAssumptionExtractor = ImplicitExtractor
EconomicAssumptionExtractor = EconomicExtractor

__all__ = [
    "ExplicitExtractor",
    "ImplicitExtractor",
    "EconomicExtractor",
    "ExplicitAssumptionExtractor",
    "ImplicitAssumptionExtractor",
    "EconomicAssumptionExtractor",
]
