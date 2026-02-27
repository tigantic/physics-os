"""Assumptions extraction module."""

from tensornet.infra.oracle.assumptions.economic_extractor import EconomicExtractor
from tensornet.infra.oracle.assumptions.explicit_extractor import ExplicitExtractor
from tensornet.infra.oracle.assumptions.implicit_extractor import ImplicitExtractor

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
