"""Assumptions extraction module."""

from ontic.infra.oracle.assumptions.economic_extractor import EconomicExtractor
from ontic.infra.oracle.assumptions.explicit_extractor import ExplicitExtractor
from ontic.infra.oracle.assumptions.implicit_extractor import ImplicitExtractor

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
