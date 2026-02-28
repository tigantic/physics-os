"""
Discovery Ingesters Package

Domain-specific data ingesters for the Autonomous Discovery Engine.
"""

from .defi import DeFiIngester, ContractData, DeFiSnapshot

__all__ = [
    "DeFiIngester",
    "ContractData", 
    "DeFiSnapshot",
]
