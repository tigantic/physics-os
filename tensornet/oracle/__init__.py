"""
ORACLE: Offensive Reasoning and Assumption-Challenging Logic Engine
====================================================================

Automated smart contract vulnerability hunting through:
1. Semantic understanding (LLM-powered intent extraction)
2. Assumption extraction (explicit + implicit)
3. Assumption challenging (reachability + impact)
4. Adversarial scenario generation
5. Multi-method verification (interval + concrete + Kantorovich)
6. Report synthesis (Immunefi-ready)

Core Insight: Bugs are ASSUMPTION FAILURES, not pattern matches.

Usage:
    from tensornet.oracle import ORACLE
    
    oracle = ORACLE(
        anthropic_key="sk-...",
        eth_rpc="https://eth-mainnet.g.alchemy.com/v2/..."
    )
    
    results = oracle.hunt(address="0x1234...abcd")
    
    for exploit in results.verified_exploits:
        report = oracle.generate_report(exploit)
        report.save("./reports/")

Constitution Compliance: Article IV (Verification), Sovereign Stack
"""

from tensornet.oracle.core.oracle import ORACLE
from tensornet.oracle.core.types import (
    Assumption,
    AssumptionType,
    AttackScenario,
    Challenge,
    Contract,
    Function,
    HuntResult,
    ImpactLevel,
    Report,
    VerifiedExploit,
)

__all__ = [
    "ORACLE",
    "Contract",
    "Function",
    "Assumption",
    "AssumptionType",
    "Challenge",
    "AttackScenario",
    "HuntResult",
    "ImpactLevel",
    "Report",
    "VerifiedExploit",
]

__version__ = "0.1.0"
