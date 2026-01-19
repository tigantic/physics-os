"""
Impact analyzer for assumption violations.

Determines WHAT HAPPENS when an assumption is violated:
- Critical: Direct fund theft
- High: Significant fund loss
- Medium: Limited loss or DoS
- Low: Minor issues
"""

from __future__ import annotations

import re
from typing import Optional

from tensornet.oracle.core.types import (
    Assumption,
    AssumptionType,
    ExecutionPath,
    ImpactLevel,
)


class ImpactAnalyzer:
    """
    Determine what happens when an assumption is violated.
    
    Maps assumption violations to Immunefi severity levels.
    """
    
    # Keywords that indicate critical impact
    CRITICAL_KEYWORDS = [
        "theft", "steal", "drain", "extract", "all funds", "total loss",
        "unauthorized withdrawal", "mint unlimited", "bypass auth",
    ]
    
    # Keywords that indicate high impact
    HIGH_KEYWORDS = [
        "significant loss", "material loss", "insolvency", "bad debt",
        "oracle manipulation", "price manipulation", "liquidation failure",
    ]
    
    # Keywords that indicate medium impact
    MEDIUM_KEYWORDS = [
        "temporary dos", "griefing", "partial loss", "front-running",
        "sandwich", "slippage", "fee extraction",
    ]
    
    # Assumption types that are typically critical
    CRITICAL_ASSUMPTION_TYPES = [
        ("reentrancy", ImpactLevel.CRITICAL),
        ("overflow", ImpactLevel.HIGH),
        ("oracle", ImpactLevel.HIGH),
        ("authorization", ImpactLevel.CRITICAL),
        ("flash loan", ImpactLevel.HIGH),
    ]
    
    def analyze(
        self,
        assumption: Assumption,
        violation_path: Optional[ExecutionPath] = None,
    ) -> tuple[ImpactLevel, str]:
        """
        Analyze impact of assumption violation.
        
        Args:
            assumption: The violated assumption
            violation_path: How the violation was reached
            
        Returns:
            (impact_level, description)
        """
        # Start with type-based assessment
        base_level = self._assess_by_type(assumption)
        
        # Refine based on statement content
        content_level = self._assess_by_content(assumption)
        
        # Use the higher severity
        impact = max(base_level, content_level)
        
        # Generate description
        description = self._generate_description(assumption, impact, violation_path)
        
        return impact, description
    
    def _assess_by_type(self, assumption: Assumption) -> ImpactLevel:
        """Assess impact based on assumption type."""
        
        # Economic assumptions often have high impact
        if assumption.type == AssumptionType.ECONOMIC:
            return ImpactLevel.HIGH
        
        # External assumptions (oracles, etc.) are dangerous
        if assumption.type == AssumptionType.EXTERNAL:
            return ImpactLevel.HIGH
        
        # Explicit assumptions (require/assert) vary
        if assumption.type == AssumptionType.EXPLICIT:
            # Check for auth-related
            if any(kw in assumption.statement.lower() 
                   for kw in ["owner", "admin", "auth", "only"]):
                return ImpactLevel.CRITICAL
            # Bounds checking
            if any(kw in assumption.statement.lower()
                   for kw in ["amount", "balance", "> 0"]):
                return ImpactLevel.MEDIUM
            return ImpactLevel.MEDIUM
        
        # Implicit assumptions are often critical (the hidden bugs)
        if assumption.type == AssumptionType.IMPLICIT:
            return ImpactLevel.HIGH
        
        # Arithmetic issues
        if assumption.type == AssumptionType.ARITHMETIC:
            return ImpactLevel.MEDIUM
        
        # Temporal issues
        if assumption.type == AssumptionType.TEMPORAL:
            return ImpactLevel.MEDIUM
        
        return ImpactLevel.LOW
    
    def _assess_by_content(self, assumption: Assumption) -> ImpactLevel:
        """Assess impact based on assumption statement content."""
        statement = assumption.statement.lower()
        
        # Check for critical keywords
        if any(kw in statement for kw in self.CRITICAL_KEYWORDS):
            return ImpactLevel.CRITICAL
        
        # Check for reentrancy
        if "reentran" in statement:
            return ImpactLevel.CRITICAL
        
        # Check for authorization bypass
        if any(kw in statement for kw in ["caller must be", "only owner", "authorized"]):
            return ImpactLevel.CRITICAL
        
        # Check for high impact keywords
        if any(kw in statement for kw in self.HIGH_KEYWORDS):
            return ImpactLevel.HIGH
        
        # Oracle/price related
        if any(kw in statement for kw in ["oracle", "price", "rate"]):
            return ImpactLevel.HIGH
        
        # Check for medium impact keywords
        if any(kw in statement for kw in self.MEDIUM_KEYWORDS):
            return ImpactLevel.MEDIUM
        
        # Balance/amount related
        if any(kw in statement for kw in ["balance", "amount", "deposit", "withdraw"]):
            return ImpactLevel.MEDIUM
        
        return ImpactLevel.LOW
    
    def _generate_description(
        self,
        assumption: Assumption,
        impact: ImpactLevel,
        violation_path: Optional[ExecutionPath],
    ) -> str:
        """Generate human-readable impact description."""
        
        base = f"Violation of: {assumption.statement}"
        
        if impact == ImpactLevel.CRITICAL:
            consequence = self._critical_consequence(assumption)
        elif impact == ImpactLevel.HIGH:
            consequence = self._high_consequence(assumption)
        elif impact == ImpactLevel.MEDIUM:
            consequence = self._medium_consequence(assumption)
        else:
            consequence = "Minor protocol misbehavior possible."
        
        path_desc = ""
        if violation_path and violation_path.calls:
            steps = [f"{call[0]}()" for call in violation_path.calls[:5]]
            path_desc = f" Attack path: {' → '.join(steps)}"
        
        return f"{base}\n\nConsequence: {consequence}{path_desc}"
    
    def _critical_consequence(self, assumption: Assumption) -> str:
        """Generate critical impact consequence."""
        statement = assumption.statement.lower()
        
        if "reentran" in statement:
            return "Attacker can drain funds through reentrancy attack."
        if "owner" in statement or "admin" in statement:
            return "Attacker can bypass authorization and access privileged functions."
        if "oracle" in statement:
            return "Attacker can manipulate oracle to steal funds."
        if "balance" in statement:
            return "Attacker can extract more funds than deposited."
        
        return "Direct theft of user or protocol funds possible."
    
    def _high_consequence(self, assumption: Assumption) -> str:
        """Generate high impact consequence."""
        statement = assumption.statement.lower()
        
        if "liquidat" in statement:
            return "Liquidation mechanism can fail, leading to protocol insolvency."
        if "oracle" in statement or "price" in statement:
            return "Price manipulation can lead to significant fund loss."
        if "collateral" in statement:
            return "Undercollateralized positions can cause bad debt."
        
        return "Significant loss of funds or protocol value possible."
    
    def _medium_consequence(self, assumption: Assumption) -> str:
        """Generate medium impact consequence."""
        statement = assumption.statement.lower()
        
        if "dos" in statement or "revert" in statement:
            return "Denial of service attack possible."
        if "precision" in statement or "rounding" in statement:
            return "Precision loss can be exploited for small profit over many transactions."
        if "slippage" in statement:
            return "Users may receive less than expected from trades."
        
        return "Limited fund loss or temporary service disruption possible."


# Convenience function
def assess_impact(
    assumption: Assumption,
    violation_path: Optional[ExecutionPath] = None,
) -> tuple[ImpactLevel, str]:
    """
    Quick impact assessment.
    
    Args:
        assumption: The violated assumption
        violation_path: How the violation was reached
        
    Returns:
        (impact_level, description)
    """
    analyzer = ImpactAnalyzer()
    return analyzer.analyze(assumption, violation_path)
