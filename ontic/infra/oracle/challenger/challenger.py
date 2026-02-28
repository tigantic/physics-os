"""
Main orchestrator for challenging assumptions.

This is where the hunt happens:
1. Take each assumption
2. Ask: "Can this be violated?"
3. If yes: "What's the impact?"
4. Generate exploit sketch
"""

from __future__ import annotations

import os
from typing import Optional

from ontic.infra.oracle.core.types import (
    Assumption,
    Challenge,
    Contract,
    ImpactLevel,
    IntentAnalysis,
)
from ontic.infra.oracle.verification.impact import ImpactAnalyzer
from ontic.infra.oracle.verification.reachability import ReachabilityChecker


class AssumptionChallenger:
    """
    For each assumption, determine if violation is reachable and impactful.
    
    This is ORACLE's core logic - the machine that finds bugs.
    """
    
    def __init__(
        self,
        contract: Contract,
        intent: Optional[IntentAnalysis] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the challenger.
        
        Args:
            contract: Contract to analyze
            intent: Semantic intent analysis
            api_key: Anthropic API key for LLM-assisted analysis
        """
        self.contract = contract
        self.intent = intent
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        
        self.reachability = ReachabilityChecker(contract)
        self.impact_analyzer = ImpactAnalyzer()
        
        self._client = None
    
    @property
    def client(self):
        """Lazy-load Anthropic client."""
        if self._client is None and self.api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                pass
        return self._client
    
    def challenge_all(
        self,
        assumptions: list[Assumption],
        min_confidence: float = 0.5,
    ) -> list[Challenge]:
        """
        Challenge every assumption, return those that can be violated.
        
        Args:
            assumptions: List of assumptions to challenge
            min_confidence: Minimum confidence threshold to consider
            
        Returns:
            List of challenges (potentially exploitable assumptions)
        """
        challenges = []
        
        for assumption in assumptions:
            # Skip low-confidence assumptions
            if assumption.confidence < min_confidence:
                continue
            
            challenge = self.challenge(assumption)
            
            # Only include if reachable and has meaningful impact
            if challenge.reachable and challenge.impact != ImpactLevel.INFORMATIONAL:
                challenges.append(challenge)
        
        # Sort by impact (critical first)
        challenges.sort(key=lambda c: c.impact, reverse=True)
        
        return challenges
    
    def challenge(self, assumption: Assumption) -> Challenge:
        """
        Challenge a single assumption.
        
        Args:
            assumption: The assumption to challenge
            
        Returns:
            Challenge result
        """
        # Generate negation
        negation = self._negate(assumption)
        
        # Check reachability
        reachable, path = self.reachability.check_assumption_violation(assumption)
        
        if not reachable:
            return Challenge(
                assumption=assumption,
                negation=negation,
                reachable=False,
                impact=ImpactLevel.INFORMATIONAL,
                impact_description="Violation not reachable",
            )
        
        # Analyze impact
        impact_level, impact_desc = self.impact_analyzer.analyze(assumption, path)
        
        # Generate exploit sketch using LLM
        exploit_sketch = self._generate_exploit_sketch(assumption, path, impact_level)
        
        return Challenge(
            assumption=assumption,
            negation=negation,
            reachable=True,
            reachability_proof=path,
            impact=impact_level,
            impact_description=impact_desc,
            exploit_sketch=exploit_sketch,
        )
    
    def _negate(self, assumption: Assumption) -> str:
        """Generate negation of assumption."""
        statement = assumption.statement
        
        # Simple negation patterns
        negations = {
            "must be positive": "is zero or negative",
            "must be non-zero": "is zero",
            "must equal": "does not equal",
            "must not equal": "equals",
            "must be at least": "is less than",
            "must be at most": "is greater than",
            "cannot be re-entered": "is re-entered",
            "is accurate": "is inaccurate/stale",
            "is trusted": "is malicious",
            "will act": "does not act",
        }
        
        result = statement
        for pattern, replacement in negations.items():
            if pattern in statement.lower():
                result = statement.lower().replace(pattern, replacement)
                return result.capitalize()
        
        return f"NOT: {statement}"
    
    def _generate_exploit_sketch(
        self,
        assumption: Assumption,
        path,
        impact: ImpactLevel,
    ) -> str:
        """Generate exploit sketch using LLM."""
        
        # Build context
        path_desc = ""
        if path and path.calls:
            steps = [f"{call[0]}({call[1]})" for call in path.calls[:5]]
            path_desc = " → ".join(steps)
        
        # Try LLM if available
        if self.client and impact >= ImpactLevel.MEDIUM:
            prompt = f"""An assumption violation was found in a smart contract:

Contract: {self.contract.name}
Assumption: {assumption.statement}
Violation reachable via: {path_desc or 'direct call'}
Impact: {impact.value}

Describe in 2-3 sentences how an attacker would exploit this.
Be specific about:
1. What the attacker does (concrete steps)
2. Why it works (the broken assumption)
3. What they gain (the profit/damage)

Keep it concise and actionable."""

            try:
                response = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            except Exception:
                pass
        
        # Fallback to template-based sketch
        return self._template_exploit_sketch(assumption, path_desc, impact)
    
    def _template_exploit_sketch(
        self,
        assumption: Assumption,
        path_desc: str,
        impact: ImpactLevel,
    ) -> str:
        """Generate template-based exploit sketch."""
        
        statement = assumption.statement.lower()
        
        # Reentrancy
        if "reentran" in statement or "callback" in statement:
            return ("1. Call vulnerable function\n"
                    "2. During external call, re-enter the function\n"
                    "3. State not yet updated, repeat extraction\n"
                    "4. Drain funds")
        
        # Oracle manipulation
        if "oracle" in statement or "price" in statement:
            return ("1. Flash loan large amount\n"
                    "2. Manipulate price (swap, deposit)\n"
                    "3. Trigger action at manipulated price (borrow, liquidate)\n"
                    "4. Profit from price discrepancy")
        
        # Authorization bypass
        if "owner" in statement or "admin" in statement or "auth" in statement:
            return ("1. Identify unprotected privileged function\n"
                    "2. Call function as unauthorized user\n"
                    "3. Execute privileged action (withdraw, upgrade)")
        
        # First depositor
        if "first" in statement and "deposit" in statement:
            return ("1. Be first depositor with tiny amount (1 wei)\n"
                    "2. Donate large amount directly to vault\n"
                    "3. Next depositor receives 0 shares due to rounding\n"
                    "4. Steal their entire deposit")
        
        # Precision/rounding
        if "precision" in statement or "rounding" in statement:
            return ("1. Identify rounding direction in conversion\n"
                    "2. Make many small transactions\n"
                    "3. Accumulate rounding error profit\n"
                    "4. Extract accumulated value")
        
        # Generic
        if path_desc:
            return f"Execute sequence: {path_desc}\nViolates: {assumption.statement}"
        
        return f"Violate: {assumption.statement}\nImpact: {impact.value}"


# Convenience function
def challenge_assumptions(
    contract: Contract,
    assumptions: list[Assumption],
) -> list[Challenge]:
    """
    Challenge all assumptions in a contract.
    
    Args:
        contract: The contract
        assumptions: Assumptions to challenge
        
    Returns:
        List of valid challenges
    """
    challenger = AssumptionChallenger(contract)
    return challenger.challenge_all(assumptions)
