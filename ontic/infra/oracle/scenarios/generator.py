"""
LLM-powered attack scenario generation.

This is where ORACLE gets creative:
- Takes vulnerable assumptions
- Generates concrete attack scenarios
- Combines multiple weaknesses
- Proposes novel attacks beyond known patterns
"""

from __future__ import annotations

import json
import os
from typing import Optional

from ontic.infra.oracle.core.types import (
    AttackScenario,
    AttackStep,
    Challenge,
    Contract,
    ImpactLevel,
    IntentAnalysis,
)


class ScenarioGenerator:
    """
    Generate creative attack scenarios using LLM + pattern library.
    
    Combines known attack patterns with LLM creativity.
    """
    
    # Known attack patterns to prime generation
    ATTACK_PATTERNS = {
        "flash_loan_manipulation": {
            "name": "Flash Loan Price Manipulation",
            "steps": [
                "Flash loan large amount from Aave/dYdX",
                "Use funds to manipulate target (swap, deposit, etc.)",
                "Profit from manipulated state",
                "Repay flash loan",
            ],
            "complexity": "MEDIUM",
            "capital": 0,  # Flash loans require no capital
        },
        "first_depositor": {
            "name": "First Depositor Inflation Attack",
            "steps": [
                "Be first to deposit tiny amount (1 wei)",
                "Donate large amount directly to vault (no shares)",
                "Next depositor gets 0 shares due to rounding",
                "Withdraw to steal their deposit",
            ],
            "complexity": "LOW",
            "capital": 1,  # Just 1 wei
        },
        "reentrancy": {
            "name": "Reentrancy Attack",
            "steps": [
                "Deploy attacker contract with receive/fallback",
                "Call vulnerable function that sends ETH",
                "Re-enter during callback before state update",
                "Repeat until drained",
            ],
            "complexity": "LOW",
            "capital": 1,  # Minimal seed capital
        },
        "oracle_manipulation": {
            "name": "Oracle Manipulation",
            "steps": [
                "Identify oracle price source",
                "Manipulate price (TWAP gaming, spot manipulation)",
                "Borrow against inflated collateral / liquidate at manipulated price",
                "Let price revert, profit from discrepancy",
            ],
            "complexity": "HIGH",
            "capital": 1000000,  # Need significant capital
        },
        "governance_attack": {
            "name": "Flash Loan Governance Attack",
            "steps": [
                "Flash loan governance tokens",
                "Create malicious proposal or vote on pending",
                "Execute if timelock allows",
                "Repay flash loan",
            ],
            "complexity": "HIGH",
            "capital": 0,
        },
        "precision_loss": {
            "name": "Precision Loss Exploitation",
            "steps": [
                "Identify rounding direction in share/asset conversion",
                "Execute many small transactions at rounding boundary",
                "Accumulate dust profits",
                "Compound over time",
            ],
            "complexity": "MEDIUM",
            "capital": 10000,
        },
        "sandwich": {
            "name": "Sandwich Attack",
            "steps": [
                "Monitor mempool for large trades",
                "Front-run with large buy",
                "Let victim trade at worse price",
                "Back-run with sell",
            ],
            "complexity": "MEDIUM",
            "capital": 100000,
        },
        "read_only_reentrancy": {
            "name": "Read-Only Reentrancy",
            "steps": [
                "Enter target during callback",
                "Read stale state from secondary protocol",
                "Use stale state to profit",
                "Exit cleanly",
            ],
            "complexity": "HIGH",
            "capital": 10000,
        },
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize scenario generator.
        
        Args:
            api_key: Anthropic API key
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
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
    
    def generate(
        self,
        contract: Contract,
        challenges: list[Challenge],
        intent: IntentAnalysis,
    ) -> list[AttackScenario]:
        """
        Generate attack scenarios from vulnerable assumptions.
        
        Args:
            contract: Target contract
            challenges: Validated vulnerable assumptions
            intent: Semantic intent analysis
            
        Returns:
            List of attack scenarios
        """
        scenarios = []
        
        # Generate scenarios from each challenge (pattern-based)
        for challenge in challenges:
            scenario = self._challenge_to_scenario(challenge, contract, intent)
            if scenario:
                scenarios.append(scenario)
        
        # Generate novel scenarios using LLM
        if self.client and challenges:
            novel = self._generate_novel_scenarios(contract, challenges, intent)
            scenarios.extend(novel)
        
        # Deduplicate
        scenarios = self._deduplicate(scenarios)
        
        # Rank by feasibility and profit
        scenarios = self._rank(scenarios)
        
        return scenarios
    
    def _challenge_to_scenario(
        self,
        challenge: Challenge,
        contract: Contract,
        intent: IntentAnalysis,
    ) -> Optional[AttackScenario]:
        """Convert a challenge to an attack scenario."""
        
        assumption = challenge.assumption
        statement = assumption.statement.lower()
        
        # Match to known patterns
        pattern_key = None
        
        if "reentran" in statement:
            pattern_key = "reentrancy"
        elif "oracle" in statement or "price" in statement:
            pattern_key = "oracle_manipulation"
        elif "first" in statement and "deposit" in statement:
            pattern_key = "first_depositor"
        elif "precision" in statement or "rounding" in statement:
            pattern_key = "precision_loss"
        elif "flash" in statement and "govern" in statement:
            pattern_key = "governance_attack"
        elif "sandwich" in statement or "front" in statement:
            pattern_key = "sandwich"
        
        if pattern_key:
            pattern = self.ATTACK_PATTERNS[pattern_key]
            
            steps = [
                AttackStep(
                    action=step,
                    target=contract.name,
                    description=step,
                )
                for step in pattern["steps"]
            ]
            
            return AttackScenario(
                name=f"{pattern['name']} on {contract.name}",
                description=f"Exploit: {challenge.assumption.statement}",
                steps=steps,
                required_capital=pattern["capital"],
                expected_profit=self._estimate_profit(challenge, contract),
                prerequisites=[f"Vulnerability: {assumption.statement}"],
                complexity=pattern["complexity"],
                challenges_addressed=[assumption.id],
            )
        
        # Generic scenario for unmatched challenges
        return AttackScenario(
            name=f"Exploit {assumption.id} in {contract.name}",
            description=challenge.exploit_sketch or f"Violate: {assumption.statement}",
            steps=[
                AttackStep(
                    action="Identify vulnerable state",
                    target=assumption.source,
                ),
                AttackStep(
                    action="Trigger assumption violation",
                    target=assumption.source,
                ),
                AttackStep(
                    action="Extract value",
                    target=contract.name,
                ),
            ],
            required_capital=10**18,  # 1 ETH default
            expected_profit=self._estimate_profit(challenge, contract),
            complexity="MEDIUM",
            challenges_addressed=[assumption.id],
        )
    
    def _generate_novel_scenarios(
        self,
        contract: Contract,
        challenges: list[Challenge],
        intent: IntentAnalysis,
    ) -> list[AttackScenario]:
        """Use LLM to generate creative attacks."""
        
        # Summarize challenges
        challenge_summary = "\n".join([
            f"- {c.assumption.statement} (Impact: {c.impact.value})"
            for c in challenges[:10]
        ])
        
        # Summarize functions
        func_summary = "\n".join([
            f"- {f.name}(): {f.visibility}, {f.mutability}"
            for f in contract.functions[:15]
        ])
        
        prompt = f"""You are an expert smart contract security researcher.

CONTRACT: {contract.name}
TYPE: {intent.protocol_type}
DESCRIPTION: {intent.description}

KEY FUNCTIONS:
{func_summary}

KNOWN VULNERABILITIES:
{challenge_summary}

Generate 3 creative attack scenarios. Consider:
1. Combining multiple vulnerabilities
2. Cross-protocol attacks (flash loans, composability)
3. Economic attacks that don't require code bugs
4. Timing/ordering attacks (MEV)
5. Multi-transaction sequences

For each scenario, provide JSON:
{{
    "scenarios": [
        {{
            "name": "Attack Name",
            "description": "2-3 sentence description",
            "steps": ["Step 1", "Step 2", ...],
            "required_capital_eth": 0,
            "expected_profit_eth": 1000,
            "complexity": "LOW|MEDIUM|HIGH",
            "prerequisites": ["Condition 1"]
        }}
    ]
}}

Be creative but realistic. Only suggest attacks that could actually work."""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return self._parse_llm_scenarios(response.content[0].text, contract)
        except Exception:
            return []
    
    def _parse_llm_scenarios(
        self,
        text: str,
        contract: Contract,
    ) -> list[AttackScenario]:
        """Parse LLM response into scenarios."""
        scenarios = []
        
        try:
            # Extract JSON
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(text[json_start:json_end])
                
                for s in data.get("scenarios", []):
                    steps = [
                        AttackStep(action=step, target=contract.name)
                        for step in s.get("steps", [])
                    ]
                    
                    scenarios.append(AttackScenario(
                        name=s.get("name", "Unknown"),
                        description=s.get("description", ""),
                        steps=steps,
                        required_capital=int(s.get("required_capital_eth", 0) * 10**18),
                        expected_profit=int(s.get("expected_profit_eth", 0) * 10**18),
                        complexity=s.get("complexity", "MEDIUM"),
                        prerequisites=s.get("prerequisites", []),
                    ))
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        
        return scenarios
    
    def _estimate_profit(self, challenge: Challenge, contract: Contract) -> int:
        """Estimate potential profit from exploit."""
        
        # Base estimates by impact level
        base_profits = {
            ImpactLevel.CRITICAL: 1_000_000 * 10**18,  # $1M
            ImpactLevel.HIGH: 100_000 * 10**18,       # $100K
            ImpactLevel.MEDIUM: 10_000 * 10**18,      # $10K
            ImpactLevel.LOW: 1_000 * 10**18,          # $1K
            ImpactLevel.INFORMATIONAL: 0,
        }
        
        return base_profits.get(challenge.impact, 0)
    
    def _deduplicate(self, scenarios: list[AttackScenario]) -> list[AttackScenario]:
        """Remove duplicate scenarios."""
        seen_names = set()
        unique = []
        
        for s in scenarios:
            normalized = s.name.lower().strip()
            if normalized not in seen_names:
                seen_names.add(normalized)
                unique.append(s)
        
        return unique
    
    def _rank(self, scenarios: list[AttackScenario]) -> list[AttackScenario]:
        """Rank scenarios by feasibility and expected profit."""
        
        def score(s: AttackScenario) -> float:
            profit = s.expected_profit
            capital = max(s.required_capital, 1)
            
            # ROI-like score, favor low capital
            roi = profit / capital
            
            # Complexity penalty
            complexity_mult = {
                "LOW": 1.0,
                "MEDIUM": 0.7,
                "HIGH": 0.4,
            }
            
            return roi * complexity_mult.get(s.complexity, 0.5)
        
        return sorted(scenarios, key=score, reverse=True)
