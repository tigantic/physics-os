"""
Extract implicit assumptions using LLM reasoning.

Implicit assumptions are the DANGEROUS ones:
- Not checked with require/assert
- Developer assumed they'd always be true
- Often the root cause of exploits
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from ontic.infra.oracle.core.types import (
    Assumption,
    AssumptionType,
    Contract,
    Function,
    IntentAnalysis,
)


class ImplicitExtractor:
    """
    Use LLM to find assumptions NOT explicitly checked.
    
    These are the bugs waiting to happen.
    """
    
    # Common implicit assumption patterns
    PATTERNS = {
        "reentrancy_safe": "Assumes no reentrant calls during execution",
        "oracle_fresh": "Assumes oracle data is current (not stale)",
        "oracle_accurate": "Assumes oracle data reflects true market value",
        "no_flash_loan": "Assumes funds weren't flash loaned",
        "monotonic_time": "Assumes block.timestamp always increases",
        "caller_is_eoa": "Assumes msg.sender is an EOA, not a contract",
        "token_standard_compliant": "Assumes ERC20 behaves according to standard",
        "no_fee_on_transfer": "Assumes token doesn't take fees on transfer",
        "no_rebasing": "Assumes token balance doesn't change without transfer",
        "sufficient_liquidity": "Assumes enough liquidity exists for operation",
        "no_front_running": "Assumes transaction ordering doesn't matter",
        "no_sandwich": "Assumes no sandwich attacks on trades",
        "gas_sufficient": "Assumes enough gas for external calls",
        "no_overflow": "Assumes arithmetic doesn't overflow",
        "precision_safe": "Assumes division rounding is acceptable",
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the implicit extractor.
        
        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None
        self._assumption_id = 0
    
    @property
    def client(self):
        """Lazy-load Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package required")
        return self._client
    
    def extract(self, contract: Contract, intent: IntentAnalysis) -> list[Assumption]:
        """
        Extract implicit assumptions from contract.
        
        Args:
            contract: Parsed contract
            intent: Semantic intent analysis
            
        Returns:
            List of implicit assumptions
        """
        self._assumption_id = 0
        assumptions = []
        
        # Per-function analysis
        for func in contract.functions:
            if func.is_public:  # Focus on externally callable functions
                func_assumptions = self._analyze_function(func, contract, intent)
                assumptions.extend(func_assumptions)
        
        # Contract-level analysis
        contract_assumptions = self._analyze_contract_level(contract, intent)
        assumptions.extend(contract_assumptions)
        
        # Deduplicate similar assumptions
        assumptions = self._deduplicate(assumptions)
        
        return assumptions
    
    def _next_id(self) -> str:
        """Generate next assumption ID."""
        self._assumption_id += 1
        return f"I{self._assumption_id:03d}"
    
    def _analyze_function(self, func: Function, contract: Contract,
                          intent: IntentAnalysis) -> list[Assumption]:
        """Analyze a single function for implicit assumptions."""
        assumptions = []
        source = func.source
        
        # Pattern-based detection (fast, no LLM needed)
        
        # 1. External calls without reentrancy guard
        # Check both func.external_calls and raw .call patterns
        has_external_call = bool(func.external_calls)
        has_low_level_call = any(
            pattern in source.lower() 
            for pattern in [".call(", ".call{", ".delegatecall(", ".delegatecall{"]
        )
        has_reentrancy_guard = (
            "nonReentrant" in func.modifiers or
            "ReentrancyGuard" in " ".join(contract.inherits) or
            "nonreentrant" in source.lower()
        )
        
        if (has_external_call or has_low_level_call) and not has_reentrancy_guard:
            assumptions.append(Assumption(
                id=self._next_id(),
                type=AssumptionType.IMPLICIT,
                source=func.name,
                statement="No reentrancy during external calls",
                formal="∀ call ∈ external_calls: no_callback(call)",
                confidence=0.95,  # HIGH confidence - this is a critical assumption
            ))
            
            # Additional: Check if state is written AFTER external call (classic reentrancy)
            if has_low_level_call:
                # Find the call position
                patterns = [".call(", ".call{", ".delegatecall(", ".delegatecall{"]
                positions = [source.find(p) for p in patterns]
                call_pos = max((p for p in positions if p >= 0), default=-1)
                
                if call_pos > 0:
                    # Skip past the call statement (find next semicolon)
                    semicolon_pos = source.find(";", call_pos)
                    if semicolon_pos > 0:
                        after_call = source[semicolon_pos + 1:]
                        # Check for state writes (including mapping/array assignments)
                        state_write_pattern = r"\w+\[.+\]\s*=\s*[^=]|\w+\s*=\s*[^=]"
                        if re.search(state_write_pattern, after_call):
                            assumptions.append(Assumption(
                                id=self._next_id(),
                                type=AssumptionType.IMPLICIT,
                                source=func.name,
                                statement="⚠️ CEI VIOLATION: State updated AFTER external call - REENTRANCY RISK",
                                formal="⚠️ CRITICAL: external_call() → state_write violates Checks-Effects-Interactions",
                                confidence=0.99,  # EXTREMELY HIGH - classic reentrancy pattern
                            ))
        
        # 2. Access control - state-changing functions without auth
        is_state_changing = (
            func.writes_state or
            ".transfer(" in source or
            ".call{" in source or
            "selfdestruct" in source.lower()
        )
        has_access_control = (
            "onlyOwner" in " ".join(func.modifiers) or
            "onlyAdmin" in " ".join(func.modifiers) or
            "require(msg.sender ==" in source or
            "require(msg.sender==" in source or
            "if (msg.sender !=" in source or
            "_checkOwner" in source or
            "Ownable" in " ".join(contract.inherits) or
            "AccessControl" in " ".join(contract.inherits)
        )
        
        if is_state_changing and not has_access_control and func.visibility in ("external", "public"):
            # Check for sensitive operations
            sensitive_ops = [
                ("owner", "owner modification"),
                ("admin", "admin modification"),
                ("withdraw", "fund withdrawal"),
                ("transfer", "token transfer"),
                ("mint", "token minting"),
                ("burn", "token burning"),
                ("pause", "contract pause"),
                ("upgrade", "contract upgrade"),
                ("set", "parameter setting"),
            ]
            for keyword, desc in sensitive_ops:
                if keyword in func.name.lower():
                    assumptions.append(Assumption(
                        id=self._next_id(),
                        type=AssumptionType.IMPLICIT,
                        source=func.name,
                        statement=f"⚠️ MISSING ACCESS CONTROL: {desc} without authorization check",
                        formal=f"⚠️ CRITICAL: {func.name}() modifies state without msg.sender verification",
                        confidence=0.97,
                    ))
                    break
        
        # 3. Token transfers without balance checks
        if ".transfer(" in source or ".safeTransfer(" in source:
            if "balanceOf" not in source:
                assumptions.append(Assumption(
                    id=self._next_id(),
                    type=AssumptionType.IMPLICIT,
                    source=func.name,
                    statement="Sufficient token balance for transfer",
                    formal="balance[this] ≥ amount",
                    confidence=0.8,
                ))
        
        # 4. Division without zero check
        if "/" in source:
            # Check if divisor is checked
            div_pattern = r'/\s*(\w+)'
            for match in re.finditer(div_pattern, source):
                divisor = match.group(1)
                if f"{divisor} > 0" not in source and f"{divisor} != 0" not in source:
                    assumptions.append(Assumption(
                        id=self._next_id(),
                        type=AssumptionType.IMPLICIT,
                        source=func.name,
                        statement=f"Divisor {divisor} is non-zero",
                        formal=f"{divisor} ≠ 0",
                        confidence=0.7,
                    ))
        
        # 4. Price/rate calculations
        if any(kw in source.lower() for kw in ["price", "rate", "oracle", "getprice"]):
            assumptions.append(Assumption(
                id=self._next_id(),
                type=AssumptionType.EXTERNAL,
                source=func.name,
                statement="Oracle price is accurate and current",
                formal="|oracle_price - market_price| < ε ∧ age(price) < max_staleness",
                confidence=0.85,
            ))
        
        # 5. Timestamp usage
        if "block.timestamp" in source:
            if intent.protocol_type in ("lending", "vault", "staking"):
                assumptions.append(Assumption(
                    id=self._next_id(),
                    type=AssumptionType.TEMPORAL,
                    source=func.name,
                    statement="Block timestamp cannot be manipulated significantly",
                    formal="block.timestamp ≈ real_time ± 15s",
                    confidence=0.75,
                ))
        
        # 6. First depositor protection (for vaults)
        if intent.protocol_type == "vault" and func.name.lower() in ("deposit", "mint"):
            if "totalSupply" in source and "totalAssets" in source:
                assumptions.append(Assumption(
                    id=self._next_id(),
                    type=AssumptionType.ECONOMIC,
                    source=func.name,
                    statement="First depositor cannot manipulate share price",
                    formal="totalSupply == 0 → shares = assets (no inflation attack)",
                    confidence=0.9,
                ))
        
        # 7. Liquidation thresholds
        if intent.protocol_type == "lending" and "liquidate" in func.name.lower():
            assumptions.append(Assumption(
                id=self._next_id(),
                type=AssumptionType.ECONOMIC,
                source=func.name,
                statement="Liquidators will act before position becomes insolvent",
                formal="∀ position: collateral_value/debt < threshold → liquidated promptly",
                confidence=0.7,
            ))
        
        return assumptions
    
    def _analyze_contract_level(self, contract: Contract, 
                                 intent: IntentAnalysis) -> list[Assumption]:
        """Analyze contract-level implicit assumptions."""
        assumptions = []
        
        # Add trust assumptions from intent analysis
        for trust in intent.trust_assumptions:
            assumptions.append(Assumption(
                id=self._next_id(),
                type=AssumptionType.EXTERNAL,
                source="global",
                statement=trust,
                confidence=0.8,
            ))
        
        # Protocol-specific assumptions
        if intent.protocol_type == "lending":
            assumptions.extend([
                Assumption(
                    id=self._next_id(),
                    type=AssumptionType.ECONOMIC,
                    source="global",
                    statement="Interest rate model maintains protocol solvency",
                    formal="∀ t: total_deposits + accrued_interest ≤ available_liquidity + outstanding_borrows",
                    confidence=0.7,
                ),
                Assumption(
                    id=self._next_id(),
                    type=AssumptionType.EXTERNAL,
                    source="global",
                    statement="Collateral assets maintain sufficient liquidity",
                    formal="∀ asset: market_liquidity(asset) > liquidation_needs",
                    confidence=0.6,
                ),
            ])
        
        elif intent.protocol_type == "dex":
            assumptions.extend([
                Assumption(
                    id=self._next_id(),
                    type=AssumptionType.ECONOMIC,
                    source="global",
                    statement="Arbitrageurs keep pool prices aligned with market",
                    formal="|pool_price - market_price| < arb_threshold",
                    confidence=0.7,
                ),
                Assumption(
                    id=self._next_id(),
                    type=AssumptionType.IMPLICIT,
                    source="global",
                    statement="Pool tokens conform to ERC20 standard",
                    formal="∀ token: compliant(token, ERC20)",
                    confidence=0.8,
                ),
            ])
        
        elif intent.protocol_type == "vault":
            assumptions.extend([
                Assumption(
                    id=self._next_id(),
                    type=AssumptionType.ARITHMETIC,
                    source="global",
                    statement="Share/asset conversion doesn't cause precision loss exploitation",
                    formal="shares_to_assets(assets_to_shares(x)) ≈ x",
                    confidence=0.85,
                ),
            ])
        
        return assumptions
    
    def analyze_with_llm(self, func: Function, contract: Contract) -> list[Assumption]:
        """
        Deep LLM analysis for complex implicit assumptions.
        
        Use this for high-value targets where pattern matching isn't enough.
        """
        prompt = f"""Analyze this Solidity function and identify ALL implicit assumptions.

An implicit assumption is something the code ASSUMES to be true but does NOT explicitly check.

Function: {func.name}
Contract: {contract.name}
Visibility: {func.visibility}
Mutability: {func.mutability}

```solidity
{func.source}
```

Categories to consider:
1. REENTRANCY: Does it assume no callbacks during execution?
2. ORACLE: Does it assume price data is fresh/accurate?
3. TOKEN BEHAVIOR: Does it assume standard ERC20/ERC721 behavior?
4. TIMING: Does it assume certain ordering of operations?
5. EXTERNAL: Does it assume external contracts behave correctly?
6. ECONOMIC: Does it assume rational/honest actors?
7. ARITHMETIC: Does it assume no overflow/precision issues?

For each assumption, provide JSON:
{{
    "assumptions": [
        {{
            "category": "REENTRANCY|ORACLE|TOKEN|TIMING|EXTERNAL|ECONOMIC|ARITHMETIC",
            "statement": "Clear description of what is assumed",
            "what_if_violated": "What could go wrong",
            "confidence": 0.0-1.0
        }}
    ]
}}

Be thorough. Think like an attacker."""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return self._parse_llm_response(response.content[0].text, func.name)
        except Exception:
            return []
    
    def _parse_llm_response(self, text: str, func_name: str) -> list[Assumption]:
        """Parse LLM response into assumptions."""
        assumptions = []
        
        try:
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(text[json_start:json_end])
                
                for a in data.get("assumptions", []):
                    category = a.get("category", "IMPLICIT")
                    type_map = {
                        "REENTRANCY": AssumptionType.IMPLICIT,
                        "ORACLE": AssumptionType.EXTERNAL,
                        "TOKEN": AssumptionType.EXTERNAL,
                        "TIMING": AssumptionType.TEMPORAL,
                        "EXTERNAL": AssumptionType.EXTERNAL,
                        "ECONOMIC": AssumptionType.ECONOMIC,
                        "ARITHMETIC": AssumptionType.ARITHMETIC,
                    }
                    
                    assumptions.append(Assumption(
                        id=self._next_id(),
                        type=type_map.get(category, AssumptionType.IMPLICIT),
                        source=func_name,
                        statement=a.get("statement", "Unknown"),
                        confidence=a.get("confidence", 0.5),
                    ))
        except (json.JSONDecodeError, KeyError):
            pass
        
        return assumptions
    
    def _deduplicate(self, assumptions: list[Assumption]) -> list[Assumption]:
        """Remove near-duplicate assumptions."""
        seen_statements = set()
        unique = []
        
        for a in assumptions:
            # Normalize statement for comparison
            normalized = a.statement.lower().strip()
            
            # Skip if we've seen something very similar
            is_dup = False
            for seen in seen_statements:
                if self._similarity(normalized, seen) > 0.8:
                    is_dup = True
                    break
            
            if not is_dup:
                seen_statements.add(normalized)
                unique.append(a)
        
        return unique
    
    def _similarity(self, s1: str, s2: str) -> float:
        """Simple word overlap similarity."""
        words1 = set(s1.split())
        words2 = set(s2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0


# Need this import for pattern matching
import re
