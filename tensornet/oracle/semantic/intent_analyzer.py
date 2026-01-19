"""
LLM-powered semantic understanding of contract intent.

This is where ORACLE differs from pattern-matching tools:
We understand WHAT the contract is trying to do, not just how it's written.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from tensornet.oracle.core.types import (
    Actor,
    Contract,
    IntentAnalysis,
    ValueFlow,
)


class IntentAnalyzer:
    """
    Extract semantic intent using LLM.
    
    Answers: "What is this contract trying to do?"
    """
    
    # Protocol type signatures for quick classification
    PROTOCOL_SIGNATURES = {
        "lending": ["borrow", "lend", "collateral", "liquidate", "healthFactor"],
        "dex": ["swap", "addLiquidity", "removeLiquidity", "getAmountOut", "pair"],
        "vault": ["deposit", "withdraw", "shares", "assets", "totalAssets", "ERC4626"],
        "bridge": ["bridge", "relay", "messageHash", "crossChain", "L1", "L2"],
        "staking": ["stake", "unstake", "reward", "delegate", "slash"],
        "governance": ["propose", "vote", "execute", "quorum", "timelock"],
        "nft": ["mint", "tokenURI", "ERC721", "ownerOf", "safeTransferFrom"],
        "oracle": ["latestAnswer", "latestRoundData", "price", "updatePrice"],
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the intent analyzer.
        
        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None
    
    @property
    def client(self):
        """Lazy-load Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package required. Install with: pip install anthropic")
        return self._client
    
    def analyze(self, contract: Contract, source: Optional[str] = None) -> IntentAnalysis:
        """
        Use LLM to understand what contract is trying to do.
        
        Args:
            contract: Parsed Contract object
            source: Original source code (uses contract.source if not provided)
            
        Returns:
            IntentAnalysis with protocol type, actors, value flows, etc.
        """
        source = source or contract.source
        
        # First, try quick classification based on function names
        quick_type = self._quick_classify(contract)
        
        # Then use LLM for deep analysis
        prompt = self._build_analysis_prompt(contract, source, quick_type)
        
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return self._parse_response(response.content[0].text, quick_type)
        except Exception as e:
            # Fallback to basic analysis without LLM
            return self._basic_analysis(contract, quick_type)
    
    def _quick_classify(self, contract: Contract) -> str:
        """Quick classification based on function signatures."""
        func_names = [f.name.lower() for f in contract.functions]
        all_text = " ".join(func_names)
        
        scores = {}
        for protocol_type, keywords in self.PROTOCOL_SIGNATURES.items():
            score = sum(1 for kw in keywords if kw.lower() in all_text)
            if score > 0:
                scores[protocol_type] = score
        
        if scores:
            return max(scores, key=scores.get)
        return "unknown"
    
    def _build_analysis_prompt(self, contract: Contract, source: str, 
                               hint_type: str) -> str:
        """Build the LLM analysis prompt."""
        
        # Summarize functions for context
        func_summary = "\n".join([
            f"  - {f.name}({', '.join(p.type_name + ' ' + p.name for p in f.parameters)})"
            f" → {', '.join(r.type_name for r in f.returns) or 'void'}"
            f" [{f.visibility}, {f.mutability}]"
            for f in contract.functions[:20]  # Limit to avoid token overflow
        ])
        
        state_summary = "\n".join([
            f"  - {sv.type_name} {sv.name} [{sv.visibility}]"
            for sv in contract.state_variables[:15]
        ])
        
        return f"""Analyze this Solidity contract and extract its semantic intent.

CONTRACT: {contract.name}
TYPE HINT: {hint_type}
INHERITS: {', '.join(contract.inherits) or 'None'}

STATE VARIABLES:
{state_summary}

FUNCTIONS:
{func_summary}

FULL SOURCE:
```solidity
{source[:8000]}  
```

Provide a JSON response with:
{{
    "protocol_type": "lending|dex|vault|bridge|staking|governance|nft|oracle|other",
    "description": "High-level description of what this contract does (2-3 sentences)",
    "actors": [
        {{"name": "depositor", "role": "description of role", "capabilities": ["deposit", "withdraw"], "trust_level": "untrusted|semi-trusted|trusted"}}
    ],
    "value_flows": [
        {{"from_actor": "depositor", "to_actor": "contract", "asset": "ETH|ERC20", "condition": "when depositing", "function": "deposit"}}
    ],
    "trust_assumptions": [
        "Oracle prices are accurate and timely",
        "Admin does not act maliciously"
    ],
    "key_invariants": [
        "Total deposits >= total withdrawals",
        "User can always withdraw their own funds"
    ]
}}

Be specific and thorough. Focus on security-relevant details."""
    
    def _parse_response(self, response_text: str, fallback_type: str) -> IntentAnalysis:
        """Parse LLM response into IntentAnalysis."""
        try:
            # Extract JSON from response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(response_text[json_start:json_end])
            else:
                raise ValueError("No JSON found in response")
            
            actors = [
                Actor(
                    name=a.get("name", "unknown"),
                    role=a.get("role", ""),
                    capabilities=a.get("capabilities", []),
                    trust_level=a.get("trust_level", "untrusted"),
                )
                for a in data.get("actors", [])
            ]
            
            value_flows = [
                ValueFlow(
                    from_actor=vf.get("from_actor", ""),
                    to_actor=vf.get("to_actor", ""),
                    asset=vf.get("asset", ""),
                    condition=vf.get("condition", ""),
                    function=vf.get("function", ""),
                )
                for vf in data.get("value_flows", [])
            ]
            
            return IntentAnalysis(
                protocol_type=data.get("protocol_type", fallback_type),
                description=data.get("description", ""),
                actors=actors,
                value_flows=value_flows,
                trust_assumptions=data.get("trust_assumptions", []),
                key_invariants=data.get("key_invariants", []),
            )
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Return basic analysis on parse failure
            return IntentAnalysis(
                protocol_type=fallback_type,
                description=f"Failed to parse LLM response: {e}",
                actors=[],
                value_flows=[],
                trust_assumptions=[],
                key_invariants=[],
            )
    
    def _basic_analysis(self, contract: Contract, protocol_type: str) -> IntentAnalysis:
        """Basic analysis without LLM."""
        
        # Infer actors from function names
        actors = []
        if any("deposit" in f.name.lower() for f in contract.functions):
            actors.append(Actor(name="depositor", role="Deposits assets", 
                               capabilities=["deposit"], trust_level="untrusted"))
        if any("withdraw" in f.name.lower() for f in contract.functions):
            actors.append(Actor(name="withdrawer", role="Withdraws assets",
                               capabilities=["withdraw"], trust_level="untrusted"))
        if any("admin" in f.name.lower() or "owner" in f.name.lower() 
               for f in contract.functions):
            actors.append(Actor(name="admin", role="Administrative functions",
                               capabilities=["admin"], trust_level="trusted"))
        
        # Standard trust assumptions
        trust_assumptions = []
        if protocol_type == "lending":
            trust_assumptions = [
                "Oracle prices are accurate",
                "Liquidators act when profitable",
                "Interest rate model is economically sound",
            ]
        elif protocol_type == "dex":
            trust_assumptions = [
                "Arbitrageurs keep prices aligned with market",
                "No sandwich attacks on large trades",
            ]
        elif protocol_type == "vault":
            trust_assumptions = [
                "Share price calculation is accurate",
                "Underlying strategy is safe",
            ]
        
        return IntentAnalysis(
            protocol_type=protocol_type,
            description=f"A {protocol_type} protocol with {len(contract.functions)} functions",
            actors=actors,
            value_flows=[],
            trust_assumptions=trust_assumptions,
            key_invariants=[],
        )
