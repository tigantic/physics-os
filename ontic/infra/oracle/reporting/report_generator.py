"""
Generate submission-ready vulnerability reports.

Output formats:
- Immunefi markdown
- Code4rena format
- JSON for archival
- Foundry test file
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

from ontic.infra.oracle.core.types import (
    ImpactLevel,
    Report,
    VerifiedExploit,
)


class ReportGenerator:
    """
    Generate Immunefi/Code4rena formatted reports.
    
    Takes verified exploits and produces submission-ready documents.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize report generator.
        
        Args:
            api_key: Anthropic API key for enhanced descriptions
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
    
    def generate(self, exploit: VerifiedExploit) -> Report:
        """
        Generate complete vulnerability report.
        
        Args:
            exploit: Verified exploit to report
            
        Returns:
            Report object ready for submission
        """
        return Report(
            title=self._generate_title(exploit),
            severity=self._classify_severity(exploit),
            summary=self._generate_summary(exploit),
            vulnerability_details=self._generate_details(exploit),
            impact=self._generate_impact(exploit),
            proof_of_concept=exploit.foundry_test,
            tools_used="ORACLE (Tigantic Holdings) + Foundry",
            recommendation=self._generate_recommendation(exploit),
        )
    
    def _generate_title(self, exploit: VerifiedExploit) -> str:
        """Generate concise, descriptive title."""
        scenario = exploit.scenario
        
        # Extract key attack type
        name = scenario.name
        
        # Make it sound professional
        if "reentrancy" in name.lower():
            return f"Critical: Reentrancy Vulnerability Allows Fund Drainage"
        elif "oracle" in name.lower():
            return f"High: Oracle Manipulation Enables Undercollateralized Borrowing"
        elif "first depositor" in name.lower():
            return f"Critical: First Depositor Inflation Attack"
        elif "flash" in name.lower() and "govern" in name.lower():
            return f"Critical: Flash Loan Governance Attack"
        elif "precision" in name.lower():
            return f"Medium: Precision Loss Allows Gradual Value Extraction"
        
        return f"{self._classify_severity(exploit)}: {name}"
    
    def _classify_severity(self, exploit: VerifiedExploit) -> str:
        """Classify according to Immunefi severity guidelines."""
        
        profit = 0
        if hasattr(exploit.proof, "profit"):
            profit = exploit.proof.profit
        else:
            profit = exploit.scenario.expected_profit
        
        # Convert to USD (assuming 1 ETH = $2000)
        profit_usd = (profit / 10**18) * 2000
        
        if profit_usd > 10_000_000:  # > $10M
            return "Critical"
        elif profit_usd > 1_000_000:  # > $1M
            return "Critical"
        elif profit_usd > 100_000:    # > $100K
            return "High"
        elif profit_usd > 10_000:     # > $10K
            return "Medium"
        elif profit_usd > 1_000:      # > $1K
            return "Low"
        else:
            return "Informational"
    
    def _generate_summary(self, exploit: VerifiedExploit) -> str:
        """Generate executive summary."""
        
        scenario = exploit.scenario
        profit = exploit.scenario.expected_profit / 10**18
        
        # Try LLM for better summary
        if self.client:
            prompt = f"""Write a clear, professional vulnerability summary for a bug bounty submission.

Attack: {scenario.name}
Steps: {[s.action for s in scenario.steps]}
Expected Profit: ~{profit:.2f} ETH
Confidence: {exploit.confidence * 100:.0f}%

The summary should:
1. State the vulnerability in one sentence
2. Explain the root cause (what assumption is broken)
3. State the impact (funds at risk)

Be concise and professional. No hype. Max 3 sentences."""

            try:
                response = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            except Exception:
                pass
        
        # Fallback to template
        return (
            f"A {scenario.complexity.lower()}-complexity attack allows an attacker to "
            f"extract approximately {profit:.2f} ETH from the protocol. "
            f"The vulnerability stems from {scenario.description[:100]}. "
            f"Exploitation requires {scenario.required_capital / 10**18:.2f} ETH in capital."
        )
    
    def _generate_details(self, exploit: VerifiedExploit) -> str:
        """Generate detailed vulnerability explanation."""
        
        scenario = exploit.scenario
        
        # Build step-by-step explanation
        steps_text = "\n".join([
            f"{i+1}. {step.action}"
            for i, step in enumerate(scenario.steps)
        ])
        
        prereqs = "\n".join([f"- {p}" for p in scenario.prerequisites]) if scenario.prerequisites else "None"
        
        return f"""### Root Cause

{scenario.description}

### Attack Prerequisites

{prereqs}

### Attack Sequence

{steps_text}

### Technical Analysis

**Complexity:** {scenario.complexity}
**Required Capital:** {scenario.required_capital / 10**18:.4f} ETH
**Expected Profit:** {scenario.expected_profit / 10**18:.4f} ETH
**Verification Confidence:** {exploit.confidence * 100:.1f}%

### Verification Method

This vulnerability was verified using {exploit.verification_method} analysis.
"""
    
    def _generate_impact(self, exploit: VerifiedExploit) -> str:
        """Generate impact description."""
        
        profit = exploit.scenario.expected_profit / 10**18
        profit_usd = profit * 2000  # Assuming $2000/ETH
        
        severity = self._classify_severity(exploit)
        
        if severity == "Critical":
            impact_desc = (
                f"**Direct theft of funds:** An attacker can extract approximately "
                f"${profit_usd:,.0f} worth of assets from the protocol in a single transaction."
            )
        elif severity == "High":
            impact_desc = (
                f"**Significant fund loss:** This vulnerability could result in "
                f"losses of approximately ${profit_usd:,.0f} if exploited."
            )
        elif severity == "Medium":
            impact_desc = (
                f"**Limited fund loss:** Exploitation could result in losses of "
                f"approximately ${profit_usd:,.0f} under specific conditions."
            )
        else:
            impact_desc = (
                f"**Minor impact:** This vulnerability has limited financial impact "
                f"(~${profit_usd:,.0f}) but represents a deviation from expected behavior."
            )
        
        return f"""{impact_desc}

### Funds at Risk

- **Immediate Risk:** {profit:.4f} ETH (~${profit_usd:,.0f})
- **Repeat Attack Possible:** {"Yes" if "flash" in exploit.scenario.name.lower() else "Depends on available capital"}
- **Affected Users:** All depositors/users of the protocol

### Attack Cost

- **Required Capital:** {exploit.scenario.required_capital / 10**18:.4f} ETH
- **Gas Costs:** Estimated 500,000 - 2,000,000 gas
- **Net Profit:** {(profit - exploit.scenario.required_capital / 10**18):.4f} ETH
"""
    
    def _generate_recommendation(self, exploit: VerifiedExploit) -> str:
        """Generate remediation recommendations."""
        
        scenario = exploit.scenario
        name = scenario.name.lower()
        
        # Pattern-based recommendations
        if "reentrancy" in name:
            return """### Recommended Fix

1. **Implement Checks-Effects-Interactions Pattern**
   - Update all state variables before making external calls
   - Move balance updates before token transfers

2. **Add ReentrancyGuard**
   ```solidity
   import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
   
   contract YourContract is ReentrancyGuard {
       function vulnerableFunction() external nonReentrant {
           // ...
       }
   }
   ```

3. **Use Pull Over Push Pattern**
   - Let users withdraw rather than pushing funds to them
"""
        
        elif "oracle" in name:
            return """### Recommended Fix

1. **Use TWAP Instead of Spot Price**
   - Implement time-weighted average price over 30+ minutes
   - Example: Uniswap V3 TWAP oracle

2. **Add Price Bounds**
   ```solidity
   require(
       currentPrice >= lastPrice * 90 / 100 &&
       currentPrice <= lastPrice * 110 / 100,
       "Price deviation too large"
   );
   ```

3. **Use Multiple Oracle Sources**
   - Aggregate prices from Chainlink, Uniswap, etc.
   - Reject if sources disagree by more than threshold
"""
        
        elif "first depositor" in name or "inflation" in name:
            return """### Recommended Fix

1. **Mint Dead Shares on Deployment**
   ```solidity
   constructor() {
       // Mint 1000 shares to zero address
       _mint(address(0), 1000);
   }
   ```

2. **Use Virtual Shares/Assets**
   ```solidity
   function totalAssets() public view returns (uint256) {
       return _totalAssets + 1;  // Virtual offset
   }
   
   function totalSupply() public view returns (uint256) {
       return _totalSupply + 1e3;  // Virtual shares
   }
   ```

3. **Require Minimum First Deposit**
   ```solidity
   require(
       totalSupply() > 0 || assets >= MIN_FIRST_DEPOSIT,
       "First deposit too small"
   );
   ```
"""
        
        elif "precision" in name or "rounding" in name:
            return """### Recommended Fix

1. **Use Higher Precision Internally**
   - Scale by 1e18 for internal calculations
   - Round in protocol's favor

2. **Add Minimum Amounts**
   ```solidity
   require(amount >= MIN_AMOUNT, "Amount too small");
   ```

3. **Round Against User for Conversions**
   ```solidity
   // When user deposits
   shares = assets.mulDivDown(totalSupply, totalAssets);
   
   // When user withdraws  
   assets = shares.mulDivDown(totalAssets, totalSupply);
   ```
"""
        
        # Generic recommendation
        return f"""### Recommended Fix

1. **Review Assumption:** `{scenario.description[:100]}`
   - Add explicit validation for this assumption
   - Consider edge cases and malicious inputs

2. **Add Input Validation**
   - Validate all user inputs
   - Check for zero/extreme values

3. **Consider Access Controls**
   - Review who can call each function
   - Add appropriate modifiers

4. **Test Thoroughly**
   - Add fuzzing tests for edge cases
   - Test with adversarial inputs
"""
    
    def export_json(self, exploit: VerifiedExploit) -> str:
        """Export exploit as JSON for archival."""
        
        return json.dumps({
            "name": exploit.scenario.name,
            "description": exploit.scenario.description,
            "severity": self._classify_severity(exploit),
            "verification_method": exploit.verification_method,
            "confidence": exploit.confidence,
            "expected_profit_wei": exploit.scenario.expected_profit,
            "required_capital_wei": exploit.scenario.required_capital,
            "complexity": exploit.scenario.complexity,
            "steps": [s.action for s in exploit.scenario.steps],
            "timestamp": datetime.now().isoformat(),
            "generator": "ORACLE v0.1.0",
        }, indent=2)
