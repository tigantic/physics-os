"""
Extract economic assumptions about incentives and rational actors.

Economic assumptions are often the most subtle:
- "Liquidators will liquidate"
- "Arbitrageurs keep prices aligned"
- "Governance acts honestly"

When these fail, protocols die (see: Mango Markets, Cream Finance).
"""

from __future__ import annotations

from ontic.infra.oracle.core.types import (
    Assumption,
    AssumptionType,
    Contract,
    IntentAnalysis,
)


class EconomicExtractor:
    """
    Identify assumptions about economic behavior.
    
    These are the assumptions that formal verification can't catch -
    they depend on game theory, not code.
    """
    
    # Economic patterns by protocol type
    LENDING_PATTERNS = [
        ("liquidation_incentive", 
         "Liquidators will liquidate underwater positions when profitable",
         "∀ pos: health(pos) < 1 → ∃ liquidator: liquidate(pos) within N blocks"),
        ("liquidation_timing",
         "Liquidation happens before position becomes insolvent",
         "∀ pos: collateral(pos) > debt(pos) when liquidated"),
        ("interest_rate_equilibrium",
         "Interest rates balance supply and demand",
         "utilization → target_utilization as t → ∞"),
        ("no_bad_debt_spiral",
         "Bad debt doesn't cascade through the system",
         "bad_debt(t) bounded for all t"),
    ]
    
    DEX_PATTERNS = [
        ("arbitrage_alignment",
         "Arbitrageurs keep pool prices aligned with external markets",
         "|pool_price - market_price| < arb_cost + slippage"),
        ("lp_profitability",
         "Liquidity provision is profitable (fees > impermanent loss)",
         "E[LP_returns] > 0 for typical volatility"),
        ("no_manipulation",
         "Price cannot be profitably manipulated within a block",
         "∀ manipulator: profit(manipulation) ≤ 0"),
    ]
    
    VAULT_PATTERNS = [
        ("strategy_profitability",
         "Underlying strategy generates positive returns",
         "E[strategy_return] > 0"),
        ("no_griefing_deposits",
         "Deposits cannot be used to grief other depositors",
         "∀ deposit: impact_on_others(deposit) < ε"),
        ("fair_withdrawal",
         "Withdrawals receive fair share of assets",
         "withdrawal_value ≈ deposit_value * (1 + returns)"),
    ]
    
    GOVERNANCE_PATTERNS = [
        ("honest_governance",
         "Governance actors act in protocol's best interest",
         "∀ proposal: passes(proposal) → beneficial(proposal)"),
        ("no_flash_governance",
         "Flash loans cannot be used to pass governance proposals",
         "voting_power(flash_loaned_tokens) = 0 for current proposal"),
        ("timelock_protection",
         "Timelock provides sufficient reaction time",
         "timelock_duration > typical_user_reaction_time"),
    ]
    
    BRIDGE_PATTERNS = [
        ("validator_honesty",
         "Bridge validators don't collude to steal funds",
         "∀ attestation: validators(attestation) ≥ threshold honest"),
        ("finality_respect",
         "Source chain transactions are final before bridge processes",
         "∀ tx: finality(tx) confirmed before bridge_process(tx)"),
    ]
    
    MEV_PATTERNS = [
        ("ordering_irrelevant",
         "Transaction ordering doesn't affect outcome significantly",
         "∀ orderings O1, O2: |result(O1) - result(O2)| < ε"),
        ("no_sandwich",
         "Users are not sandwiched on trades",
         "∀ user_tx: no profitable sandwich exists"),
        ("no_front_running",
         "Time-sensitive operations cannot be front-run",
         "∀ tx: front_run_profit(tx) ≤ 0"),
    ]
    
    def __init__(self):
        """Initialize the economic extractor."""
        self._assumption_id = 0
    
    def extract(self, contract: Contract, intent: IntentAnalysis) -> list[Assumption]:
        """
        Identify economic assumptions based on protocol type.
        
        Args:
            contract: Parsed contract
            intent: Semantic intent analysis
            
        Returns:
            List of economic assumptions
        """
        self._assumption_id = 0
        assumptions = []
        
        # Protocol-specific assumptions
        protocol_type = intent.protocol_type.lower()
        
        if protocol_type == "lending":
            assumptions.extend(self._extract_patterns(self.LENDING_PATTERNS))
        elif protocol_type == "dex":
            assumptions.extend(self._extract_patterns(self.DEX_PATTERNS))
        elif protocol_type == "vault":
            assumptions.extend(self._extract_patterns(self.VAULT_PATTERNS))
        elif protocol_type == "governance":
            assumptions.extend(self._extract_patterns(self.GOVERNANCE_PATTERNS))
        elif protocol_type == "bridge":
            assumptions.extend(self._extract_patterns(self.BRIDGE_PATTERNS))
        
        # Universal MEV assumptions (apply to most DeFi)
        if protocol_type in ("lending", "dex", "vault"):
            assumptions.extend(self._extract_patterns(self.MEV_PATTERNS))
        
        # Contract-specific analysis
        assumptions.extend(self._analyze_specific(contract, intent))
        
        return assumptions
    
    def _next_id(self) -> str:
        """Generate next assumption ID."""
        self._assumption_id += 1
        return f"EC{self._assumption_id:03d}"
    
    def _extract_patterns(self, patterns: list[tuple[str, str, str]]) -> list[Assumption]:
        """Convert pattern tuples to Assumption objects."""
        return [
            Assumption(
                id=self._next_id(),
                type=AssumptionType.ECONOMIC,
                source="global",
                statement=statement,
                formal=formal,
                confidence=0.7,  # Economic assumptions are inherently uncertain
            )
            for _, statement, formal in patterns
        ]
    
    def _analyze_specific(self, contract: Contract, 
                          intent: IntentAnalysis) -> list[Assumption]:
        """Analyze contract-specific economic patterns."""
        assumptions = []
        
        # Check for fee mechanisms
        fee_functions = [f for f in contract.functions 
                        if "fee" in f.name.lower()]
        if fee_functions:
            assumptions.append(Assumption(
                id=self._next_id(),
                type=AssumptionType.ECONOMIC,
                source="global",
                statement="Fee structure doesn't incentivize value extraction",
                formal="∀ user: fee(user) ≤ value_provided(user)",
                confidence=0.6,
            ))
        
        # Check for rewards/incentives
        reward_functions = [f for f in contract.functions 
                          if any(kw in f.name.lower() 
                                for kw in ["reward", "incentive", "bonus", "yield"])]
        if reward_functions:
            assumptions.append(Assumption(
                id=self._next_id(),
                type=AssumptionType.ECONOMIC,
                source="global",
                statement="Reward emissions don't exceed protocol revenue",
                formal="∫ rewards(t) dt ≤ ∫ revenue(t) dt over protocol lifetime",
                confidence=0.5,
            ))
        
        # Check for oracle dependencies
        oracle_vars = [sv for sv in contract.state_variables 
                      if "oracle" in sv.name.lower() or "price" in sv.name.lower()]
        if oracle_vars:
            assumptions.append(Assumption(
                id=self._next_id(),
                type=AssumptionType.ECONOMIC,
                source="global",
                statement="Oracle price cannot be profitably manipulated",
                formal="cost(manipulation) > profit(manipulation) for all strategies",
                confidence=0.65,
            ))
        
        # Check for collateral/debt mechanisms
        has_collateral = any("collateral" in sv.name.lower() 
                            for sv in contract.state_variables)
        has_debt = any("debt" in sv.name.lower() or "borrow" in sv.name.lower()
                      for sv in contract.state_variables)
        
        if has_collateral and has_debt:
            assumptions.append(Assumption(
                id=self._next_id(),
                type=AssumptionType.ECONOMIC,
                source="global",
                statement="Collateral value remains above debt value under normal conditions",
                formal="P(collateral_value < debt_value) < ε for ε small",
                confidence=0.6,
            ))
        
        # Check for time-locked operations
        if any("lock" in f.name.lower() or "timelock" in f.name.lower() 
               for f in contract.functions):
            assumptions.append(Assumption(
                id=self._next_id(),
                type=AssumptionType.ECONOMIC,
                source="global",
                statement="Lock period is sufficient to prevent timing attacks",
                formal="lock_duration > adversary_preparation_time",
                confidence=0.7,
            ))
        
        return assumptions
