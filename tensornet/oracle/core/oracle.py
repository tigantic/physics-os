"""
ORACLE: Offensive Reasoning and Assumption-Challenging Logic Engine

The main orchestrator that ties all phases together.

Usage:
    oracle = ORACLE(anthropic_key="sk-...")
    results = oracle.hunt(source=contract_source)
    
    for exploit in results.verified_exploits:
        report = oracle.generate_report(exploit)
        print(report.to_immunefi_markdown())
"""

from __future__ import annotations

import os
import time
from typing import Optional

from tensornet.oracle.core.types import (
    Assumption,
    Contract,
    HuntResult,
    IntentAnalysis,
    Report,
    VerifiedExploit,
)


class ORACLE:
    """
    ORACLE: Offensive Reasoning and Assumption-Challenging Logic Engine.
    
    Automated smart contract vulnerability hunting through:
    1. Semantic understanding (intent analysis)
    2. Assumption extraction (explicit + implicit + economic)
    3. Assumption challenging (reachability + impact)
    4. Scenario generation (pattern + LLM)
    5. Verification (interval + concrete)
    6. Report synthesis
    
    Core insight: Bugs are ASSUMPTION FAILURES, not pattern matches.
    """
    
    def __init__(
        self,
        anthropic_key: Optional[str] = None,
        eth_rpc: Optional[str] = None,
    ):
        """
        Initialize ORACLE.
        
        Args:
            anthropic_key: Anthropic API key (or set ANTHROPIC_API_KEY env)
            eth_rpc: Ethereum RPC URL for mainnet verification
        """
        self.anthropic_key = anthropic_key or os.environ.get("ANTHROPIC_API_KEY")
        self.eth_rpc = eth_rpc or os.environ.get("ETH_RPC_URL")
        
        # Lazy-load components
        self._parser = None
        self._intent_analyzer = None
        self._explicit_extractor = None
        self._implicit_extractor = None
        self._economic_extractor = None
        self._challenger = None
        self._scenario_generator = None
        self._verifier = None
        self._report_generator = None
    
    @property
    def parser(self):
        """Lazy-load Solidity parser."""
        if self._parser is None:
            from tensornet.oracle.parsing import SolidityParser
            self._parser = SolidityParser()
        return self._parser
    
    @property
    def intent_analyzer(self):
        """Lazy-load intent analyzer."""
        if self._intent_analyzer is None:
            from tensornet.oracle.semantic import IntentAnalyzer
            self._intent_analyzer = IntentAnalyzer(api_key=self.anthropic_key)
        return self._intent_analyzer
    
    @property
    def explicit_extractor(self):
        """Lazy-load explicit assumption extractor."""
        if self._explicit_extractor is None:
            from tensornet.oracle.assumptions import ExplicitExtractor
            self._explicit_extractor = ExplicitExtractor()
        return self._explicit_extractor
    
    @property
    def implicit_extractor(self):
        """Lazy-load implicit assumption extractor."""
        if self._implicit_extractor is None:
            from tensornet.oracle.assumptions import ImplicitExtractor
            self._implicit_extractor = ImplicitExtractor(api_key=self.anthropic_key)
        return self._implicit_extractor
    
    @property
    def economic_extractor(self):
        """Lazy-load economic assumption extractor."""
        if self._economic_extractor is None:
            from tensornet.oracle.assumptions import EconomicExtractor
            self._economic_extractor = EconomicExtractor()
        return self._economic_extractor
    
    @property
    def scenario_generator(self):
        """Lazy-load scenario generator."""
        if self._scenario_generator is None:
            from tensornet.oracle.scenarios import ScenarioGenerator
            self._scenario_generator = ScenarioGenerator(api_key=self.anthropic_key)
        return self._scenario_generator
    
    @property
    def report_generator(self):
        """Lazy-load report generator."""
        if self._report_generator is None:
            from tensornet.oracle.reporting import ReportGenerator
            self._report_generator = ReportGenerator(api_key=self.anthropic_key)
        return self._report_generator
    
    def hunt(
        self,
        source: Optional[str] = None,
        file_path: Optional[str] = None,
        address: Optional[str] = None,
        chain: str = "ethereum",
        min_confidence: float = 0.5,
        verbose: bool = True,
    ) -> HuntResult:
        """
        Hunt for vulnerabilities in a contract.
        
        Args:
            source: Solidity source code
            file_path: Path to Solidity file
            address: Contract address (requires eth_rpc)
            chain: Chain name (default: ethereum)
            min_confidence: Minimum confidence threshold
            verbose: Print progress
            
        Returns:
            HuntResult with all findings
        """
        start_time = time.time()
        
        # Get source code
        if file_path:
            with open(file_path, "r") as f:
                source = f.read()
        elif address:
            source = self._fetch_source(address, chain)
        
        if not source:
            raise ValueError("Must provide source, file_path, or address")
        
        if verbose:
            print("=" * 60)
            print("ORACLE: Offensive Reasoning and Assumption-Challenging Logic Engine")
            print("=" * 60)
        
        # Phase 1: Parse
        if verbose:
            print("\n[Phase 1] Parsing Solidity...")
        
        contracts = self.parser.parse(source, file_path)
        if not contracts:
            raise ValueError("No contracts found in source")
        
        contract = contracts[0]  # Analyze first contract
        if verbose:
            print(f"  Found: {contract.name}")
            print(f"  Functions: {len(contract.functions)}")
            print(f"  State Variables: {len(contract.state_variables)}")
        
        # Analyze functions
        for func in contract.functions:
            self.parser.analyze_function(func, contract)
        
        # Phase 2: Semantic Analysis
        if verbose:
            print("\n[Phase 2] Analyzing Intent...")
        
        intent = self.intent_analyzer.analyze(contract, source)
        if verbose:
            print(f"  Protocol Type: {intent.protocol_type}")
            print(f"  Actors: {len(intent.actors)}")
            print(f"  Trust Assumptions: {len(intent.trust_assumptions)}")
        
        # Phase 3: Extract Assumptions
        if verbose:
            print("\n[Phase 3] Extracting Assumptions...")
        
        assumptions: list[Assumption] = []
        
        # Explicit (from require/assert)
        explicit = self.explicit_extractor.extract(contract)
        assumptions.extend(explicit)
        if verbose:
            print(f"  Explicit: {len(explicit)}")
        
        # Implicit (pattern + LLM)
        implicit = self.implicit_extractor.extract(contract, intent)
        assumptions.extend(implicit)
        if verbose:
            print(f"  Implicit: {len(implicit)}")
        
        # Economic
        economic = self.economic_extractor.extract(contract, intent)
        assumptions.extend(economic)
        if verbose:
            print(f"  Economic: {len(economic)}")
        
        if verbose:
            print(f"  Total: {len(assumptions)}")
        
        # Phase 4: Challenge Assumptions
        if verbose:
            print("\n[Phase 4] Challenging Assumptions...")
        
        from tensornet.oracle.challenger import AssumptionChallenger
        challenger = AssumptionChallenger(contract, intent, self.anthropic_key)
        challenges = challenger.challenge_all(assumptions, min_confidence)
        
        if verbose:
            print(f"  Reachable violations: {len(challenges)}")
            for c in challenges[:5]:
                print(f"    - [{c.impact.value}] {c.assumption.statement[:60]}...")
        
        # Phase 5: Generate Scenarios
        if verbose:
            print("\n[Phase 5] Generating Attack Scenarios...")
        
        scenarios = self.scenario_generator.generate(contract, challenges, intent)
        
        if verbose:
            print(f"  Scenarios generated: {len(scenarios)}")
            for s in scenarios[:3]:
                print(f"    - {s.name} ({s.complexity})")
        
        # Phase 6: Verify
        if verbose:
            print("\n[Phase 6] Verifying Exploits...")
        
        from tensornet.oracle.execution import ExploitVerifier
        verifier = ExploitVerifier()
        
        verified_exploits: list[VerifiedExploit] = []
        for scenario in scenarios:
            exploit = verifier.verify(scenario)
            if exploit:
                verified_exploits.append(exploit)
                if verbose:
                    profit = scenario.expected_profit / 10**18
                    print(f"  ✓ VERIFIED: {scenario.name}")
                    print(f"    Profit: ~{profit:.2f} ETH")
                    print(f"    Confidence: {exploit.confidence * 100:.0f}%")
        
        # Build result
        elapsed = time.time() - start_time
        
        result = HuntResult(
            contract=contract,
            intent=intent,
            assumptions=assumptions,
            challenges=challenges,
            scenarios=scenarios,
            verified_exploits=verified_exploits,
            hunt_time_seconds=elapsed,
            contract_address=address,
            chain=chain,
        )
        
        if verbose:
            print("\n" + "=" * 60)
            print("HUNT COMPLETE")
            print("=" * 60)
            print(f"  Time: {elapsed:.1f}s")
            print(f"  Assumptions: {len(assumptions)}")
            print(f"  Challenges: {len(challenges)}")
            print(f"  Scenarios: {len(scenarios)}")
            print(f"  Verified Exploits: {len(verified_exploits)}")
            
            if verified_exploits:
                total_profit = sum(e.scenario.expected_profit for e in verified_exploits)
                print(f"\n  💰 TOTAL POTENTIAL PROFIT: {total_profit / 10**18:.2f} ETH")
        
        return result
    
    def generate_report(self, exploit: VerifiedExploit) -> Report:
        """
        Generate submission-ready report for a verified exploit.
        
        Args:
            exploit: Verified exploit
            
        Returns:
            Report object
        """
        return self.report_generator.generate(exploit)
    
    def _fetch_source(self, address: str, chain: str) -> str:
        """Fetch contract source from block explorer."""
        # Placeholder - implement with Etherscan API
        raise NotImplementedError(
            f"Source fetching not implemented. "
            f"Please provide source code directly or use file_path. "
            f"Address: {address}, Chain: {chain}"
        )
    
    def hunt_file(self, path: str, **kwargs) -> HuntResult:
        """Convenience method to hunt a file."""
        return self.hunt(file_path=path, **kwargs)
    
    def quick_scan(self, source: str) -> dict:
        """
        Quick scan without full verification.
        
        Returns summary of potential issues.
        """
        contracts = self.parser.parse(source)
        if not contracts:
            return {"error": "No contracts found"}
        
        contract = contracts[0]
        intent = self.intent_analyzer.analyze(contract, source)
        
        # Just extract assumptions
        assumptions = []
        assumptions.extend(self.explicit_extractor.extract(contract))
        assumptions.extend(self.implicit_extractor.extract(contract, intent))
        assumptions.extend(self.economic_extractor.extract(contract, intent))
        
        return {
            "contract": contract.name,
            "type": intent.protocol_type,
            "functions": len(contract.functions),
            "assumptions": len(assumptions),
            "high_risk": [
                a.statement for a in assumptions 
                if a.confidence > 0.7 and "reentran" in a.statement.lower()
                or "oracle" in a.statement.lower()
            ][:5],
        }
