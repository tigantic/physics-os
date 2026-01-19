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
        fork_verify: bool = False,
        min_profit_eth: float = 0.0,
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
            fork_verify: Use mainnet fork for verification (requires ETH_RPC_URL)
            min_profit_eth: Minimum profit threshold in ETH
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
            mode = "FORK" if fork_verify else "SYMBOLIC"
            print(f"\n[Phase 6] Verifying Exploits ({mode})...")
        
        verified_exploits: list[VerifiedExploit] = []
        
        if fork_verify and self.eth_rpc:
            # Use MainnetVerifier for real fork-based verification
            from tensornet.oracle.execution import MainnetVerifier, AnvilFork, ForkConfig
            
            if verbose:
                print(f"  🔗 Starting fork: {self.eth_rpc[:50]}...")
            
            # Start ONE fork for all scenarios
            config = ForkConfig(
                rpc_url=self.eth_rpc,
                chain_id=self._get_chain_id(chain),
            )
            anvil = AnvilFork(config)
            
            if anvil.start():
                if verbose:
                    print(f"  ✅ Fork ready at block {anvil.block_number}")
                
                fork_verifier = MainnetVerifier(
                    rpc_url=self.eth_rpc,
                    chain_id=self._get_chain_id(chain),
                )
                fork_verifier.anvil = anvil  # Reuse the fork
                
                for scenario in scenarios:
                    if verbose:
                        print(f"  Testing: {scenario.name}...")
                    
                    # Take snapshot before each test
                    snapshot = anvil.snapshot()
                    
                    result = fork_verifier.verify_scenario_with_anvil(
                        scenario=scenario,
                        source=source,
                        anvil=anvil,
                    )
                    
                    # Revert to clean state
                    anvil.revert(snapshot)
                    
                    if result and result.profit_wei > 0:
                        profit_eth = result.profit_wei / 10**18
                        if profit_eth >= min_profit_eth:
                            exploit = VerifiedExploit(
                                scenario=scenario,
                                verification_method="mainnet_fork",
                                proof=None,
                                confidence=0.95,
                                foundry_test=result.foundry_test if hasattr(result, 'foundry_test') else "",
                                fork_profit_wei=result.profit_wei,
                                fork_block=anvil.block_number,
                            )
                            verified_exploits.append(exploit)
                            if verbose:
                                print(f"    ✅ VERIFIED: {profit_eth:.4f} ETH profit")
                        else:
                            if verbose:
                                print(f"    ⚠️  Below threshold: {profit_eth:.4f} ETH")
                    else:
                        if verbose:
                            print(f"    ❌ Not exploitable")
                
                anvil.stop()
            else:
                if verbose:
                    print(f"  ❌ Could not start fork - falling back to symbolic")
                # Fall through to symbolic verification
                fork_verify = False
        
        if not fork_verify or not self.eth_rpc:
            # Use symbolic verification (faster but less accurate)
            from tensornet.oracle.execution import ExploitVerifier
            verifier = ExploitVerifier()
            
            for scenario in scenarios:
                exploit = verifier.verify(scenario)
                if exploit:
                    verified_exploits.append(exploit)
                    if verbose:
                        profit = scenario.expected_profit / 10**18
                        print(f"  ✓ VERIFIED: {scenario.name}")
                        print(f"    Profit: ~{profit:.2f} ETH (estimated)")
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
        from tensornet.oracle.execution import EtherscanClient
        
        chain_ids = {
            "ethereum": 1,
            "mainnet": 1,
            "optimism": 10,
            "arbitrum": 42161,
            "base": 8453,
            "polygon": 137,
        }
        chain_id = chain_ids.get(chain.lower(), 1)
        
        client = EtherscanClient(chain_id=chain_id)
        source = client.get_source_code(address)
        
        if not source:
            raise ValueError(
                f"Could not fetch source for {address} on {chain}. "
                f"Set ETHERSCAN_API_KEY environment variable."
            )
        
        return source
    
    def hunt_address(
        self,
        address: str,
        chain: str = "ethereum",
        fork_verify: bool = True,
        min_profit_eth: float = 0.1,
        verbose: bool = True,
    ) -> HuntResult:
        """
        Hunt vulnerabilities at a mainnet address with fork verification.
        
        This is the production mode - actually finds money.
        
        Args:
            address: Contract address (0x...)
            chain: Chain name (ethereum, arbitrum, optimism, base, polygon)
            fork_verify: Whether to verify exploits on mainnet fork
            min_profit_eth: Minimum profit threshold to report
            verbose: Print progress
            
        Returns:
            HuntResult with fork-verified exploits
        """
        if not self.eth_rpc:
            raise ValueError(
                "ETH_RPC_URL required for mainnet hunting. "
                "Set environment variable or pass eth_rpc to ORACLE()"
            )
        
        start_time = time.time()
        
        if verbose:
            print("=" * 70)
            print("ORACLE: Mainnet Bounty Hunt")
            print("=" * 70)
            print(f"\n  Target: {address}")
            print(f"  Chain: {chain}")
            print(f"  Fork Verify: {fork_verify}")
            print(f"  Min Profit: {min_profit_eth} ETH")
        
        # Fetch source
        if verbose:
            print("\n[1/7] Fetching verified source...")
        
        source = self._fetch_source(address, chain)
        if verbose:
            print(f"  ✓ Source fetched: {len(source):,} chars")
        
        # Parse
        if verbose:
            print("\n[2/7] Parsing contract...")
        
        contracts = self.parser.parse(source)
        if not contracts:
            raise ValueError("No contracts found in source")
        
        contract = contracts[0]
        if verbose:
            print(f"  ✓ {contract.name}: {len(contract.functions)} functions")
        
        # Analyze
        for func in contract.functions:
            self.parser.analyze_function(func, contract)
        
        # Semantic analysis
        if verbose:
            print("\n[3/7] Semantic analysis...")
        
        intent = self.intent_analyzer.analyze(contract, source)
        if verbose:
            print(f"  ✓ Protocol type: {intent.protocol_type}")
        
        # Extract assumptions
        if verbose:
            print("\n[4/7] Extracting assumptions...")
        
        assumptions: list[Assumption] = []
        assumptions.extend(self.explicit_extractor.extract(contract))
        assumptions.extend(self.implicit_extractor.extract(contract, intent))
        assumptions.extend(self.economic_extractor.extract(contract, intent))
        
        if verbose:
            print(f"  ✓ Found {len(assumptions)} assumptions")
        
        # Challenge assumptions
        if verbose:
            print("\n[5/7] Challenging assumptions...")
        
        from tensornet.oracle.challenger import AssumptionChallenger
        challenger = AssumptionChallenger(contract)
        challenges = challenger.challenge(assumptions, contract, intent)
        
        if verbose:
            print(f"  ✓ Generated {len(challenges)} challenges")
        
        # Generate scenarios
        if verbose:
            print("\n[6/7] Generating attack scenarios...")
        
        scenarios = self.scenario_generator.generate(contract, challenges, intent)
        if verbose:
            print(f"  ✓ Generated {len(scenarios)} scenarios")
        
        # Fork verification - THE KEY STEP
        verified_exploits: list[VerifiedExploit] = []
        
        if fork_verify and scenarios:
            if verbose:
                print("\n[7/7] Fork verification (MAINNET)...")
                print(f"  RPC: {self.eth_rpc[:40]}...")
            
            from tensornet.oracle.execution import MainnetVerifier
            
            verifier = MainnetVerifier(
                rpc_url=self.eth_rpc,
                chain_id=self._get_chain_id(chain),
            )
            
            verified_exploits = verifier.verify_all(
                scenarios=scenarios,
                contract=contract,
                target_address=address,
                verbose=verbose,
            )
            
            # Filter by minimum profit
            min_profit_wei = int(min_profit_eth * 10**18)
            verified_exploits = [
                e for e in verified_exploits
                if e.verification.proof.profit_wei >= min_profit_wei
            ]
        else:
            # Fallback to interval verification
            if verbose:
                print("\n[7/7] Interval verification (simulated)...")
            
            from tensornet.oracle.execution import ExploitVerifier
            verifier = ExploitVerifier()
            
            for scenario in scenarios:
                exploit = verifier.verify(scenario)
                if exploit:
                    verified_exploits.append(exploit)
        
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
            print("\n" + "=" * 70)
            print("HUNT COMPLETE")
            print("=" * 70)
            print(f"  Time: {elapsed:.1f}s")
            print(f"  Assumptions: {len(assumptions)}")
            print(f"  Challenges: {len(challenges)}")
            print(f"  Scenarios: {len(scenarios)}")
            print(f"  Fork-Verified Exploits: {len(verified_exploits)}")
            
            if verified_exploits:
                total_profit = sum(
                    e.verification.proof.profit_wei 
                    for e in verified_exploits
                    if hasattr(e.verification.proof, 'profit_wei')
                )
                print(f"\n  💰 VERIFIED PROFIT: {total_profit / 10**18:.4f} ETH")
                
                for e in verified_exploits:
                    profit = getattr(e.verification.proof, 'profit_wei', 0) / 10**18
                    print(f"    - {e.scenario.name}: {profit:.4f} ETH")
            else:
                print(f"\n  ✓ No profitable exploits found - contract appears secure")
        
        return result
    
    def _get_chain_id(self, chain: str) -> int:
        """Get chain ID from chain name."""
        chain_ids = {
            "ethereum": 1,
            "mainnet": 1,
            "optimism": 10,
            "arbitrum": 42161,
            "base": 8453,
            "polygon": 137,
        }
        return chain_ids.get(chain.lower(), 1)
    
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
