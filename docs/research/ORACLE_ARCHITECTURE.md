# ORACLE: Offensive Reasoning and Assumption-Challenging Logic Engine

> **Version**: 1.0 | **Date**: January 19, 2026 | **Author**: Brad Adams / Tigantic Holdings
> 
> **Purpose**: The ultimate smart contract vulnerability hunter that automates and EXCEEDS what the best human auditors do.

---

## Table of Contents

1. [The Core Insight](#the-core-insight)
2. [Why This Approach](#why-this-approach)
3. [Architecture Overview](#architecture-overview)
4. [Phase 1: Semantic Extraction](#phase-1-semantic-extraction)
5. [Phase 2: Assumption Extraction](#phase-2-assumption-extraction)
6. [Phase 3: Assumption Challenging](#phase-3-assumption-challenging)
7. [Phase 4: Adversarial Scenario Generation](#phase-4-adversarial-scenario-generation)
8. [Phase 5: Verification](#phase-5-verification)
9. [Phase 6: Synthesis](#phase-6-synthesis)
10. [Custom Stack Decisions](#custom-stack-decisions)
11. [Implementation Plan](#implementation-plan)
12. [Timeline](#timeline)
13. [Success Metrics](#success-metrics)
14. [Integration with The Physics OS](#integration-with-hypertensor)

---

## The Core Insight

### What Manual Reviewers Do That Tools Don't

Every major bug bounty payout ($500K+) was found by human auditors doing ONE thing:

> **"Wait... what ASSUMES this can't happen? And can I make it happen?"**

| Human Expert Capability | Why Existing Tools Fail |
|-------------------------|------------------------|
| Understand INTENT from code/comments | Tools see syntax, not semantics |
| "This SHOULD do X, but actually does Y" | Tools don't know "supposed to" |
| Cross-function reasoning | Tools analyze functions in isolation |
| Economic intuition | Tools don't model incentives |
| "What if someone did THIS?" | Tools don't generate adversarial scenarios |
| Assumption identification | Tools don't extract implicit assumptions |

### The Reframe

**Old framing (SINGULARITY):** "Contracts are physics. Find where conservation laws break."

**New framing (ORACLE):** "Contracts are assumptions. Challenge every one."

Bugs aren't physics failures. Bugs are **ASSUMPTION FAILURES**.

The code ASSUMES something that isn't always true. The attacker makes the assumption false.

---

## Why This Approach

### Historical Evidence

| Bug | Bounty/Loss | How Found |
|-----|-------------|-----------|
| Wormhole signature bypass | $10M bounty | Manual review (Jump) |
| Optimism bridge | $2M bounty | Manual review |
| Euler v1 hack | $200M stolen | Attacker manual analysis |
| Arbitrum sequencer | $400K bounty | Manual review |

**Pattern:** Human insight beats automated tools on the biggest bugs.

### Why Now Is Different

| Barrier | Why It's Falling |
|---------|------------------|
| LLMs couldn't understand code | Claude/GPT-4 can READ and REASON about code |
| LLMs hallucinate | Formal verification catches hallucinations |
| Formal methods are slow | LLM GUIDES formal methods (smaller search space) |
| Need domain expertise | LLM HAS domain expertise (trained on all audits) |

### The Combination

```
LLM: "I think there's a vulnerability here because..."
     ↓
Formal Methods: "Let me CHECK if that's reachable..."
     ↓
LLM: "If it's reachable, the attack would be..."
     ↓
Concrete Execution: "Let me VERIFY on a fork..."
     ↓
Output: PROVEN EXPLOIT (not hallucination)
```

LLM provides **semantic understanding** and **creative scenario generation**.
Formal methods provide **rigorous verification**.

Neither alone is sufficient. Together they exceed human auditors.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                              ORACLE                                     │
│                                                                         │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐ │
│  │   PHASE 1   │   │   PHASE 2   │   │   PHASE 3   │   │   PHASE 4   │ │
│  │  Semantic   │ → │ Assumption  │ → │ Assumption  │ → │  Scenario   │ │
│  │ Extraction  │   │ Extraction  │   │ Challenging │   │ Generation  │ │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘ │
│         │                                                     │         │
│         │               ┌─────────────┐                       │         │
│         │               │   PHASE 5   │                       │         │
│         └──────────────→│ Verification│←──────────────────────┘         │
│                         └─────────────┘                                 │
│                                │                                        │
│                         ┌─────────────┐                                 │
│                         │   PHASE 6   │                                 │
│                         │  Synthesis  │                                 │
│                         └─────────────┘                                 │
│                                │                                        │
│                         ┌─────────────┐                                 │
│                         │   OUTPUT    │                                 │
│                         │  Immunefi   │                                 │
│                         │   Report    │                                 │
│                         └─────────────┘                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Semantic Extraction

### Purpose
Read code like a human auditor. Understand what it's TRYING to do.

### Input
- Solidity/Vyper source code
- NatSpec comments
- Variable/function names

### Output
```python
@dataclass
class ContractSpecification:
    intent: str                          # "This is a lending protocol that..."
    actors: List[Actor]                  # user, admin, liquidator, oracle
    assets: List[Asset]                  # tokens, collateral types
    invariants_stated: List[str]         # from comments/docs
    functions: List[FunctionSpec]        # with semantic descriptions
    state_variables: List[StateVar]      # with purpose annotations
```

### Implementation

#### 1.1 Source Parsing (Custom)

**File:** `oracle/parsing/solidity_parser.py`

```python
"""
Custom Solidity parser using tree-sitter.
Outputs EXACTLY what ORACLE needs, nothing more.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import tree_sitter_solidity as ts_sol

@dataclass
class Function:
    name: str
    visibility: str                      # public, external, internal, private
    modifiers: List[str]                 # onlyOwner, nonReentrant, etc.
    params: List[Param]
    returns: List[Param]
    state_mutability: str                # view, pure, payable, nonpayable
    body_ast: Any                        # raw AST for deeper analysis
    natspec: Optional[str]               # documentation

@dataclass
class StateVariable:
    name: str
    type: str
    visibility: str
    initial_value: Optional[str]

@dataclass
class Contract:
    name: str
    inherits: List[str]
    state_variables: List[StateVariable]
    functions: List[Function]
    events: List[Event]
    modifiers: List[Modifier]

class SolidityParser:
    """Minimal parser for ORACLE."""
    
    def __init__(self):
        self.parser = ts_sol.Parser()
    
    def parse(self, source: str) -> Contract:
        """Parse Solidity source into Contract structure."""
        tree = self.parser.parse(bytes(source, 'utf8'))
        return self._extract_contract(tree.root_node, source)
    
    def _extract_contract(self, node, source: str) -> Contract:
        """Extract contract definition from AST."""
        # Implementation details...
        pass
    
    def extract_cfg(self, func: Function) -> ControlFlowGraph:
        """Build control flow graph for function."""
        pass
    
    def extract_dfg(self, func: Function) -> DataFlowGraph:
        """Build data flow graph for function."""
        pass
    
    def extract_call_graph(self, contract: Contract) -> CallGraph:
        """Build inter-function call graph."""
        pass
```

**Substeps:**
1. [ ] Install tree-sitter-solidity: `pip install tree-sitter-solidity`
2. [ ] Implement `SolidityParser.__init__()` - parser setup
3. [ ] Implement `parse()` - main entry point
4. [ ] Implement `_extract_contract()` - contract-level extraction
5. [ ] Implement `_extract_function()` - function-level extraction
6. [ ] Implement `_extract_state_variable()` - state variable extraction
7. [ ] Implement `extract_cfg()` - control flow graph
8. [ ] Implement `extract_dfg()` - data flow graph
9. [ ] Implement `extract_call_graph()` - call graph
10. [ ] Test on Euler v2 source

**Time estimate:** 6-8 hours

#### 1.2 Semantic Analysis (LLM)

**File:** `oracle/semantic/intent_analyzer.py`

```python
"""
LLM-powered semantic understanding of contract intent.
"""

from dataclasses import dataclass
from typing import List
import anthropic

@dataclass
class IntentAnalysis:
    protocol_type: str                   # lending, dex, vault, bridge, etc.
    description: str                     # "Users deposit collateral to borrow..."
    actors: List[Actor]                  # who interacts and how
    value_flows: List[ValueFlow]         # how money/tokens move
    trust_assumptions: List[str]         # "oracle is trusted", "admin is honest"

class IntentAnalyzer:
    """Extract semantic intent using LLM."""
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def analyze(self, contract: Contract, source: str) -> IntentAnalysis:
        """Use LLM to understand what contract is trying to do."""
        
        prompt = f"""
        Analyze this Solidity contract and extract:
        1. Protocol type (lending, DEX, vault, bridge, staking, etc.)
        2. High-level description of what it does
        3. All actors who interact with it and their roles
        4. How value/tokens flow through the system
        5. Implicit trust assumptions (what must be true for this to be safe)
        
        Contract: {contract.name}
        
        Source code:
        ```solidity
        {source}
        ```
        
        Respond in structured JSON format.
        """
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return self._parse_response(response)
```

**Substeps:**
1. [ ] Set up Anthropic client
2. [ ] Design intent extraction prompt
3. [ ] Implement `analyze()` method
4. [ ] Implement response parsing
5. [ ] Add protocol pattern recognition (lending, DEX, vault patterns)
6. [ ] Add value flow tracing
7. [ ] Add trust assumption extraction
8. [ ] Test on known protocol types

**Time estimate:** 4-6 hours

---

## Phase 2: Assumption Extraction

### Purpose
Extract EVERY assumption the code makes — explicit and implicit.

### Input
- Parsed contract (from Phase 1)
- Semantic analysis (from Phase 1)

### Output
```python
@dataclass
class Assumption:
    id: str                              # A001, A002, etc.
    type: AssumptionType                 # EXPLICIT, IMPLICIT, ECONOMIC, TEMPORAL
    source: str                          # function name or "global"
    statement: str                       # human-readable assumption
    formal: Optional[str]                # formal representation (for verification)
    confidence: float                    # how confident we are this is real
    
class AssumptionType(Enum):
    EXPLICIT = "explicit"                # require(), assert()
    IMPLICIT = "implicit"                # not checked but assumed
    ECONOMIC = "economic"                # about rational actors
    TEMPORAL = "temporal"                # about time/ordering
    EXTERNAL = "external"                # about external contracts/oracles
```

### Implementation

#### 2.1 Explicit Assumption Extraction

**File:** `oracle/assumptions/explicit_extractor.py`

```python
"""
Extract explicit assumptions from require/assert statements.
"""

@dataclass
class ExplicitAssumption:
    function: str
    line: int
    condition: str                       # the actual condition
    revert_message: Optional[str]
    formal: str                          # SMT-LIB or interval representation

class ExplicitExtractor:
    """Extract assumptions from require/assert/if-revert patterns."""
    
    def extract(self, contract: Contract) -> List[ExplicitAssumption]:
        assumptions = []
        
        for func in contract.functions:
            # Find require() statements
            assumptions.extend(self._extract_requires(func))
            
            # Find assert() statements
            assumptions.extend(self._extract_asserts(func))
            
            # Find if-revert patterns
            assumptions.extend(self._extract_if_reverts(func))
            
            # Find modifier conditions
            assumptions.extend(self._extract_modifier_conditions(func))
        
        return assumptions
    
    def _extract_requires(self, func: Function) -> List[ExplicitAssumption]:
        """Find all require(condition, message) statements."""
        pass
    
    def _condition_to_formal(self, condition_ast) -> str:
        """Convert Solidity condition to formal representation."""
        # amount > 0 → amount ∈ (0, MAX_UINT256]
        # msg.sender == owner → caller = owner_address
        pass
```

**Substeps:**
1. [ ] Implement require() extraction from AST
2. [ ] Implement assert() extraction
3. [ ] Implement if-revert pattern detection
4. [ ] Implement modifier condition extraction
5. [ ] Build condition-to-formal converter
6. [ ] Handle compound conditions (&&, ||)
7. [ ] Handle comparison operators (<, <=, ==, !=, >=, >)
8. [ ] Handle arithmetic in conditions
9. [ ] Test on sample contracts

**Time estimate:** 4-6 hours

#### 2.2 Implicit Assumption Extraction (LLM)

**File:** `oracle/assumptions/implicit_extractor.py`

```python
"""
Extract implicit assumptions using LLM reasoning.
"""

class ImplicitExtractor:
    """Use LLM to find assumptions NOT explicitly checked."""
    
    def __init__(self, client: anthropic.Anthropic):
        self.client = client
        
        # Common implicit assumption patterns
        self.patterns = [
            "reentrancy_safe",           # assumes no reentrant calls
            "oracle_fresh",              # assumes oracle data is current
            "oracle_accurate",           # assumes oracle data is correct
            "no_flash_loan",             # assumes funds weren't flash loaned
            "monotonic_time",            # assumes block.timestamp increases
            "caller_is_eoa",             # assumes msg.sender is not contract
            "token_standard_compliant",  # assumes ERC20 behaves correctly
            "no_fee_on_transfer",        # assumes token doesn't take fees
            "sufficient_liquidity",      # assumes enough liquidity exists
            "no_front_running",          # assumes tx ordering doesn't matter
        ]
    
    def extract(self, contract: Contract, func: Function) -> List[Assumption]:
        """Use LLM to identify implicit assumptions."""
        
        prompt = f"""
        Analyze this function and identify ALL implicit assumptions.
        
        An implicit assumption is something the code ASSUMES to be true
        but does NOT explicitly check with require/assert.
        
        Common categories:
        - Reentrancy: Does it assume no callbacks during execution?
        - Oracle: Does it assume price data is fresh/accurate?
        - Token behavior: Does it assume standard ERC20 behavior?
        - Timing: Does it assume certain ordering of operations?
        - External contracts: Does it assume external calls behave correctly?
        - Economic: Does it assume rational/honest actors?
        
        Function: {func.name}
        
        ```solidity
        {func.source}
        ```
        
        For each assumption, provide:
        1. Category
        2. Natural language description
        3. What could go wrong if assumption is violated
        4. Confidence level (HIGH/MEDIUM/LOW)
        
        Respond in JSON format.
        """
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return self._parse_response(response)
```

**Substeps:**
1. [ ] Design implicit assumption extraction prompt
2. [ ] Implement category-specific sub-prompts
3. [ ] Build assumption pattern library
4. [ ] Implement cross-function assumption analysis
5. [ ] Add confidence scoring
6. [ ] Implement deduplication (same assumption in multiple functions)
7. [ ] Test on known vulnerable patterns
8. [ ] Validate against historical exploits

**Time estimate:** 4-6 hours

#### 2.3 Economic Assumption Extraction

**File:** `oracle/assumptions/economic_extractor.py`

```python
"""
Extract economic assumptions about incentives and rational actors.
"""

class EconomicExtractor:
    """Identify assumptions about economic behavior."""
    
    ECONOMIC_PATTERNS = [
        # Liquidation assumptions
        ("liquidation", "Assumes liquidators will act when profitable"),
        ("liquidation_timing", "Assumes liquidation happens before insolvency"),
        
        # Arbitrage assumptions
        ("arbitrage", "Assumes arbitrageurs keep prices aligned"),
        ("no_manipulation", "Assumes prices can't be manipulated profitably"),
        
        # Governance assumptions  
        ("honest_governance", "Assumes governance acts in protocol interest"),
        ("no_flash_governance", "Assumes no flash loan governance attacks"),
        
        # MEV assumptions
        ("ordering_irrelevant", "Assumes transaction ordering doesn't matter"),
        ("no_sandwich", "Assumes no sandwich attacks on user transactions"),
    ]
    
    def extract(self, contract: Contract, intent: IntentAnalysis) -> List[Assumption]:
        """Identify economic assumptions based on protocol type."""
        
        assumptions = []
        
        if intent.protocol_type == "lending":
            assumptions.extend(self._lending_assumptions(contract))
        elif intent.protocol_type == "dex":
            assumptions.extend(self._dex_assumptions(contract))
        elif intent.protocol_type == "vault":
            assumptions.extend(self._vault_assumptions(contract))
        
        return assumptions
    
    def _lending_assumptions(self, contract: Contract) -> List[Assumption]:
        """Extract assumptions specific to lending protocols."""
        return [
            Assumption(
                type=AssumptionType.ECONOMIC,
                statement="Liquidators will liquidate underwater positions",
                formal="∀ position: health_factor < 1 → liquidated within N blocks",
            ),
            Assumption(
                type=AssumptionType.ECONOMIC,
                statement="Oracle prices reflect true market value",
                formal="∀ asset: |oracle_price - market_price| < ε",
            ),
            # ... more lending-specific assumptions
        ]
```

**Substeps:**
1. [ ] Build economic pattern library per protocol type
2. [ ] Implement lending protocol assumptions
3. [ ] Implement DEX assumptions
4. [ ] Implement vault/ERC4626 assumptions
5. [ ] Implement bridge assumptions
6. [ ] Implement staking assumptions
7. [ ] Add protocol-agnostic economic assumptions
8. [ ] Test on known economic exploits (Mango, Cream, etc.)

**Time estimate:** 3-4 hours

---

## Phase 3: Assumption Challenging

### Purpose
For each assumption, ask: "Can this be violated? If so, how?"

### Input
- All extracted assumptions (from Phase 2)
- Contract structure (from Phase 1)

### Output
```python
@dataclass
class Challenge:
    assumption: Assumption
    negation: str                        # "What if this is FALSE?"
    reachable: bool                      # Can we actually violate it?
    reachability_proof: Optional[Path]   # How to reach violation
    impact: ImpactLevel                  # What happens if violated?
    exploit_sketch: Optional[str]        # High-level attack description
```

### Implementation

#### 3.1 Reachability Checker (Custom)

**File:** `oracle/verification/reachability.py`

```python
"""
Domain-specific reachability checker.
Replaces Z3 for 95% of queries with faster, targeted analysis.
"""

from ontic.numerics.interval import Interval
from typing import Optional, List
from dataclasses import dataclass

@dataclass
class Path:
    """A sequence of function calls that reaches a state."""
    calls: List[FunctionCall]
    constraints: List[Constraint]
    terminal_state: State

class ReachabilityChecker:
    """
    Check if a state/condition is reachable from initial state.
    
    Uses interval arithmetic for bounds propagation.
    Much faster than Z3 for our domain-specific queries.
    """
    
    def __init__(self, contract: Contract, cfg: ControlFlowGraph):
        self.contract = contract
        self.cfg = cfg
    
    def can_reach(
        self, 
        target_condition: str,
        max_depth: int = 10
    ) -> tuple[bool, Optional[Path]]:
        """
        Check if target_condition can be satisfied.
        
        Returns (reachable, path_to_reach)
        """
        # Convert condition to interval constraint
        constraint = self._parse_condition(target_condition)
        
        # BFS through possible call sequences
        queue = [(self._initial_state(), [])]
        visited = set()
        
        while queue:
            state, path = queue.pop(0)
            
            # Check if target reached
            if self._satisfies(state, constraint):
                return True, Path(calls=path, terminal_state=state)
            
            # Skip if visited or too deep
            state_hash = self._hash_state(state)
            if state_hash in visited or len(path) >= max_depth:
                continue
            visited.add(state_hash)
            
            # Explore all callable functions
            for func in self._callable_functions(state):
                for params in self._generate_params(func, state):
                    new_state = self._execute_symbolic(state, func, params)
                    if new_state is not None:  # didn't revert
                        queue.append((new_state, path + [(func, params)]))
        
        return False, None
    
    def _execute_symbolic(
        self, 
        state: State, 
        func: Function, 
        params: List[Interval]
    ) -> Optional[State]:
        """
        Symbolically execute function with interval parameters.
        Returns new state (intervals) or None if definitely reverts.
        """
        # Use interval arithmetic to propagate bounds
        # This is where your interval.py shines
        pass
    
    def _satisfies(self, state: State, constraint: Constraint) -> bool:
        """Check if state intervals overlap with constraint."""
        # Interval intersection check
        pass
```

**Substeps:**
1. [ ] Design State representation with intervals
2. [ ] Implement `_parse_condition()` - condition to interval constraint
3. [ ] Implement `_initial_state()` - starting state bounds
4. [ ] Implement `_callable_functions()` - which functions can be called
5. [ ] Implement `_generate_params()` - parameter space exploration
6. [ ] Implement `_execute_symbolic()` - interval-based execution
7. [ ] Implement BFS exploration loop
8. [ ] Add depth limiting and visited tracking
9. [ ] Integrate with existing `interval.py`
10. [ ] Test on simple reachability queries
11. [ ] Benchmark against Z3 on same queries

**Time estimate:** 8-12 hours

#### 3.2 Impact Analyzer

**File:** `oracle/verification/impact.py`

```python
"""
Analyze impact of assumption violation.
"""

class ImpactLevel(Enum):
    CRITICAL = "critical"                # Direct fund theft
    HIGH = "high"                        # Significant fund loss
    MEDIUM = "medium"                    # Limited fund loss or DoS
    LOW = "low"                          # Minor issues
    INFORMATIONAL = "info"               # No direct impact

class ImpactAnalyzer:
    """Determine what happens when an assumption is violated."""
    
    def analyze(
        self, 
        assumption: Assumption,
        violation_path: Path
    ) -> tuple[ImpactLevel, str]:
        """
        Trace execution from violation to impact.
        
        Returns (severity, description)
        """
        # Execute violation path symbolically
        final_state = self._trace_execution(violation_path)
        
        # Check for critical impacts
        if self._detects_fund_theft(final_state):
            return ImpactLevel.CRITICAL, "Direct theft of user funds"
        
        if self._detects_insolvency(final_state):
            return ImpactLevel.CRITICAL, "Protocol insolvency"
        
        if self._detects_unauthorized_access(final_state):
            return ImpactLevel.HIGH, "Unauthorized access to privileged function"
        
        # ... more impact checks
        
        return ImpactLevel.INFORMATIONAL, "No direct impact detected"
    
    def _detects_fund_theft(self, state: State) -> bool:
        """Check if attacker balance increased without deposit."""
        pass
    
    def _detects_insolvency(self, state: State) -> bool:
        """Check if total_assets < total_liabilities."""
        pass
```

**Substeps:**
1. [ ] Define impact levels matching Immunefi standards
2. [ ] Implement fund theft detection
3. [ ] Implement insolvency detection
4. [ ] Implement unauthorized access detection
5. [ ] Implement DoS detection
6. [ ] Implement griefing detection
7. [ ] Add impact explanation generation
8. [ ] Test on known exploits with known impacts

**Time estimate:** 3-4 hours

#### 3.3 Assumption Challenger (Orchestrator)

**File:** `oracle/challenger/challenger.py`

```python
"""
Main orchestrator for challenging assumptions.
"""

class AssumptionChallenger:
    """
    For each assumption, determine if violation is reachable and impactful.
    """
    
    def __init__(
        self,
        contract: Contract,
        reachability: ReachabilityChecker,
        impact: ImpactAnalyzer,
        llm: anthropic.Anthropic
    ):
        self.contract = contract
        self.reachability = reachability
        self.impact = impact
        self.llm = llm
    
    def challenge_all(
        self, 
        assumptions: List[Assumption]
    ) -> List[Challenge]:
        """Challenge every assumption, return vulnerable ones."""
        
        challenges = []
        
        for assumption in assumptions:
            challenge = self.challenge(assumption)
            if challenge.reachable and challenge.impact != ImpactLevel.INFORMATIONAL:
                challenges.append(challenge)
        
        # Sort by impact
        challenges.sort(key=lambda c: c.impact.value, reverse=True)
        
        return challenges
    
    def challenge(self, assumption: Assumption) -> Challenge:
        """Challenge a single assumption."""
        
        # Generate negation
        negation = self._negate(assumption)
        
        # Check reachability
        reachable, path = self.reachability.can_reach(negation)
        
        if not reachable:
            return Challenge(
                assumption=assumption,
                negation=negation,
                reachable=False,
                impact=ImpactLevel.INFORMATIONAL
            )
        
        # Analyze impact
        impact_level, impact_desc = self.impact.analyze(assumption, path)
        
        # Generate exploit sketch using LLM
        exploit_sketch = self._generate_exploit_sketch(assumption, path)
        
        return Challenge(
            assumption=assumption,
            negation=negation,
            reachable=True,
            reachability_proof=path,
            impact=impact_level,
            exploit_sketch=exploit_sketch
        )
    
    def _negate(self, assumption: Assumption) -> str:
        """Generate negation of assumption."""
        # "oracle price is fresh" → "oracle price is stale"
        # "amount > 0" → "amount == 0"
        pass
    
    def _generate_exploit_sketch(
        self, 
        assumption: Assumption, 
        path: Path
    ) -> str:
        """Use LLM to describe the attack in human terms."""
        prompt = f"""
        An assumption violation was found:
        
        Assumption: {assumption.statement}
        Violation path: {path}
        
        Describe in 2-3 sentences how an attacker would exploit this.
        Be specific about the steps and the outcome.
        """
        
        response = self.llm.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
```

**Substeps:**
1. [ ] Implement assumption negation logic
2. [ ] Wire up reachability checker
3. [ ] Wire up impact analyzer
4. [ ] Implement exploit sketch generation
5. [ ] Add parallel processing for multiple assumptions
6. [ ] Add progress reporting
7. [ ] Add result caching
8. [ ] Test end-to-end on sample contract

**Time estimate:** 4-6 hours

---

## Phase 4: Adversarial Scenario Generation

### Purpose
Go beyond assumption challenges — generate creative attack scenarios.

### Input
- Vulnerable assumptions (from Phase 3)
- Contract structure
- Protocol type

### Output
```python
@dataclass
class AttackScenario:
    name: str                            # "Flash Loan Oracle Manipulation"
    description: str                     # detailed attack narrative
    steps: List[AttackStep]              # concrete transaction sequence
    required_capital: int                # how much $ needed
    expected_profit: int                 # how much $ gained
    prerequisites: List[str]             # what must be true
    complexity: str                      # LOW/MEDIUM/HIGH
```

### Implementation

#### 4.1 Scenario Generator (LLM)

**File:** `oracle/scenarios/generator.py`

```python
"""
LLM-powered attack scenario generation.
"""

class ScenarioGenerator:
    """Generate creative attack scenarios using LLM."""
    
    # Known attack patterns to prime the LLM
    ATTACK_PATTERNS = {
        "flash_loan_manipulation": """
            1. Flash loan large amount
            2. Use funds to manipulate price/state
            3. Profit from manipulated state
            4. Repay flash loan
        """,
        "first_depositor": """
            1. Be first to deposit tiny amount
            2. Donate large amount (no shares)
            3. Next depositor gets 0 shares due to rounding
            4. Steal their deposit
        """,
        "reentrancy": """
            1. Call vulnerable function
            2. Receive callback during execution
            3. Re-call function before state update
            4. Drain funds
        """,
        "oracle_manipulation": """
            1. Manipulate oracle price (TWAP, spot, etc.)
            2. Borrow against inflated collateral
            3. Let price revert
            4. Profit = borrowed - actual collateral value
        """,
        "governance_attack": """
            1. Flash loan governance tokens
            2. Create and pass malicious proposal
            3. Execute proposal
            4. Repay flash loan
        """,
    }
    
    def generate(
        self,
        contract: Contract,
        challenges: List[Challenge],
        intent: IntentAnalysis
    ) -> List[AttackScenario]:
        """Generate attack scenarios from vulnerable assumptions."""
        
        scenarios = []
        
        # Generate scenarios from each challenge
        for challenge in challenges:
            scenario = self._challenge_to_scenario(challenge, contract)
            if scenario:
                scenarios.append(scenario)
        
        # Generate novel scenarios using LLM creativity
        novel = self._generate_novel_scenarios(contract, intent)
        scenarios.extend(novel)
        
        # Deduplicate and rank
        scenarios = self._deduplicate(scenarios)
        scenarios = self._rank_by_feasibility(scenarios)
        
        return scenarios
    
    def _generate_novel_scenarios(
        self,
        contract: Contract,
        intent: IntentAnalysis
    ) -> List[AttackScenario]:
        """Use LLM to generate creative attacks beyond known patterns."""
        
        prompt = f"""
        You are an expert smart contract security researcher.
        
        Contract type: {intent.protocol_type}
        Description: {intent.description}
        
        Key functions:
        {self._summarize_functions(contract)}
        
        Generate 5 creative attack scenarios that could exploit this contract.
        Think beyond common patterns. Consider:
        - Multi-step attacks across multiple transactions
        - Attacks that combine multiple vulnerabilities
        - Economic attacks that don't require bugs
        - Timing-based attacks
        - Cross-protocol attacks
        
        For each scenario, provide:
        1. Attack name
        2. Step-by-step description
        3. Required capital
        4. Expected profit (order of magnitude)
        5. Probability of success (LOW/MEDIUM/HIGH)
        
        Be creative but realistic. Only suggest attacks that could actually work.
        """
        
        response = self.llm.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return self._parse_scenarios(response)
```

**Substeps:**
1. [ ] Build attack pattern library
2. [ ] Implement challenge-to-scenario conversion
3. [ ] Design novel scenario generation prompt
4. [ ] Implement scenario parsing
5. [ ] Add cross-protocol scenario generation
6. [ ] Implement deduplication
7. [ ] Implement feasibility ranking
8. [ ] Add capital requirement estimation
9. [ ] Add profit estimation
10. [ ] Test on known historical attacks

**Time estimate:** 4-6 hours

---

## Phase 5: Verification

### Purpose
Prove attack scenarios are real, not hallucinations.

### Input
- Attack scenarios (from Phase 4)

### Output
```python
@dataclass
class VerifiedExploit:
    scenario: AttackScenario
    verification_method: str             # "symbolic", "interval", "concrete"
    proof: Union[SMTProof, IntervalProof, ConcreteTrace]
    confidence: float                    # 0.0 - 1.0
    foundry_test: str                    # ready-to-run Foundry test
```

### Implementation

#### 5.1 Instrumented EVM (Custom)

**File:** `oracle/execution/instrumented_evm.py`

```python
"""
Custom EVM wrapper with full observability.
Uses py-evm as backend, adds Chi tracking and instrumentation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from eth.vm.forks.cancun import CancunVM

@dataclass
class ExecutionTrace:
    opcodes: List[Opcode]
    storage_reads: List[StorageRead]
    storage_writes: List[StorageWrite]
    balance_changes: List[BalanceChange]
    external_calls: List[ExternalCall]
    events: List[Event]
    gas_used: int
    reverted: bool
    revert_reason: Optional[str]

@dataclass 
class ChiMetrics:
    """Exploit proximity metrics (from The Physics OS CFD)."""
    storage_writes: int
    unique_slots: int
    call_depth: int
    balance_delta: int
    revert_distance: int
    
    @property
    def chi(self) -> float:
        """Compute Chi value - higher = closer to exploit."""
        return (
            self.storage_writes * 1.0 +
            self.unique_slots * 2.0 +
            self.call_depth * 5.0 +
            abs(self.balance_delta) / 1e18 * 10.0 +
            (1.0 / (self.revert_distance + 1)) * 20.0
        )

class InstrumentedEVM:
    """
    EVM with full execution observability.
    
    Tracks everything needed for exploit verification:
    - Storage changes
    - Balance changes  
    - Call stack
    - Chi metrics
    """
    
    def __init__(self):
        self.state: Dict[str, Any] = {}
        self.snapshots: List[Dict] = []
        self.chi_tracker = ChiMetrics(0, 0, 0, 0, 0)
    
    def load_state(self, fork_url: str, block: int):
        """Load state from mainnet fork."""
        # Use web3 to fetch state
        pass
    
    def execute(
        self, 
        tx: Transaction,
        track_chi: bool = True
    ) -> ExecutionTrace:
        """
        Execute transaction with full instrumentation.
        
        Returns detailed trace of everything that happened.
        """
        trace = ExecutionTrace(
            opcodes=[],
            storage_reads=[],
            storage_writes=[],
            balance_changes=[],
            external_calls=[],
            events=[],
            gas_used=0,
            reverted=False,
            revert_reason=None
        )
        
        # Hook into py-evm execution
        # Record every opcode, storage access, etc.
        
        if track_chi:
            self._update_chi(trace)
        
        return trace
    
    def execute_sequence(
        self, 
        txs: List[Transaction]
    ) -> tuple[List[ExecutionTrace], float]:
        """
        Execute transaction sequence, return traces and final Chi.
        """
        traces = []
        for tx in txs:
            trace = self.execute(tx)
            traces.append(trace)
            if trace.reverted:
                break
        
        return traces, self.chi_tracker.chi
    
    def snapshot(self) -> int:
        """Save state for rollback."""
        snap_id = len(self.snapshots)
        self.snapshots.append(copy.deepcopy(self.state))
        return snap_id
    
    def rollback(self, snap_id: int):
        """Restore state from snapshot."""
        self.state = copy.deepcopy(self.snapshots[snap_id])
    
    def _update_chi(self, trace: ExecutionTrace):
        """Update Chi metrics from execution trace."""
        self.chi_tracker.storage_writes += len(trace.storage_writes)
        self.chi_tracker.unique_slots += len(set(
            w.slot for w in trace.storage_writes
        ))
        # ... update other metrics
```

**Substeps:**
1. [ ] Set up py-evm integration
2. [ ] Implement state loading from RPC
3. [ ] Implement opcode tracing hooks
4. [ ] Implement storage access tracking
5. [ ] Implement balance change tracking
6. [ ] Implement external call tracking
7. [ ] Implement Chi metrics computation
8. [ ] Implement snapshot/rollback
9. [ ] Test on simple transactions
10. [ ] Benchmark against Anvil

**Time estimate:** 8-12 hours

#### 5.2 Multi-Method Verifier

**File:** `oracle/verification/verifier.py`

```python
"""
Multi-method exploit verification.
"""

class ExploitVerifier:
    """
    Verify exploit scenarios using multiple methods.
    
    Methods (in order of rigor):
    1. Interval arithmetic - rigorous bounds
    2. Symbolic execution - path feasibility
    3. Concrete execution - actual proof
    4. Kantorovich - mathematical certificate
    """
    
    def __init__(
        self,
        evm: InstrumentedEVM,
        interval_engine: IntervalEngine,      # from The Physics OS
        kantorovich: KantorovichVerifier      # from The Physics OS
    ):
        self.evm = evm
        self.interval = interval_engine
        self.kantorovich = kantorovich
    
    def verify(
        self, 
        scenario: AttackScenario,
        methods: List[str] = ["interval", "concrete"]
    ) -> Optional[VerifiedExploit]:
        """
        Verify scenario using specified methods.
        
        Returns VerifiedExploit if confirmed, None if not exploitable.
        """
        
        # Convert scenario to transaction sequence
        txs = self._scenario_to_txs(scenario)
        
        results = {}
        
        # Method 1: Interval verification (fastest, rigorous)
        if "interval" in methods:
            interval_result = self._verify_interval(txs)
            results["interval"] = interval_result
            
            # If interval says impossible, it's impossible
            if interval_result.impossible:
                return None
        
        # Method 2: Concrete execution (definitive)
        if "concrete" in methods:
            concrete_result = self._verify_concrete(txs)
            results["concrete"] = concrete_result
            
            if concrete_result.success:
                return VerifiedExploit(
                    scenario=scenario,
                    verification_method="concrete",
                    proof=concrete_result.trace,
                    confidence=0.99,
                    foundry_test=self._generate_foundry_test(txs, concrete_result)
                )
        
        # Method 3: Kantorovich certificate (mathematical proof)
        if "kantorovich" in methods and results.get("concrete", {}).get("success"):
            kant_result = self._verify_kantorovich(txs)
            if kant_result.certified:
                return VerifiedExploit(
                    scenario=scenario,
                    verification_method="kantorovich",
                    proof=kant_result.certificate,
                    confidence=1.0,  # Mathematical certainty
                    foundry_test=self._generate_foundry_test(txs, concrete_result)
                )
        
        return None
    
    def _verify_interval(self, txs: List[Transaction]) -> IntervalResult:
        """
        Verify using interval arithmetic.
        
        Propagate input intervals through execution.
        If profit interval is strictly > 0, exploit is GUARANTEED to work.
        If profit interval is strictly < 0, exploit is IMPOSSIBLE.
        """
        from ontic.numerics.interval import Interval
        
        # Initialize state with intervals
        state = self._initial_state_intervals()
        
        for tx in txs:
            state = self._execute_interval(state, tx)
            if state is None:  # Definitely reverts
                return IntervalResult(impossible=True)
        
        profit = state.attacker_balance - state.initial_attacker_balance
        
        if profit.lo > 0:
            return IntervalResult(
                impossible=False,
                guaranteed=True,
                profit_bounds=profit
            )
        elif profit.hi < 0:
            return IntervalResult(impossible=True)
        else:
            return IntervalResult(
                impossible=False,
                guaranteed=False,
                profit_bounds=profit
            )
    
    def _verify_concrete(self, txs: List[Transaction]) -> ConcreteResult:
        """Execute on forked mainnet."""
        
        # Snapshot before
        snap = self.evm.snapshot()
        initial_balance = self.evm.get_balance(ATTACKER)
        
        # Execute
        traces, chi = self.evm.execute_sequence(txs)
        
        # Check result
        final_balance = self.evm.get_balance(ATTACKER)
        profit = final_balance - initial_balance
        
        # Rollback
        self.evm.rollback(snap)
        
        return ConcreteResult(
            success=profit > 0,
            profit=profit,
            chi=chi,
            traces=traces
        )
    
    def _generate_foundry_test(
        self, 
        txs: List[Transaction],
        result: ConcreteResult
    ) -> str:
        """Generate Foundry test that reproduces exploit."""
        
        test = f"""
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "forge-std/Test.sol";

contract ExploitTest is Test {{
    function setUp() public {{
        // Fork mainnet at block {self.evm.block_number}
        vm.createSelectFork(vm.envString("ETH_RPC_URL"), {self.evm.block_number});
    }}
    
    function testExploit() public {{
        address attacker = address(this);
        uint256 initialBalance = attacker.balance;
        
        // Execute exploit
"""
        
        for i, tx in enumerate(txs):
            test += f"        // Step {i+1}: {tx.description}\n"
            test += f"        {self._tx_to_solidity(tx)}\n"
        
        test += f"""
        
        // Verify profit
        uint256 finalBalance = attacker.balance;
        assertGt(finalBalance, initialBalance, "Exploit should be profitable");
        
        console.log("Profit:", finalBalance - initialBalance);
    }}
}}
"""
        return test
```

**Substeps:**
1. [ ] Implement scenario-to-transactions converter
2. [ ] Implement interval verification method
3. [ ] Implement concrete execution method
4. [ ] Integrate Kantorovich verifier from The Physics OS
5. [ ] Implement Foundry test generation
6. [ ] Add transaction encoding helpers
7. [ ] Add profit calculation
8. [ ] Test verification pipeline end-to-end

**Time estimate:** 6-8 hours

---

## Phase 6: Synthesis

### Purpose
Generate submission-ready reports.

### Input
- Verified exploits (from Phase 5)

### Output
- Immunefi-format markdown report
- Foundry test file
- Impact assessment
- Remediation recommendations

### Implementation

#### 6.1 Report Generator

**File:** `oracle/reporting/report_generator.py`

```python
"""
Generate submission-ready vulnerability reports.
"""

class ReportGenerator:
    """Generate Immunefi/Code4rena formatted reports."""
    
    def generate(self, exploit: VerifiedExploit) -> Report:
        """Generate complete vulnerability report."""
        
        return Report(
            title=self._generate_title(exploit),
            severity=self._classify_severity(exploit),
            summary=self._generate_summary(exploit),
            vulnerability_details=self._generate_details(exploit),
            impact=self._generate_impact(exploit),
            proof_of_concept=exploit.foundry_test,
            tools_used="ORACLE (custom) + Foundry",
            recommendation=self._generate_recommendation(exploit),
        )
    
    def _classify_severity(self, exploit: VerifiedExploit) -> str:
        """Classify according to Immunefi severity guidelines."""
        
        profit = exploit.proof.profit if hasattr(exploit.proof, 'profit') else 0
        
        if profit > 10_000_000e18:  # > $10M
            return "Critical"
        elif profit > 1_000_000e18:  # > $1M
            return "Critical"
        elif profit > 100_000e18:    # > $100K
            return "High"
        elif profit > 10_000e18:     # > $10K
            return "Medium"
        else:
            return "Low"
    
    def _generate_summary(self, exploit: VerifiedExploit) -> str:
        """Generate executive summary."""
        
        prompt = f"""
        Write a clear, professional vulnerability summary for a bug bounty submission.
        
        Attack: {exploit.scenario.name}
        Steps: {exploit.scenario.steps}
        Profit: {exploit.proof.profit / 1e18} ETH
        
        The summary should:
        1. State the vulnerability in one sentence
        2. Explain the root cause
        3. State the impact
        
        Be concise and professional. No hype.
        """
        
        response = self.llm.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def export_immunefi(self, report: Report) -> str:
        """Export as Immunefi markdown format."""
        
        return f"""
# {report.title}

## Summary
{report.summary}

## Vulnerability Details
{report.vulnerability_details}

## Impact
{report.impact}

## Proof of Concept

```solidity
{report.proof_of_concept}
```

## Recommendation
{report.recommendation}

## Tools Used
{report.tools_used}
"""
```

**Substeps:**
1. [ ] Implement severity classification
2. [ ] Implement summary generation
3. [ ] Implement details generation
4. [ ] Implement impact description
5. [ ] Implement recommendation generation
6. [ ] Add Immunefi markdown export
7. [ ] Add Code4rena format export
8. [ ] Add JSON export for archival
9. [ ] Test on sample exploits

**Time estimate:** 2-4 hours

---

## Custom Stack Decisions

### What We Build Custom vs. Use Existing

| Component | Decision | Reason |
|-----------|----------|--------|
| **Solidity Parser** | BUILD CUSTOM | Slither's IR is designed for their detectors, not our semantic analysis |
| **Reachability Checker** | BUILD CUSTOM | interval.py + domain-specific beats general Z3 |
| **EVM Execution** | WRAP py-evm | Need Chi instrumentation; Anvil for final verify only |
| **tree-sitter** | USE | Battle-tested parser generator, no opinions |
| **py-evm** | USE AS REFERENCE | Don't reinvent EVM opcodes |
| **Claude API** | USE | The LLM is the LLM |
| **interval.py** | HAVE | Already built in The Physics OS |
| **kantorovich.py** | HAVE | Already built in The Physics OS |
| **Z3** | OPTIONAL FALLBACK | Only for the 5% of queries that need full SMT |
| **Anvil** | FINAL VERIFY ONLY | Too slow for exploration, no instrumentation |

### Dependencies

```toml
# pyproject.toml

[project]
name = "oracle"
version = "0.1.0"
dependencies = [
    "tree-sitter-solidity>=0.1.0",     # Solidity parsing
    "anthropic>=0.18.0",                # Claude API
    "web3>=6.0.0",                      # Ethereum interaction
    "py-evm>=0.7.0",                    # EVM execution (reference)
    "eth-abi>=4.0.0",                   # ABI encoding
    "eth-utils>=2.0.0",                 # Ethereum utilities
]

[project.optional-dependencies]
z3 = ["z3-solver>=4.12.0"]             # Optional SMT fallback
```

---

## Implementation Plan

### Phase 1: Foundation (Day 1)

| Task | Time | Output |
|------|------|--------|
| 1.1 Set up project structure | 1 hour | `oracle/` directory tree |
| 1.2 Install dependencies | 30 min | Working environment |
| 1.3 Implement SolidityParser | 6-8 hours | AST, CFG, DFG extraction |
| 1.4 Test parser on Euler v2 | 1 hour | Verified parsing works |

**Day 1 Deliverable:** Can parse Solidity into our structures

### Phase 2: Semantic Layer (Day 2)

| Task | Time | Output |
|------|------|--------|
| 2.1 Implement IntentAnalyzer | 4-6 hours | LLM-based intent extraction |
| 2.2 Implement ExplicitExtractor | 4-6 hours | require/assert extraction |
| 2.3 Implement ImplicitExtractor | 4-6 hours | LLM-based implicit assumptions |
| 2.4 Test assumption extraction | 1 hour | Verified on known contracts |

**Day 2 Deliverable:** Can extract all assumptions from a contract

### Phase 3: Verification Core (Day 3)

| Task | Time | Output |
|------|------|--------|
| 3.1 Implement ReachabilityChecker | 8-12 hours | Domain-specific reachability |
| 3.2 Integrate interval.py | 2 hours | Interval arithmetic working |
| 3.3 Implement ImpactAnalyzer | 3-4 hours | Impact classification |
| 3.4 Test reachability queries | 1 hour | Verified on sample queries |

**Day 3 Deliverable:** Can determine if assumptions can be violated

### Phase 4: Execution Layer (Day 4)

| Task | Time | Output |
|------|------|--------|
| 4.1 Implement InstrumentedEVM | 8-12 hours | py-evm with Chi tracking |
| 4.2 Implement state loading | 2 hours | Can load mainnet state |
| 4.3 Implement snapshot/rollback | 1 hour | State management working |
| 4.4 Test execution traces | 1 hour | Verified instrumentation |

**Day 4 Deliverable:** Can execute and trace transactions with Chi

### Phase 5: Integration (Day 5)

| Task | Time | Output |
|------|------|--------|
| 5.1 Implement AssumptionChallenger | 4-6 hours | End-to-end challenge pipeline |
| 5.2 Implement ScenarioGenerator | 4-6 hours | LLM attack scenarios |
| 5.3 Implement ExploitVerifier | 4-6 hours | Multi-method verification |
| 5.4 Wire up full pipeline | 2 hours | ORACLE runs end-to-end |

**Day 5 Deliverable:** Full ORACLE pipeline working

### Phase 6: Polish (Day 6)

| Task | Time | Output |
|------|------|--------|
| 6.1 Implement ReportGenerator | 2-4 hours | Immunefi-format reports |
| 6.2 Add Foundry test generation | 2 hours | Ready-to-submit PoCs |
| 6.3 Add CLI interface | 2 hours | `oracle hunt <contract>` |
| 6.4 Test on Euler v2 | 4 hours | Validate against Chi 250 signal |
| 6.5 Documentation | 2 hours | Usage docs |

**Day 6 Deliverable:** Production-ready ORACLE

---

## Timeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ORACLE BUILD TIMELINE                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  DAY 1: Foundation                                                      │
│  ════════════════                                                       │
│  [████████████████████████████████████████] Solidity Parser            │
│  Output: Can parse Solidity → AST/CFG/DFG                               │
│                                                                         │
│  DAY 2: Semantic Layer                                                  │
│  ═══════════════════                                                    │
│  [████████████████████] Intent Analyzer                                │
│  [████████████████████] Assumption Extractors                          │
│  Output: Can extract all assumptions                                    │
│                                                                         │
│  DAY 3: Verification Core                                               │
│  ════════════════════════                                               │
│  [████████████████████████████████████████] Reachability Checker       │
│  [██████████] Impact Analyzer                                          │
│  Output: Can verify if assumptions are violable                         │
│                                                                         │
│  DAY 4: Execution Layer                                                 │
│  ══════════════════════                                                 │
│  [████████████████████████████████████████] Instrumented EVM           │
│  Output: Can execute with Chi tracking                                  │
│                                                                         │
│  DAY 5: Integration                                                     │
│  ════════════════                                                       │
│  [████████████████████] Challenger                                     │
│  [████████████████████] Scenario Generator                             │
│  [████████████████████] Verifier                                       │
│  Output: Full pipeline working                                          │
│                                                                         │
│  DAY 6: Polish                                                          │
│  ════════════                                                           │
│  [██████████] Report Generator                                         │
│  [██████████] CLI Interface                                            │
│  [████████████████████] Testing & Docs                                 │
│  Output: Production-ready ORACLE                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

Total: ~40-60 hours = 4-6 days at normal pace
       ~4-6 days at your velocity = probably 3-4 days
```

---

## Success Metrics

### Technical Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Parse success rate | >95% | % of Solidity files parsed without error |
| Assumption extraction recall | >80% | Manual review of known vulnerable contracts |
| Reachability accuracy | >90% | Compare to Z3 on test set |
| False positive rate | <30% | Manual review of flagged issues |
| End-to-end time | <10 min/contract | Benchmark on standard contracts |

### Business Metrics

| Metric | Target | Timeframe |
|--------|--------|-----------|
| First valid submission | Within 1 week of completion | |
| First paid bounty | Within 1 month | |
| Total bounty value | >$100K | First 6 months |
| Contracts analyzed | >100 | First month |

### Validation Criteria

ORACLE is considered successful if it can:

1. [ ] Rediscover at least 2/3 historical exploits (DAO, bZx, Harvest)
2. [ ] Find a valid issue in Euler v2 (Chi 250 signal)
3. [ ] Generate a submission-ready report in <30 minutes
4. [ ] Produce zero false positives on 10 audited "safe" contracts

---

## Integration with The Physics OS

### Existing Tools to Integrate

| HyperTensor Module | ORACLE Use |
|-------------------|------------|
| `ontic/numerics/interval.py` | Rigorous bounds propagation |
| `ontic/cfd/kantorovich.py` | Mathematical proof certificates |
| `ontic/cfd/chi_diagnostic.py` | Exploit proximity tracking |
| `ontic/cfd/singularity_hunter.py` | Objective maximization |

### File Structure

```
physics-os-main/
├── ontic/
│   ├── cfd/           # Existing CFD tools
│   ├── numerics/      # interval.py lives here
│   └── oracle/        # NEW: ORACLE engine
│       ├── __init__.py
│       ├── parsing/
│       │   ├── __init__.py
│       │   └── solidity_parser.py
│       ├── semantic/
│       │   ├── __init__.py
│       │   └── intent_analyzer.py
│       ├── assumptions/
│       │   ├── __init__.py
│       │   ├── explicit_extractor.py
│       │   ├── implicit_extractor.py
│       │   └── economic_extractor.py
│       ├── challenger/
│       │   ├── __init__.py
│       │   └── challenger.py
│       ├── verification/
│       │   ├── __init__.py
│       │   ├── reachability.py
│       │   ├── impact.py
│       │   └── verifier.py
│       ├── execution/
│       │   ├── __init__.py
│       │   └── instrumented_evm.py
│       ├── scenarios/
│       │   ├── __init__.py
│       │   └── generator.py
│       └── reporting/
│           ├── __init__.py
│           └── report_generator.py
├── tests/
│   └── test_oracle_*.py
└── demos/
    └── oracle_hunt.py
```

---

## Quick Start (After Build)

```bash
# Hunt a contract
oracle hunt 0x1234...abcd --chain ethereum --output report.md

# Hunt from source file
oracle hunt ./contracts/Vault.sol --output report.md

# Hunt with specific focus
oracle hunt 0x1234...abcd --focus precision --output report.md
oracle hunt 0x1234...abcd --focus reentrancy --output report.md

# Verify specific scenario
oracle verify ./scenario.json --output verified.md
```

```python
# Programmatic usage
from ontic.oracle import ORACLE

oracle = ORACLE(
    anthropic_key="sk-...",
    eth_rpc="https://eth-mainnet.g.alchemy.com/v2/..."
)

# Hunt contract
results = oracle.hunt(
    address="0x1234...abcd",
    chain="ethereum"
)

# Get verified exploits
for exploit in results.verified_exploits:
    print(f"Found: {exploit.scenario.name}")
    print(f"Profit: {exploit.proof.profit / 1e18} ETH")
    print(f"Confidence: {exploit.confidence}")
    
    # Generate report
    report = oracle.generate_report(exploit)
    report.save("./reports/")
```

---

## Conclusion

ORACLE is not another bug bounty tool. It's the automation of what makes human auditors effective:

1. **Understanding intent** — what the code SHOULD do
2. **Extracting assumptions** — what must be true for it to be safe
3. **Challenging assumptions** — what if that's wrong?
4. **Generating scenarios** — how would an attacker exploit that?
5. **Verifying mathematically** — is this real or hallucination?

The combination of LLM semantic understanding with formal verification creates something that didn't exist before 2024: a system that reasons like a human but proves like a machine.

Build it.

---

*ORACLE Architecture v1.0 | January 19, 2026 | Tigantic Holdings*
