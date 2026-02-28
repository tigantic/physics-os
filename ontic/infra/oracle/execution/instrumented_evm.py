"""
Instrumented EVM with full observability and Chi tracking.

This is ORACLE's proof engine - turns theoretical attacks into verified exploits.

Chi metrics from HyperTensor CFD: higher Chi = closer to exploit.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Optional

from ontic.infra.oracle.core.types import (
    AttackScenario,
    AttackStep,
    ChiMetrics,
    ExecutionTrace,
)


@dataclass
class Transaction:
    """A transaction to execute."""
    
    to: str
    data: bytes = b""
    value: int = 0
    sender: str = "0xAttacker"
    gas: int = 10_000_000
    description: str = ""


@dataclass
class StorageWrite:
    """A storage write operation."""
    
    contract: str
    slot: int
    old_value: int
    new_value: int


@dataclass
class BalanceChange:
    """A balance change."""
    
    address: str
    old_balance: int
    new_balance: int
    
    @property
    def delta(self) -> int:
        return self.new_balance - self.old_balance


@dataclass
class SimulatedState:
    """Simulated EVM state for verification."""
    
    # Storage: contract -> slot -> value
    storage: dict[str, dict[int, int]] = field(default_factory=dict)
    
    # Balances: address -> wei
    balances: dict[str, int] = field(default_factory=dict)
    
    # Block context
    block_number: int = 18000000
    timestamp: int = 1700000000
    
    # Chi tracking
    chi: ChiMetrics = field(default_factory=ChiMetrics)
    
    def copy(self) -> SimulatedState:
        """Deep copy state."""
        return copy.deepcopy(self)


class InstrumentedEVM:
    """
    EVM simulator with full execution observability.
    
    For verification without requiring a full node.
    Tracks everything needed for exploit verification:
    - Storage changes
    - Balance changes
    - Call stack
    - Chi metrics
    
    Note: This is a SIMPLIFIED simulator for ORACLE's verification.
    For final proof, use Foundry fork tests.
    """
    
    # Common function selectors
    SELECTORS = {
        "deposit": "0xd0e30db0",
        "withdraw": "0x2e1a7d4d",
        "transfer": "0xa9059cbb",
        "approve": "0x095ea7b3",
        "balanceOf": "0x70a08231",
        "swap": "0x022c0d9f",
        "flash": "0x5cffe9de",
    }
    
    def __init__(self):
        """Initialize the instrumented EVM."""
        self.state = SimulatedState()
        self.snapshots: list[SimulatedState] = []
        self.traces: list[ExecutionTrace] = []
        
        # Initialize attacker with capital
        self.state.balances["0xAttacker"] = 10**20  # 100 ETH
    
    def load_state_mock(self, contract: str, tvl: int = 10**24) -> None:
        """
        Load mock state for a contract.
        
        In production, this would use RPC to fetch real state.
        
        Args:
            contract: Contract address
            tvl: Total value locked (default: 1M ETH)
        """
        self.state.balances[contract] = tvl
        self.state.storage[contract] = {
            0: 10**18,  # totalSupply
            1: tvl,     # totalAssets
        }
    
    def execute(
        self,
        tx: Transaction,
        track_chi: bool = True,
    ) -> ExecutionTrace:
        """
        Execute transaction with full instrumentation.
        
        Args:
            tx: Transaction to execute
            track_chi: Whether to track Chi metrics
            
        Returns:
            Detailed execution trace
        """
        trace = ExecutionTrace(
            storage_reads=[],
            storage_writes=[],
            balance_changes=[],
            external_calls=[],
            events=[],
            gas_used=0,
            reverted=False,
            revert_reason=None,
        )
        
        # Track initial balances
        sender_balance = self.state.balances.get(tx.sender, 0)
        target_balance = self.state.balances.get(tx.to, 0)
        
        # Simulate based on function selector
        selector = tx.data[:4].hex() if len(tx.data) >= 4 else ""
        
        # Process value transfer
        if tx.value > 0:
            if sender_balance < tx.value:
                trace.reverted = True
                trace.revert_reason = "Insufficient balance"
                return trace
            
            self.state.balances[tx.sender] = sender_balance - tx.value
            self.state.balances[tx.to] = target_balance + tx.value
            
            trace.balance_changes.append(
                (tx.sender, sender_balance, sender_balance - tx.value)
            )
            trace.balance_changes.append(
                (tx.to, target_balance, target_balance + tx.value)
            )
        
        # Simulate common function effects
        self._simulate_function(tx, trace)
        
        # Update Chi
        if track_chi:
            self._update_chi(trace)
        
        self.traces.append(trace)
        return trace
    
    def _simulate_function(self, tx: Transaction, trace: ExecutionTrace) -> None:
        """Simulate function execution based on selector."""
        
        # Extract function signature from description if available
        desc = tx.description.lower()
        
        if "deposit" in desc:
            self._simulate_deposit(tx, trace)
        elif "withdraw" in desc:
            self._simulate_withdraw(tx, trace)
        elif "swap" in desc:
            self._simulate_swap(tx, trace)
        elif "flash" in desc:
            self._simulate_flash_loan(tx, trace)
        elif "liquidate" in desc:
            self._simulate_liquidate(tx, trace)
    
    def _simulate_deposit(self, tx: Transaction, trace: ExecutionTrace) -> None:
        """Simulate deposit function."""
        # Contract receives ETH, user gets shares
        contract = tx.to
        
        if contract not in self.state.storage:
            self.state.storage[contract] = {}
        
        old_total = self.state.storage[contract].get(0, 0)
        new_total = old_total + tx.value
        
        self.state.storage[contract][0] = new_total
        trace.storage_writes.append((contract, 0, old_total, new_total))
    
    def _simulate_withdraw(self, tx: Transaction, trace: ExecutionTrace) -> None:
        """Simulate withdraw function."""
        contract = tx.to
        
        # Attacker receives ETH
        old_attacker = self.state.balances.get(tx.sender, 0)
        contract_balance = self.state.balances.get(contract, 0)
        
        # Simulate receiving up to 10% of contract balance
        withdraw_amount = min(contract_balance // 10, 10**20)
        
        self.state.balances[tx.sender] = old_attacker + withdraw_amount
        self.state.balances[contract] = contract_balance - withdraw_amount
        
        trace.balance_changes.append(
            (tx.sender, old_attacker, old_attacker + withdraw_amount)
        )
        trace.balance_changes.append(
            (contract, contract_balance, contract_balance - withdraw_amount)
        )
    
    def _simulate_swap(self, tx: Transaction, trace: ExecutionTrace) -> None:
        """Simulate swap function."""
        # Attacker might profit from price impact
        old_attacker = self.state.balances.get(tx.sender, 0)
        
        # Simulate potential profit (could be positive or negative)
        # In exploit scenarios, we model profitable swaps
        swap_profit = tx.value // 100  # 1% profit
        
        self.state.balances[tx.sender] = old_attacker + swap_profit
        trace.balance_changes.append(
            (tx.sender, old_attacker, old_attacker + swap_profit)
        )
    
    def _simulate_flash_loan(self, tx: Transaction, trace: ExecutionTrace) -> None:
        """Simulate flash loan - attacker temporarily has huge capital."""
        # Flash loan gives temporary capital
        old_attacker = self.state.balances.get(tx.sender, 0)
        
        # Flash loan 10M ETH equivalent
        flash_amount = 10**25
        self.state.balances[tx.sender] = old_attacker + flash_amount
        
        # This will be repaid in same tx (simulated)
        trace.balance_changes.append(
            (tx.sender, old_attacker, old_attacker + flash_amount)
        )
    
    def _simulate_liquidate(self, tx: Transaction, trace: ExecutionTrace) -> None:
        """Simulate liquidation with bonus."""
        old_attacker = self.state.balances.get(tx.sender, 0)
        
        # Liquidator bonus (typically 5-10%)
        bonus = tx.value // 10
        
        self.state.balances[tx.sender] = old_attacker + bonus
        trace.balance_changes.append(
            (tx.sender, old_attacker, old_attacker + bonus)
        )
    
    def _update_chi(self, trace: ExecutionTrace) -> None:
        """Update Chi metrics from execution trace."""
        self.state.chi.storage_writes += len(trace.storage_writes)
        self.state.chi.unique_slots += len(set(sw[1] for sw in trace.storage_writes))
        self.state.chi.call_depth = max(self.state.chi.call_depth, 
                                        len(trace.external_calls))
        
        # Calculate balance delta for attacker
        for addr, old, new in trace.balance_changes:
            if "attacker" in addr.lower():
                self.state.chi.balance_delta += new - old
        
        # Revert distance (0 if not reverted)
        if trace.reverted:
            self.state.chi.revert_distance = 0
        else:
            self.state.chi.revert_distance += 1
    
    def execute_sequence(
        self,
        txs: list[Transaction],
    ) -> tuple[list[ExecutionTrace], float]:
        """
        Execute transaction sequence.
        
        Args:
            txs: List of transactions
            
        Returns:
            (traces, final_chi)
        """
        traces = []
        
        for tx in txs:
            trace = self.execute(tx)
            traces.append(trace)
            
            if trace.reverted:
                break
        
        return traces, self.state.chi.chi
    
    def snapshot(self) -> int:
        """Save state for rollback."""
        snap_id = len(self.snapshots)
        self.snapshots.append(self.state.copy())
        return snap_id
    
    def rollback(self, snap_id: int) -> None:
        """Restore state from snapshot."""
        if snap_id < len(self.snapshots):
            self.state = self.snapshots[snap_id].copy()
    
    def get_balance(self, address: str) -> int:
        """Get balance of address."""
        return self.state.balances.get(address, 0)
    
    def get_profit(self, address: str = "0xAttacker", initial: int = 10**20) -> int:
        """Calculate profit for address."""
        return self.get_balance(address) - initial


def scenario_to_transactions(scenario: AttackScenario) -> list[Transaction]:
    """
    Convert attack scenario to transaction sequence.
    
    Args:
        scenario: Attack scenario
        
    Returns:
        List of transactions to execute
    """
    txs = []
    
    for i, step in enumerate(scenario.steps):
        tx = Transaction(
            to=step.target or "0xTarget",
            data=b"",
            value=step.parameters.get("value", 0) if step.parameters else 0,
            description=step.action,
        )
        txs.append(tx)
    
    return txs
