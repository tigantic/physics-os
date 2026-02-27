# Makina Protocol - QTT Kill-Chain Analysis Report

**Date:** 2025-01-XX  
**Analyst:** QTT Cross-Contract Analyzer  
**Safe Harbor:** `0x464C71f6c2F760DdA6093dCB91C24c39e5d6e18c`

---

## Targets

| Contract | Address | Bytecode |
|----------|---------|----------|
| Caliber Proxy | `0x06147e073B854521c7B778280E7d7dBAfB2D4898` | 24,274 bytes |
| DUSD/USDC Pool | `0x32E616F4f17d43f9A5cd9Be0e294727187064cb3` | 23,635 bytes |

---

## Phase 1: Disassembly Results

**Caliber Proxy:**
- Basic blocks: 1,464
- Control flow edges: 1,559
- Function selectors: harvest(), initialize(), managePositionBatch(), transferToHubMachine()

**DUSD/USDC Pool:**
- Basic blocks: 2,783
- Control flow edges: 1,358
- Function selectors: get_dy(), get_dx(), admin(), fee_receiver(), dynamic_fee()

**Cross-Contract Metrics:**
- Total reentrancy patterns detected: 23
- Total cross-call points: 132

---

## Phase 2: Tensor Construction

Combined adjacency tensor: **4247 × 4247**
- Caliber contribution: 1,464 blocks
- Pool contribution: 2,783 blocks
- Cross-contract edges: **0** (no direct call paths between contracts)

---

## Phase 3: Constraint Analysis

### Constraint 1: ReentrancyGuard Bypass
**Requirement:** Find compromised state variable or broken ReentrancyGuard modifier that can be bypassed using cross-contract state desynchronization

**Findings:**
- ReentrancyGuard blocks in Caliber: **22**
- CEI (Checks-Effects-Interactions) violations: **0**

**Status:** ❌ BLOCKED  
ReentrancyGuard is properly implemented. No CEI violations detected. Guard pattern follows SLOAD→JUMPI→SSTORE lock acquisition correctly.

---

### Constraint 2: Cross-Contract State Desynchronization
**Requirement:** Identify state desync vector between Caliber and Pool

**Findings:**
- DELEGATECALL state desync points: **1**
- SSTORE after CALL patterns: **0**

**Status:** ❌ BLOCKED  
Single DELEGATECALL point exists but there are no state writes after external calls. Cross-contract state desynchronization attack not viable.

---

### Constraint 3: Emergency Access Control
**Requirement:** Find emergency withdraw() or rescueFunds() function where access control requires specific state permutation

**Findings:**
- Emergency function candidates: **0**
- Detected function selectors: None matching emergency/rescue/withdraw patterns with state-based access control

**Status:** ❌ NOT FOUND  
No emergency extraction functions with exploitable access control detected.

---

## On-Chain State Analysis

**Caliber Proxy Storage:**
```
Slot 0: 0x0
Slot 1: 0x0
Slot 2: 0x0
Slot 3: 0x0
Slot 4: 0x0
EIP-1967 Admin: 0x0
EIP-1967 Implementation: 0x0
```
Storage appears uninitialized or wiped.

**Pool State:**
- hubMachineEndpoint(): `0x0`
- pendingTimelockExpiry(): `0`
- admin() call: reverts

**Value at Risk:**
| Token | Balance | USD Value |
|-------|---------|-----------|
| ETH | 0 | $0 |
| USDC | 10,541,453,848 | ~$10,541 |
| DUSD (DefiDollar) | 0 | $0 |

---

## Final Verdict

### **NOT_EXPLOITABLE**

**Reason:** All three requested constraints failed satisfaction:

1. ❌ ReentrancyGuard is properly implemented with 22 guard blocks and zero CEI violations
2. ❌ Cross-contract state desync blocked - no SSTORE patterns after external CALL/DELEGATECALL
3. ❌ No emergency withdraw functions with state-based access control found

The Makina Protocol contracts follow secure patterns consistent with audited Curve-style pools. The ~$10,541 USDC cannot be extracted via reentrancy, state desync, or emergency function exploitation.

---

## Attestation

```
Target: Makina Protocol
Contracts: [0x06147e073B854521c7B778280E7d7dBAfB2D4898, 0x32E616F4f17d43f9A5cd9Be0e294727187064cb3]
Value at Risk: $10,541 USDC
Verdict: NOT_EXPLOITABLE
Constraint 1 (Reentrancy): BLOCKED
Constraint 2 (State Desync): BLOCKED  
Constraint 3 (Emergency Func): NOT_FOUND
Safe Harbor: NOT_ACTIVATED - No extraction path
```
