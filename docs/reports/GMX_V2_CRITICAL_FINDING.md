# 🚨 GMX V2 ADL ORDERING VULNERABILITY - ACKNOWLEDGED IN CODE

## Executive Summary

Found an **acknowledged vulnerability** in GMX V2's ADL (Auto-Deleveraging) system that is explicitly documented in the source code but may not be widely known to users.

## Source Code Reference

**File:** `contracts/exchange/AdlHandler.sol`, lines 66-95

```solidity
// @dev auto-deleverages a position
// there is no validation that ADL is executed in order of position profit
// or position size, this is due to the limitation of the gas overhead
// required to check this ordering
//
// ADL keepers could be separately incentivised using a rebate based on
// position profit, this is not implemented within the contracts at the moment
```

## Vulnerability Analysis

### What This Means

1. **ADL positions are NOT closed in order of profitability**
2. **A keeper can choose WHICH position to ADL first**
3. **This creates an economic extraction opportunity**

### Attack Scenario

**Setup:**
- Pool is in ADL-enabled state (PnL > maxPnlFactorForAdl)
- User A has a $1M position with 50% profit ($500K)
- User B has a $100K position with 10% profit ($10K)

**Attack:**
1. ADL keeper sees both positions
2. Keeper ADLs User B's small position (takes less profit)
3. Price moves, User A can close at higher profit
4. Keeper (or related party) IS User A
5. Result: User A extracts more value from the pool

### Why This Is Significant

1. **Keeper Collusion**: ADL keepers can collude with large position holders
2. **MEV-style Extraction**: Ordering of ADL execution creates extractable value
3. **User Harm**: Small users may be ADL'd to protect large positions
4. **Trust Assumption**: Users trust keepers will act fairly (not enforced)

## Severity Assessment

| Factor | Rating |
|--------|--------|
| Severity | MEDIUM-HIGH |
| Exploitability | HIGH (requires keeper access) |
| Impact | Financial loss for ADL'd users |
| Likelihood | MODERATE (requires specific conditions) |

## Mitigation Status

From the code comments:
> "ADL keepers could be separately incentivised using a rebate based on position profit, this is not implemented within the contracts at the moment"

**GMX acknowledges this should be incentivized but HAS NOT IMPLEMENTED IT.**

## Bounty Eligibility Analysis

### Arguments FOR Bounty:
1. This is a design flaw with real economic impact
2. Users are not explicitly warned about ADL ordering
3. The fix (incentivized ordering) is acknowledged but not implemented
4. Creates MEV-style extraction opportunities

### Arguments AGAINST Bounty:
1. Documented in code (may be considered "known issue")
2. Requires keeper access (limited attack surface)
3. Gas overhead makes on-chain ordering impractical

## Recommendation

This finding should be submitted with focus on:
1. **Economic impact quantification**
2. **Proposed mitigation** (off-chain keeper incentives, position profit ordering)
3. **User disclosure** (ensure users know ADL ordering is arbitrary)

---

## Additional Vulnerabilities Identified

### 1. ADL Time Manipulation
From `AdlUtils.sol` lines 102-121:

```solidity
// since the latest ADL at is always updated, an ADL keeper could
// continually cause the latest ADL time to be updated and prevent
// ADL orders from being executed
```

**Finding:** ADL keepers can grief the system by continuously updating the ADL time, preventing other keepers from executing ADL.

### 2. Oracle Timestamp Race in ADL
```solidity
if (oracle.maxTimestamp() < latestAdlTime) {
    revert Errors.OracleTimestampsAreSmallerThanRequired(oracle.maxTimestamp(), latestAdlTime);
}
```

**Finding:** ADL execution requires oracle timestamps > latestAdlTime. Keepers can race to update latestAdlTime to block other keepers.

---

## Conclusion

The ADL ordering vulnerability is a **real issue** that GMX has acknowledged but not fixed. The question is whether this qualifies for their bounty program:

1. If GMX considers "documented in code" = "known issue" → NO BOUNTY
2. If GMX considers "not disclosed to users" = "needs fixing" → POTENTIAL BOUNTY

**Recommendation:** Submit with emphasis on USER IMPACT and PROPOSED SOLUTION.

