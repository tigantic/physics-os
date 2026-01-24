# FLUIDELITE Halo2 Circuit Analysis Report

**Generated**: 2026-01-23T14:51:39.414114
**Files Analyzed**: 82
**Total Lines**: 32,775

## Summary

| Severity | Count |
|----------|-------|
| 🚨 CRITICAL | 0 |
| ⚠️ HIGH | 0 |
| 🔶 MEDIUM | 1676 |
| ℹ️ LOW | 4 |

## Circuits Analyzed

| File | Advice | Fixed | Selectors | Gates | Lookups | Findings |
|------|--------|-------|-----------|-------|---------|----------|

## Detailed Findings

### 🔶 MEDIUM Findings

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/utils.rs`
**Line**: 57
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now get as many as necessary
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/utils.rs`
**Line**: 80
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Strange signature of the function is due to const generics bugs
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/utils.rs`
**Line**: 103
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// additive parts
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sha256_round_function/mod.rs`
**Line**: 128
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we can have a degenerate case when queue is empty, but it's a first circuit in the queue,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sha256_round_function/mod.rs`
**Line**: 129
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// so we taken default FSM state that has state.read_precompile_call = true;
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sha256_round_function/mod.rs`
**Line**: 131
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we can only skip the full circuit if we are not in any form of progress
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sha256_round_function/mod.rs`
**Line**: 153
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// main work cycle
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sha256_round_function/mod.rs`
**Line**: 160
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we are in a proper state then get the ABI from the queue
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sha256_round_function/mod.rs`
**Line**: 183
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now compute some parameters that describe the call itself
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sha256_round_function/mod.rs`
**Line**: 194
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// also set timestamps
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sha256_round_function/mod.rs`
**Line**: 202
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// timestamps have large space, so this can be expected
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sha256_round_function/mod.rs`
**Line**: 220
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Now perform few memory queries to read content
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sha256_round_function/mod.rs`
**Line**: 254
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// perform read
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sha256_round_function/mod.rs`
**Line**: 257
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we need to change endianess. Memory is BE, and each of 4 byte chunks should be interpreted as BE 
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sha256_round_function/mod.rs`
**Line**: 278
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// absorb
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sha256_round_function/mod.rs`
**Line**: 299
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// some endianess magic
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sha256_round_function/mod.rs`
**Line**: 318
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// perform write
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sha256_round_function/mod.rs`
**Line**: 323
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// update state
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sha256_round_function/mod.rs`
**Line**: 382
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// it must be trivial
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sha256_round_function/mod.rs`
**Line**: 397
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// it must be trivial
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sha256_round_function/mod.rs`
**Line**: 442
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// form the final state
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sha256_round_function/mod.rs`
**Line**: 458
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// self-check
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/input.rs`
**Line**: 31
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
pub sha256_inner_state: [UInt32<F>; 8], // 8 uint32 words of internal sha256 state
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 35
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Similar to ConditionalWitnessAllocator, but has a logical separation of sequences,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 36
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// so if a sub-sequence ended it can also allocate a boolean to indicate it by providing boolean val
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 51
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we would need to pre-pad to avoid tracking skipping first "start new sequence"
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 99
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// pop the previous sequence
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 117
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// can return anything for witness, and we decide that we return `false`
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 150
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
bias: Variable, // any variable that has to be resolved BEFORE executing witness query
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 170
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// pop the previous sequence
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 188
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// can return anything for witness, and we decide that we return `false`
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 232
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
bias: Variable, // any variable that has to be resolved BEFORE executing witness query
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 269
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we result on the FSM input trick here to add pre-padding
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 338
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// form the final state
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 356
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// self-check
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 371
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we take a request to decommit hash H into memory page X. Following our internal conventions
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 372
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we decommit individual elements starting from the index 1 in the page, and later on set a full le
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 373
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// into index 0. All elements are 32 bytes
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 408
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we need exactly 3 sponges per cycle:
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 415
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we know that if we pop then highest 32 bits are 0 by how VM constructs a queue
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 417
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we did get a fresh request from queue we expect it to follow our convention
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 420
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// turn over the endianess
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 421
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we IGNORE the highest 4 bytes
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 462
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we decommit if we either decommit or just got a new request
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 467
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// let's just pull words from witness. We know that first word is never empty if we decommit
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 482
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: we have to enforce a sequence of access to witness, so we always wait for code_word_0 to be
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 491
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if witness_1 wasn't in a circuit witness we conclude that it's the end of hash and perform finali
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 507
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// perform two writes. It's never a "pointer" type
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 535
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// even if we do not write in practice then we will never use next value too
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 549
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// mind endianess!
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 559
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// then conditionally form the second half of the block
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 563
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// padding of single byte of 1<<7 and some zeroes after, and interpret it as BE integer
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 565
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// last word is just number of bits. Note that we multiply u16 by 32*8 and it can not overflow u32
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 593
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// make it into uint256, and do not forget to ignore highest four bytes
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 620
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// finish
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 674
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Create a constraint system with proper configuration
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 771
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Create inputs for the inner function
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 789
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Run the inner function
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 800
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Check the corectness
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/code_unpacker_sha256/mod.rs`
**Line**: 879
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// mask it
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/fsm_input_output/mod.rs`
**Line**: 262
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: must be same sequence as in `flatten_as_variables`
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/fsm_input_output/mod.rs`
**Line**: 322
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// mask FSM part. Observable part is NEVER masked
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/fsm_input_output/mod.rs`
**Line**: 334
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// mask output. Observable output is zero is not the last indeed
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/fsm_input_output/mod.rs`
**Line**: 342
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and vice versa for FSM
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/fsm_input_output/mod.rs`
**Line**: 400
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we use length specialization here
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/fsm_input_output/mod.rs`
**Line**: 407
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// pad with zeroes
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/fsm_input_output/mod.rs`
**Line**: 409
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
let mut buffer_length = expected_length / AW;
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sort_decommittment_requests/mod.rs`
**Line**: 32
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// This is a sorter of logs that are kind-of "pure", F.g. event emission or L2 -> L1 messages.
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sort_decommittment_requests/mod.rs`
**Line**: 33
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Those logs do not affect a global state and may either be rolled back in full or not.
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sort_decommittment_requests/mod.rs`
**Line**: 34
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// We identify equality of logs using "timestamp" field that is a monotonic unique counter
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sort_decommittment_requests/mod.rs`
**Line**: 35
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// across the block
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sort_decommittment_requests/mod.rs`
**Line**: 51
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// as usual we assume that a caller of this fuunction has already split input queue,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sort_decommittment_requests/mod.rs`
**Line**: 52
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// so it can be comsumed in full
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sort_decommittment_requests/mod.rs`
**Line**: 54
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//use table
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sort_decommittment_requests/mod.rs`
**Line**: 79
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// passthrough must be trivial
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sort_decommittment_requests/mod.rs`
**Line**: 92
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// it must be trivial
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sort_decommittment_requests/mod.rs`
**Line**: 194
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// form the final state
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sort_decommittment_requests/mod.rs`
**Line**: 219
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// self-check
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sort_decommittment_requests/mod.rs`
**Line**: 274
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Simultaneously pop, prove sorting and resolve logic
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sort_decommittment_requests/mod.rs`
**Line**: 287
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we make encoding that is the same as defined for timestamped item
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sort_decommittment_requests/mod.rs`
**Line**: 304
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// check if keys are equal and check a value
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sort_decommittment_requests/mod.rs`
**Line**: 307
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// ensure sorting for uniqueness timestamp and rollback flag
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sort_decommittment_requests/mod.rs`
**Line**: 308
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// We know that timestamps are unique accross logs, and are also the same between write and rollback
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sort_decommittment_requests/mod.rs`
**Line**: 311
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// always ascedning
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sort_decommittment_requests/mod.rs`
**Line**: 316
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we get new hash then it my have a "first" marker
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sort_decommittment_requests/mod.rs`
**Line**: 323
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// otherwise it should have the same memory page
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sort_decommittment_requests/mod.rs`
**Line**: 335
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// decide if we should add the PREVIOUS into the queue
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sort_decommittment_requests/mod.rs`
**Line**: 339
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
record_to_add.is_first = Boolean::allocated_constant(cs, true); // we use convension to be easier co
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sort_decommittment_requests/mod.rs`
**Line**: 344
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// may be update the timestamp
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sort_decommittment_requests/mod.rs`
**Line**: 355
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if this circuit is the last one the queues must be empty and grand products must be equal
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sort_decommittment_requests/mod.rs`
**Line**: 360
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// finalization step - push the last one if necessary
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sort_decommittment_requests/mod.rs`
**Line**: 366
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
record_to_add.is_first = Boolean::allocated_constant(cs, true); // we use convension to be easier co
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sort_decommittment_requests/mod.rs`
**Line**: 382
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// LE packing so comparison is subtraction
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/sort_decommittment_requests/mod.rs`
**Line**: 497
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// add tables
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 92
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Secp256k1.p - 1 / 2
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 93
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffc2f - 0x1 / 0x2
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 96
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Decomposition constants
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 97
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Derived through algorithm 3.74 http://tomlr.free.fr/Math%E9matiques/Math%20Complete/Cryptography/
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 98
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: B2 == A1
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 107
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// assume that constructed field element is not zero
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 108
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if this is not satisfied - set the result to be F::one
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 124
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we still have to decompose it into u16 words
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 172
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we still have to decompose it into u16 words
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 207
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: caller must ensure that the field element is normalized, otherwise this will fail.
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 266
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Scalar decomposition
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 270
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// We take 8 non-zero limbs for the scalar (since it could be of any size), and 4 for B2
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 273
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// can not overflow u512
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 278
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// We take 8 non-zero limbs for the scalar (since it could be of any size), and 4 for B1
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 304
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we will need k1 and k2 to be < 2^128, so we can compare via subtraction
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 307
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// dbg!(k1.witness_hook(cs)());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 308
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// dbg!(k1_negated.witness_hook(cs)());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 317
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// dbg!(k2.witness_hook(cs)());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 318
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// dbg!(k2_negated.witness_hook(cs)());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 354
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// negated above to fit into range, we negate bases here
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 365
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now decompose every scalar we are interested in
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 375
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we do amortized double and add
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 394
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// dbg!(k1_window_idx.witness_hook(cs)());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 395
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// dbg!(k2_window_idx.witness_hook(cs)());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 396
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// dbg!(ignore_k1_part.witness_hook(cs)());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 397
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// dbg!(ignore_k2_part.witness_hook(cs)());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 430
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// dbg!(selected_k1_part_x.witness_hook(cs)());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 431
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// dbg!(selected_k1_part_y.witness_hook(cs)());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 438
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// let ((x, y), _) = acc.convert_to_affine_or_default(cs, Secp256Affine::zero());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 439
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// dbg!(x.witness_hook(cs)());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 440
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// dbg!(y.witness_hook(cs)());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 457
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we know that width is 128 bits, so just do BE decomposition and put into resulting array
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 468
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// special case
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 517
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
assert_eq!(base_canonical_limbs_canonical_limbs / 2, 8);
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 635
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// recid = (x_overflow ? 2 : 0) | (secp256k1_fe_is_odd(&r.y) ? 1 : 0)
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 636
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// The point X = (x, y) we are going to recover is not known at the start, but it is strongly relate
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 637
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// This is because x = r + kn for some integer k, where x is an element of the field F_q . In other 
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 639
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// For secp256k1 curve values of q and n are relatively close, that is,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 640
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// the probability of a random element of Fq being greater than n is about 1/{2^128}.
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 641
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// This in turn means that the overwhelming majority of r determine a unique x, however some of them
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 642
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// two: x = r and x = r + n. If x_overflow flag is set than x = r + n
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 652
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we handle x separately as it is the only element of base field of a curve (not a scalar field ele
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 653
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// check that x < q - order of base point on Secp256 curve
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 654
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if it is not actually the case - mask x to be zero
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 679
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// curve equation is y^2 = x^3 + b
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 680
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we compute t = r^3 + b and check if t is a quadratic residue or not.
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 681
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we do this by computing Legendre symbol (t, p) = t^[(p-1)/2] (mod p)
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 682
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//           p = 2^256 - 2^32 - 2^9 - 2^8 - 2^7 - 2^6 - 2^4 - 1
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 683
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// n = (p-1)/2 = 2^255 - 2^31 - 2^8 - 2^7 - 2^6 - 2^5 - 2^3 - 1
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 684
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we have to compute t^b = t^{2^255} / ( t^{2^31} * t^{2^8} * t^{2^7} * t^{2^6} * t^{2^5} * t^{2^3}
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 685
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if t is not a quadratic residue we return error and replace x by another value that will make
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 686
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// t = x^3 + b a quadratic residue
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 695
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if t is zero then just mask
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 698
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// array of powers of t of the form t^{2^i} starting from i = 0 to 255
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 715
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we can also reuse the same values to compute square root in case of p = 3 mod 4
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 716
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//           p = 2^256 - 2^32 - 2^9 - 2^8 - 2^7 - 2^6 - 2^4 - 1
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 717
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// n = (p+1)/4 = 2^254 - 2^30 - 2^7 - 2^6 - 2^5 - 2^4 - 2^2
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 737
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if lowest bit != parity bit, then we need conditionally select
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 749
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// unfortunately, if t is found to be a quadratic nonresidue, we can't simply let x to be zero,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 750
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// because then t_new = 7 is again a quadratic nonresidue. So, in this case we let x to be 9, then
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 751
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// t = 16 is a quadratic residue
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 761
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we recovered (x, y) using curve equation, so it's on curve (or was masked)
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 770
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we are going to compute the public key Q = (x, y) determined by the formula:
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 771
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Q = (s * X - hash * G) / r which is equivalent to r * Q = s * X - hash * G
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 783
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we do multiplication
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 852
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// digest is 32 bytes, but we need only 20 to recover address
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 853
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
digest_bytes[0..12].copy_from_slice(&[zero_u8; 12]); // empty out top bytes
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 920
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// it must be trivial
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 935
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// it must be trivial
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 1084
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// form the final state
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 1102
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// self-check
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 1265
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// let owned_cs = ReductionGate::<F, 4>::configure_for_cs(owned_cs, GatePlacementStrategy::UseSpecia
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 1295
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// let owned_cs = DotProductGate::<4>::configure_for_cs(owned_cs, GatePlacementStrategy::UseSpeciali
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 1311
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// add tables
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 1318
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// let table = create_naf_abs_div2_table();
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 1319
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// owned_cs.add_lookup_table::<NafAbsDiv2Table, 3>(table);
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 1321
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// let table = create_wnaf_decomp_table();
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 1322
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// owned_cs.add_lookup_table::<WnafDecompTable, 3>(table);
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 1436
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// let mut seed = Secp256Fr::from_str("1234567890").unwrap();
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 1437
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// dbg!(base);
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 1438
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// dbg!(base.mul(seed).into_affine());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 1726
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Create an r that is unrecoverable.
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 1749
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Construct a table of all combinations of correct and incorrect values
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 1750
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// for r, s, and digest.
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 1755
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// We ensure that there are no combinations where all correct items are chosen, so that we
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 1756
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// can consistently check for errors.
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 1789
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// As discussed on ethresearch forums, a caller may 'abuse' ecrecover in order to compute a
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 1790
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// secp256k1 ecmul in the EVM. This test compares the result of an ecrecover scalar mul with
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 1791
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// the output of a previously tested ecmul in the EVM.
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 1793
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// It works as follows: given a point x coordinate `r`, we set `s` to be `r * k` for some `k`.
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 1794
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// This then works out in the secp256k1 recover equation to create the equation
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 1796
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// performing a scalar multiplication.
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 1798
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// https://ethresear.ch/t/you-can-kinda-abuse-ecrecover-to-do-ecmul-in-secp256k1-today/2384
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 1804
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: This is essentially reducing a base field to a scalar field element. Due to the
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 1805
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// nature of the recovery equation turning into `(r, y) * r * k * inv(r, P)`, reducing r to
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 1806
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// a scalar value would yield the same result regardless.
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/new_optimized.rs`
**Line**: 1863
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Zero digest shouldn't give us an error
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 81
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// assume that constructed field element is not zero
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 82
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if this is not satisfied - set the result to be F::one
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 98
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we still have to decompose it into u16 words
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 145
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we still have to decompose it into u16 words
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 220
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// recid = (x_overflow ? 2 : 0) | (secp256k1_fe_is_odd(&r.y) ? 1 : 0)
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 221
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// The point X = (x, y) we are going to recover is not known at the start, but it is strongly relate
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 222
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// This is because x = r + kn for some integer k, where x is an element of the field F_q . In other 
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 224
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// For secp256k1 curve values of q and n are relatively close, that is,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 225
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// the probability of a random element of Fq being greater than n is about 1/{2^128}.
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 226
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// This in turn means that the overwhelming majority of r determine a unique x, however some of them
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 227
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// two: x = r and x = r + n. If x_overflow flag is set than x = r + n
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 237
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we handle x separately as it is the only element of base field of a curve (not a scalar field ele
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 238
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// check that x < q - order of base point on Secp256 curve
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 239
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if it is not actually the case - mask x to be zero
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 254
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NB: although it is not strictly an exception we also assume that hash is never zero as field elem
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 259
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// curve equation is y^2 = x^3 + b
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 260
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we compute t = r^3 + b and check if t is a quadratic residue or not.
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 261
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we do this by computing Legendre symbol (t, p) = t^[(p-1)/2] (mod p)
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 262
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//           p = 2^256 - 2^32 - 2^9 - 2^8 - 2^7 - 2^6 - 2^4 - 1
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 263
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// n = (p-1)/2 = 2^255 - 2^31 - 2^8 - 2^7 - 2^6 - 2^5 - 2^3 - 1
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 264
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we have to compute t^b = t^{2^255} / ( t^{2^31} * t^{2^8} * t^{2^7} * t^{2^6} * t^{2^5} * t^{2^3}
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 265
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if t is not a quadratic residue we return error and replace x by another value that will make
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 266
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// t = x^3 + b a quadratic residue
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 275
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if t is zero then just mask
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 278
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// array of powers of t of the form t^{2^i} starting from i = 0 to 255
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 295
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we can also reuse the same values to compute square root in case of p = 3 mod 4
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 296
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//           p = 2^256 - 2^32 - 2^9 - 2^8 - 2^7 - 2^6 - 2^4 - 1
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 297
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// n = (p+1)/4 = 2^254 - 2^30 - 2^7 - 2^6 - 2^5 - 2^4 - 2^2
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 313
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if lowest bit != parity bit, then we need conditionally select
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 325
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// unfortunately, if t is found to be a quadratic nonresidue, we can't simply let x to be zero,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 326
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// because then t_new = 7 is again a quadratic nonresidue. So, in this case we let x to be 9, then
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 327
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// t = 16 is a quadratic residue
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 337
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we recovered (x, y) using curve equation, so it's on curve (or was masked)
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 366
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we are going to compute the public key Q = (x, y) determined by the formula:
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 367
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Q = (s * X - hash * G) / r which is equivalent to r * Q = s * X - hash * G
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 368
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// current implementation of point by scalar multiplications doesn't support multiplication by zero
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 369
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// so we check that all s, r, hash are not zero (as FieldElements):
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 370
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if any of them is zero we reject the signature and in circuit itself replace all zero variables b
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 374
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we do multiexponentiation
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 378
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we should start from MSB, double the accumulator, then conditionally add
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 421
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// digest is 32 bytes, but we need only 20 to recover address
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 422
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
digest_bytes[0..12].copy_from_slice(&[zero_u8; 12]); // empty out top bytes
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 491
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// it must be trivial
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 506
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// it must be trivial
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 639
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// form the final state
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 657
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// self-check
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 806
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// let owned_cs = ReductionGate::<F, 4>::configure_for_cs(owned_cs, GatePlacementStrategy::UseSpecia
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 832
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// let owned_cs = DotProductGate::<4>::configure_for_cs(owned_cs, GatePlacementStrategy::UseSpeciali
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/baseline.rs`
**Line**: 848
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// add tables
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/mod.rs`
**Line**: 35
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// characteristics of the base field for secp curve
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/mod.rs`
**Line**: 37
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// order of group of points for secp curve
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/mod.rs`
**Line**: 39
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// some affine point
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ecrecover/mod.rs`
**Line**: 61
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// re-exports for integration
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/linear_hasher/input.rs`
**Line**: 87
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     serialize = "CircuitQueueRawWitness<F, LogQuery<F>, 4, LOG_QUERY_PACKED_WIDTH>: serde::Serial
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/linear_hasher/input.rs`
**Line**: 90
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     deserialize = "CircuitQueueRawWitness<F, LogQuery<F>, 4, LOG_QUERY_PACKED_WIDTH>: serde::de::
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/linear_hasher/mod.rs`
**Line**: 64
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// only 1 instance of the circuit here for now
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/linear_hasher/mod.rs`
**Line**: 69
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// it must be trivial
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/linear_hasher/mod.rs`
**Line**: 82
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we do not serialize length because it's recalculatable in L1
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/linear_hasher/mod.rs`
**Line**: 126
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// absorb if we are not done yet
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/linear_hasher/mod.rs`
**Line**: 137
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// in case if we do last round
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/linear_hasher/mod.rs`
**Line**: 146
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// unreachable, but we set it for completeness
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/linear_hasher/mod.rs`
**Line**: 155
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// absorb if it's the last round
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/linear_hasher/mod.rs`
**Line**: 177
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// squeeze
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/linear_hasher/mod.rs`
**Line**: 195
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// self-check
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 81
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// from PrecompileCallABI
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 136
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we just need to put a marker after the current fill value
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 145
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if crate::config::CIRCUIT_VERSOBE {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 146
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     dbg!(result.witness_hook(cs)().unwrap());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 201
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we can have a degenerate case when queue is empty, but it's a first circuit in the queue,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 202
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// so we taken default FSM state that has state.read_precompile_call = true;
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 237
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// main work cycle
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 267
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we are in a proper state then get the ABI from the queue
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 290
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now compute some parameters that describe the call itself
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 310
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// also set timestamps
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 318
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// timestamps have large space, so this can be expected
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 328
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and do some work! keccak256 is expensive
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 330
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we just have read a precompile call with zero length input, we want to perform only one paddin
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 337
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// otherwise we proceed with reading the input and follow the logic of padding round based on the pr
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 338
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// padding round needed/not needed in the params
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 362
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Now perform few memory queries to read content
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 371
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// conditionally reset state. Keccak256 empty state is just all 0s
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 404
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// logic in short - we always try to read from memory into buffer,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 405
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and every time execute 1 keccak256 round function
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 407
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we have a little more complex logic here, but it's homogenious
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 456
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// perform read
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 459
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// update state variables
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 482
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// update if we do not read
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 485
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// fill the buffer
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 487
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if crate::config::CIRCUIT_VERSOBE {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 488
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     dbg!(be_bytes.witness_hook(cs)().map(|el| hex::encode(&el)));
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 506
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now actually run keccak permutation
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 508
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we either mask for padding, or mask in full if it's full padding round
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 517
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: we have already precomputed if we will need a full padding round, so we just take something
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 518
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and run keccak premutation
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 559
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we absorb nothing, and "finalize" will take care of the rest
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 593
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// manually absorb and run round function
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 598
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// dbg!(absorbed_and_padded.witness_hook(cs)());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 599
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// dbg!(state.padding_round.witness_hook(cs)());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 637
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// perform write
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 642
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// update FSM state
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 651
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we need to decide on full padding round
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 663
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// otherwise we just continue
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 712
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// it must be trivial
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 727
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// it must be trivial
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 772
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// form the final state
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 788
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// self-check
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 814
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
< (keccak256::KECCAK_RATE_BYTES / keccak256::BYTES_PER_WORD)
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 836
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// copy back
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/mod.rs`
**Line**: 957
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// add tables for keccak
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/input.rs`
**Line**: 134
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     serialize = "CircuitQueueRawWitness<F, LogQuery<F>, 4, LOG_QUERY_PACKED_WIDTH>: serde::Serial
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/input.rs`
**Line**: 137
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     deserialize = "CircuitQueueRawWitness<F, LogQuery<F>, 4, LOG_QUERY_PACKED_WIDTH>: serde::de::
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/mod.rs`
**Line**: 69
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
< (keccak256::KECCAK_RATE_BYTES / keccak256::BYTES_PER_WORD)
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/mod.rs`
**Line**: 84
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we do not write then discard
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/mod.rs`
**Line**: 169
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
bias: Variable, // any variable that has to be resolved BEFORE executing witness query
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/mod.rs`
**Line**: 225
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
bias: Variable, // any variable that has to be resolved BEFORE executing witness query
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/mod.rs`
**Line**: 335
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// it must be trivial
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/mod.rs`
**Line**: 404
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if this is the last executing cycle - we do not start the parsing of the new element:
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/mod.rs`
**Line**: 405
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// instead we either complete the second iter of processing of the last element or simply do nothing
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/mod.rs`
**Line**: 413
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// last iteration can never pop, and is never "read"
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/mod.rs`
**Line**: 435
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we can decompose everythin right away
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/mod.rs`
**Line**: 445
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// update current key
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/mod.rs`
**Line**: 448
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// get path bits
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/mod.rs`
**Line**: 454
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// determine whether we need to increment enumeration index
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/mod.rs`
**Line**: 460
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// update index over which we work
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/mod.rs`
**Line**: 473
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// use next enumeration index
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/mod.rs`
**Line**: 486
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// index is done, now we need merkle path
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/mod.rs`
**Line**: 499
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we read then we save and use it for write too
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/mod.rs`
**Line**: 504
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
let src_bytes = src.to_le_bytes(cs); // NOP
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/mod.rs`
**Line**: 512
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we just processed a value from the queue then save it
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/mod.rs`
**Line**: 519
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we have write stage in progress then use saved value as the one we will use for path
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/mod.rs`
**Line**: 527
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we need to serialize leaf index as 8 bytes
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/mod.rs`
**Line**: 536
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we have everything to update state diff data
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/mod.rs`
**Line**: 552
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: we need READ index, before updating
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/mod.rs`
**Line**: 592
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// in case of read: merkle_root == computed_merkle_root == new_merkle_root
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/mod.rs`
**Line**: 593
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// new_merkle_root = select(if is_write: then new_merkle_root else computed_merkle_root);
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/mod.rs`
**Line**: 594
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// so we first compute merkle_root - either the old one or the selected one and then enforce equalit
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/mod.rs`
**Line**: 596
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// update if we write
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/mod.rs`
**Line**: 603
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// otherwise enforce equality
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/mod.rs`
**Line**: 613
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// update our accumulator
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/mod.rs`
**Line**: 615
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we use keccak256 here because it's same table structure
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/mod.rs`
**Line**: 622
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// absorb and run permutation
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/mod.rs`
**Line**: 624
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we do not write here anyway
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/mod.rs`
**Line**: 638
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// toggle control flags
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/mod.rs`
**Line**: 640
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// cur elem is processed only in the case second iter in progress or rw_flag is false;
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/mod.rs`
**Line**: 668
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we need to run padding and one more permutation for final output
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/mod.rs`
**Line**: 681
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// squeeze
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_application/mod.rs`
**Line**: 710
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// self-check
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ram_permutation/mod.rs`
**Line**: 57
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// passthrought must be trivial
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ram_permutation/mod.rs`
**Line**: 84
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// passthrought must be trivial
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ram_permutation/mod.rs`
**Line**: 177
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// form the final state
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ram_permutation/mod.rs`
**Line**: 250
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// this is an exotic way so synchronize popping from both queues
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ram_permutation/mod.rs`
**Line**: 251
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// in asynchronous resolution
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ram_permutation/mod.rs`
**Line**: 255
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we do not need any information about unsorted element other than it's encoding
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ram_permutation/mod.rs`
**Line**: 259
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// check non-deterministic writes
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ram_permutation/mod.rs`
**Line**: 292
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// check RAM ordering
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ram_permutation/mod.rs`
**Line**: 294
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// either continue the argument or do nothing
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ram_permutation/mod.rs`
**Line**: 303
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// ensure sorting
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ram_permutation/mod.rs`
**Line**: 307
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we can not have previous sorting key even to be >= than our current key
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ram_permutation/mod.rs`
**Line**: 325
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// check uninit read
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ram_permutation/mod.rs`
**Line**: 333
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we only have a difference in these flags at the first step
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ram_permutation/mod.rs`
**Line**: 338
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// check standard RW validity
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ram_permutation/mod.rs`
**Line**: 342
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// see if we continue the argument then all our checks should be valid,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ram_permutation/mod.rs`
**Line**: 343
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// otherwise only read uninit should be enforced
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ram_permutation/mod.rs`
**Line**: 345
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we start a fresh argument then our comparison
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ram_permutation/mod.rs`
**Line**: 353
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// check standard RW validity, but it can break if we are at the very start
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ram_permutation/mod.rs`
**Line**: 365
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we did pop then accumulate to grand product
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/ram_permutation/mod.rs`
**Line**: 496
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// add tables
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/secp256r1_verify/baseline.rs`
**Line**: 111
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we use non-compressed point, so we:
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/secp256r1_verify/baseline.rs`
**Line**: 115
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we check ranges upfront. We only need to check <modulus, and conversion functions will perform ma
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/secp256r1_verify/baseline.rs`
**Line**: 155
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// perform on-curve check
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/secp256r1_verify/baseline.rs`
**Line**: 171
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we can mask point to ensure that our arithmetic formulas work
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/secp256r1_verify/baseline.rs`
**Line**: 176
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// this always exists (0 was an exception and was masked)
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/secp256r1_verify/baseline.rs`
**Line**: 184
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we do multiplication
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/secp256r1_verify/baseline.rs`
**Line**: 185
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// it's safe since we checked not-on-curve above
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/secp256r1_verify/baseline.rs`
**Line**: 237
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now compare mod n. For that we go out of limbs and back
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/secp256r1_verify/baseline.rs`
**Line**: 301
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// it must be trivial
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/secp256r1_verify/baseline.rs`
**Line**: 316
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// it must be trivial
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/secp256r1_verify/baseline.rs`
**Line**: 446
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// form the final state
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/secp256r1_verify/baseline.rs`
**Line**: 464
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// self-check
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/secp256r1_verify/baseline.rs`
**Line**: 490
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// create precomputed table of size 1<<4 - 1
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/secp256r1_verify/baseline.rs`
**Line**: 491
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// there is no 0 * P in the table, we will handle it below
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/secp256r1_verify/baseline.rs`
**Line**: 497
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// 2P, 3P, ...
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/secp256r1_verify/baseline.rs`
**Line**: 504
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now decompose every scalar we are interested in
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/secp256r1_verify/baseline.rs`
**Line**: 513
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we just do double and add
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/secp256r1_verify/baseline.rs`
**Line**: 550
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we know that width is 128 bits, so just do BE decomposition and put into resulting array
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/secp256r1_verify/baseline.rs`
**Line**: 699
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// add tables
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/secp256r1_verify/mod.rs`
**Line**: 34
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// characteristics of the base field for secp curve
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/secp256r1_verify/mod.rs`
**Line**: 36
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// order of group of points for secp curve
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/secp256r1_verify/mod.rs`
**Line**: 38
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// some affine point
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/secp256r1_verify/mod.rs`
**Line**: 60
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// re-exports for integration
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/test_input.rs`
**Line**: 11
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// This witness input is generated from the old test harness, and remodeled to work in the current t
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/input.rs`
**Line**: 33
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// FSM
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 44
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we make a generation aware memory that store all the old and new values
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 45
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// for a current storage cell. There are largely 3 possible sequences that we must be aware of
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 50
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// We use extra structure with timestamping. Even though we DO have
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 51
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// timestamp field in LogQuery, such timestamp is the SAME
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 52
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// for "forward" and "rollback" items, while we do need to have them
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 53
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// on different timestamps
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 78
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// LogQuery encoding leaves last variable as < 8 bits value
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 83
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Original encoding at index 19 is only 8 bits
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 115
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE(shamatar) Using CSAllocatableExt causes cyclic dependency here, so we use workaround
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 130
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we should be able to allocate without knowing values yet
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 211
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// passthrought must be trivial
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 226
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// same logic from sorted
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 281
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// passthrought must be trivial
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 334
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// get challenges for permutation argument
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 441
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// form the input/output
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 498
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// dbg!(compact_form.create_witness());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 574
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we can recreate it here, there are two cases:
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 580
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we simultaneously pop, accumulate partial product,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 581
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and decide whether or not we should move to the next cell
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 583
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// to ensure uniqueness we place timestamps in a addition to the original values encoding access loc
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 587
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// increment it immediatelly
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 603
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: we do not need to check shard_id of unsorted item because we can just check it on sorted it
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 635
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now resolve a logic about sorting itself
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 638
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// ensure sorting. Check that previous key < this key
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 645
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if keys are the same then timestamps are sorted
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 647
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// enforce if keys are the same and not trivial
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 651
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we follow the procedure:
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 652
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if keys are different then we finish with a previous one and update parameters
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 653
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// else we just update parameters
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 655
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if new cell
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 659
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// it must always be true if we start and if we have items to work with
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 663
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// finish with the old one
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 664
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if somewhere along the way we did encounter a read at rollback depth zero (not important if there
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 665
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and if current rollback depth is 0 then we MUST issue a read
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 669
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// there may be a situation when as a result of sequence of writes
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 670
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// storage slot is CLAIMED to be unchanged. There are two options:
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 672
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//   In this case we used a temporary value, and the fact that the last action is rollback
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 673
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//   all the way to the start (to depth 0), we are not interested in what was an initial value
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 675
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//   In this case we would not need to write IF prover is honest and provides a true witness to "rea
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 676
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//   field at the first write. But we can not rely on this and have to check this fact!
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 699
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we did only writes and rollbacks then we don't need to update
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 710
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and update as we switch to the new cell with extra logic
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 718
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// re-update
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 745
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we have new non-trivial
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 746
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and if it's read then it's definatelly at depth 0
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 756
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if same cell - update
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 766
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// update rollback depth the is a result of this action
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 784
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// check consistency
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 787
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we ALWAYS ensure read consistency on write (but not rollback) and on plain read
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 792
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// decide to update
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 818
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we definately read non-trivial, and that is on depth 0, so set to true
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 828
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// always update counters
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 836
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// finalization step - out of cycle, and only if we are done just yet
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 840
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// cell state is final
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 864
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we did only writes and rollbacks then we don't need to update
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 873
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// reset flag to match simple witness generation convensions
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 883
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// output our FSM values
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 904
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// LE packing so comparison is subtraction
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 923
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
/// Check that a == b and a > b by performing a long subtraction b - a with borrow.
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 924
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
/// Both a and b are considered as least significant word first
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 954
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// use boojum::cs::EmptyToolbox;
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/storage_validity_by_grand_product/mod.rs`
**Line**: 1050
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// add tables
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/input.rs`
**Line**: 31
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// FSM
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/input.rs`
**Line**: 49
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
pub previous_packed_key: [UInt32<F>; TRANSIENT_STORAGE_VALIDITY_CHECK_PACKED_KEY_LENGTH], // it capt
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 6
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// mod test_input;
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 37
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we make a generation aware memory that store all the old and new values
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 38
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// for a current storage cell. There are largely 3 possible sequences that we must be aware of
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 43
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// We use extra structure with timestamping. Even though we DO have
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 44
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// timestamp field in LogQuery, such timestamp is the SAME
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 45
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// for "forward" and "rollback" items, while we do need to have them
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 46
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// on different timestamps
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 96
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// passthrought must be trivial
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 111
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// same logic from sorted
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 166
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// passthrought must be trivial
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 230
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// there is no code at address 0 in our case, so we can formally use it for all the purposes
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 283
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// form the input/output
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 308
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// dbg!(compact_form.create_witness());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 374
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we simultaneously pop, accumulate partial product,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 375
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and decide whether or not we should move to the next cell
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 377
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// to ensure uniqueness we place timestamps in a addition to the original values encoding access loc
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 381
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// increment it immediatelly
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 398
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: we do not need to check shard_id of unsorted item because we can just check it on sorted it
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 426
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now resolve a logic about sorting itself
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 435
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// ensure sorting. Check that previous key < this key
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 441
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if keys are the same then timestamps are sorted
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 443
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// enforce if keys are the same and not trivial
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 447
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we follow the procedure:
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 448
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if keys are different then we finish with a previous one and update parameters
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 449
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// else we just update parameters
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 456
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if new cell
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 459
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// it must always be true if we start
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 464
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we just discard the old one and that's it
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 466
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// enforce that we read 0 always for new cell
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 469
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and update as we switch to the new cell with extra logic
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 477
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// update current value
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 498
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if same cell - update
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 508
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// update rollback depth the is a result of this action
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 526
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// check consistency
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 529
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we ALWAYS ensure read consistency on write (but not rollback) and on plain read
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 534
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// decide to update
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 553
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we rolled back all the way - check if read value is 0 again
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 564
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// always update counters
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 569
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// there is no post-processing or finalization
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 571
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// output our FSM values
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 592
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// LE packing so comparison is subtraction. Since every TX is independent it's just a part of key
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 613
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// mod tests {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 614
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     use super::*;
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 615
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     use boojum::algebraic_props::poseidon2_parameters::Poseidon2GoldilocksExternalMatrix;
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 617
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     use boojum::cs::traits::gate::GatePlacementStrategy;
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 618
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     use boojum::cs::CSGeometry;
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 619
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     // use boojum::cs::EmptyToolbox;
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 620
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     use boojum::cs::*;
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 621
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     use boojum::field::goldilocks::GoldilocksField;
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 622
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     use boojum::gadgets::tables::*;
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 623
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     use boojum::implementations::poseidon2::Poseidon2Goldilocks;
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 624
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     use boojum::worker::Worker;
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 625
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     use ethereum_types::{Address, U256};
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 627
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     type F = GoldilocksField;
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 628
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     type P = GoldilocksField;
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 630
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     use boojum::cs::cs_builder::*;
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 632
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     fn configure<
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 633
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         T: CsBuilderImpl<F, T>,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 634
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         GC: GateConfigurationHolder<F>,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 635
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         TB: StaticToolboxHolder,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 637
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         builder: CsBuilder<T, F, GC, TB>,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 639
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let owned_cs = builder;
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 640
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let owned_cs = owned_cs.allow_lookup(
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 641
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             LookupParameters::UseSpecializedColumnsWithTableIdAsConstant {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 642
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//                 width: 3,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 643
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//                 num_repetitions: 8,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 644
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//                 share_table_id: true,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 647
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let owned_cs = ConstantsAllocatorGate::configure_builder(
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 648
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             owned_cs,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 649
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             GatePlacementStrategy::UseGeneralPurposeColumns,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 651
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let owned_cs = FmaGateInBaseFieldWithoutConstant::configure_builder(
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 652
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             owned_cs,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 653
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             GatePlacementStrategy::UseGeneralPurposeColumns,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 655
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let owned_cs = ReductionGate::<F, 4>::configure_builder(
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 656
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             owned_cs,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 657
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             GatePlacementStrategy::UseGeneralPurposeColumns,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 659
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let owned_cs = BooleanConstraintGate::configure_builder(
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 660
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             owned_cs,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 661
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             GatePlacementStrategy::UseGeneralPurposeColumns,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 663
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let owned_cs = UIntXAddGate::<32>::configure_builder(
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 664
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             owned_cs,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 665
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             GatePlacementStrategy::UseGeneralPurposeColumns,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 667
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let owned_cs = UIntXAddGate::<16>::configure_builder(
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 668
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             owned_cs,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 669
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             GatePlacementStrategy::UseGeneralPurposeColumns,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 671
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let owned_cs = SelectionGate::configure_builder(
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 672
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             owned_cs,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 673
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             GatePlacementStrategy::UseGeneralPurposeColumns,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 675
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let owned_cs = ZeroCheckGate::configure_builder(
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 676
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             owned_cs,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 677
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             GatePlacementStrategy::UseGeneralPurposeColumns,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 678
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             false,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 680
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let owned_cs = DotProductGate::<4>::configure_builder(
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 681
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             owned_cs,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 682
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             GatePlacementStrategy::UseGeneralPurposeColumns,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 684
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let owned_cs =
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 685
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             MatrixMultiplicationGate::<F, 12, Poseidon2GoldilocksExternalMatrix>::configure_build
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 686
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//                 owned_cs,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 687
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//                 GatePlacementStrategy::UseGeneralPurposeColumns,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 689
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let owned_cs =
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 690
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             NopGate::configure_builder(owned_cs, GatePlacementStrategy::UseGeneralPurposeColumns)
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 692
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         owned_cs
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 696
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     fn test_storage_validity_circuit() {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 697
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let geometry = CSGeometry {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 698
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             num_columns_under_copy_permutation: 100,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 699
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             num_witness_columns: 0,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 700
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             num_constant_columns: 8,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 701
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             max_allowed_constraint_degree: 4,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 704
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         use boojum::config::DevCSConfig;
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 705
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         use boojum::cs::cs_builder_reference::*;
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 707
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let builder_impl =
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 708
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             CsReferenceImplementationBuilder::<F, P, DevCSConfig>::new(geometry, 1 << 26, 1 << 20
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 709
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         use boojum::cs::cs_builder::new_builder;
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 710
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let builder = new_builder::<_, F>(builder_impl);
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 712
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let builder = configure(builder);
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 713
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let mut owned_cs = builder.build(());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 715
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         // add tables
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 716
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let table = create_xor8_table();
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 717
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         owned_cs.add_lookup_table::<Xor8Table, 3>(table);
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 719
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let cs = &mut owned_cs;
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 721
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let lhs = [Num::allocated_constant(cs, F::from_nonreduced_u64(1));
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 722
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             DEFAULT_NUM_PERMUTATION_ARGUMENT_REPETITIONS];
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 723
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let rhs = [Num::allocated_constant(cs, F::from_nonreduced_u64(1));
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 724
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             DEFAULT_NUM_PERMUTATION_ARGUMENT_REPETITIONS];
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 726
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let execute = Boolean::allocated_constant(cs, true);
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 727
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let mut original_queue = StorageLogQueue::<F, Poseidon2Goldilocks>::empty(cs);
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 728
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let unsorted_input = test_input::generate_test_input_unsorted(cs);
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 729
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         for el in unsorted_input {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 730
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             original_queue.push(cs, el, execute);
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 733
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let mut intermediate_sorted_queue = CircuitQueue::empty(cs);
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 734
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let sorted_input = test_input::generate_test_input_sorted(cs);
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 735
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         for el in sorted_input {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 736
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             intermediate_sorted_queue.push(cs, el, execute);
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 739
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let mut sorted_queue = StorageLogQueue::empty(cs);
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 741
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let is_start = Boolean::allocated_constant(cs, true);
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 742
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let cycle_idx = UInt32::allocated_constant(cs, 0);
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 743
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let round_function = Poseidon2Goldilocks;
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 744
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let fs_challenges = crate::utils::produce_fs_challenges::<
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 745
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             F,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 746
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             _,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 747
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             Poseidon2Goldilocks,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 748
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             QUEUE_STATE_WIDTH,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 750
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             DEFAULT_NUM_PERMUTATION_ARGUMENT_REPETITIONS,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 752
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             cs,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 753
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             original_queue.into_state().tail,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 754
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             intermediate_sorted_queue.into_state().tail,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 757
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let previous_packed_key = [UInt32::allocated_constant(cs, 0); PACKED_KEY_LENGTH];
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 758
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let previous_key = UInt256::allocated_constant(cs, U256::default());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 759
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let previous_address = UInt160::allocated_constant(cs, Address::default());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 760
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let previous_timestamp = UInt32::allocated_constant(cs, 0);
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 761
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let this_cell_has_explicit_read_and_rollback_depth_zero =
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 762
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             Boolean::allocated_constant(cs, false);
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 763
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let this_cell_base_value = UInt256::allocated_constant(cs, U256::default());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 764
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let this_cell_current_value = UInt256::allocated_constant(cs, U256::default());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 765
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let this_cell_current_depth = UInt32::allocated_constant(cs, 0);
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 766
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let shard_id_to_process = UInt8::allocated_constant(cs, 0);
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 767
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let limit = 16;
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 769
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         sort_and_deduplicate_storage_access_inner(
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 770
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             cs,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 771
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             lhs,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 772
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             rhs,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 776
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             is_start,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 777
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             cycle_idx,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 778
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             fs_challenges,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 779
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             previous_packed_key,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 780
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             previous_key,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 781
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             previous_address,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 782
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             previous_timestamp,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 783
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             this_cell_has_explicit_read_and_rollback_depth_zero,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 784
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             this_cell_base_value,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 785
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             this_cell_current_value,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 786
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             this_cell_current_depth,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 787
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             shard_id_to_process,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 788
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             limit,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 791
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         cs.pad_and_shrink();
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 792
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let worker = Worker::new();
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 793
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         let mut owned_cs = owned_cs.into_assembly();
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 794
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         owned_cs.print_gate_stats();
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/transient_storage_validity_by_grand_product/mod.rs`
**Line**: 795
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         assert!(owned_cs.check_if_satisfied(&worker));
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/pre_state.rs`
**Line**: 69
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// create a draft candidate for next VM state, as well as all the data required for
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/pre_state.rs`
**Line**: 70
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// opcodes to proceed
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/pre_state.rs`
**Line**: 97
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: even if we have pending exception, we should still read and cache opcode to avoid caching p
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/pre_state.rs`
**Line**: 100
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// take down the flag
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/pre_state.rs`
**Line**: 126
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// In normal execution if we do not skip cycle then we read opcode based on PC and page.
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/pre_state.rs`
**Line**: 127
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// If we have pending exception we should do it too in case if next PC hits the cache
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/pre_state.rs`
**Line**: 129
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and in addition if we did finish execution then we never care and cleanup
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/pre_state.rs`
**Line**: 140
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// precompute timestamps
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/pre_state.rs`
**Line**: 156
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we can hardly make a judgement of using or not this sponge
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/pre_state.rs`
**Line**: 157
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// for optimization purposes, so we will assume that we always run it
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/pre_state.rs`
**Line**: 171
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// update current state
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/pre_state.rs`
**Line**: 182
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// subpc is 2 bits, so it's a range from 0 to 3. 1..=3 are bitspread via the table
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/pre_state.rs`
**Line**: 185
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// default one is one corresponding to the "highest" bytes in 32 byte word in our BE machine
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/pre_state.rs`
**Line**: 215
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// mask if we would be ok with NOPing. This masks a full 8-byte opcode, and not properties bitspread
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/pre_state.rs`
**Line**: 216
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// We mask if this cycle is just NOPing till the end of circuit
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/pre_state.rs`
**Line**: 218
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we are not pending, and we have an exception to run - run it
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/pre_state.rs`
**Line**: 221
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// update super_pc and code words if we did read
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/pre_state.rs`
**Line**: 223
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// always update code page
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/pre_state.rs`
**Line**: 241
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
); // may be it can be unconditional
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/pre_state.rs`
**Line**: 243
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// update timestamp
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/pre_state.rs`
**Line**: 280
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// decoded opcode and current (yet dirty) ergs left should be passed into the opcode,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/pre_state.rs`
**Line**: 281
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// but by default we set it into context that is true for most of the opcodes
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/pre_state.rs`
**Line**: 288
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we did all the masking and "INVALID" opcode must never happed
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/pre_state.rs`
**Line**: 299
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now read source operands
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/pre_state.rs`
**Line**: 300
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// select low part of the registers
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/pre_state.rs`
**Line**: 370
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// perform actual read
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/pre_state.rs`
**Line**: 391
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// update current state
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/pre_state.rs`
**Line**: 395
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// select source0 and source1
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/pre_state.rs`
**Line**: 399
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// select if it was reg
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/pre_state.rs`
**Line**: 405
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// select if it was imm
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/pre_state.rs`
**Line**: 412
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// form an intermediate state to process the opcodes over it
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/pre_state.rs`
**Line**: 415
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// swap operands
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/pre_state.rs`
**Line**: 454
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Potentially erase fat pointer data if opcode shouldn't take pointers and we're not in kernel
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/pre_state.rs`
**Line**: 455
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// mode
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/pre_state.rs`
**Line**: 476
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// We erase fat pointer data from src1 if it exists in non-kernel mode
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 45
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// first we create a pre-state
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 50
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// synchronization point
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 52
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// dbg!(_current_state);
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 69
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// synchronization point
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 75
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// then we apply each opcode and accumulate state diffs
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 164
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and finally apply state diffs
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 174
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// potentially we can have registers that update DST0 as memory location,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 175
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// so we choose only cases where it's indeed into memory.
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 176
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// It is only a possibility for now. Later we will predicate it based on the
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 177
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// decoded opcode properties
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 180
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// select dst0 and dst1 values
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 185
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// for DST0 it's possible to have opcode-constrainted updates only into registers
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 206
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Safety: we know by orthogonality of opcodes that boolean selectors in our iterators form either a
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 207
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// or an empty mask. So we can use unchecked casts below. Even if none of the bits is set (like in N
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 208
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// it's not a problem because in the same situation we will not have an update of register/memory an
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 262
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// We know that UMA opcodes (currently by design) are not allowed to write dst argument into memory
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 263
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// in any form, so if we do the write here we always base on the state of memory from prestate
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 289
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// update tail in next state candidate
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 293
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if dst0 is not in memory then update
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 299
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// do at once for dst0 and dst1
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 301
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// case when we want to update DST0 from potentially memory-writing opcodes,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 302
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// but we address register in fact
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 317
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// We should update registers, and the only "exotic" case is if someone tries to put
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 318
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// dst0 and dst1 in the same location.
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 320
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Note that "register" is a "wide" structure, so it doesn't benefit too much from
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 321
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// multiselect, and we can just do a sequence, that
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 322
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we update each register as being:
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 327
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// outer cycle is over ALL REGISTERS
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 333
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// form an iterator for all possible candidates
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 335
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// dst1 is always register
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 338
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// unfortunately we can not use iter chaining here due to syntax constraint
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 355
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// then chain all specific register updates. Opcodes that produce specific updates do not make non-s
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 356
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// so we just place them along with dst0
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 363
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// chain removal of pointer markers at once. Same, can be placed into dst0
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 376
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// chain zeroing at once. Same, can be placed into dst0
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 390
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Safety: our update flags are preconditioned by the applicability of the opcodes, and if opcode
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 391
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// updates specific registers it does NOT write using "normal" dst0/dst1 addressing, so our mask
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 392
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// is indeed a bitmask or empty
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 394
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// as dst0
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 412
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now as dst1
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 429
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// for registers we just use parallel select, that has the same efficiency as multiselect,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 430
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// because internally it's [UInt32<F>; 8]
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 441
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// apply smaller changes to VM state, such as ergs left, etc
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 443
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// PC
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 453
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Ergs
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 471
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Pubdata revert counter at the global state
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 477
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Tx number in block
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 483
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Page counter
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 486
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Context value
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 492
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// pubdata counters
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 528
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// global pubdata reverts counter can not be < 0
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 535
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Heap limit
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 553
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Axu heap limit
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 571
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// variable queue states
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 573
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Memory due to UMA
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 582
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// decommittment due to far call or log.decommit
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 595
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// forward storage log
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 618
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// rollback log head(!)
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 651
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// flags
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 657
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and now we either replace or not the callstack in full
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 663
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// other state parts
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 667
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// conditional u32 range checks. All of those are of the fixed length per opcode, so we just select
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 680
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// add/sub relations
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 719
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we can enforce sponges. There are only 2 outcomes
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 722
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// in parallel opcodes either
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 723
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// - do not use sponges and only rely on src0/dst0
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 724
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// - can not have src0/dst0 in memory, but use sponges (UMA, near_call, far call, ret)
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 741
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we can conditionally select
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 765
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we can conditionally select
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 793
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we can conditionally select
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 822
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// ensure that we selected everything
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 828
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// dbg!(new_state.memory_queue_state.witness_hook(&*cs)().unwrap());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 829
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// dbg!(new_state.memory_queue_length.witness_hook(&*cs)().unwrap());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 831
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// actually enforce_sponges
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 836
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// synchronization point
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 839
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// dbg!(_wit.memory_queue_length);
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 897
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// absorb by replacement
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 918
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// create absorbed initial state
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 937
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// for all reasonable execution traces it's fine
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/cycle.rs`
**Line**: 955
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we should assemble all the dependencies here, and we will use AllocateExt here
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/loading.rs`
**Line**: 26
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// first create the context
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/loading.rs`
**Line**: 65
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
ctx.saved_context.caller = UInt160::zero(cs); // is called from nowhere
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/loading.rs`
**Line**: 67
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// circuit specific bit
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/loading.rs`
**Line**: 71
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// mark as kernel
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/loading.rs`
**Line**: 74
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// bootloader should not pay for resizes
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/loading.rs`
**Line**: 80
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now push that to the callstack, manually
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/loading.rs`
**Line**: 88
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
let empty_entry_encoding = empty_entry.saved_context.encode(cs); // only saved part
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/loading.rs`
**Line**: 96
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// absorb by replacement
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/loading.rs`
**Line**: 128
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// memory
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/loading.rs`
**Line**: 131
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// code decommittments
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/loading.rs`
**Line**: 134
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// rest
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/loading.rs`
**Line**: 136
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// timestamp and global counters
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/loading.rs`
**Line**: 142
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we also FORMALLY mark r1 as "pointer" type, even though we will NOT have any calldata
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/loading.rs`
**Line**: 143
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Nevertheless we put it "formally" to make an empty slice to designated page
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/register_input_view.rs`
**Line**: 12
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we can decompose register into bytes before passing it into individual opcodes
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/register_input_view.rs`
**Line**: 13
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// because eventually those bytes will go into XOR/AND/OR table as inputs and will be range checked
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/register_input_view.rs`
**Line**: 14
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// anyway
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/register_input_view.rs`
**Line**: 19
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// used for bitwise operations and as a shift
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/register_input_view.rs`
**Line**: 21
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// copied from initial decomposition
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/utils.rs`
**Line**: 127
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
/// NOTE: final state is one if we INDEED READ, so extra care should be taken to select and preserve
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/utils.rs`
**Line**: 128
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
/// if we ever need it or not
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/utils.rs`
**Line**: 150
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// dbg!(timestamp.witness_hook(&*cs)().unwrap());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/utils.rs`
**Line**: 151
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// dbg!(location.witness_hook(&*cs)().unwrap());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/utils.rs`
**Line**: 196
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// this is absorb with replacement
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/utils.rs`
**Line**: 215
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// for all reasonable execution traces it's fine
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/utils.rs`
**Line**: 245
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we assume that we did quickly select low part of the register before somehow, so we
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/utils.rs`
**Line**: 264
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we use absolute addressing then we just access reg + imm mod 2^16
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/utils.rs`
**Line**: 265
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we use relative addressing then we access sp +/- (reg + imm), and if we push/pop then we updat
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/utils.rs`
**Line**: 267
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// here we only read
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/utils.rs`
**Line**: 269
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// manually unrolled selection. We KNOW that either we will not care about this particular value,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/utils.rs`
**Line**: 270
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// or one of the bits here was set anyway
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/utils.rs`
**Line**: 281
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we have a special rule for NOP opcode: if we NOP then even though we CAN formally address the mem
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/utils.rs`
**Line**: 314
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we assume that we did quickly select low part of the register before somehow, so we
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/utils.rs`
**Line**: 331
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we use absolute addressing then we just access reg + imm mod 2^16
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/utils.rs`
**Line**: 332
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we use relative addressing then we access sp +/- (reg + imm), and if we push/pop then we updat
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/utils.rs`
**Line**: 334
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// here we only write
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/utils.rs`
**Line**: 336
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// manually unrolled selection. We KNOW that either we will not care about this particular value,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/utils.rs`
**Line**: 337
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// or one of the bits here was set anyway
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/utils.rs`
**Line**: 348
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we have a special rule for NOP opcode: if we NOP then even though we CAN formally address the mem
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/utils.rs`
**Line**: 359
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
&current_sp, // push case, we update SP only after and the memory index should be current SP.
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/utils.rs`
**Line**: 375
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
); // here we do return a new SP as this will be set on the vm state afterwards but won't
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/utils.rs`
**Line**: 376
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// affect our memory location index
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/utils.rs`
**Line**: 386
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
/// NOTE: final state is one if we INDEED READ, so extra care should be taken to select and preserve
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/utils.rs`
**Line**: 387
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
/// if we ever need it or not
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/utils.rs`
**Line**: 496
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// for all reasonable execution traces it's fine
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/mod.rs`
**Line**: 83
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we also need to create the state that reflects the "initial" state for boot process
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/mod.rs`
**Line**: 95
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// or may be it's from FSM, so select
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/mod.rs`
**Line**: 101
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we run `limit` of "normal" cycles
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/mod.rs`
**Line**: 112
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// here we have too large state to run self-tests, so we will compare it only against the full commi
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/mod.rs`
**Line**: 114
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// check for "done" flag
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/mod.rs`
**Line**: 117
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we can not fail exiting, so check for our convention that pc == 0 on success, and != 0 in failure
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/mod.rs`
**Line**: 121
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// bootloader must exist succesfully
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/mod.rs`
**Line**: 132
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// select tails
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/mod.rs`
**Line**: 134
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// memory
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/mod.rs`
**Line**: 147
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// code decommit
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/mod.rs`
**Line**: 159
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// log. We IGNORE rollbacks that never happened obviously
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/mod.rs`
**Line**: 166
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// but we CAN still check that it's potentially mergeable, basically to check that witness generatio
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/mod.rs`
**Line**: 191
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// set everything
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/mod.rs`
**Line**: 200
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we generate witness then we can self-check
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 38
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
/// we assume that
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 41
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
/// Now we need to decide either to mask into exception or into NOP, or execute
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 52
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// decode and resolve condition immediatelly
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 53
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// If we will later on mask into PANIC then we will just ignore resolved condition
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 61
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// resolve fast exceptions
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 67
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// set ergs cost to 0 if we are skipping cycle
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 79
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
let ergs_left = ergs_left.mask_negated(cs, out_of_ergs_exception); // it's 0 if we underflow
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 105
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we do have an exception then we have mask properties into PANIC
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 123
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// mask out aux bits (those are 0, but do it just in case)
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 138
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// then if we didn't mask into panic and condition was false then mask into NOP
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 147
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// mask out aux bits (those are 0, but do it just in case)
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 161
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Ok, now just decompose spreads into bitmasks, and spread and decompose register indexes
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 174
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// split encodings into 4 bit chunks unchecked
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 187
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and enforce their bit length by table access, and simultaneously get
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 188
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// bitmasks for selection
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 190
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// for every register we first need to spread integer index -> bitmask as integer, and then transfor
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 207
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// place everything into struct
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 222
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// for integer N returns a field element with value 0 if N is zero, and 1 << (N-1) otherwise
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 305
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
pub opcode_boolean_spread_data: Num<F>, // this has both flags that describe the opcode itself, and 
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 332
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
let props = witness & OPCODE_PROPS_BITMASK_FOR_BITSPREAD_ENCODING; // bits without AUX flag
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 362
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we should enforce bit length because we just did the splitting
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 365
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now just make a combination to prove equality
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 389
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
/// Decodes only necessary parts of the opcode to resolve condition
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 390
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
/// for masking into NOP if opcode does nothing.
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 391
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
/// We also output imm0/imm1 parts that will NOT be ever masked,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 392
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
/// and register index encoding parts too that would be masked into 0.
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 393
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
/// Please remember that we mask only bitspread part after condition is resolved, and we do not need
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 394
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
/// to recompute the cost(!)
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 403
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we need into total 4 elements:
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 417
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// booleanity constraints
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 469
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// enforce our claimed decomposition
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 484
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// range check parts by feeding into the tables
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 491
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// bit check variant and spread it
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 493
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// by our definition of the table we check the prices to fit into u32
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 497
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// condition is checked to be 3 bits through resolution here
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 507
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// decode the end
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/decoded_opcode.rs`
**Line**: 533
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we only need one FMA gate, so we write the routine manually
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/demux_log_queue/mod.rs`
**Line**: 127
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// passthrough must be trivial
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/demux_log_queue/mod.rs`
**Line**: 144
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// for the rest it's just select between empty or from FSM
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/demux_log_queue/mod.rs`
**Line**: 173
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// copy into observable output
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/demux_log_queue/mod.rs`
**Line**: 188
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// self-check
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/demux_log_queue/mod.rs`
**Line**: 260
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// precompute all comparisons
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/demux_log_queue/mod.rs`
**Line**: 304
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if crate::config::CIRCUIT_VERSOBE {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/demux_log_queue/mod.rs`
**Line**: 305
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     dbg!(bitmasks.witness_hook(cs)().unwrap());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/demux_log_queue/mod.rs`
**Line**: 318
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// checks in "Drop" interact badly with some tools, so we check it during testing instead
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/demux_log_queue/mod.rs`
**Line**: 319
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// debug_assert!(optimizer.is_fresh());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/demux_log_queue/mod.rs`
**Line**: 355
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// We don't need to update head
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/demux_log_queue/mod.rs`
**Line**: 478
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// add tables
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/demux_log_queue/mod.rs`
**Line**: 484
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// start test
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/auxiliary.rs`
**Line**: 131
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// We use here the naming events_deduplicator but the function is applicable for
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/auxiliary.rs`
**Line**: 132
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// storage deduplicator is well - may be we should make this fact more observable
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/auxiliary.rs`
**Line**: 167
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// We use here the naming events_deduplicator but the function is applicable for
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/auxiliary.rs`
**Line**: 168
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// storage deduplicator is well - may be we should make this fact more observable
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/auxiliary.rs`
**Line**: 333
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we do rescue prime padding and absorb
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/auxiliary.rs`
**Line**: 335
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
let mut multiple = to_absorb.len() / 8;
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 162
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// create initial queues
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 185
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// create all the intermediate output data in uncommitted form to later check for equality
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 236
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// auxilary intermediate states
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 259
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// final VM storage log state for our construction
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 262
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// form the VM input
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 269
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we can form all the observable inputs already as those are just functions of observable outputs
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 306
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// code decommiments:
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 319
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// log demultiplexer
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 328
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// all intermediate queues for sorters
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 338
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// precompiles: keccak, sha256 and ecrecover
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 480
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
assert_eq!(NUM_PROCESSABLE_SHARDS, 1); // no support of porter as of yet
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 486
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// storage acesses filter
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 499
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// storage applicator for rollup subtree (porter subtree is shut down globally currently)
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 516
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we can run all the cirucits in sequence
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 518
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now let's map it for convenience, and later on walk over it
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 614
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
[zero_num; CLOSED_FORM_COMMITTMENT_LENGTH], // formally set here
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 657
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// self-check
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 662
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we can potentially skip some circuits
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 664
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we can skip everything except VM
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 665
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and if we skip, then we should ensure some invariants over outputs!
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 667
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// decommits sorter must output empty queue
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 686
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// decommitter should produce the same memory sequence
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 703
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// demux must produce empty outputs
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 722
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// keccak, sha256 and ecrecover must not modify memory
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 769
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// well, in the very unlikely case of no RAM requests (that is unreachable because VM always starts)
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 777
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// storage filter must produce an empty output
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 786
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// storage application must leave root untouched
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 829
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// events and l2 to l1 messages filters should produce empty output
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 856
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// transient storage doesn't produce an output
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 862
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// L2 to L1 linear hasher
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 876
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if nothing to hash, we expect empty hash
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 902
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we just walk one by one
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 905
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
execution_stage_bitmask[0] = boolean_true; // VM
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 930
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we believe that prover gives us valid compact forms,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 931
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// so we check equality
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 956
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
let not_skip = skip_flag.negated(cs); // this is memoized
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 962
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
let validate_observable_input = validate; // input commitment is ALWAYS the same for all the circuit
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 983
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
let not_skip = skip_flag.negated(cs); // this is memoized
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 1044
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we can use a proper circuit type and manyally add it into single queue
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 1056
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// for any circuit that is NOT start, but is added to recursion queue we validate that previous hidd
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 1057
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// is given to this circuit as hidden FSM input
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 1059
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: we use `start_flag` from witness because we validated it's logic in the lines around
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 1071
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and here we can just update it for the next step
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 1080
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// push
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 1093
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if flag.witness_hook(cs)().unwrap_or(false) {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 1094
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     let circuit_type = BaseLayerCircuitType::from_numeric_value((_idx+1) as u8);
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 1095
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     println!(
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 1097
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         circuit_type,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 1098
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         state.witness_hook(cs)(),
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 1099
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         tail_to_use_for_update.witness_hook(cs)(),
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 1106
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// for the next stage we do shifted AND
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 1108
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// note skip(1)
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 1114
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// update
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 1123
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and check if we are done
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 1130
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// so we are done!
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 1133
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: values below are allocated constant, so their values end up in
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 1134
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// scheduler setup -> verification key
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 1156
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now form a queue for 4844
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 1161
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// eip4844 circuit
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 1192
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// add to the queue
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 1240
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: even though node/leaf circuits are defined over witness-provided (input-linked)
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 1241
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// verification keys, here we EXPECT to have specific CONSTANT verificaion parameters
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 1295
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we can collapse queues
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 1305
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Form a public block header
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 1356
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we are done with this block, process the previous one
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 1371
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// form full block hash, it's just a hash of concatenation of previous and new full content hashes
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 1377
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
let take_by = F::CAPACITY_BITS / 8;
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/mod.rs`
**Line**: 1384
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// treat as BE
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 25
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// This is a sorter of logs that are kind-of "pure", e.g. event emission or L2 -> L1 messages.
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 26
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Those logs do not affect a global state and may either be rolled back in full or not.
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 27
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// We identify equality of logs using "timestamp" field that is a monotonic unique counter
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 28
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// across the block
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 47
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//use table
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 60
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// passthrough must be trivial
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 113
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// get challenges for permutation argument
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 149
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// there is no code at address 0 in our case, so we can formally use it for all the purposes
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 158
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// there is no code at address 0 in our case, so we can formally use it for all the purposes
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 192
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// form the final state
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 262
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we can recreate it here, there are two cases:
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 266
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: scheduler guarantees that only 1 - the first - circuit will have "is_start",
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 267
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// so to take a shortcut we can only need to test if there is nothing in the queue
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 281
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// reallocate and simultaneously collapse rollbacks
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 294
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we also ensure that original items are "write" unless it's a padding
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 315
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now ensure sorting
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 317
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// sanity check - all such logs are "write into the sky"
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 322
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// check if keys are equal and check a value
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 324
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// We compare timestamps, and then resolve logic over rollbacks, so the only way when
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 325
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// keys are equal can be when we do rollback
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 328
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// ensure sorting for uniqueness timestamp and rollback flag
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 329
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// We know that timestamps are unique accross logs, and are also the same between write and rollback
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 333
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// keys are always ordered as >= unless is padding
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 342
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we pop an item and it's not trivial with different log, then it MUST be non-rollback
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 346
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if it's same non-trivial log, then previous one is always guaranteed to be not-rollback by line a
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 347
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and so this one should be rollback
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 352
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we self-check ourselves over the content of the log, even though by the construction
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 353
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// of the queue it's a guaranteed permutation
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 359
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if previous is not trivial then we always have equal content
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 367
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// decide if we should add the PREVIOUS into the queue
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 368
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// We add only if previous one is not trivial, and current one doesn't rollback it due to different 
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 369
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// OR if current one is trivial
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 382
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// cleanup some fields that are not useful
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 405
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// finalization step - same way, check if last item is not a rollback
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 443
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
/// Check that a == b and a > b by performing a long subtraction b - a with borrow.
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 444
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
/// Both a and b are considered as least significant word first
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/log_sorter/mod.rs`
**Line**: 573
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// add tables
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 37
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// turns 128 bits into a Bls12 field element.
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 43
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// compose the bytes into u16 words for the nonnative wrapper
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 46
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// since the value would be interpreted as big endian in the L1 we need to reverse our bytes to
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 47
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// get the correct value
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 50
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// for some reason there is no "from_be_bytes"
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 54
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Note: we do not need to check for overflows because the max value is 2^128 which is less
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 55
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// than the field modulus.
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 66
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// here we just interpret it as LE
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 72
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// compose the bytes into u16 words for the nonnative wrapper
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 81
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// since array_chunks drops any remaining elements that don't fit in the size requirement,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 82
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we need to manually set the last byte in limbs
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 85
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Note: we do not need to check for overflows because the max value is 2^248 which is less
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 86
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// than the field modulus.
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 97
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
/// We interpret out pubdata as chunks of 31 bytes, that are coefficients of
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 98
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
/// some polynomial, starting from the highest one. It's different from 4844 blob data format,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 99
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
/// and we provide additional functions to compute the corresponding 4844 blob data, and restore bac
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 116
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
assert_eq!(limit, ELEMENTS_PER_4844_BLOCK); // max blob length eip4844
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 148
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// create a field element out of the hash of the input hash and the kzg commitment
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 158
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// truncate hash to 128 bits
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 159
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: it is safe to draw a random scalar at max 128 bits because of the schwartz zippel
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 160
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// lemma
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 162
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// take last 16 bytes to get max 2^128
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 163
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// in big endian scenario
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 174
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// We always pad the pubdata to be 31*(2^12) bytes, no matter how many elements we fill.
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 175
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Therefore, we can run the circuit straightforwardly without needing to account for potential
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 176
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// changes in the cycle number in which the padding happens. Hence, we run the loop
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 177
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// to perform the horner's rule evaluation of the blob polynomial and then finalize the hash
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 178
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// out of the loop with a single keccak256 call.
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 184
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// polynomial evaluations via horner's rule
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 186
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// horner's rule is defined as evaluating a polynomial a_0 + a_1x + a_2x^2 + ... + a_nx^n
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 187
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// in the form of a_0 + x(a_1 + x(a_2 + x(a_3 + ... + x(a_{n-1} + xa_n))))
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 188
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// since the blob is considered to be a polynomial in monomial form, we essentially
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 190
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// add and multiply and at the last step we only add the coefficient.
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 202
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// hash equality check
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 207
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now commit to versioned hash || opening point || openinig value
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 209
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// normalize and serialize opening value as BE
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 217
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// field element is normalized, so all limbs are 16 bits
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 237
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// self-check
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 354
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and we need to bitreverse
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 356
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and now serialize in BE form as Ethereum expects
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 371
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we interpret it as coefficients starting from the top one
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 378
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
repr.read_le(&buffer[..]).unwrap(); // note that it's LE
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 379
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Since repr only has 31 bytes, repr is guaranteed to be below the modulus
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 388
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Ethereum's blob data requires that all field element representations are canonical, but we will h
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 389
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// a generic case. For BLS12-381 one can fit 2*modulus into 32 bytes, but not 3, so we need to subtr
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 390
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// to get completely canonical representation sooner or later
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 418
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and we need to bitreverse
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 420
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we need to iFFT it to get monomial form
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 422
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and now serialize in LE by BLOB_CHUNK_SIZE chunks
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 425
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// note that highest monomial goes first in byte array
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 510
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// let owned_cs = ReductionGate::<F, 4>::configure_for_cs(owned_cs, GatePlacementStrategy::UseSpecia
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 531
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// let owned_cs = DotProductGate::<4>::configure_for_cs(owned_cs, GatePlacementStrategy::UseSpeciali
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 547
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// add tables
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 567
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// make some random chunks
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 582
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we can get it as polynomial, and
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 589
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we can get some quasi-setup for KZG and make a blob
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 602
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// compute commitment
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/eip_4844/mod.rs`
**Line**: 644
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// evaluate polynomial
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/buffer/mod.rs`
**Line**: 17
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
pub filled: UInt8<F>, // assume that it's enough
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/buffer/mod.rs`
**Line**: 30
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we map a set of offset + current fill factor into "start from here" bit for 0-th byte of the buff
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/buffer/mod.rs`
**Line**: 79
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// must be called only after caller ensures enough capacity left
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/buffer/mod.rs`
**Line**: 88
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
assert!(N < 128); // kind of arbitrary constant here, in practice we would only use 32
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/buffer/mod.rs`
**Line**: 89
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we do naive implementation of the shift register
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/buffer/mod.rs`
**Line**: 93
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// base case
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/buffer/mod.rs`
**Line**: 108
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we can use a mapping function to determine based on the number of meaningful bytes and curren
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/buffer/mod.rs`
**Line**: 109
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// on which bytes to use from the start and which not. We already shifted all meaningful bytes to th
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/buffer/mod.rs`
**Line**: 110
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// so we only need 1 bit to show "start here"
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/buffer/mod.rs`
**Line**: 112
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// dbg!(shifted_input.witness_hook(cs)());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/buffer/mod.rs`
**Line**: 117
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// TODO: transpose to use linear combination
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/buffer/mod.rs`
**Line**: 119
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// buffer above is shifted all the way to the left, so if byte number 0 can use any of 0..BUFFER_SIZ
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/buffer/mod.rs`
**Line**: 120
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// then for byte number 1 we can only use markers 1..BUFFER_SIZE markers, and so on, and byte number
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/buffer/mod.rs`
**Line**: 121
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// buffer position 0
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/buffer/mod.rs`
**Line**: 123
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we also need to determine if we ever "use" this byte or should zero it out for later padding proc
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/buffer/mod.rs`
**Line**: 134
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// this will underflow and walk around the field range, but not important for our ranges of N
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/buffer/mod.rs`
**Line**: 140
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// compare no overflow
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/interblock/keccak_aggregator.rs`
**Line**: 39
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
let take_by = F::CAPACITY_BITS / 8;
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/interblock/keccak_aggregator.rs`
**Line**: 49
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// transform to bytes
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/interblock/keccak_aggregator.rs`
**Line**: 52
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
src.constraint_bit_length_as_bytes(cs, total_byte_len); // le
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/interblock/keccak_aggregator.rs`
**Line**: 55
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// assert byte is 0
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/interblock/keccak_aggregator.rs`
**Line**: 64
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// mask if necessary
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/interblock/keccak_aggregator.rs`
**Line**: 77
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// run keccak over it
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/interblock/keccak_aggregator.rs`
**Line**: 83
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and make it our publid input
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/interblock/keccak_aggregator.rs`
**Line**: 86
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// treat as BE
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/interblock/mod.rs`
**Line**: 26
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// performs recursion between "independent" units for FIXED verification key
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/interblock/mod.rs`
**Line**: 82
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// as usual - create verifier for FIXED VK, verify, aggregate inputs, output inputs
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/interblock/mod.rs`
**Line**: 91
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// use this and deal with borrow checker
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/interblock/mod.rs`
**Line**: 122
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// verify the proof
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/interblock/mod.rs`
**Line**: 139
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now actually aggregate
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/compression/mod.rs`
**Line**: 24
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// We recursively verify SINGLE proofs over FIXED VK and output it's inputs
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/compression/mod.rs`
**Line**: 64
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// as usual - create verifier for FIXED VK, verify, aggregate inputs, output inputs
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/compression/mod.rs`
**Line**: 72
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// use this and deal with borrow checker
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/compression/mod.rs`
**Line**: 97
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// verify the proof
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/leaf_layer/mod.rs`
**Line**: 55
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: does NOT allocate public inputs! we will deal with locations of public inputs being the sam
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/leaf_layer/mod.rs`
**Line**: 110
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// small trick to simplify setup. If we have nothing to verify, we do not care about VK
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/leaf_layer/mod.rs`
**Line**: 111
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// being one that we want
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/leaf_layer/mod.rs`
**Line**: 158
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// ensure that it's an expected type
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/leaf_layer/mod.rs`
**Line**: 166
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// verify the proof
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/leaf_layer/mod.rs`
**Line**: 178
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// expected proof should be valid
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/leaf_layer/mod.rs`
**Line**: 181
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// enforce publici inputs
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/leaf_layer/mod.rs`
**Line**: 200
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// for el in input_commitment.iter() {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/leaf_layer/mod.rs`
**Line**: 201
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     let gate = PublicInputGate::new(el.get_variable());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/leaf_layer/mod.rs`
**Line**: 202
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     gate.add_to_cs(cs);
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/recursion_tip/mod.rs`
**Line**: 102
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// self-check that it's indeed NODE
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/recursion_tip/mod.rs`
**Line**: 109
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// from that moment we can just use allocated key to verify below
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/recursion_tip/mod.rs`
**Line**: 145
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// verify the proof
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/recursion_tip/mod.rs`
**Line**: 175
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: we usually put inputs as fixed places for all recursive circuits, even though for this type
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/recursion_tip/mod.rs`
**Line**: 176
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we do not have to do it strictly speaking
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/recursion_tip/mod.rs`
**Line**: 178
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// for el in input_commitment.iter() {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/recursion_tip/mod.rs`
**Line**: 179
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     let gate = PublicInputGate::new(el.get_variable());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/recursion_tip/mod.rs`
**Line**: 180
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     gate.add_to_cs(cs);
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/node_layer/mod.rs`
**Line**: 58
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: does NOT allocate public inputs! we will deal with locations of public inputs being the sam
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/node_layer/mod.rs`
**Line**: 112
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// select over which branch we work
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/node_layer/mod.rs`
**Line**: 123
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we need to try to split the circuit
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/node_layer/mod.rs`
**Line**: 135
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if queue length is <= max_length_if_leafs then next layer we aggregate leafs, or aggregate nodes 
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/node_layer/mod.rs`
**Line**: 149
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// small trick to simplify setup. If we have nothing to verify, we do not care about VK
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/node_layer/mod.rs`
**Line**: 150
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// being one that we want
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/node_layer/mod.rs`
**Line**: 159
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// split the original queue into "node_layer_capacity" elements, regardless if next layer
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/node_layer/mod.rs`
**Line**: 160
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// down will aggregate leafs or nodes
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/node_layer/mod.rs`
**Line**: 172
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we aggregate leafs, then we ensure length to be small enough.
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/node_layer/mod.rs`
**Line**: 173
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// It's not mandatory, but nevertheless
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/node_layer/mod.rs`
**Line**: 175
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// check len <= leaf capacity
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/node_layer/mod.rs`
**Line**: 197
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// verify the proof
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/node_layer/mod.rs`
**Line**: 209
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if it's a meaningful proof we should also check that it indeed proofs a subqueue
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/node_layer/mod.rs`
**Line**: 243
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// for el in input_commitment.iter() {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/node_layer/mod.rs`
**Line**: 244
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     let gate = PublicInputGate::new(el.get_variable());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/node_layer/mod.rs`
**Line**: 245
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     gate.add_to_cs(cs);
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/node_layer/mod.rs`
**Line**: 263
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// our logic is that external caller provides splitting witness, and
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/node_layer/mod.rs`
**Line**: 264
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we just need to ensure that total length matches, and glue intermediate points.
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/node_layer/mod.rs`
**Line**: 266
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// We also ensure consistency of split points
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/node_layer/mod.rs`
**Line**: 284
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// add length
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/node_layer/mod.rs`
**Line**: 286
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// ensure consistency
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/recursion/node_layer/mod.rs`
**Line**: 291
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// push the last one
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/precompile_input_outputs/mod.rs`
**Line**: 2
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// universal precompiles passthrough input/output
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/precompile_input_outputs/mod.rs`
**Line**: 3
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// takes requests queue + memory state
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/precompile_input_outputs/mod.rs`
**Line**: 4
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// outputs memory state
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/memory_query/mod.rs`
**Line**: 41
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// in practice we use memory queue, so we need to have a nice way to pack memory query into
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/memory_query/mod.rs`
**Line**: 42
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// 8 field elements. In addition we can exploit the fact that when we will process the elements
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/memory_query/mod.rs`
**Line**: 43
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we will only need to exploit timestamp, page, index, value and r/w flag in their types, but
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/memory_query/mod.rs`
**Line**: 44
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// actual value can be packed more tightly into full field elements as it will only be compared,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/memory_query/mod.rs`
**Line**: 45
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// without access to it's bitwidth
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/memory_query/mod.rs`
**Line**: 71
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: must be same sequence as in `flatten_as_variables`
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/memory_query/mod.rs`
**Line**: 108
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we assume the fact that capacity of F is quite close to 64 bits
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/memory_query/mod.rs`
**Line**: 111
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// strategy: we use 3 field elements to pack timestamp, decomposition of page, index and r/w flag,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/memory_query/mod.rs`
**Line**: 112
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and 5 more elements to tightly pack 8xu32 of values
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/memory_query/mod.rs`
**Line**: 132
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// value. Those in most of the cases will be nops
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/decommit_query/mod.rs`
**Line**: 46
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we assume that page bytes are known, so it'll be nop anyway
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/decommit_query/mod.rs`
**Line**: 160
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: must be same sequence as in `flatten_as_variables`
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/register/mod.rs`
**Line**: 81
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we need to erase bits 32-64 and 64-96
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/register/mod.rs`
**Line**: 91
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: CSAllocatable is done by the macro, so it allocates in the order of declaration,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/register/mod.rs`
**Line**: 92
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and we should do the same here!
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/log_query/mod.rs`
**Line**: 50
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: (shamatar): workaround for cost generics for now
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/log_query/mod.rs`
**Line**: 53
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// because two logs that we add to the queue on write-like operation only differ by
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/log_query/mod.rs`
**Line**: 54
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// rollback flag, we want to specially define offset for rollback, so we can
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/log_query/mod.rs`
**Line**: 55
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// pack two cases for free. Also packing of the rollback should go into variable
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/log_query/mod.rs`
**Line**: 56
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// number 16 or later, so we can share sponges before it
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/log_query/mod.rs`
**Line**: 133
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we decompose "key" and mix it into other limbs because with high probability
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/log_query/mod.rs`
**Line**: 134
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// in VM decomposition of "key" will always exist beforehand
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/log_query/mod.rs`
**Line**: 138
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we want to pack tightly, so we "base" our packing on read and written values
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/log_query/mod.rs`
**Line**: 300
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// continue with written value
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/log_query/mod.rs`
**Line**: 342
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// continue mixing bytes, now from "address"
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/log_query/mod.rs`
**Line**: 464
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we can pack using some other "large" items as base
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/log_query/mod.rs`
**Line**: 515
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and the final variable is just rollback flag itself
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/log_query/mod.rs`
**Line**: 517
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: if you even change this encoding please ensure that corresponding part
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/log_query/mod.rs`
**Line**: 518
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// is updated in TimestampedStorageLogRecord
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/log_query/mod.rs`
**Line**: 606
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we should be able to allocate without knowing values yet
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/log_query/mod.rs`
**Line**: 631
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: must be same sequence as in `flatten_as_variables`
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/log_query/mod.rs`
**Line**: 652
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we will output L2 to L1 messages as byte packed messages, so let's make it
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/log_query/mod.rs`
**Line**: 674
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we truncated, so let's enforce that those were unsused
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/vm_state/saved_context.rs`
**Line**: 14
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// here we store only part of the context that keeps the data that
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/vm_state/saved_context.rs`
**Line**: 15
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// needs to be stored or restored between the calls
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/vm_state/saved_context.rs`
**Line**: 17
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// repeated note on how joining of rollback queues work
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/vm_state/saved_context.rs`
**Line**: 20
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and also declare some rollback_head, such that current_rollback_head = hash(rollback_head, log_el
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/vm_state/saved_context.rs`
**Line**: 29
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// so to proceed with joining we need to only maintain
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/vm_state/saved_context.rs`
**Line**: 40
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
pub this: UInt160<F>, // unfortunately delegatecall mangles this field - it can not be restored from
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/vm_state/saved_context.rs`
**Line**: 70
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
pub total_pubdata_spent: UInt32<F>, // actually signed two-complement
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/vm_state/saved_context.rs`
**Line**: 130
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// full field elements first for simplicity
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/vm_state/saved_context.rs`
**Line**: 164
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we have left
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/vm_state/saved_context.rs`
**Line**: 189
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and now we can just pack the rest into 5 variables
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/vm_state/saved_context.rs`
**Line**: 199
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// pack shard IDs
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/vm_state/saved_context.rs`
**Line**: 216
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// pack boolean flags
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/vm_state/saved_context.rs`
**Line**: 243
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we also need allocate extended
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/vm_state/saved_context.rs`
**Line**: 249
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// TODO: use more optimal allocation for bytes
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/vm_state/saved_context.rs`
**Line**: 526
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// skip and assign any
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/vm_state/saved_context.rs`
**Line**: 537
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// cast back
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/vm_state/mod.rs`
**Line**: 50
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// use parallel select
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/vm_state/mod.rs`
**Line**: 112
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// pub ergs_per_pubdata_byte: UInt32<F>,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/vm_state/mod.rs`
**Line**: 113
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
pub pubdata_revert_counter: UInt32<F>, // actually signed
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/vm_state/callstack.rs`
**Line**: 48
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// execution context that keeps all explicit data about the current execution frame,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/vm_state/callstack.rs`
**Line**: 49
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and avoid recomputing of quantities that also do not change between calls
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/base_structures/state_diff_record/mod.rs`
**Line**: 37
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// the only thing we need is byte encoding
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/context.rs`
**Line**: 35
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// attempt to execute in non-kernel mode for this opcode would be caught before
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/context.rs`
**Line**: 118
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// write in regards of dst0 register
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/context.rs`
**Line**: 173
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
zero_u32, // reserved
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/context.rs`
**Line**: 184
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
zero_u32, // reserved
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/context.rs`
**Line**: 185
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
zero_u32, // reserved
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/context.rs`
**Line**: 186
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
zero_u32, // reserved
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/context.rs`
**Line**: 191
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we will select in the growding width manner
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/context.rs`
**Line**: 214
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we have context
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/context.rs`
**Line**: 229
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// then we have address-like values
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/context.rs`
**Line**: 275
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and finally full register for meta
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 122
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// perform basic validation
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 133
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// this one could wrap around, so we account for it. In case if we wrapped we will skip operation an
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 155
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
aux_heap_growth = aux_heap_growth.mask_negated(cs, uf); // of we access in bounds then it's 0
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 204
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// penalize for heap out of bounds access
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 230
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// burn all the ergs if not enough
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 254
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NB: Etherium virtual machine is big endian;
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 255
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we need to determine the memory cells' indexes which will be accessed
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 256
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// every memory cell is 32 bytes long, the first cell to be accesed has idx = offset / 32
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 257
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if rem = offset % 32 is zero than it is the only one cell to be accessed
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 258
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// 1) cell_idx = offset / cell_length, rem = offset % cell_length =>
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 259
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// offset = cell_idx * cell_length + rem
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 260
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we should also enforce that cell_idx /in [0, 2^32-1] - this would require range check
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 261
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we should also enforce that 0 <= rem < cell_length = 2^5;
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 262
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// rem is actually the byte offset in the first touched cell, to compute bitoffset and shifts
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 263
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we do bit_offset = rem * 8 and then apply shift computing tables
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 264
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// flag does_cross_border = rem != 0
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 277
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// read both memory cells: in what follows we will call the first memory slot A
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 278
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and the second memory Slot B
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 298
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// wrap around
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 320
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we yet access the `a` always
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 324
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we read twice
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 356
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we would not need to read we mask it into 0. We do not care about pointer part as we set const
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 387
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: we need to evaluate this closure strictly AFTER we evaluate previous access to witness,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 388
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// so we "bias" it here
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 392
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we would not need to read we mask it into 0. We do not care about pointer part as we set const
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 395
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we can update the memory queue state
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 407
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if crate::config::CIRCUIT_VERSOBE {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 408
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     if should_apply.witness_hook(&*cs)().unwrap() {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 409
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         dbg!(should_read_a_cell.witness_hook(&*cs)().unwrap());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 410
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         dbg!(should_read_b_cell.witness_hook(&*cs)().unwrap());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 427
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// this is absorb with replacement
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 449
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if crate::config::CIRCUIT_VERSOBE {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 450
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     if should_apply.witness_hook(&*cs)().unwrap() {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 451
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         if should_read_a_cell.witness_hook(&*cs)().unwrap() {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 452
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             dbg!(initial_state.map(|el| Num::from_variable(el)).witness_hook(&*cs)().unwrap());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 453
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             dbg!(final_state_candidate.witness_hook(&*cs)().unwrap());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 471
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// for all reasonable execution traces it's fine
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 481
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now second query
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 494
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// this is absorb with replacement
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 514
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if crate::config::CIRCUIT_VERSOBE {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 515
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     if should_apply.witness_hook(&*cs)().unwrap() {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 516
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         if should_read_b_cell.witness_hook(&*cs)().unwrap() {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 517
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             dbg!(initial_state.map(|el| Num::from_variable(el)).witness_hook(&*cs)().unwrap());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 518
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             dbg!(final_state_candidate.witness_hook(&*cs)().unwrap());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 536
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// for all reasonable execution traces it's fine
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 545
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if crate::config::CIRCUIT_VERSOBE {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 546
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     if should_apply.witness_hook(&*cs)().unwrap() {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 547
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         dbg!(new_memory_queue_length_after_read.witness_hook(&*cs)().unwrap());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 551
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// the issue with UMA is that if we cleanup bytes using shifts
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 552
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// then it's just too heavy in our arithmetization compared to some implementation of shift
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 553
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// register
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 555
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we have a table that is:
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 556
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// b1000000.. LSB first if unalignment is 0
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 557
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// b0100000.. LSB first if unalignment is 1
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 558
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// so it's 32 bits max, and we use parallel select
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 564
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// implement shift register
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 574
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now mask-shift
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 577
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// idx 0 is unalignment of 0 (aligned), idx 31 is unalignment of 31
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 579
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
let src = &bytes_array[idx..(idx + 32)]; // source
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 589
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if crate::config::CIRCUIT_VERSOBE {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 590
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     if should_apply.witness_hook(&*cs)().unwrap() {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 591
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         if should_read_a_cell.witness_hook(&*cs)().unwrap() {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 592
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             let src: [_; 32] = src.to_vec().try_into().unwrap();
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 593
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             dbg!(mask_bit.witness_hook(&*cs)().unwrap());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 594
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             let src_buffer = src.witness_hook(&*cs)().unwrap();
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 595
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             dbg!(hex::encode(&src_buffer));
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 596
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             let dst_buffer = selected_word.witness_hook(&*cs)().unwrap();
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 597
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             dbg!(hex::encode(&dst_buffer));
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 603
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// in case of out-of-bounds UMA we should zero-out tail of our array
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 604
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we need to shift it once again to cleanup from out of bounds part. So we just shift right and
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 639
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// for "write" we have to keep the "leftovers"
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 640
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and replace the "inner" part with decomposition of the value from src1
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 645
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
); // we do not need set panic here, as it's "inside" of `should_skip_memory_ops`
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 648
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// make it BE
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 653
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now it's a little trickier as we have to kind-of transpose
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 667
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// place back
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 669
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
let dst = &mut written_bytes_buffer[idx..(idx + 32)]; // destination
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 687
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we should write both values in corresponding cells
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 689
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// update memory queue state again
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 705
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// read value is LE integer, while words are treated as BE
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 718
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// read value is LE integer, while words are treated as BE
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 743
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// this is absorb with replacement
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 765
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if crate::config::CIRCUIT_VERSOBE {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 766
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     if should_apply.witness_hook(&*cs)().unwrap() {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 767
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         if execute_write.witness_hook(&*cs)().unwrap() {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 768
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             dbg!(initial_state.map(|el| Num::from_variable(el)).witness_hook(&*cs)().unwrap());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 769
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             dbg!(final_state_candidate.witness_hook(&*cs)().unwrap());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 787
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// for all reasonable execution traces it's fine
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 798
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now second query
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 811
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// this is absorb with replacement
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 834
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if crate::config::CIRCUIT_VERSOBE {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 835
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     if should_apply.witness_hook(&*cs)().unwrap() {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 836
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         if execute_unaligned_write.witness_hook(&*cs)().unwrap() {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 837
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             dbg!(initial_state.map(|el| Num::from_variable(el)).witness_hook(&*cs)().unwrap());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 838
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//             dbg!(final_state_candidate.witness_hook(&*cs)().unwrap());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 856
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// for all reasonable execution traces it's fine
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 866
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// push witness updates
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 869
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we should assemble all the dependencies here, and we will use AllocateExt here
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 936
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if crate::config::CIRCUIT_VERSOBE {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 937
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     if should_apply.witness_hook(&*cs)().unwrap() {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 938
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         dbg!(new_memory_queue_length_after_writes.witness_hook(&*cs)().unwrap());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 943
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// read value is LE integer, while words are treated as BE
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 961
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// compute incremented dst0 if we increment
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 992
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// exceptions
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 996
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and memory related staff
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 1003
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// pay for growth
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 1007
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// update sponges and queue states
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 1046
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we can never address a range [2^32 - 32..2^32] this way, but we don't care because
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 1047
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// it's impossible to pay for such memory growth
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 1054
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if crate::config::CIRCUIT_VERSOBE {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 1055
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     dbg!(offset.witness_hook(&*cs)().unwrap());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 1056
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     dbg!(start.witness_hook(&*cs)().unwrap());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 1057
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     dbg!(length.witness_hook(&*cs)().unwrap());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 1060
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we need to check whether we will or not deref the fat pointer.
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 1061
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we only dereference if offset < length (or offset - length < 0)
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 1067
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// 0 of it's heap/aux heap, otherwise use what we have
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 1070
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// by prevalidating fat pointer we know that there is no overflow here,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 1071
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// so we ignore the information
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 1075
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// check that we agree in logic with out-of-circuit comparisons
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 1081
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// check that offset <= MAX_OFFSET_TO_DEREF. For that we add 32 to offset and can either trigger ove
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 1082
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// with u32::MAX
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 1093
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// but on overflow we would still have to panic even if it's a pointer operation
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 1094
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: it's an offset as an absolute value, so if fat pointer's offset is not in slice as checked 
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 1095
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and offset overflow is another
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 1116
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// only necessary for fat pointer deref: now many bytes we zero-out beyond the end of fat pointer
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 1123
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// remainder fits into 8 bits too
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 1127
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// penalize for too high offset on heap - it happens exactly if offset overflows,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 1128
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// or incremented is == u32::MAX, in case we access heap
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/uma.rs`
**Line**: 1145
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// for integer N returns a field element with value 1 << N
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 45
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// condition for overflow is if we add two number >0 and get one <0 (by highest bit),
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 46
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// or add two <0 and get one >0
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 74
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// exception is when a > 0, b < 0, and result is <0,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 75
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// or if a < 0, b > 0, and result >0
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 232
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// modify the key by replacing parts for precompile call
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 239
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// replace bits 128..160 and 160..192
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 255
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// check inplace that pubdata cost is signed, but >0
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 257
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// check that refund is >=0
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 313
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: our opcodes encoding guarantees that there is no "storage read + is first"
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 314
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// variant encodable
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 326
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// here we perform all oracle access first, and then will use values below in particular opcodes
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 345
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we should assemble all the dependencies here, and we will use AllocateExt here
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 371
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: it's possible to have cost negative, if it's e.g. 2nd write in a sequence of 0 -> X -> 0
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 373
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we should nevertheless ensure that it's 0 if it's not rollup access, and not write in general
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 380
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// check range
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 407
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we only refund storage
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 419
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and also compute cost of decommit in our standard units of 32-byte words
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 439
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// but cost of decommit is determined purely by the caller
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 442
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and check if decommit would end up a repeated one
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 447
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we know net cost
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 465
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if not enough then leave only 0
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 468
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: here we will start to use other markers that will check branches being taken below. Some ch
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 471
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and we do not execute any ops in practice
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 475
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we right away compute final cost of the operation here, and we will merge it into state when we d
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 489
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: this intrinsic L1 message used L1 calldata, while our counter is for pubdata that can be pr
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 490
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// by some other way, so we do NOT add it here
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 493
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we should assemble all the dependencies here, and we will use AllocateExt here
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 500
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we always access witness, as even for writes we have to get a claimed read value!
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 526
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we read then use the same value - convension!
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 586
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// add actual update of register in case of write
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 593
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// deal with decommit
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 610
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we should assemble all the dependencies here, and we will use AllocateExt here
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 618
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we always access witness, as even for writes we have to get a claimed read value!
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 649
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// form new candidate of decommit queue
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 668
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we can refund a full cost if it's repeated, and only if we did decommit indeed,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 669
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// otherwise there was out of ergs above and
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 679
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: cold_warm_access_ergs_refund is already masked if it's not a storage access
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 687
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// apply refund
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 690
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// assemble dst0 candidates
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 691
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// one for io-like and precompile call
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 702
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// another one for decommit. It's a fat pointer!
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 704
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we have 0 offset and 0 start, and only need length and memory page
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 705
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// page
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 707
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// length is set to the full "free" heap space, and caller is responsible to truncate it
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 718
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// or it's empty if decommit didn't work
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 721
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: if any of the ops that update DST0 fails, then we write exactly empty register (failing her
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 760
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: here it's `should_apply_opcode_base` because write should always happen, but we have
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 761
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// selected a proper value above in case if there was an exception
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 811
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: out of circuit implementation does NOT set pending here and instead just burns ergs,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 812
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// that is equivalent behavior
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 814
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE - we use `should_apply`` here, because values are preselected above via `should_decommit` th
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 855
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we should be clever and simultaneously produce 2 relations:
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 856
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// - 2 common sponges for forward/rollback that only touch the encodings
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 860
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// check that we only differ at the very end
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 868
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we absort with replacement
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 871
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// TODO: may be decide on length specialization
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 873
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// absorb by replacement
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 896
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// absorb by replacement
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 917
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// absorb by replacement
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 939
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// absorb by replacement
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 959
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
); // at the moment we do not mark which sponges are actually used and which are not
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 960
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// in the opcode, so we properly simulate all of them
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 980
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// select forward
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 989
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// select rollback
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/shifts.rs`
**Line**: 60
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// cyclic right rotation x is the same as left cyclic rotation 256 - x
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/shifts.rs`
**Line**: 64
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// no underflow here
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/shifts.rs`
**Line**: 90
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// see description of MulDivRelation to range checks in mul_div.rs, but in short:
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/shifts.rs`
**Line**: 94
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// actual enforcement:
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/shifts.rs`
**Line**: 95
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// for left_shift: a = reg, b = full_shuft, remainder = 0, high = lshift_high, low = lshift_low
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/shifts.rs`
**Line**: 96
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// for right_shift : a = rshift_q, b = full_shift, remainder = rshift_r, high = 0, low = reg
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/shifts.rs`
**Line**: 115
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// but since we can do division, we need to check that remainder < divisor. We also know that diviso
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/shifts.rs`
**Line**: 116
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// extra checks are necessary
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/shifts.rs`
**Line**: 122
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we do division then remainder will be range checked, but not the subtraction result
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/shifts.rs`
**Line**: 125
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// relation is a + b == c + of * 2^N,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/shifts.rs`
**Line**: 126
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// but we compute d - e + 2^N * borrow = f
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/shifts.rs`
**Line**: 128
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// so we need to shuffle
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/shifts.rs`
**Line**: 142
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// of * is_cyclic + limb_in
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/shifts.rs`
**Line**: 154
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Sets an eq flag if out1 is zero
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/shifts.rs`
**Line**: 163
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// flags for a case if we do not set flags
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/shifts.rs`
**Line**: 182
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// add range check request
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/shifts.rs`
**Line**: 210
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// shift + idx << 8
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mul_div.rs`
**Line**: 91
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// by convention this function set remainder to the dividend if divisor is 0 to satisfy
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mul_div.rs`
**Line**: 92
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// mul-div relation. Later in the code we set remainder to 0 in this case for convention
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mul_div.rs`
**Line**: 242
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if crate::config::CIRCUIT_VERSOBE {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mul_div.rs`
**Line**: 243
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     if (should_apply_mul.witness_hook(&*cs))().unwrap_or(false) || (should_apply_div.witness_hook
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mul_div.rs`
**Line**: 244
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         dbg!(mul_low_unchecked.witness_hook(&*cs)().unwrap());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mul_div.rs`
**Line**: 245
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         dbg!(mul_high_unchecked.witness_hook(&*cs)().unwrap());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mul_div.rs`
**Line**: 246
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         dbg!(quotient_unchecked.witness_hook(&*cs)().unwrap());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mul_div.rs`
**Line**: 247
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         dbg!(remainder_unchecked.witness_hook(&*cs)().unwrap());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mul_div.rs`
**Line**: 251
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// IMPORTANT: MulDiv relation is later enforced via `enforce_mul_relation` function, that effectivel
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mul_div.rs`
**Line**: 252
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// so we do NOT need range checkes on anything that will go into MulDiv relation
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mul_div.rs`
**Line**: 266
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// see below, but in short:
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mul_div.rs`
**Line**: 270
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we mull: src0 * src1 = mul_low + (mul_high << 256) => rem = 0, a = src0, b = src1, mul_low = m
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mul_div.rs`
**Line**: 271
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we divide: src0 = q * src1 + rem =>                   rem = rem, a = quotient, b = src1, mul_l
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mul_div.rs`
**Line**: 274
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// note that if we do division, then remainder is range-checked by "result_1" above
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mul_div.rs`
**Line**: 301
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// flags which are set in case of executing mul
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mul_div.rs`
**Line**: 312
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// flags which are set in case of executing div
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mul_div.rs`
**Line**: 315
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// check if quotient and remainder are 0
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mul_div.rs`
**Line**: 319
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// check that remainder is smaller than divisor
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mul_div.rs`
**Line**: 321
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// do remainder - divisor
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mul_div.rs`
**Line**: 325
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we do division then remainder will be range checked, but not the subtraction result
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mul_div.rs`
**Line**: 328
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// relation is a + b == c + of * 2^N,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mul_div.rs`
**Line**: 329
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// but we compute d - e + 2^N * borrow = f
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mul_div.rs`
**Line**: 331
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// so we need to shuffle
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mul_div.rs`
**Line**: 339
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// unless divisor is 0 (that we handle separately),
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mul_div.rs`
**Line**: 340
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we require that remainder is < divisor
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mul_div.rs`
**Line**: 343
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if divisor is 0, then we assume quotient is zero
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mul_div.rs`
**Line**: 345
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and by convention we set remainder to 0 if we divide by 0
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mul_div.rs`
**Line**: 379
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if crate::config::CIRCUIT_VERSOBE {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mul_div.rs`
**Line**: 380
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     if (should_apply_mul.witness_hook(&*cs))().unwrap_or(false) || (should_apply_div.witness_hook
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mul_div.rs`
**Line**: 381
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         dbg!(result_0.witness_hook(&*cs)().unwrap());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mul_div.rs`
**Line**: 382
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//         dbg!(result_1.witness_hook(&*cs)().unwrap());
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mul_div.rs`
**Line**: 401
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// add range check request. Even though it's only needed for division, it's always satisfiable
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/ptr.rs`
**Line**: 55
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// pointer + non_pointer
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/ptr.rs`
**Line**: 58
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we also want to check that src1 is "small" in case of ptr.add
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/ptr.rs`
**Line**: 69
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we add we want upper part of src1 to be zero
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/ptr.rs`
**Line**: 73
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we pack we want lower part of src1 to be zero
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/ptr.rs`
**Line**: 78
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now check overflows/underflows
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/ptr.rs`
**Line**: 104
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we just need to select the result
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/ptr.rs`
**Line**: 106
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// low 32 bits from addition or unchanged original values
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/ptr.rs`
**Line**: 114
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// low 32 bits from subtraction
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/ptr.rs`
**Line**: 122
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// higher 32 bits if shrink
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/ptr.rs`
**Line**: 127
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
&src_0.u32x8_view[3], // otherwise keep src_0 bits 96..128
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/ptr.rs`
**Line**: 177
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// only update dst0 and set exception if necessary
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/binop.rs`
**Line**: 50
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// main point of merging add/sub is to enforce single add/sub relation, that doesn't leak into any
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/binop.rs`
**Line**: 51
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// other opcodes
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/binop.rs`
**Line**: 74
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we need to select, so we first reduce, and then select
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/binop.rs`
**Line**: 89
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now select
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/binop.rs`
**Line**: 105
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we only update flags and dst0
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/binop.rs`
**Line**: 128
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we apply our composite table twice - one to get compound result, and another one as range checks
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/binop.rs`
**Line**: 129
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and add alreabraic relation
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/binop.rs`
**Line**: 142
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we need to pull out individual parts. For that we decompose a value
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/binop.rs`
**Line**: 143
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// let value = (xor_result as u64) << 32 | (or_result as u64) << 16 | (and_result as u64);
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/binop.rs`
**Line**: 176
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// lookup more, but this time using a table as a range check for all the and/or/xor chunks
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/binop.rs`
**Line**: 178
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// value is irrelevant, it's just a range check
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/binop.rs`
**Line**: 187
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// enforce. Note that there are no new variables here
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/nop.rs`
**Line**: 13
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we need to properly select and enforce
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/jump.rs`
**Line**: 26
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// main point of merging add/sub is to enforce single add/sub relation, that doesn't leak into any
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/jump.rs`
**Line**: 27
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// other opcodes
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/jump.rs`
**Line**: 37
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// save next_pc into dst0
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mod.rs`
**Line**: 124
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: fields `a`, `b` and `rem` will be range checked, and fields `mul_low` and `mul_high` are us
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mod.rs`
**Line**: 125
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// only for equality check with guaranteed 32-bit results, so they are also range checked
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mod.rs`
**Line**: 138
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// a * b + rem = mul_low + 2^256 * mul_high
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mod.rs`
**Line**: 140
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// in case of multiplication rem == 0, a and b are src0 and src1
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mod.rs`
**Line**: 141
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// in case of division a = quotient, b = src1, rem is remainder, mul_low = src0
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/mod.rs`
**Line**: 159
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// place end of chain
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret.rs`
**Line**: 21
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// call and ret are merged because their main part is manipulation over callstack,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret.rs`
**Line**: 22
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and we will keep those functions here
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret.rs`
**Line**: 77
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// select callstack that will become current
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret.rs`
**Line**: 137
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we only need select between candidates, and later on we will select on higher level between curre
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret.rs`
**Line**: 153
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// this one will be largely no-op
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret.rs`
**Line**: 168
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// manual implementation of the stack: we either take a old entry and hash along with the saved cont
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret.rs`
**Line**: 177
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we simulate absorb. Note that we have already chosen an initial state,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret.rs`
**Line**: 178
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// so we just use initial state and absorb
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret.rs`
**Line**: 190
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// absorb by replacement
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret.rs`
**Line**: 247
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// assemble a new callstack in full
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret.rs`
**Line**: 287
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// first we push relations that are common, namely callstack sponge
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret.rs`
**Line**: 297
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and now we append relations for far call, that are responsible for storage read and decommittment
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret.rs`
**Line**: 302
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now just append relations to select later on
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret.rs`
**Line**: 304
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// all the opcodes reset flags in full
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret.rs`
**Line**: 308
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// report to witness oracle
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret.rs`
**Line**: 310
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we should assemble all the dependencies here, and we will use AllocateExt here
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret.rs`
**Line**: 342
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// add everything to state diffs
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret.rs`
**Line**: 344
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we should check that opcode can not use src0/dst0 in memory
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret.rs`
**Line**: 368
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// each opcode may have different register updates
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret.rs`
**Line**: 384
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// same for zeroing out and removing ptr markers
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret.rs`
**Line**: 418
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// pending exception if any
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret.rs`
**Line**: 423
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// callstacks in full
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret.rs`
**Line**: 428
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// far call already chosen it
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/add_sub.rs`
**Line**: 15
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// main point of merging add/sub is to enforce single add/sub relation, that doesn't leak into any
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/add_sub.rs`
**Line**: 16
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// other opcodes
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/add_sub.rs`
**Line**: 33
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we need to properly select and enforce
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/add_sub.rs`
**Line**: 59
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// even though we will select for range check in final state diffs application, we already need a se
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/add_sub.rs`
**Line**: 60
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// over result here, so we just add one conditional check
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/add_sub.rs`
**Line**: 63
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we need to enforce relation
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/add_sub.rs`
**Line**: 64
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we enforce a + b = c + 2^N * of,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/add_sub.rs`
**Line**: 65
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// so if we subtract, then we need to swap some staff
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/add_sub.rs`
**Line**: 67
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// relation is a + b == c + of * 2^N,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/add_sub.rs`
**Line**: 68
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// but we compute d - e + 2^N * borrow = f,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/add_sub.rs`
**Line**: 69
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// so e + f = d + of * 2^N
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/add_sub.rs`
**Line**: 71
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Naive options
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/add_sub.rs`
**Line**: 72
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// let add_relation = AddSubRelation {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/add_sub.rs`
**Line**: 73
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     a: common_opcode_state.src0_view.u32x8_view,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/add_sub.rs`
**Line**: 74
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     b: common_opcode_state.src1_view.u32x8_view,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/add_sub.rs`
**Line**: 75
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     c: addition_result_unchecked,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/add_sub.rs`
**Line**: 76
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     of
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/add_sub.rs`
**Line**: 79
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// let sub_relation = AddSubRelation {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/add_sub.rs`
**Line**: 80
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     a: common_opcode_state.src1_view.u32x8_view,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/add_sub.rs`
**Line**: 81
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     b: subtraction_result_unchecked,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/add_sub.rs`
**Line**: 82
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     c: common_opcode_state.src0_view.u32x8_view,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/add_sub.rs`
**Line**: 83
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     of: uf,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/add_sub.rs`
**Line**: 86
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Instead we select non-common part, using the fact
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/add_sub.rs`
**Line**: 87
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// that it's summetric over a/b
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/add_sub.rs`
**Line**: 114
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we need to check for zero and output
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/add_sub.rs`
**Line**: 118
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// gt = !of & !zero, so it's !(of || zero)
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/add_sub.rs`
**Line**: 132
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we only update flags and dst0
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/add_sub.rs`
**Line**: 156
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// add range check request
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 131
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// do via multiselect
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 149
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
pub(crate) generally_invalid: Boolean<F>, // common invariants
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 159
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we can never address a range [2^32 - 32..2^32] this way, but we don't care because
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 160
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// it's impossible to pay for such memory growth
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 291
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// new callstack should be just the same a the old one, but we also need to update the pricing for p
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 332
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// perform all known modifications, like PC/SP saving
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 335
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we need a completely fresh one
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 338
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now also create target for mimic
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 343
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// - resolve caller/callee dependencies
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 357
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// in src0 lives the ABI
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 358
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// in src1 lives the destination
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 360
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we also reuse pre-parsed ABI
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 362
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// src1 is target address
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 427
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// convert ergs
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 449
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// mask flags in ABI if not applicable
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 463
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// the same as we use for LOG
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 468
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// increment next counter
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 482
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we have everything to perform code read and decommittment
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 514
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// exceptions, along with `map_to_trivial` indicate whether we will or will decommit code
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 515
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// into memory, or will just use UNMAPPED_PAGE
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 524
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we should do validation BEFORE decommittment
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 529
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// first we validate if code hash is indeed in the format that we expect
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 531
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// If we do not do "constructor call" then 2nd byte should be 0,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 532
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// otherwise it's 1
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 599
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// same logic for EVM simulator
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 619
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and over empty bytecode
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 642
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// our canonical decommitment hash format has upper 4 bytes zeroed out
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 660
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// after that logic of bytecode length is uniform
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 662
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// at the end of the day all our exceptions will lead to memory page being 0
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 680
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// normalize bytecode hash
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 684
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// resolve passed ergs, passed calldata page, etc
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 710
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// add pointer validation cases
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 748
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we readjust before heap resize
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 773
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and mask in case of exceptions
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 783
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we can resize memory
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 786
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// first mask to 0 if exceptions happened
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 788
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// then compute to penalize for out of memory access attemp
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 791
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and penalize if pointer is fresh and not addressable
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 799
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// potentially pay for memory growth for heap and aux heap
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 804
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
heap_growth = heap_growth.mask_negated(cs, uf); // if we access in bounds then it's 0
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 812
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
aux_heap_growth = aux_heap_growth.mask_negated(cs, uf); // if we access in bounds then it's 0
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 838
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
let ergs_left_after_growth = ergs_left_after_growth.mask_negated(cs, uf); // if not enough - set to 
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 861
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we have a separate table that says:
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 864
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// this is only true for system contracts, so we mask an efficient address
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 909
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
let ergs_left_after_extra_costs = ergs_left_after_extra_costs.mask_negated(cs, uf); // if not enough
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 910
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
let extra_ergs_from_caller_to_callee = extra_ergs_from_caller_to_callee.mask_negated(cs, uf); // als
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 913
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we can indeed decommit
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 919
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// We will have a logic of `add_to_decommittment_queue` set it to 0 at the very end if exceptions wo
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 962
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// on call-like path we continue the forward queue, but have to allocate the rollback queue state fr
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 972
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we always access witness, as even for writes we have to get a claimed read value!
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 997
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we should resolve all passed ergs. That means
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 998
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// that we have to read it from ABI, and then use 63/64 rule
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1004
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: max passable is 63 / 64 * preliminary_ergs_left, that is itself u32, so it's safe to just
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1005
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// mul as field elements
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1016
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// max passable is <= preliminary_ergs_left from computations above, so it's also safe
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1030
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// this one can overflow IF one above underflows, but we are not interested in it's overflow value
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1056
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// this can not overflow by construction
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1067
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// but out of thin air stipend must not overflow
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1072
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// resolve this/callee shard
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1076
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// default is normal call
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1080
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// change if delegate or mimic
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1104
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// resolve static, etc
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1109
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we call EVM simulator we actually reset static flag
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1112
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// actually parts to the new one
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1118
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we need to decide whether new frame is kernel or not for degelatecall
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1127
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// heap and aux heaps for kernel mode get extra
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1147
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// code part
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1150
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// this part
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1153
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// caller part
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1156
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// code page
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1158
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// base page
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1160
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// context u128
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1161
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we do delegatecall then we propagate current context value, otherwise
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1162
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we capture the current one
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1169
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// non-local call
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1174
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we should assemble all the dependencies here, and we will use AllocateExt here
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1209
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and update registers following our ABI rules
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1213
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we put markers of:
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1267
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// erase markers everywhere anyway
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1280
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we didn't decommit for ANY reason then we will have target memory page == UNMAPPED PAGE, that 
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1309
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// We read code hash from the storage if we have enough ergs, and mask out
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1310
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// a case if code hash is 0 into either default AA or 0 if destination is kernel
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1401
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we should assemble all the dependencies here, and we will use AllocateExt here
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1408
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we always access witness, as even for writes we have to get a claimed read value!
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1430
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
log.written_value = read_value; // our convension as in LOG opcode
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1432
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we should access pubdata cost for as it's always logged on storage access
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1435
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we should assemble all the dependencies here, and we will use AllocateExt here
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1441
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we always access witness, as even for writes we have to get a claimed read value!
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1466
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now process the sponges on whether we did read
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1512
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we absort with replacement
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1515
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// TODO: may be decide on length specialization
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1520
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: since we do merged call/ret, we simulate proper relations here always,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1521
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// because we will do join enforcement on call/ret
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1525
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// absorb by replacement
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1548
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// absorb by replacement
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1569
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// absorb by replacement
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1662
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// compute any associated extra costs
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1672
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// decommit and return new code page and queue states
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1684
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we should assemble all the dependencies here, and we will use AllocateExt here
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1692
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we always access witness, as even for writes we have to get a claimed read value!
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1719
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// kind of refund if we didn't decommit
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1739
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we do not decommit then we will eventually map into 0 page, but we didn't spend any computatio
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1760
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// actually modify queue
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1811
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// absorb by replacement
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1829
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: since we do merged call/ret, we simulate proper relations here always,
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1830
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// because we will do join enforcement on call/ret
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/near_call.rs`
**Line**: 17
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we do not need to change queues on call
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/near_call.rs`
**Line**: 48
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// new callstack should be just the same a the old one, but we also need to update the pricing for p
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/near_call.rs`
**Line**: 65
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// perform all known modifications, like PC/SP saving
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/near_call.rs`
**Line**: 68
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// for NEAR CALL the next callstack entry is largely the same
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/near_call.rs`
**Line**: 70
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// on call-like path we continue the forward queue, but have to allocate the rollback queue state fr
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/near_call.rs`
**Line**: 105
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// convert ergs
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/near_call.rs`
**Line**: 131
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we did spend some ergs on decoding, so we use one from prestate
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/near_call.rs`
**Line**: 152
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if underflow than we pass everything!
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/near_call.rs`
**Line**: 206
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// actually "apply" far call
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 22
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
pub(crate) new_forward_queue_tail: [Num<F>; QUEUE_STATE_WIDTH], // after we glue
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 46
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// new callstack should be just the same a the old one, but we also need to update the pricing for p
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 61
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// revert and panic are different only in ABI: whether we zero-out any hints (returndata) about why 
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 108
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// on panic, we should never return any data. in this case, zero out src0 data
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 114
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we may want to return to label
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 124
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// it's a composite allocation, so we handwrite it
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 127
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// this applies necessary constraints
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 176
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// pass back all the ergs (after we paid the cost of "ret" itself),
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 177
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// with may be a small charge for memory growth
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 180
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// resolve some exceptions over fat pointer use and memory growth
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 182
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// exceptions that are specific only to return from non-local frame
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 190
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// resolve returndata pointer if forwarded
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 194
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// symmetric otherwise
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 199
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we also want unidirectional movement of returndata
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 200
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// check if fat_ptr.memory_page < ctx.base_page and throw if it's the case
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 210
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we try to forward then we should be unidirectional, unless kernel knows what it's doing
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 217
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
non_local_frame_exceptions.push(is_ret_panic); // just feed it here as a shorthand
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 225
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// now we can modify fat ptr that is prevalidated
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 252
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// potentially pay for memory growth
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 256
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// first mask to 0 if exceptions happened
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 258
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// then compute to penalize for out of memory access attemp
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 260
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and penalize if pointer is fresh and not addressable
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 271
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
heap_growth = heap_growth.mask_negated(cs, uf); // of we access in bounds then it's 0
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 277
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
aux_heap_growth = aux_heap_growth.mask_negated(cs, uf); // of we access in bounds then it's 0
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 284
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// subtract
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 290
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
let ergs_left_after_growth = ergs_left_after_growth.mask_negated(cs, uf); // if not enough - set to 
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 308
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we should subtract stipend, but only if we exit local frame
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 316
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// give the rest to the original caller
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 321
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// NOTE: if we return from local frame (from near-call), then memory growth will not be triggered ab
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 322
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and so panic can not happen, and we can just propagate already existing heap bound
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 323
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// to update a previous frame. If we return from the far-call then previous frame is not local, and 
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 324
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// not affect it's upper bound at all
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 338
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// resolve merging of the queues
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 340
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// most likely it's the most interesting amount all the tricks that are pulled by this VM
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 342
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// During the execution we maintain the following queue segments of what is usually called a "storag
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 343
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// storage, events, precompiles, etc accesses
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 346
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// l1 message, etc. E.g. precompilecall is pure function and doesn't rollback, and we add nothing to
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 347
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// When frame ends we have to decide whether we discard it's changes or not. So we can do either:
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 350
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// It's easy to notice that this behavior is:
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 351
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// - local O(1): only things like heads/tails of the queues are updated. Changes do accumulate along
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 352
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// then we can apply it O(1)
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 355
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Why one can not do simpler and just memorize the state of some "forward" queue on frame entry and
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 356
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// a code like
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 357
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if (SLOAD(x)) {
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 358
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
//     revert(0, 0)
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 363
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// then we branch on result of SLOAD, but it is not observable (we discarded everything in "forward"
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 365
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we revert then we should append rollback to forward
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 366
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// if we return ok then we should prepend to the rollback of the parent
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 404
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// update forward queue
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 408
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
should_perform_revert, // it's only true if we DO execute and DO revert
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 426
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// update rollback queue of the parent
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 429
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
should_perform_ret_ok, // it's only true if we DO execute and DO return ok
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 444
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we ignore label if we return from the root, of course
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 447
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Candidates for PC to return to
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 450
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// but EH is stored in the CURRENT context
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 462
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// and update registers following our ABI rules
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 464
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// everything goes into r1, and the rest is cleared
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 473
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// the rest is cleared on far return
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 481
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// erase markers everywhere anyway
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 500
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// update pubdata counter in parent frame. If we panic - we do not add, otherwise add
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/ret.rs`
**Line**: 514
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// update global counter. If we revert - we subtract (no underflow)
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/mod.rs`
**Line**: 26
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// higher parts of highest 64 bits
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/mod.rs`
**Line**: 48
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// we can share some checks
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/block_header/mod.rs`
**Line**: 25
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Data that represents a pure state
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/block_header/mod.rs`
**Line**: 33
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Data that is something like STF(BlockPassthroughData, BlockMetaParameters) -> (BlockPassthroughDa
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/block_header/mod.rs`
**Line**: 40
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// Defining some system parameters that are configurable
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/block_header/mod.rs`
**Line**: 50
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// This is the information that represents artifacts only meaningful for this block, that will not b
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/block_header/mod.rs`
**Line**: 51
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// next block
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/block_header/mod.rs`
**Line**: 70
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// only contains information about this block (or any one block in general),
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/block_header/mod.rs`
**Line**: 71
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// without anything about the previous one
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/block_header/mod.rs`
**Line**: 82
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// everything is BE
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/block_header/mod.rs`
**Line**: 96
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// everything is BE
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/block_header/mod.rs`
**Line**: 109
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// everything is BE
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/block_header/mod.rs`
**Line**: 125
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// everything is BE
```

**Recommendation**: Add explicit zero check before division.

---

#### UNCHECKED_DIVISION

**File**: `zk_targets/zksync-circuits/src/scheduler/block_header/mod.rs`
**Line**: 152
**Description**: Division without explicit zero check. May cause constraint failure or undefined behavior.

```rust
// everything is BE
```

**Recommendation**: Add explicit zero check before division.

---

### ℹ️ LOW Findings

#### INCOMPLETE_CODE

**File**: `zk_targets/zksync-circuits/src/keccak256_round_function/buffer/mod.rs`
**Line**: 117
**Description**: TODO comment indicates incomplete implementation.

```rust
// TODO: transpose to use linear combination
```

**Recommendation**: Review and complete the implementation before production use.

---

#### INCOMPLETE_CODE

**File**: `zk_targets/zksync-circuits/src/base_structures/vm_state/saved_context.rs`
**Line**: 249
**Description**: TODO comment indicates incomplete implementation.

```rust
// TODO: use more optimal allocation for bytes
```

**Recommendation**: Review and complete the implementation before production use.

---

#### INCOMPLETE_CODE

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/log.rs`
**Line**: 871
**Description**: TODO comment indicates incomplete implementation.

```rust
// TODO: may be decide on length specialization
```

**Recommendation**: Review and complete the implementation before production use.

---

#### INCOMPLETE_CODE

**File**: `zk_targets/zksync-circuits/src/main_vm/opcodes/call_ret_impl/far_call.rs`
**Line**: 1515
**Description**: TODO comment indicates incomplete implementation.

```rust
// TODO: may be decide on length specialization
```

**Recommendation**: Review and complete the implementation before production use.

---

