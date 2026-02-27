# Trusted Setup Ceremony — FluidElite KZG Parameters

> **Document status:** Production-ready procedure for multi-party trusted setup  
> **Cryptographic basis:** BN254 KZG (Kate–Zaverucha–Goldberg) commitments  
> **Protocol:** Powers-of-tau followed by circuit-specific Phase 2  
> **Compatible with:** Ethereum KZG ceremony, Hermez Phase 1 SRS  

---

## Table of Contents

1. [Overview](#1-overview)
2. [Threat Model](#2-threat-model)
3. [Prerequisites](#3-prerequisites)
4. [Phase 1 — Powers-of-Tau](#4-phase-1--powers-of-tau)
5. [Phase 2 — Circuit-Specific Setup](#5-phase-2--circuit-specific-setup)
6. [Verification](#6-verification)
7. [Archival & Publication](#7-archival--publication)
8. [Operational Integration](#8-operational-integration)
9. [Incident Response](#9-incident-response)
10. [Appendices](#10-appendices)

---

## 1. Overview

### 1.1 Purpose

The FluidElite prover system uses **KZG polynomial commitments** over the **BN254**
curve (also called BN128 / alt_bn128) to generate zero-knowledge proofs of
computational physics simulations. These proofs attest that:

- Navier–Stokes IMEX timesteps were computed correctly
- Euler 3D conservation laws hold
- Thermal conservation constraints are satisfied
- FluidElite turbulence models converge

The **structured reference string (SRS)** — the KZG parameters — contains
curve points of the form $[s^i]_1$ and $[s^i]_2$ for a secret scalar $s$.
If any single party knows $s$, they can forge proofs. The **trusted setup
ceremony** ensures $s$ is never known to any party by using a multi-party
computation (MPC) protocol where **at least one honest participant** suffices
for security.

### 1.2 Circuit Parameters

| Domain | Circuit | Min $k$ | Max Constraints | KZG SRS Size |
|--------|---------|---------|-----------------|--------------|
| FluidElite | `FluidEliteCircuit` | 17 | 131,072 | ~256 MB |
| Euler 3D | `Euler3DConservationCircuit` | 15 | 32,768 | ~64 MB |
| NS-IMEX | `NsImexCircuit` | 17 | 131,072 | ~256 MB |
| Thermal | `ThermalConservationCircuit` | 12 | 4,096 | ~8 MB |
| Membership | `SimpleMembershipCircuit` | 10 | 1,024 | ~2 MB |

The ceremony must produce parameters for $k_{\max} = 24$ (configurable via
`MAX_K` in `fluidelite-zk/src/circuit/config.rs`) to cover all current and
future circuits. Smaller $k$ values are derived by truncating the SRS.

### 1.3 Security Level

- **Curve:** BN254 (~100-bit classical security, sufficient for ≤ 2030 targets)
- **Proof system:** Halo2 with KZG commitment scheme
- **Requirement:** 1-of-$N$ honest participant assumption

---

## 2. Threat Model

### 2.1 Attacks Prevented

| Attack | Description | Mitigation |
|--------|-------------|------------|
| **Toxic waste recovery** | Reconstruct $s$ from ceremony contributions | MPC: each participant multiplies by their own random $s_i$; attacker must corrupt **all** participants |
| **Parameter substitution** | Replace legitimate SRS with forged parameters | SHA-256 digests, on-chain VK hash, Git-signed parameter manifest |
| **Coercion** | Force participant to reveal their randomness | Each participant uses a fresh machine, destroys entropy after contribution |
| **Side-channel** | Extract $s_i$ from timing, power analysis | Air-gapped contribution machine, `getrandom` for entropy |

### 2.2 Trust Assumptions

1. At least **one** ceremony participant honestly destroys their secret $s_i$.
2. The `halo2_axiom` implementation of `ParamsKZG::setup()` is correct.
3. SHA-256 is collision-resistant.
4. BN254 discrete logarithm is hard (holds for foreseeable quantum timeline).

---

## 3. Prerequisites

### 3.1 Software

```bash
# Rust toolchain (nightly for halo2_axiom)
rustup install nightly-2024-06-01
rustup default nightly-2024-06-01

# Build the ceremony tools
cd fluidelite-zk
cargo build --release --bin ceremony_coordinator
cargo build --release --bin ceremony_contribute
cargo build --release --bin ceremony_verify
```

### 3.2 Hardware (per Participant)

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 16 GB | 32 GB |
| CPU | 4 cores | 8+ cores |
| Storage | 20 GB free | 50 GB SSD |
| Network | 10 Mbps | 100 Mbps |
| Air-gapped? | Recommended | Required for coordinator |

### 3.3 Participants

Recruit **minimum 16 participants** from diverse jurisdictions, organizations,
and backgrounds. Include at minimum:

- 3 core team members (Tigantic Holdings)
- 3 external auditors/cryptographers
- 5 community members (self-nominated, verified via GitHub attestation)
- 5 independent infrastructure operators

Each participant signs a **Participation Agreement** confirming they will:
1. Run the contribution binary on a clean machine
2. Destroy all intermediate state after contributing
3. Publish their contribution hash publicly

---

## 4. Phase 1 — Powers-of-Tau

Phase 1 generates the **universal SRS** that is circuit-independent. This is
compatible with the Ethereum KZG ceremony format and can optionally bootstrap
from Ethereum's existing powers-of-tau output.

### 4.1 Bootstrap from Ethereum Ceremony (Recommended)

The Ethereum Foundation's KZG ceremony (EIP-4844) produced a universal SRS
with 4,096+ contributions. We can build upon this existing entropy:

```bash
# Download the Ethereum ceremony output (mainnet SRS)
curl -L -o ethereum_srs.ptau \
  https://ceremony.ethereum.org/api/v1/srs/download

# Verify against the published hash
sha256sum ethereum_srs.ptau
# Expected: <published_hash_from_ethereum_ceremony>

# Convert to halo2 ParamsKZG format
cargo run --release --bin ceremony_coordinator -- \
  import-ptau \
  --input ethereum_srs.ptau \
  --output ./ceremony/phase1/base_srs.params \
  --target-k 24
```

### 4.2 Fresh Ceremony (If Not Bootstrapping)

If starting from scratch, the coordinator initializes Phase 1:

```bash
# Coordinator: Initialize ceremony with random seed
cargo run --release --bin ceremony_coordinator -- \
  init \
  --k 24 \
  --output ./ceremony/phase1/round_000.params \
  --transcript ./ceremony/phase1/transcript.jsonl

# This creates:
#   round_000.params  — Initial SRS (from OsRng)
#   transcript.jsonl  — Ceremony log (empty, append-only)
```

### 4.3 Contribution Round

Each participant performs the following on their clean machine:

```bash
# 1. Download the latest round file
#    (coordinator distributes via HTTPS or IPFS)
wget https://ceremony.fluidelite.io/phase1/round_NNN.params

# 2. Compute SHA-256 of the input
sha256sum round_NNN.params > input_hash.txt

# 3. Generate contribution
#    - Reads round_NNN.params
#    - Multiplies all G1/G2 points by a fresh random scalar s_i
#    - Outputs round_NNN+1.params
cargo run --release --bin ceremony_contribute -- \
  --input round_NNN.params \
  --output round_$(printf "%03d" $((NNN+1))).params \
  --entropy-source os \
  --attestation ./my_attestation.json

# The binary will:
#   a. Read the input SRS
#   b. Generate s_i from OsRng (getrandom syscall)
#   c. Multiply: [s^j]_1 → [s_i · s^j]_1 for all j
#   d. Write the new SRS
#   e. Output attestation JSON with:
#      - Input hash (SHA-256)
#      - Output hash (SHA-256)
#      - Contribution proof (DLOG equality proof)
#      - Participant public key
#      - Timestamp (UTC)

# 4. Verify your own contribution
cargo run --release --bin ceremony_verify -- \
  --prev round_NNN.params \
  --curr round_$(printf "%03d" $((NNN+1))).params \
  --attestation ./my_attestation.json

# 5. Upload the output and attestation
#    (send to coordinator via authenticated channel)

# 6. DESTROY YOUR ENTROPY
#    Securely wipe the machine or at minimum:
shred -vfz -n 10 /tmp/ceremony_entropy_*
# If using a VM, destroy the VM image entirely.
```

### 4.4 Coordinator Workflow

```bash
# For each participant:
# 1. Receive round_NNN+1.params and attestation
# 2. Verify the contribution
cargo run --release --bin ceremony_verify -- \
  --prev ./ceremony/phase1/round_NNN.params \
  --curr ./received/round_$(printf "%03d" $((NNN+1))).params \
  --attestation ./received/attestation_NNN+1.json

# 3. If valid, accept as new latest round
cp ./received/round_$(printf "%03d" $((NNN+1))).params \
   ./ceremony/phase1/

# 4. Append to transcript
cat ./received/attestation_NNN+1.json >> ./ceremony/phase1/transcript.jsonl

# 5. Publish updated round for next participant
```

### 4.5 Finalization

After all contributions:

```bash
# Apply a final random beacon (e.g., Bitcoin block hash at a pre-committed height)
cargo run --release --bin ceremony_coordinator -- \
  finalize \
  --input ./ceremony/phase1/round_016.params \
  --beacon-source bitcoin \
  --beacon-block-height 900000 \
  --output ./ceremony/phase1/final_srs.params \
  --manifest ./ceremony/phase1/manifest.json

# The manifest contains:
# {
#   "ceremony_id": "fluidelite-phase1-2025",
#   "k": 24,
#   "num_contributions": 16,
#   "final_srs_sha256": "abc123...",
#   "beacon": {
#     "source": "bitcoin",
#     "block_height": 900000,
#     "block_hash": "000000000000000000..."
#   },
#   "contributions": [
#     {
#       "round": 1,
#       "participant": "alice@example.com",
#       "input_hash": "...",
#       "output_hash": "...",
#       "attestation_hash": "...",
#       "timestamp": "2025-07-01T12:00:00Z"
#     },
#     ...
#   ]
# }
```

---

## 5. Phase 2 — Circuit-Specific Setup

Phase 2 specializes the universal SRS for each specific circuit. This must be
repeated whenever a circuit changes (new constraints, different $k$).

### 5.1 Generate Circuit-Specific Parameters

```bash
# For each circuit domain:
for domain in thermal euler3d ns_imex fluidelite membership; do
  cargo run --release --bin ceremony_coordinator -- \
    phase2 \
    --srs ./ceremony/phase1/final_srs.params \
    --domain "${domain}" \
    --output "./ceremony/phase2/${domain}_params.bin" \
    --vk-output "./ceremony/phase2/${domain}_vk.json" \
    --solidity-output "./ceremony/phase2/${domain}_vk.sol"
done

# This:
# 1. Loads the universal SRS
# 2. Synthesizes the circuit for the domain
# 3. Generates the proving key (PK) and verifying key (VK)
# 4. Truncates the SRS to the circuit's k value
# 5. Outputs:
#    - Binary params file (ParamsKZG serialized)
#    - VK as JSON (for off-chain verification)
#    - VK as Solidity constants (for on-chain Groth16Verifier.sol)
```

### 5.2 Cross-Verification

At least **3 independent parties** must reproduce the Phase 2 output:

```bash
# Independent verifier (different machine, different build):
cargo run --release --bin ceremony_verify -- \
  --phase2 \
  --srs ./ceremony/phase1/final_srs.params \
  --domain thermal \
  --expected-params ./ceremony/phase2/thermal_params.bin \
  --expected-vk ./ceremony/phase2/thermal_vk.json

# Output: "MATCH" or "MISMATCH" with detailed diff
# Phase 2 is deterministic given the same SRS and circuit definition,
# so all verifiers MUST produce identical output.
```

### 5.3 Verification Key Deployment

```bash
# Generate the Solidity verifier with embedded VK constants
cargo run --release --bin generate_vk -- \
  --params ./ceremony/phase2/membership_params.bin \
  --output ./fluidelite-zk/foundry/src/Groth16Verifier.sol

# Verify the VK hash matches what will be deployed on-chain
sha256sum ./ceremony/phase2/membership_vk.json
# Record this hash — it will be checked by the timelock governance contract
```

---

## 6. Verification

### 6.1 Full Ceremony Verification

Any third party can verify the entire ceremony:

```bash
# Download all ceremony artifacts
git clone https://github.com/TiganticLabz/fluidelite-ceremony.git
cd fluidelite-ceremony

# Verify Phase 1: chain of contributions
cargo run --release --bin ceremony_verify -- \
  --full-phase1 \
  --ceremony-dir ./phase1/ \
  --manifest ./phase1/manifest.json

# This checks:
# 1. round_000 → round_001: valid contribution proof
# 2. round_001 → round_002: valid contribution proof
# ...
# N. round_N → final: random beacon applied correctly
# All SHA-256 hashes in manifest match actual files

# Verify Phase 2: deterministic derivation
for domain in thermal euler3d ns_imex fluidelite membership; do
  cargo run --release --bin ceremony_verify -- \
    --phase2 \
    --srs ./phase1/final_srs.params \
    --domain "${domain}" \
    --expected-params "./phase2/${domain}_params.bin" \
    --expected-vk "./phase2/${domain}_vk.json"
done
```

### 6.2 Integrity Checks

The `params.rs` module in `fluidelite-zk` performs runtime integrity checks:

```rust
// From fluidelite-zk/src/params.rs — SHA-256 verification on every load
// 1. Reads kzg_bn254_kNN.params from cache directory
// 2. Reads kzg_bn254_kNN.sha256 companion file
// 3. Computes SHA-256(params_bytes)
// 4. Compares with stored digest
// 5. On mismatch: deletes corrupt file, regenerates from ceremony SRS
```

### 6.3 On-Chain VK Hash Verification

The deployed `Groth16Verifier.sol` contains hardcoded VK constants. Verify
they match the ceremony output:

```bash
# Extract VK constants from deployed contract
cast call $VERIFIER_ADDRESS "vk_alpha1()" --rpc-url $RPC_URL

# Compare with ceremony VK JSON
jq '.alpha_g1' ./ceremony/phase2/membership_vk.json

# The values must match exactly (same BN254 points)
```

---

## 7. Archival & Publication

### 7.1 Artifact Inventory

| Artifact | Size (k=24) | Storage | Retention |
|----------|-------------|---------|-----------|
| Phase 1 initial SRS | ~512 MB | Git LFS + IPFS | Permanent |
| Each contribution round | ~512 MB | IPFS (pinned) | 5 years |
| Attestation JSONs | ~2 KB each | Git repo | Permanent |
| Phase 1 final SRS | ~512 MB | Git LFS + IPFS + S3 | Permanent |
| Phase 1 manifest | ~10 KB | Git repo | Permanent |
| Phase 2 params (per circuit) | 2–256 MB | Git LFS + S3 | Permanent |
| Phase 2 VK JSONs | ~2 KB each | Git repo | Permanent |
| Ceremony transcript | ~50 KB | Git repo | Permanent |

### 7.2 Publication Channels

1. **Git Repository:** `TiganticLabz/fluidelite-ceremony` (public)
   - All attestations, manifest, VK JSONs, transcript
   - Phase 1/Phase 2 large files via Git LFS

2. **IPFS:** Pin all `.params` files
   - CID recorded in manifest and committed to Git
   - Multiple pinning services: Pinata, web3.storage, Infura

3. **On-Chain:** Ceremony manifest hash stored in governance contract
   ```solidity
   // CeremonyRegistry.sol
   bytes32 public ceremonySrsHash;  // SHA-256 of final_srs.params
   bytes32 public ceremonyManifestHash;  // SHA-256 of manifest.json
   ```

4. **Archive.org:** Snapshot of complete ceremony artifacts
   - URL recorded in manifest

### 7.3 PVC Deployment

For Kubernetes deployments, upload ceremony outputs to the KZG params PVC:

```bash
# Create a temporary pod to upload params
kubectl run params-uploader --image=busybox --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"uploader","image":"busybox",
    "command":["sleep","3600"],
    "volumeMounts":[{"name":"kzg","mountPath":"/data"}]}],
    "volumes":[{"name":"kzg","persistentVolumeClaim":
      {"claimName":"fluidelite-prover-kzg-params"}}]}}'

# Upload each domain's params
for domain in thermal euler3d ns_imex fluidelite membership; do
  kubectl cp "./ceremony/phase2/${domain}_params.bin" \
    fluidelite/params-uploader:/data/kzg_bn254_${domain}.params
  # Also upload the SHA-256 digest
  sha256sum "./ceremony/phase2/${domain}_params.bin" | awk '{print $1}' | \
    kubectl exec -i params-uploader -- sh -c "cat > /data/kzg_bn254_${domain}.sha256"
done

# Clean up
kubectl delete pod params-uploader
```

---

## 8. Operational Integration

### 8.1 Parameter Loading

The `fluidelite-zk` prover loads KZG parameters at startup via `params.rs`:

```
FLUIDELITE_PARAMS_DIR=/opt/trustless/data/kzg-params
```

The load order is:
1. Check `$FLUIDELITE_PARAMS_DIR` for cached `kzg_bn254_kNN.params`
2. Verify SHA-256 integrity against companion `.sha256` file
3. On failure: log error, attempt regeneration (dev mode) or abort (prod mode)

In production (`FLUIDELITE_ENV=production`), regeneration from random seed is
**disabled**. The prover will refuse to start without valid ceremony-derived
parameters. This prevents accidental use of non-ceremony SRS in production.

### 8.2 Parameter Rotation

When rotating to new ceremony parameters (e.g., after a circuit upgrade):

1. Follow the [KZG Parameter Rotation runbook](./runbooks.md#2-kzg-parameter-rotation)
2. Ensure the new VK is deployed on-chain via the timelock governance contract
3. Allow 48-hour timelock to elapse before upgrading prover pods
4. Perform blue-green deployment to avoid any proof disruption

### 8.3 Helm Values

```yaml
# values-enterprise.yaml (excerpt)
persistence:
  kzgParams:
    enabled: true
    storageClass: gp3-encrypted
    size: 5Gi
    mountPath: /opt/trustless/data/kzg-params

prover:
  kzgParamsPath: /opt/trustless/data/kzg-params
  requireCeremonyParams: true  # Refuse to start without ceremony SRS
```

---

## 9. Incident Response

### 9.1 Compromised Participant

If evidence emerges that a participant's entropy was compromised:

1. **Assess scope:** If the compromised participant was not the **only**
   honest participant, the ceremony remains secure. No action needed.
2. **If all participants may be compromised:** Immediately begin a new ceremony
   with fresh participants. Pause proof generation until new SRS is live.
3. **Communication:** Publish an advisory in the ceremony repository with the
   affected round number and participant identity.

### 9.2 Parameter File Corruption

```bash
# Detection: prover logs "SHA-256 mismatch" on startup
# Resolution:
# 1. Re-download from IPFS using the CID in manifest.json
ipfs get QmXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX \
  -o /opt/trustless/data/kzg-params/kzg_bn254_k17.params

# 2. Verify integrity
sha256sum /opt/trustless/data/kzg-params/kzg_bn254_k17.params
# Compare with manifest

# 3. Restart prover
kubectl rollout restart deployment/fluidelite-prover -n fluidelite
```

### 9.3 Circuit Upgrade Requiring New Phase 2

When a circuit is modified (new constraints, different $k$):

1. Re-run Phase 2 only (Phase 1 SRS is reusable)
2. Have 3+ independent parties verify the new Phase 2 output
3. Deploy new VK to on-chain verifier via governance timelock
4. Upload new params to PVC
5. Blue-green deploy prover with new params
6. Verify old proofs still verify under old VK (backward compatibility)
7. After migration period, deprecate old VK

---

## 10. Appendices

### A. Ceremony Timeline

| Day | Activity | Owner |
|-----|----------|-------|
| D-30 | Recruit participants, distribute instructions | Coordinator |
| D-14 | Participants confirm hardware, test contribution binary | All |
| D-7 | Coordinator initializes Phase 1 (or imports Ethereum SRS) | Coordinator |
| D-0 to D+5 | Contribution rounds (2-3 per day) | Participants |
| D+6 | Random beacon applied, Phase 1 finalized | Coordinator |
| D+7 | Phase 2 generation for all circuits | Coordinator |
| D+8-9 | Phase 2 cross-verification by 3+ parties | Verifiers |
| D+10 | Artifacts published (Git, IPFS, Archive.org) | Coordinator |
| D+11 | On-chain VK deployment via governance | Coordinator |
| D+12 | Production prover updated with ceremony params | DevOps |

### B. Contribution Attestation Schema

```json
{
  "$schema": "https://ceremony.fluidelite.io/schemas/attestation-v1.json",
  "version": 1,
  "ceremony_id": "fluidelite-phase1-2025",
  "round": 7,
  "participant": {
    "name": "Alice",
    "organization": "Independent",
    "contact": "alice@example.com",
    "github": "alice"
  },
  "input": {
    "file": "round_006.params",
    "sha256": "abc123..."
  },
  "output": {
    "file": "round_007.params",
    "sha256": "def456..."
  },
  "contribution_proof": {
    "type": "dlog_equality",
    "g1_proof": "0x...",
    "g2_proof": "0x..."
  },
  "environment": {
    "os": "Ubuntu 24.04 (air-gapped)",
    "rust_version": "nightly-2024-06-01",
    "binary_sha256": "...",
    "entropy_source": "getrandom"
  },
  "timestamp": "2025-07-01T14:23:17Z",
  "statement": "I attest that I generated fresh randomness, applied it to the input SRS, and have destroyed all intermediate state including the random scalar."
}
```

### C. Random Beacon Sources

| Source | How Used | Verification |
|--------|----------|-------------|
| Bitcoin block hash | Hash of block at pre-committed height | Any Bitcoin full node |
| Ethereum randao | `PREVRANDAO` at pre-committed slot | Any Ethereum full node |
| drand | League of Entropy round at pre-committed time | drand API + signature verification |

The beacon block height/slot/round is committed publicly **before** the last
contribution round begins. This prevents the coordinator from influencing the
beacon value.

### D. Compatibility with Existing Tools

| Tool | Compatible? | Notes |
|------|-------------|-------|
| `snarkjs` | Yes | Export `.ptau` format via `ceremony_coordinator export-ptau` |
| Hermez Phase 1 | Yes | Import via `ceremony_coordinator import-ptau` |
| Ethereum KZG ceremony | Yes | Direct import of mainnet SRS |
| `halo2_axiom::ParamsKZG` | Native | Direct serialization format |
| Foundry / `forge` | Yes | VK exported as Solidity constants |

### E. Security Checklist

- [ ] Minimum 16 participants recruited from ≥ 5 jurisdictions
- [ ] Each participant confirmed air-gapped or clean VM
- [ ] Contribution binary SHA-256 matches published hash
- [ ] All contribution proofs verify (DLOG equality)
- [ ] Random beacon committed before final contribution
- [ ] Beacon value verified independently by 3+ parties
- [ ] Phase 2 output reproduced by 3+ independent verifiers
- [ ] All artifacts published to Git, IPFS, and Archive.org
- [ ] On-chain manifest hash matches published manifest
- [ ] VK constants in `Groth16Verifier.sol` match ceremony VK
- [ ] Production prover refuses to start without ceremony params
- [ ] Old proofs still verify under backward-compatible VK

---

*Document version: 1.0.0 — Generated for FluidElite Trustless Physics Platform*  
*© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.*
