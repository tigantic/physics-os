# ═══════════════════════════════════════════════════════════════════════════════
# FluidElite — Commercial TPC Certificate Guide
# ═══════════════════════════════════════════════════════════════════════════════
#
# This document describes how to generate, deliver, and independently verify
# a Trustless Physics Certificate (TPC) for a customer CFD simulation.
#
# Target: First commercial certificate for a real customer physics simulation.
# Customer receives certificate → independently verifies → confirms correctness.
# ═══════════════════════════════════════════════════════════════════════════════

## Overview

A TPC certificate cryptographically attests that a physics simulation:

1. **Ran correctly** — the ZK proof proves each timestep satisfies the
   governing PDEs (Euler, Navier-Stokes-IMEX, or thermal conservation).
2. **Was not tampered with** — Merkle tree commits all timestep proofs;
   the root is embedded in the certificate with an Ed25519 signature.
3. **Is independently verifiable** — any third party can verify the
   certificate using only the public verification key, without re-running
   the simulation.

### Certificate Structure (TPC v1)

```
┌─────────────────────────────────────────────────────┐
│ HEADER (64 bytes)                                   │
│   Magic: "TPC\x01"  Version: 1  UUID  Timestamp     │
│   Solver SHA-256 hash                                │
├─────────────────────────────────────────────────────┤
│ LAYER A — Conservation Attestation                   │
│   JSON: conservation law statistics per timestep     │
│   (mass residual, momentum residual, energy residual)│
├─────────────────────────────────────────────────────┤
│ LAYER B — Proof Aggregation                          │
│   JSON: merkle_root, proof_hashes[]                  │
│   Blob: concatenated timestep proofs (optional)      │
├─────────────────────────────────────────────────────┤
│ LAYER C — Simulation Provenance                      │
│   JSON: solver, mesh, boundary conditions, metadata  │
├─────────────────────────────────────────────────────┤
│ METADATA                                             │
│   JSON: certificate_type, domain, timestep_count     │
├─────────────────────────────────────────────────────┤
│ SIGNATURE (128 bytes)                                │
│   Ed25519 public key (32B) + signature (64B)         │
│   + content hash (32B)                               │
└─────────────────────────────────────────────────────┘
```

---

## Step 1: Generate Certificate

### Prerequisites

```bash
# Build the prover with production features
cd fluidelite-zk
cargo build --release --features production

# Verify GPU is detected
nvidia-smi
# Should show: NVIDIA RTX 5070 (or target GPU)
```

### Run Simulation + Proof Generation

```bash
# Example: HVAC thermal analysis for a customer data center
./target/release/fluidelite-prover \
  --domain thermal \
  --mesh /path/to/customer/mesh.h5 \
  --boundary-conditions /path/to/customer/bc.json \
  --timesteps 1000 \
  --dt 0.001 \
  --output /output/simulation_results/ \
  --certificate /output/certificate.tpc \
  --embed-proofs \
  --sign-key /secrets/ca-signing-key.pem
```

**Parameters:**

| Flag | Description |
|------|-------------|
| `--domain` | Physics domain: `euler3d`, `ns_imex`, or `thermal` |
| `--mesh` | Customer mesh file (HDF5 or VTK format) |
| `--boundary-conditions` | JSON file specifying BCs |
| `--timesteps` | Number of simulation timesteps to prove |
| `--dt` | Time step size (seconds) |
| `--output` | Directory for simulation field data |
| `--certificate` | Output path for TPC certificate |
| `--embed-proofs` | Embed all proof bytes in LAYER B (larger but self-contained) |
| `--sign-key` | Ed25519 signing key for certificate authority |

### Programmatic Generation (Rust)

```rust
use fluidelite_zk::multi_timestep::{MultiTimestepProver, TimestepInput};
use fluidelite_zk::thermal_circuit::ThermalProver;

// 1. Run simulation, generating a proof per timestep
let prover = ThermalProver::new(ThermalParams::production());
let mut timestep_inputs = Vec::new();

for step in 0..num_timesteps {
    let (field_data, proof) = simulate_and_prove(step, &prover, &mesh, &bcs);
    timestep_inputs.push(TimestepInput {
        proof_bytes: proof.to_bytes(),
        proof_hash: sha256(&proof.to_bytes()),
        residual_norm: field_data.conservation_residual(),
        metadata: serde_json::json!({
            "timestep": step,
            "time": step as f64 * dt,
            "cfl": field_data.cfl_number(),
        }),
    });
}

// 2. Aggregate into TPC certificate
let aggregator = MultiTimestepProver::new(signing_keypair);
let aggregate = aggregator.aggregate(&timestep_inputs, true)?;

// 3. Write certificate file
std::fs::write("certificate.tpc", &aggregate.tpc_certificate)?;
println!("Certificate ID: {}", aggregate.certificate_id);
println!("Merkle root: {}", hex::encode(&aggregate.merkle_root));
```

---

## Step 2: Deliver Certificate to Customer

### Deliverables Package

Provide the customer with:

1. **`certificate.tpc`** — the TPC certificate binary (self-contained if `--embed-proofs` used)
2. **`verification_key.pub`** — Ed25519 public key for signature verification
3. **`field_data/`** — simulation results (VTK/HDF5 field files)
4. **`VERIFICATION_GUIDE.md`** — this document (customer-facing portion)
5. **`fluidelite-verifier`** — standalone verification binary (statically linked)

### Delivery Checksums

```bash
# Generate SHA-256 checksums for all deliverables
sha256sum certificate.tpc verification_key.pub fluidelite-verifier > CHECKSUMS.sha256
```

---

## Step 3: Customer Independent Verification

The customer can verify the certificate WITHOUT access to the simulation
software, GPU hardware, or the original mesh. Only the verifier binary
and the public key are needed.

### Quick Verification (CLI)

```bash
# Verify certificate integrity and signature
./fluidelite-verifier verify \
  --certificate certificate.tpc \
  --public-key verification_key.pub

# Expected output:
# ✓ Certificate format valid (TPC v1)
# ✓ Header integrity OK
# ✓ Merkle root: 3a7f...c9d2
# ✓ 1000 timestep proofs committed
# ✓ Ed25519 signature VALID
# ✓ Conservation residuals within tolerance
# ✓ Certificate ID: 550e8400-e29b-41d4-a716-446655440000
# ✓ CERTIFICATE VERIFIED
```

### Detailed Verification

```bash
# Verify + dump full certificate contents
./fluidelite-verifier verify \
  --certificate certificate.tpc \
  --public-key verification_key.pub \
  --verbose \
  --output-json verification_report.json

# Verify a specific timestep's inclusion in the Merkle tree
./fluidelite-verifier verify-timestep \
  --certificate certificate.tpc \
  --timestep 42 \
  --proof-hash <hex-hash>
```

### Programmatic Verification (Rust)

```rust
use fluidelite_zk::multi_timestep::{MultiTimestepProver, extract_merkle_root};

// 1. Load certificate
let cert_bytes = std::fs::read("certificate.tpc")?;

// 2. Extract and verify merkle root
let merkle_root = extract_merkle_root(&cert_bytes)?;
println!("Merkle root: {}", hex::encode(&merkle_root));

// 3. Verify Ed25519 signature
let public_key = std::fs::read("verification_key.pub")?;
// Signature is last 128 bytes: [pubkey(32) | signature(64) | hash(32)]
let sig_block = &cert_bytes[cert_bytes.len() - 128..];
let content = &cert_bytes[..cert_bytes.len() - 128];
let signature = ed25519_dalek::Signature::from_bytes(
    sig_block[32..96].try_into()?
);
let verifying_key = ed25519_dalek::VerifyingKey::from_bytes(
    sig_block[0..32].try_into()?
)?;
verifying_key.verify(&sha2::Sha256::digest(content), &signature)?;
println!("Signature VALID");
```

---

## Step 4: On-Chain Anchoring (Optional)

For regulatory or contractual requirements, the certificate's Merkle root
can be anchored on-chain for immutable timestamping.

```bash
# Anchor merkle root on Ethereum mainnet
./fluidelite-verifier anchor \
  --certificate certificate.tpc \
  --rpc-url https://eth-mainnet.g.alchemy.com/v2/<key> \
  --private-key $DEPLOYER_KEY \
  --contract 0x<FluidEliteRegistry>

# Returns: Transaction hash for on-chain anchor
```

The on-chain record allows any party to verify that the certificate existed
at a specific block height, providing non-repudiation.

---

## Verification Key Management

### Key Generation

```bash
# Generate Ed25519 signing keypair for Certificate Authority
openssl genpkey -algorithm ED25519 -out ca-signing-key.pem
openssl pkey -in ca-signing-key.pem -pubout -out verification_key.pub
```

### Key Rotation

1. Generate new keypair
2. Publish new public key to customers and on-chain registry
3. Sign a key rotation certificate with the OLD key attesting the new key
4. Update CA deployment with new signing key
5. Old certificates remain valid — verification uses the key embedded in
   the certificate's signature block

### Key Escrow

For regulated industries, signing keys should be escrowed with a trusted
third party (e.g., HSM-backed key management service).

---

## Certificate Validation Checklist

For the first commercial certificate, verify all of the following:

- [ ] Certificate generates without errors on production GPU cluster
- [ ] All timestep proofs verify individually
- [ ] Merkle tree construction is deterministic (same inputs → same root)
- [ ] Ed25519 signature verifies with published public key
- [ ] Certificate can be verified on a machine WITHOUT GPU
- [ ] Conservation residuals are within domain-specific tolerance
- [ ] Certificate file size is reasonable (< 100 MB for 1000 timesteps)
- [ ] Customer can run verification independently using delivered tools
- [ ] Verification completes in < 60 seconds for 1000-timestep certificate
- [ ] On-chain anchor transaction succeeds (if required)

---

## Troubleshooting

| Issue | Cause | Resolution |
|-------|-------|------------|
| "Invalid magic bytes" | Corrupted file or wrong format | Re-download certificate, check checksums |
| "Signature verification failed" | Wrong public key or tampered certificate | Verify using correct public key from CA |
| "Merkle root mismatch" | Proof data modified after signing | Certificate integrity compromised — reject |
| "Missing LAYER B proofs" | Certificate generated without `--embed-proofs` | Request embedded version or obtain proofs separately |
| "Timestep not in tree" | Wrong timestep index or hash | Verify timestep index is 0-based and hash matches |
