# TPC Integrator Guide

> **How to request, receive, and verify a Trustless Physics Certificate (TPC)**

This guide covers the complete workflow for integrating FluidElite's
trustless physics verification into your application — from requesting a
proof, to receiving a signed TPC certificate, to verifying it locally and
on-chain.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Quick Start](#quick-start)
5. [Certificate Lifecycle](#certificate-lifecycle)
6. [SDK Integration](#sdk-integration)
   - [Python](#python-sdk)
   - [TypeScript](#typescript-sdk)
   - [Rust (Direct)](#rust-direct)
7. [On-Chain Verification](#on-chain-verification)
8. [Certificate Format](#certificate-format)
9. [Security Model](#security-model)
10. [PQC Forward-Compatibility](#pqc-forward-compatibility)
11. [Troubleshooting](#troubleshooting)
12. [API Reference](#api-reference)

---

## Overview

A **Trustless Physics Certificate (TPC)** is a cryptographically signed
attestation that a physics simulation produced a correct result. It is the
output of the FluidElite trustless verification pipeline:

```
Simulation Data → ZK Proof → Certificate Authority → TPC Certificate → On-Chain Registration
```

Each certificate proves that a specific computation (thermal, fluid dynamics,
etc.) was performed correctly by embedding a zero-knowledge proof, signing the
result with Ed25519, and optionally registering the content hash on Ethereum.

**Trust model**: You don't need to trust the simulator, the CA, or any
intermediary. The ZK proof provides mathematical certainty. The on-chain
registration provides tamper-evidence and public verifiability.

---

## Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌───────────────────────┐
│  Your            │     │  Certificate         │     │  Ethereum             │
│  Application     │────▶│  Authority (CA)      │────▶│  (TPCCertificateReg.) │
│                  │◀────│  POST /v1/certs/issue│     │  verifyCertificate()  │
└─────────────────┘     └──────────────────────┘     └───────────────────────┘
        │                        │                            │
        │   ┌────────────────────┘                            │
        │   │  TPC Binary                                     │
        │   │  (header + layers + proof + signature)          │
        ▼   ▼                                                 │
┌─────────────────┐                                           │
│  Local           │     ┌──────────────────────┐            │
│  Verification    │────▶│  On-Chain Lookup      │◀───────────┘
│  (SDK)           │     │  (content hash match) │
└─────────────────┘     └──────────────────────┘
```

### Components

| Component | Description |
|-----------|-------------|
| **Certificate Authority** | Axum HTTP service that validates proofs, signs certificates |
| **TPCCertificateRegistry** | Solidity contract storing certificate hashes on Ethereum |
| **VKGovernance** | Multi-sig timelock for verification key updates |
| **ProofCompressor** | Gas-optimized proof submission (24.6% calldata savings) |
| **Python SDK** | `pip install fluidelite-verify` |
| **TypeScript SDK** | `npm install @fluidelite/verify` |
| **Certificate Explorer** | Web UI for inspecting TPC certificates |

---

## Prerequisites

### For Certificate Issuance

- **CA URL**: Certificate Authority endpoint (provided by FluidElite)
- **API Key**: Issued to your organization
- **Proof data**: Output of a FluidElite ZK proof generation

### For On-Chain Verification

- **RPC URL**: Ethereum JSON-RPC endpoint (Infura, Alchemy, etc.)
- **Registry Address**: TPCCertificateRegistry contract address
- **Network**: Sepolia (testnet), Base Sepolia, or Ethereum mainnet

### For Local Verification Only

- **No prerequisites** — the SDK can verify certificates offline

---

## Quick Start

### Python (30 seconds)

```bash
pip install fluidelite-verify
```

```python
from fluidelite_verify import Certificate, TPCClient, TPCVerifier

# 1. Issue a certificate via the CA
client = TPCClient(
    ca_url="https://ca.fluidelite.io",
    api_key="YOUR_API_KEY"
)

result = client.issue(
    domain="thermal",
    proof=proof_bytes,        # Your ZK proof bytes
    public_inputs=["0x01"],   # Public inputs
)
print(f"Certificate: {result.certificate_id}")

# 2. Retrieve and verify locally
cert = client.get_certificate(result.certificate_id)
verification = cert.verify()
assert verification.valid, f"Failed: {verification.error}"
print(f"Valid: {verification.valid}")
print(f"Signer: {verification.signer_pubkey}")

# 3. Verify on-chain (optional)
verifier = TPCVerifier(
    rpc_url="https://sepolia.infura.io/v3/YOUR_KEY",
    registry_address="0x..."
)
on_chain = verifier.verify_on_chain(cert)
print(f"Registered: {on_chain.registered}")
print(f"Status: {on_chain.status_name}")
```

### TypeScript (30 seconds)

```bash
npm install @fluidelite/verify
```

```typescript
import { Certificate, TPCClient, TPCVerifier } from '@fluidelite/verify';

// 1. Issue a certificate
const client = new TPCClient({
  caUrl: 'https://ca.fluidelite.io',
  apiKey: 'YOUR_API_KEY',
});

const result = await client.issue({
  domain: 'thermal',
  proof: proofHex,
  publicInputs: ['0x01'],
});
console.log(`Certificate: ${result.certificateId}`);

// 2. Retrieve and verify
const cert = await client.getCertificate(result.certificateId);
const verification = await cert.verify();
console.log(`Valid: ${verification.valid}`);

// 3. On-chain verification (requires viem)
const verifier = new TPCVerifier({
  rpcUrl: 'https://sepolia.infura.io/v3/YOUR_KEY',
  registryAddress: '0x...',
});
const onChain = await verifier.verifyOnChain(cert);
console.log(`Registered: ${onChain.registered}`);
```

---

## Certificate Lifecycle

```
┌──────────┐   Issue   ┌───────────┐   Register  ┌────────────┐
│  Proof    │─────────▶│  TPC Cert │───────────▶│  On-Chain   │
│  Created  │          │  (Signed) │            │  (Immutable)│
└──────────┘          └───────────┘            └────────────┘
                           │                         │
                           │ Verify (local)           │ Verify (on-chain)
                           ▼                         ▼
                     ┌───────────┐            ┌────────────┐
                     │  VALID    │            │  Registered │
                     │  hash ✓   │            │  status: 0  │
                     │  sig  ✓   │            │  PQC: ✓/✗   │
                     └───────────┘            └────────────┘
```

### States

| State | Description |
|-------|-------------|
| **Valid (0)** | Certificate is active and verified |
| **Revoked (1)** | Certificate has been revoked by the revoker |
| **Superseded (2)** | Replaced by a newer certificate |

---

## SDK Integration

### Python SDK

#### Installation

```bash
# Basic (local verification only)
pip install fluidelite-verify

# With on-chain support
pip install 'fluidelite-verify[web3]'
```

#### Local Verification (Offline)

```python
from fluidelite_verify import Certificate

# Load from file
cert = Certificate.from_file("simulation_result.tpc")

# Or from bytes/hex
cert = Certificate.from_bytes(raw_bytes)
cert = Certificate.from_hex(hex_string)

# Verify
result = cert.verify()
print(f"Valid:     {result.valid}")
print(f"Hash OK:   {result.hash_valid}")
print(f"Sig OK:    {result.signature_valid}")
print(f"Cert ID:   {result.certificate_id}")
print(f"Signer:    {result.signer_pubkey}")

# Strict mode (raises on failure)
cert.verify_strict()  # Raises InvalidHash or InvalidSignature

# Inspect
print(cert.summary())
print(f"Domain: {cert.domain}")
print(f"Layers: {len(cert.layers)}")
for layer in cert.layers:
    print(f"  {layer.metadata.get('type', 'unknown')}: {len(layer.blobs)} blobs")
```

#### CA Client

```python
from fluidelite_verify import TPCClient

client = TPCClient(
    ca_url="https://ca.fluidelite.io",
    api_key="YOUR_API_KEY",
    timeout=60.0,
)

# Issue
result = client.issue(
    domain="euler3d",
    proof=proof_bytes,
    public_inputs=["0x42"],
    solver_hash="abcd1234...",
    metadata={"simulation_id": "run-42"},
)

# Retrieve
cert = client.get_certificate(result.certificate_id)

# Server-side verify
response = client.verify(certificate=cert)
# or by ID: client.verify(certificate_id="uuid...")

# Stats
stats = client.stats()
print(f"Total issued: {stats['total_issued']}")
```

#### On-Chain Verification

```python
from fluidelite_verify import TPCVerifier

verifier = TPCVerifier(
    rpc_url="https://sepolia.infura.io/v3/YOUR_KEY",
    registry_address="0x1234...",
)

# Full verification (local + on-chain)
result = verifier.verify(cert)
print(f"Valid: {result.valid}")
print(f"On-chain registered: {result.on_chain.registered}")
print(f"Status: {result.on_chain.status_name}")  # valid/revoked/superseded
print(f"PQC commitment: {result.on_chain.has_pqc_commitment}")

# On-chain only
on_chain = verifier.verify_on_chain(cert)
```

### TypeScript SDK

#### Installation

```bash
# Basic (local verification only)
npm install @fluidelite/verify

# With on-chain support
npm install @fluidelite/verify viem
```

#### Usage

```typescript
import { Certificate, TPCClient, TPCVerifier } from '@fluidelite/verify';

// Local verification
const cert = Certificate.fromHex(hexData);
const result = await cert.verify();

// CA client
const client = new TPCClient({ caUrl: '...', apiKey: '...' });
const issued = await client.issue({ domain: 'thermal', proof: '...' });
const retrieved = await client.getCertificate(issued.certificateId);

// On-chain
const verifier = new TPCVerifier({
  rpcUrl: 'https://sepolia.infura.io/v3/...',
  registryAddress: '0x...',
});
const onChain = await verifier.verifyOnChain(cert);
```

### Rust (Direct)

For Rust integrations, use the `fluidelite-zk` crate directly:

```rust
use fluidelite_zk::certificate_authority::{CertificateAuthority, Domain, IssueCertificateRequest};

// Initialize CA
let ca = CertificateAuthority::new(
    &signing_key_bytes,
    PathBuf::from("./certificates"),
    None,  // prover_url
    None,  // api_key
)?;

// Issue
let req = IssueCertificateRequest {
    domain: Domain::Thermal,
    proof: hex::encode(proof_bytes),
    public_inputs: vec!["0x01".into()],
    solver_hash: None,
    metadata: None,
};
let resp = ca.issue_certificate(&req).await?;

// Retrieve and verify
let cert_data = ca.get_certificate(&resp.certificate_id).await?;
let verification = ca.verify_certificate(&cert_data).await;
assert!(verification.valid);
```

---

## On-Chain Verification

### Contract Addresses

| Network | TPCCertificateRegistry | VKGovernance |
|---------|----------------------|--------------|
| Sepolia | _TBD after deployment_ | _TBD_ |
| Base Sepolia | _TBD_ | _TBD_ |
| Ethereum Mainnet | _TBD_ | _TBD_ |
| Base | _TBD_ | _TBD_ |

### Direct Contract Interaction

```solidity
// Verify a certificate exists on-chain
(bool exists, uint8 status, uint256 index) = registry.verifyCertificate(contentHash);

// Status: 0 = Valid, 1 = Revoked, 2 = Superseded
require(exists && status == 0, "Certificate not valid");

// Check PQC binding
bool hasPQC = registry.hasPQCCommitment(index);
```

### Using ethers.js / viem

```typescript
import { createPublicClient, http } from 'viem';
import { sepolia } from 'viem/chains';

const client = createPublicClient({
  chain: sepolia,
  transport: http('https://sepolia.infura.io/v3/YOUR_KEY'),
});

const [exists, status, index] = await client.readContract({
  address: '0x...', // TPCCertificateRegistry
  abi: registryABI,
  functionName: 'verifyCertificate',
  args: ['0x' + contentHash],
});
```

---

## Certificate Format

### Binary Layout

```
┌─ Header (64 bytes) ────────────────────────────────────────┐
│ [0..4]   Magic: "TPC\x01"                                  │
│ [4..8]   Version: u32 LE                                    │
│ [8..24]  Certificate ID: UUID (16 bytes)                    │
│ [24..32] Timestamp: i64 LE (nanoseconds since epoch)        │
│ [32..64] Solver Hash: SHA-256 (32 bytes)                    │
├─ Layers (variable) ────────────────────────────────────────┤
│ For each layer:                                             │
│   json_length: u32 LE                                       │
│   json_data: [u8; json_length]                             │
│   blob_count: u32 LE                                        │
│   For each blob:                                            │
│     name_length: u16 LE                                     │
│     name: [u8; name_length]                                │
│     data_length: u32 LE                                     │
│     data: [u8; data_length]                                │
├─ Signature Section (128 bytes) ────────────────────────────┤
│ [0..32]   Ed25519 Public Key                                │
│ [32..96]  Ed25519 Signature (over SHA-256 of content)       │
│ [96..128] SHA-256 Content Hash                              │
└────────────────────────────────────────────────────────────┘
```

### Standard Layers

| Layer | Type | Description |
|-------|------|-------------|
| A | `mathematical_truth` | Proof system, curve, public inputs |
| B | `computational_integrity` | Proof hash + proof blob |
| C | `physical_fidelity` | Domain-specific physics metadata |
| D | (metadata) | CA version, issuance metadata |

### Domains

| ID | Name | Description |
|----|------|-------------|
| 0 | `thermal` | Thermal diffusion simulations |
| 1 | `euler3d` | 3D Euler fluid dynamics |
| 2 | `ns_imex` | Navier-Stokes IMEX solver |
| 3 | `fluidelite` | FluidElite ML inference |

---

## Security Model

### Trust Assumptions

| Property | Guarantee |
|----------|-----------|
| **Proof soundness** | ZK proof guarantees computation correctness (Halo2/Groth16) |
| **Certificate integrity** | SHA-256 content hash detects any tampering |
| **Signer authenticity** | Ed25519 signature binds certificate to CA |
| **On-chain immutability** | Ethereum consensus prevents retroactive changes |
| **VK governance** | 2-of-3 multi-sig + 48h timelock prevents unilateral VK changes |
| **PQC readiness** | Dilithium2 commitment hash stored for post-quantum migration |

### What the SDK Verifies

1. **Content hash**: SHA-256(certificate_content) == stored_hash
2. **Ed25519 signature**: Valid signature over content hash
3. **On-chain existence**: Certificate hash registered in TPCCertificateRegistry
4. **Status check**: Certificate not revoked or superseded
5. **PQC commitment**: Dilithium2 binding registered (optional)

### Signer Whitelist

The TPCCertificateRegistry maintains a whitelist of authorized Ed25519
signer public keys. Only certificates signed by whitelisted signers can
be registered on-chain.

---

## PQC Forward-Compatibility

Each certificate can optionally have a **Post-Quantum Cryptography (PQC)
binding** stored on-chain. This is a SHA-256 commitment:

```
commitment = SHA-256(Dilithium2_signature || Dilithium2_pubkey)
```

The full Dilithium2 signature is stored off-chain in the TPC binary's
metadata layer. When PQC verification becomes practical on-chain, the
commitment can be verified against the full signature.

### Checking PQC Status

```python
# Python
on_chain = verifier.verify_on_chain(cert)
print(f"PQC commitment: {on_chain.has_pqc_commitment}")
```

```typescript
// TypeScript
const onChain = await verifier.verifyOnChain(cert);
console.log(`PQC: ${onChain.hasPqcCommitment}`);
```

```solidity
// Solidity
bool hasPQC = registry.hasPQCCommitment(index);
```

---

## Troubleshooting

### Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `InvalidCertificate: too short` | File truncated or wrong format | Ensure file is a complete .tpc binary |
| `InvalidCertificate: Invalid TPC magic` | Not a TPC file | Check file starts with `TPC\x01` |
| `InvalidHash: Content hash mismatch` | Certificate was modified after signing | Re-download from CA |
| `InvalidSignature` | Wrong signer or tampered signature | Verify signer pubkey is correct |
| `OnChainVerificationFailed: not connected` | RPC URL not configured | Set `rpc_url` in verifier config |
| `CertificateNotFound` | ID doesn't exist on CA | Check certificate_id spelling |

### Verifying CA Signer

Always verify the CA's public key matches your expected signer:

```python
cert = Certificate.from_file("result.tpc")
expected_pubkey = "abcd1234..."  # Known CA pubkey
assert cert.signature_section.pubkey_hex == expected_pubkey
```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or with the CA client
import os
os.environ["RUST_LOG"] = "debug"
```

---

## API Reference

### Certificate Authority REST API

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/v1/certificates/issue` | POST | API Key | Issue a new certificate |
| `/v1/certificates/:id` | GET | API Key | Retrieve certificate (binary) |
| `/v1/certificates/verify` | POST | API Key | Verify certificate |
| `/v1/certificates/stats` | GET | API Key | CA statistics |
| `/health` | GET | None | Health check |
| `/metrics` | GET | None | Prometheus metrics |

### Issue Certificate

```http
POST /v1/certificates/issue
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json

{
  "domain": "thermal",
  "proof": "deadbeef...",
  "public_inputs": ["0x01"],
  "solver_hash": "abcd1234...",
  "metadata": {"simulation_id": "run-42"}
}
```

**Response (201 Created):**

```json
{
  "certificate_id": "550e8400-e29b-41d4-a716-446655440000",
  "content_hash": "a1b2c3d4...",
  "signer_pubkey": "abcd1234...",
  "domain": "thermal",
  "size_bytes": 512,
  "issued_at": "2024-01-15T10:30:00Z",
  "on_chain_status": "pending"
}
```

### Verify Certificate

```http
POST /v1/certificates/verify
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json

{
  "certificate_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Response (200 OK):**

```json
{
  "valid": true,
  "hash_valid": true,
  "signature_valid": true,
  "signer_pubkey": "abcd1234...",
  "certificate_id": "550e8400-e29b-41d4-a716-446655440000",
  "domain": "thermal",
  "error": null
}
```

---

## Next Steps

1. **Get API credentials**: Contact brad@tigantic.com
2. **Deploy on testnet**: Use the deployment scripts in `fluidelite-zk/foundry/script/`
3. **Run the explorer**: Open `apps/trustless_verify/explorer.html`
4. **Join the community**: GitHub Issues for support

---

*Copyright © 2025-2026 Bradly Biron Baker Adams / Tigantic Labs. All Rights Reserved.*
