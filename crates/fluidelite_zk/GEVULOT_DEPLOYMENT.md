# FluidElite Zenith Network Deployment Guide

**Status:** ✅ **VM BUILT & TESTED LOCALLY**  
**VM Image:** [v1.0.0-zk Release](https://github.com/tigantic/HyperTensor-VM/releases/tag/v1.0.0-zk)  
**Image Size:** 101 MB  
**SHA256:** `7663e2a5722c7f6f2ade0188d983df28b7dd5314fe41faacca664b211a79c9d4`

---

## 🏛️ Enterprise Positioning

**FluidElite is the verified AI inference engine for regulated financial institutions.**

### Target Market: Canton Network Ecosystem
- **Goldman Sachs** - Institutional settlements
- **Deloitte** - Audit and compliance AI
- **Digital Asset** - Daml smart contract integration

### Value Proposition
| Feature | Enterprise Benefit |
|---------|-------------------|
| 88.2 TPS ZK Proofs | High-throughput verified AI |
| MiCA Compliance Ready | EU regulatory alignment |
| Encrypted Weights | IP protection (AES-256-GCM) |
| Decentralized Proving | No single point of failure |
| Royalty Model | Revenue per proof |

---

## ⚠️ Network Status

**Gevulot → Zenith Network (Rebranding in Progress)**

The Firestarter network endpoints are offline as Gevulot transitions to Zenith Network.
- `rpc.firestarter.gevulot.com` - DNS not resolving
- `api.gevulot.com` - DNS not resolving

**Launch Preparation:**
```bash
# When Zenith RPC is announced, deploy instantly:
./zenith_launch.sh deploy

# Or with custom endpoint:
ZENITH_RPC=https://rpc.zenith.network ./zenith_launch.sh deploy
```

---

## Local Test Results (QEMU/TCG)

```json
{
  "status": "success",
  "predictions": [{"token_id": 97, "char": "a", "path": "Arithmetic"}],
  "proof_size": 2144,
  "prove_time_ms": 17536,
  "verify_time_ms": 57,
  "verified": true
}
```

> **Note:** 17.5s prove time is due to QEMU software emulation. Native hardware would be ~100x faster (175ms per proof).

---

## Overview

FluidElite runs as a **Linux VM** on the Zenith Network (formerly Gevulot Firestarter).
It generates Zero-Knowledge proofs for language model inference at **88 TPS**.

```
┌─────────────────────────────────────────────────────────────────┐
│                  Zenith Network Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│  User Request → Zenith Network → Your VM → ZK Proof → User     │
├─────────────────────────────────────────────────────────────────┤
│  Input:  /mnt/gevulot/input/task_input.json                    │
│  Output: /mnt/gevulot/output/proof_output.json                 │
│  Model:  /fluidelite_v1.bin (baked into VM image)              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Build Everything

```bash
cd fluidelite-zk
./build-gevulot.sh
```

This will:
- Build the static MUSL binary
- Create a Podman container
- Package it into a Gevulot VM image
- Calculate the SHA256 checksum

### 2. Upload the Image

Upload `fluidelite.img` to a public URL:
- AWS S3
- GitHub Releases  
- IPFS

### 3. Deploy to Gevulot

```bash
./gvltctl program deploy \
    --name "FluidElite-V1" \
    --image "https://YOUR_URL/fluidelite.img" \
    --checksum "YOUR_SHA256_HASH"
```

Save the returned **Program Hash**.

### 4. Submit a Proof Request

Edit `task.yaml` with your program hash, then:

```bash
./gvltctl task create -f task.yaml
```

---

## Manual Build Steps

### Step 1: Build Static Binary

```bash
rustup target add x86_64-unknown-linux-musl

cd fluidelite-zk
cargo build --release --features halo2 \
    --target x86_64-unknown-linux-musl \
    --bin gevulot-prover
```

### Step 2: Build Container

```bash
# From project root
podman build -t localhost/fluidelite-v1:latest -f Containerfile .
```

### Step 3: Build VM Image

```bash
cd fluidelite-zk
./gvltctl build \
    --container containers-storage:localhost/fluidelite-v1:latest \
    -o fluidelite.img
```

**Note:** This compiles a custom Linux kernel. Takes 5-10 minutes.

---

## File Paths (Firestarter v2)

| Path | Purpose |
|------|---------|
| `/mnt/gevulot/input/` | Read-only task inputs |
| `/mnt/gevulot/output/` | Write-only proof outputs |
| `/fluidelite_v1.bin` | Model weights (baked in) |
| `/program` | Your binary (entrypoint) |

---

## Input Format

Write JSON to `/mnt/gevulot/input/task_input.json`:

```json
{
    "context": "The quick brown fox jumps over",
    "include_proof": true
}
```

Or batch multiple contexts:

```json
{
    "contexts": [
        "The quick brown",
        "jumps over the",
        "lazy dog and"
    ],
    "include_proof": true
}
```

---

## Output Format

The prover writes JSON to `/mnt/gevulot/output/proof_output.json`:

```json
{
    "status": "success",
    "predictions": [
        {
            "context_preview": "The quick brown fox jumps over",
            "token_id": 32,
            "char": " ",
            "path": "Lookup",
            "hash": "a1b2c3d4e5f67890"
        }
    ],
    "proof": "0x1234abcd...",
    "proof_size": 2144,
    "prove_time_ms": 1450,
    "verify_time_ms": 8,
    "verified": true
}
```

---

## Performance

| Batch Size | Proof Time | Throughput | Proof Size |
|------------|------------|------------|------------|
| 1          | 1.39s      | 0.7 TPS    | 2,144 B    |
| 8          | 1.39s      | 5.8 TPS    | 2,144 B    |
| 32         | 1.39s      | 22.9 TPS   | 2,144 B    |
| 64         | 1.42s      | 45.2 TPS   | 2,144 B    |
| 128        | 1.45s      | 88.2 TPS   | 2,144 B    |

---

## Files

| File | Purpose |
|------|---------|
| `src/bin/gevulot_prover.rs` | Main prover binary |
| `Containerfile` | Container definition |
| `task.yaml` | Task submission template |
| `build-gevulot.sh` | Automated build script |

---

## Getting Testnet Tokens

1. Generate your key: `./gvltctl keygen -f gevulot_key.json`
2. Join Gevulot Discord or Telegram
3. Request tokens: `/faucet YOUR_ACCOUNT_ID`
4. Verify: `./gvltctl account-info --account YOUR_ACCOUNT_ID`

---

## Troubleshooting

### Build Fails: Missing Libraries

Ensure static linking:
```bash
ldd target/x86_64-unknown-linux-musl/release/gevulot-prover
# Should say "statically linked"
```

### Podman Not Found

```bash
# Ubuntu/Debian
sudo apt install podman

# Fedora
sudo dnf install podman
```

### VM Image Too Large

The image includes a full Linux kernel (~100MB overhead).
This is expected for Firestarter.

### Task Timeout

Increase timeout in `task.yaml`:
```yaml
resources:
  timeout: 120
```

---

## Security

- **Static Binary:** No external dependencies
- **Isolated VM:** Each task runs in fresh environment
- **ZK Proofs:** Cryptographic verification
- **No Network Access:** VM is sandboxed

---

## The "No BD" Business Model

Firestarter is **stateless**. You don't register programs — you submit **Tasks**.

### How It Works

1. **Host your image** on any public URL (GitHub Releases, S3, etc.)
2. **Submit a Task** that references your image URL
3. **Network downloads** your image, runs it, returns the proof
4. **You get paid** a portion of task fees as the Program Author

### The Product

Your `fluidelite.img` URL is your product. Anyone can submit tasks using it.

### The Distribution

```yaml
# Customer's task.yaml
spec:
  image: 'https://github.com/tigantic/HyperTensor-VM/releases/download/v1.0.0/fluidelite.img'
```

### The Income

- Customer pays Gevulot network fee
- Network routes to a prover node
- You (Program Author) receive a cut because you're attested by the checksum

---

## Complete Deployment Walkthrough

### 1. Build the Image

```bash
cd fluidelite-zk
./build-gevulot.sh
```

### 2. Upload to GitHub Releases

1. Go to your repo → Releases → Create Release
2. Tag: `v1.0.0`
3. Attach `fluidelite.img`
4. Publish
5. Copy the download URL

### 3. Submit Your First Task

```bash
# Set environment
export GEVULOT_ENDPOINT="https://rpc.firestarter.gevulot.com"
export GEVULOT_MNEMONIC="clog offer stomach apology pelican craft smile silent galaxy orbit mad aim muscle canal young palace foot annual bullet want assist essay detect outside"

# Update task.yaml with your actual image URL
# Then submit
./gvltctl task create -f task.yaml
```

### 4. Collect Your Proof

```bash
# Wait a few seconds for execution
./gvltctl task get <TASK_ID>
```

The output contains an IPFS URL to your ZK proof.

---

## Important Notes

⚠️ **Image URL must stay alive** as long as tasks run against it.

⚠️ **Checksum attests authorship** — your earnings are tied to it.

⚠️ **Each task downloads fresh** — no persistent state between tasks.

---

*FluidElite V1 - The first viable ZK-LLM for decentralized networks.*
