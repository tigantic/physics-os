# FluidElite Gevulot Firestarter Deployment Guide

**Status:** Production Ready  
**Binary:** 3.7 MB (statically linked)  
**Model:** 65.8 MB  
**Platform:** Gevulot Firestarter (Linux VM)

---

## Overview

FluidElite runs as a **Linux VM** on the Gevulot Firestarter decentralized proving network.
It generates Zero-Knowledge proofs for language model inference at **88 TPS**.

```
┌─────────────────────────────────────────────────────────────────┐
│                Gevulot Firestarter Architecture                 │
├─────────────────────────────────────────────────────────────────┤
│  User Request → Gevulot Network → Your VM → ZK Proof → User    │
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

*FluidElite V1 - The first viable ZK-LLM for decentralized networks.*
