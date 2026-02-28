# Three-Tier Design

The Physics OS uses a three-tier architecture separating the open physics
engine from the commercial execution fabric and the performance substrate.

## Tier 1: tensornet (Python)

The physics engine. Pure PyTorch tensor network library implementing MPS,
MPO, QTT, and 168 physics domain packs. Fully open-source.

**Install:** `pip install tensornet`

## Tier 2: hypertensor (Python)

Licensed execution fabric providing REST API, Python SDK, MCP tool server,
job scheduling, billing, and certificate generation. Never distributed as
source.

**Surfaces:**

- `physics_os.api` — FastAPI REST/WebSocket server
- `physics_os.sdk` — Python client library
- `physics_os.mcp` — MCP tool server (agent-native workflows)
- `physics_os.cli` — Command-line interface

## Tier 3: crates/ (Rust)

Performance substrate for GPU kernels, zero-knowledge proof generation,
IPC bridges, and real-time visualization. Compiled to native binaries.

**Key crates:**

| Crate | Purpose |
|-------|---------|
| `hyper_core` | Physics engine core (QTT, MPO, CFD operators) |
| `hyper_bridge` | RAM bridge IPC (Python <-> Rust streaming) |
| `proof_bridge` | Trustless Physics: trace -> ZK proof pipeline |
| `hyper_gpu_py` | GPU compute kernels (PyO3 bindings) |
| `fluidelite_*` | FluidElite ZK proving family |
| `qtt_cem` | Maxwell FDTD solver |
| `qtt_fea` | Hex8 static elasticity solver |
| `qtt_opt` | SIMP topology optimization |
