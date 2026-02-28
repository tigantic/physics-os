<div align="center">

# The Physics OS — Architecture

**Release v4.0.1** · **February 2026** · **Powered by The Ontic Engine**

</div>

---

## System Overview

The Physics OS is organized as a layered monorepo where each tier has a strict dependency direction: higher layers depend on lower layers, never the reverse.

```mermaid
graph TB
    subgraph "Client Tier"
        REST["REST API<br/>(9 endpoints)"]
        SDK["Python SDK<br/>(sync + async)"]
        CLI["CLI<br/>(5 commands)"]
        MCP["MCP Server<br/>(11 AI tools)"]
    end

    subgraph "Runtime Access Layer — Physics OS Platform Shell (3,965 LOC)"
        AUTH["Auth + Rate Limit"]
        ROUTER["Job Router"]
        SANITIZER["IP Sanitizer"]
        CERTS["Certificate Engine<br/>(Ed25519)"]
        METER["CU Metering"]
    end

    subgraph "The Ontic Engine — QTT VM (9.9K LOC)"
        IR["QTT IR"]
        COMPILER["7 Domain Compilers"]
        RUNTIME["VM Runtime"]
        GPU["GPU Runtime<br/>(CUDA + Triton)"]
        RANK["Rank Governor"]
    end

    subgraph "The Ontic Engine — Physics Core (471K LOC)"
        CFD["CFD<br/>77K LOC"]
        GENESIS["Genesis Layers<br/>42K LOC"]
        PACKS["Domain Packs<br/>26K LOC"]
        DISCOVER["Discovery Engine<br/>25K LOC"]
        PLATFORM["Platform Substrate<br/>15K LOC"]
        MORE["+ 93 modules"]
    end

    subgraph "Rust Substrate — crates/ (151K LOC)"
        FE_ZK["FluidElite-ZK<br/>31K LOC"]
        COCKPIT["Glass Cockpit<br/>31K LOC"]
        BRIDGE["Ontic Bridge<br/>6K LOC"]
        SOLVERS["CEM / FEA / OPT<br/>5K LOC"]
    end

    REST --> AUTH
    SDK --> AUTH
    CLI --> AUTH
    MCP --> AUTH
    AUTH --> ROUTER
    ROUTER --> SANITIZER
    ROUTER --> CERTS
    ROUTER --> METER
    SANITIZER --> IR
    IR --> COMPILER
    COMPILER --> RUNTIME
    RUNTIME --> GPU
    RUNTIME --> RANK
    RUNTIME --> CFD
    RUNTIME --> GENESIS
    RUNTIME --> PACKS
    RUNTIME --> DISCOVER
    RUNTIME --> PLATFORM
    CFD --> MORE
    BRIDGE --> RUNTIME
    FE_ZK --> CERTS
```

---

## Key Architectural Decisions

| ADR | Decision | Rationale |
|:---:|----------|-----------|
| 001 | QTT as sole representation | O(log N) memory enables exascale on commodity hardware |
| 003 | Never-Dense guarantee | All operations remain in TT format; dense materialization is structurally blocked |
| 005 | Three-layer trust stack | Lean 4 → Halo2 ZK → Ed25519 covers correctness, computation, and provenance |
| 008 | Whitelist-only IP sanitization | Positive filter (25 forbidden categories) protects all internal state |
| 012 | Register-based VM | Domain-agnostic execution enables one runtime for 168 physics nodes |
| 015 | Python↔Rust IPC via mmap | 9ms cross-language latency without serialization overhead |
| 019 | Tag-driven OIDC release | Zero stored secrets; GitHub is the only trust anchor |

Full ADR archive: [`docs/adr/`](docs/adr/) (25 records)

---

## Data Flow — Job Lifecycle

```mermaid
sequenceDiagram
    participant C as Client
    participant A as Auth Layer
    participant R as Job Router
    participant V as Physics VM
    participant S as IP Sanitizer
    participant E as Evidence Generator
    participant K as Certificate Engine

    C->>A: POST /v1/jobs (Bearer token)
    A->>R: Authenticated request
    R->>R: Create job (state: queued)
    R->>V: Compile domain → QTT IR
    V->>V: Execute (state: running)
    V-->>R: Raw result + computation trace
    R->>S: Sanitize (whitelist filter)
    S-->>R: Safe result (state: succeeded)
    R->>E: Generate validation report
    E-->>R: Conservation + stability verified (state: validated)
    R->>K: Sign(SHA-256(input ‖ output ‖ report))
    K-->>R: Ed25519 certificate (state: attested)
    R-->>C: {result, report, certificate}
```

---

## IP Boundary

The **IP Sanitizer** (`hypertensor/core/sanitizer.py`) enforces a whitelist-only boundary between the physics engine and all external surfaces. This is the single chokepoint through which every result must pass.

```
                          ┌──────────────────────────┐
    Physics Engine ──────▶│    IP SANITIZER           │──────▶ External Surface
    (full internal state)  │  25 forbidden categories  │        (safe subset only)
                          │  Whitelist-only filter     │
                          └──────────────────────────┘
```

**Forbidden categories include**: raw TT cores, bond dimensions, rank trajectories, compression ratios, internal solver parameters, convergence history, GPU kernel configs, and 17 more.

---

## Verification Architecture

```mermaid
graph LR
    subgraph "Layer A — Correctness"
        LEAN["Lean 4<br/>57+ theorems"]
    end

    subgraph "Layer B — Computation"
        TRACE["Computation Trace"]
        HALO["Halo2 ZK Circuit"]
        PROOF["ZK Proof"]
    end

    subgraph "Layer C — Provenance"
        HASH["SHA-256 Content Hash"]
        SIGN["Ed25519 Signature"]
        CERT["Trust Certificate"]
    end

    LEAN -->|proves equations| TRACE
    TRACE -->|constrains| HALO
    HALO -->|generates| PROOF
    PROOF -->|binds to| HASH
    HASH -->|signed by| SIGN
    SIGN -->|produces| CERT
```

---

## Detailed Architecture Guide

For module-level dependency graphs, algorithm workflows, and extension point documentation, see the comprehensive guide:

**[`docs/architecture/ARCHITECTURE_GUIDE.md`](docs/architecture/ARCHITECTURE_GUIDE.md)** — 354 lines, 8+ Mermaid diagrams

---

## Related Documents

| Document | Purpose |
|----------|---------|
| [`PLATFORM_SPECIFICATION.md`](PLATFORM_SPECIFICATION.md) | Complete platform specification (2,052 lines) |
| [`docs/adr/`](docs/adr/) | 25 Architecture Decision Records |
| [`API_SURFACE_FREEZE.md`](API_SURFACE_FREEZE.md) | Frozen API contract definition |
| [`DETERMINISM_ENVELOPE.md`](DETERMINISM_ENVELOPE.md) | Deterministic execution guarantees |
| [`docs/governance/CONSTITUTION.md`](docs/governance/CONSTITUTION.md) | Governance and decision authority |
