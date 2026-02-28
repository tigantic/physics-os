# The Physics OS — Documentation Index

```
╔══════════════════════════════════════════════════════════════════════╗
║                        DOCUMENTATION HUB                            ║
║            Tensor-Compressed Physics at Scale                        ║
╚══════════════════════════════════════════════════════════════════════╝
```

> **Repository structure**: 17 top-level directories, 16 root config/doc files.
> Root items reduced from 265 → 33 during the Feb 2026 reorganization.

---

## Quick Navigation

| You want to… | Go to |
|:-------------|:------|
| Understand the platform | [PLATFORM_SPECIFICATION.md](../PLATFORM_SPECIFICATION.md) |
| Start contributing | [CONTRIBUTING.md](../CONTRIBUTING.md) |
| Onboard as a new developer | [ONBOARDING.md](ONBOARDING.md) |
| Browse architecture decisions | [docs/adr/](adr/README.md) (23 ADRs) |
| Read the API reference | [docs/api/](api/README.md) (90 modules) |
| Review the physics inventory | [PHYSICS_INVENTORY.md](PHYSICS_INVENTORY.md) |
| See the repo reorg plan | [REPO_REORGANIZATION_EXECUTION_PLAN.md](REPO_REORGANIZATION_EXECUTION_PLAN.md) |

---

## Root Documents

| Document | Purpose |
|----------|---------|
| [README.md](../README.md) | Project overview, quickstart, and badges |
| [PLATFORM_SPECIFICATION.md](../PLATFORM_SPECIFICATION.md) | Authoritative platform spec (24 sections, 7 appendices) |
| [CHANGELOG.md](../CHANGELOG.md) | Version history |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | Contribution guidelines |
| [SECURITY.md](../SECURITY.md) | Security policy and vulnerability reporting |
| [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md) | Community standards |
| [LICENSE](../LICENSE) | License terms |
| [CITATION.cff](../CITATION.cff) | Academic citation metadata |
| [Makefile](../Makefile) | Build, test, lint targets |

---

## Directory Structure

```
HyperTensor-VM-main/
├── apps/               # Standalone applications (Rust binaries, Python apps)
├── archive/            # Archived prototypes and deprecated code
├── artifacts/          # Build outputs, attestations, certificates, logs
├── contracts/          # Solidity smart contracts (Halo2 verifier, Semaphore)
├── crates/             # Rust workspace crates (15 members)
├── data/               # Datasets, models, caches, material databases
├── deploy/             # Containerfile, Docker configs, deployment manifests
├── docs/               # ← You are here
├── experiments/        # Research scripts, benchmarks, notebooks
├── physics_os/        # Python package: Physics OS orchestration layer
├── integrations/       # Game engine integrations (Unity, Unreal)
├── products/           # Product-specific Python packages
├── proofs/             # Mathematical proofs (conservation, Yang-Mills, ZK)
├── ontic/          # Python package: Tensor network engine (108 modules)
├── tests/              # Test suites (Python + Rust integration)
└── tools/              # Developer tools, scripts, LOC reports
```

---

## Documentation Sections

### Architecture (`docs/architecture/`)

System design, topology, and technical specifications.

| Document | Description |
|----------|-------------|
| [SOVEREIGN_ARCHITECTURE.md](architecture/SOVEREIGN_ARCHITECTURE.md) | Sovereign engine architecture |
| [RAM_BRIDGE_SPEC.md](architecture/RAM_BRIDGE_SPEC.md) | Rust↔Python RAM bridge IPC spec |
| [HYPERTENSOR_ZK_STACK.md](architecture/HYPERTENSOR_ZK_STACK.md) | ZK proof stack architecture |
| [PHYSICS_PIPELINE_COMPLETE.md](architecture/PHYSICS_PIPELINE_COMPLETE.md) | End-to-end physics pipeline |
| [SAFE_SERIALIZATION.md](architecture/SAFE_SERIALIZATION.md) | Safe serialization guide |

### Architecture Decision Records (`docs/adr/`)

23 accepted ADRs covering core platform decisions. See [adr/README.md](adr/README.md) for the full index.

**Highlights:**
- ADR-0001: Capability Ledger as Source of Truth
- ADR-0012: Never Go Dense — All Operations in TT/QTT
- ADR-0017: Q16.16 Fixed-Point for ZK Arithmetic
- ADR-0021: Python Monolith + Rust Crates Architecture
- ADR-0023: Domain Pack Taxonomy (168 Nodes, 20 Verticals)

### API Reference (`docs/api/`)

Auto-generated module documentation for all `tensornet` and `hypertensor` submodules. See [api/README.md](api/README.md).

### Attestations (`docs/attestations/`)

Cryptographic validation records and phase attestation artifacts (85 items).

### Audits (`docs/audits/`)

| Document | Description |
|----------|-------------|
| [CAPABILITY_AUDIT.md](audits/CAPABILITY_AUDIT.md) | Capability coverage assessment |
| [PERFORMANCE_AUDIT_FINDINGS.md](audits/PERFORMANCE_AUDIT_FINDINGS.md) | Performance analysis results |
| [QTT_PERFORMANCE_AUDIT.md](audits/QTT_PERFORMANCE_AUDIT.md) | QTT-specific performance audit |
| [PIPELINE_OPTIMIZATION_AUDIT.md](audits/PIPELINE_OPTIMIZATION_AUDIT.md) | Pipeline optimization findings |

### Commercial (`docs/commercial/`)

| Document | Description |
|----------|-------------|
| [NVIDIA_QTT_PHYSICS_VM_PITCH.md](commercial/NVIDIA_QTT_PHYSICS_VM_PITCH.md) | NVIDIA partnership pitch |
| [NVIDIA_TECHNICAL_BRIEF.md](commercial/NVIDIA_TECHNICAL_BRIEF.md) | Technical brief for NVIDIA |

### Governance (`docs/governance/`)

| Document | Description |
|----------|-------------|
| [API_SURFACE_FREEZE.md](governance/API_SURFACE_FREEZE.md) | API stability policy |
| [CONSTITUTION.md](governance/CONSTITUTION.md) | Platform constitution |
| [DETERMINISM_ENVELOPE.md](governance/DETERMINISM_ENVELOPE.md) | Determinism requirements |
| [ERROR_CODE_MATRIX.md](governance/ERROR_CODE_MATRIX.md) | Error code registry |
| [FORBIDDEN_OUTPUTS.md](governance/FORBIDDEN_OUTPUTS.md) | Forbidden output constraints |
| [METERING_POLICY.md](governance/METERING_POLICY.md) | Compute metering policy |
| [QUEUE_BEHAVIOR_SPEC.md](governance/QUEUE_BEHAVIOR_SPEC.md) | Job queue behavior spec |
| [CLAIM_REGISTRY.md](governance/CLAIM_REGISTRY.md) | Mathematical claim registry |
| [DOMAIN_PACK_AUDIT.md](governance/DOMAIN_PACK_AUDIT.md) | Domain pack compliance audit |

### Operations (`docs/operations/`)

| Document | Description |
|----------|-------------|
| [OPERATIONS_RUNBOOK.md](operations/OPERATIONS_RUNBOOK.md) | Production operations runbook |
| [SECURITY_OPERATIONS.md](operations/SECURITY_OPERATIONS.md) | Security operations procedures |
| [RELEASING.md](operations/RELEASING.md) | Release procedures |
| [SERVER_CONFIGURATION.md](operations/SERVER_CONFIGURATION.md) | Server configuration guide |
| [LAUNCH_GATE_MATRIX.json](operations/LAUNCH_GATE_MATRIX.json) | Launch readiness gates |

### Phases (`docs/phases/`)

Phase planning and completion documentation (8 documents covering Phases 0–5.2).

### Product (`docs/product/`)

| Document | Description |
|----------|-------------|
| [LAUNCH_READINESS.md](product/LAUNCH_READINESS.md) | Launch readiness assessment |
| [PRICING_MODEL.md](product/PRICING_MODEL.md) | Pricing and metering model |
| [RELEASE_NOTES_v4.0.0_BASELINE.md](product/RELEASE_NOTES_v4.0.0_BASELINE.md) | v4.0.0 release notes |
| [CERTIFICATE_TEST_MATRIX.md](product/CERTIFICATE_TEST_MATRIX.md) | Certificate test matrix |

### Research (`docs/research/`)

Research documents, findings, and hypothesis records (40+ items including QTT benchmarks, drug discovery, Vlasov 6D, NTT findings).

### Roadmaps (`docs/roadmaps/`)

| Document | Description |
|----------|-------------|
| [ROADMAP.md](roadmaps/ROADMAP.md) | Master project roadmap |
| [SOVEREIGN_ENGINE_ROADMAP.md](roadmaps/SOVEREIGN_ENGINE_ROADMAP.md) | Sovereign engine roadmap |
| [UI_Roadmap.md](roadmaps/UI_Roadmap.md) | UI development roadmap |

### Strategy (`docs/strategy/`)

| Document | Description |
|----------|-------------|
| [Commercial_Execution.md](strategy/Commercial_Execution.md) | Commercial execution strategy |
| [EXASCALE_IP_EXECUTION_PLAN.md](strategy/EXASCALE_IP_EXECUTION_PLAN.md) | Exascale IP strategy |
| [OS_Evolution.md](strategy/OS_Evolution.md) | Open-source evolution plan |

### Tutorials (`docs/tutorials/`)

| Document | Description |
|----------|-------------|
| [cfd_compressible_flow.md](tutorials/cfd_compressible_flow.md) | CFD compressible flow tutorial |
| [mps_ground_state.md](tutorials/mps_ground_state.md) | MPS ground state tutorial |
| [TPC_INTEGRATOR_GUIDE.md](tutorials/TPC_INTEGRATOR_GUIDE.md) | TPC integration guide |

### Workflows (`docs/workflows/`)

| Document | Description |
|----------|-------------|
| [GIT_WORKFLOW.md](workflows/GIT_WORKFLOW.md) | Git branching and commit workflow |
| [COMMIT_CHECKLIST.md](workflows/COMMIT_CHECKLIST.md) | Pre-commit checklist |
| [REVIEWER_RUNBOOK.md](workflows/REVIEWER_RUNBOOK.md) | Code review procedures |

### Additional Sections

| Directory | Contents |
|-----------|----------|
| [audit/](audit/) | Repository review report (4 items) |
| [evolution/](evolution/) | Platform evolution tracking |
| [images/](images/) | Diagrams and screenshots |
| [legacy/](legacy/) | Historical documents (15 items) |
| [media/](media/) | Video and media assets |
| [papers/](papers/) | Academic papers and publications (14 items) |
| [regulatory/](regulatory/) | Regulatory compliance mappings |
| [reports/](reports/) | Generated reports and dashboards (44+ items) |
| [specifications/](specifications/) | Technical specifications and vision docs |

---

## Rust Workspace (`crates/`)

| Crate | Description |
|-------|-------------|
| `hyper_core` | Physics engine core (QTT, MPO, CFD operators) |
| `hyper_bridge` | RAM bridge IPC (Python ↔ Rust streaming) |
| `hyper_gpu_py` | GPU acceleration PyO3 bindings |
| `tci_core` | Tensor Cross Interpolation (PyO3) |
| `proof_bridge` | Trace → ZK proof pipeline |
| `fluidelite_core` | FluidElite core engine |
| `fluidelite_circuits` | FluidElite Halo2 circuit definitions |
| `fluidelite_zk` | FluidElite ZK proving layer |
| `fluidelite_infra` | FluidElite infrastructure utilities |
| `qtt_cem` | CEM-QTT: Maxwell FDTD solver (Q16.16) |
| `qtt_fea` | FEA-QTT: Hex8 static elasticity solver |
| `qtt_opt` | OPT-QTT: SIMP topology optimization |

---

*Last updated: February 25, 2026*
