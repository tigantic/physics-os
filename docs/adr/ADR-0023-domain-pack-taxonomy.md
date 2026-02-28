# ADR-0023: Domain Pack Taxonomy (168 Nodes, 20 Verticals)

| Field | Value |
|-------|-------|
| **Status** | accepted |
| **Date** | 2026-02-25 |
| **Author** | @core-team |

## Context

HyperTensor-VM provides physics simulation across a wide range of engineering disciplines: fluid dynamics, structural mechanics, electromagnetics, quantum chemistry, drug discovery, weather prediction, and more. Without an organizing taxonomy, capabilities are discovered ad hoc, coverage gaps are invisible, and customers cannot quickly determine whether the platform supports their use case.

Enterprise physics platforms (ANSYS, COMSOL, Dassault) organize capabilities into product lines and modules. The Physics OS needs an equivalent organizational structure that is:

1. **Hierarchical** — grouped by physics domain, not by implementation artifact.
2. **Exhaustive** — every capability has exactly one home.
3. **Machine-readable** — enabling automated dashboards, CI gates, and billing aggregation.
4. **Extensible** — new domains can be added without restructuring existing ones.

## Decision

**All platform capabilities are organized into a taxonomy of 168 nodes across 20 domain packs.** Specifically:

1. Each domain pack corresponds to a physics vertical (e.g., `PHY-I: Core Tensor Algebra`, `PHY-IV: Electromagnetics`, `PHY-XI: Weather & Climate`).
2. Each node within a pack represents a specific capability (e.g., `PHY-IV.3: FDTD Maxwell Solver`).
3. Node definitions live in the capability ledger (`apps/ledger/nodes/*.yaml`) per ADR-0001.
4. Each node has a V-state (V0–V5) indicating maturity from concept to production-hardened.
5. The `PLATFORM_SPECIFICATION.md` §3 defines the full taxonomy tree.
6. Domain packs map to `tensornet/` module groups and are the unit of licensing and metering.
7. The `atlas_*.json` files in `data/atlas/` provide the benchmark reference data for each node.

### Taxonomy Structure (Top Level)

| Pack | Domain | Nodes |
|------|--------|-------|
| PHY-I | Core Tensor Algebra | 12 |
| PHY-II | Classical Fluids (CFD) | 15 |
| PHY-III | Structural Mechanics (FEA) | 10 |
| PHY-IV | Electromagnetics (CEM) | 8 |
| PHY-V | Quantum Many-Body | 10 |
| PHY-VI | Topology Optimization | 6 |
| PHY-VII | Molecular Dynamics | 8 |
| PHY-VIII | Drug Discovery | 7 |
| PHY-IX | Nuclear Engineering | 6 |
| PHY-X | Plasma Physics | 8 |
| PHY-XI | Weather & Climate | 9 |
| PHY-XII | Geophysics | 6 |
| PHY-XIII | Acoustics | 5 |
| PHY-XIV | Combustion | 7 |
| PHY-XV | Multiphysics Coupling | 8 |
| PHY-XVI | Digital Twins | 7 |
| PHY-XVII | Certification & V&V | 8 |
| PHY-XVIII | Trustless Physics (ZK) | 10 |
| PHY-XIX | AI/ML-Physics Hybrid | 9 |
| PHY-XX | Exascale Infrastructure | 9 |

## Consequences

- **Easier:** Customers can browse capabilities by domain — "show me all weather nodes at V3 or above."
- **Easier:** Coverage dashboards are automated from ledger data.
- **Easier:** Billing aggregates by domain pack — clear value-per-vertical pricing.
- **Harder:** Taxonomy maintenance — adding or moving nodes requires cross-team coordination.
- **Harder:** Some capabilities span multiple domains (e.g., fluid-structure interaction). Resolved by placing in the primary domain with cross-references.
- **Risk:** Taxonomy bloat — 168 nodes is already large. Mitigated by requiring evidence (test, benchmark, or proof) for each node to justify its existence.
