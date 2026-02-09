# ADR-0011: Phase 7 — Productization & Ecosystem Hardening

**Status:** Accepted  
**Date:** 2026-02-09  
**Decision Makers:** Platform team  

## Context

Phases 0–6 built the entire computational physics platform: canonical
protocols, data model, solver orchestration, 20 domain packs (167 taxonomy
nodes), verification & validation, QTT acceleration, coupled physics,
inverse problems, UQ, and optimization.  All internal capabilities are
functional and tested (268+ tests passing).

The next challenge is **external usability**: can a team that did not build
this platform pick it up, compose workflows, export results, and upgrade
versions without breaking?  This is the Phase 7 exit gate from
`Commercial_Execution.md`:

> "External teams can build on it without internal support, and upgrade
> versions without breaking workflows."

## Decision

### 1. Stable SDK Surface (`tensornet/sdk/`)

**Architecture:** Thin re-export layer + fluent `WorkflowBuilder`.

- `tensornet.sdk.__init__` re-exports a curated, stable subset of platform
  internals.  External code imports from `tensornet.sdk`, never from
  `tensornet.platform.*` directly.
- `WorkflowBuilder` provides a fluent DSL: `.domain()` → `.field()` →
  `.solver()` → `.time()` → `.export()` → `.build()` → `.run()`.
- `ExecutedWorkflow.run()` handles the full pipeline: mesh generation,
  initial conditions, solver dispatch, observable recording, export, and
  provenance capture.
- `RunResult` bundles `SolveResult`, `SimulationState`, provenance dict,
  exported file paths, wall time, and lineage DAG.

**Rationale:** A single builder pattern reduces the cognitive load from
~40 platform classes to one entry point.  Re-exports decouple consumers
from internal refactors.

### 2. Recipe Book (`tensornet/sdk/recipes.py`)

**Architecture:** Global registry + per-domain factory functions.

- `register_recipe()` decorator registers factory functions.
- `get_recipe(name, **kwargs)` returns a pre-configured `WorkflowBuilder`.
- `list_recipes(domain=...)` enables discovery.
- Built-in recipes: `harmonic_oscillator`, `lorenz_attractor`, `burgers_1d`,
  `sod_shock_tube`, `maxwell_1d`, `advection_diffusion_1d`,
  `heisenberg_chain`, `landau_damping`, `kohn_sham_1d`.

**Rationale:** Recipes encode domain expertise (mesh resolution, CFL,
boundary conditions) so novice users can run meaningful simulations
immediately.  Power users customize by chaining additional builder calls.

### 3. Export / Import (`tensornet/platform/export.py`, `mesh_import.py`)

**Export formats:**

| Format | Library | Use case |
|--------|---------|----------|
| VTU (VTK XML) | stdlib `xml.etree` + `base64` | ParaView / VisIt interop |
| XDMF + HDF5 | `h5py` (optional) | Large-scale parallel I/O |
| CSV | stdlib `csv` | Observable time series → Excel / pandas |
| JSON | stdlib `json` | Provenance metadata, CI artifacts |

**Import formats:**

- GMSH v2 ASCII
- GMSH v4 ASCII (non-sequential node ID remapping)
- Raw arrays (torch/numpy → `UnstructuredMesh`)

**Rationale:** VTU is the de facto standard for scientific visualization.
GMSH is the most common open-source mesher.  CSV/JSON cover lightweight
data exchange.

### 4. Post-processing (`tensornet/platform/postprocess.py`)

Operations: `probe()`, `slice_field()`, `integrate()`, `field_statistics()`,
`fft_field()`, `gradient_field()`, `histogram()`.

All operate on `FieldData` and return torch Tensors or dataclasses.
Structured meshes get fast-path implementations (index-based probe, axis
slicing).

**Rationale:** Users should not have to write their own numpy/scipy
post-processing glue.  Built-in ops cover 90% of common analysis tasks.

### 5. Visualization (`tensornet/platform/visualize.py`)

Matplotlib-based (optional dependency).  Functions: `plot_field_1d()`,
`plot_field_2d()`, `plot_convergence()`, `plot_observable_history()`,
`plot_spectrum()`.  All return `(fig, ax)` and accept `save_path`.

**Rationale:** Quick-look plotting should be built in.  Matplotlib is
ubiquitous in the Python scientific ecosystem.  Making it optional avoids
bloating minimal installations.

### 6. Deprecation Policy (`tensornet/platform/deprecation.py`)

- `VersionInfo` frozen dataclass: parses SemVer, supports comparison.
- `PLATFORM_VERSION = VersionInfo(2, 0, 0)`.
- `@deprecated(removal_version, alternative)`: emits `DeprecationWarning`
  if `PLATFORM_VERSION < removal_version`; raises `RuntimeError` if >=.
- `@since(version)`: documents when an API was introduced.
- `check_version_gate()`: CI-callable scan for overdue removals.

**Rationale:** Explicit version gates prevent "accidental API retention."
The CI check ensures deprecated code is actually removed on schedule.

### 7. Security Posture (`tensornet/platform/security.py`)

- `generate_sbom()` — CycloneDX-lite SBOM from `importlib.metadata`.
- `audit_dependencies()` — Offline heuristic check against known CVEs.
- `license_audit()` — Classifies packages as permissive/copyleft/unknown.

**Rationale:** Supply-chain security is non-negotiable for production.
Offline checks provide a baseline; production deployments should integrate
Trivy/Grype for real-time scanning.

### 8. CI/CD Hardening (`.github/workflows/hardening.yml`)

Three jobs:

1. **full-test** — Matrix (Python 3.11, 3.12), runs all 5 test files.
2. **determinism** — Runs twice with same seed, compares SHA-256 hashes.
3. **supply-chain** — SBOM generation, license audit, deprecation gate.

**Rationale:** Determinism gate catches subtle non-determinism regressions.
Supply-chain job enforces the security posture automatically.

## Consequences

- External teams can compose simulations via `WorkflowBuilder` without
  reading platform internals.
- Version upgrades are safe: deprecated APIs warn before removal, and
  CI enforces the schedule.
- Results are interoperable with ParaView, VisIt, pandas, Excel.
- Security is auditable: SBOM + license report generated on every push.

## Test Coverage

55 new tests in `tests/test_productization.py`:

- Export: VTU 1D/2D, CSV roundtrip, JSON, ExportBundle
- Mesh import: raw arrays (2D, 3D), GMSH v2, GMSH v4, format detection
- Post-processing: probe 1D/2D, slice, integrate, statistics, FFT,
  gradient, histogram
- Deprecation: VersionInfo parse/compare, @deprecated warning/raise,
  @since, check_version_gate
- Security: SBOM generation, dependency audit, license audit
- SDK: all re-exports importable, WorkflowBuilder config/validation/chaining
- Recipes: list, filter by domain, get with overrides, unknown raises
- Visualization: ensure_matplotlib, plot_field_1d, plot_field_2d,
  plot_convergence
- Platform __init__: version updated, new exports present

## Metrics

| Metric | Value |
|--------|-------|
| New modules | 10 (7 platform + 3 SDK) |
| New tests | 55 |
| Total tests passing | 268+ (1 skipped) |
| Platform version | 2.0.0 |
| SDK version | 2.0.0 |
| Recipes | 8 built-in |
| Export formats | 4 (VTU, XDMF+HDF5, CSV, JSON) |
| Import formats | 3 (GMSH v2, GMSH v4, raw) |
