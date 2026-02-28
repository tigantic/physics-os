# Rank Atlas Benchmark Specification — Version 1.0

**Status:** Released  
**Date:** 2026-02-24  
**Maintainer:** Brad Tigantic (Physics OS Project)  
**Repository:** physics-os  
**Companion Papers:**
- Paper A: *χ-Regularity and Rank Atlas* (`docs/research/paper_a_chi_regularity_atlas.md`)
- Paper B: *QTT Physics VM* (`docs/research/paper_b_qtt_physics_vm.md`)

---

## 1. Purpose

The Rank Atlas Benchmark measures **QTT bond-dimension behavior across
physics domains** as a function of grid resolution and problem complexity.
It provides a standardized, reproducible protocol for evaluating whether
QTT compression achieves bounded or polylogarithmic rank growth for PDE
solutions — the empirical foundation of the χ-regularity conjecture.

The benchmark is designed to be:

1. **Reproducible.** Deterministic configurations, fixed random seeds,
   pinned software versions.
2. **Extensible.** New physics domains can be added by implementing the
   pack interface and registering a configuration.
3. **Falsifiable.** Explicit acceptance/rejection criteria are built into
   the validation checker.
4. **Vendor-neutral.** Results from any QTT/TT solver can be submitted
   in the standard schema.

---

## 2. Measurement Protocol

### 2.1 Core Protocol

For each (pack, complexity, n_bits) triple:

1. **Instantiate** the domain pack's anchor problem with the specified
   complexity parameter.
2. **Initialize** the QTT state with `max_rank = 2048` and
   `svd_tolerance = 1e-6`.
3. **Evolve** the solution for the problem-specific integration time
   (defined per pack in the configuration).
4. **Extract** the QTT bond dimensions and full singular value spectra.
5. **Compute** entanglement entropy via von Neumann formula.
6. **Validate** physics by checking the domain-specific invariant.
7. **Record** the measurement in the standard schema (Section 3).

### 2.2 Resolution Sweep

Each pack must be measured at resolutions:

$$n_{\text{bits}} \in \{6, 7, 8, 9, 10\}$$

corresponding to $N = 64$ to $1024$ grid points per axis.

### 2.3 Complexity Sweep

For V0.4 packs (those with adjustable complexity parameters), the
complexity parameter must be swept through **10 logarithmically-spaced
values** between the configured `sweep_lo` and `sweep_hi`.

For V0.2 packs (fixed complexity), a single complexity value is used.

### 2.4 Trials

Each (pack, complexity, n_bits) configuration must be run with **3
independent trials** using seeds `{42, 137, 2026}`.

### 2.5 Baselines

Two baseline measurement modes are defined:

- **Baseline A (in-solver QTT):** The standard protocol above — extract
  bond dimensions from the QTT solver state after time integration.
- **Baseline B (dense-to-QTT encode):** Solve the same problem with a
  dense (non-QTT) solver using identical parameters. Export the final
  dense field. Compress to QTT via standalone TT-SVD at the same
  `svd_tolerance`. Record bond dimensions of the compressed result.

Baseline B measures *intrinsic* field compressibility without solver
truncation history. The comparison between A and B isolates solver-induced
rank overhead.

---

## 3. Measurement Schema

All measurements must conform to the JSON schema defined in
`benchmarks/rank_atlas/schema.json`.

### 3.1 Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `pack_id` | string | Pack identifier (I–XX) |
| `domain_name` | string | Physics domain name |
| `problem_name` | string | Specific problem within the pack |
| `n_bits` | integer | Binary resolution (6–14) |
| `n_cells` | integer | Total grid cells = `(2^n_bits)^d` |
| `complexity_param_name` | string | Name of swept parameter |
| `complexity_param_value` | number | Value of swept parameter |
| `svd_tolerance` | number | SVD truncation tolerance (default: 1e-6) |
| `max_rank_ceiling` | integer | Maximum allowed bond dimension (≥ 2048) |
| `n_sites` | integer | Number of QTT sites = `d * n_bits` |
| `bond_dimensions` | array[integer] | Bond dimension at each of `n_sites - 1` bonds |
| `singular_value_spectra` | array[array[number]] | Full SV spectrum at each bond |
| `max_rank` | integer | max(bond_dimensions) |
| `mean_rank` | number | mean(bond_dimensions) |
| `median_rank` | integer | median(bond_dimensions) |
| `rank_std` | number | std(bond_dimensions) |
| `peak_rank_site` | integer | Site index of maximum rank |
| `rank_utilization` | number | mean_rank / max_rank |
| `dense_bytes` | integer | Dense storage in bytes |
| `qtt_bytes` | integer | QTT storage in bytes |
| `compression_ratio` | number | dense_bytes / qtt_bytes |
| `wall_time_s` | number | Wall time in seconds |
| `peak_gpu_mem_mb` | number | Peak GPU memory in MB (0 if CPU) |
| `device` | string | "cuda" or "cpu" |
| `timestamp` | string | ISO 8601 timestamp |
| `seed` | integer | Random seed used |

### 3.2 Optional Fields (Extensions)

| Field | Type | Description |
|-------|------|-------------|
| `sv_decay_exponents` | array[number] | Per-bond SV decay fit exponent |
| `sv_decay_mean` | number | Mean SV decay exponent |
| `spectral_gaps` | array[number] | σ₁/σ₂ at each bond |
| `bond_entropies` | array[number] | Von Neumann entropy at each bond |
| `total_entropy` | number | Sum of bond entropies |
| `entropy_density` | number | total_entropy / n_sites |
| `effective_ranks` | array[number] | e^{S_k} at each bond |
| `area_law_exponent` | number | Entanglement scaling fit exponent |
| `area_law_r_squared` | number | Fit quality |
| `area_law_type` | string | "area" / "volume" / "log_corrected" |
| `reconstruction_error` | number | ‖u_QTT − u_ref‖ / ‖u_ref‖ |
| `physics_invariant_name` | string | Name of conserved quantity |
| `physics_invariant_value` | number | Relative invariant violation |
| `physics_invariant_passed` | boolean | Whether invariant test passed |

---

## 4. Pack Configurations

### 4.1 V0.4 Packs (Complexity Sweeps)

| Pack | Taxonomy ID | Complexity Parameter | Sweep Lo | Sweep Hi |
|------|------------|---------------------|----------|----------|
| II | PHY-II.1 | nu (viscosity) | 0.001 | 0.1 |
| III | PHY-III.3 | sigma_pulse (width) | 0.05 | 2.0 |
| V | PHY-V.5 | alpha (diffusivity) | 0.0001 | 1.0 |
| VII | PHY-VII.2 | J (exchange coupling) | 0.1 | 10.0 |
| VIII | PHY-VIII.1 | Z (atomic number) | 1.0 | 10.0 |
| XI | PHY-XI.1 | epsilon (perturbation) | 0.001 | 0.5 |

### 4.2 V0.2 Packs (Fixed Complexity)

| Pack | Taxonomy ID | Problem |
|------|------------|---------|
| I | PHY-I.1 | N-body gravity |
| IV | PHY-IV.1 | Ray tracing |
| VI | PHY-VI.1 | Band structure |
| IX | PHY-IX.1 | Shell model |
| X | PHY-X.4 | Lattice QCD |
| XII | PHY-XII.1 | Stellar structure |
| XIII | PHY-XIII.1 | Seismic wave |
| XIV | PHY-XIV.1 | Molecular dynamics |
| XV | PHY-XV.1 | Reaction kinetics |
| XVI | PHY-XVI.1 | Crystal growth |
| XVII | PHY-XVII.1 | Linear acoustics |
| XVIII | PHY-XVIII.1 | Weather prediction |
| XIX | PHY-XIX.6 | Quantum sensing |
| XX | PHY-XX.1 | Solitons |

---

## 5. Acceptance / Rejection Criteria

### 5.1 Grid Independence (Conjecture A)

For each (pack, complexity) pair, fit:

$$\chi_{\max} = a + b \cdot n_{\text{bits}}$$

**Accept** if $|b| / a < 0.05$ (rank changes by less than 5% per
resolution doubling).

### 5.2 Polylogarithmic Growth (Conjecture B)

For packs with non-constant rank growth, fit:

$$\chi_{\max} = a + b \cdot n_{\text{bits}}^q$$

**Accept** Conjecture B if $q \leq 2$.

### 5.3 Scaling Classification

| Class | Criterion | Interpretation |
|-------|-----------|----------------|
| A (Bounded) | α < 0.1 and χ_max < 50 | Rank effectively constant |
| B (Weak growth) | 0.1 ≤ α < 0.5 | Rank grows sublinearly |
| C (Strong growth) | α ≥ 0.5 | Conjecture weakened |
| D (Divergent) | χ_max → max_rank_ceiling | Conjecture falsified |

### 5.4 Universality

Compute gap statistic (Tibshirani et al., 2001) over per-pack feature
vectors. **Accept** universality if gap test selects $k = 1$ or
silhouette-optimal $k = 2$ does not align with PDE class boundaries.

### 5.5 Overall Verdict

The benchmark is **PASSED** if:
- All 20 packs are Class A or B (no Class D)
- Grid independence holds for ≥ 18/20 packs
- Gap statistic selects k = 1

The benchmark is **FAILED** if:
- ≥ 3 packs are Class D
- Clusters align with PDE classification boundaries (k ≥ 3)

---

## 6. Result Submission Format

Results must be submitted as a single JSON file containing an array of
measurement objects conforming to the schema. The file should be named:

```
rank_atlas_results_<implementation>_<date>.json
```

Example: `rank_atlas_results_hypertensor_20260224.json`

The validation checker (`benchmarks/rank_atlas/validate.py`) must pass
with zero errors before results are considered valid.

---

## 7. Reference Results

### 7.1 HyperTensor Baseline (2026-02-24)

| Metric | Value |
|--------|-------|
| Total measurements | 751+ |
| Packs measured | 20/20 |
| Resolution range | n_bits 4–12 |
| Class D packs | 0 |
| Grid independence (pack-level) | 16/20 strict, ≥18/20 lenient |
| Gap statistic k | 1 |
| Verdict | **PASSED** |

Reference data files:
- `rank_atlas_20pack.json` (352 measurements, n_bits 4–7)
- `rank_atlas_deep_III_VI.json` (162 measurements, n_bits 4–9)
- `data/rank_atlas_v04_nbits10.json` (180 measurements, n_bits 10)
- `data/rank_atlas_v02_nbits10.json` (42 measurements, n_bits 10)
- `data/rank_atlas_pack_vi_highres.json` (15 measurements, n_bits 8–12)

---

## 8. Versioning

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-24 | Initial specification. 20 packs, 5 resolutions, 3 trials, standardized schema, validation checker, acceptance criteria. |

---

*Rank Atlas Benchmark v1.0 — Physics OS Project*
