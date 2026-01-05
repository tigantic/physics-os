# Project HyperTensor: Constitutional Law

**Ratified**: 2025-12-20  
**Version**: 1.2.0  
**Authority**: Principal Investigator  
**Enforcement**: Mandatory for all contributions

---

## Preamble

This Constitution establishes the inviolable standards, protocols, and operational law governing all development within Project HyperTensor. These laws exist to ensure mathematical rigor, reproducibility, and the scientific integrity required for a system intended to solve physics in real-time on safety-critical platforms.

**Violation of any Article herein constitutes grounds for immediate rejection of contribution.**

---

## Article I: Mathematical Proof Standards

### Section 1.1 — Proof Requirements

Every algorithmic claim SHALL be accompanied by numerical proof satisfying:

1. **Reproducibility**: All tests use `torch.manual_seed(42)` or explicitly documented seeds
2. **Quantitative Measurement**: Errors measured numerically in float64 with explicit tolerances
3. **Falsifiability**: Clear PASS/FAIL criteria with numerical thresholds
4. **Reference Comparison**: Validation against analytical solutions, exact diagonalization, or established libraries (TeNPy, ITensor)

### Section 1.2 — Tolerance Hierarchy

| Category | Default Tolerance | Justification |
|----------|------------------|---------------|
| Machine Precision | `1e-14` | IEEE 754 float64 epsilon |
| Numerical Stability | `1e-10` | Accumulated roundoff |
| Algorithm Convergence | `1e-8` | Iterative method residuals |
| Physics Validation | `1e-6` | Discretization error |
| Benchmark Comparison | `5%` relative | Cross-library variance |

### Section 1.3 — Proof Artifacts

Every proof MUST generate:
- Human-readable Markdown report in `proofs/`
- Machine-readable JSON artifact with SHA256 hash
- Git commit hash linking to exact codebase state

---

## Article II: Code Architecture Standards

### Section 2.1 — Module Organization

```
Project HyperTensor/
├── tensornet/              # Core library (pip-installable)
│   ├── substrate/          # Layer 0: Field Oracle API (THE SPINE)
│   │   ├── field.py        # sample(), slice(), step(), stats(), serialize()
│   │   ├── stats.py        # FieldStats telemetry dashboard
│   │   ├── bundle.py       # FieldBundle serialization (.htf format)
│   │   └── bounded.py      # Bounded latency mode, caching
│   ├── fieldops/           # Layer 1: Physics operators (FieldGraph)
│   ├── visual/             # Layer 2: HyperVisual rendering
│   ├── envs/               # Layer 4: HyperEnv AI environments
│   ├── provenance/         # Layer 5: Attestation, audit, replay
│   ├── intent/             # Layer 6: Intentional fields, FDL
│   ├── runtime/            # Layer 7: Field scheduler, QoS
│   ├── core/               # Fundamental operations (SVD, QR, contractions, GPU)
│   ├── mps/                # Matrix Product States, Operators, Hamiltonians
│   ├── algorithms/         # DMRG, TEBD, TDVP, Lanczos, fermionic
│   ├── cfd/                # Computational fluid dynamics solvers
│   ├── quantum/            # Quantum-classical hybrid, error mitigation
│   ├── certification/      # DO-178C, hardware deployment
│   ├── neural/             # Neural-enhanced tensor networks
│   ├── autonomy/           # Mission planning, path planning, decisions
│   ├── coordination/       # Swarm, formation, consensus
│   ├── distributed_tn/     # Distributed DMRG, parallel TEBD
│   ├── validation/         # Physical validation, V&V
│   ├── simulation/         # HIL, mission simulation
│   ├── deployment/         # TensorRT, embedded systems
│   ├── ml_surrogates/      # PINNs, DeepONet, FNO
│   ├── digital_twin/       # State sync, anomaly detection
│   ├── adaptive/           # Bond optimization, compression
│   ├── realtime/           # Inference engine, memory management
│   ├── integration/        # Workflow orchestration
│   ├── guidance/           # 6-DOF trajectory, GNC
│   ├── flight_validation/  # Telemetry, flight data
│   ├── visualization/      # TensorSlicer, QTT rendering
│   ├── docs/               # API documentation generation
│   └── benchmarks/         # Internal benchmark utilities
├── benchmarks/             # Performance and accuracy benchmarks (Layer 3)
├── notebooks/              # Exploratory and demonstration notebooks
├── proofs/                 # Mathematical verification artifacts (proof_*.py)
├── tests/                  # pytest test suite
│   └── integration/        # Integration tests
├── docs/                   # Documentation and specifications
│   └── api/                # Auto-generated API reference
├── ROADMAP.md              # Strategic roadmap (see this document)
└── results/                # Generated outputs (gitignored except README)
```

**Amendment History**:
- v1.0.0 (2025-12-20): Initial ratification
- v1.1.0 (2025-12-20): Extended module structure to reflect 20-phase implementation
- v1.2.0 (2025-12-24): Added Substrate Layer architecture (Layer 0-8), ROADMAP.md reference

### Section 2.2 — Naming Conventions

| Entity | Convention | Example |
|--------|-----------|---------|
| Modules | `snake_case` | `hamiltonians.py` |
| Classes | `PascalCase` | `MatrixProductState` |
| Functions | `snake_case` | `heisenberg_mpo()` |
| Constants | `SCREAMING_SNAKE` | `DEFAULT_TOLERANCE` |
| Private | `_leading_underscore` | `_contract_left()` |
| Type hints | Required for public API | `def dmrg(psi: MPS, H: MPO) -> tuple[MPS, float, dict]:` |

### Section 2.3 — Tensor Index Conventions

All MPS tensors follow the canonical ordering:

```
A[i] : (χ_left, d, χ_right)
       └─ bond ─┘ └─ physical ─┘ └─ bond ─┘

MPO W[i] : (D_left, d_out, d_in, D_right)
           └─ bond ─┘ └── physical ──┘ └─ bond ─┘
```

**Violation**: Using (d, χ_left, χ_right) or other orderings without explicit documentation.

### Section 2.4 — Docstring Requirements

All public functions MUST include:

```python
def function_name(param: Type) -> ReturnType:
    """
    One-line summary ending with period.
    
    Extended description if necessary, including mathematical
    formulation in LaTeX: $H = \sum_i S_i \cdot S_{i+1}$
    
    Args:
        param: Description of parameter with units if applicable
        
    Returns:
        Description of return value(s)
        
    Raises:
        ValueError: When invalid input is provided
        
    Example:
        >>> result = function_name(input)
        >>> assert result == expected
        
    References:
        [1] Author, "Title", arXiv:XXXX.XXXXX (Year)
    """
```

---

## Article III: Testing Protocols

### Section 3.1 — Test Categories

| Category | Location | Trigger | Purpose |
|----------|----------|---------|---------|
| Unit Tests | `tests/test_*.py` | Every commit | Component correctness |
| Proof Tests | `proofs/proof_*.py` | Every release | Mathematical verification |
| Benchmarks | `benchmarks/*.py` | Weekly/Manual | Performance regression |
| Integration | `tests/integration/` | Pre-merge | End-to-end validation |

### Section 3.2 — Test Naming

```python
def test_<component>_<behavior>_<condition>():
    """Test that <component> <expected behavior> when <condition>."""
```

Example: `test_svd_truncated_preserves_frobenius_optimality()`

### Section 3.3 — Coverage Requirements

- **Core Library**: Minimum 90% line coverage
- **Algorithms**: Minimum 85% line coverage  
- **All New Code**: Must include tests in same PR

### Section 3.4 — Benchmark Baselines

Benchmarks MUST record:
- Exact hardware specification
- PyTorch version and CUDA version (if applicable)
- Wall-clock time with standard deviation (N≥3 runs)
- Memory high-water mark

---

## Article IV: Physics Validity Standards

### Section 4.1 — Hamiltonian Requirements

Every Hamiltonian implementation MUST verify:

1. **Hermiticity**: $\|H - H^\dagger\|_F < 10^{-14}$
2. **Correct Spectrum**: Ground state energy matches exact diagonalization for L ≤ 12
3. **Symmetry Preservation**: Conserved quantities remain constant under time evolution

### Section 4.2 — Canonical Model Benchmarks

New algorithms MUST pass validation against:

| Model | Observable | Reference |
|-------|-----------|-----------|
| Heisenberg L=10 | $E_0$ | -4.258035207282883 (exact) |
| TFIM g=1.0 L=10 | $E_0$ | -12.566370614359172 (exact) |
| Heisenberg L→∞ | $E_0/L$ | $1/4 - \ln(2)$ (Bethe ansatz) |
| TFIM critical | Central charge | $c = 0.5$ (Ising CFT) |

### Section 4.3 — Conservation Laws

Time evolution algorithms MUST conserve:
- **Energy**: $|\langle H(t) \rangle - \langle H(0) \rangle| / |E_0| < 10^{-6}$
- **Norm**: $|\langle\psi|\psi\rangle - 1| < 10^{-10}$
- **Symmetry Sectors**: $S_z$, particle number preserved exactly

---

## Article V: Numerical Stability Requirements

### Section 5.1 — Floating Point Discipline

1. **Default Precision**: `torch.float64` for all physics calculations
2. **Mixed Precision**: Permitted only with explicit error analysis
3. **Accumulation**: Summations use Kahan or pairwise summation for >1000 terms
4. **Condition Numbers**: Log warning when $\kappa > 10^{10}$

### Section 5.2 — SVD Truncation Protocol

```python
# REQUIRED: Truncation with explicit error tracking
U, S, Vh, info = svd_truncated(A, chi_max=chi, return_info=True)
assert info['truncation_error'] < tolerance
```

### Section 5.3 — Gradient Stability

Differentiable operations MUST:
- Pass `torch.autograd.gradcheck` with `eps=1e-6`
- Handle edge cases (zero singular values, degenerate eigenvalues)
- Document numerical radius of validity

---

## Article VI: Documentation Standards

### Section 6.1 — Required Documents

| Document | Location | Content |
|----------|----------|---------|
| README.md | Repository root | Installation, quickstart, badges |
| CONSTITUTION.md | Repository root | This document |
| ROADMAP.md | Repository root | Strategic roadmap, layer definitions, milestones |
| EXECUTION_TRACKER.md | Repository root | Project status and phase tracking |
| CHANGELOG.md | Repository root | Version history |
| API Reference | `docs/api/` | Auto-generated from docstrings |

### Section 6.2 — Notebook Standards

Jupyter notebooks MUST:
- Begin with installation/setup cell
- Include markdown explanations between code cells
- Run top-to-bottom without error
- Clear all outputs before commit (or use `nbstripout`)

### Section 6.3 — Citation Requirements

External algorithms MUST cite original papers:
```python
"""
DMRG algorithm following White (1992).

References:
    [1] S. R. White, "Density matrix formulation for quantum 
        renormalization groups", Phys. Rev. Lett. 69, 2863 (1992)
"""
```

---

## Article VII: Version Control Discipline

### Section 7.1 — Branch Strategy

```
main              ← Protected, requires PR + passing CI
├── develop       ← Integration branch
├── feature/*     ← New functionality
├── fix/*         ← Bug fixes  
├── proof/*       ← New mathematical proofs
└── experiment/*  ← Exploratory (not merged to main)
```

### Section 7.2 — Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types: `feat`, `fix`, `proof`, `docs`, `refactor`, `test`, `benchmark`

Example:
```
feat(algorithms): implement TDVP-2 time evolution

- Add two-site TDVP with adaptive bond dimension
- Verify energy conservation to 1e-8 over t=10

Refs: #42
```

### Section 7.3 — Pre-Commit Requirements

Before any commit:
1. `pytest tests/ -q` passes
2. `ruff check .` reports no errors
3. `ruff format --check .` passes
4. No secrets or API keys in diff

---

## Article VIII: Performance Standards

### Section 8.1 — Complexity Requirements

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| MPS contraction | $O(L \cdot \chi^3 \cdot d)$ | $O(L \cdot \chi^2 \cdot d)$ |
| DMRG sweep | $O(L \cdot \chi^3 \cdot d^2 \cdot D)$ | $O(L \cdot \chi^2 \cdot D)$ |
| TEBD step | $O(L \cdot \chi^3 \cdot d^2)$ | $O(\chi^2 \cdot d^2)$ |

Implementations exceeding these bounds require justification.

### Section 8.2 — Memory Discipline

1. **In-place Operations**: Prefer when mathematically equivalent
2. **Intermediate Cleanup**: Delete large temporaries explicitly
3. **Streaming**: For tensors > 1GB, use chunked processing
4. **Profiling**: Memory-intensive functions must include `@profile` decorator option

---

## Article IX: Security and Reproducibility

### Section 9.1 — Dependency Pinning

```toml
# pyproject.toml
[project]
dependencies = [
    "torch>=2.0.0,<3.0.0",
    "numpy>=1.24.0,<2.0.0",
]
```

### Section 9.2 — Environment Lockfile

Every release MUST include `requirements-lock.txt` generated via:
```bash
pip freeze > requirements-lock.txt
```

### Section 9.3 — Reproducibility Artifacts

Published results MUST include:
- Complete environment specification
- Random seed(s) used
- Exact command to reproduce
- Hardware specification

---

## Article X: Amendment Process

### Section 10.1 — Proposal

Amendments require:
1. Written proposal with justification
2. Review period of 7 days
3. Demonstration that change improves project integrity

### Section 10.2 — Ratification

Amendments are ratified by:
1. Principal Investigator approval
2. Updated version number (MAJOR.MINOR.PATCH)
3. Entry in CHANGELOG.md

---

## Appendix A: Quick Reference Checklist

### Before Committing Code

- [ ] All tests pass locally
- [ ] New code has docstrings with type hints
- [ ] Complex algorithms have mathematical proof
- [ ] No hardcoded paths or magic numbers
- [ ] Commit message follows format

### Before Opening PR

- [ ] Branch is up-to-date with `develop`
- [ ] All CI checks pass
- [ ] Documentation updated if API changed
- [ ] CHANGELOG.md updated for user-facing changes

### Before Release

- [ ] All proofs regenerated and passing
- [ ] Benchmarks show no regression
- [ ] Version number incremented
- [ ] requirements-lock.txt updated
- [ ] Release notes written

---

## Appendix B: Approved External Dependencies

| Package | Version | Purpose | Approval Status |
|---------|---------|---------|-----------------|
| torch | ≥2.0 | Core tensor operations | ✅ Approved |
| numpy | ≥1.24 | Array utilities | ✅ Approved |
| scipy | ≥1.10 | Sparse matrices, special functions | ✅ Approved |
| matplotlib | ≥3.7 | Visualization | ✅ Approved |
| pytest | ≥7.0 | Testing framework | ✅ Approved |
| tenpy | ≥0.10 | Benchmark reference | ✅ Approved (dev only) |

Adding new dependencies requires explicit justification and PI approval.

---

*This Constitution is the supreme law of Project HyperTensor. All code, documentation, and artifacts are subject to its provisions.*
