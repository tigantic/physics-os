# Agent Context: Project The Physics OS

> **Purpose**: This document provides complete situational awareness for any agent or engineer joining the project. Read this first.

**Last Updated**: 2026-01-02  
**Version**: 1.5.0  
**Status**: Production-ready, V&V 100% Mature

---

## 🎯 Mission Statement

Project The Physics OS is a **proprietary physics simulation engine** that applies quantum-inspired tensor network mathematics (MPS, MPO, DMRG) to computational fluid dynamics. The core innovation: turbulent flow fields obey an **Area Law** (like quantum entanglement), enabling **10,000x compression** from O(N³) to O(N·D²).

**Owner**: Tigantic Holdings LLC  
**License**: Proprietary (see [LICENSE](LICENSE))

---

## 🏗️ Architecture At-A-Glance

```
The Physics OS/
├── ontic/           # Core Python library (147,000+ LOC)
│   ├── algorithms/      # DMRG, TEBD, Lanczos, fermionic ops
│   ├── mps/             # Matrix Product States
│   ├── mpo/             # Matrix Product Operators  
│   ├── cfd/             # CFD solvers (Euler, N-S, Riemann)
│   ├── cuda/            # GPU kernels, cuBLAS integration
│   ├── sovereign/       # Real-time streaming, Glass Cockpit bridge
│   ├── validation/      # V&V framework, MMS, benchmarks
│   ├── provenance/      # Merkle DAG, cryptographic signing
│   └── [40+ domain modules]
├── tests/               # 1,124 test cases
│   ├── integration/     # MMS, physics validation, benchmarks
│   └── unit/            # Component-level tests
├── apps/
│   ├── glass_cockpit/   # Python side of real-time viz
│   └── global_eye/      # Planetary monitoring system
├── glass-cockpit/       # Rust visualizer (120 FPS)
├── crates/              # Rust workspace (hyper_core, hyper_bridge)
├── docs/                # Organized documentation
├── scripts/             # Automation, profiling, testing utilities
├── proofs/              # Mathematical verification scripts
└── media/               # Videos, rendered outputs
```

---

## 🔑 Critical Files (Read These)

| File | Purpose | Priority |
|------|---------|:--------:|
| [CONSTITUTION.md](../governance/CONSTITUTION.md) | Coding standards, governance, principles | 🔴 HIGH |
| [HYPERTENSOR_VV_FRAMEWORK.md](HYPERTENSOR_VV_FRAMEWORK.md) | V&V methodology, ASME/NASA alignment | 🔴 HIGH |
| [docs/INDEX.md](docs/INDEX.md) | Master documentation hub | 🟡 MEDIUM |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines | 🟡 MEDIUM |
| [CHANGELOG.md](CHANGELOG.md) | Version history, recent changes | 🟢 REFERENCE |

---

## 💻 Technology Stack

### Core Languages
- **Python 3.11+**: Primary implementation (tensornet library)
- **Rust 1.70+**: Glass Cockpit visualizer, performance-critical paths
- **CUDA**: GPU kernels for tensor operations

### Key Dependencies
```
torch >= 2.0          # Tensor backend, GPU support
numpy >= 1.24         # Numerical operations
scipy                 # Scientific computing
pytest                # Testing framework
ruff, black, isort    # Code formatting (enforced)
mypy                  # Type checking
```

### Rust Crates
```
glass-cockpit/        # Real-time visualization (egui, wgpu)
crates/hyper_core/    # Core tensor operations
crates/hyper_bridge/  # Python-Rust FFI bridge
```

---

## 🧠 Core Concepts

### Tensor Networks
- **MPS (Matrix Product State)**: 1D tensor chain representation
- **MPO (Matrix Product Operator)**: Operators acting on MPS
- **DMRG**: Density Matrix Renormalization Group (ground state solver)
- **TEBD**: Time-Evolving Block Decimation (dynamics)
- **QTT**: Quantized Tensor Train (multi-scale compression)

### CFD Integration
- Navier-Stokes equations represented in tensor format
- Riemann solvers (Godunov, Roe, HLLC)
- TVD limiters (minmod, superbee, van Leer)
- 7 canonical benchmarks validated (Sod, Lax, Taylor-Green, etc.)

### Key Insight
```
Traditional CFD:  O(N³) memory, O(N⁴) compute
Tensor Train:     O(N·D²) memory, O(N·D³) compute
                  where D = bond dimension << N
```

---

## 📊 Current State (As of v1.5.0)

### ✅ Completed Phases
| Phase | Deliverable | Status |
|:-----:|-------------|:------:|
| 1 | Tensor Network Core (MPS, MPO, DMRG) | ✅ |
| 2 | 1D/2D CFD Solvers | ✅ |
| 3 | Glass Cockpit (120 FPS visualization) | ✅ |
| 4 | Multi-Domain Physics (15 domains) | ✅ |
| 5 | V&V Framework (ASME aligned) | ✅ |

### 🔄 In Progress
| Phase | Objective | Target |
|:-----:|-----------|:------:|
| 6 | 3D N-S with QTT Compression | Q2 2026 |
| 7 | Digital Twin Integration | Q3 2026 |
| 8 | Cloud Deployment (AWS/GCP) | Q4 2026 |

### V&V Maturity: 100%
- Code Verification: MMS tests, unit tests, conservation laws
- Solution Verification: Grid convergence (GCI), error estimation
- Validation: 7/7 canonical benchmarks passing
- Provenance: Dilithium2 (PQC) signing, Merkle DAG
- Reproducibility: Deterministic execution, CI/CD pipelines

---

## 🌐 15 Physics Domains

Each domain has validated benchmarks. Key modules:

| Domain | Module | Benchmark |
|--------|--------|-----------|
| CFD Core | `ontic.cfd` | Sod shock tube |
| Hypersonic | `ontic.hypersim` | Sutton-Graves |
| GPU/CUDA | `ontic.cuda` | cuBLAS validated |
| Fusion | `ontic.fusion` | Gyration ratio |
| Medical | `ontic.medical` | Poiseuille flow |
| Urban | `ontic.urban` | Venturi analytical |
| Defense | `ontic.defense` | Snell's law |
| Financial | `ontic.financial` | Conservation laws |
| Energy | `ontic.energy` | Betz limit |
| Emergency | `ontic.emergency` | Fire diffusion |

See README for complete list.

---

## 🧪 Testing Requirements

### Running Tests
```bash
# Activate venv
source .venv/bin/activate

# Full suite
pytest tests/ -v

# By category
pytest -m unit          # Fast unit tests
pytest -m benchmark     # Known-solution validation
pytest -m mms           # Method of Manufactured Solutions
pytest -m conservation  # Conservation law verification
pytest -m physics       # Physics validation
```

### Test Markers
All physics code MUST have:
- `@pytest.mark.benchmark` for known-solution tests
- `@pytest.mark.mms` for manufactured solution verification
- `@pytest.mark.conservation` for conservation law checks

### Coverage Requirements
- Core modules: 90%+ coverage
- New physics: MMS or benchmark validation required

---

## 📝 Code Standards (CONSTITUTION.md)

### Must-Have
- ✅ Type hints on ALL public functions
- ✅ Docstrings with mathematical notation (LaTeX OK)
- ✅ Proofs for algorithmic claims (in `proofs/`)
- ✅ Black/isort/ruff formatting (CI enforced)
- ✅ No `print()` in library code (use `logging`)

### Naming Conventions
```python
# Classes: PascalCase
class TensorNetwork:

# Functions/methods: snake_case  
def compute_ground_state():

# Constants: UPPER_SNAKE
MAX_BOND_DIMENSION = 256

# Private: leading underscore
def _internal_helper():
```

### Commit Messages
Use conventional commits:
```
feat: Add 3D Euler solver
fix: Correct boundary condition in HLLC
docs: Update V&V framework documentation
test: Add MMS tests for advection
refactor: Simplify tensor contraction logic
chore: Update dependencies
```

---

## 🔧 Development Workflow

### Git Remotes
```bash
origin     → https://github.com/tigantic/The Physics OS.git      # Primary
vm         → https://github.com/tigantic/HyperTensor-VM.git   # Mirror
```

Push to both:
```bash
git push origin main && git push vm main
```

### Branch Strategy
- `main`: Production-ready, V&V validated
- `develop`: Integration branch (if used)
- `feature/*`: New features
- `fix/*`: Bug fixes

### Pre-Commit
```bash
pip install pre-commit
pre-commit install
```

---

## ⚠️ Common Pitfalls

### 1. Tensor Dimension Ordering
```python
# CORRECT: (batch, chi_left, d, chi_right)
tensor.shape = (B, chi_l, d, chi_r)

# WRONG: Mixing up bond dimensions
```

### 2. GPU Memory
```python
# Always move tensors explicitly
tensor = tensor.to(device)

# Clear cache when needed
torch.cuda.empty_cache()
```

### 3. Numerical Precision
```python
# Use float64 for physics validation
dtype=torch.float64

# Conservation tests expect machine epsilon (~1e-15)
```

### 4. Import Cycles
- Run `python tools/scripts/check_import_cycles.py` before committing
- Avoid circular imports between tensornet submodules

### 5. File Naming
- When saving files, avoid double extensions (`.png.png`)
- Use snake_case for all Python files

---

## 🚧 Current Pain Points / Known Issues

> **Important**: These are known limitations. Don't waste cycles rediscovering them.

### Technical Debt

| Issue | Impact | Workaround | Priority |
|-------|--------|------------|:--------:|
| **CI workflows `continue-on-error`** | Lint/test failures don't block merge | Manual review required | 🟡 |
| **560 remaining ruff warnings** | Unused variables, minor issues | Non-blocking, fix opportunistically | 🟢 |
| **Some tests require GPU** | Skip on CPU-only CI runners | `pytest -m "not gpu"` | 🟡 |
| **Large image files in repo** | Slows clone (~50MB in images/) | Consider Git LFS for future | 🟢 |

### Known Limitations

| Limitation | Details | Planned Fix |
|------------|---------|-------------|
| **3D N-S not production-ready** | Phase 6 in progress, basic implementation exists | Q2 2026 |
| **QTT compression ratio varies** | Depends on flow smoothness; shocks need higher bond dim | Adaptive bonding (research) |
| **Glass Cockpit Windows-only tested** | Linux/Mac may have shared memory path issues | Cross-platform testing needed |
| **No cloud deployment yet** | Phase 8 planned | Q4 2026 |

### Performance Bottlenecks

| Bottleneck | Location | Notes |
|------------|----------|-------|
| **Tensor contraction** | `ontic/core/` | GPU helps, but large chi still slow |
| **DMRG sweeps** | `ontic/algorithms/dmrg.py` | O(chi³) per site; chi > 256 gets expensive |
| **QTT 2D shifts** | `ontic/cfd/qtt_2d_shift.py` | Boundary handling adds overhead |
| **Shared memory latency** | Python → Rust bridge | ~1-2ms per frame; acceptable for 120 FPS |

### Flaky Tests (Intermittent Failures)

| Test | Reason | Status |
|------|--------|--------|
| `test_cuda_kernel.py` | GPU memory contention on shared runners | Skip in CI, run locally |
| `test_determinism.py` | CUDA non-determinism edge cases | Use `CUBLAS_WORKSPACE_CONFIG` |
| `test_parallel.py` | Race conditions in MPI tests | Needs refactor |

### Environment Quirks

```bash
# Windows: Zone.Identifier files from downloads
# Fix: Delete them or add to .gitignore
find . -name "*:Zone.Identifier" -delete

# WSL2: Shared memory path differs
# Glass Cockpit expects /dev/shm on Linux

# Apple Silicon: Some CUDA tests skip automatically
# Use CPU backend for development
```

### API Rough Edges

| API | Issue | Recommended Usage |
|-----|-------|-------------------|
| `MPS.random()` | Doesn't normalize by default | Call `.normalize()` after |
| `Euler1D.step()` | CFL not auto-calculated | Manually set `dt` conservatively |
| `dmrg()` return signature | Returns tuple, order matters | `psi, E, info = dmrg(...)` |
| `ontic.cfd` imports | Some modules have lazy imports | Import specific classes directly |

### Documentation Gaps

- [ ] `ontic/hypersim/` lacks docstrings in some modules
- [ ] `ontic/sovereign/` protocol documentation incomplete
- [ ] Rust crate documentation minimal (use `cargo doc`)
- [ ] Some `proofs/` scripts reference outdated module paths

---

## 🖥️ Glass Cockpit (Real-Time Viz)

### Architecture
```
Python (tensornet)          Rust (glass-cockpit)
      │                            │
      │  Shared Memory (12MB)      │
      └──────────────────────────►│
         Zero-copy tensor data     │
                                   ▼
                            egui/wgpu render
                               120 FPS
```

### Running
```bash
# Terminal 1: Python streamer
python -c "
from ontic.sovereign.realtime_tensor_stream import test_realtime_stream
test_realtime_stream(duration=60, pattern='turbulence', fps=120)
"

# Terminal 2: Rust visualizer
cd glass-cockpit && cargo run --release --bin phase3
```

---

## 📁 Key Directories

| Path | Contents |
|------|----------|
| `ontic/sovereign/` | Real-time streaming, QTT extraction |
| `ontic/cfd/` | All CFD solvers, Riemann, WENO |
| `ontic/validation/` | V&V framework, regression detection |
| `tests/integration/` | MMS tests, benchmark validations |
| `tools/scripts/` | Automation, profiling, CI helpers |
| `proofs/` | Mathematical verification scripts |
| `docs/architecture/` | System design documents |
| `docs/phases/` | Phase completion reports |

---

## 🔗 Quick Links

### Documentation
- [README.md](README.md) - Project overview
- [docs/INDEX.md](docs/INDEX.md) - Documentation hub
- [docs/architecture/](docs/architecture/) - System design

### V&V
- [HYPERTENSOR_VV_FRAMEWORK.md](HYPERTENSOR_VV_FRAMEWORK.md) - V&V methodology
- [tests/integration/test_*_mms.py](tests/integration/) - MMS tests
- [docs/audits/](docs/audits/) - Audit reports

### Development
- [CONTRIBUTING.md](CONTRIBUTING.md) - How to contribute
- [CONSTITUTION.md](../governance/CONSTITUTION.md) - Standards and governance
- [.github/workflows/](../.github/workflows/) - CI/CD pipelines

---

## 🚀 Getting Started (New Agent Checklist)

1. ☐ Read this document completely
2. ☐ Skim [CONSTITUTION.md](../governance/CONSTITUTION.md) for code standards
3. ☐ Review [HYPERTENSOR_VV_FRAMEWORK.md](HYPERTENSOR_VV_FRAMEWORK.md) for V&V requirements
4. ☐ Understand the tensornet module structure (`ls ontic/`)
5. ☐ Run tests to verify environment: `pytest tests/ -x -v --tb=short`
6. ☐ Check current git status: `git log --oneline -10`
7. ☐ Review recent changes: `git diff HEAD~5 --stat`

---

## 📞 Context for Conversation

When resuming work, the agent should:

1. **Check conversation history** for recent tasks/decisions
2. **Verify file state** before editing (use `read_file` or `grep_search`)
3. **Run tests** after changes to physics code
4. **Push to both remotes** after commits: `origin` and `vm`
5. **Update CHANGELOG.md** for significant changes

---

## 🎖️ Success Criteria

A task is complete when:

- ✅ Code follows CONSTITUTION.md standards
- ✅ Tests pass (`pytest tests/ -v`)
- ✅ No ruff/mypy errors on changed files
- ✅ Changes committed with conventional commit message
- ✅ Pushed to both `origin` and `vm` remotes
- ✅ User confirms acceptance

---

*"In God we trust. All others must bring data." — W. Edwards Deming*

**You are now ready to execute. Welcome to Project The Physics OS.**
