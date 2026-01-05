# HyperTensor Onboarding Guide

**Your First 30 Minutes with HyperTensor**

Welcome to HyperTensor! This guide will get you up and running with the project in under 30 minutes.

---

## ⏱️ Quick Start (5 minutes)

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/tigantic/hypertensor.git
cd hypertensor

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

### 2. Verify Installation

```bash
# Run quick tests
pytest tests/test_mps.py -v

# Import the library
python -c "import tensornet; print(f'Version: {tensornet.__version__}')"
```

---

## 🎯 Core Concepts (10 minutes)

### What is HyperTensor?

HyperTensor is a quantum-inspired tensor network library for:
- **Many-body physics**: DMRG, TEBD for quantum systems
- **CFD**: Tensor-compressed fluid dynamics solvers
- **QTT**: Quantized tensor train compression

### Key Data Structures

| Structure | Description | Use Case |
|-----------|-------------|----------|
| `MPS` | Matrix Product State | Quantum wavefunctions, 1D systems |
| `MPO` | Matrix Product Operator | Hamiltonians, observables |
| `QTT` | Quantized Tensor Train | High-dimensional field data |

### Hello World Example

```python
from tensornet import MPS, heisenberg_mpo, dmrg

# Create a 10-site Heisenberg chain Hamiltonian
H = heisenberg_mpo(L=10, J=1.0, Jz=1.0)

# Random initial MPS with bond dimension 32
psi = MPS.random(L=10, d=2, chi=32)

# Find ground state with DMRG
psi, energy, info = dmrg(psi, H, num_sweeps=10, chi_max=64)

print(f"Ground state energy: {energy:.6f}")
print(f"Final bond dimension: {psi.bond_dimensions}")
```

---

## 📁 Project Structure (5 minutes)

```
hypertensor/
├── tensornet/              # Main library
│   ├── core/               # MPS, MPO, decompositions
│   ├── algorithms/         # DMRG, TEBD, TDVP
│   ├── cfd/                # CFD solvers (Euler, QTT)
│   ├── mps/                # Hamiltonians, observables
│   └── ...
├── tests/                  # Unit and integration tests
├── proofs/                 # Mathematical proof scripts
├── demos/                  # Example applications
├── benchmarks/             # Performance benchmarks
└── docs/                   # Documentation
```

### Key Files to Know

| File | Purpose |
|------|---------|
| `tensornet/__init__.py` | Public API exports |
| `tensornet/core/mps.py` | MPS implementation |
| `tensornet/algorithms/dmrg.py` | DMRG algorithm |
| `tests/conftest.py` | Pytest fixtures |
| `pyproject.toml` | Project configuration |

---

## 🧪 Running Tests (5 minutes)

```bash
# Run all unit tests
pytest tests/ -v --ignore=tests/integration

# Run specific test file
pytest tests/test_dmrg.py -v

# Run tests with coverage
pytest tests/ --cov=tensornet --cov-report=html

# Run only fast tests
make test-fast

# Run physics validation tests
make test-physics
```

---

## 🔧 Development Workflow (5 minutes)

### 1. Create a Branch

```bash
git checkout -b feature/my-feature
```

### 2. Make Changes

Edit files in `tensornet/` and add tests in `tests/`.

### 3. Lint and Format

```bash
ruff check . --fix
ruff format .
```

### 4. Run Tests

```bash
pytest tests/ -v
```

### 5. Commit and Push

```bash
git add .
git commit -m "Add my feature"
git push origin feature/my-feature
```

### 6. Create Pull Request

Open a PR on GitHub using the PR template.

---

## 📚 Next Steps

1. **Read the [CONTRIBUTING.md](../CONTRIBUTING.md)** for coding standards
2. **Explore [demos/](../demos/)** for application examples
3. **Run [proofs/](../proofs/)** to see mathematical verification
4. **Check [ROADMAP.md](../ROADMAP.md)** for project direction

---

## 🆘 Getting Help

- **Questions?** Open a GitHub Discussion
- **Found a bug?** File an issue using the bug report template
- **Want to contribute?** Check issues labeled `good first issue`

---

## 📊 Quick Reference

### Common Commands

```bash
make test          # Run all tests
make lint          # Check code style
make format        # Auto-format code
make docs          # Build documentation
make clean         # Clean build artifacts
```

### Test Markers

```bash
pytest -m unit         # Fast unit tests only
pytest -m integration  # Integration tests
pytest -m physics      # Physics validation
pytest -m slow         # Long-running tests
pytest -m gpu          # GPU-required tests
```

---

*Welcome to the team! 🚀*
