# Contributing to HyperTensor

Thank you for your interest in contributing to HyperTensor! This document provides a quick-start guide for new contributors.

## Quick Start

1. **Fork & Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/hypertensor.git
   cd hypertensor
   ```

2. **Set Up Environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   pip install -e ".[dev]"
   ```

3. **Run Tests**
   ```bash
   pytest tests/ -v
   ```

## Finding Issues to Work On

Look for issues labeled:
- 🟢 **good first issue** — Great for newcomers
- 📖 **documentation** — Help improve docs
- 🧪 **test coverage** — Add tests for uncovered code

## Code Standards

### TL;DR of the Constitution

Our [Constitution](CONSTITUTION.md) outlines coding standards. Key points:

1. **Test Naming**: `test_<module>_<function>_<scenario>_<expected>`
   ```python
   def test_dmrg_ground_state_heisenberg_converges():
   ```

2. **Docstrings**: NumPy style with Args, Returns, Examples
   ```python
   def compute_energy(H, psi):
       """Compute expectation value <psi|H|psi>.
       
       Args:
           H: Hamiltonian MPO
           psi: MPS wavefunction
           
       Returns:
           float: Energy expectation value
       """
   ```

3. **Type Hints**: Required for public APIs
   ```python
   def dmrg(H: MPO, chi_max: int = 64, ...) -> DMRGResult:
   ```

4. **Proof Tests**: New algorithms need proof tests that validate correctness
   ```python
   # In proofs/proof_my_algorithm.py
   def test_algorithm_satisfies_invariant():
       ...
   ```

## Pull Request Process

1. Create a branch: `git checkout -b feature/your-feature`
2. Make changes and add tests
3. Run the test suite: `pytest tests/ -v`
4. Run linting: `ruff check . && black --check .`
5. Submit PR with clear description

## ⚠️ REQUIRED READING: Performance-Critical Code

Before modifying ANY performance-critical code, you MUST read:

**[SOVEREIGN_ATTESTATION.md](SOVEREIGN_ATTESTATION.md)**

This document contains:
- Validated benchmarks (244 FPS @ 4K achieved)
- Locked optimizations (rSVD threshold, Morton encoding, etc.)
- Optimization-first mandates
- Validation commands to run before/after changes

**Files under protection:**
- `tensornet/sovereign/morton.py` — O(1) Morton encoding
- `tensornet/adaptive/*.py` — rSVD threshold = 100
- `tensornet/core/mps.py` — Tensor network core

Changes to these files require:
1. Reading the attestation document
2. Running validation benchmarks BEFORE changes
3. Running validation benchmarks AFTER changes
4. Proving measurable improvement (not "cleaner" code)

**The Sovereign thesis: You don't need more compute — you need smarter representations.**

## Development Tips

### Running Specific Tests
```bash
# Run unit tests only
pytest tests/unit/ -v

# Run integration tests
pytest tests/integration/ -v

# Run tests matching a pattern
pytest -k "dmrg" -v
```

### Profiling Performance
```bash
python scripts/profile_performance.py --dmrg --tebd
```

### Building Documentation
```bash
python tensornet/docs/api_reference.py
# Output in docs/api/
```

## Development Environment Setup

### Prerequisites
- Python 3.11 or 3.12
- Git
- (Optional) Rust toolchain for TCI acceleration

### Initial Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/tigantic/hypertensor.git
   cd hypertensor
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   
   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   
   # Windows (cmd)
   .venv\Scripts\activate.bat
   
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install the package in editable mode with dev dependencies**
   ```bash
   pip install -e ".[dev]"
   pip install -r requirements-dev.txt
   ```

4. **Verify installation**
   ```bash
   pytest tests/ -v --collect-only  # Should list all tests
   python -c "import tensornet; print(tensornet.__version__)"
   ```

### Development Tools

The project uses the following development tools (pinned in `requirements-dev.txt`):

| Tool | Purpose | Command |
|------|---------|---------|
| pytest | Testing | `pytest tests/ -v` |
| ruff | Linting/formatting | `ruff check . && ruff format .` |
| mypy | Type checking | `mypy tensornet/` |
| bandit | Security scanning | `bandit -r tensornet/` |
| detect-secrets | Secret detection | `detect-secrets scan` |
| pre-commit | Git hooks | `pre-commit install` |

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=tensornet --cov-report=html

# Run only unit tests (fast)
pytest tests/ -v -m unit

# Run physics validation tests
pytest tests/ -v -m physics

# Run tests requiring GPU
pytest tests/ -v -m gpu

# Run tests for specific module
pytest tests/test_dmrg.py -v

# Run with specific random seed (for reproducibility debugging)
HYPERTENSOR_TEST_SEED=12345 pytest tests/ -v
```

### Test Reproducibility

All tests are seeded for reproducibility via `tests/conftest.py`:
- Default seed: `42`
- Override with `HYPERTENSOR_TEST_SEED` environment variable
- Seeds: `random`, `numpy.random`, `torch.manual_seed`

If a test fails sporadically, try running with different seeds to reproduce.

### Pre-commit Hooks

Install pre-commit hooks to automatically run checks before commits:

```bash
pip install pre-commit
pre-commit install
```

### Makefile Targets

Common tasks are available via `make`:

```bash
make test          # Run tests
make lint          # Run linting
make format        # Format code
make security      # Run security scan
make dev-deps      # Install dev dependencies
make clean         # Clean build artifacts
```

## Dependency Lockfiles

We maintain lockfiles for reproducible builds:

| Package | Lockfile | Update Command |
|---------|----------|----------------|
| tensornet (root) | `requirements-lock.txt` | `pip freeze > requirements-lock.txt` |
| qtt-sdk | `sdk/qtt-sdk/requirements-lock.txt` | `cd sdk/qtt-sdk && pip freeze > requirements-lock.txt` |

### Updating Lockfiles

1. **When to update**: After changing dependencies in `pyproject.toml`
2. **Process**:
   ```bash
   # Ensure you're in a clean virtual environment
   pip install -e ".[dev]"
   pip freeze > requirements-lock.txt
   ```
3. **CI Check**: CI verifies lockfile consistency with `pip install -r requirements-lock.txt`

### Using Lockfiles

For reproducible builds (CI, deployment):
```bash
pip install -r requirements-lock.txt
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Logging Conventions

Use the centralized logging module instead of `print()` statements:

```python
from tensornet.logging_config import get_logger

logger = get_logger(__name__)

# Use appropriate log levels
logger.debug("Detailed debugging information")
logger.info("General informational messages")
logger.warning("Warning: something unexpected happened")
logger.error("Error: operation failed")

# Special methods for domain-specific logging
logger.computation("Starting DMRG calculation")  # [COMPUTE] prefix
logger.physics("Energy converged")               # [PHYSICS] prefix
logger.convergence(iteration=10, value=1e-8)     # Convergence tracking
```

### Log Level Guidelines

| Level | Use For |
|-------|---------|
| DEBUG | Detailed diagnostic info, variable values |
| INFO | Progress updates, successful operations |
| WARNING | Unexpected but recoverable situations |
| ERROR | Failed operations, exceptions |
| CRITICAL | System-level failures |

### When to Use `print()`

`print()` is acceptable in:
- Demo scripts (`demos/`)
- CLI tools with user-facing output
- Proof scripts for result display

Avoid `print()` in:
- Library code (`tensornet/`)
- Test files (use pytest's capfd instead)
- Server code (use structured logging)

## Need Help?

- Open a GitHub Discussion for questions
- Check existing issues before creating new ones
- Read the [README](README.md) for project overview

## Code of Conduct

Be respectful and constructive. We're building something cool together!
