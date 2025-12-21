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

## Need Help?

- Open a GitHub Discussion for questions
- Check existing issues before creating new ones
- Read the [README](README.md) for project overview

## Code of Conduct

Be respectful and constructive. We're building something cool together!
