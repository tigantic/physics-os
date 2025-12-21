# Integration Tests

This directory contains integration tests that validate end-to-end functionality
across multiple modules of the HyperTensor framework.

## Test Categories

| Test | Description |
|------|-------------|
| `test_dmrg_physics.py` | DMRG + Hamiltonians → ground state validation |
| `test_cfd_pipeline.py` | CFD solver → MPS compression → reconstruction |
| `test_quantum_certification.py` | Quantum hybrid → certification workflow |

## Running Integration Tests

```bash
pytest tests/integration/ -v
```

## Constitutional Compliance

Per Article III, Section 3.1, integration tests live in `tests/integration/`.
