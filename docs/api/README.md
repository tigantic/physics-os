# API Reference

This directory contains auto-generated API documentation for the ontic library.

## Generation

API documentation is generated using Sphinx:

```bash
cd docs
sphinx-apidoc -o api ../ontic -f
make html
```

## Structure

| File | Content |
|------|---------|
| `ontic.core.rst` | Core operations (MPS, MPO, decompositions) |
| `ontic.algorithms.rst` | DMRG, TEBD, TDVP, Lanczos |
| `ontic.cfd.rst` | CFD solvers (Euler, Navier-Stokes) |
| `ontic.quantum.rst` | Quantum-classical hybrid |
| `ontic.certification.rst` | DO-178C compliance |

## Constitutional Compliance

Per Article VI, Section 6.1, API reference is auto-generated from docstrings.
