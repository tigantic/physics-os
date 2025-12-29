# API Reference

This directory contains auto-generated API documentation for the tensornet library.

## Generation

API documentation is generated using Sphinx:

```bash
cd docs
sphinx-apidoc -o api ../tensornet -f
make html
```

## Structure

| File | Content |
|------|---------|
| `tensornet.core.rst` | Core operations (MPS, MPO, decompositions) |
| `tensornet.algorithms.rst` | DMRG, TEBD, TDVP, Lanczos |
| `tensornet.cfd.rst` | CFD solvers (Euler, Navier-Stokes) |
| `tensornet.quantum.rst` | Quantum-classical hybrid |
| `tensornet.certification.rst` | DO-178C compliance |

## Constitutional Compliance

Per Article VI, Section 6.1, API reference is auto-generated from docstrings.
