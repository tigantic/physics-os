# Changelog

All notable changes to QTeneT will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-31

### Added
- **Holy Grail Demo**: 6D Vlasov-Maxwell simulation with 1 billion grid points
  - 21,229× compression (4.29 GB → 198 KB)
  - O(log N) complexity per timestep
- **TCI Module**: Black-box function to QTT compression
  - `from_function()` for arbitrary callable compression
  - `from_function_nd()` for multi-dimensional functions
  - Smart pivot initialization for better accuracy
- **Operators Module**: N-dimensional curse-breaking operators
  - `shift_nd()`: O(log N) shift in any dimension
  - `apply_shift()`: Apply shift MPO to QTT state
  - `laplacian_nd()`: N-dimensional Laplacian
  - `gradient_nd()`: N-dimensional gradient
- **Solvers Module**: Physics solvers in native QTT format
  - `Vlasov5D`: 5D phase-space Vlasov-Poisson
  - `Vlasov6D`: 6D phase-space Vlasov-Maxwell (THE HOLY GRAIL)
  - `EulerND`: N-dimensional compressible Euler
- **Benchmarks Module**: Curse-breaking performance validation
  - `curse_of_dimensionality()`: Scaling across dimensions
  - `dimension_scaling()`: Compression vs dimension
  - `rank_scaling()`: Accuracy vs rank tradeoff
- **Genesis Module**: Advanced QTT primitives
  - Optimal Transport (Layer 20)
  - Spectral Graph Wavelets (Layer 21)
  - Random Matrix Theory (Layer 22)
  - Tropical Geometry (Layer 23)
  - Kernel Methods / RKHS (Layer 24)
  - Persistent Homology (Layer 25)
  - Geometric Algebra (Layer 26)
- **Test Suite**: 66 unit tests covering all modules

### Fixed
- `apply_shift()` parameter order now matches upstream API: `(qtt_cores, mpo_cores)`
- `Vlasov5D.two_stream_ic()` now correctly calls upstream without unsupported kwargs
- MSB-first bit ordering for consistent QTT evaluation

### Known Limitations
- Single-sweep TCI may have reconstruction errors for peaked functions on large grids
- Energy conservation not tracked (would require expensive dense operations)

---

*ELITE Engineering — Tigantic Holdings LLC*
