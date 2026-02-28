# Module `ontic.adaptive`

Adaptive bond dimension management for tensor network simulations.

This module provides intelligent adaptive truncation strategies that dynamically
adjust bond dimensions during time evolution to balance accuracy and computational
cost. Key features include:

- Real-time entanglement entropy monitoring
- Area law validation and scaling analysis
- Multiple compression strategies (SVD, randomized, variational)
- Automatic bond dimension adaptation based on truncation error targets
