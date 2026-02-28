# Module `ontic.ml_surrogates`

Machine Learning Surrogate Models for The Physics OS.

This module provides neural network-based surrogate models for
accelerating CFD simulations. These models learn from high-fidelity
data to provide fast predictions with uncertainty quantification.

Components:
    - CFDSurrogate: Base class for CFD surrogate models
    - PhysicsInformedNet: Physics-informed neural networks (PINNs)
    - DeepONet: Deep Operator Networks for PDE solutions
    - FourierNeuralOperator: Spectral methods for operator learning
    - UncertaintyQuantifier: Bayesian/ensemble uncertainty estimation

**Contents:**

- [Submodules](#submodules)

## Submodules

- [`ml_surrogates.surrogate_base`](#ml_surrogates-surrogate_base)
