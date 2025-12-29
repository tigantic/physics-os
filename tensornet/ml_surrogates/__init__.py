"""
Machine Learning Surrogate Models for HyperTensor.

This module provides neural network-based surrogate models for
accelerating CFD simulations. These models learn from high-fidelity
data to provide fast predictions with uncertainty quantification.

Components:
    - CFDSurrogate: Base class for CFD surrogate models
    - PhysicsInformedNet: Physics-informed neural networks (PINNs)
    - DeepONet: Deep Operator Networks for PDE solutions
    - FourierNeuralOperator: Spectral methods for operator learning
    - UncertaintyQuantifier: Bayesian/ensemble uncertainty estimation

Example:
    >>> from tensornet.ml_surrogates import PhysicsInformedNet, PINNConfig
    >>> config = PINNConfig(
    ...     input_dim=4,  # (x, y, z, t)
    ...     output_dim=5,  # (rho, u, v, w, p)
    ...     hidden_layers=[128, 128, 128],
    ...     physics_weight=1.0
    ... )
    >>> pinn = PhysicsInformedNet(config)
    >>> pinn.train(x_data, y_data, boundary_conditions)

Author: HyperTensor Team
"""

from .surrogate_base import (
    SurrogateConfig,
    CFDSurrogate,
    MLPSurrogate,
    ResNetSurrogate,
    SurrogateMetrics,
    evaluate_surrogate,
    create_surrogate,
)

from .physics_informed import (
    PINNConfig,
    PhysicsLoss,
    PhysicsInformedNet,
    NavierStokesPINN,
    EulerPINN,
    create_pinn_for_equation,
    compute_physics_residual,
)

from .deep_onet import (
    DeepONetConfig,
    BranchNet,
    TrunkNet,
    DeepONet,
    MultiInputDeepONet,
    create_deeponet,
    train_deeponet,
)

from .fourier_operator import (
    FNOConfig,
    SpectralConv2d,
    SpectralConv3d,
    FourierNeuralOperator,
    FNO2d,
    FNO3d,
    TFNO2d,
    create_fno,
)

from .uncertainty import (
    UncertaintyConfig,
    UncertaintyType,
    UncertaintyQuantifier,
    EnsembleUQ,
    MCDropoutUQ,
    BayesianUQ,
    compute_prediction_interval,
    calibrate_uncertainty,
)

from .training import (
    TrainingConfig,
    SurrogateTrainer,
    DataAugmentor,
    ActiveLearner,
    train_surrogate,
    cross_validate,
)

__all__ = [
    # Base
    "SurrogateConfig",
    "CFDSurrogate",
    "MLPSurrogate",
    "ResNetSurrogate",
    "SurrogateMetrics",
    "evaluate_surrogate",
    "create_surrogate",
    # Physics-Informed
    "PINNConfig",
    "PhysicsLoss",
    "PhysicsInformedNet",
    "NavierStokesPINN",
    "EulerPINN",
    "create_pinn_for_equation",
    "compute_physics_residual",
    # DeepONet
    "DeepONetConfig",
    "BranchNet",
    "TrunkNet",
    "DeepONet",
    "MultiInputDeepONet",
    "create_deeponet",
    "train_deeponet",
    # Fourier Neural Operator
    "FNOConfig",
    "SpectralConv2d",
    "SpectralConv3d",
    "FourierNeuralOperator",
    "FNO2d",
    "FNO3d",
    "TFNO2d",
    "create_fno",
    # Uncertainty
    "UncertaintyConfig",
    "UncertaintyType",
    "UncertaintyQuantifier",
    "EnsembleUQ",
    "MCDropoutUQ",
    "BayesianUQ",
    "compute_prediction_interval",
    "calibrate_uncertainty",
    # Training
    "TrainingConfig",
    "SurrogateTrainer",
    "DataAugmentor",
    "ActiveLearner",
    "train_surrogate",
    "cross_validate",
]
