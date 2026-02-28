"""
QTT-RKHS: Kernel Methods in Quantized Tensor Train Format

TENSOR GENESIS Protocol — Layer 24

Reproducing Kernel Hilbert Spaces (RKHS) provide a powerful framework
for kernel machines and Gaussian processes. The kernel matrix K with
K_ij = k(x_i, x_j) captures pairwise relationships.

QTT Insight: For grid-based data, many kernels have low-rank structure:
    - RBF/Gaussian: K_ij = exp(-||x_i - x_j||²/2σ²)
    - Polynomial: K_ij = (x_i · x_j + c)^d
    - Matérn: K_ij = Matérn_ν(||x_i - x_j||)

These kernels can be represented in QTT format with rank O(log N),
enabling O(r³ log N) kernel matrix operations vs O(N³).

Applications:
    - Gaussian processes at trillion-point scale
    - Kernel ridge regression
    - Maximum Mean Discrepancy (MMD)
    - Kernel PCA

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from ontic.genesis.rkhs.kernels import (
    Kernel,
    RBFKernel,
    MaternKernel,
    PolynomialKernel,
    PeriodicKernel,
    LinearKernel,
    CompositeKernel,
    SumKernel,
    ProductKernel,
)

from ontic.genesis.rkhs.kernel_matrix import (
    QTTKernelMatrix,
    kernel_matrix,
    kernel_vector,
)

from ontic.genesis.rkhs.gp import (
    GPPrior,
    GPPosterior,
    GPRegressor,
    SparseGP,
    gp_predict,
    gp_posterior_sample,
    gp_marginal_likelihood,
)

from ontic.genesis.rkhs.ridge import (
    kernel_ridge_regression,
    solve_krr,
    KernelRidgeRegressor,
)

from ontic.genesis.rkhs.mmd import (
    maximum_mean_discrepancy,
    mmd_squared,
    mmd_test,
    mmd_qtt_native,
    rbf_kernel_mpo,
    QTTKernelMPO,
)

__all__ = [
    # Kernels
    "Kernel",
    "RBFKernel",
    "MaternKernel",
    "PolynomialKernel",
    "PeriodicKernel",
    "LinearKernel",
    "CompositeKernel",
    "SumKernel",
    "ProductKernel",
    # Kernel matrices
    "QTTKernelMatrix",
    "kernel_matrix",
    "kernel_vector",
    # Gaussian processes
    "GPPrior",
    "GPPosterior",
    "GPRegressor",
    "SparseGP",
    "gp_predict",
    "gp_posterior_sample",
    "gp_marginal_likelihood",
    # Kernel ridge regression
    "kernel_ridge_regression",
    "solve_krr",
    "KernelRidgeRegressor",
    # MMD
    "maximum_mean_discrepancy",
    "mmd_squared",
    "mmd_test",
    # QTT-native MMD
    "mmd_qtt_native",
    "rbf_kernel_mpo",
    "QTTKernelMPO",
]
