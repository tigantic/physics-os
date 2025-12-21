"""
Quantum Module for HyperTensor
==============================

Quantum-classical hybrid algorithms, error mitigation, and quantum-inspired
optimization methods for tensor network computations.

Submodules:
    - hybrid: VQE, QAOA, tensor network Born machines
    - error_mitigation: ZNE, PEC, CDR, quantum error correction codes
"""

from .hybrid import (
    # Gate primitives
    GateType,
    QuantumGate,
    QuantumCircuit,
    GateMatrices,
    
    # TN Simulator
    TNQuantumSimulator,
    
    # VQE
    AnsatzType,
    VQEConfig,
    VQE,
    
    # QAOA
    QAOAConfig,
    QAOA,
    
    # Born Machine
    TensorNetworkBornMachine,
    
    # Quantum-inspired
    QuantumInspiredOptimizer,
    
    # Utilities
    create_ising_hamiltonian,
    create_maxcut_hamiltonian,
)

from .error_mitigation import (
    # Noise models
    NoiseType,
    NoiseChannel,
    NoiseModel,
    KrausChannel,
    
    # ZNE
    ExtrapolationMethod,
    ZNEConfig,
    ZeroNoiseExtrapolator,
    
    # PEC
    PECConfig,
    ProbabilisticErrorCancellation,
    
    # CDR
    CDRConfig,
    CliffordDataRegression,
    
    # QEC codes
    QECCode,
    BitFlipCode,
    PhaseFlipCode,
    ShorCode,
    
    # Noise-aware optimization
    NoiseAwareVQEConfig,
    NoiseAwareOptimizer,
    
    # Utilities
    apply_error_mitigation,
    create_device_noise_model,
)

__all__ = [
    # hybrid.py
    'GateType',
    'QuantumGate',
    'QuantumCircuit',
    'GateMatrices',
    'TNQuantumSimulator',
    'AnsatzType',
    'VQEConfig',
    'VQE',
    'QAOAConfig',
    'QAOA',
    'TensorNetworkBornMachine',
    'QuantumInspiredOptimizer',
    'create_ising_hamiltonian',
    'create_maxcut_hamiltonian',
    
    # error_mitigation.py
    'NoiseType',
    'NoiseChannel',
    'NoiseModel',
    'KrausChannel',
    'ExtrapolationMethod',
    'ZNEConfig',
    'ZeroNoiseExtrapolator',
    'PECConfig',
    'ProbabilisticErrorCancellation',
    'CDRConfig',
    'CliffordDataRegression',
    'QECCode',
    'BitFlipCode',
    'PhaseFlipCode',
    'ShorCode',
    'NoiseAwareVQEConfig',
    'NoiseAwareOptimizer',
    'apply_error_mitigation',
    'create_device_noise_model',
]
