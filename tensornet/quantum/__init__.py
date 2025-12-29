"""
Quantum Module for HyperTensor
==============================

Quantum-classical hybrid algorithms, error mitigation, quantum-inspired
optimization methods, and QTT-native rendering architecture.

Submodules:
    - hybrid: VQE, QAOA, tensor network Born machines
    - error_mitigation: ZNE, PEC, CDR, quantum error correction codes
    - cpu_qtt_evaluator: i9-optimized sparse QTT evaluation
    - hybrid_qtt_renderer: CPU-GPU co-design for real-time synthesis
    - qtt_glsl_bridge: GPU shader integration tools

The i9-5070 Co-Design:
    CPU (i9-14900HX): Factorization Engine - sparse QTT evaluation
    GPU (RTX 5070): Synthesis Engine - bicubic interpolation
    Performance: 4K @ 183 FPS (128² sparse) or 78 FPS (256² sparse)
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
    # QTT-Native Rendering
    'CPUQTTEvaluator',
    'HybridQTTRenderer',
    'QTTShaderParams',
    'pack_qtt_for_shader',
]

# Lazy imports for QTT rendering (avoid heavy dependencies unless needed)
def __getattr__(name):
    if name == 'CPUQTTEvaluator':
        from .cpu_qtt_evaluator import CPUQTTEvaluator
        return CPUQTTEvaluator
    elif name == 'HybridQTTRenderer':
        from .hybrid_qtt_renderer import HybridQTTRenderer
        return HybridQTTRenderer
    elif name == 'QTTShaderParams':
        from .qtt_glsl_bridge import QTTShaderParams
        return QTTShaderParams
    elif name == 'pack_qtt_for_shader':
        from .qtt_glsl_bridge import pack_qtt_for_shader
        return pack_qtt_for_shader
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
