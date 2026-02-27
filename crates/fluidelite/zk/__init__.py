"""FluidElite ZK Module - Zero-Knowledge Proof Components for FluidElite Inference."""

from fluidelite.zk.circuit_analysis import (
    FluidEliteConfig,
    CircuitStats,
    analyze_fluidelite_step,
    analyze_fluidelite_inference,
    analyze_transformer_attention,
    compare_architectures,
    detailed_breakdown,
)

from fluidelite.zk.proof_simulation import (
    FluidEliteZKProver,
    FluidEliteZKVerifier,
    ZKProof,
    Commitment,
    ProofTranscript,
)

from fluidelite.zk.prover_node import (
    FluidEliteProver,
    ProofRequest,
    ProofResponse,
    ProverStats,
)

__all__ = [
    # Circuit analysis
    'FluidEliteConfig',
    'CircuitStats',
    'analyze_fluidelite_step',
    'analyze_fluidelite_inference',
    'analyze_transformer_attention',
    'compare_architectures',
    'detailed_breakdown',
    # Proof simulation
    'FluidEliteZKProver',
    'FluidEliteZKVerifier',
    'ZKProof',
    'Commitment',
    'ProofTranscript',
    # Prover node
    'FluidEliteProver',
    'ProofRequest',
    'ProofResponse',
    'ProverStats',
]
