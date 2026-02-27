"""
FluidElite ZK Python Bindings
=============================

Python interface to the Rust ZK prover.

Usage:
    from fluidelite_zk import FluidEliteProver, MPS
    
    # Create prover with default weights
    prover = FluidEliteProver.new_with_identity_weights(num_sites=8, chi_max=16)
    
    # Create context
    context = MPS(num_sites=8, chi_max=16)
    
    # Generate proof
    proof = prover.prove(context, token_id=42)
    print(f"Proof size: {proof.size()} bytes")
"""

try:
    from .fluidelite_zk import (
        MPS,
        MPO,
        Proof,
        FluidEliteProver,
        CircuitConfig,
        version,
        prove_inference,
    )
except ImportError:
    # Fallback for development without compiled extension
    import warnings
    warnings.warn(
        "fluidelite_zk Rust extension not found. "
        "Build with: maturin develop --features python"
    )
    
    # Provide stub implementations for development
    class MPS:
        """Stub MPS for development"""
        def __init__(self, num_sites=8, chi_max=16, phys_dim=2):
            self.num_sites = num_sites
            self.chi_max = chi_max
            self.phys_dim = phys_dim
    
    class MPO:
        """Stub MPO for development"""
        @staticmethod
        def identity(num_sites, phys_dim=2):
            return MPO()
    
    class Proof:
        """Stub Proof for development"""
        pass
    
    class FluidEliteProver:
        """Stub Prover for development"""
        @staticmethod
        def new_with_identity_weights(num_sites=8, chi_max=16, vocab_size=256):
            return FluidEliteProver()
    
    class CircuitConfig:
        """Stub CircuitConfig for development"""
        pass
    
    def version():
        return "0.1.0-dev"
    
    def prove_inference(*args, **kwargs):
        raise NotImplementedError("Rust extension required")

__all__ = [
    "MPS",
    "MPO", 
    "Proof",
    "FluidEliteProver",
    "CircuitConfig",
    "version",
    "prove_inference",
]

__version__ = "0.1.0"
