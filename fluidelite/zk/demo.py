#!/usr/bin/env python3
"""
FluidElite ZK End-to-End Demo
=============================

Demonstrates complete ZK proof generation for FluidElite inference:
1. Initialize FluidElite model
2. Run inference on tokens
3. Generate ZK proof of correct execution
4. Verify the proof

This is a simulation that shows the complete workflow.
Real deployment would use Halo2/Plonk circuits.
"""

import sys
import time
import hashlib
import numpy as np
from typing import List, Tuple

# Try to import FluidElite
try:
    import torch
    from fluidelite.llm.fluid_elite import FluidElite
    from fluidelite.core.mps import MPS
    HAS_FLUIDELITE = True
except ImportError:
    HAS_FLUIDELITE = False
    print("Warning: FluidElite not available, using simulation mode")


class ZKFluidEliteDemo:
    """
    Demonstrates ZK proof generation for FluidElite inference.
    """
    
    BABY_BEAR = 2013265921  # ZK-friendly prime
    
    def __init__(self, L: int = 16, chi: int = 64, vocab_size: int = 256):
        self.L = L
        self.chi = chi
        self.vocab_size = vocab_size
        
        if HAS_FLUIDELITE:
            self.model = FluidElite(
                num_sites=L,
                rank=chi,
                mpo_rank=1,
                vocab_size=vocab_size,
                dtype=torch.float32
            )
            print(f"Initialized FluidElite: L={L}, χ={chi}, vocab={vocab_size}")
        else:
            self.model = None
            print(f"Simulating FluidElite: L={L}, χ={chi}, vocab={vocab_size}")
    
    def run_inference(self, tokens: List[int]) -> Tuple[np.ndarray, dict]:
        """
        Run inference and collect witness data.
        
        Returns:
            (output_logits, witness_dict)
        """
        print(f"\n[1] Running inference on {len(tokens)} tokens...")
        start = time.perf_counter()
        
        if HAS_FLUIDELITE and self.model is not None:
            with torch.no_grad():
                # Get output logits
                logits = self.model.forward(tokens)
                output = logits.cpu().numpy()
                
                # Collect intermediate states (simplified - just model params)
                param_bytes = b''.join(p.data.cpu().numpy().tobytes() for p in self.model.parameters())
                witness = {
                    'tokens': np.array(tokens),
                    'output': output,
                    'model_hash': hashlib.sha256(param_bytes).hexdigest()[:16]
                }
        else:
            # Simulation mode
            np.random.seed(sum(tokens) % 1000)
            output = np.random.randn(self.vocab_size).astype(np.float32)
            witness = {
                'tokens': np.array(tokens),
                'output': output,
                'model_hash': 'simulated_model'
            }
        
        elapsed = (time.perf_counter() - start) * 1000
        print(f"    Inference complete: {elapsed:.2f}ms")
        print(f"    Output shape: {output.shape}")
        print(f"    Top prediction: token {output.argmax()}")
        
        return output, witness
    
    def generate_commitment(self, data: np.ndarray, label: str) -> bytes:
        """Generate commitment to data (simulated Merkle root)."""
        h = hashlib.sha256()
        h.update(label.encode())
        h.update(data.tobytes())
        return h.digest()
    
    def fiat_shamir_challenge(self, transcript: bytes, label: str) -> int:
        """Generate Fiat-Shamir challenge."""
        h = hashlib.sha256()
        h.update(transcript)
        h.update(label.encode())
        return int.from_bytes(h.digest()[:8], 'little') % self.BABY_BEAR
    
    def generate_proof(self, tokens: List[int], output: np.ndarray, witness: dict) -> dict:
        """
        Generate ZK proof of correct inference.
        
        In real system: This would be a Plonk/Halo2 proof.
        Here: Simulated with Fiat-Shamir commitments.
        """
        print(f"\n[2] Generating ZK proof...")
        start = time.perf_counter()
        
        # Step 1: Commit to public inputs
        input_commit = self.generate_commitment(np.array(tokens), "input")
        output_commit = self.generate_commitment(output, "output")
        
        # Step 2: Commit to witness (model + intermediates)
        # In real system: Merkle tree over all witness values
        witness_data = np.concatenate([
            witness['tokens'].astype(np.float32),
            witness['output'],
            np.frombuffer(witness['model_hash'].encode(), dtype=np.uint8).view(np.float32)
        ])
        witness_commit = self.generate_commitment(witness_data, "witness")
        
        # Step 3: Build transcript
        transcript = input_commit + output_commit + witness_commit
        
        # Step 4: Generate challenges
        alpha = self.fiat_shamir_challenge(transcript, "alpha")
        beta = self.fiat_shamir_challenge(transcript + alpha.to_bytes(8, 'little'), "beta")
        gamma = self.fiat_shamir_challenge(transcript + beta.to_bytes(8, 'little'), "gamma")
        
        # Step 5: Compute evaluations (simulated polynomial openings)
        # In real system: Evaluate constraint polynomials at random point
        z = self.fiat_shamir_challenge(transcript + gamma.to_bytes(8, 'little'), "z")
        
        evaluations = []
        for i in range(5):  # 5 random evaluations
            idx = (z + i * alpha) % len(witness_data)
            evaluations.append(int(witness_data[int(idx)] * 1000) % self.BABY_BEAR)
        
        # Step 6: Compute constraint polynomial quotient
        # In real system: q(X) = (constraint(X)) / Z_H(X)
        quotient_eval = sum(evaluations) % self.BABY_BEAR
        
        elapsed = (time.perf_counter() - start) * 1000
        
        # Estimate constraint count
        constraints_per_token = self.L * self.chi * self.chi * 2  # Linear mode
        total_constraints = constraints_per_token * len(tokens) + self.chi * self.vocab_size
        
        proof = {
            'input_commitment': input_commit.hex(),
            'output_commitment': output_commit.hex(),
            'witness_commitment': witness_commit.hex(),
            'challenges': [alpha, beta, gamma, z],
            'evaluations': evaluations,
            'quotient_eval': quotient_eval,
            'num_constraints': total_constraints,
            'prover_time_ms': elapsed
        }
        
        print(f"    Proof generated: {elapsed:.2f}ms")
        print(f"    Constraints: {total_constraints:,}")
        print(f"    Proof size: {self._proof_size(proof)} bytes")
        
        return proof
    
    def _proof_size(self, proof: dict) -> int:
        """Calculate proof size in bytes."""
        # 3 commitments × 32 bytes
        commits = 3 * 32
        # 4 challenges × 8 bytes
        challenges = 4 * 8
        # 5 evaluations × 8 bytes
        evals = 5 * 8
        # 1 quotient eval
        quotient = 8
        # Fixed overhead
        overhead = 64
        return commits + challenges + evals + quotient + overhead
    
    def verify_proof(self, tokens: List[int], output: np.ndarray, proof: dict) -> bool:
        """
        Verify ZK proof of correct inference.
        
        Verifier only sees:
        - Public inputs (tokens)
        - Claimed output
        - Proof
        
        Does NOT see:
        - Model weights
        - Intermediate computations
        """
        print(f"\n[3] Verifying proof...")
        start = time.perf_counter()
        
        # Step 1: Recompute input/output commitments
        input_commit = self.generate_commitment(np.array(tokens), "input")
        output_commit = self.generate_commitment(output, "output")
        
        # Step 2: Check commitments match
        if input_commit.hex() != proof['input_commitment']:
            print("    ❌ Input commitment mismatch")
            return False
        
        if output_commit.hex() != proof['output_commitment']:
            print("    ❌ Output commitment mismatch")
            return False
        
        # Step 3: Regenerate challenges (Fiat-Shamir)
        witness_commit = bytes.fromhex(proof['witness_commitment'])
        transcript = input_commit + output_commit + witness_commit
        
        alpha = self.fiat_shamir_challenge(transcript, "alpha")
        beta = self.fiat_shamir_challenge(transcript + alpha.to_bytes(8, 'little'), "beta")
        gamma = self.fiat_shamir_challenge(transcript + beta.to_bytes(8, 'little'), "gamma")
        z = self.fiat_shamir_challenge(transcript + gamma.to_bytes(8, 'little'), "z")
        
        if [alpha, beta, gamma, z] != proof['challenges']:
            print("    ❌ Challenge mismatch (transcript tampering)")
            return False
        
        # Step 4: Verify evaluations are in valid range
        for eval_val in proof['evaluations']:
            if eval_val < 0 or eval_val >= self.BABY_BEAR:
                print(f"    ❌ Evaluation {eval_val} out of field range")
                return False
        
        # Step 5: Verify quotient polynomial (simulated)
        # In real system: Check pairing equations
        expected_quotient = sum(proof['evaluations']) % self.BABY_BEAR
        if expected_quotient != proof['quotient_eval']:
            print("    ❌ Quotient polynomial mismatch")
            return False
        
        elapsed = (time.perf_counter() - start) * 1000
        print(f"    ✅ Proof valid: {elapsed:.2f}ms")
        
        return True
    
    def run_demo(self, tokens: List[int]):
        """Run complete ZK inference demo."""
        print("=" * 60)
        print("FluidElite ZK Inference Demo")
        print("=" * 60)
        print(f"\nInput: {len(tokens)} tokens")
        print(f"Model: L={self.L}, χ={self.chi}, vocab={self.vocab_size}")
        
        # Step 1: Inference
        output, witness = self.run_inference(tokens)
        
        # Step 2: Prove
        proof = self.generate_proof(tokens, output, witness)
        
        # Step 3: Verify
        valid = self.verify_proof(tokens, output, proof)
        
        # Summary
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"  Input tokens:    {len(tokens)}")
        print(f"  Constraints:     {proof['num_constraints']:,}")
        print(f"  Prover time:     {proof['prover_time_ms']:.2f}ms (simulated)")
        print(f"  Proof size:      {self._proof_size(proof)} bytes")
        print(f"  Proof valid:     {'✅ YES' if valid else '❌ NO'}")
        
        # Performance projections
        print("\n  Real-world estimates (GPU):")
        real_prover_time = proof['num_constraints'] / 1000 * 0.5
        print(f"    Prover time:   {real_prover_time:.1f}ms")
        print(f"    Verifier time: 3-5ms (constant)")
        print(f"    Proof size:    500-1000 bytes (constant)")
        
        return valid


def main():
    """Run demo with various configurations."""
    print("\n" + "=" * 60)
    print("FluidElite ZK Demo - Multiple Configurations")
    print("=" * 60)
    
    # Test configurations
    configs = [
        (8, 32, 10),     # Small model, short sequence
        (16, 64, 50),    # Medium model, medium sequence
        (16, 64, 200),   # Medium model, long sequence
    ]
    
    for L, chi, n_tokens in configs:
        print(f"\n{'='*60}")
        print(f"Config: L={L}, χ={chi}, tokens={n_tokens}")
        print("=" * 60)
        
        demo = ZKFluidEliteDemo(L=L, chi=chi, vocab_size=256)
        tokens = list(range(n_tokens))
        demo.run_demo(tokens)
    
    print("\n" + "=" * 60)
    print("All demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
