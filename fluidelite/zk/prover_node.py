#!/usr/bin/env python3
"""
FluidElite Prover Node Simulation
=================================

Simulates the complete workflow of a FluidElite prover on a network like Gevulot.

This demonstrates:
1. Receiving a proof request from the network
2. Running FluidElite inference (the actual compute)
3. Generating ZK proof of correct execution
4. Submitting proof for verification
5. Calculating profit margin

Usage:
    python3 -m fluidelite.zk.prover_node
"""

import time
import hashlib
import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import sys

# Import FluidElite components
try:
    import torch
    from fluidelite.llm.fluid_elite_zk import FluidEliteZK
    from fluidelite.core.mps import MPS
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: torch not available, running in simulation mode")


@dataclass
class ProofRequest:
    """Incoming proof request from the network."""
    request_id: str
    tokens: List[int]
    model_commitment: str  # Hash of model weights
    max_response_time_ms: int
    reward_usd: float


@dataclass
class ProofResponse:
    """Proof submitted to the network."""
    request_id: str
    output_logits: List[float]
    predicted_token: int
    proof_commitment: str
    witness_commitment: str
    transcript_hash: str
    num_constraints: int
    prover_time_ms: float


@dataclass
class ProverStats:
    """Statistics for prover session."""
    jobs_completed: int = 0
    total_tokens_proved: int = 0
    total_constraints: int = 0
    total_prover_time_ms: float = 0.0
    total_reward_usd: float = 0.0
    total_electricity_cost_usd: float = 0.0
    
    @property
    def profit_usd(self) -> float:
        return self.total_reward_usd - self.total_electricity_cost_usd
    
    @property
    def profit_margin(self) -> float:
        if self.total_reward_usd == 0:
            return 0
        return self.profit_usd / self.total_reward_usd


class FluidEliteProver:
    """
    Prover node for FluidElite inference on decentralized networks.
    
    This class simulates:
    - Receiving jobs from network mempool
    - Running verified inference
    - Generating ZK proofs
    - Submitting proofs and collecting rewards
    """
    
    # Economic constants
    GPU_POWER_WATTS = 300  # RTX 4090
    ELECTRICITY_RATE_USD_KWH = 0.10
    
    def __init__(self, num_sites: int = 16, chi_max: int = 64, vocab_size: int = 256):
        self.config = {
            'num_sites': num_sites,
            'chi_max': chi_max,
            'vocab_size': vocab_size
        }
        
        if HAS_TORCH:
            self.model = FluidEliteZK(
                num_sites=num_sites,
                chi_max=chi_max,
                vocab_size=vocab_size
            )
            # Compute model commitment (hash of weights)
            param_bytes = b''.join(
                p.data.cpu().numpy().tobytes() 
                for p in self.model.parameters()
            )
            self.model_commitment = hashlib.sha256(param_bytes).hexdigest()[:16]
        else:
            self.model = None
            self.model_commitment = "simulation_mode"
        
        self.stats = ProverStats()
        
        print(f"FluidElite Prover Node Initialized")
        print(f"  Config: L={num_sites}, χ={chi_max}, vocab={vocab_size}")
        print(f"  Model commitment: {self.model_commitment}")
    
    def process_request(self, request: ProofRequest) -> ProofResponse:
        """
        Process a proof request: run inference + generate proof.
        """
        start_time = time.perf_counter()
        
        # Validate model commitment
        if request.model_commitment != self.model_commitment:
            raise ValueError(f"Model mismatch: expected {self.model_commitment}, got {request.model_commitment}")
        
        # Run inference
        if HAS_TORCH and self.model is not None:
            self.model.reset_step_count()
            with torch.no_grad():
                logits = self.model.forward(request.tokens)
                output_logits = logits.cpu().numpy().tolist()
                predicted_token = int(logits.argmax().item())
        else:
            # Simulation mode
            np.random.seed(sum(request.tokens) % 1000)
            output_logits = np.random.randn(self.config['vocab_size']).tolist()
            predicted_token = int(np.argmax(output_logits))
        
        inference_time = (time.perf_counter() - start_time) * 1000
        
        # Generate ZK proof (simulated)
        proof_start = time.perf_counter()
        
        # Compute commitments
        h = hashlib.sha256()
        h.update(json.dumps(request.tokens).encode())
        h.update(json.dumps(output_logits).encode())
        proof_commitment = h.hexdigest()[:32]
        
        h.update(b"witness")
        witness_commitment = h.hexdigest()[:32]
        
        h.update(b"transcript")
        transcript_hash = h.hexdigest()[:32]
        
        # Constraint count
        if self.model is not None:
            num_constraints = self.model.total_constraint_count(len(request.tokens))
            estimated_proof_time = self.model.estimate_proof_time_ms(len(request.tokens))
        else:
            num_constraints = 131104 * len(request.tokens) + 16384
            estimated_proof_time = num_constraints / 1000 * 0.05
        
        # Simulate actual proof generation time
        # In real system, this would be the actual prover
        proof_gen_time = (time.perf_counter() - proof_start) * 1000
        
        total_time = inference_time + estimated_proof_time
        
        # Update stats
        self.stats.jobs_completed += 1
        self.stats.total_tokens_proved += len(request.tokens)
        self.stats.total_constraints += num_constraints
        self.stats.total_prover_time_ms += total_time
        self.stats.total_reward_usd += request.reward_usd
        
        # Calculate electricity cost
        gpu_hours = total_time / 1000 / 3600
        kwh = gpu_hours * (self.GPU_POWER_WATTS / 1000)
        electricity_cost = kwh * self.ELECTRICITY_RATE_USD_KWH
        self.stats.total_electricity_cost_usd += electricity_cost
        
        return ProofResponse(
            request_id=request.request_id,
            output_logits=output_logits,
            predicted_token=predicted_token,
            proof_commitment=proof_commitment,
            witness_commitment=witness_commitment,
            transcript_hash=transcript_hash,
            num_constraints=num_constraints,
            prover_time_ms=total_time
        )
    
    def print_stats(self):
        """Print prover statistics."""
        s = self.stats
        print("\n" + "=" * 60)
        print("PROVER SESSION STATISTICS")
        print("=" * 60)
        print(f"  Jobs completed:      {s.jobs_completed}")
        print(f"  Tokens proved:       {s.total_tokens_proved:,}")
        print(f"  Total constraints:   {s.total_constraints:,}")
        print(f"  Total prover time:   {s.total_prover_time_ms:.1f}ms")
        print(f"  ")
        print(f"  Total rewards:       ${s.total_reward_usd:.6f}")
        print(f"  Electricity cost:    ${s.total_electricity_cost_usd:.6f}")
        print(f"  Net profit:          ${s.profit_usd:.6f}")
        print(f"  Profit margin:       {s.profit_margin * 100:.1f}%")
        
        if s.total_tokens_proved > 0:
            print(f"  ")
            print(f"  Per-token metrics:")
            print(f"    Avg constraints:   {s.total_constraints / s.total_tokens_proved:,.0f}")
            print(f"    Avg prover time:   {s.total_prover_time_ms / s.total_tokens_proved:.2f}ms")
            print(f"    Revenue per token: ${s.total_reward_usd / s.total_tokens_proved:.8f}")
        
        print("=" * 60)


def simulate_prover_session():
    """Simulate a prover session with multiple jobs."""
    print("=" * 60)
    print("FLUIDELITE PROVER NODE SIMULATION")
    print("=" * 60)
    print()
    
    # Initialize prover
    prover = FluidEliteProver(num_sites=16, chi_max=64, vocab_size=256)
    
    # Simulate job queue
    jobs = [
        ProofRequest(
            request_id=f"job_{i}",
            tokens=list(range(n_tokens)),
            model_commitment=prover.model_commitment,
            max_response_time_ms=10000,
            reward_usd=0.001 * (n_tokens / 100)  # $0.001 per 100 tokens
        )
        for i, n_tokens in enumerate([10, 50, 100, 200, 500])
    ]
    
    print(f"\nProcessing {len(jobs)} jobs from network queue...\n")
    print("-" * 60)
    
    for job in jobs:
        print(f"Job {job.request_id}: {len(job.tokens)} tokens, reward=${job.reward_usd:.4f}")
        
        try:
            response = prover.process_request(job)
            print(f"  ✅ Proof generated: {response.num_constraints:,} constraints")
            print(f"     Prover time: {response.prover_time_ms:.1f}ms")
            print(f"     Predicted token: {response.predicted_token}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        print()
    
    prover.print_stats()
    
    # Project to 24h operation
    print("\n" + "=" * 60)
    print("24-HOUR PROJECTION")
    print("=" * 60)
    
    s = prover.stats
    if s.total_prover_time_ms > 0:
        jobs_per_hour = 3600 * 1000 / (s.total_prover_time_ms / s.jobs_completed)
        daily_jobs = jobs_per_hour * 24
        daily_revenue = (s.total_reward_usd / s.jobs_completed) * daily_jobs
        daily_cost = (s.total_electricity_cost_usd / s.jobs_completed) * daily_jobs
        daily_profit = daily_revenue - daily_cost
        
        print(f"  Estimated jobs/hour:    {jobs_per_hour:,.0f}")
        print(f"  Estimated jobs/day:     {daily_jobs:,.0f}")
        print(f"  Daily revenue:          ${daily_revenue:.2f}")
        print(f"  Daily electricity:      ${daily_cost:.4f}")
        print(f"  Daily profit:           ${daily_profit:.2f}")
        print(f"  Monthly profit:         ${daily_profit * 30:.2f}")
    
    print("=" * 60)


def compare_transformer_costs():
    """Compare FluidElite vs Transformer proving costs."""
    print("\n" + "=" * 60)
    print("FLUIDELITE vs TRANSFORMER PROVER ECONOMICS")
    print("=" * 60)
    
    # FluidElite costs (from our analysis)
    fe_constraints_per_token = 131104  # L=16, chi=64
    fe_proof_time_per_token_ms = fe_constraints_per_token / 1000 * 0.05
    
    # Transformer costs (from literature)
    # GPT-2 small (117M params): ~50M constraints per token for attention alone
    tf_constraints_per_token = 50_000_000
    tf_proof_time_per_token_ms = tf_constraints_per_token / 1000 * 0.05
    
    print("\nPer-token costs:")
    print(f"  FluidElite:   {fe_constraints_per_token:>12,} constraints → {fe_proof_time_per_token_ms:.1f}ms")
    print(f"  Transformer:  {tf_constraints_per_token:>12,} constraints → {tf_proof_time_per_token_ms:.1f}ms")
    print(f"  Ratio:        {tf_constraints_per_token / fe_constraints_per_token:.0f}x more expensive for Transformer")
    
    print("\nFor 1000 tokens:")
    fe_1000 = fe_constraints_per_token * 1000
    tf_1000 = tf_constraints_per_token * 1000
    print(f"  FluidElite:   {fe_1000:>15,} constraints → {fe_1000 / 1000 * 0.05 / 1000:.1f}s")
    print(f"  Transformer:  {tf_1000:>15,} constraints → {tf_1000 / 1000 * 0.05 / 1000:.1f}s")
    
    print("\nHardware requirements:")
    print(f"  FluidElite:   RTX 3070 (8GB) - ${800}")
    print(f"  Transformer:  A100 (80GB) - ${15,000}")
    
    print("\nBreak-even (at $0.001 per 1000 tokens):")
    fe_profit_per_1000 = 0.001 - (fe_1000 / 1000 * 0.05 / 1000 / 3600 * 0.3 * 0.10)
    tf_profit_per_1000 = 0.001 - (tf_1000 / 1000 * 0.05 / 1000 / 3600 * 0.4 * 0.10)
    print(f"  FluidElite:   ${fe_profit_per_1000:.6f} profit per 1000 tokens")
    print(f"  Transformer:  ${tf_profit_per_1000:.6f} profit per 1000 tokens (likely negative)")
    
    print("\n" + "=" * 60)
    print("CONCLUSION: FluidElite enables profitable micro-proving")
    print("            Transformers require economies of scale")
    print("=" * 60)


if __name__ == "__main__":
    simulate_prover_session()
    compare_transformer_costs()
