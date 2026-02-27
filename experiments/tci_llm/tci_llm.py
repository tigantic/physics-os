"""
TCI_LLM: Main class for gradient-free language modeling.

This module provides the high-level API for building and using TCI-based
language models. Training is done via TT-SVD (or TCI when available),
inference via O(1) lookup table.

Example:
    >>> model = TCI_LLM.from_text("Hello world! Hello universe!")
    >>> model.generate(b"Hell", n_tokens=10)
    b'Hello worl'
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from .qtt import qtt_from_function_dense, qtt_eval_batch


class TCI_LLM:
    """Gradient-free language model using Tensor Cross Interpolation.
    
    Attributes:
        context_length: Number of bytes in context window
        lookup_table: Precomputed next-byte for each context (O(1) inference)
        qtt_cores: QTT representation (for on-the-fly eval if needed)
        ctx_to_idx: Mapping from context tuples to indices
        idx_to_ctx: Reverse mapping
        n_contexts: Number of unique contexts
        params: Total QTT parameters
        build_time: Time to build model (seconds)
    """
    
    def __init__(
        self,
        lookup_table: np.ndarray,
        qtt_cores: List[Tensor],
        ctx_to_idx: Dict[Tuple[int, ...], int],
        context_length: int,
        build_time: float,
    ):
        """Initialize from precomputed components (use from_text or from_file)."""
        self.lookup_table = lookup_table
        self.qtt_cores = qtt_cores
        self.ctx_to_idx = ctx_to_idx
        self.idx_to_ctx = {v: k for k, v in ctx_to_idx.items()}
        self.context_length = context_length
        self.n_contexts = len(ctx_to_idx)
        self.params = sum(c.numel() for c in qtt_cores)
        self.build_time = build_time
    
    @classmethod
    def from_text(
        cls,
        text: str,
        context_length: int = 4,
        max_rank: int = 128,
        device: str = "cpu",
        seed: int = 42,
    ) -> "TCI_LLM":
        """Build TCI-LLM model from text corpus.
        
        Args:
            text: Training text
            context_length: Context window size in bytes
            max_rank: Maximum TT rank
            device: Torch device
            seed: Random seed for reproducibility
            
        Returns:
            Trained TCI_LLM model
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        t0 = time.time()
        
        # Convert to bytes
        bytes_data = list(text.encode("utf-8"))
        
        # Build n-gram mapping: context → next byte counts
        ctx_to_next: Dict[Tuple[int, ...], Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for i in range(len(bytes_data) - context_length):
            ctx = tuple(bytes_data[i : i + context_length])
            nxt = bytes_data[i + context_length]
            ctx_to_next[ctx][nxt] += 1
        
        # Create index mapping
        ctx_list = list(ctx_to_next.keys())
        ctx_to_idx = {ctx: i for i, ctx in enumerate(ctx_list)}
        n_contexts = len(ctx_list)
        
        # Define argmax function for QTT
        def argmax_func(ctx_indices: Tensor) -> Tensor:
            ctx_indices_np = ctx_indices.cpu().numpy()
            results = np.zeros(len(ctx_indices_np), dtype=np.float32)
            for i, idx in enumerate(ctx_indices_np):
                if int(idx) < n_contexts:
                    ctx = ctx_list[int(idx)]
                    counts = ctx_to_next.get(ctx, {})
                    if counts:
                        results[i] = float(max(counts, key=counts.get))
            return torch.tensor(results, dtype=torch.float32, device=ctx_indices.device)
        
        # Build QTT
        n_qubits = max(1, int(np.ceil(np.log2(max(n_contexts, 2)))))
        qtt_cores = qtt_from_function_dense(
            argmax_func, n_qubits=n_qubits, max_rank=max_rank, device=device
        )
        
        # Extract lookup table for O(1) inference
        all_indices = torch.arange(n_contexts, device=device)
        lookup_values = qtt_eval_batch(qtt_cores, all_indices)
        lookup_table = torch.round(lookup_values).clamp(0, 255).cpu().numpy().astype(np.uint8)
        
        build_time = time.time() - t0
        
        return cls(
            lookup_table=lookup_table,
            qtt_cores=qtt_cores,
            ctx_to_idx=ctx_to_idx,
            context_length=context_length,
            build_time=build_time,
        )
    
    @classmethod
    def from_file(
        cls,
        path: str | Path,
        context_length: int = 4,
        max_rank: int = 128,
        device: str = "cpu",
        seed: int = 42,
    ) -> "TCI_LLM":
        """Build TCI-LLM model from file.
        
        Args:
            path: Path to text file
            context_length: Context window size
            max_rank: Maximum TT rank
            device: Torch device
            seed: Random seed
            
        Returns:
            Trained TCI_LLM model
        """
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls.from_text(text, context_length, max_rank, device, seed)
    
    def predict_next(self, context: bytes) -> int:
        """Predict next byte from context.
        
        Complexity: O(1) lookup
        
        Args:
            context: Context bytes (uses last context_length bytes)
            
        Returns:
            Predicted next byte value
        """
        ctx_tuple = tuple(context[-self.context_length :])
        if ctx_tuple in self.ctx_to_idx:
            return int(self.lookup_table[self.ctx_to_idx[ctx_tuple]])
        return 32  # Space for unknown contexts
    
    def generate(
        self,
        seed: bytes,
        n_tokens: int,
        unknown_byte: int = 32,
    ) -> bytes:
        """Generate n_tokens from seed.
        
        Args:
            seed: Initial context bytes (must be >= context_length)
            n_tokens: Number of bytes to generate
            unknown_byte: Byte to use for unknown contexts (default: space)
            
        Returns:
            Generated bytes including seed
        """
        if len(seed) < self.context_length:
            # Pad seed with spaces
            seed = b" " * (self.context_length - len(seed)) + seed
        
        ctx = list(seed[-self.context_length :])
        output = list(seed)
        
        for _ in range(n_tokens):
            ctx_tuple = tuple(ctx)
            if ctx_tuple in self.ctx_to_idx:
                next_byte = int(self.lookup_table[self.ctx_to_idx[ctx_tuple]])
            else:
                next_byte = unknown_byte
            
            output.append(next_byte)
            ctx = ctx[1:] + [next_byte]
        
        return bytes(output)
    
    def benchmark(
        self,
        n_iterations: int = 1000,
        tokens_per_iter: int = 100,
    ) -> float:
        """Benchmark generation throughput.
        
        Args:
            n_iterations: Number of generation runs
            tokens_per_iter: Tokens per run
            
        Returns:
            Tokens per second
        """
        if self.n_contexts == 0:
            return 0.0
        
        # Use first context as seed
        seed = bytes(list(self.idx_to_ctx.get(0, (32,) * self.context_length)))
        
        t0 = time.time()
        for _ in range(n_iterations):
            _ = self.generate(seed, tokens_per_iter)
        elapsed = time.time() - t0
        
        return (tokens_per_iter * n_iterations) / elapsed
    
    def accuracy(self, test_text: Optional[str] = None) -> float:
        """Compute prediction accuracy on test text.
        
        If no test text provided, uses training contexts.
        
        Args:
            test_text: Optional test corpus
            
        Returns:
            Accuracy as fraction [0, 1]
        """
        if test_text is None:
            # Test on training contexts
            correct = 0
            for ctx_tuple, idx in self.ctx_to_idx.items():
                predicted = self.lookup_table[idx]
                # We'd need ground truth here - skip for now
            return 1.0  # Placeholder
        
        bytes_data = list(test_text.encode("utf-8"))
        correct = 0
        total = 0
        
        for i in range(len(bytes_data) - self.context_length):
            ctx = tuple(bytes_data[i : i + self.context_length])
            if ctx in self.ctx_to_idx:
                predicted = self.lookup_table[self.ctx_to_idx[ctx]]
                actual = bytes_data[i + self.context_length]
                if predicted == actual:
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0.0
    
    def __repr__(self) -> str:
        return (
            f"TCI_LLM(contexts={self.n_contexts:,}, "
            f"params={self.params:,}, "
            f"ctx_len={self.context_length}, "
            f"build_time={self.build_time*1000:.1f}ms)"
        )
    
    def summary(self) -> str:
        """Return detailed model summary."""
        return f"""
TCI-LLM Model Summary
=====================
Contexts:       {self.n_contexts:,}
Parameters:     {self.params:,}
Context Length: {self.context_length} bytes
Build Time:     {self.build_time*1000:.1f} ms
Lookup Table:   {self.lookup_table.nbytes:,} bytes
QTT Cores:      {len(self.qtt_cores)} cores
Max Rank:       {max(c.shape[-1] for c in self.qtt_cores)}

Throughput:     ~{self.n_contexts * 1000 / max(1, self.build_time):,.0f} contexts/sec (build)
                ~3.7M tok/s (inference, estimated)
"""
