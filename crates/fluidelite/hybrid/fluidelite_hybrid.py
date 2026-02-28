#!/usr/bin/env python3
"""
FluidEliteHybrid: Production ZK-LLM Model
==========================================

Combines the two winning approaches from FINDINGS.md:
1. LOOKUP TABLE (Fast Path): 100% accuracy on seen contexts, O(1) lookup
2. LEAST SQUARES (Fallback): 46% accuracy on unseen contexts, sparse features

Key Insight: 76% of tokens hit the Fast Path (Lookup), avoiding expensive math.
This reduces ZK proof generation time by ~75%.

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                  FluidEliteHybrid                        │
    │                                                          │
    │  Context (8 bytes) ──→ Hash ──→ ctx_idx                 │
    │                            │                             │
    │                            ▼                             │
    │               ┌─────── Lookup ───────┐                  │
    │               │  ctx_idx in table?   │                  │
    │               └──────────┬───────────┘                  │
    │                    Yes / │ \ No                         │
    │                        /   \                            │
    │               ┌──────▼     ▼──────┐                     │
    │               │ Fast Path │ Fallback│                   │
    │               │ table[idx]│ x @ W   │                   │
    │               │   O(1)    │ O(F×V)  │                   │
    │               └─────┬─────┴────┬────┘                   │
    │                     │          │                         │
    │                     ▼          ▼                         │
    │                  next_token (0-255)                      │
    └─────────────────────────────────────────────────────────┘

ZK Circuit Implications:
- Fast Path: Lookup Argument (100x cheaper than matmul)
- Fallback: Sparse matmul with rank-24 compressed W

Usage:
    model = FluidEliteHybrid.from_corpus(train_bytes)
    model.save("model.bin")
    
    # Inference
    next_byte = model.predict(context_bytes)

Author: TiganticLabz
Date: January 2026
"""

import numpy as np
import torch
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import svds
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import struct
import json
import time


@dataclass
class HybridConfig:
    """
    Production Configuration for FluidEliteHybrid V1.
    
    THE SWEET SPOT (from FINDINGS.md):
    - Total Params: ~16,000 (16.04k)
    - Chi (Rank): 24 (NOT 64 - that was the old dense head)
    - L (Sites): 12 (2^12 = 4096 context states)
    - Vocab: 256 (byte-level)
    
    WHAT WE KILLED:
    - MPO contraction (no mpo_rank needed)
    - Truncation (no SVDs, no chi explosion)
    - Backpropagation (no gradients)
    
    The ZK circuit is now purely Arithmetic:
    - Verify Hash (for Lookup)
    - Verify Multiply+Add (for Least Squares)
    """
    # Core architecture
    context_len: int = 12         # L - lookback window (12 bytes)
    vocab_size: int = 256         # Byte-level vocabulary
    rank: int = 24                # Chi - compression rank (THE SWEET SPOT!)
    
    # Sparse features for Least Squares fallback
    # Total: 22,528 dimensions (hashed to prevent explosion)
    feature_dim: int = 22528      
    n_unigram: int = 2048         # Position × byte
    n_bigram: int = 8192          # Hashed bigrams
    n_trigram: int = 8192         # Hashed trigrams  
    n_skipgram: int = 4096        # Hashed skipgrams
    
    # Training
    regularization: float = 0.1   # Ridge regression lambda
    
    # Computed properties
    @property
    def params(self) -> int:
        """Total parameters in the model."""
        # Compressed W: [feature_dim × rank] + [vocab_size × rank]
        return self.feature_dim * self.rank + self.vocab_size * self.rank
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.rank <= 32, f"Rank should be ≤32 for efficiency, got {self.rank}"
        assert self.vocab_size == 256, f"Vocab must be 256 (byte-level), got {self.vocab_size}"


class FluidEliteHybrid:
    """
    Production ZK-LLM combining Lookup + Least Squares.
    
    Fast Path (76% of tokens): O(1) lookup table
    Fallback (24% of tokens): Sparse features → compressed W
    """
    
    def __init__(self, config: HybridConfig = None):
        self.config = config or HybridConfig()
        
        # Lookup table: ctx_hash → next_byte (Fast Path)
        self.lookup_table: Dict[bytes, int] = {}
        
        # Sparse feature weights: [feature_dim, vocab_size] (Fallback)
        # Stored as rank-24 compressed: W = U @ V.T
        self.W_compressed: Optional[Tuple[np.ndarray, np.ndarray]] = None
        
        # Statistics
        self.n_seen_contexts = 0
        self.n_total_predictions = 0
        self.n_lookup_hits = 0
    
    @classmethod
    def from_corpus(cls, data: bytes, config: HybridConfig = None) -> 'FluidEliteHybrid':
        """
        Build hybrid model from raw bytes.
        
        Phase 1: Build lookup table (100% accuracy on seen)
        Phase 2: Build sparse features → least squares (46% on unseen)
        Phase 3: Compress W to rank-24 via SVD
        """
        config = config or HybridConfig()
        model = cls(config)
        
        print("=" * 60)
        print("FluidEliteHybrid: Building from corpus")
        print("=" * 60)
        print(f"Corpus size: {len(data):,} bytes")
        print(f"Context length: {config.context_len}")
        print(f"Target rank: {config.rank}")
        
        # ========================================
        # PHASE 1: Build Lookup Table (Fast Path)
        # ========================================
        print("\n📋 Phase 1: Building Lookup Table...")
        start = time.time()
        
        context_counts: Dict[bytes, Dict[int, int]] = {}
        
        for i in range(len(data) - config.context_len):
            ctx = data[i:i + config.context_len]
            next_byte = data[i + config.context_len]
            
            if ctx not in context_counts:
                context_counts[ctx] = {}
            context_counts[ctx][next_byte] = context_counts[ctx].get(next_byte, 0) + 1
        
        # Store argmax for each context
        for ctx, counts in context_counts.items():
            model.lookup_table[ctx] = max(counts, key=counts.get)
        
        model.n_seen_contexts = len(model.lookup_table)
        print(f"   Unique contexts: {model.n_seen_contexts:,}")
        print(f"   Lookup table size: {model.n_seen_contexts * (config.context_len + 1) / 1024:.1f} KB")
        print(f"   Time: {time.time() - start:.2f}s")
        
        # ========================================
        # PHASE 2-3: STREAMING COVARIANCE ON GPU
        # ========================================
        print("\n🔢 Phase 2-3: Streaming covariance accumulation (GPU)...")
        start = time.time()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Device: {device}")
        
        contexts = list(context_counts.keys())
        n_contexts = len(contexts)
        F = config.feature_dim
        V = config.vocab_size
        
        # Accumulate X^T X and X^T Y on GPU (never materialize full X)
        XtX = torch.zeros((F, F), dtype=torch.float32, device=device)
        XtY = torch.zeros((F, V), dtype=torch.float32, device=device)
        
        batch_size = 50000  # Process in batches
        n_batches = (n_contexts + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, n_contexts)
            batch_contexts = contexts[batch_start:batch_end]
            B = len(batch_contexts)
            
            # Build batch feature matrix on GPU
            X_batch = torch.zeros((B, F), dtype=torch.float32, device=device)
            Y_batch = torch.zeros((B, V), dtype=torch.float32, device=device)
            
            for i, ctx in enumerate(batch_contexts):
                features = model._extract_features(ctx)
                for f in features:
                    X_batch[i, f] = 1.0
                
                counts = context_counts[ctx]
                total = sum(counts.values())
                for byte_val, count in counts.items():
                    Y_batch[i, byte_val] = count / total
            
            # Accumulate covariance
            XtX += X_batch.T @ X_batch
            XtY += X_batch.T @ Y_batch
            
            del X_batch, Y_batch
            torch.cuda.empty_cache() if device.type == 'cuda' else None
            
            if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
                print(f"   Batch [{batch_idx+1}/{n_batches}] - {batch_end:,}/{n_contexts:,} contexts")
        
        # Add regularization
        XtX += config.regularization * torch.eye(F, dtype=torch.float32, device=device)
        
        print(f"   XtX: {XtX.shape}, XtY: {XtY.shape}")
        print(f"   Solving linear system on GPU...")
        
        # Solve: W = (XtX)^{-1} XtY
        W_dense = torch.linalg.solve(XtX, XtY)  # [F, V]
        
        del XtX, XtY
        torch.cuda.empty_cache() if device.type == 'cuda' else None
        
        print(f"   W shape: {W_dense.shape}")
        print(f"   Time: {time.time() - start:.2f}s")
        
        # ========================================
        # PHASE 4: Compress W to Rank-24 (GPU SVD)
        # ========================================
        print(f"\n📦 Phase 4: Compressing W to rank-{config.rank} via SVD (GPU)...")
        start = time.time()
        
        k = config.rank
        
        # Full SVD on GPU
        U, S, Vh = torch.linalg.svd(W_dense, full_matrices=False)
        
        # Keep only top-k
        U_k = U[:, :k]           # [F, k]
        S_k = S[:k]              # [k]
        Vh_k = Vh[:k, :]         # [k, V]
        
        # Store as W = (U * S) @ V.T
        model.W_compressed = (
            (U_k * S_k).cpu().numpy().astype(np.float32),  # [F, k]
            Vh_k.T.cpu().numpy().astype(np.float32)         # [V, k]
        )
        
        del W_dense, U, S, Vh, U_k, S_k, Vh_k
        torch.cuda.empty_cache() if device.type == 'cuda' else None
        
        # Stats
        compressed_size = (model.W_compressed[0].size + model.W_compressed[1].size) * 4
        dense_size = F * V * 4
        
        print(f"   Compressed shape: [{F} × {k}] + [{V} × {k}]")
        print(f"   Compressed size: {compressed_size / 1024:.1f} KB")
        print(f"   Compression ratio: {dense_size / compressed_size:.1f}×")
        
        # Verify reconstruction error
        W_reconstructed = model.W_compressed[0] @ model.W_compressed[1].T
        reconstruction_error = np.linalg.norm(W_dense - W_reconstructed) / np.linalg.norm(W_dense)
        print(f"   Reconstruction error: {reconstruction_error:.4f}")
        print(f"   Time: {time.time() - start:.2f}s")
        
        print("\n" + "=" * 60)
        print("✅ FluidEliteHybrid ready!")
        print(f"   Fast Path: {model.n_seen_contexts:,} contexts in lookup")
        print(f"   Fallback: Rank-{k} compressed W ({compressed_size/1024:.1f} KB)")
        print("=" * 60)
        
        return model
    
    def _extract_features(self, ctx: bytes) -> List[int]:
        """
        Extract sparse n-gram features from context.
        
        Returns list of active feature indices (sparse representation).
        """
        cfg = self.config
        features = []
        ctx_len = len(ctx)
        
        # Unigrams: position × byte (last 4 positions only for efficiency)
        for i in range(max(0, ctx_len - 4), ctx_len):
            pos = i - (ctx_len - 4)  # Relative position 0-3
            idx = (pos * 256 + ctx[i]) % cfg.n_unigram
            features.append(idx)
        
        # Bigrams
        base = cfg.n_unigram
        for i in range(ctx_len - 1):
            h = (i * 65537 + ctx[i] * 257 + ctx[i+1]) % cfg.n_bigram
            features.append(base + h)
        
        # Trigrams
        base += cfg.n_bigram
        for i in range(ctx_len - 2):
            h = (ctx[i] * 65537 + ctx[i+1] * 257 + ctx[i+2]) % cfg.n_trigram
            features.append(base + h)
        
        # Skipgrams
        base += cfg.n_trigram
        for i in range(ctx_len - 2):
            h = (ctx[i] * 257 + ctx[i+2]) % cfg.n_skipgram
            features.append(base + h)
        
        return features
    
    def predict(self, context: bytes) -> int:
        """
        Predict next byte given context.
        
        Fast Path: Lookup table (O(1))
        Fallback: Sparse features → compressed W matmul
        """
        self.n_total_predictions += 1
        
        # Ensure context is correct length
        if len(context) < self.config.context_len:
            context = b'\x00' * (self.config.context_len - len(context)) + context
        elif len(context) > self.config.context_len:
            context = context[-self.config.context_len:]
        
        # FAST PATH: Lookup table
        if context in self.lookup_table:
            self.n_lookup_hits += 1
            return self.lookup_table[context]
        
        # FALLBACK: Sparse features → W
        return self._predict_fallback(context)
    
    def _predict_fallback(self, context: bytes) -> int:
        """Fallback prediction via sparse features + compressed W."""
        if self.W_compressed is None:
            return ord(' ')  # Default to space
        
        # Extract sparse features
        feature_indices = self._extract_features(context)
        
        # Build sparse feature vector
        x = np.zeros(self.config.feature_dim, dtype=np.float32)
        for f in feature_indices:
            x[f] = 1.0
        
        # Compressed matmul: x @ U @ V.T
        U, Vt = self.W_compressed
        hidden = x @ U          # [k]
        logits = hidden @ Vt.T  # [vocab_size]
        
        return int(np.argmax(logits))
    
    def predict_probs(self, context: bytes) -> np.ndarray:
        """Get full probability distribution (for sampling)."""
        # For lookup, return one-hot
        if len(context) >= self.config.context_len:
            ctx = context[-self.config.context_len:]
            if ctx in self.lookup_table:
                probs = np.zeros(self.config.vocab_size, dtype=np.float32)
                probs[self.lookup_table[ctx]] = 1.0
                return probs
        
        # Fallback: softmax over logits
        if self.W_compressed is None:
            return np.ones(self.config.vocab_size, dtype=np.float32) / self.config.vocab_size
        
        feature_indices = self._extract_features(context[-self.config.context_len:])
        x = np.zeros(self.config.feature_dim, dtype=np.float32)
        for f in feature_indices:
            x[f] = 1.0
        
        U, Vt = self.W_compressed
        logits = (x @ U) @ Vt.T
        
        # Stable softmax
        logits = logits - logits.max()
        probs = np.exp(logits)
        return probs / probs.sum()
    
    def generate(self, seed: bytes, n_tokens: int, temperature: float = 0.0) -> bytes:
        """Generate text from seed."""
        context = bytearray(seed[-self.config.context_len:])
        output = bytearray(seed)
        
        for _ in range(n_tokens):
            if temperature == 0.0:
                # Greedy
                next_byte = self.predict(bytes(context))
            else:
                # Sample
                probs = self.predict_probs(bytes(context))
                probs = np.power(probs, 1.0 / temperature)
                probs = probs / probs.sum()
                next_byte = np.random.choice(256, p=probs)
            
            output.append(next_byte)
            context = context[1:] + bytearray([next_byte])
        
        return bytes(output)
    
    def get_stats(self) -> dict:
        """Get inference statistics."""
        hit_rate = self.n_lookup_hits / max(1, self.n_total_predictions)
        return {
            "total_predictions": self.n_total_predictions,
            "lookup_hits": self.n_lookup_hits,
            "lookup_hit_rate": hit_rate,
            "n_seen_contexts": self.n_seen_contexts,
        }
    
    # ========================================
    # SERIALIZATION FOR RUST PROVER
    # ========================================
    
    def save_binary(self, path: str):
        """
        Export model to binary format for Rust ZK prover.
        
        Format:
            [Header]
            - magic: 4 bytes "FLEH"
            - version: 4 bytes (1)
            - config: JSON length (4 bytes) + JSON
            
            [Lookup Table]
            - n_entries: 4 bytes
            - entries: [context_len bytes + 1 byte] × n_entries
            
            [Compressed W]
            - rank: 4 bytes
            - U: [feature_dim × rank] float32
            - V: [vocab_size × rank] float32
        """
        print(f"\n💾 Exporting to {path}...")
        
        with open(path, 'wb') as f:
            # Header
            f.write(b'FLEH')  # Magic
            f.write(struct.pack('<I', 1))  # Version
            
            # Config as JSON
            config_json = json.dumps({
                "context_len": self.config.context_len,
                "vocab_size": self.config.vocab_size,
                "feature_dim": self.config.feature_dim,
                "rank": self.config.rank,
                "n_unigram": self.config.n_unigram,
                "n_bigram": self.config.n_bigram,
                "n_trigram": self.config.n_trigram,
                "n_skipgram": self.config.n_skipgram,
            }).encode('utf-8')
            f.write(struct.pack('<I', len(config_json)))
            f.write(config_json)
            
            # Lookup table
            f.write(struct.pack('<I', len(self.lookup_table)))
            for ctx, next_byte in self.lookup_table.items():
                f.write(ctx)
                f.write(bytes([next_byte]))
            
            # Compressed W
            if self.W_compressed is not None:
                U, V = self.W_compressed
                rank = U.shape[1]
                f.write(struct.pack('<I', rank))
                f.write(U.astype(np.float32).tobytes())
                f.write(V.astype(np.float32).tobytes())
            else:
                f.write(struct.pack('<I', 0))
        
        import os
        size = os.path.getsize(path)
        print(f"   ✅ Saved {path} ({size / 1024:.1f} KB)")
    
    @classmethod
    def load_binary(cls, path: str) -> 'FluidEliteHybrid':
        """Load model from binary format."""
        with open(path, 'rb') as f:
            # Header
            magic = f.read(4)
            assert magic == b'FLEH', f"Invalid magic: {magic}"
            version = struct.unpack('<I', f.read(4))[0]
            assert version == 1, f"Unknown version: {version}"
            
            # Config
            config_len = struct.unpack('<I', f.read(4))[0]
            config_json = json.loads(f.read(config_len).decode('utf-8'))
            config = HybridConfig(**config_json)
            
            model = cls(config)
            
            # Lookup table
            n_entries = struct.unpack('<I', f.read(4))[0]
            for _ in range(n_entries):
                ctx = f.read(config.context_len)
                next_byte = f.read(1)[0]
                model.lookup_table[ctx] = next_byte
            model.n_seen_contexts = n_entries
            
            # Compressed W
            rank = struct.unpack('<I', f.read(4))[0]
            if rank > 0:
                U_size = config.feature_dim * rank * 4
                V_size = config.vocab_size * rank * 4
                U = np.frombuffer(f.read(U_size), dtype=np.float32).reshape(config.feature_dim, rank)
                V = np.frombuffer(f.read(V_size), dtype=np.float32).reshape(config.vocab_size, rank)
                model.W_compressed = (U.copy(), V.copy())
        
        return model


def evaluate_model(model: FluidEliteHybrid, test_data: bytes, n_samples: int = 10000) -> dict:
    """Evaluate model on test data."""
    print("\n📊 Evaluating model...")
    
    ctx_len = model.config.context_len
    n_samples = min(n_samples, len(test_data) - ctx_len - 1)
    
    # Random sample positions
    positions = np.random.choice(len(test_data) - ctx_len - 1, size=n_samples, replace=False)
    
    correct_total = 0
    correct_seen = 0
    correct_unseen = 0
    n_seen = 0
    n_unseen = 0
    
    start = time.time()
    for pos in positions:
        ctx = test_data[pos:pos + ctx_len]
        target = test_data[pos + ctx_len]
        
        pred = model.predict(ctx)
        is_seen = ctx in model.lookup_table
        
        if pred == target:
            correct_total += 1
            if is_seen:
                correct_seen += 1
            else:
                correct_unseen += 1
        
        if is_seen:
            n_seen += 1
        else:
            n_unseen += 1
    
    elapsed = time.time() - start
    
    results = {
        "n_samples": n_samples,
        "accuracy_total": correct_total / n_samples,
        "accuracy_seen": correct_seen / max(1, n_seen),
        "accuracy_unseen": correct_unseen / max(1, n_unseen),
        "pct_seen": n_seen / n_samples,
        "throughput": n_samples / elapsed,
    }
    
    print(f"   Total accuracy: {100*results['accuracy_total']:.1f}%")
    print(f"   Seen contexts:  {100*results['accuracy_seen']:.1f}% ({100*results['pct_seen']:.1f}% of test)")
    print(f"   Unseen contexts: {100*results['accuracy_unseen']:.1f}% ({100*(1-results['pct_seen']):.1f}% of test)")
    print(f"   Throughput: {results['throughput']:.0f} predictions/sec")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FluidEliteHybrid: Production ZK-LLM")
    parser.add_argument("--train", type=str, help="Training corpus file")
    parser.add_argument("--test", type=str, help="Test corpus file")
    parser.add_argument("--output", type=str, default="model.bin", help="Output binary file")
    parser.add_argument("--context_len", type=int, default=8, help="Context length")
    parser.add_argument("--rank", type=int, default=24, help="Compression rank")
    parser.add_argument("--generate", type=int, default=0, help="Generate N tokens")
    parser.add_argument("--seed", type=str, default="The ", help="Generation seed")
    args = parser.parse_args()
    
    if args.train:
        # Load training data
        with open(args.train, 'rb') as f:
            train_data = f.read()
        
        # Build model
        config = HybridConfig(context_len=args.context_len, rank=args.rank)
        model = FluidEliteHybrid.from_corpus(train_data, config)
        
        # Evaluate on test if provided
        if args.test:
            with open(args.test, 'rb') as f:
                test_data = f.read()
            evaluate_model(model, test_data)
        
        # Save
        model.save_binary(args.output)
        
        # Generate sample
        if args.generate > 0:
            print("\n📝 Sample generation:")
            sample = model.generate(args.seed.encode(), args.generate)
            print(sample.decode('utf-8', errors='replace'))
    else:
        print("Usage: python fluidelite_hybrid.py --train corpus.txt --output model.bin")
