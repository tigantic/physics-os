"""
Bounded Memory Training for FluidElite
=======================================

The core QTT thesis: Memory is O(1) regardless of sequence length.

This applies to TRAINING, not just inference.

Problem:
    PyTorch autograd stores every intermediate tensor:
    - step 1 → save activation  
    - step 2 → save activation
    - ...
    - step 1000 → save activation
    - backward → needs all 1000 saved tensors
    
    Result: Training memory = O(seq_len), defeats the whole point.

Solution:
    Gradient checkpointing with MPS state reconstruction.
    
    Forward: Run full sequence, checkpoint MPS every N steps
    Backward: Recompute activations from checkpoints on demand
    
    Memory = O(num_checkpoints) = O(seq_len / checkpoint_interval)
    With checkpoint_interval = seq_len: Memory = O(1)

Usage:
    >>> from fluidelite.core.bounded_training import bounded_train_step
    >>> 
    >>> loss = bounded_train_step(model, tokens, checkpoint_every=16)
    >>> loss.backward()  # Memory bounded!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import List, Tuple, Optional
from dataclasses import dataclass


def _compute_segment_loss(model, tokens: torch.Tensor, start_idx: int, 
                          mps_tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute loss for a segment of tokens.
    This is checkpointed - intermediates are recomputed on backward.
    """
    from fluidelite.core.mps import MPS
    
    # Restore MPS from tensor list
    ctx_mps = MPS(list(mps_tensors))  # mps_tensors is a tuple from checkpoint
    
    segment_loss = torch.tensor(0.0, device=tokens.device, dtype=torch.float32)
    
    for t in range(len(tokens) - 1):
        # Predict
        logits = model.predict(ctx_mps)
        target = tokens[t + 1]
        step_loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
        segment_loss = segment_loss + step_loss
        
        # Step
        ctx_mps = model.step(ctx_mps, tokens[t].item())
    
    return segment_loss


def bounded_train_step(model, tokens: torch.Tensor, initial_ctx=None,
                       segment_len: int = 16) -> torch.Tensor:
    """
    Memory-bounded training step using gradient checkpointing.
    
    Instead of storing all intermediates, we:
    1. Break sequence into segments
    2. Checkpoint each segment (PyTorch's checkpoint recomputes on backward)
    3. Total memory = O(num_segments × segment_size), not O(seq_len × activation_size)
    
    Args:
        model: FluidElite model
        tokens: Input token tensor [seq_len]
        initial_ctx: Initial MPS context (or None for random)
        segment_len: Tokens per segment (lower = less memory, more recompute)
        
    Returns:
        Loss tensor
    """
    from fluidelite.core.mps import MPS
    
    device = tokens.device
    dtype = next(model.parameters()).dtype
    seq_len = len(tokens)
    
    # Initialize context
    if initial_ctx is None:
        initial_ctx = MPS.random(model.L, d=2, chi=1, device=device, dtype=dtype)
    
    # Process in segments with checkpointing
    total_loss = torch.tensor(0.0, device=device, dtype=dtype)
    ctx_mps = initial_ctx
    
    num_segments = (seq_len - 1 + segment_len - 1) // segment_len
    
    for seg_idx in range(num_segments):
        start = seg_idx * segment_len
        end = min(start + segment_len + 1, seq_len)
        
        if start >= end - 1:
            break
        
        segment_tokens = tokens[start:end]
        
        # Pass MPS tensors as tuple (checkpoint needs tensors, not objects)
        mps_tensors = tuple(ctx_mps.tensors)
        
        # Checkpoint this segment - recomputes on backward to save memory
        segment_loss = checkpoint(
            _compute_segment_loss,
            model, segment_tokens, start, mps_tensors,
            use_reentrant=False
        )
        
        total_loss = total_loss + segment_loss
        
        # Advance context to end of segment (without grad to save memory)
        with torch.no_grad():
            for t in range(len(segment_tokens) - 1):
                ctx_mps = model.step(ctx_mps, segment_tokens[t].item())
    
    return total_loss


class BoundedTrainer:
    """
    High-level trainer with memory-bounded training.
    
    Example:
        >>> trainer = BoundedTrainer(model, lr=1e-3, segment_len=16)
        >>> for epoch in range(100):
        ...     loss = trainer.train_step(tokens)
        ...     print(f"Epoch {epoch}: loss={loss:.4f}")
    """
    
    def __init__(self, model, lr: float = 1e-3, segment_len: int = 16,
                 grad_clip: float = 1.0):
        self.model = model
        self.segment_len = segment_len
        self.grad_clip = grad_clip
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
    def train_step(self, tokens: torch.Tensor, initial_ctx=None) -> float:
        """
        Single training step with bounded memory.
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        loss = bounded_train_step(
            self.model, tokens, initial_ctx, 
            segment_len=self.segment_len
        )
        
        loss.backward()
        
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, data: torch.Tensor, bptt_len: int = 64, 
                    log_every: int = 100) -> dict:
        """
        Train for one epoch.
        """
        import time
        from fluidelite.core.mps import MPS
        
        self.model.train()
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        
        num_windows = (len(data) - 1) // bptt_len
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        start_time = time.time()
        
        for window_idx in range(num_windows):
            start = window_idx * bptt_len
            end = start + bptt_len + 1
            if end > len(data):
                break
            
            window = data[start:end].to(device)
            ctx = MPS.random(self.model.L, d=2, chi=1, device=device, dtype=dtype)
            
            loss = self.train_step(window, ctx)
            total_loss += loss
            
            # Track accuracy
            with torch.no_grad():
                ctx = MPS.random(self.model.L, d=2, chi=1, device=device, dtype=dtype)
                for t in range(len(window) - 1):
                    logits = self.model.predict(ctx)
                    if logits.argmax() == window[t + 1]:
                        total_correct += 1
                    total_tokens += 1
                    ctx = self.model.step(ctx, window[t].item())
            
            if (window_idx + 1) % log_every == 0:
                elapsed = time.time() - start_time
                tok_per_sec = total_tokens / elapsed
                avg_loss = total_loss / (window_idx + 1)
                acc = 100.0 * total_correct / total_tokens
                
                if torch.cuda.is_available():
                    mem_gb = torch.cuda.max_memory_allocated() / 1e9
                    print(f"  [{window_idx+1:5d}/{num_windows}] "
                          f"loss={avg_loss:.4f}, acc={acc:.1f}%, "
                          f"{tok_per_sec:.0f} tok/s, VRAM={mem_gb:.2f}GB")
                else:
                    print(f"  [{window_idx+1:5d}/{num_windows}] "
                          f"loss={avg_loss:.4f}, acc={acc:.1f}%, "
                          f"{tok_per_sec:.0f} tok/s")
        
        elapsed = time.time() - start_time
        return {
            "loss": total_loss / max(1, num_windows),
            "acc": 100.0 * total_correct / max(1, total_tokens),
            "tok_per_sec": total_tokens / elapsed
        }


# Keep old name for compatibility
def bounded_forward(model, tokens, initial_ctx=None, checkpoint_every=16):
    """Alias for bounded_train_step."""
    return bounded_train_step(model, tokens, initial_ctx, segment_len=checkpoint_every)
