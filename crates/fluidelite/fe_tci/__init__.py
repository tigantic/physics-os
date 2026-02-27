"""
FluidElite-TCI: Gradient-Free MPS Language Model

This is NOT a compressed transformer. This is a NEW ARCHITECTURE:
- Recurrent state in MPS (not parallel attention)
- O(1) memory regardless of context length
- Infinite context window
- Trained via TCI (no gradients)

Architecture:
    Token_n → Binary Encode → MPS_input (χ=1)
    MPS_state → MPO W_hidden → MPS_temp
    MPS_temp + MPS_input → MPS_state_new (truncate to χ_max)
    MPS_state_new → TCI Predict Head → Token_n+1

The whole model is just: f(context_tokens) → next_token
TCI the end-to-end function and let the decomposition fall out.

No gradients. No backprop. No truncation gradient problem.
"""

__version__ = "0.1.0"
__author__ = "HyperTensor Labs"

from .context_encoder import ContextEncoder
from .fluidelite_model import FluidEliteModel

__all__ = ["ContextEncoder", "FluidEliteModel"]
