"""
Standard MPS States
===================

Factory functions for common quantum states.
"""

from typing import Optional, List
import torch
from torch import Tensor
import math

from tensornet.core.mps import MPS


def ghz_mps(
    L: int,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> MPS:
    """
    Create GHZ state: |GHZ⟩ = (|00...0⟩ + |11...1⟩) / √2
    
    This is a maximally entangled state with S = ln(2) at every bond.
    
    Args:
        L: Number of sites
        dtype: Data type
        device: Device
        
    Returns:
        MPS representation of GHZ state
    """
    if device is None:
        device = torch.device('cpu')
    
    tensors = []
    
    # First site: (1, 2, 2)
    A0 = torch.zeros(1, 2, 2, dtype=dtype, device=device)
    A0[0, 0, 0] = 1.0 / math.sqrt(2)
    A0[0, 1, 1] = 1.0 / math.sqrt(2)
    tensors.append(A0)
    
    # Middle sites: (2, 2, 2) - identity on each branch
    for i in range(1, L - 1):
        A = torch.zeros(2, 2, 2, dtype=dtype, device=device)
        A[0, 0, 0] = 1.0
        A[1, 1, 1] = 1.0
        tensors.append(A)
    
    # Last site: (2, 2, 1)
    AL = torch.zeros(2, 2, 1, dtype=dtype, device=device)
    AL[0, 0, 0] = 1.0
    AL[1, 1, 0] = 1.0
    tensors.append(AL)
    
    return MPS(tensors)


def product_mps(
    states: List[Tensor],
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> MPS:
    """
    Create product state MPS from local states.
    
    |ψ⟩ = |ψ₀⟩ ⊗ |ψ₁⟩ ⊗ ... ⊗ |ψ_{L-1}⟩
    
    Args:
        states: List of local state vectors, each of shape (d,)
        dtype: Data type
        device: Device
        
    Returns:
        MPS with bond dimension 1
    """
    if device is None:
        device = torch.device('cpu')
    
    tensors = []
    for state in states:
        # Reshape (d,) -> (1, d, 1)
        A = state.to(dtype=dtype, device=device).reshape(1, -1, 1)
        tensors.append(A)
    
    return MPS(tensors)


def random_mps(
    L: int,
    d: int,
    chi: int,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
    normalize: bool = True,
) -> MPS:
    """
    Create random MPS.
    
    Alias for MPS.random() for convenience.
    
    Args:
        L: Number of sites
        d: Physical dimension
        chi: Bond dimension
        dtype: Data type
        device: Device
        normalize: Normalize the state
        
    Returns:
        Random MPS
    """
    return MPS.random(L, d, chi, dtype=dtype, device=device, normalize=normalize)


def all_up_mps(
    L: int,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> MPS:
    """
    Create |↑↑...↑⟩ state.
    
    Args:
        L: Number of sites
        dtype: Data type
        device: Device
        
    Returns:
        All-up product state
    """
    up = torch.tensor([1.0, 0.0], dtype=dtype, device=device)
    return product_mps([up] * L, dtype=dtype, device=device)


def all_down_mps(
    L: int,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> MPS:
    """
    Create |↓↓...↓⟩ state.
    
    Args:
        L: Number of sites
        dtype: Data type
        device: Device
        
    Returns:
        All-down product state
    """
    down = torch.tensor([0.0, 1.0], dtype=dtype, device=device)
    return product_mps([down] * L, dtype=dtype, device=device)


def neel_mps(
    L: int,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> MPS:
    """
    Create Néel state: |↑↓↑↓...⟩
    
    Args:
        L: Number of sites
        dtype: Data type
        device: Device
        
    Returns:
        Néel product state
    """
    up = torch.tensor([1.0, 0.0], dtype=dtype, device=device)
    down = torch.tensor([0.0, 1.0], dtype=dtype, device=device)
    states = [up if i % 2 == 0 else down for i in range(L)]
    return product_mps(states, dtype=dtype, device=device)


def domain_wall_mps(
    L: int,
    position: int,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> MPS:
    """
    Create domain wall state: |↑↑...↑↓↓...↓⟩
    
    Args:
        L: Number of sites
        position: Position of domain wall (sites 0..position-1 are up)
        dtype: Data type
        device: Device
        
    Returns:
        Domain wall product state
    """
    up = torch.tensor([1.0, 0.0], dtype=dtype, device=device)
    down = torch.tensor([0.0, 1.0], dtype=dtype, device=device)
    states = [up if i < position else down for i in range(L)]
    return product_mps(states, dtype=dtype, device=device)
