"""
Differentiable QTT Module
=========================

Fully differentiable Tensor Train operations for end-to-end gradient-based
optimization. Enables backpropagation through the entire QTT pipeline.

Key Components:
1. DifferentiableQTTCores - nn.Module wrapping TT cores as Parameters
2. NuclearNormRegularizer - Complexity penalty for Occam's Razor optimization
3. RankAdaptiveQTT - Dynamic rank expansion/contraction during training

Mathematical Foundation:
- TT decomposition: A ≈ A^(1) × A^(2) × ... × A^(d)
- Nuclear norm: R = λ Σ_k ||σ(A^(k))||_1
- Rank adaptation: Monitor energy residue, expand/contract bonds

References:
- Oseledets (2011): Tensor-Train Decomposition
- Halko et al. (2011): Finding Structure with Randomness
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# NUCLEAR NORM REGULARIZER
# =============================================================================

class NuclearNormRegularizer(nn.Module):
    """
    Nuclear norm regularizer for QTT cores.
    
    Adds a penalty based on the sum of singular values at each bond,
    encouraging low-rank (simpler) representations.
    
    Loss:
        R_nuclear = λ Σ_{k=1}^{d-1} ||σ(A^(k))||_1
    
    This implements Occam's Razor for tensor networks:
    - Large singular values → important structure (signal)
    - Small singular values → noise (penalized away)
    
    The regularizer is fully differentiable since SVD gradients flow
    through distinct singular values.
    """
    
    def __init__(
        self,
        lambda_reg: float = 1e-4,
        normalize: bool = True,
        per_bond: bool = False,
        device: torch.device = DEVICE
    ):
        """
        Args:
            lambda_reg: Regularization strength (higher = more compression)
            normalize: If True, normalize by number of bonds
            per_bond: If True, return per-bond breakdown
            device: Computation device
        """
        super().__init__()
        self.lambda_reg = lambda_reg
        self.normalize = normalize
        self.per_bond = per_bond
        self.device = device
        
        # Track singular value statistics for monitoring
        self.register_buffer('_sv_history', torch.zeros(100))
        self.register_buffer('_sv_idx', torch.tensor(0))
    
    def forward(
        self,
        cores: List[torch.Tensor],
        return_breakdown: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Compute nuclear norm regularization loss.
        
        Args:
            cores: List of TT cores, each shape (r_left, d_k, r_right)
            return_breakdown: If True, return per-bond singular values
            
        Returns:
            loss: Scalar regularization loss (differentiable)
            breakdown: (optional) Dict with per-bond statistics
        """
        if not cores:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        total_nuclear_norm = torch.tensor(0.0, device=self.device)
        bond_norms = []
        bond_svs = []
        
        for k, core in enumerate(cores[:-1]):  # Last core has no right bond to regularize
            # Unfold core: (r_left, d_k, r_right) -> (r_left * d_k, r_right)
            r_left, d_k, r_right = core.shape
            unfolded = core.reshape(r_left * d_k, r_right)
            
            # Compute singular values (differentiable)
            # Use torch.linalg.svdvals for efficiency - only computes σ, not U,V
            try:
                sigma = torch.linalg.svdvals(unfolded)
            except RuntimeError:
                # Fallback for edge cases
                _, sigma, _ = torch.svd(unfolded, compute_uv=False)
            
            # Nuclear norm = sum of singular values
            bond_norm = sigma.sum()
            total_nuclear_norm = total_nuclear_norm + bond_norm
            
            bond_norms.append(bond_norm)
            bond_svs.append(sigma.detach())
        
        # Normalize by number of bonds if requested
        num_bonds = len(cores) - 1
        if self.normalize and num_bonds > 0:
            total_nuclear_norm = total_nuclear_norm / num_bonds
        
        # Apply regularization strength
        loss = self.lambda_reg * total_nuclear_norm
        
        # Update history for monitoring
        with torch.no_grad():
            idx = int(self._sv_idx.item()) % 100
            self._sv_history[idx] = total_nuclear_norm.item()
            self._sv_idx += 1
        
        if return_breakdown or self.per_bond:
            breakdown = {
                'total_nuclear_norm': total_nuclear_norm.detach(),
                'bond_norms': [bn.detach() for bn in bond_norms],
                'bond_singular_values': bond_svs,
                'num_bonds': num_bonds,
                'lambda': self.lambda_reg,
                'loss': loss.detach()
            }
            return loss, breakdown
        
        return loss
    
    def get_energy_distribution(self, cores: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute energy distribution across bonds.
        
        Energy at bond k = Σ σ_i² (Frobenius norm squared)
        
        Returns:
            Tensor of shape (num_bonds,) with energy per bond
        """
        energies = []
        for k, core in enumerate(cores[:-1]):
            r_left, d_k, r_right = core.shape
            unfolded = core.reshape(r_left * d_k, r_right)
            sigma = torch.linalg.svdvals(unfolded)
            energy = (sigma ** 2).sum()
            energies.append(energy)
        
        return torch.stack(energies) if energies else torch.tensor([0.0], device=self.device)
    
    def get_rank_profile(self, cores: List[torch.Tensor], threshold: float = 1e-6) -> List[int]:
        """
        Compute effective rank at each bond (singular values above threshold).
        
        Args:
            cores: TT cores
            threshold: Relative threshold for counting significant singular values
            
        Returns:
            List of effective ranks per bond
        """
        effective_ranks = []
        for k, core in enumerate(cores[:-1]):
            r_left, d_k, r_right = core.shape
            unfolded = core.reshape(r_left * d_k, r_right)
            sigma = torch.linalg.svdvals(unfolded)
            
            # Count singular values above relative threshold
            rel_threshold = threshold * sigma[0] if len(sigma) > 0 else threshold
            effective_rank = (sigma > rel_threshold).sum().item()
            effective_ranks.append(int(effective_rank))
        
        return effective_ranks


# =============================================================================
# DIFFERENTIABLE QTT CORES MODULE
# =============================================================================

class DifferentiableQTTCores(nn.Module):
    """
    Wraps TT cores as trainable nn.Parameters.
    
    This enables gradient-based optimization of the core values while
    keeping the rank structure fixed. For rank adaptation, use
    RankAdaptiveQTT.
    
    The forward pass reconstructs the full tensor (for small tensors)
    or computes contractions (for large tensors).
    """
    
    def __init__(
        self,
        cores: List[torch.Tensor],
        requires_grad: bool = True,
        device: torch.device = DEVICE
    ):
        """
        Args:
            cores: Initial TT cores from decomposition
            requires_grad: Whether cores are trainable
            device: Computation device
        """
        super().__init__()
        self.device = device
        self.ndim = len(cores)
        
        # Store shape information
        self.shape = self._infer_shape(cores)
        self.ranks = self._get_ranks(cores)
        
        # Register cores as parameters
        self.cores = nn.ParameterList([
            nn.Parameter(core.to(device).clone(), requires_grad=requires_grad)
            for core in cores
        ])
    
    def _infer_shape(self, cores: List[torch.Tensor]) -> Tuple[int, ...]:
        """Infer original tensor shape from cores."""
        return tuple(core.shape[1] for core in cores)
    
    def _get_ranks(self, cores: List[torch.Tensor]) -> List[int]:
        """Get bond dimensions."""
        return [core.shape[2] for core in cores[:-1]]
    
    def forward(self, contract: bool = False) -> torch.Tensor:
        """
        Forward pass - either return cores or contract to tensor.
        
        Args:
            contract: If True, contract cores to full tensor (expensive!)
                     If False, return list of cores
        
        Returns:
            Full tensor if contract=True, else list of cores
        """
        if not contract:
            return list(self.cores)
        
        # Contract cores to full tensor (only for small tensors!)
        # This is O(d * n * r²) - expensive for large tensors
        result = self.cores[0].squeeze(0)  # (d_0, r_0)
        
        for core in self.cores[1:]:
            # result: (..., r_k)
            # core: (r_k, d_k, r_{k+1})
            result = torch.einsum('...r,rdr2->...dr2', result, core)
        
        return result.squeeze(-1)  # Remove final dimension (should be 1)
    
    def contract_with_vector(self, vectors: List[torch.Tensor]) -> torch.Tensor:
        """
        Contract TT with vectors at each mode.
        
        Args:
            vectors: List of vectors, one per mode, shape (d_k,)
            
        Returns:
            Scalar result of contraction
        """
        result = self.cores[0]  # (1, d_0, r_0)
        result = torch.einsum('ldr,d->lr', result, vectors[0])  # (1, r_0)
        
        for k, (core, vec) in enumerate(zip(self.cores[1:], vectors[1:]), 1):
            # result: (1, r_k)
            # core: (r_k, d_k, r_{k+1})
            contracted = torch.einsum('rds,d->rs', core, vec)  # (r_k, r_{k+1})
            result = result @ contracted  # (1, r_{k+1})
        
        return result.squeeze()
    
    def get_cores(self) -> List[torch.Tensor]:
        """Return list of cores (for compatibility)."""
        return list(self.cores)
    
    def nuclear_norm(self) -> torch.Tensor:
        """Compute total nuclear norm of all bonds."""
        total = torch.tensor(0.0, device=self.device)
        for core in self.cores[:-1]:
            r_left, d_k, r_right = core.shape
            unfolded = core.reshape(r_left * d_k, r_right)
            sigma = torch.linalg.svdvals(unfolded)
            total = total + sigma.sum()
        return total
    
    def frobenius_norm(self) -> torch.Tensor:
        """Compute Frobenius norm of the TT (efficient, no full reconstruction)."""
        # ||A||_F² = <A, A> in TT format
        # Use core-wise contraction
        result = torch.einsum('ldr,ldr->r', self.cores[0], self.cores[0])  # (r_0,)
        
        for core in self.cores[1:]:
            # result: (r_k,) representing <partial, partial>
            # Need: result @ (core ⊗ core contracted over d)
            core_gram = torch.einsum('rds,rds->rs', core, core)  # (r_k, r_{k+1})
            result = torch.einsum('r,rs->s', result, core_gram)
        
        return torch.sqrt(result.sum())


# =============================================================================
# RANK ADAPTIVE QTT
# =============================================================================

@dataclass
class RankAdaptationConfig:
    """Configuration for rank adaptation."""
    chi_min: int = 2
    chi_max: int = 128
    energy_threshold: float = 1e-6  # Expand if discarded energy > this
    compression_threshold: float = 1e-8  # Contract if singular values < this
    adaptation_rate: float = 1.5  # Factor for rank change
    cooldown_steps: int = 10  # Steps between adaptations


class RankAdaptiveQTT(nn.Module):
    """
    QTT with dynamic rank adaptation.
    
    Monitors the singular value spectrum during training and:
    - EXPANDS rank when discarded energy exceeds threshold (underfitting)
    - CONTRACTS rank when many singular values are near zero (overfitting)
    
    The adaptation is triggered by the TruncationPolicy or by explicit calls.
    """
    
    def __init__(
        self,
        cores: List[torch.Tensor],
        config: Optional[RankAdaptationConfig] = None,
        device: torch.device = DEVICE
    ):
        """
        Args:
            cores: Initial TT cores
            config: Rank adaptation configuration
            device: Computation device
        """
        super().__init__()
        self.device = device
        self.config = config or RankAdaptationConfig()
        
        # Initialize differentiable cores
        self.qtt = DifferentiableQTTCores(cores, requires_grad=True, device=device)
        
        # Track adaptation state
        self.register_buffer('_step', torch.tensor(0))
        self.register_buffer('_last_adaptation', torch.tensor(0))
        self.register_buffer('_discarded_energy', torch.zeros(len(cores) - 1))
    
    def forward(self) -> List[torch.Tensor]:
        """Return current cores."""
        return self.qtt.get_cores()
    
    def compute_discarded_energy(
        self,
        original_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute energy discarded by current rank truncation.
        
        Args:
            original_tensor: Full tensor to compare against
            
        Returns:
            total_discarded: Scalar total discarded energy
            per_bond: Per-bond discarded energy
        """
        # Reconstruct tensor from cores
        reconstructed = self.qtt(contract=True)
        
        # Compute residual energy
        residual = original_tensor.to(self.device) - reconstructed
        discarded = (residual ** 2).sum()
        
        # Per-bond analysis would require intermediate reconstructions
        # For now, return uniform distribution as approximation
        num_bonds = len(self.qtt.cores) - 1
        per_bond = discarded / num_bonds * torch.ones(num_bonds, device=self.device)
        
        self._discarded_energy = per_bond.detach()
        
        return discarded, per_bond
    
    def should_expand(self) -> Tuple[bool, List[int]]:
        """
        Check if any bonds should be expanded.
        
        Returns:
            should: Whether expansion is needed
            bonds: List of bond indices to expand
        """
        # Check cooldown
        if self._step - self._last_adaptation < self.config.cooldown_steps:
            return False, []
        
        bonds_to_expand = []
        for k, energy in enumerate(self._discarded_energy):
            if energy > self.config.energy_threshold:
                current_rank = self.qtt.ranks[k] if k < len(self.qtt.ranks) else 1
                if current_rank < self.config.chi_max:
                    bonds_to_expand.append(k)
        
        return len(bonds_to_expand) > 0, bonds_to_expand
    
    def should_contract(self) -> Tuple[bool, List[int]]:
        """
        Check if any bonds should be contracted.
        
        Returns:
            should: Whether contraction is needed
            bonds: List of bond indices to contract
        """
        if self._step - self._last_adaptation < self.config.cooldown_steps:
            return False, []
        
        bonds_to_contract = []
        for k, core in enumerate(self.qtt.cores[:-1]):
            r_left, d_k, r_right = core.shape
            unfolded = core.reshape(r_left * d_k, r_right)
            sigma = torch.linalg.svdvals(unfolded)
            
            # Check how many singular values are below threshold
            rel_threshold = self.config.compression_threshold * sigma[0]
            effective_rank = (sigma > rel_threshold).sum().item()
            
            if effective_rank < r_right * 0.5:  # Can reduce by half
                bonds_to_contract.append(k)
        
        return len(bonds_to_contract) > 0, bonds_to_contract
    
    def expand_rank(self, bond_indices: List[int]) -> None:
        """
        Expand rank at specified bonds by zero-padding.
        
        This preserves the current learned state while adding capacity.
        
        Args:
            bond_indices: Which bonds to expand
        """
        with torch.no_grad():
            for k in bond_indices:
                if k >= len(self.qtt.cores) - 1:
                    continue
                
                core = self.qtt.cores[k]
                next_core = self.qtt.cores[k + 1]
                
                r_left, d_k, r_right = core.shape
                new_r_right = min(
                    int(r_right * self.config.adaptation_rate),
                    self.config.chi_max
                )
                delta = new_r_right - r_right
                
                if delta <= 0:
                    continue
                
                # Pad current core's right dimension
                padding = torch.zeros(r_left, d_k, delta, device=self.device)
                new_core = torch.cat([core.data, padding], dim=2)
                self.qtt.cores[k].data = new_core
                
                # Pad next core's left dimension
                r_left_next, d_next, r_right_next = next_core.shape
                padding_next = torch.zeros(delta, d_next, r_right_next, device=self.device)
                new_next_core = torch.cat([next_core.data, padding_next], dim=0)
                self.qtt.cores[k + 1].data = new_next_core
            
            self._last_adaptation = self._step.clone()
            self.qtt.ranks = self.qtt._get_ranks(list(self.qtt.cores))
    
    def contract_rank(self, bond_indices: List[int], target_ranks: Optional[List[int]] = None) -> None:
        """
        Contract rank at specified bonds via SVD truncation.
        
        Args:
            bond_indices: Which bonds to contract
            target_ranks: Target ranks (if None, use effective rank)
        """
        with torch.no_grad():
            for i, k in enumerate(bond_indices):
                if k >= len(self.qtt.cores) - 1:
                    continue
                
                core = self.qtt.cores[k]
                next_core = self.qtt.cores[k + 1]
                
                r_left, d_k, r_right = core.shape
                
                # Compute effective rank
                unfolded = core.reshape(r_left * d_k, r_right)
                U, S, Vh = torch.linalg.svd(unfolded, full_matrices=False)
                
                rel_threshold = self.config.compression_threshold * S[0]
                effective_rank = max((S > rel_threshold).sum().item(), self.config.chi_min)
                
                if target_ranks is not None and i < len(target_ranks):
                    new_rank = target_ranks[i]
                else:
                    new_rank = effective_rank
                
                new_rank = max(new_rank, self.config.chi_min)
                
                if new_rank >= r_right:
                    continue
                
                # Truncate
                U_trunc = U[:, :new_rank]
                S_trunc = S[:new_rank]
                Vh_trunc = Vh[:new_rank, :]
                
                # Update current core
                new_core = U_trunc.reshape(r_left, d_k, new_rank)
                self.qtt.cores[k].data = new_core
                
                # Absorb S @ Vh into next core
                absorbed = torch.diag(S_trunc) @ Vh_trunc  # (new_rank, r_right)
                r_left_next, d_next, r_right_next = next_core.shape
                # next_core: (r_right, d_next, r_right_next)
                # absorbed @ next_core reshaped
                next_reshaped = next_core.reshape(r_right, d_next * r_right_next)
                new_next = (absorbed @ next_reshaped).reshape(new_rank, d_next, r_right_next)
                self.qtt.cores[k + 1].data = new_next
            
            self._last_adaptation = self._step.clone()
            self.qtt.ranks = self.qtt._get_ranks(list(self.qtt.cores))
    
    def step(self) -> None:
        """Increment step counter."""
        self._step += 1


# =============================================================================
# DIFFERENTIABLE DISCOVERY LOSS
# =============================================================================

class DifferentiableDiscoveryLoss(nn.Module):
    """
    Composite loss function for differentiable discovery.
    
    Combines:
    1. Task-specific loss (e.g., reconstruction, prediction)
    2. Nuclear norm regularization (complexity penalty)
    3. Optional entropy regularization
    
    Total loss:
        L_total = L_task + λ_nuclear * R_nuclear + λ_entropy * R_entropy
    """
    
    def __init__(
        self,
        lambda_nuclear: float = 1e-4,
        lambda_entropy: float = 0.0,
        device: torch.device = DEVICE
    ):
        """
        Args:
            lambda_nuclear: Weight for nuclear norm regularization
            lambda_entropy: Weight for entropy regularization (0 = disabled)
            device: Computation device
        """
        super().__init__()
        self.device = device
        self.lambda_nuclear = lambda_nuclear
        self.lambda_entropy = lambda_entropy
        
        self.nuclear_reg = NuclearNormRegularizer(
            lambda_reg=1.0,  # We apply lambda_nuclear ourselves
            normalize=True,
            device=device
        )
    
    def forward(
        self,
        task_loss: torch.Tensor,
        cores: List[torch.Tensor],
        return_breakdown: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Compute total loss.
        
        Args:
            task_loss: Primary task loss (must be differentiable)
            cores: TT cores for regularization
            return_breakdown: If True, return component losses
            
        Returns:
            total_loss: Combined loss (differentiable)
            breakdown: (optional) Dict with component losses
        """
        # Nuclear norm regularization
        nuclear_loss, nuclear_breakdown = self.nuclear_reg(cores, return_breakdown=True)
        nuclear_loss = self.lambda_nuclear * nuclear_loss
        
        # Entropy regularization (optional)
        entropy_loss = torch.tensor(0.0, device=self.device)
        if self.lambda_entropy > 0:
            entropy_loss = self._compute_entropy_regularization(cores)
            entropy_loss = self.lambda_entropy * entropy_loss
        
        total_loss = task_loss + nuclear_loss + entropy_loss
        
        if return_breakdown:
            breakdown = {
                'total': total_loss.detach(),
                'task': task_loss.detach(),
                'nuclear': nuclear_loss.detach(),
                'entropy': entropy_loss.detach(),
                'nuclear_breakdown': nuclear_breakdown
            }
            return total_loss, breakdown
        
        return total_loss
    
    def _compute_entropy_regularization(self, cores: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute entropy regularization based on singular value distribution.
        
        High entropy = flat singular value spectrum = overly complex
        Low entropy = peaked spectrum = good compression
        """
        total_entropy = torch.tensor(0.0, device=self.device)
        
        for core in cores[:-1]:
            r_left, d_k, r_right = core.shape
            unfolded = core.reshape(r_left * d_k, r_right)
            sigma = torch.linalg.svdvals(unfolded)
            
            # Normalize to probability distribution
            sigma_norm = sigma / (sigma.sum() + 1e-10)
            
            # Compute entropy: -Σ p log p
            entropy = -torch.sum(sigma_norm * torch.log(sigma_norm + 1e-10))
            total_entropy = total_entropy + entropy
        
        return total_entropy / max(len(cores) - 1, 1)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def qtt_from_tensor(
    tensor: torch.Tensor,
    max_rank: int = 32,
    tolerance: float = 1e-6,
    device: torch.device = DEVICE
) -> DifferentiableQTTCores:
    """
    Create differentiable QTT from full tensor via TT-SVD.
    
    Args:
        tensor: Full tensor to decompose
        max_rank: Maximum bond dimension
        tolerance: Truncation tolerance (relative to largest singular value)
        device: Computation device
        
    Returns:
        DifferentiableQTTCores module with trained parameters
    """
    tensor = tensor.to(device)
    shape = tensor.shape
    ndim = tensor.ndim
    
    cores = []
    current = tensor.reshape(shape[0], -1)
    ranks = []
    
    for i in range(ndim - 1):
        m, n = current.shape
        k = min(max_rank, m, n)
        
        U, S, Vh = torch.linalg.svd(current, full_matrices=False)
        
        # Truncate by tolerance
        if tolerance > 0 and len(S) > 0:
            rel_threshold = tolerance * S[0]
            k_eff = max((S > rel_threshold).sum().item(), 1)
            k_eff = min(k_eff, k)
        else:
            k_eff = k
        
        U = U[:, :k_eff]
        S = S[:k_eff]
        Vh = Vh[:k_eff, :]
        
        r_left = 1 if i == 0 else ranks[-1]
        d_i = shape[i]
        r_right = k_eff
        
        core = U.reshape(r_left, d_i, r_right)
        cores.append(core)
        ranks.append(r_right)
        
        current = torch.diag(S) @ Vh
        
        if i < ndim - 2:
            d_next = shape[i + 1]
            current = current.reshape(k_eff * d_next, -1)
    
    # Last core
    r_last = ranks[-1] if ranks else 1
    d_last = shape[-1]
    last_core = current.reshape(r_last, d_last, 1)
    cores.append(last_core)
    
    return DifferentiableQTTCores(cores, requires_grad=True, device=device)


def reconstruct_from_cores(cores: List[torch.Tensor]) -> torch.Tensor:
    """
    Reconstruct full tensor from TT cores using stable contraction.
    
    Uses batched matrix multiplication instead of einsum to avoid
    subscript overflow for large tensors.
    
    Args:
        cores: List of TT cores, each with shape (r_left, d_k, r_right)
        
    Returns:
        Reconstructed tensor with shape (d_1, d_2, ..., d_n)
    """
    if len(cores) == 0:
        raise ValueError("Empty cores list")
    
    # Start with first core: (1, d_1, r_1) -> (d_1, r_1)
    result = cores[0].squeeze(0)  # (d_1, r_1)
    
    # Iteratively contract: result has shape (d_1, ..., d_k, r_k)
    for core in cores[1:]:
        # core has shape (r_k, d_{k+1}, r_{k+1})
        r_left, d_k, r_right = core.shape
        
        # Reshape result to (..., r_k)
        batch_shape = result.shape[:-1]
        batch_size = result[..., 0].numel()
        
        # Flatten batch dims: (batch_size, r_k)
        result_flat = result.reshape(batch_size, r_left)
        
        # Reshape core: (r_k, d_k * r_right) for matmul
        core_flat = core.reshape(r_left, d_k * r_right)
        
        # Contract: (batch_size, d_k * r_right)
        contracted = torch.mm(result_flat, core_flat)
        
        # Reshape back: (*batch_shape, d_k, r_right)
        result = contracted.reshape(*batch_shape, d_k, r_right)
    
    # Final result: (*all_dims, 1) -> (*all_dims)
    return result.squeeze(-1)


def compute_reconstruction_loss(
    cores: List[torch.Tensor],
    target: torch.Tensor,
    loss_type: str = 'mse'
) -> torch.Tensor:
    """
    Compute reconstruction loss between QTT and target tensor.
    
    Args:
        cores: TT cores
        target: Target tensor (same shape as reconstructed)
        loss_type: 'mse', 'mae', or 'huber'
        
    Returns:
        Differentiable reconstruction loss
    """
    reconstructed = reconstruct_from_cores(cores)
    
    if loss_type == 'mse':
        return F.mse_loss(reconstructed, target)
    elif loss_type == 'mae':
        return F.l1_loss(reconstructed, target)
    elif loss_type == 'huber':
        return F.huber_loss(reconstructed, target)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
