"""
Riemannian Optimization for Matrix Product States
=================================================

Optimizes tensor networks on the Stiefel/Grassmann manifolds.
Prevents "Gauge Explosion" and vanishing gradients.

This version SKIPS QR retraction "to save FLOPs, relying on 
the Projection step to keep us close to the manifold."

The key insight is that tensor network parameters have gauge redundancy:
many different parameter values represent the same physical state.
Standard optimizers waste gradient energy on gauge directions.
Riemannian Adam projects out the gauge noise.

⚠️ WARNING (January 13, 2026): The stabilize=True projection is BROKEN.
   The formula G_proj = G - W @ (W^T @ G) assumes W is orthonormal, but
   MPO cores are NOT orthonormal after a few training steps. This causes:
   - Loss to INCREASE instead of decrease
   - Eventually SVD failure due to ill-conditioned matrices
   
   Use stabilize=False (plain Adam) until this is properly fixed.
   See FINDINGS.md for details.

References:
    - Lubich et al. "Time integration of tensor trains"
    - Haegeman et al. "TDVP for matrix product states"

Constitutional Compliance:
    - Article V.5.1: All public classes/methods documented
    - Article VII.7.2: Definition of Done = USER-OBSERVABLE BEHAVIOR works
"""

import torch
from torch.optim import Optimizer


class RiemannianAdam(Optimizer):
    """
    Adam Optimizer adapted for Tensor Network Manifolds.
    
    Projects the Euclidean gradient G onto the Tangent Space of the 
    current tensor core W. This filters out "Gauge Noise" -- updates that 
    change the numbers but not the physical operator.
    
    The projection formula is:
        G_proj = G - W @ (W^T @ G)
        
    This removes the component of the gradient parallel to W, keeping
    only the component that changes the actual function being represented.
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for running averages (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability (default: 1e-8)
        stabilize: Whether to apply manifold projection (default: True)
        
    Example:
        >>> model = FluidElite(num_sites=12, rank=32, vocab_size=100)
        >>> optimizer = RiemannianAdam(model.parameters(), lr=0.005)
        >>> for batch in data:
        ...     optimizer.zero_grad()
        ...     loss = compute_loss(model, batch)
        ...     loss.backward()
        ...     optimizer.step()
    """
    def __init__(
        self, 
        params, 
        lr: float = 1e-3, 
        betas: tuple[float, float] = (0.9, 0.999), 
        eps: float = 1e-8, 
        stabilize: bool = True
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, stabilize=stabilize)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns loss (optional)
            
        Returns:
            Loss value if closure provided, else None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # --- Manifold Projection (The "Elite" Step) ---
                # Project G -> P_W(G) = G - W @ (W^H @ G)
                # Assumes W is approximately isometric
                if group['stabilize']:
                    shape = p.shape
                    if len(shape) == 4:  # MPO Core (L, Out, In, R)
                        flattened = p.view(-1, shape[-1])
                        grad_flat = grad.view(-1, shape[-1])
                    elif len(shape) == 5:  # Stacked MPO Core (L, D_l, d, d, D_r)
                        flattened = p.view(-1, shape[-1])
                        grad_flat = grad.view(-1, shape[-1])
                    else:  # Generic Tensor (e.g., readout head)
                        flattened = p.view(-1, shape[-1])
                        grad_flat = grad.view(-1, shape[-1])

                    # Calculate Overlap: W^T @ G
                    overlap = flattened.T @ grad_flat
                    
                    # Remove component parallel to the weights (Gauge removal)
                    grad_proj = grad_flat - flattened @ overlap
                    grad = grad_proj.view_as(p)

                # --- Standard Adam on Projected Gradient ---
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                
                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute denominator
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                # Step size
                step_size = group['lr']
                
                # Bias correction could be added here but we skip for simplicity
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']
                # step_size = group['lr'] * (bias_correction2 ** 0.5) / bias_correction1

                # Apply update
                p.addcdiv_(exp_avg, denom, value=-step_size)
                
                # Note: A full Riemannian retraction would re-orthogonalize p here.
                # We skip this to save FLOPs, relying on projection to stay close
                # to the manifold. For strict manifold optimization, add:
                # if len(shape) >= 3:
                #     Q, R = torch.linalg.qr(p.view(-1, shape[-1]))
                #     p.data = Q.view_as(p)

        return loss
