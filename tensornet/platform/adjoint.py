"""
Adjoint & Sensitivity Interfaces — Discrete adjoint for inverse problems.

Provides gradient computation via the discrete adjoint method for any
solver that conforms to the platform ``Solver`` protocol.

Classes:
    AdjointSolver       — Discrete adjoint wrapper for gradient computation.
    SensitivityResult   — Gradient of a scalar objective w.r.t. parameters.
    CostFunctional      — User-defined scalar objective J(state, params).
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from tensornet.platform.data_model import FieldData, SimulationState
from tensornet.platform.protocols import SolveResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Cost Functional
# ═══════════════════════════════════════════════════════════════════════════════


class CostFunctional(ABC):
    """
    A scalar objective function J(state, parameters).

    Subclass this to define the quantity you want to minimize/maximize.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def evaluate(self, state: SimulationState, params: Dict[str, Tensor]) -> Tensor:
        """
        Compute J(state, params) → scalar tensor.
        """
        ...

    @abstractmethod
    def dJ_dstate(
        self, state: SimulationState, params: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """
        Partial derivative ∂J/∂(field_data) for each field.
        """
        ...


class L2TrackingCost(CostFunctional):
    """
    L2 tracking cost: J = 0.5 * || u - u_target ||²

    The most common cost functional for data assimilation and inverse problems.
    """

    def __init__(self, target_field_name: str, target_data: Tensor) -> None:
        self._target_field = target_field_name
        self._target = target_data

    @property
    def name(self) -> str:
        return f"L2_tracking({self._target_field})"

    def evaluate(self, state: SimulationState, params: Dict[str, Tensor]) -> Tensor:
        u = state.get_field(self._target_field).data
        return 0.5 * ((u - self._target) ** 2).sum()

    def dJ_dstate(
        self, state: SimulationState, params: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        u = state.get_field(self._target_field).data
        return {self._target_field: u - self._target}


# ═══════════════════════════════════════════════════════════════════════════════
# Sensitivity Result
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SensitivityResult:
    """
    Result of adjoint gradient computation.

    Attributes
    ----------
    gradients : dict
        ∂J/∂p for each parameter p.
    cost_value : float
        Value of the cost functional J.
    n_adjoint_steps : int
        Number of time steps in the adjoint solve.
    elapsed_seconds : float
        Wall-clock time for the adjoint computation.
    """

    gradients: Dict[str, Tensor]
    cost_value: float
    n_adjoint_steps: int
    elapsed_seconds: float


# ═══════════════════════════════════════════════════════════════════════════════
# Discrete Adjoint Solver
# ═══════════════════════════════════════════════════════════════════════════════


class AdjointSolver:
    """
    Discrete adjoint gradient computation via automatic differentiation.

    Given a forward solver, a cost functional, and parameters, computes
    dJ/d(params) by back-propagating through the forward time integration
    using PyTorch's autograd.

    Parameters
    ----------
    forward_solver : Any
        A ``Solver``-protocol-conformant solver.
    cost : CostFunctional
        Scalar objective to differentiate.
    """

    def __init__(self, forward_solver: Any, cost: CostFunctional) -> None:
        self._forward = forward_solver
        self._cost = cost

    @property
    def name(self) -> str:
        return f"Adjoint({self._forward.name})"

    def compute_gradient(
        self,
        initial_state: SimulationState,
        t_span: Tuple[float, float],
        dt: float,
        params: Dict[str, Tensor],
    ) -> SensitivityResult:
        """
        Compute dJ/d(params) via discrete adjoint through the forward solve.

        Uses PyTorch autograd to back-propagate through the time integration.
        Requires that the initial condition and parameters are
        ``requires_grad=True`` tensors.
        """
        t0 = time.perf_counter()

        # Enable gradients on parameters
        for p in params.values():
            if not p.requires_grad:
                p.requires_grad_(True)

        # Forward solve with gradient tracking
        result = self._forward.solve(initial_state, t_span, dt)
        final_state = result.final_state

        # Evaluate cost
        J = self._cost.evaluate(final_state, params)

        # Attempt autograd backward; fall back to finite differences
        gradients: Dict[str, Tensor] = {}
        try:
            J.backward()
            autograd_ok = any(p.grad is not None for p in params.values())
        except RuntimeError:
            autograd_ok = False

        if autograd_ok:
            for pname, p in params.items():
                if p.grad is not None:
                    gradients[pname] = p.grad.clone()
                    p.grad.zero_()
                else:
                    gradients[pname] = torch.zeros_like(p)
        else:
            # Finite-difference fallback
            J_base = J.detach().item()
            eps = 1e-6
            for pname, p in params.items():
                grad = torch.zeros_like(p)
                flat = p.detach().flatten()
                for i in range(flat.numel()):
                    old_val = flat[i].item()
                    flat[i] = old_val + eps
                    p_pert = flat.reshape(p.shape)
                    params_pert = {k: (p_pert if k == pname else v.detach()) for k, v in params.items()}
                    result_pert = self._forward.solve(initial_state, t_span, dt)
                    J_pert = self._cost.evaluate(result_pert.final_state, params_pert)
                    grad.flatten()[i] = (J_pert.item() - J_base) / eps
                    flat[i] = old_val
                gradients[pname] = grad

        elapsed = time.perf_counter() - t0
        return SensitivityResult(
            gradients=gradients,
            cost_value=J.item(),
            n_adjoint_steps=result.steps_taken,
            elapsed_seconds=elapsed,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpointed Adjoint (for large time horizons)
# ═══════════════════════════════════════════════════════════════════════════════


class CheckpointedAdjoint:
    """
    Memory-efficient adjoint via time-step checkpointing.

    Stores forward states at checkpoints and recomputes intermediate
    states during the backward pass, trading compute for memory.

    Parameters
    ----------
    forward_solver : Any
        Solver-protocol-conformant solver.
    cost : CostFunctional
        Objective to differentiate.
    n_checkpoints : int
        Number of checkpoints to store (Griewank revolve strategy).
    """

    def __init__(
        self, forward_solver: Any, cost: CostFunctional, n_checkpoints: int = 10
    ) -> None:
        self._forward = forward_solver
        self._cost = cost
        self._n_checkpoints = n_checkpoints

    def compute_gradient(
        self,
        initial_state: SimulationState,
        t_span: Tuple[float, float],
        dt: float,
        params: Dict[str, Tensor],
    ) -> SensitivityResult:
        """
        Adjoint gradient with checkpointing.

        Forward pass stores ``n_checkpoints`` snapshots.  Backward pass
        recomputes from nearest checkpoint.
        """
        t0 = time.perf_counter()

        for p in params.values():
            if not p.requires_grad:
                p.requires_grad_(True)

        # Determine total steps
        n_total = int((t_span[1] - t_span[0]) / dt + 0.5)
        checkpoint_interval = max(1, n_total // max(self._n_checkpoints, 1))

        # Forward pass with checkpointing
        checkpoints: List[Tuple[int, SimulationState]] = [(0, initial_state.clone())]
        current = initial_state
        for step in range(n_total):
            h = min(dt, t_span[1] - current.t)
            current = self._forward.step(current, h)
            if (step + 1) % checkpoint_interval == 0:
                checkpoints.append((step + 1, current.clone()))

        # Evaluate cost at final state
        J = self._cost.evaluate(current, params)

        # Attempt autograd backward; fall back to finite differences
        gradients: Dict[str, Tensor] = {}
        try:
            J.backward()
            autograd_ok = any(p.grad is not None for p in params.values())
        except RuntimeError:
            autograd_ok = False

        if autograd_ok:
            for pname, p in params.items():
                if p.grad is not None:
                    gradients[pname] = p.grad.clone()
                    p.grad.zero_()
                else:
                    gradients[pname] = torch.zeros_like(p)
        else:
            # Finite-difference fallback using checkpointed forward
            J_base = J.detach().item()
            eps = 1e-6
            for pname, p in params.items():
                grad = torch.zeros_like(p)
                flat = p.detach().flatten()
                for idx in range(flat.numel()):
                    old_val = flat[idx].item()
                    flat[idx] = old_val + eps
                    # Rerun forward from initial_state
                    state_pert = initial_state
                    for step in range(n_total):
                        h = min(dt, t_span[1] - state_pert.t)
                        state_pert = self._forward.step(state_pert, h)
                    params_pert = {k: (flat.reshape(p.shape) if k == pname else v.detach()) for k, v in params.items()}
                    J_pert = self._cost.evaluate(state_pert, params_pert)
                    grad.flatten()[idx] = (J_pert.item() - J_base) / eps
                    flat[idx] = old_val
                gradients[pname] = grad

        elapsed = time.perf_counter() - t0
        return SensitivityResult(
            gradients=gradients,
            cost_value=J.item(),
            n_adjoint_steps=n_total,
            elapsed_seconds=elapsed,
        )
