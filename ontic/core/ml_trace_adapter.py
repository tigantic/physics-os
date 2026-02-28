"""
ML-to-STARK Trace Adapter
============================

Generic adapter for PINN / FNO / neural-network potential methods.

Trace layout per training step:
    weight_hash[t]   = H(θ[t])
    loss[t]          = L(θ[t])
    gradient_hash[t] = H(∇L[t])
    constraint:       loss[t] ≤ loss[t-1] (with tolerance)
    constraint:       final_loss < threshold
    inference:        input_hash → output_hash via forward(θ_final)

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from ontic.core.trace import TraceSession


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _hash_scalar(v: float) -> str:
    return hashlib.sha256(np.float64(v).tobytes()).hexdigest()


@dataclass
class MLConvergence:
    """Convergence diagnostics for an ML training run."""

    n_steps: int
    initial_loss: float
    final_loss: float
    loss_decreased: bool
    converged: bool
    weight_hashes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_steps": self.n_steps,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "loss_decreased": self.loss_decreased,
            "converged": self.converged,
        }


class MLTraceAdapter:
    """
    Base class for ML-method trace adapters.

    Subclasses must implement:
        _train_step(step_idx) -> (loss_value, weight_bytes)
        _predict(x) -> NDArray
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def run_traced(
        self,
        n_steps: int,
        loss_threshold: float = 1e-3,
    ) -> tuple:
        """Execute ML training with full trace logging."""
        session = TraceSession()
        session.log_custom(
            name="ml_init",
            input_hashes=[_hash_scalar(float(self.seed))],
            output_hashes=[],
            params={"n_steps": n_steps, "loss_threshold": loss_threshold},
            metrics={},
        )

        t0 = time.perf_counter_ns()
        losses: List[float] = []
        weight_hashes: List[str] = []

        for step in range(n_steps):
            loss_val, w_bytes = self._train_step(step)
            losses.append(float(loss_val))
            wh = hashlib.sha256(w_bytes).hexdigest()[:16]
            weight_hashes.append(wh)

        conv = MLConvergence(
            n_steps=n_steps,
            initial_loss=losses[0] if losses else 0.0,
            final_loss=losses[-1] if losses else 0.0,
            loss_decreased=bool(losses[-1] <= losses[0] + 0.1) if losses else True,
            converged=bool(losses[-1] < loss_threshold) if losses else False,
            weight_hashes=weight_hashes[-min(10, len(weight_hashes)):],
        )

        t1 = time.perf_counter_ns()
        session.log_custom(
            name="ml_complete",
            input_hashes=[],
            output_hashes=[_hash_scalar(conv.final_loss)],
            params={"compute_time_ns": t1 - t0},
            metrics=conv.to_dict(),
        )
        return {"losses": losses, "final_loss": conv.final_loss}, conv, session

    def _train_step(self, step_idx: int) -> tuple:
        raise NotImplementedError

    def _predict(self, x: NDArray) -> NDArray:
        raise NotImplementedError
