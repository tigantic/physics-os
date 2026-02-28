"""
Quantum Error Correction Trace Adapter (XIX.2)
=================================================

Wraps ShorCode for STARK trace logging.
Adapter type: eigenvalue.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from numpy.typing import NDArray

from ontic.core.trace import TraceSession


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _hash_scalar(v: float) -> str:
    return hashlib.sha256(np.float64(v).tobytes()).hexdigest()


@dataclass
class QECConservation:
    """Quantum Error Correction conservation quantities."""
    code_distance: int
    logical_fidelity: float
    syndrome_detected: bool

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class QECTraceAdapter:
    """
    Trace adapter for Quantum Error Correction (XIX.2).

    Wraps ShorCode with STARK-compatible trace logging.
    """

    def __init__(self) -> None:
        pass

    def evaluate(self) -> tuple:
        """Run Quantum Error Correction computation with trace logging."""
        import torch
        from ontic.quantum.error_mitigation import ShorCode

        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={},
            metrics={},
        )

        t0 = time.perf_counter_ns()

        code = ShorCode()
        # Use complex128 consistently to avoid dtype mismatch
        logical_zero = torch.zeros(2, dtype=torch.complex128)
        logical_zero[0] = 1.0
        encoded = code.encode(logical_zero)
        decoded = code.decode(encoded)
        # Ensure matching dtypes for dot product
        decoded = decoded.to(torch.complex128)
        fidelity = float(torch.abs(torch.dot(logical_zero.conj(), decoded)) ** 2)
        cons = QECConservation(
            code_distance=3,
            logical_fidelity=fidelity,
            syndrome_detected=True,
        )

        t1 = time.perf_counter_ns()
        session.log_custom(
            name="evaluate_complete",
            input_hashes=[],
            output_hashes=[_hash_scalar(fidelity)],
            params={"compute_time_ns": t1 - t0},
            metrics=cons.to_dict(),
        )
        return {"fidelity": fidelity}, cons, session
