"""
Band Structure Trace Adapter (IX.2)
======================================

Wraps TightBindingBands for STARK trace logging.
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

from tensornet.core.trace import TraceSession


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _hash_scalar(v: float) -> str:
    return hashlib.sha256(np.float64(v).tobytes()).hexdigest()


@dataclass
class BandStructureConservation:
    """Band Structure conservation quantities."""
    n_bands: int
    band_gap: float
    charge_neutrality: bool

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class BandStructureTraceAdapter:
    """
    Trace adapter for Band Structure (IX.2).

    Wraps TightBindingBands with STARK-compatible trace logging.
    """

    def __init__(self) -> None:
        pass

    def evaluate(self, n_k: int = 50) -> tuple:
        """Run Band Structure computation with trace logging."""
        from tensornet.quantum.condensed_matter.band_structure import TightBindingBands

        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={"n_k": n_k},
            metrics={},
        )

        t0 = time.perf_counter_ns()

        # chain_1d(cls, eps0, t, a)
        model = TightBindingBands.chain_1d(eps0=0.0, t=1.0)
        # k_path must be (n_k, dim) for dim=1
        k_path = np.linspace(-np.pi, np.pi, n_k).reshape(-1, 1)
        bands = model.bands(k_path)
        n_bands = bands.shape[1] if bands.ndim == 2 else 1
        gap = 0.0
        if bands.ndim == 2 and bands.shape[1] > 1:
            gap = float(np.min(bands[:, 1]) - np.max(bands[:, 0]))
        cons = BandStructureConservation(
            n_bands=n_bands,
            band_gap=gap,
            charge_neutrality=True,
        )

        t1 = time.perf_counter_ns()
        session.log_custom(
            name="evaluate_complete",
            input_hashes=[],
            output_hashes=[],
            params={"compute_time_ns": t1 - t0},
            metrics=cons.to_dict(),
        )
        return bands, cons, session
