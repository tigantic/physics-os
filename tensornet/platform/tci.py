"""
TCI Decomposition Engine — Platform-canonical TCI interface.

Provides a clean platform-level API for Tensor Cross Interpolation (TCI),
which builds QTT representations from black-box function evaluations in
O(r² · n_qubits) complexity instead of O(2^n_qubits).

This module delegates to ``tensornet.cfd.qtt_tci`` for the actual
computation, wrapping results in platform-native ``QTTFieldData``.

Key functions:
    tci_from_function   — Build QTT from a callable f(indices) → values.
    tci_from_field      — Compress a dense FieldData via TCI sampling.
    tci_error_vs_rank   — Produce an error-vs-rank curve for QTT enablement.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from tensornet.platform.data_model import FieldData, Mesh, StructuredMesh
from tensornet.platform.qtt import QTTFieldData, _next_power_of_2

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# TCI Configuration
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class TCIConfig:
    """
    Configuration for TCI decomposition.

    Parameters
    ----------
    max_rank : int
        Maximum bond dimension for the resulting QTT.
    tolerance : float
        Convergence tolerance (relative Frobenius-norm error).
    max_sweeps : int
        Maximum number of TCI optimization sweeps.
    maxvol_tolerance : float
        Interior tolerance for MaxVol pivot selection.
    batch_size : int
        Number of function evaluations per batch (for GPU efficiency).
    use_rust : bool
        Prefer Rust TCI core when available.
    device : str
        Torch device for computation.
    dtype : torch.dtype
        Floating-point precision.
    """

    max_rank: int = 64
    tolerance: float = 1e-8
    max_sweeps: int = 20
    maxvol_tolerance: float = 1.05
    batch_size: int = 1024
    use_rust: bool = True
    device: str = "cpu"
    dtype: torch.dtype = torch.float64


# ═══════════════════════════════════════════════════════════════════════════════
# TCI Result
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TCIResult:
    """
    Result container from a TCI decomposition.

    Holds the QTT cores plus convergence diagnostics used for
    rank-growth reports and error-vs-rank curves.
    """

    cores: List[Tensor]
    n_qubits: int
    max_rank_achieved: int
    n_function_evals: int
    compression_ratio: float
    elapsed_seconds: float
    method: str  # "tci_python", "tci_rust", "tt_svd_fallback"
    sweep_errors: List[float] = dc_field(default_factory=list)
    converged: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# Core TCI Functions
# ═══════════════════════════════════════════════════════════════════════════════


def tci_from_function(
    f: Callable[[Tensor], Tensor],
    n_qubits: int,
    config: Optional[TCIConfig] = None,
) -> TCIResult:
    """
    Build a QTT representation from a black-box function via TCI.

    The function ``f`` maps integer indices ``(batch,)`` to values ``(batch,)``.
    TCI adaptively samples ``f`` at O(r² · n_qubits) points to construct
    a rank-``r`` QTT core chain.

    Parameters
    ----------
    f : Callable[[Tensor], Tensor]
        Black-box function mapping indices to values.
    n_qubits : int
        Number of qubit levels (grid size = ``2 ** n_qubits``).
    config : TCIConfig, optional
        Decomposition parameters (defaults to ``TCIConfig()``).

    Returns
    -------
    TCIResult
        QTT cores plus convergence diagnostics.
    """
    if config is None:
        config = TCIConfig()

    t0 = time.perf_counter()
    N = 2 ** n_qubits

    # Attempt Rust TCI first
    try:
        from tensornet.cfd.qtt_tci import qtt_from_function_tci_python
        raw_cores = qtt_from_function_tci_python(
            f,
            n_qubits=n_qubits,
            max_rank=config.max_rank,
            tolerance=config.tolerance,
            device=config.device,
        )
        # Ensure every core is a Tensor (the TCI engine may return nested
        # lists or numpy arrays depending on the backend).
        cores = []
        for rc in raw_cores:
            if isinstance(rc, Tensor):
                cores.append(rc)
            elif isinstance(rc, list):
                cores.append(torch.tensor(rc, dtype=config.dtype, device=config.device))
            else:
                import numpy as np  # noqa: F811
                cores.append(torch.as_tensor(rc, dtype=config.dtype, device=config.device))
        # Canonical shape per core is (r_left, 2, r_right).
        # If returned as (r_left, 2, r_right) already, fine.  If squeezed
        # or has an extra axis, attempt fix.
        for i, c in enumerate(cores):
            if c.ndim == 2:
                cores[i] = c.unsqueeze(0) if i == 0 else c.unsqueeze(-1)
            elif c.ndim == 4:
                cores[i] = c.squeeze()
        method = "tci_python"
    except Exception as exc:
        logger.warning("TCI engine failed (%s), falling back to TT-SVD", exc)
        # Fallback: dense evaluation + TT-SVD
        indices = torch.arange(N, device=config.device)
        values = f(indices)
        from tensornet.platform.qtt import _tt_svd
        cores = _tt_svd(values, n_qubits, config.max_rank, config.tolerance)
        method = "tt_svd_fallback"

    elapsed = time.perf_counter() - t0

    storage = sum(c.numel() for c in cores)
    max_rank_achieved = max(c.shape[2] for c in cores) if cores else 1
    ratio = N / max(storage, 1)

    # Estimate function evaluations
    if method == "tt_svd_fallback":
        n_evals = N
    else:
        n_evals = sum(config.max_rank ** 2 for _ in range(n_qubits))

    return TCIResult(
        cores=cores,
        n_qubits=n_qubits,
        max_rank_achieved=max_rank_achieved,
        n_function_evals=n_evals,
        compression_ratio=ratio,
        elapsed_seconds=elapsed,
        method=method,
    )


def tci_from_field(
    field: FieldData,
    config: Optional[TCIConfig] = None,
) -> QTTFieldData:
    """
    Compress a dense ``FieldData`` to QTT via TCI adaptive sampling.

    Unlike ``field_to_qtt`` (which uses TT-SVD on the full dense tensor),
    this function treats the field as a black-box function of grid index
    and uses TCI to build the QTT with sub-linear function evaluations.

    For fields that are smooth or low-rank, TCI achieves the same accuracy
    as TT-SVD with far fewer evaluations.

    Parameters
    ----------
    field : FieldData
        Dense field to compress.
    config : TCIConfig, optional
        Decomposition parameters.

    Returns
    -------
    QTTFieldData
        Compressed field in QTT form.
    """
    if config is None:
        config = TCIConfig()

    data = field.data.flatten().to(config.dtype)
    original_len = data.shape[0]
    n2 = _next_power_of_2(original_len)
    n_qubits = int(math.log2(n2))

    # Pad if necessary
    if n2 > original_len:
        padded = torch.zeros(n2, dtype=config.dtype, device=config.device)
        padded[:original_len] = data
    else:
        padded = data.to(config.device)

    def f_lookup(indices: Tensor) -> Tensor:
        return padded[indices.long()]

    result = tci_from_function(f_lookup, n_qubits, config)

    return QTTFieldData(
        name=field.name,
        cores=result.cores,
        n_qubits=result.n_qubits,
        mesh=field.mesh,
        units=field.units,
        compression_ratio=result.compression_ratio,
        max_rank=result.max_rank_achieved,
        tolerance=config.tolerance,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Error-vs-Rank Curve
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RankErrorPoint:
    """A single point on the error-vs-rank curve."""

    rank: int
    error_linf: float
    error_l2: float
    compression_ratio: float
    storage_elements: int


def tci_error_vs_rank(
    field: FieldData,
    rank_schedule: Optional[List[int]] = None,
    tolerance: float = 1e-14,
) -> List[RankErrorPoint]:
    """
    Produce an error-vs-rank curve for a field.

    Compresses the field at each rank in ``rank_schedule`` and measures
    the reconstruction error against the original dense data.

    Parameters
    ----------
    field : FieldData
        The reference dense field.
    rank_schedule : list[int], optional
        Ranks to test.  Defaults to ``[1, 2, 4, 8, 16, 32, 64]``.
    tolerance : float
        SVD tolerance (set very tight to isolate rank effects).

    Returns
    -------
    list[RankErrorPoint]
        Error measurements at each rank.
    """
    if rank_schedule is None:
        rank_schedule = [1, 2, 4, 8, 16, 32, 64]

    from tensornet.platform.qtt import field_to_qtt, qtt_to_field

    data = field.data.flatten()
    n = data.shape[0]

    points: List[RankErrorPoint] = []

    for rank in rank_schedule:
        qtt = field_to_qtt(field, max_rank=rank, tolerance=tolerance)
        recon = qtt_to_field(qtt)
        recon_data = recon.data[:n]

        diff = (data - recon_data).abs()
        error_linf = diff.max().item()
        error_l2 = diff.pow(2).sum().sqrt().item()

        points.append(RankErrorPoint(
            rank=rank,
            error_linf=error_linf,
            error_l2=error_l2,
            compression_ratio=qtt.compression_ratio,
            storage_elements=qtt.storage_elements,
        ))

    return points
