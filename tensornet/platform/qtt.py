"""
QTT Bridging Layer — Field-to-QTT mappings and QTT-native platform types.

This module bridges the existing QTT primitives in ``tensornet.cfd``
with the platform data model, making QTT a first-class field representation.

Key types:
    QTTFieldData    — A named field stored in QTT (compressed) form.
    QTTOperator     — An MPO-wrapped operator satisfying ``OperatorProto``.
    QTTDiscretization — A ``Discretization`` that produces QTT operators.

Key functions:
    field_to_qtt    — Convert dense ``FieldData`` to ``QTTFieldData``.
    qtt_to_field    — Reconstruct dense ``FieldData`` from ``QTTFieldData``.

Design: Platform FieldData uses ``torch.Tensor``.  This module adds a
parallel representation in QTT cores, with lossless round-trip guarantees
up to the user-specified tolerance.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from tensornet.platform.data_model import FieldData, Mesh, StructuredMesh

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# QTT Field Representation
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class QTTFieldData:
    """
    A named field stored as QTT (Quantized Tensor-Train) cores.

    This is the QTT analog of ``FieldData``.  The ``cores`` list holds
    tensors of shape ``(r_left, 2, r_right)`` — one per qubit level.

    Attributes
    ----------
    name : str
        Field name (matches the corresponding dense field).
    cores : list[Tensor]
        QTT cores, each shape ``(r_left, 2, r_right)``.
    n_qubits : int
        Number of qubit levels.  Grid size = ``2 ** n_qubits``.
    mesh : Mesh
        The mesh this field lives on (for reconstruction).
    units : str
        SI-compatible unit string.
    compression_ratio : float
        Ratio of dense size to QTT storage.
    max_rank : int
        Maximum bond dimension across all cores.
    tolerance : float
        Truncation tolerance used during compression.
    """

    name: str
    cores: List[Tensor]
    n_qubits: int
    mesh: Mesh
    units: str = "1"
    compression_ratio: float = 1.0
    max_rank: int = 1
    tolerance: float = 1e-10

    @property
    def grid_size(self) -> int:
        """Number of grid points representable by this QTT."""
        return 2 ** self.n_qubits

    @property
    def ranks(self) -> List[int]:
        """Bond dimensions between adjacent cores."""
        return [c.shape[2] for c in self.cores[:-1]]

    @property
    def storage_elements(self) -> int:
        """Total number of stored floating-point values."""
        return sum(c.numel() for c in self.cores)

    @property
    def dense_elements(self) -> int:
        """Number of elements in the uncompressed representation."""
        return self.grid_size

    def clone(self) -> "QTTFieldData":
        """Return a deep copy."""
        return QTTFieldData(
            name=self.name,
            cores=[c.clone() for c in self.cores],
            n_qubits=self.n_qubits,
            mesh=self.mesh,
            units=self.units,
            compression_ratio=self.compression_ratio,
            max_rank=self.max_rank,
            tolerance=self.tolerance,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Dense ↔ QTT Conversions
# ═══════════════════════════════════════════════════════════════════════════════


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _pad_to_power_of_2(data: Tensor) -> Tuple[Tensor, int]:
    """Pad a 1-D tensor to the next power of 2, return (padded, original_len)."""
    n = data.shape[0]
    n2 = _next_power_of_2(n)
    if n2 == n:
        return data, n
    padded = torch.zeros(n2, dtype=data.dtype, device=data.device)
    padded[:n] = data
    return padded, n


def _tt_svd(
    data: Tensor,
    n_qubits: int,
    max_rank: int,
    tolerance: float,
) -> List[Tensor]:
    """
    TT-SVD decomposition of a 1-D tensor into QTT cores.

    Reshapes the length-``2^n_qubits`` vector into a chain of
    ``(r_left, 2, r_right)`` cores via sequential SVD truncation.
    """
    N = 2 ** n_qubits
    if data.numel() != N:
        raise ValueError(f"Data has {data.numel()} elements, expected {N}")

    remaining = data.reshape(1, N).to(torch.float64)
    cores: List[Tensor] = []
    r_left = 1

    for k in range(n_qubits):
        n_right = remaining.numel() // (r_left * 2)
        mat = remaining.reshape(r_left * 2, n_right)

        if k < n_qubits - 1:
            U, S, Vh = torch.linalg.svd(mat, full_matrices=False)

            # Truncate by tolerance
            cumulative = torch.cumsum(S ** 2, dim=0)
            total = cumulative[-1]
            if total > 0:
                keep = int(torch.searchsorted(cumulative, total * (1 - tolerance ** 2)).item()) + 1
            else:
                keep = 1
            keep = min(keep, max_rank, S.shape[0])

            U = U[:, :keep]
            S = S[:keep]
            Vh = Vh[:keep, :]

            core = U.reshape(r_left, 2, keep)
            remaining = torch.diag(S) @ Vh
            r_left = keep
        else:
            core = mat.reshape(r_left, 2, 1)

        cores.append(core)

    return cores


def field_to_qtt(
    field: FieldData,
    max_rank: int = 64,
    tolerance: float = 1e-10,
) -> QTTFieldData:
    """
    Compress a dense ``FieldData`` into QTT form via TT-SVD.

    Parameters
    ----------
    field : FieldData
        Dense field on a structured mesh.
    max_rank : int
        Maximum bond dimension.
    tolerance : float
        SVD truncation tolerance (relative).

    Returns
    -------
    QTTFieldData
        Compressed field with QTT cores.

    Raises
    ------
    ValueError
        If the mesh is not structured or data length is not compatible.
    """
    data = field.data.flatten()
    original_len = data.shape[0]
    padded, _ = _pad_to_power_of_2(data)
    n_qubits = int(math.log2(padded.shape[0]))

    t0 = time.perf_counter()
    cores = _tt_svd(padded, n_qubits, max_rank, tolerance)
    elapsed = time.perf_counter() - t0

    storage = sum(c.numel() for c in cores)
    ratio = original_len / max(storage, 1)
    actual_max_rank = max(c.shape[2] for c in cores) if cores else 1

    logger.debug(
        "field_to_qtt '%s': %d → %d elements (%.1fx), max_rank=%d, %.3fs",
        field.name, original_len, storage, ratio, actual_max_rank, elapsed,
    )

    return QTTFieldData(
        name=field.name,
        cores=cores,
        n_qubits=n_qubits,
        mesh=field.mesh,
        units=field.units,
        compression_ratio=ratio,
        max_rank=actual_max_rank,
        tolerance=tolerance,
    )


def qtt_to_field(qtt_field: QTTFieldData, n_points: Optional[int] = None) -> FieldData:
    """
    Reconstruct a dense ``FieldData`` from QTT cores.

    Parameters
    ----------
    qtt_field : QTTFieldData
        QTT-compressed field.
    n_points : int, optional
        If given, truncate the reconstructed tensor to this length
        (handles non-power-of-2 original grids).

    Returns
    -------
    FieldData
        Reconstructed dense field.
    """
    # Contract cores left-to-right
    result = qtt_field.cores[0]  # (1, 2, r1)
    for core in qtt_field.cores[1:]:
        # result: (1, 2^k, r_k), core: (r_k, 2, r_{k+1})
        r_left = result.shape[0]
        n_accumulated = result.shape[1]
        r_mid = core.shape[0]
        r_right = core.shape[2]

        # Reshape for contraction
        result = result.reshape(r_left * n_accumulated, r_mid)
        core_mat = core.reshape(r_mid, 2 * r_right)
        result = result @ core_mat
        result = result.reshape(r_left, n_accumulated * 2, r_right)

    data = result.squeeze(0).squeeze(-1)

    if n_points is not None:
        data = data[:n_points]
    elif isinstance(qtt_field.mesh, StructuredMesh):
        data = data[:qtt_field.mesh.n_cells]

    return FieldData(
        name=qtt_field.name,
        data=data,
        mesh=qtt_field.mesh,
        units=qtt_field.units,
    )


def qtt_roundtrip_error(field: FieldData, max_rank: int = 64, tolerance: float = 1e-10) -> float:
    """
    Compute the L-infinity round-trip error: field → QTT → field.

    Returns the maximum absolute difference.
    """
    qtt = field_to_qtt(field, max_rank=max_rank, tolerance=tolerance)
    reconstructed = qtt_to_field(qtt)
    n = min(field.data.shape[0], reconstructed.data.shape[0])
    return (field.data[:n] - reconstructed.data[:n]).abs().max().item()


# ═══════════════════════════════════════════════════════════════════════════════
# QTT Operator (MPO wrapper satisfying OperatorProto)
# ═══════════════════════════════════════════════════════════════════════════════


class QTTOperator:
    """
    An MPO-based operator that acts on QTT fields.

    Satisfies the ``OperatorProto`` protocol.  When ``apply`` is called
    with a dense tensor, it:
    1. Compresses to QTT.
    2. Applies the MPO in compressed form.
    3. Reconstructs to dense.

    When called on ``QTTFieldData`` directly (via ``apply_qtt``), the
    entire computation stays in compressed form.

    Parameters
    ----------
    name : str
        Operator name.
    mpo_cores : list[Tensor]
        MPO cores, each shape ``(r_left, d_out, d_in, r_right)``.
    max_rank : int
        Maximum output rank after MPO application + truncation.
    tolerance : float
        Truncation tolerance for post-application rounding.
    """

    def __init__(
        self,
        name: str,
        mpo_cores: List[Tensor],
        max_rank: int = 64,
        tolerance: float = 1e-10,
    ) -> None:
        self._name = name
        self._mpo_cores = mpo_cores
        self._max_rank = max_rank
        self._tolerance = tolerance

    @property
    def name(self) -> str:
        return self._name

    def apply(self, field: Tensor, **kwargs: Any) -> Tensor:
        """
        Apply the MPO operator to a dense tensor.

        Compresses → applies MPO → reconstructs.  For high-dimensional
        fields, prefer ``apply_qtt`` to avoid dense round-trips.
        """
        n = field.shape[0]
        n2 = _next_power_of_2(n)
        n_qubits = int(math.log2(n2))

        padded, _ = _pad_to_power_of_2(field.flatten())
        cores = _tt_svd(padded, n_qubits, self._max_rank, self._tolerance)
        result_cores = self._apply_mpo_to_cores(cores)
        result_cores = _truncate_cores(result_cores, self._max_rank, self._tolerance)

        # Reconstruct
        result = result_cores[0]
        for core in result_cores[1:]:
            r_left = result.shape[0]
            n_accumulated = result.shape[1]
            r_mid = core.shape[0]
            r_right = core.shape[2]
            result = result.reshape(r_left * n_accumulated, r_mid)
            core_mat = core.reshape(r_mid, 2 * r_right)
            result = result @ core_mat
            result = result.reshape(r_left, n_accumulated * 2, r_right)

        return result.squeeze(0).squeeze(-1)[:n]

    def apply_qtt(self, qtt_field: QTTFieldData) -> QTTFieldData:
        """
        Apply the MPO to a QTT field, staying in compressed form.

        Returns a new QTTFieldData with the operator applied.
        """
        result_cores = self._apply_mpo_to_cores(qtt_field.cores)
        result_cores = _truncate_cores(result_cores, self._max_rank, self._tolerance)

        storage = sum(c.numel() for c in result_cores)
        ratio = qtt_field.grid_size / max(storage, 1)
        actual_max_rank = max(c.shape[2] for c in result_cores) if result_cores else 1

        return QTTFieldData(
            name=qtt_field.name,
            cores=result_cores,
            n_qubits=qtt_field.n_qubits,
            mesh=qtt_field.mesh,
            units=qtt_field.units,
            compression_ratio=ratio,
            max_rank=actual_max_rank,
            tolerance=self._tolerance,
        )

    def _apply_mpo_to_cores(self, mps_cores: List[Tensor]) -> List[Tensor]:
        """
        Contract MPO cores with MPS cores site by site.

        MPS core shape: (r_left_mps, 2, r_right_mps)
        MPO core shape: (r_left_mpo, 2, 2, r_right_mpo)
        Result core:    (r_left_mps * r_left_mpo, 2, r_right_mps * r_right_mpo)
        """
        n_sites = min(len(mps_cores), len(self._mpo_cores))
        result: List[Tensor] = []

        for k in range(n_sites):
            mps_c = mps_cores[k]    # (rL_m, 2, rR_m)
            mpo_c = self._mpo_cores[k]  # (rL_o, 2, 2, rR_o)

            rL_m, _, rR_m = mps_c.shape
            rL_o, d_out, d_in, rR_o = mpo_c.shape

            # Einsum: contract over physical input index σ_in
            # mps: (rLm=a, σ_in=s, rRm=b), mpo: (rLo=c, σ_out=p, σ_in=s, rRo=d)
            # result: (a, c, p, b, d) → reshape to (a*c, p, b*d)
            out = torch.einsum("asb,cpsd->acpbd", mps_c.to(torch.float64), mpo_c.to(torch.float64))
            out = out.reshape(rL_m * rL_o, d_out, rR_m * rR_o)
            result.append(out)

        return result


def _truncate_cores(
    cores: List[Tensor],
    max_rank: int,
    tolerance: float,
) -> List[Tensor]:
    """
    SVD-based rank truncation sweep (left-to-right canonical form).

    Rounds a QTT with inflated ranks back to controlled bond dimensions.
    """
    n = len(cores)
    if n <= 1:
        return cores

    new_cores: List[Tensor] = []
    current = cores[0]

    for k in range(n - 1):
        r_left, d, r_right = current.shape
        mat = current.reshape(r_left * d, r_right)

        U, S, Vh = torch.linalg.svd(mat, full_matrices=False)

        # Truncate
        cumulative = torch.cumsum(S ** 2, dim=0)
        total = cumulative[-1]
        if total > 0:
            keep = int(torch.searchsorted(cumulative, total * (1 - tolerance ** 2)).item()) + 1
        else:
            keep = 1
        keep = min(keep, max_rank, S.shape[0])

        U = U[:, :keep]
        S = S[:keep]
        Vh = Vh[:keep, :]

        new_cores.append(U.reshape(r_left, d, keep))

        # Absorb S·Vh into the next core
        SVh = torch.diag(S) @ Vh
        next_core = cores[k + 1]
        r_left_next, d_next, r_right_next = next_core.shape
        next_mat = SVh @ next_core.reshape(r_right, d_next * r_right_next)
        current = next_mat.reshape(keep, d_next, r_right_next)

    new_cores.append(current)
    return new_cores


# ═══════════════════════════════════════════════════════════════════════════════
# QTT Discretization (satisfies Discretization protocol)
# ═══════════════════════════════════════════════════════════════════════════════


class QTTDiscretization:
    """
    A discretization that produces QTT-native operators.

    Satisfies the ``Discretization`` protocol.  Given a ``ProblemSpec``
    and a ``StructuredMesh``, produces MPO-based differential operators
    (Laplacian, gradient, advection) in QTT form.

    Parameters
    ----------
    max_rank : int
        Maximum rank for operator MPOs and field compression.
    tolerance : float
        Truncation tolerance.
    order : int
        Formal accuracy order of the finite-difference stencil encoded
        in the MPO (default: 2, i.e. central differences).
    """

    def __init__(
        self,
        max_rank: int = 64,
        tolerance: float = 1e-10,
        order: int = 2,
    ) -> None:
        self._max_rank = max_rank
        self._tolerance = tolerance
        self._order = order

    @property
    def method(self) -> str:
        return "QTT"

    @property
    def order(self) -> int:
        return self._order

    def discretize(self, spec: Any, mesh: Any) -> Dict[str, QTTOperator]:
        """
        Build QTT-native differential operators for the given problem.

        Returns a dict of named ``QTTOperator`` instances:
        - ``'laplacian'`` — second-order central-difference Laplacian MPO
        - ``'shift_left'`` — left shift MPO
        - ``'shift_right'`` — right shift MPO

        These are constructed analytically (not from dense matrices).
        """
        if not isinstance(mesh, StructuredMesh):
            raise TypeError(
                f"QTTDiscretization requires StructuredMesh, got {type(mesh).__name__}"
            )

        n_cells = mesh.shape[0] if mesh.ndim == 1 else mesh.n_cells
        n_qubits = int(math.ceil(math.log2(max(n_cells, 2))))
        dx = mesh.dx[0]

        operators: Dict[str, QTTOperator] = {}

        # Build shift MPOs analytically
        shift_l_cores = _build_shift_mpo(n_qubits, direction="left")
        shift_r_cores = _build_shift_mpo(n_qubits, direction="right")

        operators["shift_left"] = QTTOperator(
            "shift_left", shift_l_cores, self._max_rank, self._tolerance,
        )
        operators["shift_right"] = QTTOperator(
            "shift_right", shift_r_cores, self._max_rank, self._tolerance,
        )

        # Laplacian = (S_L + S_R - 2·I) / dx²
        # Built from shift MPOs and identity
        laplacian_cores = _build_laplacian_mpo(n_qubits, dx)
        operators["laplacian"] = QTTOperator(
            "laplacian", laplacian_cores, self._max_rank, self._tolerance,
        )

        return operators


def _build_shift_mpo(n_qubits: int, direction: str = "right") -> List[Tensor]:
    """
    Build a single-site shift MPO for a 1-D QTT grid.

    The shift operator maps index i → i+1 (right) or i → i-1 (left)
    in binary, which is a carry-propagation chain at the qubit level.

    Each MPO core has shape ``(r_left, 2, 2, r_right)`` with bond
    dimension 2 (carry bit).
    """
    cores: List[Tensor] = []

    for k in range(n_qubits):
        # MPO core with 2 bond states: (no-carry, carry)
        core = torch.zeros(2, 2, 2, 2, dtype=torch.float64)

        if direction == "right":
            # Increment: propagate carry from LSB to MSB
            # State 0 (no carry): identity — out=in
            core[0, 0, 0, 0] = 1.0  # 0→0, no carry out
            core[0, 1, 1, 0] = 1.0  # 1→1, no carry out
            # State 1 (carry in): add 1
            core[1, 1, 0, 0] = 1.0  # 0+1=1, no carry out
            core[1, 0, 1, 1] = 1.0  # 1+1=0, carry out
        else:
            # Decrement: borrow propagation
            core[0, 0, 0, 0] = 1.0
            core[0, 1, 1, 0] = 1.0
            core[1, 0, 1, 0] = 1.0  # 1-1=0, no borrow out
            core[1, 1, 0, 1] = 1.0  # 0-1=1, borrow out

        if k == 0:
            # First core: inject carry for shift
            core = core[1:2, :, :, :]  # (1, 2, 2, 2) — start with carry
        elif k == n_qubits - 1:
            # Last core: drop carry (periodic or clamp)
            core = core[:, :, :, 0:1]  # (2, 2, 2, 1) — discard carry
        cores.append(core)

    return cores


def _build_laplacian_mpo(n_qubits: int, dx: float) -> List[Tensor]:
    """
    Build a 1-D discrete Laplacian MPO: (u[i+1] + u[i-1] - 2·u[i]) / dx².

    Constructed as a sum of shift MPOs and identity, resulting in
    bond dimension 4 (2 for each shift + overlap).

    For production use at scale, prefer ``tensornet.cfd.pure_qtt_ops.laplacian_mpo``
    which uses the fused MPO construction from Kazeev & Schwab (2015).
    """
    shift_r = _build_shift_mpo(n_qubits, "right")
    shift_l = _build_shift_mpo(n_qubits, "left")

    cores: List[Tensor] = []
    coeff = 1.0 / (dx * dx)

    for k in range(n_qubits):
        sr = shift_r[k]
        sl = shift_l[k]

        rL_r, d_out_r, d_in_r, rR_r = sr.shape
        rL_l, d_out_l, d_in_l, rR_l = sl.shape

        # Identity MPO core
        id_core = torch.zeros(1, 2, 2, 1, dtype=torch.float64)
        id_core[0, 0, 0, 0] = 1.0
        id_core[0, 1, 1, 0] = 1.0

        # Block-diagonal: [S_R, S_L, -2·I]
        rL = rL_r + rL_l + 1
        rR = rR_r + rR_l + 1

        core = torch.zeros(rL, 2, 2, rR, dtype=torch.float64)

        # S_R block
        core[:rL_r, :, :, :rR_r] = sr * coeff
        # S_L block
        core[rL_r:rL_r + rL_l, :, :, rR_r:rR_r + rR_l] = sl * coeff
        # -2I block
        core[rL_r + rL_l:, :, :, rR_r + rR_l:] = id_core * (-2.0 * coeff)

        cores.append(core)

    return cores
