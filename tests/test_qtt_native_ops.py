"""
Tests for tensornet.cfd.qtt_native_ops
=======================================

Covers:
- QTT round-trip (fold → unfold) identity via test helpers
- rSVD truncation correctness and rank bound
- QTT arithmetic (add, scalar multiply) linearity
- Hadamard product (compress-as-multiply and DMRG modes)
- QTT checkpoint save/load round-trip
- Inner product symmetry and positivity
- QTT point/batch evaluation
- Fuzz: random rank/dimension combos
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import List

import pytest
import torch
from torch import Tensor

# Skip entire module if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for QTT native ops",
)

from tensornet.cfd.qtt_native_ops import (
    QTTCores,
    qtt_truncate_now,
    qtt_truncate_sweep,
    qtt_add_native,
    qtt_scale_native,
    qtt_hadamard_native,
    qtt_inner_native,
    qtt_norm_native,
    qtt_eval_point,
    qtt_eval_batch,
    qtt_save,
    qtt_load,
    _RSVD_DEFAULT_POWER_ITER,
)

DEVICE = "cuda"
DTYPE = torch.float32


# ─── Test helpers: TT-SVD fold / full-contraction unfold ────────────
#
# These are NOT part of qtt_native_ops (the module works entirely in
# compressed QTT space).  We implement them here purely for verification.

def qtt_fold(x: Tensor, n_bits: int) -> QTTCores:
    """
    Convert a dense vector of length 2^n_bits to exact QTT via TT-SVD.

    Performs a left-to-right sweep of full SVDs so that the resulting
    cores represent x exactly (up to floating-point precision).
    """
    assert x.numel() == 2 ** n_bits
    cores: List[Tensor] = []
    remaining = x.reshape(1 * 2, -1)  # (r_prev*2, 2^(n_bits-1))
    r_prev = 1

    for i in range(n_bits - 1):
        U, S, Vh = torch.linalg.svd(remaining, full_matrices=False)
        r_next = S.shape[0]
        core = U.reshape(r_prev, 2, r_next)
        cores.append(core)
        remaining = (torch.diag(S) @ Vh).reshape(r_next * 2, -1)
        r_prev = r_next

    # Last core
    cores.append(remaining.reshape(r_prev, 2, 1))
    return QTTCores(cores)


def qtt_unfold(qtt: QTTCores) -> Tensor:
    """
    Contract all QTT cores into a dense vector of length 2^L.

    Uses sequential einsum contraction.
    """
    result = qtt.cores[0]  # (1, 2, r1)
    for core in qtt.cores[1:]:
        r_left = result.shape[0]
        n_left = result.shape[1]
        r_next = core.shape[2]
        # result[a, i, b] * core[b, j, c] → new[a, i*2+j, c]
        result = torch.einsum('aib,bjc->aijc', result, core)
        result = result.reshape(r_left, n_left * 2, r_next)
    return result.squeeze(0).squeeze(-1)


def _random_qtt(n_bits: int, max_rank: int) -> QTTCores:
    """Create a random QTT core chain for testing."""
    cores: List[Tensor] = []
    r_prev = 1
    for i in range(n_bits):
        r_next = 1 if i == n_bits - 1 else min(max_rank, 2 ** (i + 1))
        core = torch.randn(r_prev, 2, r_next, device=DEVICE, dtype=DTYPE)
        cores.append(core)
        r_prev = r_next
    return QTTCores(cores)


def _qtt_frobenius(qtt: QTTCores) -> float:
    """Compute ||QTT||_F via inner product."""
    return math.sqrt(max(0.0, qtt_inner_native(qtt, qtt).item()))


# ─── Round-trip identity ────────────────────────────────────────────

class TestFoldUnfold:
    """qtt_fold → qtt_unfold should recover original vector."""

    @pytest.mark.parametrize("n_bits", [4, 7, 10])
    def test_round_trip(self, n_bits: int) -> None:
        N = 2 ** n_bits
        x = torch.randn(N, device=DEVICE, dtype=DTYPE)
        qtt = qtt_fold(x, n_bits)
        x_rec = qtt_unfold(qtt)
        assert x_rec.shape == (N,)
        torch.testing.assert_close(x_rec, x, atol=1e-4, rtol=1e-4)

    def test_fold_output_shapes(self) -> None:
        N = 128  # 2^7
        x = torch.randn(N, device=DEVICE, dtype=DTYPE)
        qtt = qtt_fold(x, 7)
        assert qtt.num_sites == 7
        for c in qtt.cores:
            assert c.ndim == 3
            assert c.shape[1] == 2


# ─── Truncation ─────────────────────────────────────────────────────

class TestTruncation:
    """rSVD truncation respects rank bounds and preserves accuracy."""

    @pytest.mark.parametrize("max_rank", [4, 8, 16, 32])
    def test_rank_bound(self, max_rank: int) -> None:
        n_bits = 8
        x = torch.randn(2 ** n_bits, device=DEVICE, dtype=DTYPE)
        qtt = qtt_fold(x, n_bits)
        truncated = qtt_truncate_now(qtt, max_rank)
        for c in truncated.cores:
            assert c.shape[0] <= max_rank
            assert c.shape[2] <= max_rank

    def test_exact_representation_unchanged(self) -> None:
        """A low-rank signal should survive truncation losslessly."""
        n_bits = 6
        N = 2 ** n_bits
        # Constant vector → rank-1 QTT
        x = torch.ones(N, device=DEVICE, dtype=DTYPE) * 3.14
        qtt = qtt_fold(x, n_bits)
        truncated = qtt_truncate_now(qtt, max_rank=64)
        x_rec = qtt_unfold(truncated)
        torch.testing.assert_close(x_rec, x, atol=1e-4, rtol=1e-4)

    def test_configurable_power_iter(self) -> None:
        """_RSVD_DEFAULT_POWER_ITER is accessible and positive."""
        assert isinstance(_RSVD_DEFAULT_POWER_ITER, int)
        assert _RSVD_DEFAULT_POWER_ITER > 0


# ─── Arithmetic ─────────────────────────────────────────────────────

class TestArithmetic:
    """QTT addition and scalar multiplication obey linear algebra."""

    def test_add_commutativity(self) -> None:
        n_bits = 6
        a = _random_qtt(n_bits, 8)
        b = _random_qtt(n_bits, 8)
        ab = qtt_unfold(qtt_add_native(a, b, max_rank=64))
        ba = qtt_unfold(qtt_add_native(b, a, max_rank=64))
        torch.testing.assert_close(ab, ba, atol=1e-4, rtol=1e-4)

    def test_scalar_mul_linearity(self) -> None:
        n_bits = 6
        a = _random_qtt(n_bits, 8)
        x = qtt_unfold(a)
        scaled = qtt_unfold(qtt_scale_native(a, 2.5))
        torch.testing.assert_close(scaled, x * 2.5, atol=1e-4, rtol=1e-4)

    def test_add_zero(self) -> None:
        n_bits = 6
        N = 2 ** n_bits
        a = _random_qtt(n_bits, 8)
        zero = qtt_fold(torch.zeros(N, device=DEVICE, dtype=DTYPE), n_bits)
        result = qtt_unfold(qtt_add_native(a, zero, max_rank=64))
        expected = qtt_unfold(a)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)


# ─── Hadamard product ──────────────────────────────────────────────

class TestHadamard:
    """Element-wise product in QTT format."""

    def test_hadamard_correctness(self) -> None:
        n_bits = 6
        N = 2 ** n_bits
        a = torch.randn(N, device=DEVICE, dtype=DTYPE)
        b = torch.randn(N, device=DEVICE, dtype=DTYPE)
        ca = qtt_fold(a, n_bits)
        cb = qtt_fold(b, n_bits)
        result = qtt_unfold(qtt_hadamard_native(ca, cb, max_rank=64))
        expected = a * b
        rel_err = (result - expected).norm() / expected.norm()
        assert rel_err.item() < 0.02, f"Hadamard rel_err = {rel_err:.4f}"

    def test_hadamard_with_ones(self) -> None:
        """Hadamard with all-ones is identity."""
        n_bits = 6
        N = 2 ** n_bits
        x = torch.randn(N, device=DEVICE, dtype=DTYPE)
        ones = torch.ones(N, device=DEVICE, dtype=DTYPE)
        cx = qtt_fold(x, n_bits)
        c1 = qtt_fold(ones, n_bits)
        result = qtt_unfold(qtt_hadamard_native(cx, c1, max_rank=64))
        torch.testing.assert_close(result, x, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("mode", [
        "compress",
        pytest.param("dmrg", marks=pytest.mark.xfail(
            reason="Known einsum dimension bug in _hadamard_dmrg",
            strict=True,
        )),
    ])
    def test_hadamard_modes(self, mode: str) -> None:
        n_bits = 5
        N = 2 ** n_bits
        a = torch.randn(N, device=DEVICE, dtype=DTYPE)
        b = torch.randn(N, device=DEVICE, dtype=DTYPE)
        ca = qtt_fold(a, n_bits)
        cb = qtt_fold(b, n_bits)
        result = qtt_unfold(
            qtt_hadamard_native(ca, cb, max_rank=32, mode=mode)
        )
        expected = a * b
        rel_err = (result - expected).norm() / expected.norm()
        assert rel_err.item() < 0.05, (
            f"Hadamard mode={mode} rel_err = {rel_err:.4f}"
        )


# ─── Inner product ──────────────────────────────────────────────────

class TestInner:
    """QTT inner product properties."""

    def test_inner_positivity(self) -> None:
        a = _random_qtt(6, 8)
        val = qtt_inner_native(a, a).item()
        assert val > 0, f"Self inner product must be positive, got {val}"

    def test_inner_symmetry(self) -> None:
        a = _random_qtt(6, 8)
        b = _random_qtt(6, 8)
        ab = qtt_inner_native(a, b).item()
        ba = qtt_inner_native(b, a).item()
        assert abs(ab - ba) < 1e-4 * max(abs(ab), 1), (
            f"Inner product not symmetric: <a,b>={ab}, <b,a>={ba}"
        )

    def test_inner_matches_dense(self) -> None:
        n_bits = 6
        N = 2 ** n_bits
        a = torch.randn(N, device=DEVICE, dtype=DTYPE)
        b = torch.randn(N, device=DEVICE, dtype=DTYPE)
        ca = qtt_fold(a, n_bits)
        cb = qtt_fold(b, n_bits)
        qtt_val = qtt_inner_native(ca, cb).item()
        dense_val = (a * b).sum().item()
        assert abs(qtt_val - dense_val) < 1e-3 * max(abs(dense_val), 1), (
            f"Inner mismatch: qtt={qtt_val:.6f}, dense={dense_val:.6f}"
        )

    def test_norm_matches_dense(self) -> None:
        """qtt_norm_native matches dense L2 norm."""
        n_bits = 6
        x = torch.randn(2 ** n_bits, device=DEVICE, dtype=DTYPE)
        qtt = qtt_fold(x, n_bits)
        qtt_n = qtt_norm_native(qtt).item()
        dense_n = x.norm().item()
        assert abs(qtt_n - dense_n) < 1e-3 * max(dense_n, 1), (
            f"Norm mismatch: qtt={qtt_n:.6f}, dense={dense_n:.6f}"
        )


# ─── Point & batch evaluation ──────────────────────────────────────

class TestEval:
    """qtt_eval_point / qtt_eval_batch match dense."""

    def test_eval_point_matches_dense(self) -> None:
        n_bits = 6
        N = 2 ** n_bits
        x = torch.randn(N, device=DEVICE, dtype=DTYPE)
        qtt = qtt_fold(x, n_bits)
        for idx in [0, 1, N // 2, N - 1]:
            val = qtt_eval_point(qtt, idx).item()
            assert abs(val - x[idx].item()) < 1e-4, (
                f"Point eval mismatch at {idx}: got {val}, expected {x[idx].item()}"
            )

    def test_eval_batch_matches_dense(self) -> None:
        n_bits = 7
        N = 2 ** n_bits
        x = torch.randn(N, device=DEVICE, dtype=DTYPE)
        qtt = qtt_fold(x, n_bits)
        indices = torch.arange(N, device=DEVICE)
        vals = qtt_eval_batch(qtt, indices)
        torch.testing.assert_close(vals, x, atol=1e-4, rtol=1e-4)


# ─── Checkpoint save/load ──────────────────────────────────────────

class TestCheckpoint:
    """QTT save/load round-trip."""

    def test_save_load_round_trip(self) -> None:
        a = _random_qtt(7, 16)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = Path(f.name)

        try:
            qtt_save(a, str(path))
            loaded = qtt_load(str(path), device=DEVICE)
            assert len(loaded.cores) == len(a.cores)
            for c_orig, c_load in zip(a.cores, loaded.cores):
                torch.testing.assert_close(c_orig, c_load)
        finally:
            path.unlink(missing_ok=True)

    def test_save_load_preserves_values(self) -> None:
        n_bits = 6
        N = 2 ** n_bits
        x = torch.randn(N, device=DEVICE, dtype=DTYPE)
        qtt = qtt_fold(x, n_bits)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = Path(f.name)

        try:
            qtt_save(qtt, str(path))
            loaded = qtt_load(str(path), device=DEVICE)
            x_rec = qtt_unfold(loaded)
            torch.testing.assert_close(x_rec, x, atol=1e-4, rtol=1e-4)
        finally:
            path.unlink(missing_ok=True)


# ─── QTTCores dataclass ────────────────────────────────────────────

class TestQTTCoresAPI:
    """QTTCores properties and methods."""

    def test_properties(self) -> None:
        a = _random_qtt(6, 8)
        assert a.num_sites == 6
        assert len(a.ranks) == 7  # L+1 bond dims
        assert a.ranks[0] == 1
        assert a.ranks[-1] == 1
        assert a.max_rank <= 8
        assert 1.0 <= a.mean_rank <= 8.0
        assert a.total_params > 0
        assert a.device.type == "cuda"
        assert a.dtype == DTYPE

    def test_clone(self) -> None:
        a = _random_qtt(5, 4)
        b = a.clone()
        # Modify original — clone should be independent
        a.cores[0].fill_(0.0)
        assert b.cores[0].abs().sum().item() > 0


# ─── Fuzz ───────────────────────────────────────────────────────────

class TestFuzz:
    """Random-parameter stress tests."""

    @pytest.mark.parametrize("n_bits", [3, 5, 7, 9])
    @pytest.mark.parametrize("max_rank", [2, 8, 32])
    def test_fold_truncate_unfold(self, n_bits: int, max_rank: int) -> None:
        N = 2 ** n_bits
        x = torch.randn(N, device=DEVICE, dtype=DTYPE)
        qtt = qtt_fold(x, n_bits)
        truncated = qtt_truncate_now(qtt, max_rank)
        x_rec = qtt_unfold(truncated)
        assert x_rec.shape == (N,)
        # At minimum, rank-1 approximation captures dominant component
        assert x_rec.norm().item() > 0

    @pytest.mark.parametrize("seed", range(5))
    def test_arithmetic_consistency(self, seed: int) -> None:
        torch.manual_seed(seed)
        n_bits = 5
        N = 2 ** n_bits
        a = torch.randn(N, device=DEVICE, dtype=DTYPE)
        b = torch.randn(N, device=DEVICE, dtype=DTYPE)
        alpha = torch.randn(1).item()

        ca = qtt_fold(a, n_bits)
        cb = qtt_fold(b, n_bits)

        # Test: alpha*a + b in QTT ≈ alpha*a + b dense
        ca_scaled = qtt_scale_native(ca, alpha)
        result = qtt_unfold(qtt_add_native(ca_scaled, cb, max_rank=64))
        expected = alpha * a + b
        rel_err = (result - expected).norm() / expected.norm()
        assert rel_err.item() < 0.02, (
            f"seed={seed}, alpha={alpha:.3f}, rel_err={rel_err:.4f}"
        )
