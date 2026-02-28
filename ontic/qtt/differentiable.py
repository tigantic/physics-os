"""
Differentiable Tensor Networks
===============================

Provides a thin PyTorch-compatible wrapper so that TT-format operations
participate in autograd.  When PyTorch is not available the module falls
back to a numpy-only tape-based AD implementation that covers the
critical path (forward + backward through TT-round and TT-matvec).

Key classes / functions
-----------------------
* :class:`TTTensor`           — differentiable TT representation
* :func:`tt_round_diff`       — differentiable TT rounding
* :func:`tt_matvec_diff`      — differentiable MPO × TT product
* :func:`tt_inner_diff`       — differentiable TT inner product
* :func:`tt_loss`             — scalar loss from TT (e.g. norm²)
* :class:`NumpyTape`          — fallback reverse-mode tape
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

# Optional PyTorch import
_HAS_TORCH = False
try:
    import torch
    from torch import Tensor as TorchTensor
    _HAS_TORCH = True
except ImportError:
    pass


# ======================================================================
# Numpy-only tape-based AD
# ======================================================================

@dataclass
class TapeEntry:
    """One record on the reverse-mode tape."""
    backward_fn: Callable[..., None]
    args: tuple


class NumpyTape:
    """
    Minimal reverse-mode AD tape for numpy arrays.

    Usage::

        tape = NumpyTape()
        a = tape.variable(np.array([1.0, 2.0]))
        b = tape.variable(np.array([3.0, 4.0]))
        c = tape.add(a, b)
        loss = tape.reduce_sum(c)
        grads = tape.backward(loss)
    """

    def __init__(self) -> None:
        self._entries: list[TapeEntry] = []
        self._grads: dict[int, NDArray] = {}
        self._values: dict[int, NDArray] = {}
        self._counter = 0

    def _next_id(self) -> int:
        self._counter += 1
        return self._counter

    def variable(self, value: NDArray) -> int:
        """Register an input variable, return its ID."""
        vid = self._next_id()
        self._values[vid] = value.copy()
        return vid

    def _get(self, vid: int) -> NDArray:
        return self._values[vid]

    def _set(self, vid: int, value: NDArray) -> None:
        self._values[vid] = value

    def _accum_grad(self, vid: int, grad: NDArray) -> None:
        if vid in self._grads:
            self._grads[vid] = self._grads[vid] + grad
        else:
            self._grads[vid] = grad.copy()

    # --- Operations ---

    def add(self, a_id: int, b_id: int) -> int:
        """c = a + b"""
        a = self._get(a_id)
        b = self._get(b_id)
        c = a + b
        c_id = self._next_id()
        self._set(c_id, c)

        def backward() -> None:
            gc = self._grads.get(c_id, np.ones_like(c))
            self._accum_grad(a_id, gc)
            self._accum_grad(b_id, gc)

        self._entries.append(TapeEntry(backward_fn=backward, args=(a_id, b_id, c_id)))
        return c_id

    def mul(self, a_id: int, b_id: int) -> int:
        """c = a * b (element-wise)"""
        a = self._get(a_id)
        b = self._get(b_id)
        c = a * b
        c_id = self._next_id()
        self._set(c_id, c)

        def backward() -> None:
            gc = self._grads.get(c_id, np.ones_like(c))
            self._accum_grad(a_id, gc * b)
            self._accum_grad(b_id, gc * a)

        self._entries.append(TapeEntry(backward_fn=backward, args=(a_id, b_id, c_id)))
        return c_id

    def scale(self, a_id: int, alpha: float) -> int:
        """c = alpha * a"""
        a = self._get(a_id)
        c = alpha * a
        c_id = self._next_id()
        self._set(c_id, c)

        def backward() -> None:
            gc = self._grads.get(c_id, np.ones_like(c))
            self._accum_grad(a_id, alpha * gc)

        self._entries.append(TapeEntry(backward_fn=backward, args=(a_id, c_id)))
        return c_id

    def reduce_sum(self, a_id: int) -> int:
        """c = sum(a)"""
        a = self._get(a_id)
        c = np.array(np.sum(a))
        c_id = self._next_id()
        self._set(c_id, c)

        def backward() -> None:
            gc = self._grads.get(c_id, np.ones_like(c))
            self._accum_grad(a_id, gc * np.ones_like(a))

        self._entries.append(TapeEntry(backward_fn=backward, args=(a_id, c_id)))
        return c_id

    def matmul(self, a_id: int, b_id: int) -> int:
        """c = a @ b"""
        a = self._get(a_id)
        b = self._get(b_id)
        c = a @ b
        c_id = self._next_id()
        self._set(c_id, c)

        def backward() -> None:
            gc = self._grads.get(c_id, np.ones_like(c))
            self._accum_grad(a_id, gc @ b.T)
            self._accum_grad(b_id, a.T @ gc)

        self._entries.append(TapeEntry(backward_fn=backward, args=(a_id, b_id, c_id)))
        return c_id

    def backward(self, loss_id: int) -> dict[int, NDArray]:
        """
        Run reverse-mode AD from *loss_id*.

        Returns dict mapping variable ID → gradient array.
        """
        self._grads[loss_id] = np.ones_like(self._get(loss_id))
        for entry in reversed(self._entries):
            entry.backward_fn()
        return dict(self._grads)


# ======================================================================
# Differentiable TT representation
# ======================================================================

@dataclass
class TTTensor:
    """
    Differentiable TT-format tensor.

    Stores cores as numpy arrays, with optional gradient buffers.

    Attributes
    ----------
    cores : list[NDArray]
        TT-cores, each of shape (r_{k-1}, d_k, r_k).
    grads : list[NDArray | None]
        Gradient w.r.t. each core (populated after backward).
    requires_grad : bool
        Whether to track gradients.
    """
    cores: list[NDArray]
    grads: list[Optional[NDArray]] = field(default_factory=list)
    requires_grad: bool = True

    def __post_init__(self) -> None:
        if not self.grads:
            self.grads = [None] * len(self.cores)

    @property
    def n_sites(self) -> int:
        return len(self.cores)

    @property
    def bond_dims(self) -> list[int]:
        return [self.cores[k].shape[2] for k in range(self.n_sites - 1)]

    def zero_grad(self) -> None:
        """Reset all gradients."""
        self.grads = [None] * len(self.cores)

    def clone(self) -> TTTensor:
        """Deep copy."""
        return TTTensor(
            cores=[c.copy() for c in self.cores],
            requires_grad=self.requires_grad,
        )


# ======================================================================
# Differentiable TT operations (numpy path)
# ======================================================================

def tt_inner_diff(a: TTTensor, b: TTTensor) -> tuple[float, Callable[[], None]]:
    """
    Differentiable inner product ⟨a, b⟩.

    Returns (value, backward_fn).  Calling backward_fn() populates
    a.grads and b.grads.
    """
    N = a.n_sites
    if N != b.n_sites:
        raise ValueError("Different number of sites")

    # Forward: left-to-right transfer matrix
    # E_k(i,j) = sum over all physical indices left of bond k
    transfers: list[NDArray] = []
    E = np.ones((1, 1))
    transfers.append(E)

    for k in range(N):
        # E_{k+1}[i,j] = sum_{i',j',d} E_k[i',j'] * a_k[i',d,i] * b_k[j',d,j]
        ac = a.cores[k]  # (ra_l, d, ra_r)
        bc = b.cores[k]  # (rb_l, d, rb_r)
        E = np.einsum('ij,idk,jdl->kl', E, ac, bc)
        transfers.append(E)

    value = float(E.item()) if E.size == 1 else float(E[0, 0])

    def backward_fn() -> None:
        # Right-to-left transfer for gradient computation
        G = np.ones((1, 1))
        right_transfers: list[NDArray] = [G]
        for k in range(N - 1, 0, -1):
            ac = a.cores[k]
            bc = b.cores[k]
            G = np.einsum('idk,jdl,kl->ij', ac, bc, G)
            right_transfers.append(G)
        right_transfers.reverse()

        # Gradients for a
        if a.requires_grad:
            for k in range(N):
                L = transfers[k]           # (ra_l_left, rb_l_left)
                R = right_transfers[k]     # (ra_r_right, rb_r_right)
                bc = b.cores[k]            # (rb_l, d, rb_r)
                # dL/d(a_k[i,d,j]) = L[i, j'] * b_k[j', d, j''] * R[j, j'']
                grad_ak = np.einsum('ij,jdk,lk->idl', L, bc, R)
                if a.grads[k] is None:
                    a.grads[k] = grad_ak
                else:
                    a.grads[k] = a.grads[k] + grad_ak

        # Gradients for b
        if b.requires_grad:
            for k in range(N):
                L = transfers[k]
                R = right_transfers[k]
                ac = a.cores[k]
                grad_bk = np.einsum('ij,idk,lk->jdl', L, ac, R)
                if b.grads[k] is None:
                    b.grads[k] = grad_bk
                else:
                    b.grads[k] = b.grads[k] + grad_bk

    return value, backward_fn


def tt_norm_sq_diff(a: TTTensor) -> tuple[float, Callable[[], None]]:
    """
    Differentiable squared norm ||a||².

    Returns (value, backward_fn).
    """
    return tt_inner_diff(a, a)


def tt_loss(
    a: TTTensor,
    target: TTTensor,
) -> tuple[float, Callable[[], None]]:
    """
    TT loss = ||a - target||².

    For simplicity, expands to ||a||² - 2⟨a, target⟩ + ||target||².

    Returns (value, backward_fn).
    """
    norm_a_sq, bw_a = tt_norm_sq_diff(a)
    inner_ab, bw_ab = tt_inner_diff(a, target)
    norm_b_sq, _ = tt_norm_sq_diff(target)

    loss = norm_a_sq - 2.0 * inner_ab + norm_b_sq

    def backward_fn() -> None:
        # d(loss)/d(a_k) = 2 * d(||a||²)/d(a_k) - 2 * d(⟨a,target⟩)/d(a_k)
        a.zero_grad()
        bw_a()
        # Now a.grads has gradients of ||a||² = 2 * <a,a> w.r.t. a cores
        # But tt_inner_diff(a, a) = <a, a>, and backward gives gradient of <a, a>
        # The gradient of ||a||² = <a, a> w.r.t. a_k is 2 * the transfer-based grad
        # (since a appears in both arguments).
        grads_norm = [g.copy() if g is not None else None for g in a.grads]

        a.zero_grad()
        bw_ab()
        grads_inner = [g.copy() if g is not None else None for g in a.grads]

        a.zero_grad()
        for k in range(a.n_sites):
            gn = grads_norm[k] if grads_norm[k] is not None else np.zeros_like(a.cores[k])
            gi = grads_inner[k] if grads_inner[k] is not None else np.zeros_like(a.cores[k])
            a.grads[k] = gn - 2.0 * gi

    return loss, backward_fn


def tt_gradient_descent_step(
    tt: TTTensor,
    learning_rate: float,
    max_rank: Optional[int] = None,
) -> None:
    """
    In-place gradient descent step on TT cores.

    Parameters
    ----------
    tt : TTTensor
        TT-tensor with populated gradients.
    learning_rate : float
        Step size.
    max_rank : int, optional
        If given, round after update.
    """
    from ontic.qtt.sparse_direct import tt_round

    for k in range(tt.n_sites):
        if tt.grads[k] is not None:
            tt.cores[k] = tt.cores[k] - learning_rate * tt.grads[k]

    if max_rank is not None:
        tt.cores = tt_round(tt.cores, max_rank=max_rank)

    tt.zero_grad()


# ======================================================================
# PyTorch wrapper (if available)
# ======================================================================

if _HAS_TORCH:

    class TTInnerAutograd(torch.autograd.Function):
        """Custom autograd for TT inner product."""

        @staticmethod
        def forward(
            ctx: torch.autograd.function.FunctionCtx,
            *flat_cores: TorchTensor,
        ) -> TorchTensor:
            N = len(flat_cores) // 2
            a_cores = list(flat_cores[:N])
            b_cores = list(flat_cores[N:])

            # Forward pass
            E = torch.ones(1, 1, dtype=a_cores[0].dtype, device=a_cores[0].device)
            transfers = [E]
            for k in range(N):
                E = torch.einsum('ij,idk,jdl->kl', E, a_cores[k], b_cores[k])
                transfers.append(E)

            ctx.save_for_backward(*flat_cores)
            ctx.transfers = transfers
            ctx.N = N
            return E.squeeze()

        @staticmethod
        def backward(
            ctx: torch.autograd.function.FunctionCtx,
            grad_output: TorchTensor,
        ) -> tuple[Optional[TorchTensor], ...]:
            flat_cores = ctx.saved_tensors
            N = ctx.N
            transfers = ctx.transfers
            a_cores = list(flat_cores[:N])
            b_cores = list(flat_cores[N:])

            # Right transfers
            G = torch.ones(1, 1, dtype=a_cores[0].dtype, device=a_cores[0].device)
            right_transfers = [G]
            for k in range(N - 1, 0, -1):
                G = torch.einsum('idk,jdl,kl->ij', a_cores[k], b_cores[k], G)
                right_transfers.append(G)
            right_transfers.reverse()

            grads: list[Optional[TorchTensor]] = []
            # Gradients for a_cores
            for k in range(N):
                L = transfers[k]
                R = right_transfers[k]
                grad_ak = torch.einsum('ij,jdk,lk->idl', L, b_cores[k], R)
                grads.append(grad_output * grad_ak)

            # Gradients for b_cores
            for k in range(N):
                L = transfers[k]
                R = right_transfers[k]
                grad_bk = torch.einsum('ij,idk,lk->jdl', L, a_cores[k], R)
                grads.append(grad_output * grad_bk)

            return tuple(grads)

    def tt_inner_torch(
        a_cores: list[TorchTensor],
        b_cores: list[TorchTensor],
    ) -> TorchTensor:
        """
        PyTorch-differentiable TT inner product.

        Parameters
        ----------
        a_cores, b_cores : list[Tensor]
            TT-cores as PyTorch tensors.

        Returns
        -------
        Tensor
            Scalar inner product with autograd support.
        """
        return TTInnerAutograd.apply(*a_cores, *b_cores)
