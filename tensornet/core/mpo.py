"""
Matrix Product Operator (MPO)
=============================

MPO representation for operators on 1D systems (Hamiltonians, time evolution).

Tensor convention:
    W[i] : (D_left, d_out, d_in, D_right)

    Operator O = Σ W[0]_{1,σ₀,σ'₀,α₀} W[1]_{α₀,σ₁,σ'₁,α₁} ... |σ⟩⟨σ'|
"""

from __future__ import annotations

import torch
from torch import Tensor


class MPO:
    """
    Matrix Product Operator representation.

    Used for Hamiltonians, time evolution operators, and observables.

    Attributes:
        tensors: List of tensors W[i] with shape (D_left, d_out, d_in, D_right)
        L: Number of sites
        d: Physical dimension
        D: Maximum MPO bond dimension

    Example:
        >>> from tensornet import heisenberg_mpo
        >>> H = heisenberg_mpo(L=10, J=1.0)
        >>> print(f"MPO bond dimension: {H.D}")
    """

    def __init__(self, tensors: list[Tensor]):
        """
        Initialize MPO from list of tensors.

        Args:
            tensors: List of tensors with shape (D_left, d_out, d_in, D_right)
        """
        self.tensors = tensors

    @property
    def L(self) -> int:
        """Number of sites."""
        return len(self.tensors)

    @property
    def d(self) -> int:
        """Physical dimension."""
        return self.tensors[0].shape[1]

    @property
    def D(self) -> int:
        """Maximum MPO bond dimension."""
        return max(max(t.shape[0], t.shape[3]) for t in self.tensors)

    @property
    def dtype(self) -> torch.dtype:
        """Data type."""
        return self.tensors[0].dtype

    @property
    def device(self) -> torch.device:
        """Device."""
        return self.tensors[0].device

    def to_matrix(self) -> Tensor:
        """
        Contract MPO to dense matrix.

        Warning: Exponential in L. Only for small systems.

        Returns:
            Dense matrix of shape (d^L, d^L)
        """
        # Start with first tensor
        result = self.tensors[0]  # (1, d, d, D)

        for i in range(1, self.L):
            # Contract bond indices
            result = torch.einsum("...ija,aklb->...ikjlb", result, self.tensors[i])
            # Merge physical indices
            shape = result.shape
            result = result.reshape(
                *shape[:-5], shape[-5] * shape[-4], shape[-3] * shape[-2], shape[-1]
            )

        # Remove boundary bonds and reshape
        result = result.squeeze(0).squeeze(-1)
        return result

    def apply(self, mps) -> MPS:
        """
        Apply MPO to MPS: |ψ'⟩ = O|ψ⟩

        The result has bond dimension χ * D.

        Args:
            mps: Input MPS

        Returns:
            New MPS with O applied
        """
        from tensornet.core.mps import MPS

        new_tensors = []

        for i in range(self.L):
            A = mps.tensors[i]  # (χ_l, d, χ_r)
            W = self.tensors[i]  # (D_l, d_out, d_in, D_r)

            # Contract physical index
            # B_{(χ_l,D_l), d_out, (χ_r,D_r)} = Σ_d W_{D_l,d_out,d,D_r} A_{χ_l,d,χ_r}
            # W: (D_l, d_out, d_in, D_r) = abcd, A: (χ_l, d_in, χ_r) = ecf -> (e,a,b,f,d)
            B = torch.einsum("abcd,ecf->eabfd", W, A)

            # Reshape to combined bond dimensions
            chi_l, D_l = A.shape[0], W.shape[0]
            d_out = W.shape[1]
            chi_r, D_r = A.shape[2], W.shape[3]

            B = B.reshape(chi_l * D_l, d_out, chi_r * D_r)
            new_tensors.append(B)

        return MPS(new_tensors)

    def expectation(self, mps) -> Tensor:
        """
        Compute ⟨ψ|O|ψ⟩.

        Args:
            mps: MPS state

        Returns:
            Expectation value
        """
        # Contract environments from left
        # env[a,w,b]: (χ_ket, D_mpo, χ_bra) represents contracted sites 0..i-1
        # a = ket left bond, w = mpo left bond, b = bra left bond

        chi_l = mps.tensors[0].shape[0]
        D_l = self.tensors[0].shape[0]
        env = torch.zeros(chi_l, D_l, chi_l, dtype=self.dtype, device=self.device)
        env[0, 0, 0] = 1.0

        for i in range(self.L):
            A = mps.tensors[i]  # (χ_l, d, χ_r) = [a, s, c]
            W = self.tensors[i]  # (D_l, d_out, d_in, D_r) = [w, t, s, x]
            A_conj = A.conj()  # [b, t, d]

            # Contract: env[a,w,b] A[a,s,c] W[w,t,s,x] A*[b,t,d] -> new_env[c,x,d]
            # Step 1: env[a,w,b] A[a,s,c] -> [w,b,s,c]
            temp1 = torch.einsum("awb,asc->wbsc", env, A)
            # Step 2: [w,b,s,c] W[w,t,s,x] -> [b,t,c,x]
            temp2 = torch.einsum("wbsc,wtsx->btcx", temp1, W)
            # Step 3: [b,t,c,x] A*[b,t,d] -> [c,x,d]
            env = torch.einsum("btcx,btd->cxd", temp2, A_conj)

        # Final contraction - env should be (1, 1, 1)
        return env.squeeze()

    def is_hermitian(self, tol: float = 1e-10) -> bool:
        """
        Check if MPO is Hermitian.

        Args:
            tol: Tolerance for comparison

        Returns:
            True if H = H†
        """
        H = self.to_matrix()
        return torch.allclose(H, H.T.conj(), atol=tol)

    def copy(self) -> MPO:
        """Return deep copy."""
        return MPO([t.clone() for t in self.tensors])

    def __repr__(self) -> str:
        return f"MPO(L={self.L}, d={self.d}, D={self.D}, dtype={self.dtype})"


def mpo_sum(mpo1: MPO, mpo2: MPO) -> MPO:
    """
    Sum two MPOs: O = O1 + O2.

    Result has bond dimension D1 + D2.

    Args:
        mpo1: First MPO
        mpo2: Second MPO

    Returns:
        Sum MPO
    """
    L = mpo1.L
    assert mpo2.L == L

    tensors = []

    for i in range(L):
        W1 = mpo1.tensors[i]
        W2 = mpo2.tensors[i]

        D1_l, d, _, D1_r = W1.shape
        D2_l, _, _, D2_r = W2.shape

        if i == 0:
            # First site: concatenate along right bond
            W = torch.cat([W1, W2], dim=3)
        elif i == L - 1:
            # Last site: concatenate along left bond
            W = torch.cat([W1, W2], dim=0)
        else:
            # Bulk: block diagonal
            W = torch.zeros(
                D1_l + D2_l, d, d, D1_r + D2_r, dtype=W1.dtype, device=W1.device
            )
            W[:D1_l, :, :, :D1_r] = W1
            W[D1_l:, :, :, D1_r:] = W2

        tensors.append(W)

    return MPO(tensors)
