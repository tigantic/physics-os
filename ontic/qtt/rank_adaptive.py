"""
Automatic Rank Adaptation for TT Decompositions
=================================================

Bayesian and information-theoretic criteria for selecting TT-rank
automatically, eliminating manual rank tuning.

Algorithms
----------
* **AIC / BIC rank selection** — Akaike / Bayesian information criterion on
  the singular-value spectrum: select rank that minimises AIC/BIC.
* **MDL (Minimum Description Length)** — two-part code: model cost (rank
  parameters) vs. data fit.
* **Bayesian rank estimation** — marginal-likelihood approximation via
  Laplace or evidence framework.
* **Residual-based adaptive rounding** — iteratively increase rank until
  ``||A - TT(A)||_F / ||A||_F < ε``.

Key functions
-------------
* :func:`rank_aic`     — rank per bond via AIC
* :func:`rank_bic`     — rank per bond via BIC
* :func:`rank_mdl`     — rank per bond via MDL
* :func:`adaptive_round` — residual-driven iterative rounding
* :func:`bayesian_rank` — evidence-maximisation rank selection
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray


# ======================================================================
# Information criteria on a singular-value spectrum
# ======================================================================

def _log_likelihood(S: NDArray, k: int) -> float:
    """
    Gaussian log-likelihood for keeping the top-*k* singular values
    and modelling the rest as isotropic noise.
    """
    n = len(S)
    if k >= n or k < 1:
        return -np.inf
    signal = S[:k]
    noise = S[k:]
    sigma2 = float(np.mean(noise ** 2)) if len(noise) > 0 else 1e-30
    sigma2 = max(sigma2, 1e-30)
    ll = -0.5 * float(np.sum(signal ** 2 / sigma2 + np.log(sigma2)))
    ll -= 0.5 * float(np.sum(noise ** 2 / sigma2 + np.log(sigma2)))
    return ll


def _n_params(k: int, m: int, n: int) -> int:
    """Number of free parameters for a rank-k truncation of an m×n matrix."""
    return k * (m + n - k)


def rank_aic(S: NDArray, m: int, n: int) -> int:
    """
    Select rank via Akaike Information Criterion.

    AIC = -2 ln L + 2 p

    Parameters
    ----------
    S : NDArray
        Singular values (descending order).
    m, n : int
        Original matrix dimensions.

    Returns
    -------
    int
        Optimal rank (≥ 1).
    """
    best_k = 1
    best_aic = np.inf
    for k in range(1, len(S)):
        ll = _log_likelihood(S, k)
        p = _n_params(k, m, n)
        aic = -2.0 * ll + 2.0 * p
        if aic < best_aic:
            best_aic = aic
            best_k = k
    return best_k


def rank_bic(S: NDArray, m: int, n: int, n_samples: Optional[int] = None) -> int:
    """
    Select rank via Bayesian Information Criterion.

    BIC = -2 ln L + p ln N_samples

    Parameters
    ----------
    S : NDArray
        Singular values.
    m, n : int
        Matrix dimensions.
    n_samples : int, optional
        Effective sample size (defaults to ``m * n``).

    Returns
    -------
    int
        Optimal rank.
    """
    if n_samples is None:
        n_samples = m * n
    best_k = 1
    best_bic = np.inf
    for k in range(1, len(S)):
        ll = _log_likelihood(S, k)
        p = _n_params(k, m, n)
        bic = -2.0 * ll + p * np.log(n_samples)
        if bic < best_bic:
            best_bic = bic
            best_k = k
    return best_k


def rank_mdl(S: NDArray, m: int, n: int) -> int:
    """
    Select rank via Minimum Description Length.

    Two-part code: bit cost of storing the low-rank model +
    bit cost of the residual at noise level.

    Parameters
    ----------
    S : NDArray
        Singular values.
    m, n : int
        Matrix dimensions.

    Returns
    -------
    int
        Optimal rank.
    """
    total_energy = float(np.sum(S ** 2))
    best_k = 1
    best_mdl = np.inf
    for k in range(1, len(S)):
        residual_energy = float(np.sum(S[k:] ** 2))
        n_p = _n_params(k, m, n)
        # Model cost (in nats)
        model_cost = 0.5 * n_p * np.log(total_energy + 1e-30)
        # Data cost
        data_cost = 0.5 * (m * n - n_p) * np.log(
            residual_energy / max(m * n - n_p, 1) + 1e-30
        )
        mdl = model_cost + data_cost
        if mdl < best_mdl:
            best_mdl = mdl
            best_k = k
    return best_k


# ======================================================================
# Bayesian rank estimation (evidence approximation)
# ======================================================================

def bayesian_rank(
    S: NDArray,
    m: int,
    n: int,
    alpha_prior: float = 1.0,
) -> int:
    """
    Evidence-maximisation rank selection (Laplace approximation).

    Computes the log-marginal-likelihood for each candidate rank and
    returns the maximiser.

    Parameters
    ----------
    S : NDArray
        Singular values.
    m, n : int
        Matrix dimensions.
    alpha_prior : float
        Prior precision hyper-parameter.

    Returns
    -------
    int
        Optimal rank.
    """
    best_k = 1
    best_evidence = -np.inf
    total_sq = float(np.sum(S ** 2))

    for k in range(1, len(S)):
        residual = float(np.sum(S[k:] ** 2))
        sigma2 = residual / max(m * n - _n_params(k, m, n), 1)
        sigma2 = max(sigma2, 1e-30)
        n_p = _n_params(k, m, n)

        # Log-likelihood
        ll = -0.5 * residual / sigma2 - 0.5 * (m * n) * np.log(2 * np.pi * sigma2)
        # Log-prior
        lp = 0.5 * n_p * np.log(alpha_prior) - 0.5 * alpha_prior * float(np.sum(S[:k] ** 2))
        # Laplace correction (log det Hessian)
        lh = -0.5 * n_p * np.log(2 * np.pi)

        evidence = ll + lp + lh
        if evidence > best_evidence:
            best_evidence = evidence
            best_k = k

    return best_k


# ======================================================================
# Adaptive TT-rounding
# ======================================================================

@dataclass
class AdaptiveRoundResult:
    """
    Result of adaptive rounding.

    Attributes
    ----------
    cores : list[NDArray]
        Rounded TT-cores.
    ranks : list[int]
        Bond dimensions after rounding.
    relative_error : float
        Achieved relative error.
    rank_history : list[list[int]]
        Bond dimensions at each iteration.
    """
    cores: list[NDArray]
    ranks: list[int]
    relative_error: float
    rank_history: list[list[int]]


def adaptive_round(
    cores: list[NDArray],
    tol: float = 1e-6,
    max_rank: int = 256,
    initial_rank: int = 2,
    growth_factor: float = 1.5,
    criterion: str = "bic",
) -> AdaptiveRoundResult:
    """
    Adaptively round a TT until the relative error is below *tol*.

    Strategy: start with ``initial_rank``, round, estimate error from
    discarded singular values, and increase rank by ``growth_factor``
    if error exceeds *tol*.

    Optionally, at each bond an information criterion (``aic``, ``bic``,
    ``mdl``, ``bayesian``) selects the local rank instead of the global
    cap.

    Parameters
    ----------
    cores : list[NDArray]
        Input TT-cores ``(r_l, d, r_r)``.
    tol : float
        Target relative Frobenius error.
    max_rank : int
        Hard upper bound on bond dimension.
    initial_rank : int
        Starting rank guess.
    growth_factor : float
        Multiplicative increase when error exceeds tolerance.
    criterion : str
        Rank selection criterion: ``"aic"``, ``"bic"``, ``"mdl"``,
        ``"bayesian"``, or ``"none"`` (use max_rank directly).

    Returns
    -------
    AdaptiveRoundResult
    """
    from ontic.qtt.sparse_direct import tt_round

    selectors = {
        "aic": rank_aic,
        "bic": rank_bic,
        "mdl": rank_mdl,
        "bayesian": bayesian_rank,
    }

    N = len(cores)
    current_rank = initial_rank
    history: list[list[int]] = []

    while current_rank <= max_rank:
        # Round with current global rank cap
        rounded = tt_round(cores, max_rank=current_rank)

        # Estimate relative error from discarded singular values
        total_sq = 0.0
        discard_sq = 0.0
        ranks: list[int] = []

        for k in range(N - 1):
            r_l, d, r_r = cores[k].shape
            mat = cores[k].reshape(r_l * d, r_r)
            S = np.linalg.svd(mat, compute_uv=False)
            total_sq += float(np.sum(S ** 2))

            if criterion in selectors:
                sel_rank = selectors[criterion](S, r_l * d, r_r)
                sel_rank = min(sel_rank, current_rank)
            else:
                sel_rank = min(current_rank, len(S))

            if sel_rank < len(S):
                discard_sq += float(np.sum(S[sel_rank:] ** 2))
            ranks.append(sel_rank)

        ranks.append(rounded[-1].shape[2] if len(rounded) > 0 else 1)
        history.append(ranks)

        rel_err = np.sqrt(discard_sq / max(total_sq, 1e-30))
        if rel_err <= tol or current_rank >= max_rank:
            # Apply per-bond ranks from the criterion
            final = _round_per_bond(cores, ranks)
            return AdaptiveRoundResult(
                cores=final,
                ranks=[c.shape[2] for c in final],
                relative_error=float(rel_err),
                rank_history=history,
            )

        current_rank = min(int(np.ceil(current_rank * growth_factor)), max_rank)

    # Fallback
    rounded = tt_round(cores, max_rank=max_rank)
    return AdaptiveRoundResult(
        cores=rounded,
        ranks=[c.shape[2] for c in rounded],
        relative_error=float(rel_err) if 'rel_err' in dir() else 1.0,
        rank_history=history,
    )


def _round_per_bond(
    cores: list[NDArray],
    per_bond_ranks: list[int],
) -> list[NDArray]:
    """
    Round TT with different rank limits at each bond.
    """
    N = len(cores)
    out = [c.copy() for c in cores]

    # Left-to-right QR
    for k in range(N - 1):
        r_l, d, r_r = out[k].shape
        mat = out[k].reshape(r_l * d, r_r)
        Q, R = np.linalg.qr(mat)
        new_r = Q.shape[1]
        out[k] = Q.reshape(r_l, d, new_r)
        out[k + 1] = np.einsum('ij,jkl->ikl', R, out[k + 1])

    # Right-to-left SVD truncation with per-bond ranks
    for k in range(N - 1, 0, -1):
        r_l, d, r_r = out[k].shape
        mat = out[k].reshape(r_l, d * r_r)
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)
        keep = min(per_bond_ranks[k - 1] if k - 1 < len(per_bond_ranks) else len(S), len(S))
        keep = max(keep, 1)
        out[k] = (np.diag(S[:keep]) @ Vh[:keep]).reshape(keep, d, r_r)
        out[k - 1] = np.einsum('ijk,kl->ijl', out[k - 1], U[:, :keep])

    return out
