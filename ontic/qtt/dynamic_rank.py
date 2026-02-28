"""
Dynamic Rank Adaptation During Time Integration
=================================================

Rank grows when physics demands it (e.g. shock formation) and shrinks
when the solution returns to a smooth regime.

Strategies
----------
* **Residual-based**: after each time step, check the SVD-tail energy
  at every bond; increase rank where it exceeds tolerance, decrease
  where it is far below.
* **Entropy-based**: monitor bond entanglement entropy; high entropy ⇒
  increase rank.
* **Gradient-based**: monitor temporal gradient ∂u/∂t in TT; steep
  gradients trigger local rank increases.

Key classes / functions
-----------------------
* :class:`DynamicRankConfig`  — strategy parameters
* :class:`DynamicRankState`   — current rank profile + diagnostics
* :func:`adapt_ranks`         — one adaptation step
* :func:`dynamic_rank_step`   — time-step + rank adaptation (wrapper)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from ontic.qtt.sparse_direct import tt_round


# ======================================================================
# Configuration
# ======================================================================

class RankStrategy(Enum):
    RESIDUAL = auto()
    ENTROPY = auto()
    GRADIENT = auto()


@dataclass
class DynamicRankConfig:
    """
    Configuration for dynamic rank adaptation.

    Attributes
    ----------
    strategy : RankStrategy
        Adaptation criterion.
    tol_increase : float
        Relative SVD-tail threshold above which rank increases.
    tol_decrease : float
        Relative SVD-tail threshold below which rank decreases.
    rank_min : int
        Minimum allowed bond dimension.
    rank_max : int
        Maximum allowed bond dimension.
    growth_step : int
        How much rank grows per adaptation.
    shrink_step : int
        How much rank shrinks per adaptation.
    check_interval : int
        Adapt every *check_interval* time steps.
    """
    strategy: RankStrategy = RankStrategy.RESIDUAL
    tol_increase: float = 1e-4
    tol_decrease: float = 1e-8
    rank_min: int = 2
    rank_max: int = 256
    growth_step: int = 2
    shrink_step: int = 1
    check_interval: int = 1


@dataclass
class DynamicRankState:
    """
    Current state of dynamic rank adaptation.

    Attributes
    ----------
    ranks : list[int]
        Bond dimensions at each bond.
    svd_tails : list[float]
        Relative tail energy at each bond.
    entropies : list[float]
        Bond entanglement entropy at each bond.
    step_count : int
        Number of time steps since last adaptation.
    history : list[list[int]]
        Rank history over time.
    """
    ranks: list[int]
    svd_tails: list[float] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)
    step_count: int = 0
    history: list[list[int]] = field(default_factory=list)


# ======================================================================
# Core adaptation logic
# ======================================================================

def _bond_diagnostics(cores: list[NDArray]) -> tuple[list[float], list[float]]:
    """
    Compute per-bond SVD tail energy and entanglement entropy.

    Returns (tail_energies, entropies) each of length N-1.
    """
    N = len(cores)
    tails: list[float] = []
    entropies: list[float] = []

    # Left-to-right QR to put in left-canonical form
    lc = [c.copy() for c in cores]
    for k in range(N - 1):
        r_l, d, r_r = lc[k].shape
        mat = lc[k].reshape(r_l * d, r_r)
        Q, R = np.linalg.qr(mat)
        new_r = Q.shape[1]
        lc[k] = Q.reshape(r_l, d, new_r)
        lc[k + 1] = np.einsum('ij,jkl->ikl', R, lc[k + 1])

    # Now right-to-left SVD to get spectra
    for k in range(N - 1, 0, -1):
        r_l, d, r_r = lc[k].shape
        mat = lc[k].reshape(r_l, d * r_r)
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)

        # Tail energy: fraction of total in discarded singular values
        total_sq = float(np.sum(S ** 2))
        if total_sq > 1e-30:
            # Current rank is r_l; tail = everything beyond current
            tails.append(0.0)  # Nothing discarded at current rank
            # But we can look at the distribution
            # Entropy
            p = S ** 2 / total_sq
            p = p[p > 1e-30]
            entropy = -float(np.sum(p * np.log(p)))
        else:
            tails.append(0.0)
            entropy = 0.0
        entropies.append(entropy)

        # Absorb into left
        lc[k] = (np.diag(S) @ Vh).reshape(len(S), d, r_r)
        lc[k - 1] = np.einsum('ijk,kl->ijl', lc[k - 1], U)

    tails.reverse()
    entropies.reverse()
    return tails, entropies


def adapt_ranks(
    cores: list[NDArray],
    config: DynamicRankConfig,
    state: Optional[DynamicRankState] = None,
) -> tuple[list[NDArray], DynamicRankState]:
    """
    Perform one rank adaptation step.

    Analyses the SVD spectrum at each bond and adjusts the per-bond
    rank limits up or down.

    Parameters
    ----------
    cores : list[NDArray]
        Current TT-vector cores.
    config : DynamicRankConfig
        Adaptation parameters.
    state : DynamicRankState, optional
        Previous state (for history tracking).

    Returns
    -------
    (new_cores, state)
    """
    N = len(cores)
    tails, entropies = _bond_diagnostics(cores)

    current_ranks = [cores[k].shape[2] for k in range(N - 1)]
    new_ranks = list(current_ranks)

    if config.strategy == RankStrategy.RESIDUAL:
        for i in range(len(current_ranks)):
            # Re-compute tail at this bond with potential rank increase
            r_l, d, r_r = cores[i].shape
            mat = cores[i].reshape(r_l * d, r_r)
            S = np.linalg.svd(mat, compute_uv=False)
            total_sq = float(np.sum(S ** 2))
            if total_sq > 1e-30 and len(S) > 1:
                # Check if last singular value is large (need more rank)
                tail_ratio = float(S[-1] ** 2 / total_sq)
                if tail_ratio > config.tol_increase:
                    new_ranks[i] = min(
                        current_ranks[i] + config.growth_step,
                        config.rank_max,
                    )
                elif tail_ratio < config.tol_decrease:
                    new_ranks[i] = max(
                        current_ranks[i] - config.shrink_step,
                        config.rank_min,
                    )

    elif config.strategy == RankStrategy.ENTROPY:
        for i in range(len(current_ranks)):
            if i < len(entropies):
                # High entropy → need more rank
                if entropies[i] > np.log(current_ranks[i]) * 0.9:
                    new_ranks[i] = min(
                        current_ranks[i] + config.growth_step,
                        config.rank_max,
                    )
                elif entropies[i] < np.log(max(current_ranks[i], 2)) * 0.3:
                    new_ranks[i] = max(
                        current_ranks[i] - config.shrink_step,
                        config.rank_min,
                    )

    elif config.strategy == RankStrategy.GRADIENT:
        # Use SVD tail as proxy for temporal gradient
        for i in range(len(current_ranks)):
            r_l, d, r_r = cores[i].shape
            mat = cores[i].reshape(r_l * d, r_r)
            S = np.linalg.svd(mat, compute_uv=False)
            if len(S) > 1:
                gradient_proxy = float(S[0] / (S[-1] + 1e-30))
                if gradient_proxy > 1.0 / config.tol_increase:
                    new_ranks[i] = min(
                        current_ranks[i] + config.growth_step,
                        config.rank_max,
                    )
                elif gradient_proxy < 1.0 / config.tol_decrease:
                    new_ranks[i] = max(
                        current_ranks[i] - config.shrink_step,
                        config.rank_min,
                    )

    # Apply new ranks
    max_rank = max(new_ranks) if new_ranks else config.rank_max
    new_cores = tt_round(cores, max_rank=max_rank)

    if state is None:
        state = DynamicRankState(ranks=new_ranks)
    else:
        state.ranks = new_ranks
        state.svd_tails = tails
        state.entropies = entropies
        state.step_count += 1

    state.history.append(list(new_ranks))
    return new_cores, state


# ======================================================================
# Combined time-step + rank adaptation
# ======================================================================

def dynamic_rank_step(
    cores: list[NDArray],
    rhs_fn: Callable[[list[NDArray]], list[NDArray]],
    dt: float,
    config: DynamicRankConfig,
    state: Optional[DynamicRankState] = None,
) -> tuple[list[NDArray], DynamicRankState]:
    """
    One time step with dynamic rank adaptation.

    Performs explicit Euler: ``u_{n+1} = u_n + dt * rhs(u_n)`` then
    adapts ranks according to *config*.

    Parameters
    ----------
    cores : list[NDArray]
        Current solution TT-cores.
    rhs_fn : callable
        Right-hand-side function ``f(u) → du/dt`` operating on TT-cores.
    dt : float
        Time step.
    config : DynamicRankConfig
        Adaptation parameters.
    state : DynamicRankState, optional
        Previous adaptation state.

    Returns
    -------
    (new_cores, state)
    """
    from ontic.qtt.eigensolvers import tt_axpy

    # Forward Euler step
    dudt = rhs_fn(cores)
    new_cores = tt_axpy(dt, dudt, cores, max_rank=config.rank_max)

    # Rank adaptation
    if state is None:
        state = DynamicRankState(ranks=[c.shape[2] for c in new_cores[:-1]])
    state.step_count += 1

    if state.step_count % config.check_interval == 0:
        new_cores, state = adapt_ranks(new_cores, config, state)

    return new_cores, state
