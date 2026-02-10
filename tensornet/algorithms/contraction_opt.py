"""
Tensor Network Contraction Optimisation
=========================================

Optimal and near-optimal contraction-order search for arbitrary tensor
networks, going beyond ``opt_einsum`` with:

* **Exhaustive dynamic-programming** search (exact optimal for small nets)
* **Greedy + local search** heuristic for large networks
* **Random-greedy** with restarts
* **Cost model** based on FLOPs and memory

Key classes / functions
-----------------------
* :class:`TNGraph`            — tensor-network graph
* :class:`ContractionPlan`    — ordered list of pairwise contractions
* :func:`optimal_order`       — exact DP (exponential in #tensors)
* :func:`greedy_order`        — O(n² log n) heuristic
* :func:`random_greedy_order` — best of *k* random greedy trials
* :func:`estimate_cost`       — FLOP estimate for a contraction plan
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import FrozenSet, Optional, Sequence

import numpy as np


# ======================================================================
# Data structures
# ======================================================================

@dataclass(frozen=True)
class Edge:
    """A contraction index shared by two or more tensors."""
    name: str
    dim: int  # Dimension of this index


@dataclass
class TensorNode:
    """
    A single tensor in the network.

    Attributes
    ----------
    id : int
        Unique identifier.
    indices : list[str]
        Names of the indices carried by this tensor.
    shape : tuple[int, ...]
        Shape of the tensor.
    """
    id: int
    indices: list[str]
    shape: tuple[int, ...]

    @property
    def size(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 1


@dataclass
class TNGraph:
    """
    Graph representation of a tensor network.

    Attributes
    ----------
    nodes : list[TensorNode]
        Tensor nodes.
    index_dims : dict[str, int]
        Dimension of each named index.
    output_indices : list[str]
        Indices that remain un-contracted (open legs).
    """
    nodes: list[TensorNode]
    index_dims: dict[str, int]
    output_indices: list[str] = field(default_factory=list)

    @property
    def n_tensors(self) -> int:
        return len(self.nodes)


@dataclass
class ContractionStep:
    """A single pairwise contraction."""
    left: int          # ID of left tensor
    right: int         # ID of right tensor
    result_id: int     # ID assigned to the result
    contracted: list[str]  # Indices summed over
    cost_flops: float  # Estimated FLOPs


@dataclass
class ContractionPlan:
    """Ordered sequence of pairwise contractions."""
    steps: list[ContractionStep]
    total_flops: float
    peak_memory: float  # In elements


# ======================================================================
# Cost estimation
# ======================================================================

def _pairwise_cost(
    shape_left: dict[str, int],
    shape_right: dict[str, int],
    contracted: set[str],
) -> float:
    """FLOP cost of contracting two tensors."""
    all_indices = set(shape_left) | set(shape_right)
    dims = {}
    for idx in all_indices:
        dims[idx] = shape_left.get(idx, 1) * shape_right.get(idx, 1)
        if idx in shape_left and idx in shape_right:
            dims[idx] = shape_left[idx]  # They should agree

    # Collect all dimensions properly
    total = 1
    for idx in all_indices:
        d = shape_left.get(idx, None) or shape_right.get(idx, None) or 1
        total *= d

    return float(total)


def _result_shape(
    shape_left: dict[str, int],
    shape_right: dict[str, int],
    contracted: set[str],
) -> dict[str, int]:
    """Shape of the result after pairwise contraction."""
    result: dict[str, int] = {}
    for idx in set(shape_left) | set(shape_right):
        if idx not in contracted:
            result[idx] = shape_left.get(idx, None) or shape_right.get(idx, None) or 1
    return result


def estimate_cost(plan: ContractionPlan) -> float:
    """Total FLOP count for a contraction plan."""
    return plan.total_flops


# ======================================================================
# Greedy contraction order
# ======================================================================

def greedy_order(graph: TNGraph) -> ContractionPlan:
    """
    Greedy contraction order: at each step contract the pair with
    minimum intermediate size.

    Complexity: :math:`O(n^2 \\log n)`.

    Parameters
    ----------
    graph : TNGraph
        Tensor network graph.

    Returns
    -------
    ContractionPlan
    """
    # Build shape dicts
    shapes: dict[int, dict[str, int]] = {}
    for node in graph.nodes:
        shapes[node.id] = {
            idx: node.shape[i] for i, idx in enumerate(node.indices)
        }

    output_set = set(graph.output_indices)
    active: set[int] = {n.id for n in graph.nodes}
    next_id = max(active) + 1
    steps: list[ContractionStep] = []
    total_flops = 0.0
    peak_mem = 0.0

    while len(active) > 1:
        # Find best pair
        best_pair = None
        best_cost = np.inf
        best_contracted: set[str] = set()

        active_list = sorted(active)
        for i in range(len(active_list)):
            for j in range(i + 1, len(active_list)):
                a, b = active_list[i], active_list[j]
                shared = set(shapes[a]) & set(shapes[b])
                contracted = shared - output_set
                if not contracted and not shared:
                    # No shared indices — skip (outer product unless last)
                    if len(active) > 2:
                        continue
                cost = _pairwise_cost(shapes[a], shapes[b], contracted)
                if cost < best_cost:
                    best_cost = cost
                    best_pair = (a, b)
                    best_contracted = contracted

        if best_pair is None:
            # Force outer product of first two
            active_list = sorted(active)
            best_pair = (active_list[0], active_list[1])
            best_contracted = set()
            best_cost = _pairwise_cost(
                shapes[best_pair[0]], shapes[best_pair[1]], set()
            )

        a, b = best_pair
        res_shape = _result_shape(shapes[a], shapes[b], best_contracted)

        steps.append(ContractionStep(
            left=a,
            right=b,
            result_id=next_id,
            contracted=sorted(best_contracted),
            cost_flops=best_cost,
        ))
        total_flops += best_cost

        result_size = float(np.prod(list(res_shape.values()))) if res_shape else 1.0
        peak_mem = max(peak_mem, result_size)

        shapes[next_id] = res_shape
        active.discard(a)
        active.discard(b)
        active.add(next_id)
        next_id += 1

    return ContractionPlan(
        steps=steps,
        total_flops=total_flops,
        peak_memory=peak_mem,
    )


# ======================================================================
# Optimal contraction order (exact DP — small networks only)
# ======================================================================

def optimal_order(graph: TNGraph) -> ContractionPlan:
    """
    Exact optimal contraction order via dynamic programming.

    Exponential in the number of tensors — practical for ≤ ~15 tensors.

    Parameters
    ----------
    graph : TNGraph
        Tensor network graph.

    Returns
    -------
    ContractionPlan
        Plan with minimum total FLOP count.
    """
    n = graph.n_tensors
    if n > 20:
        raise ValueError(
            f"optimal_order is exponential; refusing n={n} > 20. "
            "Use greedy_order or random_greedy_order instead."
        )

    shapes: dict[int, dict[str, int]] = {}
    for node in graph.nodes:
        shapes[node.id] = {
            idx: node.shape[i] for i, idx in enumerate(node.indices)
        }

    output_set = set(graph.output_indices)
    ids = frozenset(n.id for n in graph.nodes)

    # DP cache: subset → (min_cost, best_split, result_shape)
    cache: dict[FrozenSet[int], tuple[float, Optional[tuple], dict[str, int]]] = {}

    # Base cases: single tensors
    for nid in ids:
        cache[frozenset([nid])] = (0.0, None, shapes[nid])

    # Fill DP for subsets of increasing size
    id_list = sorted(ids)
    for size in range(2, n + 1):
        for subset_tuple in itertools.combinations(id_list, size):
            subset = frozenset(subset_tuple)
            best_cost = np.inf
            best_split: Optional[tuple] = None
            best_shape: dict[str, int] = {}

            # Try all non-trivial splits
            subset_list = sorted(subset)
            for r in range(1, size):
                for left_tuple in itertools.combinations(subset_list, r):
                    left = frozenset(left_tuple)
                    right = subset - left
                    if not right or left not in cache or right not in cache:
                        continue

                    cost_l, _, shape_l = cache[left]
                    cost_r, _, shape_r = cache[right]

                    shared = set(shape_l) & set(shape_r)
                    contracted = shared - output_set
                    pair_cost = _pairwise_cost(shape_l, shape_r, contracted)
                    total = cost_l + cost_r + pair_cost

                    if total < best_cost:
                        best_cost = total
                        best_split = (left, right, contracted)
                        best_shape = _result_shape(shape_l, shape_r, contracted)

            if best_split is not None:
                cache[subset] = (best_cost, best_split, best_shape)

    # Reconstruct plan
    steps: list[ContractionStep] = []
    next_id = max(ids) + 1
    subset_to_id: dict[FrozenSet[int], int] = {
        frozenset([nid]): nid for nid in ids
    }

    def _reconstruct(subset: FrozenSet[int]) -> int:
        nonlocal next_id
        if len(subset) == 1:
            return next(iter(subset))
        _, split, _ = cache[subset]
        if split is None:
            return next(iter(subset))
        left_set, right_set, contracted = split
        left_id = _reconstruct(left_set)
        right_id = _reconstruct(right_set)
        result_id = next_id
        next_id += 1
        pair_cost = _pairwise_cost(
            cache.get(left_set, (0, None, {}))[2],
            cache.get(right_set, (0, None, {}))[2],
            contracted,
        )
        steps.append(ContractionStep(
            left=left_id,
            right=right_id,
            result_id=result_id,
            contracted=sorted(contracted),
            cost_flops=pair_cost,
        ))
        subset_to_id[subset] = result_id
        return result_id

    _reconstruct(ids)
    total_flops = cache.get(ids, (0.0, None, {}))[0]

    return ContractionPlan(
        steps=steps,
        total_flops=total_flops,
        peak_memory=0.0,  # Not tracked in DP
    )


# ======================================================================
# Random-greedy with restarts
# ======================================================================

def random_greedy_order(
    graph: TNGraph,
    n_trials: int = 64,
    temperature: float = 1.0,
    seed: Optional[int] = None,
) -> ContractionPlan:
    """
    Random-greedy contraction order: run *n_trials* randomised greedy
    searches and return the best.

    At each step, instead of always picking the cheapest pair, sample
    pairs with probability ∝ exp(-cost / temperature).

    Parameters
    ----------
    graph : TNGraph
        Tensor network.
    n_trials : int
        Number of random restarts.
    temperature : float
        Boltzmann temperature for pair selection.
    seed : int, optional
        RNG seed.

    Returns
    -------
    ContractionPlan
        Best plan found.
    """
    rng = np.random.default_rng(seed)
    best_plan: Optional[ContractionPlan] = None

    for _ in range(n_trials):
        shapes: dict[int, dict[str, int]] = {}
        for node in graph.nodes:
            shapes[node.id] = {
                idx: node.shape[i] for i, idx in enumerate(node.indices)
            }

        output_set = set(graph.output_indices)
        active: set[int] = {n.id for n in graph.nodes}
        next_id = max(active) + 1
        steps: list[ContractionStep] = []
        total_flops = 0.0
        peak_mem = 0.0

        while len(active) > 1:
            candidates: list[tuple[int, int, set[str], float]] = []
            active_list = sorted(active)
            for i in range(len(active_list)):
                for j in range(i + 1, len(active_list)):
                    a, b = active_list[i], active_list[j]
                    shared = set(shapes[a]) & set(shapes[b])
                    contracted = shared - output_set
                    cost = _pairwise_cost(shapes[a], shapes[b], contracted)
                    candidates.append((a, b, contracted, cost))

            if not candidates:
                break

            # Boltzmann sampling
            costs = np.array([c[3] for c in candidates])
            log_weights = -costs / max(temperature, 1e-30)
            log_weights -= np.max(log_weights)  # Numerical stability
            weights = np.exp(log_weights)
            weights /= weights.sum() + 1e-30
            choice = rng.choice(len(candidates), p=weights)
            a, b, contracted, cost = candidates[choice]

            res_shape = _result_shape(shapes[a], shapes[b], contracted)
            steps.append(ContractionStep(
                left=a, right=b, result_id=next_id,
                contracted=sorted(contracted), cost_flops=cost,
            ))
            total_flops += cost
            result_size = float(np.prod(list(res_shape.values()))) if res_shape else 1.0
            peak_mem = max(peak_mem, result_size)

            shapes[next_id] = res_shape
            active.discard(a)
            active.discard(b)
            active.add(next_id)
            next_id += 1

        plan = ContractionPlan(
            steps=steps, total_flops=total_flops, peak_memory=peak_mem,
        )
        if best_plan is None or plan.total_flops < best_plan.total_flops:
            best_plan = plan

    assert best_plan is not None
    return best_plan
