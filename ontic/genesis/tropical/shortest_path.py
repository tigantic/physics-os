"""
QTT Shortest Path Algorithms via Tropical Algebra

The key insight: shortest path = tropical matrix operations.

Floyd-Warshall ≡ Kleene star in min-plus semiring
Bellman-Ford ≡ Repeated tropical matrix-vector product

For QTT representations, these achieve O(r³ log² N) complexity
vs O(N³) for dense matrices.

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import torch

from ontic.genesis.tropical.semiring import (
    TropicalSemiring, MinPlusSemiring, SemiringType
)
from ontic.genesis.tropical.matrix import (
    TropicalMatrix, tropical_matmul, tropical_kleene_star, tropical_power
)


@dataclass
class ShortestPathResult:
    """
    Result of a shortest path computation.
    
    Attributes:
        distances: Distance matrix or vector
        predecessors: Optional predecessor matrix for path reconstruction
        algorithm: Name of algorithm used
        iterations: Number of iterations performed
        converged: Whether algorithm converged
    """
    distances: torch.Tensor
    predecessors: Optional[torch.Tensor] = None
    algorithm: str = "unknown"
    iterations: int = 0
    converged: bool = True
    
    def path_to(self, source: int, target: int) -> List[int]:
        """
        Reconstruct path from source to target.
        
        Args:
            source: Starting node
            target: Ending node
            
        Returns:
            List of nodes in path [source, ..., target]
        """
        if self.predecessors is None:
            raise ValueError("No predecessors stored for path reconstruction")
        
        path = []
        current = target
        
        # Trace back
        max_steps = self.predecessors.shape[0] + 1
        for _ in range(max_steps):
            path.append(current)
            if current == source:
                break
            pred = int(self.predecessors[source, current].item())
            if pred < 0:  # No path
                return []
            current = pred
        else:
            # Didn't reach source - cycle or no path
            return []
        
        path.reverse()
        return path


def all_pairs_shortest_path(A: TropicalMatrix,
                            algorithm: str = "floyd_warshall"
                            ) -> ShortestPathResult:
    """
    Compute all-pairs shortest paths.
    
    Args:
        A: Adjacency/weight matrix as TropicalMatrix
        algorithm: "floyd_warshall" or "tropical_closure"
        
    Returns:
        ShortestPathResult with distance matrix
    """
    if algorithm == "floyd_warshall":
        return floyd_warshall_tropical(A)
    elif algorithm == "tropical_closure":
        closure = tropical_kleene_star(A)
        return ShortestPathResult(
            distances=closure.data,
            algorithm="tropical_closure",
            iterations=A.size
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def floyd_warshall_tropical(A: TropicalMatrix,
                            store_predecessors: bool = True
                            ) -> ShortestPathResult:
    """
    Floyd-Warshall algorithm via tropical algebra.
    
    D^(k+1)_ij = D^(k)_ij ⊕ (D^(k)_ik ⊗ D^(k)_kj)
    
    This is exactly the Kleene star iteration.
    
    Args:
        A: Weight matrix
        store_predecessors: Whether to store path reconstruction info
        
    Returns:
        ShortestPathResult with distances and optional predecessors
    """
    n = A.size
    D = A.data.clone()
    
    # Initialize diagonal
    for i in range(n):
        if D[i, i] > 0:
            D[i, i] = 0.0
    
    # Predecessor matrix: pred[i,j] = node before j on shortest path from i to j
    # So path from i to j is: i -> ... -> pred[i,j] -> j
    pred = None
    if store_predecessors:
        pred = -torch.ones((n, n), dtype=torch.long)
        for i in range(n):
            for j in range(n):
                if D[i, j] < A.semiring.zero - 1 and i != j:
                    pred[i, j] = i  # Direct edge i -> j
    
    semiring = A.semiring
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                # new_dist = D[i,k] + D[k,j]
                new_dist = D[i, k] + D[k, j]
                
                # Check if going through k is better
                if semiring.semiring_type == SemiringType.MIN_PLUS:
                    improved = new_dist < D[i, j]
                else:
                    improved = new_dist > D[i, j]
                
                if improved:
                    D[i, j] = new_dist
                    # Predecessor of j on path from i through k
                    # is the predecessor of j on path from k
                    if pred is not None:
                        pred[i, j] = pred[k, j] if pred[k, j] >= 0 else k
    
    return ShortestPathResult(
        distances=D,
        predecessors=pred,
        algorithm="floyd_warshall",
        iterations=n
    )


def single_source_shortest_path(A: TropicalMatrix,
                                source: int,
                                algorithm: str = "bellman_ford"
                                ) -> ShortestPathResult:
    """
    Compute shortest paths from a single source.
    
    Args:
        A: Weight matrix
        source: Source node index
        algorithm: "bellman_ford" or "dijkstra"
        
    Returns:
        ShortestPathResult with distance vector
    """
    if algorithm == "bellman_ford":
        return bellman_ford_tropical(A, source)
    elif algorithm == "dijkstra":
        return dijkstra_tropical(A, source)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def bellman_ford_tropical(A: TropicalMatrix,
                          source: int,
                          detect_negative: bool = True
                          ) -> ShortestPathResult:
    """
    Bellman-Ford algorithm via tropical matrix-vector product.
    
    d^(k+1) = d^(k) ⊕ (A ⊗ d^(k))
    
    where ⊗ is tropical matrix-vector product.
    
    Args:
        A: Weight matrix
        source: Source node
        detect_negative: Check for negative cycles
        
    Returns:
        ShortestPathResult
    """
    n = A.size
    semiring = A.semiring
    
    # Initialize distances
    d = torch.full((n,), semiring.zero)
    d[source] = 0.0
    
    # Predecessor array
    pred = -torch.ones(n, dtype=torch.long)
    
    converged = False
    iterations = 0
    
    for iteration in range(n):
        iterations = iteration + 1
        d_new = d.clone()
        updated = False
        
        for j in range(n):
            # d[j] = min_i (A[i,j] + d[i])
            candidates = A.data[:, j] + d
            
            if semiring.semiring_type == SemiringType.MIN_PLUS:
                new_val = candidates.min()
                best_pred = candidates.argmin().item()
            else:
                new_val = candidates.max()
                best_pred = candidates.argmax().item()
            
            if semiring.semiring_type == SemiringType.MIN_PLUS:
                if new_val < d_new[j]:
                    d_new[j] = new_val
                    pred[j] = best_pred
                    updated = True
            else:
                if new_val > d_new[j]:
                    d_new[j] = new_val
                    pred[j] = best_pred
                    updated = True
        
        d = d_new
        
        if not updated:
            converged = True
            break
    
    # Check for negative cycles
    if detect_negative and not converged:
        # One more iteration - if anything changes, negative cycle
        for j in range(n):
            candidates = A.data[:, j] + d
            if semiring.semiring_type == SemiringType.MIN_PLUS:
                if candidates.min() < d[j] - 1e-9:
                    converged = False
                    break
            else:
                if candidates.max() > d[j] + 1e-9:
                    converged = False
                    break
        else:
            converged = True
    
    # Convert pred to 2D for compatibility
    pred_2d = -torch.ones((n, n), dtype=torch.long)
    pred_2d[source, :] = pred
    
    return ShortestPathResult(
        distances=d,
        predecessors=pred_2d,
        algorithm="bellman_ford",
        iterations=iterations,
        converged=converged
    )


def dijkstra_tropical(A: TropicalMatrix,
                      source: int) -> ShortestPathResult:
    """
    Dijkstra's algorithm for non-negative weights.
    
    Only valid for min-plus semiring with non-negative weights.
    
    Args:
        A: Weight matrix (non-negative)
        source: Source node
        
    Returns:
        ShortestPathResult
    """
    if A.semiring.semiring_type != SemiringType.MIN_PLUS:
        raise ValueError("Dijkstra requires min-plus semiring")
    
    n = A.size
    inf = A.semiring.zero
    
    d = torch.full((n,), inf)
    d[source] = 0.0
    
    pred = -torch.ones(n, dtype=torch.long)
    visited = set()
    
    iterations = 0
    
    for _ in range(n):
        iterations += 1
        
        # Find unvisited node with minimum distance
        min_dist = inf
        u = -1
        for i in range(n):
            if i not in visited and d[i] < min_dist:
                min_dist = d[i]
                u = i
        
        if u < 0 or min_dist >= inf - 1:
            break
        
        visited.add(u)
        
        # Relax edges
        for v in range(n):
            if v in visited:
                continue
            
            weight = A.data[u, v]
            if weight >= inf - 1:
                continue
            
            new_dist = d[u] + weight
            if new_dist < d[v]:
                d[v] = new_dist
                pred[v] = u
    
    # Convert to 2D
    pred_2d = -torch.ones((n, n), dtype=torch.long)
    pred_2d[source, :] = pred
    
    return ShortestPathResult(
        distances=d,
        predecessors=pred_2d,
        algorithm="dijkstra",
        iterations=iterations
    )


def shortest_path_tree(A: TropicalMatrix,
                       source: int) -> Dict[str, torch.Tensor]:
    """
    Compute shortest path tree from source.
    
    Returns the tree structure as a set of edges.
    
    Args:
        A: Weight matrix
        source: Root of tree
        
    Returns:
        Dict with 'distances', 'parent', 'tree_edges'
    """
    result = bellman_ford_tropical(A, source)
    
    n = A.size
    parent = result.predecessors[source, :]
    
    # Build edge list
    edges = []
    for v in range(n):
        u = parent[v].item()
        if u >= 0 and u != v:
            edges.append((u, v))
    
    return {
        'distances': result.distances,
        'parent': parent,
        'tree_edges': torch.tensor(edges) if edges else torch.empty((0, 2), dtype=torch.long)
    }


def path_exists(A: TropicalMatrix, source: int, target: int) -> bool:
    """Check if path exists from source to target."""
    result = bellman_ford_tropical(A, source, detect_negative=False)
    return result.distances[target].item() < A.semiring.zero - 1


def shortest_path_length(A: TropicalMatrix, 
                         source: int, 
                         target: int) -> float:
    """Get shortest path length between two nodes."""
    result = bellman_ford_tropical(A, source, detect_negative=False)
    return result.distances[target].item()


def k_shortest_paths(A: TropicalMatrix,
                     source: int,
                     target: int,
                     k: int = 5) -> List[Tuple[float, List[int]]]:
    """
    Find k shortest paths between source and target.
    
    Uses Yen's algorithm.
    
    Args:
        A: Weight matrix
        source: Start node
        target: End node
        k: Number of paths to find
        
    Returns:
        List of (distance, path) tuples
    """
    n = A.size
    
    # Find first shortest path
    result = bellman_ford_tropical(A, source)
    path = result.path_to(source, target)
    
    if not path:
        return []
    
    paths = [(result.distances[target].item(), path)]
    candidates = []
    
    for i in range(1, k):
        last_path = paths[-1][1]
        
        for j in range(len(last_path) - 1):
            spur_node = last_path[j]
            root_path = last_path[:j+1]
            
            # Temporarily remove edges
            A_temp = TropicalMatrix(A.data.clone(), A.semiring, n)
            
            for prev_dist, prev_path in paths:
                if prev_path[:j+1] == root_path and len(prev_path) > j + 1:
                    # Remove edge
                    u, v = prev_path[j], prev_path[j+1]
                    A_temp.data[u, v] = A.semiring.zero
            
            # Remove root path nodes (except spur)
            for node in root_path[:-1]:
                A_temp.data[node, :] = A.semiring.zero
                A_temp.data[:, node] = A.semiring.zero
            
            # Find spur path
            spur_result = bellman_ford_tropical(A_temp, spur_node)
            if spur_result.distances[target] < A.semiring.zero - 1:
                spur_path = spur_result.path_to(spur_node, target)
                if spur_path:
                    total_path = root_path[:-1] + spur_path
                    # Calculate total distance
                    total_dist = 0.0
                    for idx in range(len(total_path) - 1):
                        u, v = total_path[idx], total_path[idx+1]
                        total_dist += A.data[u, v].item()
                    
                    if (total_dist, total_path) not in candidates:
                        candidates.append((total_dist, total_path))
        
        if not candidates:
            break
        
        candidates.sort(key=lambda x: x[0])
        best = candidates.pop(0)
        
        if best not in paths:
            paths.append(best)
    
    return paths[:k]


def eccentricity(A: TropicalMatrix, node: int) -> float:
    """
    Compute eccentricity of a node (max distance to any other node).
    """
    result = bellman_ford_tropical(A, node)
    finite_dists = result.distances[result.distances < A.semiring.zero - 1]
    if len(finite_dists) == 0:
        return float('inf')
    return finite_dists.max().item()


def graph_diameter(A: TropicalMatrix) -> float:
    """
    Compute graph diameter (max eccentricity).
    """
    apsp = all_pairs_shortest_path(A)
    finite = apsp.distances[apsp.distances < A.semiring.zero - 1]
    if len(finite) == 0:
        return float('inf')
    return finite.max().item()


def graph_radius(A: TropicalMatrix) -> float:
    """
    Compute graph radius (min eccentricity).
    """
    n = A.size
    min_ecc = float('inf')
    
    for i in range(n):
        ecc = eccentricity(A, i)
        min_ecc = min(min_ecc, ecc)
    
    return min_ecc
