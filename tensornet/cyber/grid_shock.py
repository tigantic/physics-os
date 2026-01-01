"""
Cyber Grid Shock - DDoS Attack Visualization as Fluid Dynamics

The Physics of Network Attacks

A computer network is isomorphic to a pipe network:
- Nodes = Servers (pressure reservoirs)
- Edges = Links (pipes with bandwidth capacity)
- Packets = Fluid particles
- Bandwidth = Pipe diameter
- Congestion = Pressure buildup

A DDoS attack is equivalent to opening an infinite pressure source:
- Attacker injects massive "fluid" at one node
- Pressure wave propagates through the network
- When pressure exceeds capacity: NODE FAILURE
- Failures cascade as traffic reroutes

This module visualizes network attacks using the Heat Equation
on graphs - the same math used for thermal diffusion.

∂u/∂t = α ∇²u  (Diffusion on a graph via the Laplacian)

Reference: Network science meets fluid dynamics in this
physics-informed approach to cybersecurity visualization.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Set
import torch
import numpy as np

# NetworkX for graph operations (standard in data science)
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("[WARN] NetworkX not installed. Using fallback graph.")


@dataclass
class CascadeReport:
    """
    Report on cascade failure from network attack.
    """
    attack_source: int
    attack_target: int
    
    total_nodes: int
    failed_nodes: int
    surviving_nodes: int
    
    peak_pressure: float
    peak_node: int
    
    time_to_cascade: int  # Steps until 50% failure
    cascade_occurred: bool
    
    failure_sequence: List[int]  # Order of node failures
    
    status: str
    recommendation: str
    
    def __str__(self) -> str:
        cascade_status = "⛔ CASCADE FAILURE" if self.cascade_occurred else "✅ CONTAINED"
        return (
            f"[CASCADE REPORT]\n"
            f"  Attack: Node {self.attack_source} → Node {self.attack_target}\n"
            f"  Nodes Failed: {self.failed_nodes}/{self.total_nodes} "
            f"({100*self.failed_nodes/self.total_nodes:.1f}%)\n"
            f"  Peak Pressure: {self.peak_pressure:.1f}% at Node {self.peak_node}\n"
            f"  Time to Cascade: {self.time_to_cascade} steps\n"
            f"  Status: {cascade_status}\n"
            f"  First 5 Failures: {self.failure_sequence[:5]}\n"
            f"  Recommendation: {self.recommendation}"
        )


@dataclass
class AttackSimulation:
    """
    Results of attack simulation.
    """
    pressure_history: torch.Tensor  # [steps, nodes]
    failed_nodes: Set[int]
    cascade_report: CascadeReport


class CyberGrid:
    """
    Cyber Network as a Fluid Grid.
    
    Models network traffic as fluid flow:
    - Each node has a "pressure" (packet queue depth)
    - Edges allow flow proportional to pressure difference
    - Capacity limits cause "explosions" (node failure)
    
    Attack modeling:
    - DDoS: Inject infinite pressure at attacker node
    - Cascade: Failed nodes redirect traffic, overloading neighbors
    """
    
    def __init__(
        self,
        num_nodes: int = 50,
        connection_probability: float = 0.1,
        base_capacity: float = 100.0,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize cyber grid.
        
        Args:
            num_nodes: Number of servers/routers
            connection_probability: Erdős-Rényi edge probability
            base_capacity: Default bandwidth capacity per node
            device: Torch device
        """
        self.num_nodes = num_nodes
        self.base_capacity = base_capacity
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Build network graph
        if HAS_NETWORKX:
            self.graph = nx.erdos_renyi_graph(num_nodes, connection_probability)
            # Ensure connected
            if not nx.is_connected(self.graph):
                # Add edges to connect components
                components = list(nx.connected_components(self.graph))
                for i in range(len(components) - 1):
                    u = list(components[i])[0]
                    v = list(components[i + 1])[0]
                    self.graph.add_edge(u, v)
        else:
            # Fallback: simple ring + random edges
            self.graph = self._create_fallback_graph(num_nodes, connection_probability)
        
        # Build adjacency data for GPU
        self.edges = list(self.graph.edges())
        self.num_edges = len(self.edges)
        
        # Edge tensor for fast GPU access
        self.edge_tensor = torch.tensor(
            [[e[0], e[1]] for e in self.edges],
            device=self.device,
            dtype=torch.long
        )
        
        # State: Pressure (packet load) at each node
        self.pressure = torch.zeros(num_nodes, device=self.device, dtype=torch.float64)
        
        # Capacity: Maximum pressure before failure
        # Add some variation (seeded for reproducibility)
        torch.manual_seed(42)  # Per Article III, Section 3.2
        np.random.seed(42)
        self.capacity = torch.ones(num_nodes, device=self.device, dtype=torch.float64) * base_capacity
        self.capacity += torch.randn(num_nodes, device=self.device, dtype=torch.float64) * (base_capacity * 0.2)
        self.capacity = torch.clamp(self.capacity, min=base_capacity * 0.5)
        
        # Node status: 1 = alive, 0 = failed
        self.alive = torch.ones(num_nodes, device=self.device)
        
        # Track failure order
        self.failure_sequence: List[int] = []
        
        avg_degree = 2 * self.num_edges / num_nodes
        print(f"[CYBER] Grid initialized: {num_nodes} nodes, {self.num_edges} edges")
        print(f"[CYBER] Average degree: {avg_degree:.2f}")
        print(f"[CYBER] Base capacity: {base_capacity}")
        print(f"[CYBER] Device: {self.device}")
    
    def _create_fallback_graph(self, n: int, p: float):
        """Create simple graph without NetworkX."""
        class SimpleGraph:
            def __init__(self):
                self.edges_set = set()
                self.nodes = set()
            
            def add_edge(self, u, v):
                self.edges_set.add((min(u, v), max(u, v)))
                self.nodes.add(u)
                self.nodes.add(v)
            
            def edges(self):
                return list(self.edges_set)
        
        g = SimpleGraph()
        
        # Ring for connectivity
        for i in range(n):
            g.add_edge(i, (i + 1) % n)
        
        # Random edges
        for i in range(n):
            for j in range(i + 2, n):
                if np.random.random() < p:
                    g.add_edge(i, j)
        
        return g
    
    def reset(self):
        """Reset grid to initial state."""
        self.pressure = torch.zeros(self.num_nodes, device=self.device)
        self.alive = torch.ones(self.num_nodes, device=self.device)
        self.failure_sequence = []
    
    def simulate_attack(
        self,
        attacker_node: int = 0,
        target_node: int = 10,
        attack_rate: float = 1000.0,
        diffusion_rate: float = 0.1,
        steps: int = 50,
        verbose: bool = True,
    ) -> AttackSimulation:
        """
        Simulate a DDoS attack as fluid injection.
        
        The attack injects "pressure" at the attacker node,
        which diffuses through the network. When any node's
        pressure exceeds capacity, it fails and stops routing.
        
        Args:
            attacker_node: Source of attack traffic
            target_node: Intended target (for reporting)
            attack_rate: Pressure injected per step
            diffusion_rate: How fast pressure spreads (0-1)
            steps: Simulation duration
            verbose: Print progress
            
        Returns:
            AttackSimulation with full history and report
        """
        self.reset()
        
        if verbose:
            print(f"\n[CYBER] DDoS Attack Simulation")
            print(f"[CYBER] Attacker: Node {attacker_node}")
            print(f"[CYBER] Target: Node {target_node}")
            print(f"[CYBER] Attack rate: {attack_rate} units/step")
            print()
        
        pressure_history = []
        time_to_cascade = steps  # Default if no cascade
        
        for t in range(steps):
            # ========================================
            # 1. INJECT ATTACK TRAFFIC
            # ========================================
            if self.alive[attacker_node] > 0.5:
                self.pressure[attacker_node] += attack_rate
            
            # ========================================
            # 2. DIFFUSION (Heat equation on graph)
            # ========================================
            # Traffic flows from high pressure to low pressure
            # This is Laplacian smoothing on the graph
            
            new_pressure = self.pressure.clone()
            
            for u, v in self.edges:
                # Only flow through alive nodes
                if self.alive[u] < 0.5 or self.alive[v] < 0.5:
                    continue
                
                # Flow proportional to pressure difference
                diff = self.pressure[v] - self.pressure[u]
                flow = diff * diffusion_rate
                
                new_pressure[u] += flow
                new_pressure[v] -= flow
            
            # Ensure non-negative
            self.pressure = torch.clamp(new_pressure, min=0)
            
            # ========================================
            # 3. CHECK FOR OVERLOAD (Node Failure)
            # ========================================
            overloaded = (self.pressure > self.capacity) & (self.alive > 0.5)
            
            # Mark overloaded nodes as failed
            newly_failed = overloaded.nonzero(as_tuple=True)[0].tolist()
            
            for node in newly_failed:
                if node not in self.failure_sequence:
                    self.failure_sequence.append(node)
                    self.alive[node] = 0.0
                    # Failed nodes dump pressure to neighbors
                    # (simulating traffic rerouting)
                    self.pressure[node] = 0.0
            
            # Track pressure history
            pressure_history.append(self.pressure.cpu().clone())
            
            # ========================================
            # 4. PROGRESS REPORTING
            # ========================================
            if verbose and t % 10 == 0:
                max_p = self.pressure.max().item()
                victim = self.pressure.argmax().item()
                alive_count = self.alive.sum().item()
                failed_count = self.num_nodes - alive_count
                
                print(f"   T={t:3d}: Max Load {max_p:6.0f}% at Node {victim:2d} | "
                      f"Failed: {failed_count:.0f}/{self.num_nodes}")
            
            # Check for cascade
            failed_ratio = 1.0 - (self.alive.sum().item() / self.num_nodes)
            if failed_ratio >= 0.5 and time_to_cascade == steps:
                time_to_cascade = t
                if verbose:
                    print(f"\n   [ALERT] CASCADE FAILURE at T={t}. "
                          f"50% of network down!")
        
        # ========================================
        # 5. GENERATE REPORT
        # ========================================
        failed_count = len(self.failure_sequence)
        cascade_occurred = failed_count >= self.num_nodes // 2
        
        peak_pressure = max(h.max().item() for h in pressure_history)
        peak_node = pressure_history[-1].argmax().item()
        
        # Recommendation based on attack success
        if cascade_occurred:
            recommendation = "CRITICAL: Implement rate limiting at edge nodes. " \
                           "Add redundant paths. Consider BGP blackholing."
        elif failed_count > 0:
            recommendation = f"PARTIAL DAMAGE: {failed_count} nodes failed. " \
                           "Increase capacity or add load balancers."
        else:
            recommendation = "ATTACK MITIGATED: Network absorbed the attack. " \
                           "Continue monitoring."
        
        report = CascadeReport(
            attack_source=attacker_node,
            attack_target=target_node,
            total_nodes=self.num_nodes,
            failed_nodes=failed_count,
            surviving_nodes=self.num_nodes - failed_count,
            peak_pressure=peak_pressure,
            peak_node=peak_node,
            time_to_cascade=time_to_cascade,
            cascade_occurred=cascade_occurred,
            failure_sequence=self.failure_sequence.copy(),
            status="CASCADE" if cascade_occurred else "CONTAINED",
            recommendation=recommendation,
        )
        
        if verbose:
            print()
            print(report)
        
        return AttackSimulation(
            pressure_history=torch.stack(pressure_history),
            failed_nodes=set(self.failure_sequence),
            cascade_report=report,
        )
    
    def find_critical_nodes(self, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Find the most critical nodes (highest impact if attacked).
        
        Uses betweenness centrality - nodes that bridge many paths.
        """
        if not HAS_NETWORKX:
            # Fallback: use degree
            degrees = [0] * self.num_nodes
            for u, v in self.edges:
                degrees[u] += 1
                degrees[v] += 1
            ranked = sorted(enumerate(degrees), key=lambda x: x[1], reverse=True)
            return ranked[:top_k]
        
        centrality = nx.betweenness_centrality(self.graph)
        ranked = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
    
    def simulate_defense(
        self,
        attacker_node: int,
        defense_nodes: List[int],
        attack_rate: float = 1000.0,
        steps: int = 50,
    ) -> AttackSimulation:
        """
        Simulate attack with defensive rate limiting.
        
        Defense nodes have increased capacity (like adding CDN/DDoS protection).
        """
        # Boost capacity of defense nodes
        original_capacity = self.capacity.clone()
        
        for node in defense_nodes:
            self.capacity[node] *= 5.0  # 5× capacity boost
        
        print(f"[DEFENSE] Hardened nodes: {defense_nodes}")
        
        result = self.simulate_attack(
            attacker_node=attacker_node,
            attack_rate=attack_rate,
            steps=steps,
            verbose=True,
        )
        
        # Restore original capacity
        self.capacity = original_capacity
        
        return result


def run_attack_demo():
    """
    Full demonstration of cyber attack visualization.
    """
    print("=" * 70)
    print("GRID SHOCK: DDoS Attack Visualization")
    print("Phase 10: Cybersecurity as Fluid Dynamics")
    print("=" * 70)
    print()
    
    # Create network
    grid = CyberGrid(num_nodes=50, connection_probability=0.1)
    
    print()
    
    # Find critical nodes
    print("[RECON] Identifying critical infrastructure...")
    critical = grid.find_critical_nodes(top_k=5)
    print("Most critical nodes (highest betweenness centrality):")
    for node, score in critical:
        print(f"  Node {node}: Centrality = {score:.4f}")
    
    print()
    
    # Simulate attack on critical node
    target = critical[0][0]  # Most critical node
    
    sim = grid.simulate_attack(
        attacker_node=0,
        target_node=target,
        attack_rate=500.0,
        steps=50,
    )
    
    return sim


if __name__ == "__main__":
    run_attack_demo()
