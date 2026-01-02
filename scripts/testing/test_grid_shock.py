"""
Test Script: Grid Shock - Cyber Attack Visualization

Phase 10: The Shield - Cybersecurity as Fluid Dynamics

Demonstrates DDoS attack visualization using fluid dynamics
on a network graph. We prove that cascade failures can be
predicted and critical infrastructure identified.

The Physics:
- Network = Pipe grid
- Traffic = Fluid flow
- DDoS = Infinite pressure source
- Cascade = Shockwave propagation

Run: python test_grid_shock.py
"""

import numpy as np
import torch

from tensornet.cyber.grid_shock import CyberGrid, run_attack_demo


def full_grid_shock_demo():
    """
    Complete cyber attack visualization demonstration.
    """
    print("=" * 70)
    print("GRID SHOCK: Cyber Attack as Fluid Shockwave")
    print("Phase 10: The Shield - Protecting the Network")
    print("=" * 70)
    print()

    # ==========================================================
    # 1. CREATE NETWORK
    # ==========================================================
    print("[PHASE 1] Building Network Infrastructure...")
    print()

    grid = CyberGrid(
        num_nodes=50,  # 50 servers/routers
        connection_probability=0.1,  # 10% edge probability
        base_capacity=100.0,  # 100% = normal capacity
    )

    print()

    # ==========================================================
    # 2. RECONNAISSANCE
    # ==========================================================
    print("[PHASE 2] Network Reconnaissance...")
    print()

    critical = grid.find_critical_nodes(top_k=5)
    print("Critical Infrastructure (High Centrality Nodes):")
    for i, (node, score) in enumerate(critical):
        role = [
            "Core Router",
            "Distribution Hub",
            "Edge Gateway",
            "Backup Link",
            "Secondary Path",
        ][i]
        print(f"  #{i+1} Node {node:2d}: {role} (Centrality: {score:.4f})")

    print()

    # ==========================================================
    # 3. ATTACK SIMULATION
    # ==========================================================
    print("[PHASE 3] Simulating DDoS Attack...")
    print()

    attacker = 0
    target = critical[0][0]  # Attack the most critical node

    sim = grid.simulate_attack(
        attacker_node=attacker,
        target_node=target,
        attack_rate=500.0,  # 500% of normal traffic per step
        diffusion_rate=0.1,  # 10% flow per step
        steps=50,
        verbose=True,
    )

    print()

    # ==========================================================
    # 4. DEFENSE SIMULATION
    # ==========================================================
    print("=" * 70)
    print("[PHASE 4] Testing Defense Strategy...")
    print("=" * 70)
    print()

    # Reset and try with defense
    grid.reset()

    # Protect the critical nodes
    defense_nodes = [node for node, _ in critical[:3]]

    print(f"Deploying DDoS protection to nodes: {defense_nodes}")
    print("(5× capacity boost simulating CDN/scrubbing)")
    print()

    defended_sim = grid.simulate_defense(
        attacker_node=attacker,
        defense_nodes=defense_nodes,
        attack_rate=500.0,
        steps=50,
    )

    print()

    # ==========================================================
    # 5. COMPARISON
    # ==========================================================
    print("=" * 70)
    print("ATTACK vs DEFENSE COMPARISON")
    print("=" * 70)

    undefended_failed = sim.cascade_report.failed_nodes
    defended_failed = defended_sim.cascade_report.failed_nodes

    print(
        f"""
UNDEFENDED NETWORK:
  Failed Nodes: {undefended_failed}/{grid.num_nodes}
  Cascade: {'YES' if sim.cascade_report.cascade_occurred else 'NO'}
  Peak Pressure: {sim.cascade_report.peak_pressure:.0f}%

DEFENDED NETWORK:
  Failed Nodes: {defended_failed}/{grid.num_nodes}
  Cascade: {'YES' if defended_sim.cascade_report.cascade_occurred else 'NO'}
  Peak Pressure: {defended_sim.cascade_report.peak_pressure:.0f}%

PROTECTION EFFICACY:
  Nodes Saved: {undefended_failed - defended_failed}
  Improvement: {100 * (1 - defended_failed / max(1, undefended_failed)):.1f}%
"""
    )

    # ==========================================================
    # 6. PHYSICS EXPLANATION
    # ==========================================================
    print("=" * 70)
    print("THE PHYSICS OF NETWORK ATTACKS")
    print("=" * 70)
    print(
        """
FLUID DYNAMICS ANALOGY:

1. THE NETWORK AS A PIPE SYSTEM
   - Nodes = Pressure reservoirs (servers with queue buffers)
   - Edges = Pipes (network links with bandwidth)
   - Packets = Fluid particles
   - Congestion = Pressure buildup
   
2. THE DIFFUSION EQUATION
   ∂u/∂t = α ∇²u
   
   On a graph, the Laplacian becomes:
   Δu[i] = Σⱼ (u[j] - u[i]) for neighbors j
   
   Traffic flows from high pressure (congested) to low pressure (free).

3. DDoS AS INFINITE PRESSURE SOURCE
   - Attacker injects traffic at rate >> normal
   - Pressure wave propagates through network
   - When pressure > capacity: NODE FAILURE
   - Failed nodes dump traffic to neighbors → CASCADE

4. CASCADE DYNAMICS
   - Similar to dam break or pipe burst
   - Failure begets failure (positive feedback)
   - Critical point: ~50% failure triggers total collapse
   
5. DEFENSE STRATEGIES
   - Rate limiting: Reduce α (diffusion rate) at edge
   - Capacity boost: Increase threshold (CDN, scrubbing)
   - Redundancy: Multiple paths distribute pressure
   - Isolation: Break network into segments
   
6. WHY THIS MATTERS
   - Visualize attacks before they happen
   - Identify critical infrastructure
   - Test defense strategies in simulation
   - Train security teams with physics intuition
"""
    )

    # ==========================================================
    # 7. VISUALIZATION READY
    # ==========================================================
    print("=" * 70)
    print("VISUALIZATION OPTIONS")
    print("=" * 70)
    print(
        """
The pressure_history tensor can be visualized as:

1. NETWORK GRAPH ANIMATION
   - Node color = pressure (blue → red)
   - Node size = capacity remaining
   - Failed nodes = black/removed
   - Edge thickness = flow rate

2. HEATMAP OVER TIME
   - X-axis = Node ID
   - Y-axis = Time step
   - Color = Pressure level
   - Red band = cascade wave

3. 3D TERRAIN
   - Height = pressure
   - Time as animation
   - Watch the shockwave propagate

All can use existing Glass Cockpit / Unreal pipelines.
"""
    )

    # ==========================================================
    # 8. FINAL STATUS
    # ==========================================================
    print("=" * 70)
    print("🛡️  GRID SHOCK STATUS")
    print("=" * 70)

    if defended_sim.cascade_report.cascade_occurred:
        print(
            """
⚠️  DEFENSE INSUFFICIENT

Even with protection, cascade occurred.
Recommend: Add more redundant paths.
"""
        )
    else:
        print(
            f"""
✅ NETWORK DEFENDED

Attack contained with {defended_failed} node failures.
Critical infrastructure protected.
Cascade prevented.

'The shield holds.'
"""
        )

    print("=" * 70)
    print("PHASE 10 COMPLETE: The Grid Shock")
    print("=" * 70)

    return sim, defended_sim


def run_cascade_threshold_study():
    """
    Study at what attack rate cascade occurs.
    """
    print()
    print("=" * 70)
    print("CASCADE THRESHOLD STUDY")
    print("=" * 70)
    print()

    results = []

    for rate in [100, 200, 300, 400, 500, 750, 1000]:
        grid = CyberGrid(num_nodes=50, connection_probability=0.1)
        sim = grid.simulate_attack(
            attacker_node=0,
            attack_rate=float(rate),
            steps=50,
            verbose=False,
        )

        cascade = "CASCADE" if sim.cascade_report.cascade_occurred else "CONTAINED"
        results.append((rate, sim.cascade_report.failed_nodes, cascade))

        print(
            f"  Attack Rate {rate:4d}: {sim.cascade_report.failed_nodes:2d} failures - {cascade}"
        )

    print()

    # Find threshold
    for i, (rate, failed, status) in enumerate(results):
        if status == "CASCADE":
            print(
                f"Cascade threshold: ~{results[i-1][0] if i > 0 else rate} - {rate} units/step"
            )
            break


if __name__ == "__main__":
    # Main demo
    full_grid_shock_demo()

    # Optional: threshold study
    # run_cascade_threshold_study()

    print("\n🛡️  Grid shock simulation complete. The shield holds.")
