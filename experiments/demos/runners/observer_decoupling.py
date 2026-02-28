#!/usr/bin/env python3
"""
Ontic Unique Capability Demo
==================================

What others CAN do: TT-SVD compression, pointwise query
What Ontic CAN do: STORE and OPERATE on fields at scale

Scroll to increase grid size.
Watch: grid hits BILLIONS, memory stays KB.

Dense can't even allocate. We can.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ontic.cfd.pure_qtt_ops import QTTState, dense_to_qtt, qtt_to_dense, qtt_norm, qtt_add

# ---------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------

n_qubits = 10  # Start: 2^10 = 1,024 points
max_qubits = 40  # Max: 2^40 = 1 TRILLION points

# ---------------------------------------------------------------------
# FIGURE
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 8))

# We'll show a bar chart: dense memory vs QTT memory
bars = ax.bar(['Dense\n(impossible)', 'QTT\n(actual)'], [0, 0], color=['red', 'green'])
ax.set_ylabel('Memory', fontsize=12)
ax.set_ylim(0, 1)

title = fig.suptitle("", fontsize=14, fontweight='bold')
info = fig.text(0.5, 0.02, "", ha='center', fontsize=12)
status = ax.text(0.5, 0.5, "", ha='center', va='center', fontsize=16, 
                 transform=ax.transAxes, fontweight='bold')

# ---------------------------------------------------------------------
# CREATE QTT FOR HUGE GRIDS
# ---------------------------------------------------------------------

def create_low_rank_qtt(n_qubits: int, rank: int = 4) -> QTTState:
    """
    Create a low-rank QTT directly (no dense intermediate).
    
    This is how you handle 2^40 point grids - you never build dense.
    """
    cores = []
    for i in range(n_qubits):
        r_left = 1 if i == 0 else rank
        r_right = 1 if i == n_qubits - 1 else rank
        # Random low-rank core (represents a smooth function)
        core = torch.randn(r_left, 2, r_right, dtype=torch.float32) * 0.5
        cores.append(core)
    return QTTState(cores=cores, num_qubits=n_qubits)


# ---------------------------------------------------------------------
# RENDER
# ---------------------------------------------------------------------

def render(n_qubits: int):
    N = 2 ** n_qubits
    
    # Create QTT directly (no dense!)
    qtt = create_low_rank_qtt(n_qubits, rank=4)
    
    # Also create a second one and ADD them (in QTT form!)
    qtt2 = create_low_rank_qtt(n_qubits, rank=4)
    qtt_sum = qtt_add(qtt, qtt2, max_bond=8)
    
    # Compute norm (without densifying!)
    norm = qtt_norm(qtt_sum)
    
    # Memory calculations
    qtt_bytes = sum(c.numel() * 4 for c in qtt_sum.cores)
    dense_bytes = N * 4  # float32
    
    # Format dense memory
    if dense_bytes < 1024:
        dense_str = f"{dense_bytes} B"
        dense_val = dense_bytes
        unit_factor = 1
    elif dense_bytes < 1024**2:
        dense_str = f"{dense_bytes/1024:.1f} KB"
        dense_val = dense_bytes / 1024
        unit_factor = 1024
    elif dense_bytes < 1024**3:
        dense_str = f"{dense_bytes/1024**2:.1f} MB"
        dense_val = dense_bytes / 1024**2
        unit_factor = 1024**2
    elif dense_bytes < 1024**4:
        dense_str = f"{dense_bytes/1024**3:.1f} GB"
        dense_val = dense_bytes / 1024**3
        unit_factor = 1024**3
    else:
        dense_str = f"{dense_bytes/1024**4:.1f} TB"
        dense_val = dense_bytes / 1024**4
        unit_factor = 1024**4
    
    # Update bars (normalize so QTT is visible)
    qtt_normalized = qtt_bytes / unit_factor
    
    # Clear and redraw
    ax.clear()
    
    if dense_bytes > 16 * 1024**3:  # > 16 GB
        # Can't even show dense bar - it's "infinity"
        ax.bar(['Dense\n(IMPOSSIBLE)', 'QTT\n(actual)'], 
               [1.0, 0.01], color=['darkred', 'green'])
        ax.set_ylim(0, 1.1)
        ax.text(0, 0.5, f"∞\n{dense_str}", ha='center', fontsize=14, fontweight='bold', color='white')
        ax.text(1, 0.05, f"{qtt_bytes:,} B", ha='center', fontsize=12, fontweight='bold')
    else:
        max_val = max(dense_val, qtt_normalized) * 1.2
        ax.bar(['Dense', 'QTT'], [dense_val, qtt_normalized], color=['red', 'green'])
        ax.set_ylim(0, max_val)
        ax.text(0, dense_val + max_val*0.02, dense_str, ha='center', fontsize=11)
        ax.text(1, qtt_normalized + max_val*0.02, f"{qtt_bytes:,} B", ha='center', fontsize=11)
    
    ax.set_ylabel('Memory')
    
    title.set_text(
        f"Grid: 2^{n_qubits} = {N:,} points\n"
        f"Operations: Created 2 fields + Added them + Computed norm = {norm:.4f}"
    )
    
    compression = dense_bytes / qtt_bytes if qtt_bytes > 0 else 0
    
    if N > 1e12:
        info.set_text(f"★ TRILLION+ POINTS - Dense needs {dense_str}. We used {qtt_bytes:,} bytes. Ratio: {compression:,.0f}× ★")
    elif N > 1e9:
        info.set_text(f"★ BILLION+ POINTS - Dense needs {dense_str}. We used {qtt_bytes:,} bytes. Ratio: {compression:,.0f}× ★")
    elif N > 1e6:
        info.set_text(f"Millions of points. Dense: {dense_str}, QTT: {qtt_bytes:,} bytes. Ratio: {compression:,.0f}×")
    else:
        info.set_text(f"Scroll UP to scale grid. Dense: {dense_str}, QTT: {qtt_bytes:,} bytes. Ratio: {compression:,.0f}×")


# ---------------------------------------------------------------------
# EVENTS
# ---------------------------------------------------------------------

def on_scroll(event):
    global n_qubits
    if event.button == "up":
        n_qubits = min(n_qubits + 2, max_qubits)
    elif event.button == "down":
        n_qubits = max(n_qubits - 2, 6)
    
    render(n_qubits)
    fig.canvas.draw_idle()

fig.canvas.mpl_connect("scroll_event", on_scroll)

# ---------------------------------------------------------------------
# INITIAL RENDER
# ---------------------------------------------------------------------

render(n_qubits)
plt.tight_layout(rect=[0, 0.05, 1, 0.93])
plt.show()
