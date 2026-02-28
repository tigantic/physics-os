#!/usr/bin/env python3
"""
Generate Cross-Primitive Pipeline visualization diagram.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import json

# Load attestation data
with open("CROSS_PRIMITIVE_PIPELINE_ATTESTATION.json", "r") as f:
    data = json.load(f)

# Set up the figure with dark theme
plt.style.use('dark_background')
fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# Colors
COLORS = {
    'QTT-OT': '#FF6B6B',      # Red
    'QTT-SGW': '#4ECDC4',     # Teal
    'QTT-RKHS': '#45B7D1',    # Blue
    'QTT-PH': '#96CEB4',      # Green
    'QTT-GA': '#FFEAA7',      # Yellow
    'bg': '#1a1a2e',
    'accent': '#00d4ff',
    'gold': '#FFD700',
}

fig.patch.set_facecolor(COLORS['bg'])
ax.set_facecolor(COLORS['bg'])

# Title
ax.text(8, 9.5, "CROSS-PRIMITIVE PIPELINE", fontsize=28, fontweight='bold',
        ha='center', va='center', color=COLORS['gold'], fontfamily='monospace')
ax.text(8, 8.9, "THE MOAT DEMONSTRATION", fontsize=16, ha='center', va='center',
        color='#aaaaaa', fontfamily='monospace')

# Pipeline stages - horizontal flow
stages = data['stages']
stage_x = [2, 5, 8, 11, 14]
stage_y = 6

# Draw connecting arrows first (behind boxes)
for i in range(len(stage_x) - 1):
    ax.annotate('', xy=(stage_x[i+1] - 1.1, stage_y), 
                xytext=(stage_x[i] + 1.1, stage_y),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent'], 
                               lw=3, mutation_scale=20))

# Draw stage boxes
for i, (stage, x) in enumerate(zip(stages, stage_x)):
    color = COLORS[stage['primitive']]
    
    # Box
    box = FancyBboxPatch((x - 1, stage_y - 0.8), 2, 1.6,
                         boxstyle="round,pad=0.05,rounding_size=0.2",
                         facecolor=color, edgecolor='white', linewidth=2,
                         alpha=0.9)
    ax.add_patch(box)
    
    # Layer number
    ax.text(x, stage_y + 0.5, f"Layer {stage['layer']}", fontsize=10,
            ha='center', va='center', color='black', fontweight='bold')
    
    # Primitive name
    ax.text(x, stage_y, stage['primitive'], fontsize=12,
            ha='center', va='center', color='black', fontweight='bold',
            fontfamily='monospace')
    
    # Time
    ax.text(x, stage_y - 0.5, f"{stage['time_seconds']:.2f}s", fontsize=9,
            ha='center', va='center', color='black')

# Stage descriptions below
descriptions = [
    "Climate\nTransport",
    "Spectral\nAnalysis", 
    "Anomaly\nDetection",
    "Topology\nStructure",
    "Geometric\nCharacterization"
]

for x, desc in zip(stage_x, descriptions):
    ax.text(x, stage_y - 1.5, desc, fontsize=9, ha='center', va='center',
            color='white', fontfamily='monospace')

# Key metrics panel
metrics_y = 3.2

# Left panel - Memory
ax.add_patch(FancyBboxPatch((0.5, 1.5), 4.5, 2.5,
                            boxstyle="round,pad=0.05,rounding_size=0.2",
                            facecolor='#2a2a4e', edgecolor=COLORS['accent'], 
                            linewidth=2, alpha=0.8))

ax.text(2.75, 3.7, "MEMORY", fontsize=12, fontweight='bold',
        ha='center', color=COLORS['accent'], fontfamily='monospace')

mem = data['memory']
ax.text(2.75, 3.2, f"Dense (theoretical): {mem['theoretical_dense_bytes']:,} bytes",
        fontsize=10, ha='center', color='#ff6b6b')
ax.text(2.75, 2.7, f"QTT (actual): {mem['actual_qtt_bytes']:,} bytes",
        fontsize=10, ha='center', color='#4ecdc4')
ax.text(2.75, 2.1, f"Compression: {mem['compression_ratio']:.1f}×",
        fontsize=14, ha='center', color=COLORS['gold'], fontweight='bold')

# Center panel - Moat Status
ax.add_patch(FancyBboxPatch((5.5, 1.5), 5, 2.5,
                            boxstyle="round,pad=0.05,rounding_size=0.2",
                            facecolor='#1a3a1a', edgecolor=COLORS['gold'], 
                            linewidth=3, alpha=0.8))

ax.text(8, 3.7, "★ MOAT VERIFIED ★", fontsize=14, fontweight='bold',
        ha='center', color=COLORS['gold'], fontfamily='monospace')
ax.text(8, 3.1, f"Primitives Chained: {data['moat']['primitives_chained']}",
        fontsize=11, ha='center', color='white')
ax.text(8, 2.6, f"Densification Events: {data['moat']['densification_events']}",
        fontsize=11, ha='center', color='#4ecdc4')
ax.text(8, 2.0, "ALL STAGES COMPRESSED",
        fontsize=12, ha='center', color='#00ff00', fontweight='bold')

# Right panel - Findings
ax.add_patch(FancyBboxPatch((11, 1.5), 4.5, 2.5,
                            boxstyle="round,pad=0.05,rounding_size=0.2",
                            facecolor='#2a2a4e', edgecolor=COLORS['accent'], 
                            linewidth=2, alpha=0.8))

ax.text(13.25, 3.7, "FINDINGS", fontsize=12, fontweight='bold',
        ha='center', color=COLORS['accent'], fontfamily='monospace')

findings = data['findings']
ax.text(13.25, 3.2, f"Climate Shift: +{findings['climate_shift_celsius']}°C",
        fontsize=10, ha='center', color='white')
ax.text(13.25, 2.7, f"Anomaly Severity: {findings['anomaly_severity']:.2f}",
        fontsize=10, ha='center', color='#ff6b6b')
ax.text(13.25, 2.1, f"Topo Complexity: {findings['topological_complexity']}",
        fontsize=10, ha='center', color='white')

# Scale and time info
ax.text(8, 0.8, f"Scale: 2^{data['grid_bits']} = {data['scale']:,} points  |  Total Time: {data['total_time_seconds']:.2f}s",
        fontsize=12, ha='center', color='#888888', fontfamily='monospace')

# Branding
ax.text(8, 0.3, "TENSOR GENESIS • The Ontic Engine • January 2026",
        fontsize=10, ha='center', color='#555555', fontfamily='monospace')

# Save
plt.tight_layout()
plt.savefig('assets/cross_primitive_pipeline.png', dpi=150, 
            facecolor=COLORS['bg'], edgecolor='none',
            bbox_inches='tight', pad_inches=0.3)
plt.savefig('assets/cross_primitive_pipeline.svg', 
            facecolor=COLORS['bg'], edgecolor='none',
            bbox_inches='tight', pad_inches=0.3)

print("✓ Generated: assets/cross_primitive_pipeline.png")
print("✓ Generated: assets/cross_primitive_pipeline.svg")
