"""
HyperFOAM Solver Visualizer
===========================

Real-time and post-simulation CFD visualization.

Article VII, Section 7.2: Show the physics, not "trust me bro".
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class VisualizationFrame:
    """Single frame of solver state for visualization."""
    iteration: int
    residual: float
    u_slice: np.ndarray  # X-velocity at mid-height
    v_slice: np.ndarray  # Y-velocity at mid-height  
    w_slice: np.ndarray  # Z-velocity at mid-height
    T_slice: np.ndarray  # Temperature at mid-height
    velocity_mag: np.ndarray  # Velocity magnitude at mid-height
    max_velocity: float
    mean_temp: float
    

def extract_frame_from_state(state, iteration: int, config) -> VisualizationFrame:
    """Extract visualization data from solver state."""
    # Get mid-height slice (z = nz//2)
    mid_z = config.nz // 2
    
    # Convert torch tensors to numpy
    u_np = state.u.cpu().numpy() if hasattr(state.u, 'cpu') else np.array(state.u)
    v_np = state.v.cpu().numpy() if hasattr(state.v, 'cpu') else np.array(state.v)
    w_np = state.w.cpu().numpy() if hasattr(state.w, 'cpu') else np.array(state.w)
    T_np = state.T.cpu().numpy() if hasattr(state.T, 'cpu') else np.array(state.T)
    vel_mag = state.velocity_magnitude.cpu().numpy() if hasattr(state.velocity_magnitude, 'cpu') else np.array(state.velocity_magnitude)
    
    return VisualizationFrame(
        iteration=iteration,
        residual=state.residual_history[-1] if state.residual_history else 0,
        u_slice=u_np[:, :, mid_z],
        v_slice=v_np[:, :, mid_z],
        w_slice=w_np[:, :, mid_z],
        T_slice=T_np[:, :, mid_z],
        velocity_mag=vel_mag[:, :, mid_z],
        max_velocity=vel_mag.max(),
        mean_temp=T_np.mean(),
    )


def create_live_figure(config) -> go.Figure:
    """Create the live visualization figure with subplots."""
    if not PLOTLY_AVAILABLE:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '🌀 Velocity Magnitude (m/s)', 
            '🌡️ Temperature (°C)',
            '📉 Convergence History',
            '📊 Simulation Stats'
        ),
        specs=[
            [{"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "scatter"}, {"type": "table"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    
    # Initialize empty heatmaps
    nx, ny = config.nx, config.ny
    
    # Velocity magnitude heatmap
    fig.add_trace(
        go.Heatmap(
            z=np.zeros((ny, nx)),
            colorscale='Turbo',
            showscale=True,
            colorbar=dict(title="m/s", x=0.45),
            name='Velocity'
        ),
        row=1, col=1
    )
    
    # Temperature heatmap
    fig.add_trace(
        go.Heatmap(
            z=np.ones((ny, nx)) * 24,
            colorscale='RdYlBu_r',
            showscale=True,
            colorbar=dict(title="°C", x=1.0),
            name='Temperature'
        ),
        row=1, col=2
    )
    
    # Convergence plot
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[1],
            mode='lines+markers',
            name='Residual',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4),
        ),
        row=2, col=1
    )
    
    # Stats table
    fig.add_trace(
        go.Table(
            header=dict(
                values=['Metric', 'Value'],
                fill_color='#1976d2',
                font=dict(color='white', size=12),
                align='left'
            ),
            cells=dict(
                values=[
                    ['Iteration', 'Residual', 'Max Velocity', 'Mean Temp', 'Status'],
                    ['0', '1.0e+00', '0.00 m/s', '24.0 °C', '⏳ Running...']
                ],
                fill_color='#f5f5f5',
                align='left',
                font=dict(size=11),
                height=25,
            )
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=700,
        showlegend=False,
        title=dict(
            text='🔥 HyperFOAM CFD Solver - Live Simulation',
            font=dict(size=18),
        ),
        margin=dict(t=80, b=40, l=40, r=40),
    )
    
    # Set axis labels
    fig.update_xaxes(title_text="X (cells)", row=1, col=1)
    fig.update_yaxes(title_text="Y (cells)", row=1, col=1)
    fig.update_xaxes(title_text="X (cells)", row=1, col=2)
    fig.update_yaxes(title_text="Y (cells)", row=1, col=2)
    fig.update_xaxes(title_text="Iteration", type="log", row=2, col=1)
    fig.update_yaxes(title_text="Residual", type="log", row=2, col=1)
    
    return fig


def update_live_figure(fig: go.Figure, frame: VisualizationFrame, 
                       residual_history: List[float]) -> go.Figure:
    """Update the live figure with new frame data."""
    if fig is None:
        return None
    
    # Update velocity heatmap (trace 0)
    fig.data[0].z = frame.velocity_mag.T
    
    # Update temperature heatmap (trace 1)
    fig.data[1].z = frame.T_slice.T
    
    # Update convergence plot (trace 2)
    iterations = list(range(1, len(residual_history) + 1))
    fig.data[2].x = iterations
    fig.data[2].y = residual_history
    
    # Update stats table (trace 3)
    status = '✅ Converged' if frame.residual < 1e-5 else '⏳ Running...'
    fig.data[3].cells.values = [
        ['Iteration', 'Residual', 'Max Velocity', 'Mean Temp', 'Status'],
        [
            str(frame.iteration),
            f'{frame.residual:.2e}',
            f'{frame.max_velocity:.3f} m/s',
            f'{frame.mean_temp:.1f} °C',
            status
        ]
    ]
    
    return fig


def create_final_results_figure(final_frame: VisualizationFrame, 
                                residual_history: List[float],
                                config,
                                runtime_seconds: float) -> go.Figure:
    """Create comprehensive final results visualization."""
    if not PLOTLY_AVAILABLE:
        return None
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            '🌀 Velocity Field (m/s)',
            '🌡️ Temperature Field (°C)', 
            '➡️ X-Velocity Component',
            '📉 Convergence History',
            '⬇️ Y-Velocity Component',
            '📊 Final Statistics'
        ),
        specs=[
            [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "scatter"}, {"type": "heatmap"}, {"type": "table"}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.08,
    )
    
    # 1. Velocity magnitude
    fig.add_trace(
        go.Heatmap(
            z=final_frame.velocity_mag.T,
            colorscale='Turbo',
            showscale=True,
            colorbar=dict(title="m/s", x=0.28, len=0.4, y=0.8),
        ),
        row=1, col=1
    )
    
    # 2. Temperature
    fig.add_trace(
        go.Heatmap(
            z=final_frame.T_slice.T,
            colorscale='RdYlBu_r',
            showscale=True,
            colorbar=dict(title="°C", x=0.63, len=0.4, y=0.8),
        ),
        row=1, col=2
    )
    
    # 3. X-velocity
    fig.add_trace(
        go.Heatmap(
            z=final_frame.u_slice.T,
            colorscale='RdBu_r',
            zmid=0,
            showscale=True,
            colorbar=dict(title="m/s", x=0.98, len=0.4, y=0.8),
        ),
        row=1, col=3
    )
    
    # 4. Convergence history
    iterations = list(range(1, len(residual_history) + 1))
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=residual_history,
            mode='lines',
            line=dict(color='#1f77b4', width=2),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.2)',
        ),
        row=2, col=1
    )
    
    # Add convergence threshold line
    fig.add_hline(y=1e-5, line_dash="dash", line_color="green", 
                  annotation_text="Converged", row=2, col=1)
    
    # 5. Y-velocity
    fig.add_trace(
        go.Heatmap(
            z=final_frame.v_slice.T,
            colorscale='RdBu_r',
            zmid=0,
            showscale=True,
            colorbar=dict(title="m/s", x=0.63, len=0.4, y=0.2),
        ),
        row=2, col=2
    )
    
    # 6. Stats table
    converged = "✅ Yes" if final_frame.residual < 1e-5 else "❌ No"
    fig.add_trace(
        go.Table(
            header=dict(
                values=['<b>Metric</b>', '<b>Value</b>'],
                fill_color='#1976d2',
                font=dict(color='white', size=12),
                align='left',
                height=30,
            ),
            cells=dict(
                values=[
                    ['Grid Size', 'Total Cells', 'Iterations', 'Final Residual', 
                     'Max Velocity', 'Mean Temperature', 'Runtime', 'Converged'],
                    [f'{config.nx}×{config.ny}×{config.nz}',
                     f'{config.nx * config.ny * config.nz:,}',
                     str(final_frame.iteration),
                     f'{final_frame.residual:.2e}',
                     f'{final_frame.max_velocity:.4f} m/s',
                     f'{final_frame.mean_temp:.2f} °C',
                     f'{runtime_seconds:.1f} seconds',
                     converged]
                ],
                fill_color=[['#f0f0f0', 'white'] * 4],
                align='left',
                font=dict(size=11),
                height=28,
            )
        ),
        row=2, col=3
    )
    
    fig.update_layout(
        height=800,
        showlegend=False,
        title=dict(
            text='🎯 HyperFOAM Simulation Results - Mid-Height Cross-Section',
            font=dict(size=20),
        ),
        margin=dict(t=100, b=50, l=50, r=50),
    )
    
    # Update axes
    for row in [1, 2]:
        for col in [1, 2, 3]:
            if not (row == 2 and col == 3):  # Skip table
                fig.update_xaxes(title_text="X (cells)", row=row, col=col)
                fig.update_yaxes(title_text="Y (cells)", row=row, col=col)
    
    fig.update_xaxes(title_text="Iteration", type="log", row=2, col=1)
    fig.update_yaxes(title_text="Residual", type="log", row=2, col=1)
    
    return fig


def create_velocity_vector_plot(frame: VisualizationFrame, 
                                config, 
                                skip: int = 4) -> go.Figure:
    """Create velocity vector field visualization."""
    if not PLOTLY_AVAILABLE:
        return None
    
    nx, ny = frame.u_slice.shape
    
    # Create coordinate grids
    x = np.arange(0, nx, skip)
    y = np.arange(0, ny, skip)
    X, Y = np.meshgrid(x, y)
    
    # Get velocity components (downsampled)
    U = frame.u_slice[::skip, ::skip].T
    V = frame.v_slice[::skip, ::skip].T
    
    # Normalize for arrow length
    mag = np.sqrt(U**2 + V**2)
    mag_max = mag.max() if mag.max() > 0 else 1
    scale = 3.0 / mag_max
    
    fig = go.Figure()
    
    # Background: velocity magnitude
    fig.add_trace(go.Heatmap(
        z=frame.velocity_mag.T,
        colorscale='Turbo',
        showscale=True,
        colorbar=dict(title="Velocity (m/s)"),
        opacity=0.7,
    ))
    
    # Velocity vectors as arrows using annotations
    # (Plotly doesn't have native quiver, so we use cone or annotations)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if mag[i, j] > 0.01 * mag_max:  # Skip tiny vectors
                fig.add_annotation(
                    x=X[i, j],
                    y=Y[i, j],
                    ax=X[i, j] + U[i, j] * scale * skip,
                    ay=Y[i, j] + V[i, j] * scale * skip,
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1.5,
                    arrowcolor="white",
                )
    
    fig.update_layout(
        title="🏹 Velocity Vector Field",
        xaxis_title="X (cells)",
        yaxis_title="Y (cells)",
        height=500,
    )
    
    return fig


def create_streamlit_placeholder_content() -> str:
    """Return placeholder HTML for when plotly isn't available."""
    return """
    <div style="text-align:center; padding:40px; background:#f5f5f5; border-radius:8px;">
        <h3>📊 Visualization requires Plotly</h3>
        <p>Install with: <code>pip install plotly</code></p>
    </div>
    """
