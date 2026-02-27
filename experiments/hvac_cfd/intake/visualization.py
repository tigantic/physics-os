"""
3D Visualization Module
=======================

Provides interactive 3D visualization of HVAC scenes using Plotly.
Renders rooms, vents, occupants, and airflow paths.
"""

import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


def create_box_mesh(
    position: List[float],
    dimensions: List[float],
    color: str = 'lightblue',
    opacity: float = 0.3,
    name: str = ''
) -> go.Mesh3d:
    """
    Create a 3D box mesh for rooms/obstacles.
    
    Args:
        position: [x, y, z] of corner
        dimensions: [length, height, width] (x, y, z)
        color: Fill color
        opacity: Transparency (0-1)
        name: Label for hover
    
    Returns:
        Plotly Mesh3d trace
    """
    x0, y0, z0 = position
    dx, dy, dz = dimensions
    
    # 8 vertices of a box
    vertices = np.array([
        [x0, y0, z0],
        [x0 + dx, y0, z0],
        [x0 + dx, y0 + dy, z0],
        [x0, y0 + dy, z0],
        [x0, y0, z0 + dz],
        [x0 + dx, y0, z0 + dz],
        [x0 + dx, y0 + dy, z0 + dz],
        [x0, y0 + dy, z0 + dz],
    ])
    
    # 12 triangles (2 per face)
    i = [0, 0, 0, 0, 4, 4, 0, 1, 1, 2, 4, 5]
    j = [1, 2, 4, 5, 5, 6, 3, 2, 5, 6, 7, 6]
    k = [2, 3, 5, 1, 6, 7, 7, 6, 2, 3, 3, 2]
    
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=i, j=j, k=k,
        color=color,
        opacity=opacity,
        name=name,
        hoverinfo='name',
        flatshading=True,
    )


def create_wireframe_box(
    position: List[float],
    dimensions: List[float],
    color: str = 'black',
    width: int = 2,
    name: str = ''
) -> go.Scatter3d:
    """Create wireframe edges for a box."""
    x0, y0, z0 = position
    dx, dy, dz = dimensions
    
    # Define edges
    edges_x = []
    edges_y = []
    edges_z = []
    
    # Bottom face
    for (x1, y1, z1), (x2, y2, z2) in [
        ((x0, y0, z0), (x0+dx, y0, z0)),
        ((x0+dx, y0, z0), (x0+dx, y0, z0+dz)),
        ((x0+dx, y0, z0+dz), (x0, y0, z0+dz)),
        ((x0, y0, z0+dz), (x0, y0, z0)),
    ]:
        edges_x.extend([x1, x2, None])
        edges_y.extend([y1, y2, None])
        edges_z.extend([z1, z2, None])
    
    # Top face
    for (x1, y1, z1), (x2, y2, z2) in [
        ((x0, y0+dy, z0), (x0+dx, y0+dy, z0)),
        ((x0+dx, y0+dy, z0), (x0+dx, y0+dy, z0+dz)),
        ((x0+dx, y0+dy, z0+dz), (x0, y0+dy, z0+dz)),
        ((x0, y0+dy, z0+dz), (x0, y0+dy, z0)),
    ]:
        edges_x.extend([x1, x2, None])
        edges_y.extend([y1, y2, None])
        edges_z.extend([z1, z2, None])
    
    # Vertical edges
    for x, z in [(x0, z0), (x0+dx, z0), (x0+dx, z0+dz), (x0, z0+dz)]:
        edges_x.extend([x, x, None])
        edges_y.extend([y0, y0+dy, None])
        edges_z.extend([z, z, None])
    
    return go.Scatter3d(
        x=edges_x, y=edges_y, z=edges_z,
        mode='lines',
        line=dict(color=color, width=width),
        name=name,
        hoverinfo='name',
    )


def create_vent_marker(
    position: List[float],
    dimensions: List[float],
    direction: List[float],
    vent_type: int,  # 0=supply, 1=return
    name: str = '',
    flow_rate: float = 0
) -> List[go.Scatter3d]:
    """
    Create vent visualization with airflow direction arrow.
    
    Returns list of traces (vent box + arrow).
    """
    traces = []
    
    # Vent color based on type
    color = '#00BFFF' if vent_type == 0 else '#FF6B6B'  # Blue=supply, Red=return
    
    # Vent as small box
    x, y, z = position
    dx, dy, dz = dimensions
    
    # Create vent cube
    traces.append(go.Mesh3d(
        x=[x-dx/2, x+dx/2, x+dx/2, x-dx/2, x-dx/2, x+dx/2, x+dx/2, x-dx/2],
        y=[y-dy/2, y-dy/2, y+dy/2, y+dy/2, y-dy/2, y-dy/2, y+dy/2, y+dy/2],
        z=[z-dz/2, z-dz/2, z-dz/2, z-dz/2, z+dz/2, z+dz/2, z+dz/2, z+dz/2],
        i=[0, 0, 0, 0, 4, 4, 0, 1, 1, 2, 4, 5],
        j=[1, 2, 4, 5, 5, 6, 3, 2, 5, 6, 7, 6],
        k=[2, 3, 5, 1, 6, 7, 7, 6, 2, 3, 3, 2],
        color=color,
        opacity=0.8,
        name=name,
        hovertemplate=f"{name}<br>Flow: {flow_rate*2118.88:.0f} CFM<extra></extra>",
    ))
    
    # Airflow direction arrow
    arrow_length = 0.8
    dir_x, dir_y, dir_z = direction
    
    traces.append(go.Cone(
        x=[x + dir_x * 0.3],
        y=[y + dir_y * 0.3],
        z=[z + dir_z * 0.3],
        u=[dir_x * arrow_length],
        v=[dir_y * arrow_length],
        w=[dir_z * arrow_length],
        colorscale=[[0, color], [1, color]],
        showscale=False,
        sizemode='absolute',
        sizeref=0.3,
        name=f"{name} flow",
        hoverinfo='skip',
    ))
    
    return traces


def create_occupant_marker(
    position: List[float],
    name: str = '',
    heat_output: float = 100
) -> go.Scatter3d:
    """Create occupant marker (sphere/point)."""
    x, y, z = position
    
    # Place occupant at standing height
    return go.Scatter3d(
        x=[x],
        y=[y + 0.9],  # Head height ~0.9m above floor
        z=[z],
        mode='markers+text',
        marker=dict(
            size=12,
            color='#FFD700',
            symbol='circle',
            line=dict(color='#B8860B', width=2),
        ),
        text=[f"👤"],
        textposition='top center',
        name=name,
        hovertemplate=f"{name}<br>Heat: {heat_output:.0f} W<extra></extra>",
    )


def create_floor_grid(
    x_range: Tuple[float, float],
    z_range: Tuple[float, float],
    grid_size: float = 1.0
) -> go.Scatter3d:
    """Create floor grid lines."""
    lines_x = []
    lines_y = []
    lines_z = []
    
    x_min, x_max = x_range
    z_min, z_max = z_range
    
    # X-parallel lines
    for z in np.arange(z_min, z_max + grid_size, grid_size):
        lines_x.extend([x_min, x_max, None])
        lines_y.extend([0, 0, None])
        lines_z.extend([z, z, None])
    
    # Z-parallel lines
    for x in np.arange(x_min, x_max + grid_size, grid_size):
        lines_x.extend([x, x, None])
        lines_y.extend([0, 0, None])
        lines_z.extend([z_min, z_max, None])
    
    return go.Scatter3d(
        x=lines_x, y=lines_y, z=lines_z,
        mode='lines',
        line=dict(color='#444444', width=1),
        name='Grid',
        hoverinfo='skip',
    )


def visualize_job_spec(job_spec: Dict[str, Any], show_grid: bool = True) -> go.Figure:
    """
    Create complete 3D visualization from job_spec.
    
    Args:
        job_spec: Complete job specification dict
        show_grid: Whether to show floor grid
    
    Returns:
        Plotly Figure object
    """
    traces = []
    
    # Extract geometry
    geometry = job_spec.get('geometry', {})
    rooms = geometry.get('rooms', [])
    hvac = job_spec.get('hvac', {})
    vents = hvac.get('vents', [])
    sources = job_spec.get('sources', {})
    occupants = sources.get('occupants', [])
    
    # Track bounds for camera
    all_x = [0]
    all_y = [0]
    all_z = [0]
    
    # Color palette for rooms
    room_colors = ['#87CEEB', '#98FB98', '#DDA0DD', '#F0E68C', '#E0FFFF']
    
    # Add rooms
    for i, room in enumerate(rooms):
        pos = room.get('position', [0, 0, 0])
        dims = room.get('dimensions', [10, 3, 10])
        name = room.get('name', f'Room {i+1}')
        color = room_colors[i % len(room_colors)]
        
        # Room as semi-transparent box
        traces.append(create_box_mesh(pos, dims, color=color, opacity=0.15, name=name))
        # Room wireframe
        traces.append(create_wireframe_box(pos, dims, color='#333333', width=2, name=f'{name} outline'))
        
        # Add dimension annotations as 3D text markers
        dx, dy, dz = dims
        x0, y0, z0 = pos
        
        # Length annotation (X axis) - along the floor
        traces.append(go.Scatter3d(
            x=[x0 + dx/2],
            y=[y0 - 0.3],
            z=[z0 - 0.5],
            mode='text',
            text=[f'{dx:.1f}m'],
            textfont=dict(size=14, color='#00FFFF'),
            textposition='middle center',
            showlegend=False,
            hoverinfo='skip',
        ))
        
        # Width annotation (Z axis)
        traces.append(go.Scatter3d(
            x=[x0 - 0.5],
            y=[y0 - 0.3],
            z=[z0 + dz/2],
            mode='text',
            text=[f'{dz:.1f}m'],
            textfont=dict(size=14, color='#00FFFF'),
            textposition='middle center',
            showlegend=False,
            hoverinfo='skip',
        ))
        
        # Height annotation (Y axis)
        traces.append(go.Scatter3d(
            x=[x0 - 0.5],
            y=[y0 + dy/2],
            z=[z0 - 0.5],
            mode='text',
            text=[f'{dy:.1f}m'],
            textfont=dict(size=14, color='#00FFFF'),
            textposition='middle center',
            showlegend=False,
            hoverinfo='skip',
        ))
        
        # Update bounds
        all_x.extend([pos[0], pos[0] + dims[0]])
        all_y.extend([pos[1], pos[1] + dims[1]])
        all_z.extend([pos[2], pos[2] + dims[2]])
    
    # Add vents
    for vent in vents:
        pos = vent.get('position', [0, 0, 0])
        dims = vent.get('dimensions', [0.6, 0.15, 0.6])
        direction = vent.get('direction', [0, -1, 0])
        vent_type = vent.get('type', 0)
        name = vent.get('name', 'Vent')
        flow = vent.get('flowRate', 0)
        
        traces.extend(create_vent_marker(pos, dims, direction, vent_type, name, flow))
        
        all_x.append(pos[0])
        all_y.append(pos[1])
        all_z.append(pos[2])
    
    # Add occupants
    for occ in occupants:
        pos = occ.get('position', [0, 0, 0])
        name = occ.get('name', 'Person')
        heat = occ.get('heatOutput', 100)
        
        traces.append(create_occupant_marker(pos, name, heat))
        
        all_x.append(pos[0])
        all_z.append(pos[2])
    
    # Add floor grid
    if show_grid and rooms:
        x_range = (min(all_x) - 1, max(all_x) + 1)
        z_range = (min(all_z) - 1, max(all_z) + 1)
        traces.append(create_floor_grid(x_range, z_range))
    
    # Create figure
    fig = go.Figure(data=traces)
    
    # Configure layout
    x_range = max(all_x) - min(all_x)
    y_range = max(all_y) - min(all_y)  
    z_range = max(all_z) - min(all_z)
    max_range = max(x_range, y_range, z_range) or 10
    
    center_x = (max(all_x) + min(all_x)) / 2
    center_y = (max(all_y) + min(all_y)) / 2
    center_z = (max(all_z) + min(all_z)) / 2
    
    # Calculate camera position based on actual scene bounds
    # Position camera to show true proportions - higher Y for rooms with low ceiling
    cam_dist = max_range * 0.8
    aspect_ratio = x_range / (z_range or 1)
    
    fig.update_layout(
        title=dict(
            text=f"🏢 {job_spec.get('deliverables', {}).get('projectName', 'HVAC Scene')}",
            font=dict(size=20),
        ),
        scene=dict(
            xaxis=dict(
                title='X (m)', 
                gridcolor='#555', 
                showbackground=True, 
                backgroundcolor='#1a1a1a',
                range=[min(all_x) - 0.5, max(all_x) + 0.5],
            ),
            yaxis=dict(
                title='Height (m)', 
                gridcolor='#555', 
                showbackground=True, 
                backgroundcolor='#1a1a1a',
                range=[min(all_y) - 0.5, max(all_y) + 0.5],
            ),
            zaxis=dict(
                title='Z (m)', 
                gridcolor='#555', 
                showbackground=True, 
                backgroundcolor='#1a1a1a',
                range=[min(all_z) - 0.5, max(all_z) + 0.5],
            ),
            aspectmode='data',  # True proportions
            camera=dict(
                # Isometric-ish view that shows room proportions clearly
                eye=dict(x=1.8, y=0.8, z=1.3),
                center=dict(x=0, y=-0.1, z=0),
                up=dict(x=0, y=1, z=0),
            ),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='#0e1117',
        font=dict(color='white'),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(0,0,0,0.5)',
        ),
    )
    
    return fig


def visualize_form_data(form_data: Dict[str, Any]) -> go.Figure:
    """
    Create 3D preview from intake form data (before job_spec generation).
    
    Args:
        form_data: Form field values from Streamlit session
    
    Returns:
        Plotly Figure object
    """
    # Extract dimensions (in user's units - typically feet for imperial)
    length = float(form_data.get('room_length', 30) or 30)
    width = float(form_data.get('room_width', 20) or 20)
    height = float(form_data.get('room_height', 10) or 10)
    
    # Get unit system - check both possible keys
    unit_system = form_data.get('unit_system', 'imperial')
    
    # Convert to meters for 3D visualization (internal units)
    if unit_system != 'metric':
        # Imperial: feet to meters
        length_m = length * 0.3048
        width_m = width * 0.3048
        height_m = height * 0.3048
    else:
        # Already metric
        length_m = length
        width_m = width
        height_m = height
    
    room_name = form_data.get('room_name', 'Main Room') or 'Main Room'
    project_name = form_data.get('project_name', 'Preview') or 'Preview'
    
    # Get the correct unit label
    unit_label = "m" if unit_system == 'metric' else "ft"
    
    # Format title: show user units + solver units (SI)
    # "Sandwich Method": User sees ft, solver uses m internally
    if unit_system != 'metric':
        title_dims = f"{length:.0f}×{width:.0f}×{height:.0f} ft → {length_m:.1f}×{width_m:.1f}×{height_m:.1f} m"
    else:
        title_dims = f"{length:.1f}×{width:.1f}×{height:.1f} m"
    
    # Build minimal job_spec for visualization
    # dimensions: [X-length, Y-height, Z-width] for proper 3D orientation
    preview_spec = {
        'deliverables': {'projectName': f"{project_name} ({title_dims})"},
        'geometry': {
            'rooms': [{
                'name': room_name,
                'position': [0, 0, 0],
                'dimensions': [length_m, height_m, width_m],  # X, Y(up), Z
            }]
        },
        'hvac': {
            'vents': []
        },
        'sources': {
            'occupants': []
        }
    }
    
    # Add vents if specified
    vent_count = int(form_data.get('vent_count', 1) or 1)
    supply_cfm = float(form_data.get('supply_airflow', 100) or 100)
    supply_m3s = supply_cfm * 0.000471947  # CFM to m³/s
    
    for i in range(vent_count):
        x_pos = length_m * (i + 1) / (vent_count + 1)
        preview_spec['hvac']['vents'].append({
            'name': f'Supply {i+1}',
            'type': 0,
            'position': [x_pos, height_m - 0.2, width_m / 2],
            'dimensions': [0.6, 0.15, 0.6],
            'direction': [0, -1, 0],
            'flowRate': supply_m3s / vent_count,
        })
    
    # Add return vents
    return_count = int(form_data.get('return_count', 1) or 1)
    for i in range(return_count):
        x_pos = length_m * (i + 1) / (return_count + 1)
        preview_spec['hvac']['vents'].append({
            'name': f'Return {i+1}',
            'type': 1,
            'position': [x_pos, 0.3, width_m / 2],
            'dimensions': [0.6, 0.15, 0.6],
            'direction': [0, 1, 0],
            'flowRate': supply_m3s / return_count,
        })
    
    # Add occupants
    occupancy = int(form_data.get('occupancy', 1) or 1)
    for i in range(occupancy):
        x_pos = length_m * (i + 1) / (occupancy + 1)
        preview_spec['sources']['occupants'].append({
            'name': f'Person {i+1}',
            'position': [x_pos, 0, width_m / 2],
            'heatOutput': 100,
        })
    
    return visualize_job_spec(preview_spec)


# Quick test
if __name__ == "__main__":
    # Test with sample data
    sample_spec = {
        'deliverables': {'projectName': 'Test Room'},
        'geometry': {
            'rooms': [
                {'name': 'Office', 'position': [0, 0, 0], 'dimensions': [8, 3, 6]},
            ]
        },
        'hvac': {
            'vents': [
                {'name': 'Supply 1', 'type': 0, 'position': [4, 2.8, 3], 
                 'dimensions': [0.6, 0.15, 0.6], 'direction': [0, -1, 0], 'flowRate': 0.05},
                {'name': 'Return 1', 'type': 1, 'position': [4, 0.3, 3],
                 'dimensions': [0.6, 0.15, 0.6], 'direction': [0, 1, 0], 'flowRate': 0.05},
            ]
        },
        'sources': {
            'occupants': [
                {'name': 'Person 1', 'position': [2, 0, 3], 'heatOutput': 100},
                {'name': 'Person 2', 'position': [6, 0, 3], 'heatOutput': 100},
            ]
        }
    }
    
    fig = visualize_job_spec(sample_spec)
    fig.show()
