"""
HyperFOAM Dashboard: HVAC Digital Twin Interface

A Streamlit web app for real-time HVAC simulation.

Run: python -m hyperfoam.dashboard
  or: streamlit run hyperfoam/dashboard.py
"""

import sys
from pathlib import Path

# Ensure hyperfoam is importable
_pkg_root = Path(__file__).parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

import streamlit as st
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

from hyperfoam import Solver, ConferenceRoom
from hyperfoam.presets import setup_conference_room

# Page config
st.set_page_config(
    page_title="HyperFOAM Dashboard",
    page_icon="🌡️",
    layout="wide"
)

# Custom CSS for metrics
st.markdown("""
<style>
    .metric-pass { color: #00c853 !important; }
    .metric-fail { color: #ff1744 !important; }
    .stMetric { background-color: #1e1e1e; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("🌡️ HyperFOAM: HVAC Digital Twin")
st.markdown("### Series A Demo: Conference Room Simulation")
st.markdown("---")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🎛️ Simulation Parameters")
st.sidebar.markdown("---")

st.sidebar.subheader("Occupancy")
n_people = st.sidebar.slider(
    "Number of Occupants",
    min_value=0,
    max_value=24,
    value=12,
    step=2,
    help="Each person generates 100W of heat and exhales CO2"
)

st.sidebar.subheader("HVAC Settings")
supply_vel = st.sidebar.slider(
    "Supply Velocity (m/s)",
    min_value=0.3,
    max_value=2.0,
    value=0.8,
    step=0.1,
    help="Higher velocity = more air changes but higher draft risk"
)

supply_angle = st.sidebar.slider(
    "Diffuser Angle (°)",
    min_value=15,
    max_value=75,
    value=60,
    step=5,
    help="Angle from vertical - higher = more horizontal spread"
)

supply_temp = st.sidebar.slider(
    "Supply Temperature (°C)",
    min_value=16.0,
    max_value=24.0,
    value=20.0,
    step=0.5,
    help="Temperature of air from supply vents"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Simulation")
duration = st.sidebar.selectbox(
    "Duration (seconds)",
    options=[60, 120, 300, 600],
    index=2,
    help="Physical time to simulate"
)

run_button = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)

# --- MAIN CONTENT ---
if not run_button:
    # Show instructions when idle
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to HyperFOAM
        
        This is a **GPU-accelerated CFD solver** for HVAC simulation.
        
        ### How to Use
        1. Adjust the sliders in the sidebar
        2. Click **Run Simulation**
        3. Watch the room conditions evolve in real-time
        
        ### What's Being Simulated
        - **Navier-Stokes** equations (fluid flow)
        - **Energy transport** (temperature)
        - **Species transport** (CO2 from breathing)
        - **Buoyancy** (hot air rises)
        
        ### ASHRAE 55 Comfort Criteria
        | Metric | Target | Description |
        |--------|--------|-------------|
        | Temperature | 20-24°C | Thermal comfort zone |
        | CO2 | < 1000 ppm | Indoor air quality |
        | Velocity | < 0.25 m/s | Draft risk prevention |
        """)
    
    with col2:
        st.markdown("### Current Settings")
        st.metric("Occupants", f"{n_people} people", f"{n_people * 100}W heat load")
        st.metric("Supply Velocity", f"{supply_vel} m/s")
        st.metric("Diffuser Angle", f"{supply_angle}°")
        st.metric("Supply Temp", f"{supply_temp}°C")

else:
    # --- RUN SIMULATION ---
    
    # Setup placeholders
    status_placeholder = st.empty()
    progress_bar = st.progress(0, text="Initializing...")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        temp_metric = st.empty()
    with col2:
        co2_metric = st.empty()
    with col3:
        vel_metric = st.empty()
    
    plot_col1, plot_col2 = st.columns(2)
    with plot_col1:
        heatmap_placeholder = st.empty()
    with plot_col2:
        history_placeholder = st.empty()
    
    # Initialize solver
    status_placeholder.info("🔧 Initializing Physics Engine...")
    
    config = ConferenceRoom()
    config.supply_velocity = supply_vel
    config.supply_angle = float(supply_angle)
    config.supply_temp = supply_temp
    
    solver = Solver(config)
    setup_conference_room(solver, n_occupants=n_people)
    
    status_placeholder.info("⚡ Running GPU Simulation...")
    
    # History for plots
    time_history = []
    temp_history = []
    co2_history = []
    vel_history = []
    
    # Run simulation with live updates
    steps = int(duration / config.dt)
    log_steps = int(1.0 / config.dt)  # Update every 1 second of sim time
    
    start_time = time.time()
    
    for step in range(steps):
        solver.step()
        
        # Update display every 1 second of physical time
        if step % log_steps == 0:
            t = step * config.dt
            metrics = solver._compute_zone_metrics()
            
            # Store history
            time_history.append(t)
            temp_history.append(metrics['T'])
            co2_history.append(metrics['CO2'])
            vel_history.append(metrics['V'])
            
            # Update progress
            prog = min(1.0, t / duration)
            elapsed = time.time() - start_time
            eta = (elapsed / max(prog, 0.01)) * (1 - prog)
            progress_bar.progress(prog, text=f"t = {t:.0f}s / {duration}s  |  ETA: {eta:.0f}s")
            
            # Update metrics (every 5 seconds)
            if t % 5 < config.dt:
                # Temperature
                temp_pass = 20.0 <= metrics['T'] <= 24.0
                temp_delta = metrics['T'] - 22.0  # Midpoint
                temp_metric.metric(
                    "🌡️ Temperature",
                    f"{metrics['T']:.1f}°C",
                    f"{'✓ PASS' if temp_pass else '✗ FAIL'}",
                    delta_color="normal" if temp_pass else "inverse"
                )
                
                # CO2
                co2_pass = metrics['CO2'] < 1000.0
                co2_metric.metric(
                    "💨 CO2 Level",
                    f"{metrics['CO2']:.0f} ppm",
                    f"{'✓ PASS' if co2_pass else '✗ FAIL'}",
                    delta_color="normal" if co2_pass else "inverse"
                )
                
                # Velocity
                vel_pass = metrics['V'] < 0.25
                vel_metric.metric(
                    "🌬️ Draft Risk",
                    f"{metrics['V']:.3f} m/s",
                    f"{'✓ PASS' if vel_pass else '✗ FAIL'}",
                    delta_color="normal" if vel_pass else "inverse"
                )
            
            # Update plots (every 10 seconds)
            if t % 10 < config.dt and t > 0:
                # Heatmap - mid-plane slice
                if solver.thermal_solver:
                    T_field = solver.thermal_solver.temperature.phi[:, config.ny//2, :].cpu().numpy().T
                    T_field = T_field - 273.15  # Convert to Celsius
                    
                    fig1, ax1 = plt.subplots(figsize=(8, 4))
                    im = ax1.imshow(
                        T_field, 
                        origin='lower', 
                        cmap='RdYlBu_r', 
                        vmin=18, vmax=26,
                        aspect='auto',
                        extent=[0, config.lx, 0, config.lz]
                    )
                    ax1.set_xlabel("X (m)")
                    ax1.set_ylabel("Z (m)")
                    ax1.set_title(f"Temperature Cross-Section (t = {t:.0f}s)")
                    plt.colorbar(im, label="Temperature (°C)")
                    heatmap_placeholder.pyplot(fig1)
                    plt.close(fig1)
                
                # History plot
                fig2, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
                
                # Temperature
                axes[0].plot(time_history, temp_history, 'r-', linewidth=2)
                axes[0].axhspan(20, 24, alpha=0.2, color='green')
                axes[0].set_ylabel("Temp (°C)")
                axes[0].set_ylim(18, 28)
                axes[0].grid(True, alpha=0.3)
                
                # CO2
                axes[1].plot(time_history, co2_history, 'g-', linewidth=2)
                axes[1].axhline(1000, color='red', linestyle='--', alpha=0.5)
                axes[1].set_ylabel("CO2 (ppm)")
                axes[1].grid(True, alpha=0.3)
                
                # Velocity
                axes[2].plot(time_history, vel_history, 'b-', linewidth=2)
                axes[2].axhline(0.25, color='red', linestyle='--', alpha=0.5)
                axes[2].set_ylabel("Velocity (m/s)")
                axes[2].set_xlabel("Time (s)")
                axes[2].grid(True, alpha=0.3)
                
                plt.tight_layout()
                history_placeholder.pyplot(fig2)
                plt.close(fig2)
    
    # Final results
    elapsed = time.time() - start_time
    progress_bar.progress(1.0, text=f"Complete! {steps/elapsed:.0f} steps/s")
    
    final_metrics = solver.get_comfort_metrics()
    
    if final_metrics['overall_pass']:
        status_placeholder.success("✅ **SYSTEM VALIDATED** - All ASHRAE criteria met!")
    else:
        failures = []
        if not final_metrics['temp_pass']:
            failures.append("Temperature")
        if not final_metrics['co2_pass']:
            failures.append("CO2")
        if not final_metrics['velocity_pass']:
            failures.append("Velocity")
        status_placeholder.error(f"⚠️ **TUNING REQUIRED** - Failed: {', '.join(failures)}")
    
    # Summary
    st.markdown("---")
    st.markdown("### Final Results")
    
    result_col1, result_col2, result_col3, result_col4 = st.columns(4)
    
    with result_col1:
        st.metric(
            "Final Temperature",
            f"{final_metrics['temperature']:.2f}°C",
            "✓ PASS" if final_metrics['temp_pass'] else "✗ FAIL"
        )
    
    with result_col2:
        st.metric(
            "Final CO2",
            f"{final_metrics['co2']:.0f} ppm",
            "✓ PASS" if final_metrics['co2_pass'] else "✗ FAIL"
        )
    
    with result_col3:
        st.metric(
            "Final Velocity",
            f"{final_metrics['velocity']:.3f} m/s",
            "✓ PASS" if final_metrics['velocity_pass'] else "✗ FAIL"
        )
    
    with result_col4:
        st.metric(
            "Performance",
            f"{steps/elapsed:.0f} steps/s",
            f"{duration/elapsed:.1f}× real-time"
        )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "HyperFOAM v0.1.0 | GPU-Native CFD for HVAC Digital Twins | TiganticLabz"
    "</div>",
    unsafe_allow_html=True
)
