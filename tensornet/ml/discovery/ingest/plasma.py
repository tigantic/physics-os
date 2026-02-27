#!/usr/bin/env python3
"""
Fusion Plasma Data Ingester for Autonomous Discovery Engine

Converts tokamak/stellarator data into tensor format for discovery.
Supports: Magnetic fields, particle distributions, energy flux, MHD modes.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone


@dataclass
class PlasmaShot:
    """A single plasma discharge/shot."""
    shot_id: str
    device: str = ""  # ITER, JET, DIII-D, etc.
    timestamp: Optional[datetime] = None
    duration_ms: float = 0.0
    plasma_current_kA: float = 0.0
    magnetic_field_T: float = 0.0
    electron_density_m3: float = 0.0
    electron_temp_keV: float = 0.0
    ion_temp_keV: float = 0.0
    stored_energy_MJ: float = 0.0
    confinement_mode: str = ""  # L-mode, H-mode, I-mode
    elm_events: List[Dict] = field(default_factory=list)
    disruption: bool = False
    

@dataclass  
class MagneticField3D:
    """3D magnetic field configuration."""
    B_r: torch.Tensor  # Radial component
    B_phi: torch.Tensor  # Toroidal component
    B_z: torch.Tensor  # Vertical component
    grid_r: torch.Tensor  # R coordinates
    grid_z: torch.Tensor  # Z coordinates
    n_phi: int = 1  # Number of toroidal slices
    

@dataclass
class PlasmaProfile:
    """1D radial plasma profile."""
    rho: torch.Tensor  # Normalized radius (0=core, 1=edge)
    values: torch.Tensor  # Profile values
    profile_type: str = ""  # Te, Ti, ne, q, pressure, etc.
    units: str = ""


class PlasmaIngester:
    """
    Ingest fusion plasma data for discovery analysis.
    
    Converts:
        - Magnetic field → QTT 3D field tensor
        - Particle distribution → QTT PDF
        - Energy flux → time series tensor
        - MHD modes → spectral tensor
        - ELM events → event distribution
    """
    
    def __init__(self, grid_bits: int = 10):
        self.grid_bits = grid_bits
        self.grid_size = 2 ** grid_bits
        
    def from_eqdsk(self, eqdsk_data: Dict) -> MagneticField3D:
        """
        Parse G-EQDSK equilibrium file format.
        
        Standard tokamak equilibrium format containing:
        - Poloidal flux (psi)
        - Pressure profile
        - Safety factor (q) profile
        - Boundary shape
        """
        # Extract grid dimensions
        nr = eqdsk_data.get("nr", 65)
        nz = eqdsk_data.get("nz", 65)
        
        # Get flux function (psi)
        psi = eqdsk_data.get("psi", torch.zeros(nr, nz))
        if isinstance(psi, np.ndarray):
            psi = torch.from_numpy(psi).float()
        
        # Compute B from psi: B_r = -1/R * dpsi/dz, B_z = 1/R * dpsi/dr
        # For now, approximate with gradients
        R = torch.linspace(
            eqdsk_data.get("rmin", 1.0),
            eqdsk_data.get("rmax", 2.5),
            nr
        )
        Z = torch.linspace(
            eqdsk_data.get("zmin", -1.5),
            eqdsk_data.get("zmax", 1.5),
            nz
        )
        
        # Compute gradients using central differences
        dpsi_dr = torch.zeros_like(psi)
        dpsi_dz = torch.zeros_like(psi)
        
        dpsi_dr[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2 * (R[1] - R[0]))
        dpsi_dz[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2 * (Z[1] - Z[0]))
        
        # Poloidal B components from Grad-Shafranov (B = nabla x (psi/R * phi_hat))
        # B_R = -(1/R) * dpsi/dZ, B_Z = (1/R) * dpsi/dR
        R_grid = R.unsqueeze(1).expand(nr, nz)
        B_r = -dpsi_dz / (R_grid + 1e-10)
        B_z = dpsi_dr / (R_grid + 1e-10)
        
        # Toroidal field: B_phi = F(psi) / R
        # Vacuum approximation: F = constant = B0 * R0 (acceptable outside plasma)
        # For full implementation, F(psi) would come from EQDSK fpol array
        B_phi_0 = eqdsk_data.get("bcentr", 5.0)  # Vacuum toroidal field at axis
        R_0 = eqdsk_data.get("rcentr", 1.7)      # Major radius at axis
        B_phi = B_phi_0 * R_0 / (R_grid + 1e-10)
        
        return MagneticField3D(
            B_r=B_r,
            B_phi=B_phi,
            B_z=B_z,
            grid_r=R,
            grid_z=Z,
            n_phi=1,
        )
    
    def build_field_magnitude_tensor(
        self, 
        mag_field: MagneticField3D
    ) -> torch.Tensor:
        """
        Build |B| magnitude field for discovery.
        
        Returns: (nr, nz) tensor of field magnitude
        """
        B_mag = torch.sqrt(
            mag_field.B_r**2 + 
            mag_field.B_phi**2 + 
            mag_field.B_z**2
        )
        return B_mag
    
    def build_q_profile_tensor(
        self,
        q_values: List[float],
        target_length: int = None,
    ) -> torch.Tensor:
        """
        Build safety factor q profile tensor.
        
        q = rB_phi / RB_theta determines MHD stability.
        Rational surfaces (q = m/n) are important.
        """
        if target_length is None:
            target_length = min(self.grid_size, 256)
        
        q = torch.tensor(q_values, dtype=torch.float64)
        
        if len(q) != target_length:
            # Interpolate
            x_old = torch.linspace(0, 1, len(q))
            x_new = torch.linspace(0, 1, target_length)
            q = torch.nn.functional.interpolate(
                q.unsqueeze(0).unsqueeze(0),
                size=target_length,
                mode='linear',
                align_corners=True
            ).squeeze()
        
        return q
    
    def build_profile_distribution(
        self,
        profile: PlasmaProfile,
        n_bins: int = None,
    ) -> torch.Tensor:
        """
        Build distribution tensor from plasma profile.
        
        Returns: (n_bins,) probability distribution
        """
        if n_bins is None:
            n_bins = min(self.grid_size, 512)
        
        values = profile.values
        if len(values) == 0:
            return torch.ones(n_bins) / n_bins
        
        # Normalize
        v_min, v_max = float(values.min()), float(values.max())
        if v_max == v_min:
            return torch.ones(n_bins) / n_bins
        
        values_norm = (values - v_min) / (v_max - v_min)
        
        # Build histogram
        hist = torch.histc(values_norm, bins=n_bins, min=0, max=1)
        hist = hist / (hist.sum() + 1e-10)
        
        return hist
    
    def build_elm_event_distribution(
        self,
        elm_events: List[Dict],
        n_bins: int = 256,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Build ELM (Edge Localized Mode) event distribution.
        
        ELMs are periodic instabilities that cause energy loss.
        Distribution of ELM sizes/frequencies is diagnostic.
        
        Returns:
            (distribution, statistics dict)
        """
        if not elm_events:
            return torch.ones(n_bins) / n_bins, {"count": 0}
        
        # Extract ELM energies
        energies = torch.tensor([
            float(e.get("energy_kJ", e.get("amplitude", 1.0)))
            for e in elm_events
        ])
        
        # Extract inter-ELM times
        times = sorted([float(e.get("time_ms", 0)) for e in elm_events])
        if len(times) > 1:
            inter_elm = torch.tensor([
                times[i+1] - times[i] for i in range(len(times)-1)
            ])
            elm_frequency = 1000.0 / float(inter_elm.mean()) if inter_elm.mean() > 0 else 0
        else:
            inter_elm = torch.tensor([0.0])
            elm_frequency = 0
        
        # Build energy distribution
        if energies.max() > energies.min():
            e_norm = (energies - energies.min()) / (energies.max() - energies.min())
            hist = torch.histc(e_norm, bins=n_bins, min=0, max=1)
            hist = hist / (hist.sum() + 1e-10)
        else:
            hist = torch.ones(n_bins) / n_bins
        
        stats = {
            "count": len(elm_events),
            "mean_energy_kJ": float(energies.mean()),
            "max_energy_kJ": float(energies.max()),
            "frequency_Hz": elm_frequency,
            "mean_inter_elm_ms": float(inter_elm.mean()) if len(inter_elm) > 0 else 0,
        }
        
        return hist, stats
    
    def build_energy_flux_series(
        self,
        power_values: List[float],
        time_values: List[float] = None,
        target_length: int = None,
    ) -> torch.Tensor:
        """
        Build energy flux time series tensor.
        
        Args:
            power_values: Power in MW over time
            time_values: Time points in ms
            target_length: Resample to this length
            
        Returns: (target_length,) tensor
        """
        if target_length is None:
            target_length = min(self.grid_size, len(power_values))
        
        power = torch.tensor(power_values, dtype=torch.float64)
        
        if len(power) != target_length:
            power = torch.nn.functional.interpolate(
                power.unsqueeze(0).unsqueeze(0),
                size=target_length,
                mode='linear',
                align_corners=True
            ).squeeze()
        
        return power
    
    def build_mhd_mode_spectrum(
        self,
        signal: torch.Tensor,
        n_modes: int = 32,
    ) -> torch.Tensor:
        """
        Build MHD mode spectrum from fluctuation signal.
        
        MHD modes (m, n) determine stability.
        Returns: (n_modes,) spectral power distribution
        """
        if len(signal) < n_modes * 2:
            return torch.ones(n_modes) / n_modes
        
        # FFT
        spectrum = torch.fft.rfft(signal)
        power = torch.abs(spectrum[:n_modes]) ** 2
        
        # Normalize
        power = power / (power.sum() + 1e-10)
        
        return power
    
    def ingest_shot(
        self,
        shot: PlasmaShot,
        profiles: Dict[str, PlasmaProfile] = None,
        time_traces: Dict[str, List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Ingest complete plasma shot data.
        
        Returns dict with tensors for discovery pipeline:
            - Te_distribution: Electron temperature profile dist
            - ne_distribution: Density profile dist
            - elm_distribution: ELM energy distribution
            - power_series: Heating power time series
            - stored_energy_series: Stored energy time series
        """
        result = {
            "shot_id": shot.shot_id,
            "device": shot.device,
            "confinement_mode": shot.confinement_mode,
        }
        
        # Process profiles
        if profiles:
            for name, profile in profiles.items():
                result[f"{name}_distribution"] = self.build_profile_distribution(profile)
        
        # Process time traces
        if time_traces:
            for name, values in time_traces.items():
                result[f"{name}_series"] = self.build_energy_flux_series(values)
        
        # Process ELM events
        elm_dist, elm_stats = self.build_elm_event_distribution(shot.elm_events)
        result["elm_distribution"] = elm_dist
        result["elm_stats"] = elm_stats
        
        # Add scalar parameters as tensor
        result["parameters"] = torch.tensor([
            shot.plasma_current_kA,
            shot.magnetic_field_T,
            shot.electron_density_m3 / 1e19,  # Normalize to 10^19 m^-3
            shot.electron_temp_keV,
            shot.ion_temp_keV,
            shot.stored_energy_MJ,
        ])
        
        return result
    
    def compute_confinement_time(
        self,
        stored_energy_MJ: float,
        heating_power_MW: float,
    ) -> float:
        """Compute energy confinement time tau_E = W / P."""
        if heating_power_MW <= 0:
            return 0.0
        return stored_energy_MJ / heating_power_MW * 1000  # in ms
    
    def detect_h_mode_transition(
        self,
        d_alpha_signal: torch.Tensor,
        threshold_drop: float = 0.5,
    ) -> List[int]:
        """
        Detect L-H transition from D-alpha signal.
        
        H-mode transition shows sudden drop in D-alpha emission.
        Returns: List of time indices where transitions detected
        """
        if len(d_alpha_signal) < 10:
            return []
        
        transitions = []
        
        # Look for sudden drops
        for i in range(5, len(d_alpha_signal) - 5):
            before = float(d_alpha_signal[i-5:i].mean())
            after = float(d_alpha_signal[i:i+5].mean())
            
            if before > 0 and (after / before) < threshold_drop:
                transitions.append(i)
        
        return transitions
    
    def compute_pressure_gradient(
        self,
        pressure_profile: PlasmaProfile,
    ) -> torch.Tensor:
        """
        Compute pressure gradient dp/dr.
        
        Important for stability analysis (peeling-ballooning).
        """
        p = pressure_profile.values
        rho = pressure_profile.rho
        
        if len(p) < 3:
            return torch.zeros(1)
        
        # Central difference
        dp_drho = torch.zeros_like(p)
        dr = rho[1] - rho[0] if len(rho) > 1 else 1.0
        dp_drho[1:-1] = (p[2:] - p[:-2]) / (2 * dr)
        dp_drho[0] = (p[1] - p[0]) / dr
        dp_drho[-1] = (p[-1] - p[-2]) / dr
        
        return dp_drho


def main():
    """Test the Plasma ingester."""
    ingester = PlasmaIngester(grid_bits=10)
    
    print("Testing Plasma Ingester...")
    
    # Create synthetic plasma shot
    shot = PlasmaShot(
        shot_id="TEST-001",
        device="SYNTHETIC",
        plasma_current_kA=1500,
        magnetic_field_T=5.3,
        electron_density_m3=8e19,
        electron_temp_keV=10.0,
        ion_temp_keV=8.0,
        stored_energy_MJ=350,
        confinement_mode="H-mode",
        elm_events=[
            {"time_ms": 100, "energy_kJ": 50},
            {"time_ms": 150, "energy_kJ": 45},
            {"time_ms": 210, "energy_kJ": 55},
            {"time_ms": 260, "energy_kJ": 48},
            {"time_ms": 320, "energy_kJ": 52},
        ]
    )
    
    # Create synthetic profiles
    rho = torch.linspace(0, 1, 100)
    Te = 10 * (1 - rho**2)**2  # Parabolic-ish profile
    
    profiles = {
        "Te": PlasmaProfile(rho=rho, values=Te, profile_type="Te", units="keV"),
    }
    
    time_traces = {
        "power": [10 + 0.5 * np.sin(i * 0.1) for i in range(200)],
    }
    
    result = ingester.ingest_shot(shot, profiles, time_traces)
    
    print(f"✓ Shot ingested: {result['shot_id']}")
    print(f"✓ Te distribution: shape={result['Te_distribution'].shape}")
    print(f"✓ ELM events: {result['elm_stats']['count']} at {result['elm_stats']['frequency_Hz']:.1f} Hz")
    print(f"✓ Parameters: {result['parameters'].shape}")
    
    # Test q profile
    q_values = [1.0 + i * 0.05 for i in range(50)]
    q_profile = ingester.build_q_profile_tensor(q_values)
    print(f"✓ q profile: shape={q_profile.shape}, q_95≈{float(q_profile[-5]):.2f}")
    
    # Test ELM distribution
    elm_dist, elm_stats = ingester.build_elm_event_distribution(shot.elm_events)
    print(f"✓ ELM distribution: shape={elm_dist.shape}, sum={float(elm_dist.sum()):.4f}")
    
    # Test MHD spectrum
    mhd_signal = torch.randn(1024)
    spectrum = ingester.build_mhd_mode_spectrum(mhd_signal)
    print(f"✓ MHD spectrum: shape={spectrum.shape}")
    
    # Test confinement time
    tau_E = ingester.compute_confinement_time(350, 50)
    print(f"✓ Confinement time: τ_E = {tau_E:.1f} ms")
    
    print("\n✅ Plasma Ingester ready for Phase 2")


if __name__ == "__main__":
    main()
