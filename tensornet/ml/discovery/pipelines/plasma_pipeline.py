#!/usr/bin/env python3
"""
Fusion Plasma Discovery Pipeline

Chains Genesis primitives to analyze tokamak/stellarator data.
Detects: ELM precursors, H-mode transitions, MHD instabilities, confinement anomalies.

Uses the proven V2 engine architecture from DeFi pipeline.

Integration:
    - tensornet.fusion.tokamak: Boris pusher particle simulation
    - tensornet.fusion.marrs_simulator: MARRS solid-state fusion
    - tomahawk_cfd_gauntlet: TT-compressed MHD instability control
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import json

# Discovery engine imports
from ..engine_v2 import DiscoveryEngineV2 as DiscoveryEngine
from ..findings import Finding
from ..hypothesis import HypothesisGenerator

# Ingester
from ..ingest.plasma import (
    PlasmaIngester,
    PlasmaShot,
    PlasmaProfile,
    MagneticField3D,
)

# Fusion physics modules (Phase 9/21/22)
from tensornet.plasma_nuclear.fusion import (
    TokamakReactor,
    ConfinementReport,
    PlasmaState as FusionPlasmaState,
)


@dataclass
class PlasmaPipelineResult:
    """Results from plasma discovery pipeline."""
    shot_id: str
    device: str
    confinement_mode: str
    findings: List[Dict] = field(default_factory=list)
    hypotheses: List[Dict] = field(default_factory=list)
    statistics: Dict = field(default_factory=dict)
    primitive_results: Dict = field(default_factory=dict)
    execution_time_ms: float = 0.0
    timestamp: str = ""


class PlasmaDiscoveryPipeline:
    """
    Autonomous discovery pipeline for fusion plasma analysis.
    
    Analysis Focus:
        1. ELM cycle analysis (energy burst patterns)
        2. Confinement mode detection (L/H/I-mode)
        3. MHD stability assessment
        4. Profile consistency checks
        5. Disruption precursor detection
        
    Primitives Used:
        - OT: Distribution drift (plasma profiles over time)
        - SGW: Multi-scale MHD mode structure
        - RKHS: Anomaly detection vs baseline shots
        - PH: Topological features in parameter space
        - GA: Geometric invariants (rotational symmetry)
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.ingester = PlasmaIngester()
        self.engine = DiscoveryEngine()
        self.hypothesis_generator = HypothesisGenerator()
        
        # Baseline thresholds (from ITER/JET operational limits)
        self.thresholds = {
            "elm_frequency_high_Hz": 100,  # Type-I ELMs typically 50-100 Hz
            "elm_frequency_low_Hz": 10,    # Grassy ELMs
            "elm_energy_large_kJ": 100,    # Large ELM threshold
            "q95_min": 3.0,                # Minimum edge safety factor
            "greenwald_fraction_max": 0.85, # Density limit
            "beta_limit": 3.5,             # Troyon beta limit factor
            "mhd_mode_threshold": 0.3,     # MHD mode amplitude warning
            "distribution_shift_threshold": 0.3,
            "mmd_anomaly_threshold": 0.5,
        }
    
    def analyze_shot(
        self,
        shot: PlasmaShot,
        profiles: Dict[str, PlasmaProfile] = None,
        time_traces: Dict[str, List[float]] = None,
        baseline_shots: List[Dict] = None,
    ) -> PlasmaPipelineResult:
        """
        Run full discovery pipeline on a plasma shot.
        
        Args:
            shot: Plasma shot metadata
            profiles: Dict of plasma profiles (Te, ne, q, pressure, etc.)
            time_traces: Dict of time series (power, stored_energy, etc.)
            baseline_shots: Reference shots for comparison
            
        Returns:
            PlasmaPipelineResult with findings and hypotheses
        """
        import time
        start_time = time.time()
        
        result = PlasmaPipelineResult(
            shot_id=shot.shot_id,
            device=shot.device,
            confinement_mode=shot.confinement_mode,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        
        # Stage 1: Ingest data
        if self.verbose:
            print(f"[PLASMA] Ingesting shot {shot.shot_id}...")
        
        ingested = self.ingester.ingest_shot(shot, profiles, time_traces)
        
        findings = []
        primitive_results = {}
        
        # Stage 2: Analyze ELM distribution
        if self.verbose:
            print("[PLASMA] Stage 2: ELM analysis...")
        
        elm_findings = self._analyze_elm_pattern(shot, ingested)
        findings.extend(elm_findings)
        
        # Stage 3: Profile analysis with OT (distribution shift)
        if profiles and self.verbose:
            print("[PLASMA] Stage 3: Profile analysis (OT)...")
        
        profile_findings, ot_results = self._analyze_profiles_ot(profiles, ingested)
        findings.extend(profile_findings)
        primitive_results["ot"] = ot_results
        
        # Stage 4: MHD mode spectrum (SGW - multi-scale)
        if time_traces and self.verbose:
            print("[PLASMA] Stage 4: MHD mode analysis (SGW)...")
        
        mhd_findings, sgw_results = self._analyze_mhd_modes_sgw(time_traces)
        findings.extend(mhd_findings)
        primitive_results["sgw"] = sgw_results
        
        # Stage 5: Anomaly detection vs baseline (RKHS)
        if baseline_shots and self.verbose:
            print("[PLASMA] Stage 5: Anomaly detection (RKHS)...")
        
        anomaly_findings, rkhs_results = self._detect_anomalies_rkhs(
            ingested, baseline_shots
        )
        findings.extend(anomaly_findings)
        primitive_results["rkhs"] = rkhs_results
        
        # Stage 6: Parameter space topology (PH)
        if self.verbose:
            print("[PLASMA] Stage 6: Parameter topology (PH)...")
        
        topo_findings, ph_results = self._analyze_topology_ph(ingested)
        findings.extend(topo_findings)
        primitive_results["ph"] = topo_findings
        
        # Stage 7: Geometric symmetry analysis (GA)
        if self.verbose:
            print("[PLASMA] Stage 7: Symmetry analysis (GA)...")
        
        sym_findings, ga_results = self._analyze_symmetry_ga(ingested)
        findings.extend(sym_findings)
        primitive_results["ga"] = ga_results
        
        # Stage 8: Safety factor analysis
        if self.verbose:
            print("[PLASMA] Stage 8: Safety factor analysis...")
        
        q_findings = self._analyze_safety_factor(shot)
        findings.extend(q_findings)
        
        # Stage 9: Disruption risk assessment
        if self.verbose:
            print("[PLASMA] Stage 9: Disruption risk assessment...")
        
        disruption_findings = self._assess_disruption_risk(shot, ingested)
        findings.extend(disruption_findings)
        
        # Generate hypotheses
        if self.verbose:
            print("[PLASMA] Generating hypotheses...")
        
        hypotheses = self._generate_hypotheses(findings, shot)
        
        # Compile statistics
        stats = self._compile_statistics(shot, ingested, findings)
        
        execution_time = (time.time() - start_time) * 1000
        
        result.findings = [f for f in findings]
        result.hypotheses = hypotheses
        result.statistics = stats
        result.primitive_results = primitive_results
        result.execution_time_ms = execution_time
        
        if self.verbose:
            print(f"[PLASMA] Complete: {len(findings)} findings, "
                  f"{len(hypotheses)} hypotheses in {execution_time:.1f}ms")
        
        return result
    
    def simulate_confinement(
        self,
        shot: PlasmaShot,
        n_particles: int = 1000,
        n_steps: int = 500,
        dt: float = 1e-9,
    ) -> Tuple[ConfinementReport, List[Dict]]:
        """
        Run Boris pusher particle simulation for confinement analysis.
        
        Uses tensornet.fusion.TokamakReactor for real particle-in-cell
        simulation of plasma confinement.
        
        Args:
            shot: Plasma shot with geometry parameters
            n_particles: Number of test particles
            n_steps: Simulation timesteps
            dt: Time step in seconds
            
        Returns:
            (ConfinementReport, findings list)
        """
        findings = []
        
        # Extract geometry from shot
        # Default to ITER-like if not specified
        major_radius = getattr(shot, 'major_radius_m', 6.2)
        minor_radius = getattr(shot, 'minor_radius_m', 2.0)
        B0 = shot.magnetic_field_T if shot.magnetic_field_T > 0 else 5.3
        
        # Estimate safety factor from shot parameters
        if shot.plasma_current_kA > 0 and shot.magnetic_field_T > 0:
            q = 5 * minor_radius**2 * B0 / (major_radius * shot.plasma_current_kA / 1000)
        else:
            q = 3.0  # Default safe value
        
        # Create reactor simulation
        reactor = TokamakReactor(
            major_radius=major_radius,
            minor_radius=minor_radius,
            B0=B0,
            safety_factor=q,
        )
        
        # Create plasma with thermal properties from shot
        temperature = shot.electron_temp_keV if shot.electron_temp_keV > 0 else 10.0
        particles = reactor.create_plasma(
            num_particles=n_particles,
            temperature=temperature,
            toroidal_flow=10.0,
            seed=42,
        )
        
        # Run Boris pusher simulation
        final_particles, escape_history = reactor.push_particles(
            particles,
            dt=dt,
            steps=n_steps,
            q_over_m=1.0,
            verbose=self.verbose,
        )
        
        # Analyze confinement
        report = reactor.analyze_confinement(final_particles, escape_history)
        
        # Generate findings from confinement report
        if report.confinement_ratio < 0.5:
            findings.append({
                "type": "CONFINEMENT_FAILURE",
                "severity": "critical",
                "primitive": "boris_pusher",
                "description": f"Particle simulation shows {report.confinement_ratio*100:.1f}% confinement",
                "value": report.confinement_ratio,
                "threshold": 0.5,
                "recommendation": report.recommendation,
            })
        elif report.confinement_ratio < 0.8:
            findings.append({
                "type": "CONFINEMENT_MARGINAL",
                "severity": "warning",
                "primitive": "boris_pusher",
                "description": f"Marginal confinement: {report.confinement_ratio*100:.1f}%",
                "value": report.confinement_ratio,
                "threshold": 0.8,
                "recommendation": report.recommendation,
            })
        else:
            findings.append({
                "type": "CONFINEMENT_GOOD",
                "severity": "info",
                "primitive": "boris_pusher",
                "description": f"Good confinement: {report.confinement_ratio*100:.1f}%",
                "value": report.confinement_ratio,
                "status": report.status,
            })
        
        # Check for edge losses
        if report.max_rho > minor_radius * 0.9:
            findings.append({
                "type": "EDGE_LOSS_RISK",
                "severity": "warning",
                "primitive": "boris_pusher",
                "description": f"Particles reaching ρ={report.max_rho:.3f}m (edge at {minor_radius:.2f}m)",
                "value": report.max_rho,
                "threshold": minor_radius * 0.9,
            })
        
        return report, findings

    def _analyze_elm_pattern(
        self, 
        shot: PlasmaShot,
        ingested: Dict,
    ) -> List[Dict]:
        """Analyze ELM (Edge Localized Mode) patterns."""
        findings = []
        elm_stats = ingested.get("elm_stats", {})
        
        if elm_stats.get("count", 0) == 0:
            return findings
        
        frequency = elm_stats.get("frequency_Hz", 0)
        mean_energy = elm_stats.get("mean_energy_kJ", 0)
        max_energy = elm_stats.get("max_energy_kJ", 0)
        
        # Check ELM frequency regime
        if frequency > self.thresholds["elm_frequency_high_Hz"]:
            findings.append({
                "type": "ELM_HIGH_FREQUENCY",
                "severity": "info",
                "primitive": "statistical",
                "description": f"High ELM frequency: {frequency:.1f} Hz (grassy regime)",
                "value": frequency,
                "threshold": self.thresholds["elm_frequency_high_Hz"],
            })
        elif frequency < self.thresholds["elm_frequency_low_Hz"] and frequency > 0:
            findings.append({
                "type": "ELM_LOW_FREQUENCY",
                "severity": "warning",
                "primitive": "statistical",
                "description": f"Low ELM frequency: {frequency:.1f} Hz (Type-I regime, larger ELMs)",
                "value": frequency,
                "threshold": self.thresholds["elm_frequency_low_Hz"],
            })
        
        # Check ELM size
        if max_energy > self.thresholds["elm_energy_large_kJ"]:
            findings.append({
                "type": "ELM_LARGE_ENERGY",
                "severity": "critical",
                "primitive": "statistical",
                "description": f"Large ELM detected: {max_energy:.1f} kJ (wall damage risk)",
                "value": max_energy,
                "threshold": self.thresholds["elm_energy_large_kJ"],
            })
        
        # ELM distribution analysis using pipeline
        elm_dist = ingested.get("elm_distribution")
        if elm_dist is not None and elm_dist.numel() > 0:
            # Check for bimodal distribution (different ELM types)
            peaks = self._find_distribution_peaks(elm_dist)
            if len(peaks) > 1:
                findings.append({
                    "type": "ELM_BIMODAL",
                    "severity": "info",
                    "primitive": "OT",
                    "description": f"Bimodal ELM distribution: {len(peaks)} peaks detected",
                    "value": len(peaks),
                })
        
        return findings
    
    def _analyze_profiles_ot(
        self,
        profiles: Dict[str, PlasmaProfile],
        ingested: Dict,
    ) -> Tuple[List[Dict], Dict]:
        """Analyze profile consistency using Optimal Transport."""
        findings = []
        ot_results = {}
        
        if not profiles:
            return findings, ot_results
        
        # Compare Te and Ti profiles (should be similar in hot plasmas)
        Te_dist = ingested.get("Te_distribution")
        Ti_dist = ingested.get("Ti_distribution")
        
        if Te_dist is not None and Ti_dist is not None:
            # Compute Wasserstein-2 distance using quantile (sliced) approach
            # W2^2 = integral_0^1 (F_Te^{-1}(p) - F_Ti^{-1}(p))^2 dp
            # Approximated via empirical quantiles
            
            # Normalize to probability distributions
            Te_prob = Te_dist / (Te_dist.sum() + 1e-10)
            Ti_prob = Ti_dist / (Ti_dist.sum() + 1e-10)
            
            # Compute CDFs
            Te_cdf = torch.cumsum(Te_prob, dim=0)
            Ti_cdf = torch.cumsum(Ti_prob, dim=0)
            
            # Quantile function approximation via sorting
            n_quantiles = min(100, len(Te_dist))
            quantile_levels = torch.linspace(0.01, 0.99, n_quantiles)
            
            # Find quantile values (inverse CDF)
            Te_quantiles = torch.zeros(n_quantiles)
            Ti_quantiles = torch.zeros(n_quantiles)
            
            for i, q in enumerate(quantile_levels):
                Te_quantiles[i] = (Te_cdf >= q).float().argmax().float()
                Ti_quantiles[i] = (Ti_cdf >= q).float().argmax().float()
            
            # W2 distance: RMS of quantile differences
            W2 = float(torch.sqrt(((Te_quantiles - Ti_quantiles) ** 2).mean()))
            
            # Normalize by profile length
            W2 = W2 / len(Te_dist)
            
            ot_result = {"wasserstein_distance": float(W2), "method": "quantile-W2"}
            ot_results["Te_Ti_transport"] = ot_result
            
            if float(W2) > self.thresholds["distribution_shift_threshold"]:
                findings.append({
                    "type": "PROFILE_DECOUPLING",
                    "severity": "warning",
                    "primitive": "OT",
                    "description": f"Te-Ti decoupling detected: W₂ ≈ {float(W2):.3f}",
                    "value": float(W2),
                    "threshold": self.thresholds["distribution_shift_threshold"],
                })
        
        # Check pressure gradient (pedestal stability)
        if "pressure" in profiles:
            pressure_profile = profiles["pressure"]
            grad = self.ingester.compute_pressure_gradient(pressure_profile)
            max_grad = float(grad.abs().max())
            
            # High gradient at edge indicates strong pedestal
            edge_grad = float(grad[-10:].abs().mean()) if len(grad) > 10 else 0
            
            ot_results["pressure_gradient"] = {
                "max_gradient": max_grad,
                "edge_gradient": edge_grad,
            }
            
            if edge_grad > max_grad * 0.5:
                findings.append({
                    "type": "STRONG_PEDESTAL",
                    "severity": "info",
                    "primitive": "OT",
                    "description": f"Strong edge pressure gradient: {edge_grad:.2f} (pedestal region)",
                    "value": edge_grad,
                })
        
        return findings, ot_results
    
    def _analyze_mhd_modes_sgw(
        self,
        time_traces: Dict[str, List[float]],
    ) -> Tuple[List[Dict], Dict]:
        """Analyze MHD modes using Spectral Graph Wavelets (multi-scale)."""
        findings = []
        sgw_results = {}
        
        if not time_traces:
            return findings, sgw_results
        
        # Look for magnetic fluctuation signals
        for trace_name, values in time_traces.items():
            if len(values) < 64:
                continue
            
            signal = torch.tensor(values, dtype=torch.float64)
            
            # Build MHD spectrum
            spectrum = self.ingester.build_mhd_mode_spectrum(signal, n_modes=32)
            
            sgw_results[trace_name] = {
                "spectrum": spectrum.tolist()[:8],  # Top 8 modes
                "dominant_mode": int(spectrum.argmax()),
                "peak_power": float(spectrum.max()),
            }
            
            # Multi-scale energy analysis
            scales = self._compute_multiscale_energy(signal)
            sgw_results[f"{trace_name}_scales"] = scales
            
            # Check for dominant mode
            if float(spectrum.max()) > self.thresholds["mhd_mode_threshold"]:
                mode_num = int(spectrum.argmax())
                findings.append({
                    "type": "MHD_MODE_DOMINANT",
                    "severity": "warning",
                    "primitive": "SGW",
                    "description": f"Dominant MHD mode n={mode_num} in {trace_name} (power={float(spectrum.max()):.3f})",
                    "value": float(spectrum.max()),
                    "mode_number": mode_num,
                    "trace": trace_name,
                })
            
            # Check for energy concentration at specific scale
            if len(scales) > 2:
                max_scale = max(range(len(scales)), key=lambda i: scales[i])
                if scales[max_scale] > 0.5:  # >50% energy at one scale
                    findings.append({
                        "type": "ENERGY_LOCALIZATION",
                        "severity": "info",
                        "primitive": "SGW",
                        "description": f"Energy localized at scale 2^{max_scale} in {trace_name}",
                        "value": scales[max_scale],
                        "scale": max_scale,
                    })
        
        return findings, sgw_results
    
    def _detect_anomalies_rkhs(
        self,
        current_shot: Dict,
        baseline_shots: List[Dict],
    ) -> Tuple[List[Dict], Dict]:
        """Detect anomalies using RKHS kernel methods (MMD)."""
        findings = []
        rkhs_results = {}
        
        if not baseline_shots:
            return findings, rkhs_results
        
        # Compare parameter vectors
        current_params = current_shot.get("parameters")
        if current_params is None:
            return findings, rkhs_results
        
        baseline_params = torch.stack([
            torch.tensor(s.get("parameters", [0]*6)) 
            for s in baseline_shots if "parameters" in s
        ])
        
        if len(baseline_params) == 0:
            return findings, rkhs_results
        
        # Compute MMD between current and baseline using Genesis RKHS primitive
        from tensornet.genesis.rkhs import RBFKernel
        
        kernel = RBFKernel(length_scale=1.0, variance=1.0)
        current = current_params.unsqueeze(0).float()
        baseline = baseline_params.float()
        
        # Compute MMD: E[k(X,X')] + E[k(Y,Y')] - 2*E[k(X,Y)]
        K_xx = kernel(current, current)  # (1, 1)
        K_yy = kernel(baseline, baseline)  # (m, m)
        K_xy = kernel(current, baseline)  # (1, m)
        
        mmd_squared = float(K_xx.mean()) + float(K_yy.mean()) - 2 * float(K_xy.mean())
        mmd_score = max(0, mmd_squared) ** 0.5
        
        mmd_result = {"mmd_score": mmd_score}
        rkhs_results["mmd"] = mmd_result
        
        if mmd_score > self.thresholds["mmd_anomaly_threshold"]:
            findings.append({
                "type": "PARAMETER_ANOMALY",
                "severity": "critical",
                "primitive": "RKHS",
                "description": f"Shot parameters anomalous vs baseline: MMD = {mmd_score:.3f}",
                "value": mmd_score,
                "threshold": self.thresholds["mmd_anomaly_threshold"],
            })
        
        return findings, rkhs_results
    
    def _analyze_topology_ph(
        self,
        ingested: Dict,
    ) -> Tuple[List[Dict], Dict]:
        """Analyze topological features using Persistent Homology."""
        findings = []
        
        # Build parameter point cloud
        params = ingested.get("parameters")
        if params is None or len(params) < 3:
            return findings, {}
        
        # Create synthetic point cloud from parameters
        # In real usage, this would be multi-shot parameter space
        point_cloud = params.unsqueeze(0).expand(10, -1).clone()  # Minimal cloud
        noise = torch.randn_like(point_cloud) * 0.1
        point_cloud = point_cloud + noise
        
        # Compute Betti numbers using Genesis PH primitive
        from tensornet.genesis.topology import RipsComplex
        from tensornet.genesis.topology.boundary import betti_numbers_from_boundary
        
        complex_result = RipsComplex.from_points(
            point_cloud.float(), 
            max_radius=2.0, 
            max_dim=2
        )
        betti = betti_numbers_from_boundary(complex_result)
        
        ph_result = {"betti_numbers": betti}
        
        if len(betti) > 1 and betti[1] > 0:
            findings.append({
                "type": "TOPOLOGICAL_CYCLE",
                "severity": "info",
                "primitive": "PH",
                "description": f"Parameter space has {betti[1]} topological cycles (β₁)",
                "value": betti[1],
            })
        
        return findings, ph_result
    
    def _analyze_symmetry_ga(
        self,
        ingested: Dict,
    ) -> Tuple[List[Dict], Dict]:
        """Analyze geometric symmetry using Geometric Algebra."""
        findings = []
        
        params = ingested.get("parameters")
        if params is None or len(params) < 3:
            return findings, {}
        
        # Use first 3 parameters as 3D point
        point = params[:3].tolist()
        
        # Compute geometric invariants using Genesis GA primitive
        from tensornet.genesis.ga import CliffordAlgebra, vector
        
        ga = CliffordAlgebra(p=3, q=0, r=0)  # 3D Euclidean
        mv = vector(ga, point)
        
        norm = mv.norm()
        grade_0 = float(mv.scalar_part())
        
        ga_result = {"norm": norm, "grade_0": grade_0}
        
        # Check for rotational invariant
        if abs(norm) > 0:
            findings.append({
                "type": "GEOMETRIC_INVARIANT",
                "severity": "info",
                "primitive": "GA",
                "description": f"Geometric norm: {norm:.3f}, scalar component: {grade_0:.3f}",
                "value": norm,
            })
        
        return findings, ga_result
    
    def _analyze_safety_factor(self, shot: PlasmaShot) -> List[Dict]:
        """Analyze safety factor q for MHD stability."""
        findings = []
        
        # q95 analysis (safety factor at 95% flux surface)
        # In real data, this would come from equilibrium reconstruction
        # Estimate from parameters
        if shot.plasma_current_kA > 0 and shot.magnetic_field_T > 0:
            # Rough q95 estimate: q95 ≈ 5 * a^2 * B_T / (R * I_p)
            # Assume typical aspect ratio
            R = 6.2  # ITER major radius (m)
            a = 2.0  # ITER minor radius (m)
            q95_estimate = 5 * a**2 * shot.magnetic_field_T / (R * shot.plasma_current_kA / 1000)
            
            if q95_estimate < self.thresholds["q95_min"]:
                findings.append({
                    "type": "Q95_LOW",
                    "severity": "critical",
                    "primitive": "physics",
                    "description": f"Low safety factor q₉₅ ≈ {q95_estimate:.2f} (disruption risk)",
                    "value": q95_estimate,
                    "threshold": self.thresholds["q95_min"],
                })
        
        return findings
    
    def _assess_disruption_risk(
        self, 
        shot: PlasmaShot,
        ingested: Dict,
    ) -> List[Dict]:
        """Assess overall disruption risk."""
        findings = []
        risk_score = 0.0
        risk_factors = []
        
        # Already disrupted
        if shot.disruption:
            return [{
                "type": "DISRUPTION_OCCURRED",
                "severity": "critical",
                "primitive": "statistical",
                "description": "Plasma disruption recorded in this shot",
                "value": 1.0,
            }]
        
        # Check Greenwald density limit
        if shot.electron_density_m3 > 0 and shot.plasma_current_kA > 0:
            # Greenwald limit: n_GW = I_p / (π a^2) in 10^20 m^-3
            a = 2.0  # Assume ITER-like
            n_gw = shot.plasma_current_kA / (np.pi * a**2) * 1e20
            greenwald_fraction = shot.electron_density_m3 / n_gw
            
            if greenwald_fraction > self.thresholds["greenwald_fraction_max"]:
                risk_score += 0.3
                risk_factors.append(f"Greenwald fraction {greenwald_fraction:.2f}")
        
        # Check ELM impact
        elm_stats = ingested.get("elm_stats", {})
        if elm_stats.get("max_energy_kJ", 0) > self.thresholds["elm_energy_large_kJ"]:
            risk_score += 0.2
            risk_factors.append("Large ELM events")
        
        # Low q95
        if shot.plasma_current_kA > 0 and shot.magnetic_field_T > 0:
            R, a = 6.2, 2.0
            q95_est = 5 * a**2 * shot.magnetic_field_T / (R * shot.plasma_current_kA / 1000)
            if q95_est < self.thresholds["q95_min"]:
                risk_score += 0.4
                risk_factors.append(f"Low q95 ({q95_est:.2f})")
        
        if risk_score > 0.3:
            findings.append({
                "type": "DISRUPTION_RISK",
                "severity": "critical" if risk_score > 0.5 else "warning",
                "primitive": "composite",
                "description": f"Disruption risk score: {risk_score:.2f} - Factors: {', '.join(risk_factors)}",
                "value": risk_score,
                "factors": risk_factors,
            })
        
        return findings
    
    def _generate_hypotheses(
        self,
        findings: List[Dict],
        shot: PlasmaShot,
    ) -> List[Dict]:
        """Generate hypotheses from findings using hypothesis generator."""
        if not findings:
            return []
        
        # Convert findings to engine format for hypothesis generator
        from ..engine_v2 import Finding as EngineFinding, DiscoveryResult
        
        engine_findings = []
        for f in findings:
            # Map severity string to expected format
            sev_map = {
                "info": "INFO",
                "warning": "MEDIUM",
                "critical": "CRITICAL",
            }
            severity = sev_map.get(f.get("severity", "info"), "INFO")
            
            engine_findings.append(EngineFinding(
                primitive=f.get("primitive", "unknown"),
                severity=severity,
                summary=f.get("description", "Unknown finding"),
                evidence={
                    "value": f.get("value", 0),
                    "threshold": f.get("threshold"),
                    "type": f.get("type", "unknown"),
                },
            ))
        
        # Create a DiscoveryResult wrapper
        discovery_result = DiscoveryResult(findings=engine_findings)
        
        hypotheses = self.hypothesis_generator.generate(discovery_result)
        
        # Add plasma-specific hypotheses
        plasma_hypotheses = []
        
        # Check for ELM mitigation opportunity
        elm_findings = [f for f in findings if "ELM" in f.get("type", "")]
        if elm_findings:
            plasma_hypotheses.append({
                "id": "plasma-h1",
                "type": "elm_mitigation",
                "confidence": 0.7,
                "summary": "ELM mitigation may be beneficial",
                "details": f"Detected {len(elm_findings)} ELM-related findings. "
                          "Consider RMP (Resonant Magnetic Perturbation) or pellet injection.",
                "action": "Apply ELM mitigation techniques",
            })
        
        # Check for confinement mode transition
        profile_findings = [f for f in findings if "PROFILE" in f.get("type", "") or "PEDESTAL" in f.get("type", "")]
        if profile_findings and shot.confinement_mode == "L-mode":
            plasma_hypotheses.append({
                "id": "plasma-h2",
                "type": "h_mode_access",
                "confidence": 0.6,
                "summary": "H-mode access possible",
                "details": "Profile structure suggests approaching H-mode threshold. "
                          "Increase heating power to trigger L-H transition.",
                "action": "Increase NBI/ECRH power",
            })
        
        # Combine with standard hypotheses
        result = plasma_hypotheses
        for h in hypotheses[:3]:  # Top 3 from generator
            result.append({
                "id": h.hypothesis_id if hasattr(h, 'hypothesis_id') else h.id,
                "type": h.hypothesis_type if hasattr(h, 'hypothesis_type') else "general",
                "confidence": h.confidence,
                "summary": h.title,
                "details": h.description,
                "action": h.recommended_action,
            })
        
        return result
    
    def _compile_statistics(
        self,
        shot: PlasmaShot,
        ingested: Dict,
        findings: List[Dict],
    ) -> Dict:
        """Compile pipeline statistics."""
        elm_stats = ingested.get("elm_stats", {})
        
        # Compute confinement time if we have the data
        tau_E = 0
        if shot.stored_energy_MJ > 0:
            # Estimate heating power from plasma current
            P_heat = shot.plasma_current_kA * 0.01  # Very rough estimate
            if P_heat > 0:
                tau_E = self.ingester.compute_confinement_time(
                    shot.stored_energy_MJ, P_heat
                )
        
        return {
            "shot_id": shot.shot_id,
            "device": shot.device,
            "confinement_mode": shot.confinement_mode,
            "plasma_current_kA": shot.plasma_current_kA,
            "magnetic_field_T": shot.magnetic_field_T,
            "stored_energy_MJ": shot.stored_energy_MJ,
            "confinement_time_ms": tau_E,
            "elm_count": elm_stats.get("count", 0),
            "elm_frequency_Hz": elm_stats.get("frequency_Hz", 0),
            "findings_count": len(findings),
            "critical_findings": len([f for f in findings if f.get("severity") == "critical"]),
            "warning_findings": len([f for f in findings if f.get("severity") == "warning"]),
        }
    
    def _find_distribution_peaks(self, dist: torch.Tensor, min_height: float = 0.02) -> List[int]:
        """Find peaks in a distribution."""
        peaks = []
        for i in range(1, len(dist) - 1):
            if dist[i] > dist[i-1] and dist[i] > dist[i+1] and dist[i] > min_height:
                peaks.append(i)
        return peaks
    
    def _compute_multiscale_energy(self, signal: torch.Tensor, n_scales: int = 5) -> List[float]:
        """Compute energy at different scales (wavelet-like)."""
        energies = []
        current = signal.float()
        
        for _ in range(n_scales):
            if len(current) < 4:
                break
            
            # High-pass (detail)
            detail = current[1:] - current[:-1]
            energy = float((detail ** 2).sum())
            energies.append(energy)
            
            # Low-pass (downsample)
            if len(current) % 2 == 1:
                current = current[:-1]
            current = (current[::2] + current[1::2]) / 2
        
        # Normalize
        total = sum(energies) + 1e-10
        return [e / total for e in energies]
    
    def generate_report(
        self,
        result: PlasmaPipelineResult,
        format: str = "markdown",
    ) -> str:
        """Generate analysis report."""
        if format == "markdown":
            return self._generate_markdown_report(result)
        elif format == "json":
            return self._generate_json_report(result)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def _generate_markdown_report(self, result: PlasmaPipelineResult) -> str:
        """Generate Markdown report."""
        lines = [
            "# Plasma Discovery Report",
            "",
            f"**Shot:** {result.shot_id}",
            f"**Device:** {result.device}",
            f"**Mode:** {result.confinement_mode}",
            f"**Analysis Time:** {result.execution_time_ms:.1f} ms",
            f"**Timestamp:** {result.timestamp}",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            f"- **Total Findings:** {len(result.findings)}",
            f"- **Critical:** {result.statistics.get('critical_findings', 0)}",
            f"- **Warnings:** {result.statistics.get('warning_findings', 0)}",
            f"- **Hypotheses Generated:** {len(result.hypotheses)}",
            "",
            "## Plasma Parameters",
            "",
            f"| Parameter | Value |",
            f"|-----------|-------|",
            f"| Plasma Current | {result.statistics.get('plasma_current_kA', 0):.0f} kA |",
            f"| Magnetic Field | {result.statistics.get('magnetic_field_T', 0):.2f} T |",
            f"| Stored Energy | {result.statistics.get('stored_energy_MJ', 0):.1f} MJ |",
            f"| Confinement Time | {result.statistics.get('confinement_time_ms', 0):.1f} ms |",
            f"| ELM Frequency | {result.statistics.get('elm_frequency_Hz', 0):.1f} Hz |",
            "",
        ]
        
        # Findings by severity
        if result.findings:
            lines.extend([
                "## Findings",
                "",
            ])
            
            for severity in ["critical", "warning", "info"]:
                sev_findings = [f for f in result.findings if f.get("severity") == severity]
                if sev_findings:
                    emoji = {"critical": "🔴", "warning": "🟡", "info": "🔵"}[severity]
                    lines.append(f"### {emoji} {severity.upper()}")
                    lines.append("")
                    for f in sev_findings:
                        lines.append(f"- **{f.get('type', 'Unknown')}** [{f.get('primitive', '')}]")
                        lines.append(f"  {f.get('description', '')}")
                        lines.append("")
        
        # Hypotheses
        if result.hypotheses:
            lines.extend([
                "## Hypotheses",
                "",
            ])
            for i, h in enumerate(result.hypotheses, 1):
                lines.append(f"### Hypothesis {i}: {h.get('summary', 'Unknown')}")
                lines.append("")
                lines.append(f"**Type:** {h.get('type', '')} | **Confidence:** {h.get('confidence', 0):.0%}")
                lines.append("")
                lines.append(h.get('details', ''))
                lines.append("")
                lines.append(f"**Recommended Action:** {h.get('action', 'None')}")
                lines.append("")
        
        # Footer
        lines.extend([
            "---",
            "",
            "*Generated by HyperTensor Autonomous Discovery Engine - Phase 2: Fusion Plasma*",
        ])
        
        return "\n".join(lines)
    
    def _generate_json_report(self, result: PlasmaPipelineResult) -> str:
        """Generate JSON report."""
        return json.dumps({
            "shot_id": result.shot_id,
            "device": result.device,
            "confinement_mode": result.confinement_mode,
            "timestamp": result.timestamp,
            "execution_time_ms": result.execution_time_ms,
            "statistics": result.statistics,
            "findings": result.findings,
            "hypotheses": result.hypotheses,
        }, indent=2)


def run_demo() -> PlasmaPipelineResult:
    """
    Run demo analysis on synthetic plasma data.
    
    ⚠️  DEMONSTRATION ONLY - NOT FOR PRODUCTION USE
    
    This function:
    - Uses SYNTHETIC fusion plasma data (not real experimental data)
    - Creates fake ELM events and plasma parameters
    - Intended for testing pipeline functionality and visualization
    
    For production analysis, use:
        pipeline = PlasmaDiscoveryPipeline()
        result = pipeline.analyze_shot(real_plasma_shot)
    
    Returns:
        PlasmaPipelineResult with findings from synthetic data analysis
    """
    import logging
    logging.getLogger(__name__).warning(
        "run_demo() uses SYNTHETIC plasma data - not for production use"
    )
    print("=" * 60)
    print("PLASMA DISCOVERY PIPELINE - DEMO")
    print("=" * 60)
    print()
    
    # Create synthetic ITER-like shot
    shot = PlasmaShot(
        shot_id="DEMO-ITER-001",
        device="ITER-SYNTHETIC",
        plasma_current_kA=15000,
        magnetic_field_T=5.3,
        electron_density_m3=1.0e20,
        electron_temp_keV=12.0,
        ion_temp_keV=10.0,
        stored_energy_MJ=350,
        confinement_mode="H-mode",
        elm_events=[
            {"time_ms": 100, "energy_kJ": 80},
            {"time_ms": 180, "energy_kJ": 75},
            {"time_ms": 270, "energy_kJ": 120},  # Large ELM
            {"time_ms": 350, "energy_kJ": 85},
            {"time_ms": 430, "energy_kJ": 90},
            {"time_ms": 520, "energy_kJ": 70},
        ],
        disruption=False,
    )
    
    # Create synthetic profiles
    rho = torch.linspace(0, 1, 100)
    
    # Electron temperature: peaked + pedestal
    Te = 12 * (1 - rho**2)**1.5
    Te[-20:] = Te[-20:] * torch.linspace(1, 0.2, 20)  # Pedestal drop
    
    # Ion temperature
    Ti = 10 * (1 - rho**2)**1.2
    Ti[-20:] = Ti[-20:] * torch.linspace(1, 0.3, 20)
    
    # Pressure
    ne = 1e20 * (1 - rho**2)**0.8
    pressure = ne * (Te + Ti) * 1.6e-19 / 1e3  # Rough
    
    profiles = {
        "Te": PlasmaProfile(rho=rho, values=Te, profile_type="Te", units="keV"),
        "Ti": PlasmaProfile(rho=rho, values=Ti, profile_type="Ti", units="keV"),
        "pressure": PlasmaProfile(rho=rho, values=pressure, profile_type="p", units="kPa"),
    }
    
    # Create synthetic time traces
    time_traces = {
        "power": [50 + 5 * np.sin(i * 0.05) + np.random.randn() for i in range(500)],
        "stored_energy": [350 + 10 * np.sin(i * 0.02) + 0.5 * np.random.randn() for i in range(500)],
        "magnetic_fluct": [0.01 * np.sin(i * 0.3) + 0.005 * np.random.randn() for i in range(500)],
    }
    
    # Create baseline shots for comparison
    baseline_shots = [
        {
            "parameters": torch.tensor([14000, 5.2, 9.5, 11.5, 9.5, 340.0]),
        },
        {
            "parameters": torch.tensor([15500, 5.3, 10.5, 12.5, 10.0, 360.0]),
        },
        {
            "parameters": torch.tensor([14500, 5.25, 10.0, 11.8, 9.8, 345.0]),
        },
    ]
    
    # Run pipeline
    pipeline = PlasmaDiscoveryPipeline(verbose=True)
    result = pipeline.analyze_shot(
        shot,
        profiles=profiles,
        time_traces=time_traces,
        baseline_shots=baseline_shots,
    )
    
    # Print summary
    print()
    print("=" * 60)
    print("DEMO RESULTS")
    print("=" * 60)
    print(f"Shot: {result.shot_id}")
    print(f"Findings: {len(result.findings)}")
    print(f"  Critical: {result.statistics.get('critical_findings', 0)}")
    print(f"  Warning: {result.statistics.get('warning_findings', 0)}")
    print(f"Hypotheses: {len(result.hypotheses)}")
    print()
    
    # Show top findings
    print("Top Findings:")
    for f in result.findings[:5]:
        sev = f.get("severity", "info")
        emoji = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(sev, "⚪")
        print(f"  {emoji} [{f.get('primitive', '')}] {f.get('type', '')}")
        print(f"     {f.get('description', '')}")
    
    print()
    print("Top Hypotheses:")
    for h in result.hypotheses[:3]:
        print(f"  → {h.get('summary', '')} ({h.get('confidence', 0):.0%})")
    
    return result


if __name__ == "__main__":
    result = run_demo()
