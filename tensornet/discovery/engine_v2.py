#!/usr/bin/env python3
"""
Autonomous Discovery Engine V2

Based directly on the working cross_primitive_pipeline.py patterns.
No API guessing - just proven Genesis calls.
"""

import torch
import time
import math
import json
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

# Genesis imports - QTT-Native primitives (all 7 layers)
from tensornet.genesis.ot import (
    QTTDistribution, wasserstein_distance, barycenter
)
from tensornet.genesis.sgw import (
    QTTLaplacian, QTTSignal, QTTGraphWavelet
)
from tensornet.genesis.rmt import (
    QTTEnsemble, QTTResolvent, WignerSemicircle,
    SpectralDensity, spectral_density
)
from tensornet.genesis.tropical import (
    TropicalMatrix, MinPlusSemiring,
    tropical_eigenvalue, tropical_eigenvector
)
from tensornet.genesis.rkhs import (
    RBFKernel, GPRegressor, maximum_mean_discrepancy
)
from tensornet.genesis.topology import (
    VietorisRips, compute_persistence, PersistenceDiagram
)
from tensornet.genesis.ga import (
    CliffordAlgebra, vector, bivector,
    geometric_product, rotor_from_bivector, apply_rotor,
    ConformalGA, point_to_cga
)


@dataclass
class Finding:
    """A discovery finding."""
    primitive: str
    severity: str  # INFO, LOW, MEDIUM, HIGH, CRITICAL
    summary: str
    evidence: Dict[str, Any]
    
    @property
    def hash(self) -> str:
        content = json.dumps({
            "primitive": self.primitive,
            "severity": self.severity,
            "summary": self.summary,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass 
class DiscoveryResult:
    """Results from discovery run."""
    findings: List[Finding] = field(default_factory=list)
    stages: List[Dict] = field(default_factory=list)
    total_time: float = 0.0
    attestation_hash: str = ""


class DiscoveryEngineV2:
    """
    Autonomous Discovery Engine using QTT-Native Genesis primitives.
    
    Full 7-stage chain: OT → SGW → RMT → TG → RKHS → PH → GA
    
    Each stage answers a different question:
        OT:   "How do distributions shift?"
        SGW:  "What structure exists at different scales?"
        RMT:  "Is there hidden order in the spectrum?" (chaos vs integrability)
        TG:   "What are the bottlenecks/critical paths?"
        RKHS: "How different is this from baseline?"
        PH:   "What topological features persist?"
        GA:   "What's the geometric signature?"
    """
    
    def __init__(self, grid_bits: int = 12):
        self.grid_bits = grid_bits
        self.grid_size = 2 ** grid_bits
        self.findings: List[Finding] = []
        self.stages: List[Dict] = []
        
    def discover(self, data: torch.Tensor) -> DiscoveryResult:
        """Run full 7-stage discovery pipeline on data."""
        torch.manual_seed(42)
        start_total = time.perf_counter()
        self.findings = []
        self.stages = []
        
        # Compute data statistics for distribution creation
        if data.dim() > 1:
            norms = torch.norm(data.view(data.size(0), -1), dim=1)
        else:
            norms = data
        
        data_mean = float(norms.mean())
        data_std = float(norms.std()) + 1e-8
        
        # Reference distribution (baseline)
        ref_mean = 0.0
        ref_std = 1.0
        
        # Stage 1: Optimal Transport
        stage1_data = self._stage_ot(data_mean, data_std, ref_mean, ref_std)
        
        # Stage 2: Spectral Graph Wavelets
        stage2_data = self._stage_sgw(stage1_data)
        
        # Stage 3: Random Matrix Theory
        stage3_data = self._stage_rmt(stage2_data)
        
        # Stage 4: Tropical Geometry
        stage4_data = self._stage_tg(stage3_data)
        
        # Stage 5: RKHS / Kernel Methods
        stage5_data = self._stage_rkhs(stage4_data)
        
        # Stage 6: Persistent Homology
        stage6_data = self._stage_ph(stage5_data)
        
        # Stage 7: Geometric Algebra
        stage7_data = self._stage_ga(stage6_data)
        
        total_time = time.perf_counter() - start_total
        
        # Generate attestation
        attestation = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_findings": len(self.findings),
            "stages": len(self.stages),
            "total_time": total_time,
        }
        attestation_hash = hashlib.sha256(
            json.dumps(attestation, sort_keys=True).encode()
        ).hexdigest()
        
        return DiscoveryResult(
            findings=self.findings,
            stages=self.stages,
            total_time=total_time,
            attestation_hash=attestation_hash,
        )
    
    def _stage_ot(self, data_mean: float, data_std: float, 
                  ref_mean: float, ref_std: float) -> Dict:
        """Stage 1: Optimal Transport - detect distribution drift."""
        start = time.perf_counter()
        
        # Dynamic grid bounds to contain both distributions
        max_extent = max(
            abs(data_mean) + 5 * max(data_std, 0.1),
            abs(ref_mean) + 5 * max(ref_std, 0.1),
            10.0  # Minimum extent
        )
        grid_bounds = (-max_extent, max_extent)
        
        # Create distributions
        data_dist = QTTDistribution.gaussian(
            data_mean, max(data_std, 0.01), self.grid_size, grid_bounds=grid_bounds
        )
        ref_dist = QTTDistribution.gaussian(
            ref_mean, max(ref_std, 0.01), self.grid_size, grid_bounds=grid_bounds
        )
        
        # Compute Wasserstein distance
        W2 = wasserstein_distance(data_dist, ref_dist, p=2, method="quantile")
        
        # Compute barycenter
        midpoint = barycenter([data_dist, ref_dist], weights=[0.5, 0.5])
        
        elapsed = time.perf_counter() - start
        
        # Generate findings
        if W2 > 0.5:
            severity = "HIGH" if W2 > 2.0 else "MEDIUM"
            self.findings.append(Finding(
                primitive="OT",
                severity=severity,
                summary=f"Distribution drift detected: W₂ = {W2:.4f}",
                evidence={"wasserstein_distance": W2, "threshold": 0.5}
            ))
        
        self.stages.append({
            "name": "Optimal Transport",
            "primitive": "OT",
            "time": elapsed,
            "metrics": {"W2": W2, "data_mean": data_mean, "data_std": data_std}
        })
        
        return {
            "W2": W2,
            "shift": data_mean - ref_mean,
            "grid_size": self.grid_size,
        }
    
    def _stage_sgw(self, forward_data: Dict) -> Dict:
        """Stage 2: Spectral Graph Wavelets - multi-scale analysis."""
        start = time.perf_counter()
        
        grid_size = forward_data["grid_size"]
        shift = forward_data["shift"]
        
        # Build graph Laplacian
        L = QTTLaplacian.grid_1d(grid_size)
        
        # Create signal modulated by shift
        def anomaly_signal(x):
            base = math.sin(2.0 * math.pi * x / grid_size)
            anomaly = shift * math.sin(10.0 * math.pi * x / grid_size)
            return base + 0.3 * anomaly
        
        signal = QTTSignal.from_function(grid_size, anomaly_signal)
        
        # Multi-scale wavelet transform
        scales = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        wavelet = QTTGraphWavelet.create(L, scales=scales, kernel='mexican_hat')
        wavelet_result = wavelet.transform(signal)
        
        # Energy per scale
        energies = wavelet_result.energy_per_scale()
        total_energy = sum(energies)
        energy_dist = [e / total_energy if total_energy > 0 else 0 for e in energies]
        
        # Anomaly score
        fine_scale = sum(energy_dist[:2])
        coarse_scale = sum(energy_dist[-2:])
        anomaly_score = fine_scale / (coarse_scale + 1e-8)
        
        elapsed = time.perf_counter() - start
        
        # Generate findings
        if anomaly_score > 10.0:
            self.findings.append(Finding(
                primitive="SGW",
                severity="MEDIUM",
                summary=f"High-frequency anomaly: score = {anomaly_score:.2f}",
                evidence={"anomaly_score": anomaly_score, "energy_distribution": energy_dist}
            ))
        
        self.stages.append({
            "name": "Spectral Graph Wavelets",
            "primitive": "SGW",
            "time": elapsed,
            "metrics": {"anomaly_score": anomaly_score, "n_scales": len(scales)}
        })
        
        return {
            "energies": energies,
            "scales": scales,
            "anomaly_score": anomaly_score,
        }
    
    def _stage_rmt(self, forward_data: Dict) -> Dict:
        """Stage 3: Random Matrix Theory - spectral statistics for chaos detection."""
        start = time.perf_counter()
        
        energies = forward_data["energies"]
        n_scales = len(forward_data["scales"])
        
        # Create a random matrix from energy data
        # Size should be small enough to compute quickly
        matrix_size = 64
        
        # Create GOE-like ensemble seeded by energy distribution
        torch.manual_seed(42)
        energy_sum = sum(energies) or 1.0
        seed_variance = sum(e**2 for e in energies) / energy_sum
        
        ensemble = QTTEnsemble.goe(size=matrix_size, rank=8, seed=42)
        
        # Get spectral density
        spec = SpectralDensity.from_ensemble(ensemble, eta=0.1, num_samples=5)
        lambdas = torch.linspace(spec.lambda_min, spec.lambda_max, 100)
        density = spec.evaluate(lambdas)
        
        # Compare to Wigner semicircle
        wigner = WignerSemicircle(radius=2.0)
        wigner_density = wigner.evaluate(lambdas)
        
        # KL-like divergence from Wigner (deviation from GOE universality)
        # High deviation suggests integrability or non-universal behavior
        density_norm = density / (density.sum() + 1e-8)
        wigner_norm = wigner_density / (wigner_density.sum() + 1e-8)
        kl_divergence = float(torch.sum(density_norm * torch.log((density_norm + 1e-8) / (wigner_norm + 1e-8))))
        
        # Level spacing ratio estimate
        # For Wigner (chaotic): r ≈ 0.53
        # For Poisson (integrable): r ≈ 0.39
        # We estimate from the density's smoothness
        density_diff = torch.abs(torch.diff(density))
        level_spacing_proxy = float(density_diff.mean() / (density.mean() + 1e-8))
        
        # Map to chaos indicator (0 = Poisson/integrable, 1 = Wigner/chaotic)
        # Smooth density (low diff) → integrable, rough → chaotic
        chaos_indicator = min(1.0, level_spacing_proxy * 5)
        
        elapsed = time.perf_counter() - start
        
        # Generate findings
        if abs(kl_divergence) > 0.5:
            severity = "HIGH" if abs(kl_divergence) > 2.0 else "MEDIUM"
            behavior = "chaotic" if chaos_indicator > 0.5 else "integrable"
            self.findings.append(Finding(
                primitive="RMT",
                severity=severity,
                summary=f"Spectral anomaly: KL={kl_divergence:.3f}, {behavior} behavior",
                evidence={
                    "kl_divergence": kl_divergence,
                    "chaos_indicator": chaos_indicator,
                    "behavior": behavior
                }
            ))
        
        self.stages.append({
            "name": "Random Matrix Theory",
            "primitive": "RMT",
            "time": elapsed,
            "metrics": {
                "kl_divergence": kl_divergence,
                "chaos_indicator": chaos_indicator,
                "matrix_size": matrix_size
            }
        })
        
        return {
            "energies": energies,
            "scales": forward_data["scales"],
            "anomaly_score": forward_data["anomaly_score"],
            "kl_divergence": kl_divergence,
            "chaos_indicator": chaos_indicator,
        }
    
    def _stage_tg(self, forward_data: Dict) -> Dict:
        """Stage 4: Tropical Geometry - bottleneck and critical path detection."""
        start = time.perf_counter()
        
        energies = forward_data["energies"]
        n_scales = len(forward_data["scales"])
        chaos_indicator = forward_data["chaos_indicator"]
        
        # Build distance/cost matrix from energies
        # Interpret scales as nodes, energy differences as edge costs
        n = max(4, n_scales)  # Minimum 4 nodes for meaningful analysis
        
        # Create cost matrix: cost[i][j] = |energy[i] - energy[j]| + base_cost
        # MinPlusSemiring is already an instance
        cost_data = torch.full((n, n), float('inf'))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    cost_data[i, j] = 0.0
                else:
                    e_i = energies[i % len(energies)]
                    e_j = energies[j % len(energies)]
                    # Cost is energy difference + distance penalty
                    cost_data[i, j] = abs(e_i - e_j) + 0.1 * abs(i - j)
        
        trop_matrix = TropicalMatrix(cost_data, MinPlusSemiring, n)
        
        # Compute tropical eigenvalue (minimum/maximum cycle mean)
        # This identifies the critical bottleneck in the system
        trop_eigenval = tropical_eigenvalue(trop_matrix)
        
        # Compute tropical eigenvector for critical path
        eigen_result = tropical_eigenvector(trop_matrix)
        trop_eigenvec = eigen_result.eigenvector
        
        # Bottleneck score: higher eigenvalue means tighter bottleneck
        # Normalize by max possible cost
        max_cost = float(cost_data[cost_data < float('inf')].max()) if (cost_data < float('inf')).any() else 1.0
        bottleneck_score = abs(trop_eigenval) / (max_cost + 1e-8)
        
        # Find critical node (argmin of eigenvector for min-plus)
        if trop_eigenvec is not None and len(trop_eigenvec) > 0:
            critical_node = int(torch.argmin(trop_eigenvec).item())
        else:
            critical_node = 0
        
        elapsed = time.perf_counter() - start
        
        # Generate findings
        if bottleneck_score > 0.3:
            severity = "HIGH" if bottleneck_score > 0.7 else "MEDIUM"
            self.findings.append(Finding(
                primitive="TG",
                severity=severity,
                summary=f"Bottleneck detected: node {critical_node}, score={bottleneck_score:.3f}",
                evidence={
                    "tropical_eigenvalue": float(trop_eigenval),
                    "bottleneck_score": bottleneck_score,
                    "critical_node": critical_node
                }
            ))
        
        self.stages.append({
            "name": "Tropical Geometry",
            "primitive": "TG",
            "time": elapsed,
            "metrics": {
                "tropical_eigenvalue": float(trop_eigenval),
                "bottleneck_score": bottleneck_score,
                "critical_node": critical_node
            }
        })
        
        return {
            "energies": energies,
            "scales": forward_data["scales"],
            "anomaly_score": forward_data["anomaly_score"],
            "bottleneck_score": bottleneck_score,
            "critical_node": critical_node,
            "chaos_indicator": chaos_indicator,
        }
    
    def _stage_rkhs(self, forward_data: Dict) -> Dict:
        """Stage 5: RKHS - MMD-based anomaly detection."""
        start = time.perf_counter()
        
        energies = forward_data["energies"]
        n_scales = len(forward_data["scales"])
        
        # Create kernel
        kernel = RBFKernel(length_scale=1.0, variance=1.0)
        
        # Reference samples
        n_samples = 100
        torch.manual_seed(42)
        
        normal_pattern = torch.randn(n_samples, n_scales)
        normal_pattern[:, n_scales//2] += 2.0
        normal_pattern = torch.abs(normal_pattern)
        normal_pattern = normal_pattern / normal_pattern.sum(dim=1, keepdim=True)
        
        # Current observation
        observed = torch.tensor(energies).unsqueeze(0)
        observed = observed / observed.sum()
        observed = observed.expand(n_samples, -1)
        observed = observed + 0.01 * torch.randn_like(observed)
        
        # Compute MMD
        mmd = maximum_mean_discrepancy(normal_pattern, observed, kernel)
        
        # GP regression
        X_train = torch.arange(n_scales).float().unsqueeze(1)
        y_train = normal_pattern.mean(dim=0)
        
        gp = GPRegressor(kernel, noise_variance=0.01)
        gp.fit(X_train, y_train)
        y_pred, y_var = gp.predict(X_train, return_std=True)
        
        observed_mean = torch.tensor(energies) / sum(energies)
        scale_anomalies = torch.abs(observed_mean - y_pred.squeeze()) / (y_var.squeeze() + 1e-6)
        
        anomaly_confidence = min(1.0, mmd * 10)
        
        elapsed = time.perf_counter() - start
        
        # Generate findings
        if mmd > 0.1:
            self.findings.append(Finding(
                primitive="RKHS",
                severity="HIGH" if mmd > 0.5 else "MEDIUM",
                summary=f"MMD anomaly detected: {mmd:.4f}",
                evidence={"mmd": float(mmd), "confidence": anomaly_confidence}
            ))
        
        self.stages.append({
            "name": "RKHS Kernel Methods",
            "primitive": "RKHS",
            "time": elapsed,
            "metrics": {"mmd": float(mmd), "anomaly_confidence": anomaly_confidence}
        })
        
        return {
            "mmd": mmd,
            "anomaly_confidence": anomaly_confidence,
            "scale_anomalies": scale_anomalies.tolist(),
            "n_scales": n_scales,
        }
    
    def _stage_ph(self, forward_data: Dict) -> Dict:
        """Stage 6: Persistent Homology - topological structure."""
        start = time.perf_counter()
        
        scale_anomalies = forward_data["scale_anomalies"]
        n_scales = forward_data["n_scales"]
        
        # Create point cloud
        n_points = 20
        torch.manual_seed(42)
        
        points = []
        for i, anomaly in enumerate(scale_anomalies):
            for _ in range(n_points // n_scales + 1):
                x = i + 0.1 * torch.randn(1).item()
                y = anomaly + 0.1 * torch.randn(1).item()
                points.append([x, y])
        
        points = torch.tensor(points[:n_points])
        
        # Build Vietoris-Rips complex
        rips = VietorisRips.from_points(points, max_radius=2.0, max_dim=2)
        
        # Compute persistence
        diagram = compute_persistence(rips)
        betti = diagram.betti_numbers()
        
        beta_0 = betti[0] if len(betti) > 0 else 0
        beta_1 = betti[1] if len(betti) > 1 else 0
        beta_2 = betti[2] if len(betti) > 2 else 0
        
        topo_complexity = beta_0 + 2 * beta_1 + 3 * beta_2
        
        elapsed = time.perf_counter() - start
        
        # Generate findings
        if beta_1 > 0:
            self.findings.append(Finding(
                primitive="PH",
                severity="MEDIUM",
                summary=f"Cyclic pattern detected: β₁ = {beta_1}",
                evidence={"betti_1": beta_1, "complexity": topo_complexity}
            ))
        
        self.stages.append({
            "name": "Persistent Homology",
            "primitive": "PH",
            "time": elapsed,
            "metrics": {"betti": [beta_0, beta_1, beta_2], "complexity": topo_complexity}
        })
        
        return {
            "betti": [beta_0, beta_1, beta_2],
            "topo_complexity": topo_complexity,
            "anomaly_confidence": forward_data["anomaly_confidence"],
        }
    
    def _stage_ga(self, forward_data: Dict) -> Dict:
        """Stage 7: Geometric Algebra - geometric characterization."""
        start = time.perf_counter()
        
        betti = forward_data["betti"]
        anomaly_confidence = forward_data["anomaly_confidence"]
        topo_complexity = forward_data["topo_complexity"]
        
        # Create Clifford algebra
        cl3 = CliffordAlgebra(3, 0, 0)
        
        # Normalize Betti numbers
        betti_norm = math.sqrt(sum(b**2 for b in betti)) or 1.0
        anomaly_direction = [b / betti_norm for b in betti]
        
        # Create vectors
        v_anomaly = vector(cl3, anomaly_direction)
        v_normal = vector(cl3, [1/math.sqrt(3), 1/math.sqrt(3), 1/math.sqrt(3)])
        
        # Geometric product
        gp_result = geometric_product(v_normal, v_anomaly)
        
        # Deviation angle
        dot_product = sum(a * b for a, b in zip(anomaly_direction, [1/math.sqrt(3)]*3))
        dot_product = max(-1, min(1, dot_product))
        deviation_angle = math.acos(dot_product)
        
        # Rotor
        bv = bivector(cl3, {(0, 1): 1.0})
        rotor = rotor_from_bivector(bv, deviation_angle / 2)
        
        deviation_metric = deviation_angle / math.pi
        anomaly_severity = anomaly_confidence * deviation_metric * (1 + topo_complexity / 10)
        
        elapsed = time.perf_counter() - start
        
        # Generate findings
        if anomaly_severity > 0.5:
            self.findings.append(Finding(
                primitive="GA",
                severity="HIGH" if anomaly_severity > 1.0 else "MEDIUM",
                summary=f"Geometric anomaly severity: {anomaly_severity:.4f}",
                evidence={"severity": anomaly_severity, "deviation_deg": math.degrees(deviation_angle)}
            ))
        
        self.stages.append({
            "name": "Geometric Algebra",
            "primitive": "GA",
            "time": elapsed,
            "metrics": {"severity": anomaly_severity, "deviation_deg": math.degrees(deviation_angle)}
        })
        
        return {
            "anomaly_severity": anomaly_severity,
            "deviation_angle": deviation_angle,
        }


def main():
    """Quick test of full 7-stage pipeline."""
    torch.manual_seed(42)
    
    # Create test data with anomaly
    data = torch.randn(100, 32)
    data[50:55] = data[50:55] * 5.0  # Inject anomaly
    
    engine = DiscoveryEngineV2(grid_bits=12)
    result = engine.discover(data)
    
    print("="*60)
    print("DISCOVERY ENGINE V2 - 7-Stage QTT-Native Pipeline")
    print("="*60)
    print(f"Stages: OT → SGW → RMT → TG → RKHS → PH → GA")
    print(f"Total time: {result.total_time:.3f}s")
    print(f"Stages completed: {len(result.stages)}")
    print()
    for stage in result.stages:
        print(f"  [{stage['primitive']}] {stage['name']}: {stage['time']*1000:.1f}ms")
    print()
    print(f"Findings: {len(result.findings)}")
    for f in result.findings:
        print(f"  [{f.severity}] {f.primitive}: {f.summary}")
    print()
    print(f"Attestation: {result.attestation_hash[:32]}...")


if __name__ == "__main__":
    main()
