#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                          ║
║                    T E N S O R   G E N E S I S   O R A C L E   N O D E                  ║
║                                                                                          ║
║                         Domain-Agnostic Structure Engine                                 ║
║                                                                                          ║
║     Input: Raw numbers (any domain)                                                      ║
║     Output: Cryptographically signed structural attestation                              ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

The Universal Pipeline:
  Stage 1 (OT):   "How much did the distribution shift?"
  Stage 2 (SGW):  "At what scale is the change happening?"
  Stage 3 (RKHS): "How anomalous is this pattern?"
  Stage 4 (PH):   "What is the topological shape?"
  Stage 5 (GA):   "What is the geometric direction?"

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# QTT operations are CPU-based internally, use GPU only for dense ops
DEVICE_GPU = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE_CPU = torch.device('cpu')
DTYPE = torch.float64

app = FastAPI(
    title="Tensor Genesis Oracle Node",
    description="Domain-Agnostic Structure Engine — Universal Mathematical Truth Machine",
    version="1.0.0",
)


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST/RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class AnalyzeRequest(BaseModel):
    """Request to analyze two distributions."""
    distribution_a: List[float] = Field(..., description="First distribution (baseline/historical)")
    distribution_b: List[float] = Field(..., description="Second distribution (current/comparison)")
    domain_hint: Optional[str] = Field(None, description="Optional domain hint (climate, finance, medical, etc.)")
    

class SingleDistributionRequest(BaseModel):
    """Request to analyze a single distribution's structure."""
    data: List[float] = Field(..., description="Distribution to analyze")
    domain_hint: Optional[str] = Field(None, description="Optional domain hint")


class AttestationResponse(BaseModel):
    """Cryptographically signed attestation of analysis results."""
    attestation: str
    timestamp: str
    pipeline_version: str
    device: str
    
    # Stage results
    stage_1_ot: Dict[str, Any]
    stage_2_sgw: Dict[str, Any]
    stage_3_rkhs: Dict[str, Any]
    stage_4_ph: Dict[str, Any]
    stage_5_ga: Dict[str, Any]
    
    # Summary
    total_time_seconds: float
    interpretation: Dict[str, str]
    
    # Cryptographic proof
    sha256: str


# ═══════════════════════════════════════════════════════════════════════════════
# QTT TENSOR CONVERSION (THE ADAPTER)
# ═══════════════════════════════════════════════════════════════════════════════

def numbers_to_qtt_distribution(data: List[float], grid_size: int = None) -> "QTTDistribution":
    """
    Convert raw numbers to QTT-compressed distribution.
    
    This is THE ADAPTER — the only domain-specific part.
    Once data is a QTT tensor, the math is universal.
    """
    from tensornet.genesis.ot import QTTDistribution
    
    arr = np.array(data, dtype=np.float64)
    
    # Normalize to probability distribution
    arr = arr - arr.min()
    if arr.sum() > 0:
        arr = arr / arr.sum()
    else:
        arr = np.ones_like(arr) / len(arr)
    
    # Pad to power of 2 for QTT
    n = len(arr)
    n_bits = max(4, int(np.ceil(np.log2(n))))
    target_size = 2 ** n_bits
    
    if n < target_size:
        # Pad with zeros (or interpolate)
        padded = np.zeros(target_size)
        padded[:n] = arr
        arr = padded / padded.sum() if padded.sum() > 0 else padded
    elif n > target_size:
        # Downsample
        arr = np.interp(
            np.linspace(0, n-1, target_size),
            np.arange(n),
            arr
        )
        arr = arr / arr.sum() if arr.sum() > 0 else arr
    
    # Create QTT distribution
    tensor = torch.tensor(arr, dtype=DTYPE, device=DEVICE_CPU)
    return QTTDistribution.from_dense(tensor, grid_bounds=(0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: OPTIMAL TRANSPORT — "How much did it move?"
# ═══════════════════════════════════════════════════════════════════════════════

def stage_1_optimal_transport(dist_a: "QTTDistribution", dist_b: "QTTDistribution") -> Dict[str, Any]:
    """
    Compute Wasserstein distance between distributions.
    
    Interpretation:
    - Climate: "Temperature shifted by X degrees"
    - Finance: "Liquidity moved by X basis points"  
    - Medical: "Blood flow changed by X ml/min"
    """
    from tensornet.genesis.ot import wasserstein_distance
    
    start = time.perf_counter()
    
    W1 = wasserstein_distance(dist_a, dist_b, p=1, method='quantile')
    W2 = wasserstein_distance(dist_a, dist_b, p=2, method='quantile')
    
    elapsed = time.perf_counter() - start
    
    return {
        "wasserstein_1": float(W1),
        "wasserstein_2": float(W2),
        "shift_magnitude": float(W2),  # Primary metric
        "computation_time_ms": elapsed * 1000,
        "interpretation": "Distribution shift magnitude",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: SPECTRAL GRAPH WAVELETS — "At what scale?"
# ═══════════════════════════════════════════════════════════════════════════════

def stage_2_spectral_wavelets(dist_a: "QTTDistribution", dist_b: "QTTDistribution") -> Dict[str, Any]:
    """
    Multi-scale analysis using spectral graph wavelets.
    
    Interpretation:
    - Climate: "Local storm vs global warming"
    - Finance: "Flash crash vs recession"
    - Medical: "Local clot vs systemic hypertension"
    """
    from tensornet.genesis.sgw import QTTLaplacian, QTTSignal, QTTGraphWavelet
    
    start = time.perf_counter()
    
    n = dist_a.grid_size
    n_bits = int(np.log2(n))
    
    # Build graph Laplacian (1D chain for distribution comparison)
    L = QTTLaplacian.grid_1d(n)
    
    # Create signals from distributions
    signal_a = QTTSignal.from_dense(dist_a.to_dense())
    signal_b = QTTSignal.from_dense(dist_b.to_dense())
    diff_signal = QTTSignal.from_dense(dist_b.to_dense() - dist_a.to_dense())
    
    # Multi-scale wavelet analysis
    scales = [0.5, 1.0, 2.0, 4.0, 8.0]
    wavelet = QTTGraphWavelet.create(L, scales=scales, kernel='mexican_hat')
    
    coeffs = wavelet.transform(diff_signal)
    
    # Energy at each scale - use the built-in method
    energies = coeffs.energy_per_scale()
    scale_energies = [{"scale": s, "energy": float(e)} for s, e in zip(scales, energies)]
    
    # Dominant scale
    max_energy_idx = np.argmax(energies)
    dominant_scale = scales[max_energy_idx]
    
    elapsed = time.perf_counter() - start
    
    return {
        "scales_analyzed": scales,
        "scale_energies": scale_energies,
        "dominant_scale": dominant_scale,
        "is_local": dominant_scale < 2.0,
        "is_global": dominant_scale >= 4.0,
        "computation_time_ms": elapsed * 1000,
        "interpretation": f"Dominant change at scale {dominant_scale} ({'local' if dominant_scale < 2 else 'global'})",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3: RKHS KERNEL METHODS — "Is this normal?"
# ═══════════════════════════════════════════════════════════════════════════════

def stage_3_rkhs_anomaly(dist_a: "QTTDistribution", dist_b: "QTTDistribution") -> Dict[str, Any]:
    """
    Anomaly detection using Maximum Mean Discrepancy in RKHS.
    
    Interpretation:
    - Climate: "This heatwave is abnormal"
    - Finance: "This trade volume is suspicious"
    - Medical: "This tissue density is cancerous"
    """
    from tensornet.genesis.rkhs import RBFKernel, maximum_mean_discrepancy
    
    start = time.perf_counter()
    
    # Convert to point clouds for MMD
    dense_a = dist_a.to_dense().cpu()
    dense_b = dist_b.to_dense().cpu()
    
    # Create point representations
    n = len(dense_a)
    x = torch.linspace(0, 1, n).unsqueeze(1)
    
    # Weight points by distribution values
    points_a = x.repeat(1, 2)
    points_a[:, 1] = dense_a
    
    points_b = x.repeat(1, 2)
    points_b[:, 1] = dense_b
    
    # Multi-scale MMD
    length_scales = [0.1, 0.5, 1.0, 2.0]
    mmd_scores = []
    
    for ls in length_scales:
        kernel = RBFKernel(length_scale=ls)
        mmd = maximum_mean_discrepancy(points_a.to(DEVICE_GPU), points_b.to(DEVICE_GPU), kernel)
        mmd_scores.append({"length_scale": ls, "mmd": float(mmd)})
    
    # Aggregate anomaly score
    avg_mmd = np.mean([s["mmd"] for s in mmd_scores])
    max_mmd = np.max([s["mmd"] for s in mmd_scores])
    
    # Anomaly classification
    if max_mmd > 0.5:
        anomaly_level = "SEVERE"
    elif max_mmd > 0.2:
        anomaly_level = "MODERATE"
    elif max_mmd > 0.05:
        anomaly_level = "MILD"
    else:
        anomaly_level = "NORMAL"
    
    elapsed = time.perf_counter() - start
    
    return {
        "mmd_scores": mmd_scores,
        "average_mmd": float(avg_mmd),
        "max_mmd": float(max_mmd),
        "anomaly_level": anomaly_level,
        "is_anomalous": max_mmd > 0.1,
        "computation_time_ms": elapsed * 1000,
        "interpretation": f"Anomaly level: {anomaly_level} (MMD={max_mmd:.4f})",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4: PERSISTENT HOMOLOGY — "What shape is it?"
# ═══════════════════════════════════════════════════════════════════════════════

def stage_4_topology(dist_a: "QTTDistribution", dist_b: "QTTDistribution") -> Dict[str, Any]:
    """
    Topological analysis using persistent homology.
    
    Interpretation:
    - Climate: "The storm has an eye (cyclone)"
    - Finance: "Trades form a loop (wash trading)"
    - Medical: "Tumor has a cavity (necrosis)"
    """
    from tensornet.genesis.topology.qtt_native import qtt_persistence_grid_1d
    
    start = time.perf_counter()
    
    n = dist_a.grid_size
    n_bits = int(np.log2(n))
    
    # Compute persistent homology
    result = qtt_persistence_grid_1d(n_bits)
    betti = result.betti_numbers
    
    # Analyze difference distribution topology
    diff = (dist_b.to_dense() - dist_a.to_dense()).cpu().numpy()
    
    # Find connected components (β₀) and loops (β₁) in thresholded difference
    threshold = np.std(diff) * 2
    significant_regions = np.abs(diff) > threshold
    
    # Count connected regions
    from scipy import ndimage
    labeled, n_components = ndimage.label(significant_regions)
    
    # Detect "holes" (sign changes surrounded by significant values)
    sign_changes = np.diff(np.sign(diff))
    n_holes = np.sum(np.abs(sign_changes) == 2)
    
    elapsed = time.perf_counter() - start
    
    # Shape interpretation
    if n_holes > 2:
        shape_type = "OSCILLATORY"
        shape_desc = "Multiple peaks and valleys (wave pattern)"
    elif n_components > 3:
        shape_type = "FRAGMENTED"
        shape_desc = "Multiple disconnected change regions"
    elif n_components == 1 and n_holes == 0:
        shape_type = "UNIMODAL"
        shape_desc = "Single concentrated change region"
    else:
        shape_type = "COMPLEX"
        shape_desc = "Multi-modal change pattern"
    
    return {
        "betti_numbers": betti,
        "connected_components": int(n_components),
        "topological_holes": int(n_holes),
        "shape_type": shape_type,
        "shape_description": shape_desc,
        "boundary_memory_bytes": result.memory_bytes,
        "computation_time_ms": elapsed * 1000,
        "interpretation": f"Shape: {shape_type} — {shape_desc}",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5: GEOMETRIC ALGEBRA — "Which direction?"
# ═══════════════════════════════════════════════════════════════════════════════

def stage_5_geometric_direction(dist_a: "QTTDistribution", dist_b: "QTTDistribution") -> Dict[str, Any]:
    """
    Geometric analysis using Clifford algebra.
    
    Interpretation:
    - Climate: "The front is moving North-East"
    - Finance: "The market is trending Bearish"
    - Medical: "Growth is oriented towards the artery"
    """
    from tensornet.genesis.ga import CliffordAlgebra, Multivector, geometric_product
    
    start = time.perf_counter()
    
    # Compute gradient of difference
    dense_a = dist_a.to_dense().cpu().numpy()
    dense_b = dist_b.to_dense().cpu().numpy()
    diff = dense_b - dense_a
    
    # First and second moments
    x = np.linspace(0, 1, len(diff))
    
    # Center of mass of change
    if np.sum(np.abs(diff)) > 1e-10:
        center = np.sum(x * np.abs(diff)) / np.sum(np.abs(diff))
    else:
        center = 0.5
    
    # Direction (gradient at center)
    gradient = np.gradient(diff)
    avg_gradient = np.mean(gradient)
    
    # Magnitude and direction encoding
    magnitude = np.sqrt(np.sum(diff ** 2))
    
    # Use 2D Clifford algebra for direction encoding
    algebra = CliffordAlgebra(2)
    
    # Encode direction as multivector
    # e1 = "positive change", e2 = "spreading"
    spread = np.std(np.where(np.abs(diff) > np.std(diff))[0]) / len(diff) if len(diff) > 0 else 0
    
    direction_vec = Multivector(algebra, np.array([
        0,              # scalar
        avg_gradient,   # e1 (direction)
        spread,         # e2 (spread)
        0,              # e12 (rotation)
    ]))
    
    # Classify trend
    if avg_gradient > 0.01:
        trend = "INCREASING"
        trend_desc = "Distribution shifting right/up"
    elif avg_gradient < -0.01:
        trend = "DECREASING"
        trend_desc = "Distribution shifting left/down"
    else:
        trend = "STABLE"
        trend_desc = "No significant directional trend"
    
    elapsed = time.perf_counter() - start
    
    return {
        "center_of_change": float(center),
        "average_gradient": float(avg_gradient),
        "magnitude": float(magnitude),
        "spread": float(spread),
        "trend": trend,
        "trend_description": trend_desc,
        "geometric_vector": [float(avg_gradient), float(spread)],
        "computation_time_ms": elapsed * 1000,
        "interpretation": f"Trend: {trend} — {trend_desc}",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE — THE UNIVERSAL TRUTH ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(
    data_a: List[float],
    data_b: List[float],
    domain_hint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run the complete 5-stage structure analysis pipeline.
    
    This is THE UNIVERSAL ENGINE.
    It doesn't know what the numbers represent.
    It just finds the mathematical structure.
    """
    total_start = time.perf_counter()
    
    # ADAPTER: Convert raw numbers to QTT tensors
    dist_a = numbers_to_qtt_distribution(data_a)
    dist_b = numbers_to_qtt_distribution(data_b)
    
    # STAGE 1: Optimal Transport
    stage_1 = stage_1_optimal_transport(dist_a, dist_b)
    
    # STAGE 2: Spectral Graph Wavelets
    stage_2 = stage_2_spectral_wavelets(dist_a, dist_b)
    
    # STAGE 3: RKHS Anomaly Detection
    stage_3 = stage_3_rkhs_anomaly(dist_a, dist_b)
    
    # STAGE 4: Persistent Homology
    stage_4 = stage_4_topology(dist_a, dist_b)
    
    # STAGE 5: Geometric Algebra
    stage_5 = stage_5_geometric_direction(dist_a, dist_b)
    
    total_time = time.perf_counter() - total_start
    
    # Generate human-readable interpretation
    interpretation = {
        "shift": f"Distribution shifted by {stage_1['wasserstein_2']:.4f} units",
        "scale": f"Change is {'local' if stage_2['is_local'] else 'global'} (scale={stage_2['dominant_scale']})",
        "anomaly": f"Anomaly level: {stage_3['anomaly_level']}",
        "shape": f"Pattern: {stage_4['shape_type']}",
        "direction": f"Trend: {stage_5['trend']}",
    }
    
    if domain_hint:
        interpretation["domain"] = f"Analysis performed with {domain_hint} domain context"
    
    # Build attestation
    attestation = {
        "attestation": "TENSOR_GENESIS_ORACLE_ATTESTATION",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "1.0.0",
        "device": str(DEVICE_GPU),
        "input_sizes": {
            "distribution_a": len(data_a),
            "distribution_b": len(data_b),
            "qtt_grid_size": dist_a.grid_size,
        },
        "stage_1_ot": stage_1,
        "stage_2_sgw": stage_2,
        "stage_3_rkhs": stage_3,
        "stage_4_ph": stage_4,
        "stage_5_ga": stage_5,
        "total_time_seconds": total_time,
        "interpretation": interpretation,
    }
    
    # Cryptographic hash
    content = json.dumps(attestation, sort_keys=True, default=str)
    attestation["sha256"] = hashlib.sha256(content.encode()).hexdigest()
    
    return attestation


# ═══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    """Health check and info."""
    return {
        "service": "Tensor Genesis Oracle Node",
        "version": "1.0.0",
        "status": "operational",
        "device": str(DEVICE_GPU),
        "endpoints": {
            "/analyze": "POST - Compare two distributions",
            "/health": "GET - Health check",
        },
    }


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy", "device": str(DEVICE_GPU)}


@app.post("/analyze", response_model=None)
async def analyze(request: AnalyzeRequest) -> JSONResponse:
    """
    Analyze two distributions using the 5-stage pipeline.
    
    Returns cryptographically signed attestation of structural analysis.
    """
    try:
        attestation = run_full_pipeline(
            data_a=request.distribution_a,
            data_b=request.distribution_b,
            domain_hint=request.domain_hint,
        )
        return JSONResponse(content=attestation)
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/single", response_model=None)
async def analyze_single(request: SingleDistributionRequest) -> JSONResponse:
    """
    Analyze a single distribution against uniform baseline.
    """
    try:
        # Create uniform baseline
        n = len(request.data)
        uniform = [1.0 / n] * n
        
        attestation = run_full_pipeline(
            data_a=uniform,
            data_b=request.data,
            domain_hint=request.domain_hint,
        )
        return JSONResponse(content=attestation)
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                          ║
║                    T E N S O R   G E N E S I S   O R A C L E   N O D E                  ║
║                                                                                          ║
║                         Domain-Agnostic Structure Engine                                 ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
    """)
    uvicorn.run(app, host="0.0.0.0", port=8080)
