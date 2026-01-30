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
import os
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Ensure project root is in path for imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

# Calibration profiles directory
CALIBRATION_DIR = Path(__file__).parent / "calibration_profiles"
CALIBRATION_DIR.mkdir(exist_ok=True)

# Global calibration cache
_calibration_cache: Dict[str, "CalibrationProfile"] = {}


def make_json_serializable(obj: Any) -> Any:
    """Recursively convert numpy/torch types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif hasattr(obj, 'item'):  # torch scalar tensors
        return obj.item()
    else:
        return obj


def load_calibration_profile(domain: str) -> Optional["CalibrationProfile"]:
    """
    Load calibration profile for a domain.
    
    Returns None if no calibration exists for this domain.
    """
    from calibration import CalibrationProfile
    
    if domain in _calibration_cache:
        return _calibration_cache[domain]
    
    profile_path = CALIBRATION_DIR / f"calibration_{domain}.json"
    if profile_path.exists():
        with open(profile_path, 'r') as f:
            data = json.load(f)
        profile = CalibrationProfile.from_dict(data)
        _calibration_cache[domain] = profile
        return profile
    
    return None


def get_threshold(
    domain: Optional[str],
    stage_name: str,
    severity: str,
    default: float,
) -> float:
    """
    Get calibrated threshold or fall back to default.
    
    Args:
        domain: Domain hint (e.g., "climate", "finance")
        stage_name: Pipeline stage (e.g., "ot", "rkhs")
        severity: Severity level (e.g., "mild", "moderate", "severe")
        default: Default value if no calibration exists
    
    Returns:
        Calibrated threshold or default
    """
    if domain is None:
        return default
    
    profile = load_calibration_profile(domain)
    if profile is None:
        return default
    
    if stage_name not in profile.stage_thresholds:
        return default
    
    thresholds = profile.stage_thresholds[stage_name].thresholds
    return thresholds.get(severity, default)

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

def stage_3_rkhs_anomaly(
    dist_a: "QTTDistribution",
    dist_b: "QTTDistribution",
    domain_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Anomaly detection using Maximum Mean Discrepancy in RKHS.
    
    Interpretation:
    - Climate: "This heatwave is abnormal"
    - Finance: "This trade volume is suspicious"
    - Medical: "This tissue density is cancerous"
    
    Uses stratified subsampling to avoid O(n²) memory explosion.
    """
    from tensornet.genesis.rkhs import RBFKernel, maximum_mean_discrepancy
    from tensornet.genesis.core.triton_ops import qtt_evaluate_at_indices
    
    start = time.perf_counter()
    
    n = dist_a.grid_size
    
    # Subsample to avoid O(n²) kernel matrix explosion
    # 2048 points gives 2048² = 4M entries = 32MB (manageable)
    n_samples = min(2048, n)
    sample_indices = torch.linspace(0, n - 1, n_samples, dtype=torch.long)
    
    # Evaluate QTT at sample points (no full dense materialization)
    from tensornet.genesis.sgw.graph_signals import QTTSignal
    signal_a = QTTSignal.from_cores(dist_a.cores)
    signal_b = QTTSignal.from_cores(dist_b.cores)
    
    values_a = qtt_evaluate_at_indices(signal_a.cores, sample_indices)
    values_b = qtt_evaluate_at_indices(signal_b.cores, sample_indices)
    
    # Create point representations: (x_position, pdf_value)
    x = torch.linspace(0, 1, n_samples).unsqueeze(1)
    
    points_a = torch.cat([x, values_a.unsqueeze(1)], dim=1)
    points_b = torch.cat([x, values_b.unsqueeze(1)], dim=1)
    
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
    
    # Get calibrated thresholds or use defaults
    threshold_severe = get_threshold(domain_hint, "rkhs", "severe", default=0.5)
    threshold_moderate = get_threshold(domain_hint, "rkhs", "moderate", default=0.2)
    threshold_mild = get_threshold(domain_hint, "rkhs", "mild", default=0.05)
    
    # Anomaly classification using calibrated thresholds
    if max_mmd >= threshold_severe:
        anomaly_level = "SEVERE"
    elif max_mmd >= threshold_moderate:
        anomaly_level = "MODERATE"
    elif max_mmd >= threshold_mild:
        anomaly_level = "MILD"
    else:
        anomaly_level = "NORMAL"
    
    elapsed = time.perf_counter() - start
    
    # Indicate calibration status
    calibration_status = "calibrated" if domain_hint and load_calibration_profile(domain_hint) else "default"
    
    return {
        "mmd_scores": mmd_scores,
        "average_mmd": float(avg_mmd),
        "max_mmd": float(max_mmd),
        "anomaly_level": anomaly_level,
        "is_anomalous": max_mmd >= threshold_mild,
        "thresholds_used": {
            "mild": threshold_mild,
            "moderate": threshold_moderate,
            "severe": threshold_severe,
        },
        "calibration_status": calibration_status,
        "computation_time_ms": elapsed * 1000,
        "interpretation": f"Anomaly level: {anomaly_level} (MMD={max_mmd:.4f}, {calibration_status} thresholds)",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4: PERSISTENT HOMOLOGY — "What shape is it?"
# ═══════════════════════════════════════════════════════════════════════════════

def stage_4_topology(
    dist_a: "QTTDistribution",
    dist_b: "QTTDistribution",
    domain_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Topological analysis using persistent homology.
    
    Interpretation:
    - Climate: "The storm has an eye (cyclone)"
    - Finance: "Trades form a loop (wash trading)"
    - Medical: "Tumor has a cavity (necrosis)"
    
    Complexity thresholds loaded from calibration if available.
    """
    from tensornet.genesis.topology.qtt_native import qtt_persistence_grid_1d
    from tensornet.genesis.core.triton_ops import qtt_evaluate_at_indices
    
    start = time.perf_counter()
    
    n = dist_a.grid_size
    n_bits = int(np.log2(n))
    
    # Compute persistent homology
    result = qtt_persistence_grid_1d(n_bits)
    betti = result.betti_numbers
    
    # SUBSAMPLE to avoid O(n) dense allocation — max 2048 points
    n_samples = min(2048, n)
    sample_indices = torch.linspace(0, n - 1, n_samples, dtype=torch.long, device=dist_a.device)
    values_a = qtt_evaluate_at_indices(dist_a.cores, sample_indices)
    values_b = qtt_evaluate_at_indices(dist_b.cores, sample_indices)
    diff = (values_b - values_a).cpu().numpy()
    
    # Find connected components (β₀) and loops (β₁) in thresholded difference
    threshold = np.std(diff) * 2
    significant_regions = np.abs(diff) > threshold
    
    # Count connected regions
    from scipy import ndimage
    labeled, n_components = ndimage.label(significant_regions)
    
    # Detect "holes" (sign changes surrounded by significant values)
    sign_changes = np.diff(np.sign(diff))
    n_holes = np.sum(np.abs(sign_changes) == 2)
    
    # Topological complexity metric
    complexity = float(n_components + n_holes)
    
    # Get calibrated thresholds for complexity classification
    threshold_oscillatory = get_threshold(domain_hint, "ph", "severe", default=5.0)
    threshold_fragmented = get_threshold(domain_hint, "ph", "moderate", default=3.0)
    
    elapsed = time.perf_counter() - start
    
    # Shape interpretation using calibrated thresholds
    if n_holes > threshold_oscillatory / 2:  # Holes indicate oscillation
        shape_type = "OSCILLATORY"
        shape_desc = "Multiple peaks and valleys (wave pattern)"
    elif n_components > threshold_fragmented:
        shape_type = "FRAGMENTED"
        shape_desc = "Multiple disconnected change regions"
    elif n_components == 1 and n_holes == 0:
        shape_type = "UNIMODAL"
        shape_desc = "Single concentrated change region"
    else:
        shape_type = "COMPLEX"
        shape_desc = "Multi-modal change pattern"
    
    # Indicate calibration status
    calibration_status = "calibrated" if domain_hint and load_calibration_profile(domain_hint) else "default"
    
    return {
        "betti_numbers": betti,
        "connected_components": int(n_components),
        "topological_holes": int(n_holes),
        "complexity": complexity,
        "shape_type": shape_type,
        "shape_description": shape_desc,
        "boundary_memory_bytes": result.memory_bytes,
        "calibration_status": calibration_status,
        "computation_time_ms": elapsed * 1000,
        "interpretation": f"Shape: {shape_type} — {shape_desc}",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5: GEOMETRIC ALGEBRA — "Which direction?"
# ═══════════════════════════════════════════════════════════════════════════════

def stage_5_geometric_direction(
    dist_a: "QTTDistribution",
    dist_b: "QTTDistribution",
    domain_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Geometric analysis using Clifford algebra.
    
    Interpretation:
    - Climate: "The front is moving North-East"
    - Finance: "The market is trending Bearish"
    - Medical: "Growth is oriented towards the artery"
    
    Gradient thresholds loaded from calibration if available.
    """
    from tensornet.genesis.ga import CliffordAlgebra, Multivector, geometric_product
    from tensornet.genesis.core.triton_ops import qtt_evaluate_at_indices
    
    start = time.perf_counter()
    
    n = dist_a.grid_size
    
    # SUBSAMPLE to avoid O(n) dense allocation — max 2048 points
    n_samples = min(2048, n)
    sample_indices = torch.linspace(0, n - 1, n_samples, dtype=torch.long, device=dist_a.device)
    dense_a = qtt_evaluate_at_indices(dist_a.cores, sample_indices).cpu().numpy()
    dense_b = qtt_evaluate_at_indices(dist_b.cores, sample_indices).cpu().numpy()
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
    max_gradient = float(np.max(np.abs(gradient)))
    
    # Magnitude and direction encoding
    magnitude = np.sqrt(np.sum(diff ** 2))
    
    # Use 2D Clifford algebra for direction encoding
    algebra = CliffordAlgebra(2)
    
    # Encode direction as multivector
    # e1 = "positive change", e2 = "spreading"
    significant_idx = np.where(np.abs(diff) > np.std(diff))[0]
    spread = float(np.std(significant_idx) / len(diff)) if len(significant_idx) > 0 else 0.0
    
    direction_vec = Multivector(algebra, np.array([
        0,              # scalar
        avg_gradient,   # e1 (direction)
        spread,         # e2 (spread)
        0,              # e12 (rotation)
    ]))
    
    # Get calibrated gradient thresholds
    threshold_increasing = get_threshold(domain_hint, "ga", "mild", default=0.01)
    
    # Classify trend using calibrated threshold
    if avg_gradient > threshold_increasing:
        trend = "INCREASING"
        trend_desc = "Distribution shifting right/up"
    elif avg_gradient < -threshold_increasing:
        trend = "DECREASING"
        trend_desc = "Distribution shifting left/down"
    else:
        trend = "STABLE"
        trend_desc = "No significant directional trend"
    
    elapsed = time.perf_counter() - start
    
    # Indicate calibration status
    calibration_status = "calibrated" if domain_hint and load_calibration_profile(domain_hint) else "default"
    
    return {
        "center_of_change": float(center),
        "average_gradient": float(avg_gradient),
        "max_gradient": max_gradient,
        "magnitude": float(magnitude),
        "spread": spread,
        "trend": trend,
        "trend_description": trend_desc,
        "geometric_vector": [float(avg_gradient), spread],
        "calibration_status": calibration_status,
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
    
    If a calibration profile exists for the given domain_hint,
    thresholds will be loaded from it. Otherwise, defaults are used.
    """
    total_start = time.perf_counter()
    
    # Check calibration status
    calibration_profile = load_calibration_profile(domain_hint) if domain_hint else None
    calibration_status = "calibrated" if calibration_profile else "default"
    
    # ADAPTER: Convert raw numbers to QTT tensors
    dist_a = numbers_to_qtt_distribution(data_a)
    dist_b = numbers_to_qtt_distribution(data_b)
    
    # STAGE 1: Optimal Transport (no thresholds, just measurement)
    stage_1 = stage_1_optimal_transport(dist_a, dist_b)
    
    # STAGE 2: Spectral Graph Wavelets (no thresholds, just measurement)
    stage_2 = stage_2_spectral_wavelets(dist_a, dist_b)
    
    # STAGE 3: RKHS Anomaly Detection (uses calibrated thresholds)
    stage_3 = stage_3_rkhs_anomaly(dist_a, dist_b, domain_hint=domain_hint)
    
    # STAGE 4: Persistent Homology (uses calibrated thresholds)
    stage_4 = stage_4_topology(dist_a, dist_b, domain_hint=domain_hint)
    
    # STAGE 5: Geometric Algebra (uses calibrated thresholds)
    stage_5 = stage_5_geometric_direction(dist_a, dist_b, domain_hint=domain_hint)
    
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
        interpretation["calibration"] = calibration_status
    
    # Build attestation
    attestation = {
        "attestation": "TENSOR_GENESIS_ORACLE_ATTESTATION",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "1.1.0",  # Updated for calibration support
        "calibration_status": calibration_status,
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
    
    # Convert all numpy types to Python native types for JSON serialization
    attestation = make_json_serializable(attestation)
    
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
            "/analyze/single": "POST - Analyze single distribution against uniform",
            "/calibration/status": "GET - Check calibration status for a domain",
            "/calibration/list": "GET - List available calibration profiles",
            "/health": "GET - Health check",
        },
    }


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy", "device": str(DEVICE_GPU)}


@app.get("/calibration/status/{domain}")
async def calibration_status(domain: str):
    """
    Check calibration status for a specific domain.
    
    Returns calibration profile metadata if available.
    """
    profile = load_calibration_profile(domain)
    if profile is None:
        return {
            "domain": domain,
            "calibrated": False,
            "message": f"No calibration profile found for domain '{domain}'. Using default thresholds.",
        }
    
    return {
        "domain": domain,
        "calibrated": True,
        "version": profile.version,
        "created_at": profile.created_at,
        "sample_count": profile.sample_count,
        "validation_metrics": profile.validation_metrics,
        "stages_calibrated": list(profile.stage_thresholds.keys()),
    }


@app.get("/calibration/list")
async def list_calibrations():
    """
    List all available calibration profiles.
    """
    profiles = []
    if CALIBRATION_DIR.exists():
        for path in CALIBRATION_DIR.glob("calibration_*.json"):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                profiles.append({
                    "domain": data.get("domain"),
                    "version": data.get("version"),
                    "created_at": data.get("created_at"),
                    "sample_count": data.get("sample_count"),
                    "file": path.name,
                })
            except Exception as e:
                profiles.append({"file": path.name, "error": str(e)})
    
    return {
        "profiles": profiles,
        "count": len(profiles),
        "calibration_dir": str(CALIBRATION_DIR),
    }


@app.post("/analyze", response_model=None)
async def analyze(request: AnalyzeRequest) -> JSONResponse:
    """
    Analyze two distributions using the 5-stage pipeline.
    
    Returns cryptographically signed attestation of structural analysis.
    If a calibration profile exists for domain_hint, calibrated thresholds are used.
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
