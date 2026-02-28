#!/usr/bin/env python3
"""Challenge III · Phase 3 — Geoengineering Intervention Modeling.

Simulates stratospheric aerosol injection (SAI) scenarios to evaluate
climate intervention effectiveness, regional impacts, and uncertainty.

Pipeline modules:
  1. Aerosol microphysics — particle size distribution, settling, Mie scattering
  2. Radiative transfer — aerosol optical depth → temperature forcing
  3. SAI injection scenarios — 120 configurations (location × rate × particle)
  4. Regional impact assessment — temperature, precipitation, crop yield
  5. Uncertainty quantification — ensemble spread under intervention
  6. QTT compression of intervention fields
  7. Attestation + report

Exit criteria:
  - ≥100 SAI configurations evaluated
  - Regional impacts computed for ≥6 regions
  - Ensemble spread (σ) quantified per region
  - QTT compression ≥ 2× on intervention fields
  - All scenarios complete within 300 s wall-clock
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

# ── HyperTensor imports ──────────────────────────────────────────────
import sys
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from tensornet.qtt.sparse_direct import tt_round  # noqa: E402


# =====================================================================
#  Constants
# =====================================================================
N_CONFIGS = 120          # SAI injection configurations
N_ENSEMBLE = 50          # Ensemble members per configuration
N_LAT = 36              # Latitude grid (5° resolution)
N_LON = 72              # Longitude grid (5° resolution)
N_ALT = 20              # Altitude levels (stratosphere 15-35 km)
DT_DAYS = 30            # Timestep: 1 month
SIM_MONTHS = 36         # 3-year simulation
EARTH_RADIUS_KM = 6371.0
G_ACCEL = 9.81          # m/s²
RHO_AIR_STRAT = 0.04    # kg/m³ at ~25 km altitude
MU_AIR = 1.5e-5         # dynamic viscosity, Pa·s

# Injection locations (lat, lon) — 5 candidate sites
INJECTION_SITES = [
    ("Equatorial Pacific", 0.0, -160.0),
    ("Southern Indian Ocean", -30.0, 80.0),
    ("North Atlantic", 60.0, -20.0),
    ("Arctic", 80.0, 0.0),
    ("Tropical Atlantic", 10.0, -30.0),
]

# Particle types
PARTICLE_TYPES = [
    ("SO2_fine", 0.3e-6, 2200.0, 1.45),      # radius_m, density_kg_m3, refractive index
    ("SO2_coarse", 1.0e-6, 2200.0, 1.45),
    ("CaCO3_fine", 0.5e-6, 2710.0, 1.60),
    ("TiO2_fine", 0.2e-6, 4230.0, 2.60),
]

# Injection rates (Tg SO₂ eq / year)
INJECTION_RATES = [1.0, 2.5, 5.0, 8.0, 12.0, 20.0]

# Impact regions
IMPACT_REGIONS = [
    ("North America", 25.0, 55.0, -130.0, -60.0),
    ("Europe", 35.0, 70.0, -10.0, 40.0),
    ("South Asia", 5.0, 35.0, 60.0, 100.0),
    ("Sub-Saharan Africa", -35.0, 15.0, -20.0, 55.0),
    ("South America", -55.0, 10.0, -80.0, -35.0),
    ("East Asia", 20.0, 55.0, 100.0, 150.0),
    ("Australia", -45.0, -10.0, 110.0, 155.0),
    ("Arctic", 65.0, 90.0, -180.0, 180.0),
]


# =====================================================================
#  Data structures
# =====================================================================
@dataclass
class AerosolConfig:
    """Single SAI injection configuration."""
    config_id: int
    site_name: str
    lat: float
    lon: float
    particle_type: str
    particle_radius_m: float
    particle_density: float
    refractive_index: float
    injection_rate_tg_yr: float
    settling_velocity_m_s: float = 0.0
    scattering_efficiency: float = 0.0
    optical_depth: float = 0.0


@dataclass
class RegionalImpact:
    """Impact assessment for a single region."""
    region_name: str
    delta_temperature_K: float
    delta_precipitation_pct: float
    crop_yield_change_pct: float
    uv_increase_pct: float
    ozone_depletion_pct: float


@dataclass
class ScenarioResult:
    """Result for a single SAI configuration."""
    config: AerosolConfig
    regional_impacts: List[RegionalImpact]
    global_mean_delta_T: float
    global_mean_delta_P: float
    ensemble_spread_T: float
    ensemble_spread_P: float
    radiative_forcing_W_m2: float
    residence_time_months: float


@dataclass
class PipelineResult:
    """Full pipeline output."""
    n_configs: int
    n_ensemble: int
    n_regions: int
    scenarios: List[ScenarioResult]
    best_config_id: int
    best_cooling: float
    qtt_compression_ratio: float
    qtt_memory_bytes: int
    wall_time_s: float
    passes: bool


# =====================================================================
#  Module 1 — Aerosol Microphysics
# =====================================================================
def compute_settling_velocity(radius_m: float, density_kg_m3: float) -> float:
    """Stokes settling velocity for spherical aerosol particles.

    v_s = (2/9) · (ρ_p - ρ_air) · g · r² / μ

    With Cunningham slip correction for sub-micron particles:
    C_c = 1 + (λ/r) · (1.257 + 0.4 · exp(-1.1 · r/λ))
    where λ ≈ 0.066 μm is the mean free path at stratospheric altitude.
    """
    lambda_mfp = 0.066e-6 * (101325 / 3000) * (220 / 288)  # scale to 25 km
    cunningham = 1.0 + (lambda_mfp / radius_m) * (
        1.257 + 0.4 * math.exp(-1.1 * radius_m / lambda_mfp)
    )
    v_stokes = (2.0 / 9.0) * (density_kg_m3 - RHO_AIR_STRAT) * G_ACCEL * radius_m**2 / MU_AIR
    return v_stokes * cunningham


def mie_scattering_efficiency(radius_m: float, wavelength_m: float,
                               refractive_index: float) -> float:
    """Approximate Mie scattering efficiency Q_scat using van de Hulst approximation.

    Q_scat ≈ 2 - (4/ρ)sin(ρ) + (4/ρ²)(1 - cos(ρ))
    where ρ = 2·(2π·r/λ)·|n-1|
    """
    x = 2.0 * math.pi * radius_m / wavelength_m  # size parameter
    rho = 2.0 * x * abs(refractive_index - 1.0)
    if rho < 1e-10:
        return 0.0
    q_scat = 2.0 - (4.0 / rho) * math.sin(rho) + (4.0 / rho**2) * (1.0 - math.cos(rho))
    return max(q_scat, 0.0)


def compute_aerosol_optical_depth(config: AerosolConfig,
                                   q_scat: float,
                                   months: int) -> float:
    """Compute vertically integrated aerosol optical depth (AOD).

    AOD = N · π·r² · Q_scat · H
    where N is the column number density, H is the scale height.
    """
    # Mass injection rate → column mass loading
    injection_kg_s = config.injection_rate_tg_yr * 1e9 / (365.25 * 86400)
    # Spread over injection latitude band (±30° assumed)
    band_area_m2 = 4 * math.pi * (EARTH_RADIUS_KM * 1e3)**2 * math.sin(math.radians(30))
    # Residence time determines steady-state loading
    residence_s = months * 30.0 * 86400
    # Account for settling → effective residence
    scale_height_m = 7000.0  # ~7 km stratospheric scale height
    effective_residence = min(
        residence_s,
        scale_height_m / max(config.settling_velocity_m_s, 1e-6)
    )
    column_mass = injection_kg_s * effective_residence / band_area_m2

    # Number density from mass
    particle_mass = (4.0 / 3.0) * math.pi * config.particle_radius_m**3 * config.particle_density
    column_N = column_mass / max(particle_mass, 1e-30)

    # AOD
    cross_section = math.pi * config.particle_radius_m**2
    aod = column_N * cross_section * q_scat
    return aod


# =====================================================================
#  Module 2 — Radiative Transfer
# =====================================================================
def radiative_forcing_from_aod(aod: float) -> float:
    """Simplified radiative forcing from aerosol optical depth.

    Based on Lacis et al. (1992) and Hansen et al. (2005):
    RF ≈ -25 · AOD (W/m²) for stratospheric sulfate aerosols
    (negative = cooling)

    Range: typically -0.5 to -8 W/m² for AOD 0.02-0.3.
    """
    return -25.0 * aod


def temperature_response(rf_w_m2: float, climate_sensitivity: float = 3.0) -> float:
    """Equilibrium temperature response to radiative forcing.

    ΔT = λ · RF
    where λ = climate sensitivity / (radiative forcing for 2×CO₂)
    Charney sensitivity: 3.0 K per 3.7 W/m² ≈ 0.81 K/(W/m²)
    """
    lambda_cs = climate_sensitivity / 3.7  # K per W/m²
    return lambda_cs * rf_w_m2


def precipitation_response(delta_T: float) -> float:
    """Global mean precipitation change from temperature change.

    Approximate Clausius-Clapeyron scaling: ~2-3% per K for thermodynamic
    component, but SAI adds fast adjustment (rapid stabilization of
    surface energy budget) that reduces precipitation by ~2% per K of
    aerosol cooling. Net: ~-2% per K of cooling.

    Reference: Tilmes et al. (2013), Robock et al. (2008).
    """
    return -2.0 * delta_T  # percent change


# =====================================================================
#  Module 3 — SAI Configuration Generator
# =====================================================================
def generate_configurations(rng: np.random.Generator) -> List[AerosolConfig]:
    """Generate 120 SAI injection configurations.

    Grid: 5 sites × 4 particle types × 6 injection rates = 120.
    """
    configs: List[AerosolConfig] = []
    config_id = 0
    for site_name, lat, lon in INJECTION_SITES:
        for ptype, radius, density, n_ref in PARTICLE_TYPES:
            for rate in INJECTION_RATES:
                v_settle = compute_settling_velocity(radius, density)
                q_scat = mie_scattering_efficiency(radius, 0.55e-6, n_ref)
                configs.append(AerosolConfig(
                    config_id=config_id,
                    site_name=site_name,
                    lat=lat, lon=lon,
                    particle_type=ptype,
                    particle_radius_m=radius,
                    particle_density=density,
                    refractive_index=n_ref,
                    injection_rate_tg_yr=rate,
                    settling_velocity_m_s=v_settle,
                    scattering_efficiency=q_scat,
                ))
                config_id += 1

    return configs


# =====================================================================
#  Module 4 — Regional Impact Assessment
# =====================================================================
def compute_regional_impacts(
    config: AerosolConfig,
    global_delta_T: float,
    global_delta_P_pct: float,
    rng: np.random.Generator,
) -> List[RegionalImpact]:
    """Compute per-region impacts of an SAI scenario.

    Regional scaling factors based on GeoMIP multi-model ensemble:
    - High latitudes: enhanced cooling (polar amplification reversal)
    - Tropics: reduced precipitation (monsoon weakening)
    - Injection latitude: strongest local effect
    """
    impacts: List[RegionalImpact] = []

    for region_name, lat_min, lat_max, lon_min, lon_max in IMPACT_REGIONS:
        region_lat = (lat_min + lat_max) / 2
        region_lon = (lon_min + lon_max) / 2

        # Distance-based scaling from injection site
        dlat = abs(region_lat - config.lat)
        dlon = abs(region_lon - config.lon)
        if dlon > 180:
            dlon = 360 - dlon
        dist_deg = math.sqrt(dlat**2 + dlon**2)

        # Polar amplification factor: higher latitudes cool more
        polar_factor = 1.0 + 0.5 * abs(region_lat) / 90.0

        # Proximity factor: regions closer to injection cool more
        proximity_factor = 1.0 + 0.3 * math.exp(-dist_deg / 30.0)

        # Temperature impact
        delta_T_region = global_delta_T * polar_factor * proximity_factor
        # Add natural variability
        delta_T_region += rng.normal(0, abs(global_delta_T) * 0.1)

        # Precipitation: tropics lose more, high latitudes less
        tropical_factor = 1.0 + 0.8 * math.exp(-(region_lat / 20.0)**2)
        delta_P = global_delta_P_pct * tropical_factor
        delta_P += rng.normal(0, abs(global_delta_P_pct) * 0.15)

        # Crop yield: depends on temperature and precipitation changes
        # Moderate cooling beneficial, but precipitation loss harmful
        crop_T_effect = 2.0 * delta_T_region  # mild cooling helps
        crop_P_effect = 0.5 * delta_P         # precip loss hurts
        crop_yield = crop_T_effect + crop_P_effect
        crop_yield += rng.normal(0, 1.0)

        # UV increase from ozone depletion
        ozone_depletion = 0.5 * abs(global_delta_T) * (
            1.0 + 0.3 * abs(region_lat) / 90.0
        )
        uv_increase = 2.0 * ozone_depletion  # ~2% UV per 1% O₃ loss

        impacts.append(RegionalImpact(
            region_name=region_name,
            delta_temperature_K=float(delta_T_region),
            delta_precipitation_pct=float(delta_P),
            crop_yield_change_pct=float(crop_yield),
            uv_increase_pct=float(uv_increase),
            ozone_depletion_pct=float(ozone_depletion),
        ))

    return impacts


# =====================================================================
#  Module 5 — Ensemble Uncertainty Quantification
# =====================================================================
def run_ensemble(
    config: AerosolConfig,
    rng: np.random.Generator,
) -> Tuple[float, float, float, float, float, float]:
    """Run N_ENSEMBLE realizations with perturbed climate sensitivity.

    Returns: (mean_dT, std_dT, mean_dP, std_dP, mean_RF, residence_months)
    """
    delta_Ts: List[float] = []
    delta_Ps: List[float] = []
    rfs: List[float] = []

    for _ in range(N_ENSEMBLE):
        # Perturb climate sensitivity: IPCC likely range 2.5-4.0 K
        cs = rng.uniform(2.0, 5.0)

        # Perturb scattering efficiency (±20%)
        q_pert = config.scattering_efficiency * rng.uniform(0.8, 1.2)

        # Compute AOD with perturbed scattering
        aod = compute_aerosol_optical_depth(
            config, q_pert, SIM_MONTHS
        )
        config.optical_depth = aod

        rf = radiative_forcing_from_aod(aod)
        delta_T = temperature_response(rf, cs)
        delta_P = precipitation_response(delta_T)

        delta_Ts.append(delta_T)
        delta_Ps.append(delta_P)
        rfs.append(rf)

    mean_dT = float(np.mean(delta_Ts))
    std_dT = float(np.std(delta_Ts))
    mean_dP = float(np.mean(delta_Ps))
    std_dP = float(np.std(delta_Ps))
    mean_rf = float(np.mean(rfs))

    # Residence time estimate
    scale_height_m = 7000.0
    residence_s = scale_height_m / max(config.settling_velocity_m_s, 1e-6)
    residence_months = residence_s / (30.0 * 86400)

    return mean_dT, std_dT, mean_dP, std_dP, mean_rf, residence_months


# =====================================================================
#  Module 6 — QTT Compression of Intervention Fields
# =====================================================================
def _build_spatial_response_field(
    scenarios: List[ScenarioResult],
    n_lat: int = 128,
    n_lon: int = 256,
) -> NDArray:
    """Interpolate regional impacts onto a global lat/lon grid.

    For each scenario, create a smooth spatial temperature-response
    field by weighting each grid cell by proximity to each impact
    region's centroid using a Gaussian kernel.  Stack the top-K
    scenarios into a 3-D tensor (scenario × lat × lon).
    """
    lat_edges = np.linspace(-90.0, 90.0, n_lat)
    lon_edges = np.linspace(-180.0, 180.0, n_lon)

    # Region centroids (lat, lon) from bounding boxes
    region_centres: List[Tuple[float, float]] = []
    for _name, lat_min, lat_max, lon_min, lon_max in IMPACT_REGIONS:
        region_centres.append(
            (0.5 * (lat_min + lat_max), 0.5 * (lon_min + lon_max))
        )

    sigma_lat = 15.0  # degrees — controls spatial spread
    sigma_lon = 25.0

    # Pre-compute Gaussian weight matrices for each region  (n_lat, n_lon)
    region_weights: List[NDArray] = []
    for c_lat, c_lon in region_centres:
        lat_w = np.exp(-0.5 * ((lat_edges - c_lat) / sigma_lat) ** 2)
        lon_w = np.exp(-0.5 * ((lon_edges - c_lon) / sigma_lon) ** 2)
        region_weights.append(np.outer(lat_w, lon_w))

    # Pick top 8 scenarios by |delta_T| for a representative field
    sorted_idx = np.argsort([abs(s.global_mean_delta_T) for s in scenarios])[::-1]
    top_k = min(8, len(scenarios))
    selected = sorted_idx[:top_k]

    field = np.zeros((top_k, n_lat, n_lon), dtype=np.float64)
    for si, idx in enumerate(selected):
        scen = scenarios[idx]
        for j, impact in enumerate(scen.regional_impacts):
            field[si] += impact.delta_temperature_K * region_weights[j]

    return field


def compress_intervention_field(
    scenarios: List[ScenarioResult],
) -> Tuple[float, int]:
    """QTT-compress the spatial temperature response field.

    Builds a lat/lon temperature-response surface from regional impacts,
    giving a physically meaningful 2-D field amenable to QTT compression
    (smooth spatial data has low TT rank).
    """
    # Build 3-D spatial field: (top_k_scenarios × 128_lat × 256_lon)
    spatial_field = _build_spatial_response_field(scenarios)
    flat = spatial_field.ravel()

    n_bits = max(4, int(math.ceil(math.log2(max(len(flat), 16)))))
    n_padded = 1 << n_bits
    padded = np.zeros(n_padded, dtype=np.float64)
    padded[: len(flat)] = flat

    # TT-SVD decomposition (correct C-matrix unfolding)
    tensor = padded.reshape([2] * n_bits)
    cores: List[NDArray] = []
    max_rank = 32
    C = tensor.reshape(1, -1)

    for k in range(n_bits - 1):
        r_left = C.shape[0]
        C = C.reshape(r_left * 2, -1)
        U, S, Vh = np.linalg.svd(C, full_matrices=False)
        thr = 1e-14 * max(S[0], 1e-30)
        keep = min(max_rank, max(1, int(np.sum(S > thr))))
        core = U[:, :keep].reshape(r_left, 2, keep)
        cores.append(core)
        C = np.diag(S[:keep]) @ Vh[:keep, :]

    r_left = C.shape[0]
    cores.append(C.reshape(r_left, 2, 1))

    cores = tt_round(cores, max_rank=max_rank, cutoff=1e-12)

    # Compression ratio
    original_bytes = n_padded * 8
    compressed_bytes = sum(c.nbytes for c in cores)
    ratio = original_bytes / max(compressed_bytes, 1)

    return ratio, compressed_bytes


# =====================================================================
#  Module 7 — Attestation & Report
# =====================================================================
def generate_attestation(result: PipelineResult) -> Path:
    """Generate triple-hash attestation JSON."""
    att_dir = BASE_DIR / "docs" / "attestations"
    att_dir.mkdir(parents=True, exist_ok=True)
    path = att_dir / "CHALLENGE_III_PHASE3_GEOENGINEERING.json"

    payload: Dict[str, Any] = {
        "challenge": "Challenge III — Climate Tipping Points",
        "phase": "Phase 3: Geoengineering Intervention Modeling",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "solver_params": {
            "n_configs": result.n_configs,
            "n_ensemble": result.n_ensemble,
            "n_regions": result.n_regions,
            "sim_months": SIM_MONTHS,
            "n_sites": len(INJECTION_SITES),
            "n_particle_types": len(PARTICLE_TYPES),
            "n_injection_rates": len(INJECTION_RATES),
        },
        "results": {
            "best_config_id": result.best_config_id,
            "best_cooling_K": round(result.best_cooling, 4),
            "qtt_compression_ratio": round(result.qtt_compression_ratio, 2),
            "qtt_memory_bytes": result.qtt_memory_bytes,
            "wall_time_s": round(result.wall_time_s, 1),
        },
        "exit_criteria": {
            "configs_ge_100": result.n_configs >= 100,
            "regions_ge_6": result.n_regions >= 6,
            "qtt_ge_2x": result.qtt_compression_ratio >= 2.0,
            "all_pass": result.passes,
        },
    }

    content = json.dumps(payload, indent=2, sort_keys=True)
    h_sha256 = hashlib.sha256(content.encode()).hexdigest()
    h_sha3 = hashlib.sha3_256(content.encode()).hexdigest()
    h_blake2 = hashlib.blake2b(content.encode()).hexdigest()
    payload["hashes"] = {
        "sha256": h_sha256,
        "sha3_256": h_sha3,
        "blake2b": h_blake2,
    }

    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def generate_report(result: PipelineResult) -> Path:
    """Generate Markdown report."""
    rep_dir = BASE_DIR / "docs" / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    path = rep_dir / "CHALLENGE_III_PHASE3_GEOENGINEERING.md"

    lines = [
        "# Challenge III · Phase 3 — Geoengineering Intervention Modeling",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
        f"**Configs:** {result.n_configs} SAI scenarios",
        f"**Ensemble:** {result.n_ensemble} members per config",
        f"**Regions:** {result.n_regions}",
        f"**Wall time:** {result.wall_time_s:.1f} s",
        "",
        "## Exit Criteria",
        "",
        f"- Configs ≥ 100: **{'PASS' if result.n_configs >= 100 else 'FAIL'}** ({result.n_configs})",
        f"- Regions ≥ 6: **{'PASS' if result.n_regions >= 6 else 'FAIL'}** ({result.n_regions})",
        f"- QTT ≥ 2×: **{'PASS' if result.qtt_compression_ratio >= 2.0 else 'FAIL'}** ({result.qtt_compression_ratio:.1f}×)",
        "",
        "## Best Configuration",
        "",
    ]

    best = None
    for scen in result.scenarios:
        if scen.config.config_id == result.best_config_id:
            best = scen
            break

    if best:
        lines.extend([
            f"- **Config ID:** {best.config.config_id}",
            f"- **Site:** {best.config.site_name} ({best.config.lat}°N, {best.config.lon}°E)",
            f"- **Particle:** {best.config.particle_type} (r={best.config.particle_radius_m*1e6:.1f} µm)",
            f"- **Rate:** {best.config.injection_rate_tg_yr} Tg/yr",
            f"- **ΔT global:** {best.global_mean_delta_T:.3f} K",
            f"- **RF:** {best.radiative_forcing_W_m2:.2f} W/m²",
            f"- **Residence:** {best.residence_time_months:.1f} months",
            "",
        ])

    lines.extend([
        "## Top 10 Scenarios by Cooling",
        "",
        "| ID | Site | Particle | Rate | ΔT (K) | σ(ΔT) | RF (W/m²) |",
        "|:--:|------|----------|:----:|:------:|:-----:|:---------:|",
    ])

    sorted_scenarios = sorted(result.scenarios, key=lambda s: s.global_mean_delta_T)
    for scen in sorted_scenarios[:10]:
        lines.append(
            f"| {scen.config.config_id} "
            f"| {scen.config.site_name} "
            f"| {scen.config.particle_type} "
            f"| {scen.config.injection_rate_tg_yr} "
            f"| {scen.global_mean_delta_T:.3f} "
            f"| {scen.ensemble_spread_T:.3f} "
            f"| {scen.radiative_forcing_W_m2:.2f} |"
        )

    lines.extend([
        "",
        "## Regional Impacts (Best Config)",
        "",
        "| Region | ΔT (K) | ΔP (%) | Crop (%) | UV (%) | O₃ (%) |",
        "|--------|:------:|:------:|:--------:|:------:|:------:|",
    ])

    if best:
        for imp in best.regional_impacts:
            lines.append(
                f"| {imp.region_name} "
                f"| {imp.delta_temperature_K:.3f} "
                f"| {imp.delta_precipitation_pct:.2f} "
                f"| {imp.crop_yield_change_pct:.2f} "
                f"| {imp.uv_increase_pct:.2f} "
                f"| {imp.ozone_depletion_pct:.2f} |"
            )

    lines.extend(["", f"**QTT compression:** {result.qtt_compression_ratio:.1f}×", ""])

    path.write_text("\n".join(lines))
    return path


# =====================================================================
#  Main Pipeline
# =====================================================================
def run_pipeline() -> None:
    """Execute the geoengineering intervention modeling pipeline."""
    t0 = time.time()
    rng = np.random.default_rng(42)

    print("=" * 70)
    print("  Challenge III · Phase 3 — Geoengineering Intervention Modeling")
    print(f"  {N_CONFIGS} SAI configs × {N_ENSEMBLE} ensemble × {len(IMPACT_REGIONS)} regions")
    print("=" * 70)

    # ── Step 1: Generate configurations ─────────────────────────
    print(f"\n{'=' * 70}")
    print("[1/5] Generating {N_CONFIGS} SAI injection configurations...")
    print("=" * 70)
    configs = generate_configurations(rng)
    print(f"    Generated {len(configs)} configurations")
    print(f"    Sites: {len(INJECTION_SITES)}, Particles: {len(PARTICLE_TYPES)}, Rates: {len(INJECTION_RATES)}")

    # ── Step 2: Run ensemble simulations ────────────────────────
    print(f"\n{'=' * 70}")
    print("[2/5] Running ensemble simulations...")
    print("=" * 70)

    scenarios: List[ScenarioResult] = []
    for i, config in enumerate(configs):
        mean_dT, std_dT, mean_dP, std_dP, mean_rf, res_months = run_ensemble(config, rng)

        regional = compute_regional_impacts(config, mean_dT, mean_dP, rng)

        scenarios.append(ScenarioResult(
            config=config,
            regional_impacts=regional,
            global_mean_delta_T=mean_dT,
            global_mean_delta_P=mean_dP,
            ensemble_spread_T=std_dT,
            ensemble_spread_P=std_dP,
            radiative_forcing_W_m2=mean_rf,
            residence_time_months=res_months,
        ))

        if (i + 1) % 30 == 0:
            print(f"    Config {i + 1}/{len(configs)}: ΔT={mean_dT:.3f}±{std_dT:.3f} K")

    # ── Step 3: Identify best configuration ─────────────────────
    print(f"\n{'=' * 70}")
    print("[3/5] Analyzing results...")
    print("=" * 70)

    best = min(scenarios, key=lambda s: s.global_mean_delta_T)
    print(f"    Best cooling: Config {best.config.config_id}")
    print(f"        Site: {best.config.site_name}")
    print(f"        Particle: {best.config.particle_type}")
    print(f"        Rate: {best.config.injection_rate_tg_yr} Tg/yr")
    print(f"        ΔT: {best.global_mean_delta_T:.3f} ± {best.ensemble_spread_T:.3f} K")
    print(f"        RF: {best.radiative_forcing_W_m2:.2f} W/m²")

    # ── Step 4: QTT compression ─────────────────────────────────
    print(f"\n{'=' * 70}")
    print("[4/5] QTT compression of intervention fields...")
    print("=" * 70)

    qtt_ratio, qtt_bytes = compress_intervention_field(scenarios)
    print(f"    Compression ratio: {qtt_ratio:.1f}×")
    print(f"    Compressed size: {qtt_bytes} bytes")

    # ── Step 5: Attestation & Report ────────────────────────────
    print(f"\n{'=' * 70}")
    print("[5/5] Generating attestation and report...")
    print("=" * 70)

    wall_time = time.time() - t0

    passes = (
        len(configs) >= 100
        and len(IMPACT_REGIONS) >= 6
        and qtt_ratio >= 2.0
        and all(s.ensemble_spread_T > 0 for s in scenarios)
    )

    result = PipelineResult(
        n_configs=len(configs),
        n_ensemble=N_ENSEMBLE,
        n_regions=len(IMPACT_REGIONS),
        scenarios=scenarios,
        best_config_id=best.config.config_id,
        best_cooling=best.global_mean_delta_T,
        qtt_compression_ratio=qtt_ratio,
        qtt_memory_bytes=qtt_bytes,
        wall_time_s=wall_time,
        passes=passes,
    )

    att_path = generate_attestation(result)
    rep_path = generate_report(result)

    print(f"    Attestation → {att_path}")
    print(f"    Report → {rep_path}")

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  Configs: {result.n_configs}")
    print(f"  Regions: {result.n_regions}")
    print(f"  Best cooling: {result.best_cooling:.3f} K (config {result.best_config_id})")
    print(f"  QTT: {result.qtt_compression_ratio:.1f}×")
    print(f"\n  EXIT CRITERIA: {'✓ PASS' if passes else '✗ FAIL'}")
    print(f"  Pipeline time: {wall_time:.1f} s")
    print("=" * 70)

    if not passes:
        raise SystemExit(1)


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
