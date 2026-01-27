# Tensor Genesis Oracle Node

**Domain-Agnostic Structure Engine** — The Universal Truth Machine

## What It Does

```
Raw Numbers → [OT → SGW → RKHS → PH → GA] → Signed Attestation
```

The Oracle Node takes **any numerical data** and produces a **cryptographically signed proof** of its mathematical structure. It doesn't know if your data is:
- Climate temperatures
- Stock prices
- Blood flow measurements
- Network traffic
- Sensor readings

It just analyzes the **structure**.

## The 5-Stage Pipeline

| Stage | Math | Question Answered |
|-------|------|-------------------|
| 1. OT | Optimal Transport | "How much did the distribution shift?" |
| 2. SGW | Spectral Wavelets | "At what scale is the change?" |
| 3. RKHS | Kernel Methods | "Is this anomalous?" |
| 4. PH | Persistent Homology | "What shape is the pattern?" |
| 5. GA | Geometric Algebra | "Which direction is the trend?" |

## Quick Start

### Run Locally

```bash
# Install dependencies
pip install fastapi uvicorn numpy torch scipy

# Start the Oracle Node
python oracle_node/server.py

# In another terminal, test it
python oracle_node/test_client.py
```

### Run with Docker

```bash
# Build
docker build -t tensor-genesis-oracle -f oracle_node/Dockerfile .

# Run
docker run -p 8080:8080 tensor-genesis-oracle

# Test
curl http://localhost:8080/health
```

## API Usage

### Compare Two Distributions

```bash
curl -X POST http://localhost:8080/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "distribution_a": [1.0, 2.0, 3.0, ...],
    "distribution_b": [1.5, 2.5, 3.5, ...],
    "domain_hint": "climate"
  }'
```

### Response

```json
{
  "attestation": "TENSOR_GENESIS_ORACLE_ATTESTATION",
  "timestamp": "2026-01-27T12:00:00Z",
  "stage_1_ot": {
    "wasserstein_2": 0.0234,
    "interpretation": "Distribution shift magnitude"
  },
  "stage_2_sgw": {
    "dominant_scale": 4.0,
    "is_global": true,
    "interpretation": "Dominant change at scale 4.0 (global)"
  },
  "stage_3_rkhs": {
    "anomaly_level": "MODERATE",
    "interpretation": "Anomaly level: MODERATE (MMD=0.1523)"
  },
  "stage_4_ph": {
    "shape_type": "UNIMODAL",
    "interpretation": "Shape: UNIMODAL — Single concentrated change region"
  },
  "stage_5_ga": {
    "trend": "INCREASING",
    "interpretation": "Trend: INCREASING — Distribution shifting right/up"
  },
  "total_time_seconds": 0.042,
  "sha256": "a1b2c3d4e5f6..."
}
```

## The Universal Translator

The same pipeline handles completely different domains:

| Stage | Climate | Finance | Medical |
|-------|---------|---------|---------|
| OT | "Temperature shifted 2.5°C" | "Liquidity moved 50bps" | "Blood flow changed 20ml/min" |
| SGW | "Local storm vs global warming" | "Flash crash vs recession" | "Clot vs hypertension" |
| RKHS | "This heatwave is abnormal" | "This volume is suspicious" | "This density is cancerous" |
| PH | "Storm has an eye (cyclone)" | "Trades form a loop (wash trading)" | "Tumor has a cavity (necrosis)" |
| GA | "Front moving North-East" | "Market trending Bearish" | "Growth toward artery" |

## Why This Is A Moat

Traditional approach:
- Build a climate model
- Build a finance model  
- Build a medical model
- 3 teams, 3 codebases, 3x cost

Your approach:
- Build **one** Structure Engine
- Swap only the data adapter
- Same math, infinite domains

## License

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
