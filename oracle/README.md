# 🌌 Galaxy Feed Oracle - Trinity System

**Real-time market entropy analysis with GPU-accelerated regime detection.**

## Overview

The Galaxy Feed Oracle processes live Binance Futures data through a Trinity gate system
to generate swing trading signals. It combines three data streams into a coherent signal
generator that fires only when all conditions align.

## Trinity Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TRINITY SIGNAL SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   FIREHOSE 1: aggTrade       FIREHOSE 2: !forceOrder     FIREHOSE 3: !markPrice
│   ─────────────────────      ────────────────────────    ────────────────────
│   • 40 perpetual pairs       • All liquidation events    • All funding rates
│   • Price normalization      • Long/short tracking       • Weighted average
│   • Volume distribution      • $5M cascade threshold     • 0.03% tension level
│   • EMA smoothing (α=0.1)    • Rolling 60s window        • Bias detection
│                                                                             │
│                              ┌───────────┐                                  │
│   Gate 1: FUNDING ──────────▶│           │                                  │
│   Is market tense?           │  TRINITY  │──────▶ SWING_LONG / SWING_SHORT  │
│                              │  ENGINE   │                                  │
│   Gate 2: ENTROPY ──────────▶│           │                                  │
│   Is structure breaking?     │ (All 3    │──────▶ LONG_SETUP / SHORT_SETUP  │
│                              │  gates    │                                  │
│   Gate 3: LIQUIDATIONS ─────▶│  align)   │──────▶ SQUEEZE DETECTION         │
│   Is cascade happening?      │           │                                  │
│                              └───────────┘                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Signal Types

| Signal | Condition | Meaning |
|--------|-----------|---------|
| 🟢 SWING LONG | All 3 gates + short liquidation cascade | High conviction long entry |
| 🔴 SWING SHORT | All 3 gates + long liquidation cascade | High conviction short entry |
| ⚡ LONG SETUP | Funding + Entropy aligned | Preparing for long, needs confirmation |
| ⚡ SHORT SETUP | Funding + Entropy aligned | Preparing for short, needs confirmation |
| 🔥 LONG SQUEEZE | Extreme long funding + liquidation cascade | Forced long exits |
| 🔥 SHORT SQUEEZE | Extreme short funding + liquidation cascade | Forced short exits |

## Regime Classification

Based on EMA-smoothed entropy (H):

| Regime | H Range | Trading Mode |
|--------|---------|--------------|
| QUIET | H < 3.0 | Accumulation/distribution |
| BUILDING | 3.0 ≤ H < 5.0 | Position building |
| VOLATILE | 5.0 ≤ H < 7.0 | Active trading |
| CHAOTIC | 7.0 ≤ H < 9.0 | High conviction signals only |
| PLASMA | H ≥ 9.0 | Do not trade raw |

## Quick Start

```bash
# Run the Trinity System (detection only)
python3 oracle/galaxy_feed_v3.py

# Run the NS Predictor (detection + prediction)
python3 oracle/ns_predictor.py

# Output format:
# [HH:MM:SS] BTC:$XX,XXX ETH:$X,XXX SOL:$XXX.X | REGIME H=X.X(raw) Δ=±X.XX | F:BIAS +X.XXX% | L:$X.XM S:$X.XM | Gates:🟢🟢🟢 | XXX/s
```

## NS Predictor - Navier-Stokes Market Prediction

The NS Predictor treats the market as a **turbulent fluid** and uses physics-based
forward integration to predict entropy evolution, regime transitions, and price
distribution dynamics.

### Physical Model

```
Market as Turbulent Fluid:
━━━━━━━━━━━━━━━━━━━━━━━━━━
• Price momentum → Velocity field
• Volume flow → Density field  
• Entropy → Turbulent kinetic energy
• Regime transitions → Reynolds number thresholds
• Liquidation cascades → Energy injection (forcing)
• Funding rate → Pressure gradient

Governing Equations:
━━━━━━━━━━━━━━━━━━━━
1. Vorticity Transport:
   ∂ω/∂t + (u·∇)ω = ν∇²ω + F_liq + F_funding

2. Stream Function:
   ∇²ψ = -ω

3. Kolmogorov Cascade:
   E(k) ~ k^(-5/3)
```

### Prediction Horizons

| Horizon | Confidence | Use Case |
|---------|------------|----------|
| 1 min | 97% | Scalping signals |
| 5 min | 85% | Short-term positioning |
| 15 min | 61% | Swing entry timing |
| 30 min | 37% | Trend confirmation |
| 60 min | 14% | Regime outlook |

### Trading Signals

| Signal | Condition |
|--------|-----------|
| 🟢 LONG MOMENTUM | Rising entropy + negative funding |
| 🔴 SHORT MOMENTUM | Rising entropy + positive funding |
| 🔵 LONG REVERSION | Vol fading + shorts overextended |
| 🔵 SHORT REVERSION | Vol fading + longs overextended |
| 📈 VOL EXPANSION | Entropy rising through horizons |
| 📉 VOL CONTRACTION | Entropy falling through horizons |
| ⚠️ REDUCE EXPOSURE | Cascade probability > 30% |
| 🛡️ HEDGE | Vol multiplier > 2x in chaotic regime |

## Performance

| Metric | Value |
|--------|-------|
| Symbols | 40 perpetual contracts |
| Throughput | 700+ trades/sec sustained |
| GPU Memory | ~1.1 GB |
| Latency | <100ms tick-to-decision |
| Entropy Smoothing | EMA α=0.1 |

## Dependencies

- Python 3.10+
- PyTorch 2.0+
- Triton 3.0+
- websockets
- NVIDIA GPU (CUDA)

## Architecture

### Core Components

1. **GalaxySlicer** - GPU-accelerated entropy calculation
   - Triton JIT kernel for zero-loop processing
   - Normalized price/volume inputs
   - EMA smoothing for noise reduction

2. **LiquidationTracker** - Cascade detection
   - Parses `!forceOrder@arr` stream
   - Tracks long/short liquidation volume
   - 60-second rolling window
   - $1M spike / $5M cascade thresholds

3. **FundingTracker** - Market positioning
   - Parses `!markPrice@arr` stream
   - Weighted average by market cap
   - 0.03% tension / 0.10% extreme thresholds
   - Long/Short/Neutral bias detection

4. **TrinityEngine** - Signal generation
   - Evaluates all three gates
   - 30-second cooldown between signals
   - Only fires when gates align

## Files

| File | Purpose |
|------|---------|
| `galaxy_feed_v3.py` | **Production** - Trinity System (detection) |
| `ns_predictor.py` | **Production** - NS Predictor (prediction) |
| `binance_firehose.py` | Standalone Binance perpetuals feed |
| `coinbase_firehose.py` | Coinbase USD/USDC pairs feed |
| `triton_slicer.py` | Base Triton kernel implementation |
| `zero_loop_kernel.py` | Optimized zero-loop kernel |

## Regional Notes

Binance Global REST API is blocked in some regions (HTTP 451). The WebSocket
Futures endpoints (`wss://fstream.binance.com`) bypass this restriction and
work globally.

## Author

Genesis Stack / HyperTensor VM

## License

Proprietary - Part of HyperTensor-VM
