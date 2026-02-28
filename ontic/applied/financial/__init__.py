"""
TensorNet Financial Module - Liquidity Weather System

Phase 6: Real-Time Order Book Physics
=====================================

The "Alpha" Pipeline:
    Order Book → Density Field → Navier-Stokes → Price Prediction

Core Hypothesis:
    Price is a particle suspended in a liquidity fluid.
    It follows the path of least resistance (low pressure).
    By modeling the order book as a compressible fluid,
    we can predict "dam breaks" before price moves.

Physics Mapping:
    - Space (x): Price levels ($95,000 - $105,000 for BTC)
    - Density (ρ): log(Volume) at each price level
    - Pressure (P): Limit order concentration
    - Velocity (u): Market order flow
    - Temperature (T): Volatility (σ of last 100 trades)

Solver:
    ∂u/∂t = -∇P + ν∇²u
    (Price acceleration = -Pressure gradient + Friction)

Visual Output:
    - Green Fog: Deep buy liquidity (support)
    - Red Fog: Deep sell liquidity (resistance)
    - White Particle: Current price
    - Signal: Red fog thins → Breakout imminent

Target Market:
    - Proprietary Trading Desks
    - Crypto Market Makers
    - Quantitative Hedge Funds
"""

from ontic.applied.financial.feed import MarketDataFeed, OrderBookFluid
from ontic.applied.financial.solver import LiquiditySolver, solve_price_flow

__all__ = [
    "OrderBookFluid",
    "MarketDataFeed",
    "LiquiditySolver",
    "solve_price_flow",
]
