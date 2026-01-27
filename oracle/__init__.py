"""
Oracle Module
=============
Quantum Tensor Train (QTT) based market prediction engine.

Components:
- qtt_encoder: Order Book → Tensor Train encoding
- oracle_engine: DMRG training + Monte Carlo simulation
- live_oracle: Real-time Coinbase integration

The Core Thesis:
  Compression IS prediction.
  The same math that compresses state also predicts it.
"""

from .qtt_encoder import (
    QTTEncoder,
    OrderBook,
    TensorTrain,
    TensorCore,
    FeatureVector,
    MarketState
)

from .oracle_engine import (
    OracleEngine,
    DMRGTrainer,
    MatrixProductOperator,
    MPOCore,
    SimulationResult
)

from .live_oracle import (
    LiveOracle,
    Prediction
)

__all__ = [
    # Encoder
    'QTTEncoder',
    'OrderBook', 
    'TensorTrain',
    'TensorCore',
    'FeatureVector',
    'MarketState',
    
    # Engine
    'OracleEngine',
    'DMRGTrainer',
    'MatrixProductOperator',
    'MPOCore',
    'SimulationResult',
    
    # Live
    'LiveOracle',
    'Prediction'
]

__version__ = '0.1.0'
