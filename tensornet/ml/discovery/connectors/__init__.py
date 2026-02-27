"""
Live Data Connectors for Autonomous Discovery Engine

Phase 5-8: Real-time and historical data sources.

Connectors:
- CoinbaseL2Connector: WebSocket L2 order book feed
- HistoricalDataLoader: Load historical market events
- StreamingPipeline: Real-time sliding window analysis
- DeFiConnector: Ethereum DeFi protocols (Uniswap, Aave)
- MolecularConnector: RCSB PDB + AlphaFold structures
- FusionConnector: Tokamak plasma + TT-compressed MHD

Environment Variables (for production):
- ALCHEMY_API_KEY: Alchemy API key for Ethereum RPC
- INFURA_API_KEY: Infura API key for Ethereum RPC
- ETHEREUM_RPC_URL: Custom Ethereum RPC endpoint
- PDB_CACHE_DIR: Directory for caching PDB structures
"""

from .coinbase_l2 import CoinbaseL2Connector, L2Update, L2Snapshot
from .historical import HistoricalDataLoader, HistoricalEvent
from .streaming import StreamingPipeline, StreamingConfig, StreamingResult

# Phase 8: Real data connectors
from .ethereum import (
    DeFiConnector,
    UniswapV3Connector,
    AaveV3Connector,
    EthereumConfig,
    PoolState,
    SwapEvent,
    LendingPosition,
    LiquidationEvent,
    Protocol,
)
from .molecular_pdb import (
    MolecularConnector,
    RCSBConnector,
    AlphaFoldConnector,
    PDBConfig,
    PDBEntry,
    SequenceAlignment,
    LigandInfo,
)

# Phase 9: Fusion plasma connector
from .fusion import (
    FusionConnector,
    FusionConfig,
    FusionAnalysisResult,
    MHDMode,
    ConfinementMetrics,
    TTCompressedMHDAnalyzer,
)

__all__ = [
    # Markets
    "CoinbaseL2Connector",
    "L2Update",
    "L2Snapshot",
    "HistoricalDataLoader",
    "HistoricalEvent",
    "StreamingPipeline",
    "StreamingConfig",
    "StreamingResult",
    # DeFi
    "DeFiConnector",
    "UniswapV3Connector",
    "AaveV3Connector",
    "EthereumConfig",
    "PoolState",
    "SwapEvent",
    "LendingPosition",
    "LiquidationEvent",
    "Protocol",
    # Molecular
    "MolecularConnector",
    "RCSBConnector",
    "AlphaFoldConnector",
    "PDBConfig",
    "PDBEntry",
    "SequenceAlignment",
    "LigandInfo",
    # Fusion
    "FusionConnector",
    "FusionConfig",
    "FusionAnalysisResult",
    "MHDMode",
    "ConfinementMetrics",
    "TTCompressedMHDAnalyzer",
]
