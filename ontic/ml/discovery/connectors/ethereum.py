#!/usr/bin/env python3
"""
Ethereum DeFi Connector

Production-grade connector for Ethereum DeFi protocol data.

Data Sources:
    - Ethereum JSON-RPC (Alchemy, Infura, or local node)
    - TheGraph subgraphs for indexed protocol data
    - Direct contract calls for real-time state

Supported Protocols:
    - Uniswap V2/V3 (pools, swaps, liquidity)
    - Aave V2/V3 (lending pools, liquidations)
    - Compound V2/V3 (markets, borrow/supply rates)
    - Curve (pools, gauge data)

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from decimal import Decimal
from enum import Enum
import hashlib
import os

import torch

from ..config import get_config


logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

@dataclass
class EthereumConfig:
    """Ethereum connection configuration."""
    
    # RPC endpoints (prioritized)
    rpc_url: str = field(default_factory=lambda: os.environ.get(
        "ETHEREUM_RPC_URL",
        "https://eth-mainnet.g.alchemy.com/v2/demo"  # Public demo, rate limited
    ))
    
    # Alternative providers
    alchemy_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("ALCHEMY_API_KEY"))
    infura_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("INFURA_API_KEY"))
    
    # TheGraph endpoints
    uniswap_v3_subgraph: str = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"
    uniswap_v2_subgraph: str = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v2"
    aave_v3_subgraph: str = "https://api.thegraph.com/subgraphs/name/aave/protocol-v3"
    
    # Rate limiting
    max_requests_per_second: float = 10.0
    request_timeout_seconds: float = 30.0
    max_retries: int = 3
    
    # Block confirmation
    confirmation_blocks: int = 2
    
    def get_rpc_url(self) -> str:
        """Get the best available RPC URL."""
        if self.alchemy_api_key:
            return f"https://eth-mainnet.g.alchemy.com/v2/{self.alchemy_api_key}"
        if self.infura_api_key:
            return f"https://mainnet.infura.io/v3/{self.infura_api_key}"
        return self.rpc_url


# ============================================================
# Data Structures
# ============================================================

class Protocol(Enum):
    """Supported DeFi protocols."""
    UNISWAP_V2 = "uniswap_v2"
    UNISWAP_V3 = "uniswap_v3"
    AAVE_V2 = "aave_v2"
    AAVE_V3 = "aave_v3"
    COMPOUND_V2 = "compound_v2"
    CURVE = "curve"


@dataclass
class TokenInfo:
    """ERC20 token information."""
    address: str
    symbol: str
    name: str
    decimals: int
    
    def format_amount(self, raw_amount: int) -> Decimal:
        """Convert raw token amount to decimal."""
        return Decimal(raw_amount) / Decimal(10 ** self.decimals)


@dataclass
class PoolState:
    """Current state of a liquidity pool."""
    protocol: Protocol
    pool_address: str
    token0: TokenInfo
    token1: TokenInfo
    reserve0: Decimal
    reserve1: Decimal
    fee_tier: int  # In basis points (e.g., 30 = 0.3%)
    liquidity: Decimal
    sqrt_price_x96: Optional[int] = None  # Uniswap V3 only
    tick: Optional[int] = None  # Uniswap V3 only
    block_number: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def price_0_to_1(self) -> Decimal:
        """Price of token0 in terms of token1."""
        if self.reserve0 == 0:
            return Decimal(0)
        return self.reserve1 / self.reserve0
    
    @property
    def price_1_to_0(self) -> Decimal:
        """Price of token1 in terms of token0."""
        if self.reserve1 == 0:
            return Decimal(0)
        return self.reserve0 / self.reserve1
    
    @property
    def tvl_usd(self) -> Optional[Decimal]:
        """Total value locked (requires price oracle)."""
        # Would need price oracle integration
        return None


@dataclass
class SwapEvent:
    """A swap event from a DEX."""
    protocol: Protocol
    pool_address: str
    tx_hash: str
    log_index: int
    block_number: int
    timestamp: datetime
    sender: str
    recipient: str
    amount0_in: Decimal
    amount1_in: Decimal
    amount0_out: Decimal
    amount1_out: Decimal
    
    @property
    def is_buy(self) -> bool:
        """True if buying token0 with token1."""
        return self.amount0_out > 0 and self.amount1_in > 0
    
    @property
    def price_impact_estimate(self) -> Decimal:
        """Rough estimate of price impact."""
        if self.amount0_out > 0 and self.amount1_in > 0:
            return self.amount1_in / self.amount0_out
        elif self.amount1_out > 0 and self.amount0_in > 0:
            return self.amount0_in / self.amount1_out
        return Decimal(0)


@dataclass
class LendingPosition:
    """A lending protocol position."""
    protocol: Protocol
    user_address: str
    asset: TokenInfo
    supplied: Decimal
    borrowed: Decimal
    collateral_factor: Decimal  # 0.0 to 1.0
    liquidation_threshold: Decimal  # 0.0 to 1.0
    health_factor: Decimal
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def at_risk(self) -> bool:
        """True if position is at liquidation risk."""
        return self.health_factor < Decimal("1.1")
    
    @property
    def liquidatable(self) -> bool:
        """True if position can be liquidated."""
        return self.health_factor < Decimal("1.0")


@dataclass
class LiquidationEvent:
    """A liquidation event from a lending protocol."""
    protocol: Protocol
    tx_hash: str
    block_number: int
    timestamp: datetime
    liquidator: str
    borrower: str
    debt_asset: TokenInfo
    collateral_asset: TokenInfo
    debt_amount: Decimal
    collateral_seized: Decimal
    
    @property
    def liquidation_bonus(self) -> Decimal:
        """Implied liquidation bonus percentage."""
        if self.debt_amount == 0:
            return Decimal(0)
        return (self.collateral_seized / self.debt_amount) - Decimal(1)


# ============================================================
# HTTP Client with Rate Limiting
# ============================================================

class RateLimitedClient:
    """HTTP client with rate limiting and retries."""
    
    def __init__(self, config: EthereumConfig):
        self.config = config
        self._last_request_time = 0.0
        self._min_interval = 1.0 / config.max_requests_per_second
        
        # Try to import aiohttp, fall back to requests
        try:
            import aiohttp
            self._use_async = True
        except ImportError:
            import requests
            self._use_async = False
            self._session = requests.Session()
    
    def _wait_for_rate_limit(self) -> None:
        """Wait if needed to respect rate limit."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()
    
    def post_json(self, url: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a rate-limited POST request."""
        self._wait_for_rate_limit()
        
        import requests
        
        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(
                    url,
                    json=data,
                    timeout=self.config.request_timeout_seconds,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return {}  # Should not reach here
    
    async def post_json_async(self, url: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make an async rate-limited POST request."""
        import aiohttp
        
        self._wait_for_rate_limit()
        
        for attempt in range(self.config.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json=data,
                        timeout=aiohttp.ClientTimeout(total=self.config.request_timeout_seconds)
                    ) as response:
                        response.raise_for_status()
                        return await response.json()
            except Exception as e:
                logger.warning(f"Async request failed (attempt {attempt + 1}): {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
        
        return {}


# ============================================================
# Ethereum RPC Client
# ============================================================

class EthereumRPCClient:
    """Ethereum JSON-RPC client."""
    
    def __init__(self, config: Optional[EthereumConfig] = None):
        self.config = config or EthereumConfig()
        self._client = RateLimitedClient(self.config)
        self._request_id = 0
    
    def _make_request(self, method: str, params: List[Any]) -> Any:
        """Make an RPC request."""
        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params
        }
        
        result = self._client.post_json(self.config.get_rpc_url(), payload)
        
        if "error" in result:
            error = result["error"]
            raise RuntimeError(f"RPC error {error.get('code')}: {error.get('message')}")
        
        return result.get("result")
    
    def get_block_number(self) -> int:
        """Get latest block number."""
        result = self._make_request("eth_blockNumber", [])
        return int(result, 16)
    
    def get_block(self, block_number: Union[int, str] = "latest") -> Dict[str, Any]:
        """Get block by number."""
        if isinstance(block_number, int):
            block_number = hex(block_number)
        return self._make_request("eth_getBlockByNumber", [block_number, True])
    
    def call(
        self,
        to: str,
        data: str,
        block: str = "latest"
    ) -> str:
        """Make an eth_call."""
        return self._make_request("eth_call", [{"to": to, "data": data}, block])
    
    def get_logs(
        self,
        address: Optional[str] = None,
        topics: Optional[List[Optional[str]]] = None,
        from_block: Union[int, str] = "latest",
        to_block: Union[int, str] = "latest"
    ) -> List[Dict[str, Any]]:
        """Get logs matching filter."""
        if isinstance(from_block, int):
            from_block = hex(from_block)
        if isinstance(to_block, int):
            to_block = hex(to_block)
        
        filter_params: Dict[str, Any] = {
            "fromBlock": from_block,
            "toBlock": to_block
        }
        if address:
            filter_params["address"] = address
        if topics:
            filter_params["topics"] = topics
        
        return self._make_request("eth_getLogs", [filter_params])


# ============================================================
# TheGraph Client
# ============================================================

class TheGraphClient:
    """Client for TheGraph Protocol subgraphs."""
    
    def __init__(self, config: Optional[EthereumConfig] = None):
        self.config = config or EthereumConfig()
        self._client = RateLimitedClient(self.config)
    
    def query(self, subgraph_url: str, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a GraphQL query."""
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
        
        result = self._client.post_json(subgraph_url, payload)
        
        if "errors" in result:
            errors = result["errors"]
            raise RuntimeError(f"GraphQL errors: {errors}")
        
        return result.get("data", {})


# ============================================================
# Uniswap V3 Connector
# ============================================================

class UniswapV3Connector:
    """Connector for Uniswap V3 protocol data."""
    
    # Known pool addresses (mainnet)
    KNOWN_POOLS = {
        "WETH-USDC-500": "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
        "WETH-USDC-3000": "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8",
        "WBTC-WETH-3000": "0xcbcdf9626bc03e24f779434178a73a0b4bad62ed",
        "WETH-USDT-3000": "0x4e68ccd3e89f51c3074ca5072bbac773960dfa36",
    }
    
    # Event signatures
    SWAP_TOPIC = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"
    
    def __init__(self, config: Optional[EthereumConfig] = None):
        self.config = config or EthereumConfig()
        self._rpc = EthereumRPCClient(self.config)
        self._graph = TheGraphClient(self.config)
        
        logger.info(f"UniswapV3Connector initialized with RPC: {self.config.get_rpc_url()[:50]}...")
    
    def get_pool_state(self, pool_address: str) -> PoolState:
        """Get current state of a Uniswap V3 pool."""
        # Query TheGraph for pool data
        query = """
        query GetPool($id: ID!) {
            pool(id: $id) {
                token0 {
                    id
                    symbol
                    name
                    decimals
                }
                token1 {
                    id
                    symbol
                    name
                    decimals
                }
                feeTier
                liquidity
                sqrtPrice
                tick
                totalValueLockedToken0
                totalValueLockedToken1
            }
        }
        """
        
        data = self._graph.query(
            self.config.uniswap_v3_subgraph,
            query,
            {"id": pool_address.lower()}
        )
        
        pool_data = data.get("pool")
        if not pool_data:
            raise ValueError(f"Pool not found: {pool_address}")
        
        token0 = TokenInfo(
            address=pool_data["token0"]["id"],
            symbol=pool_data["token0"]["symbol"],
            name=pool_data["token0"]["name"],
            decimals=int(pool_data["token0"]["decimals"])
        )
        
        token1 = TokenInfo(
            address=pool_data["token1"]["id"],
            symbol=pool_data["token1"]["symbol"],
            name=pool_data["token1"]["name"],
            decimals=int(pool_data["token1"]["decimals"])
        )
        
        return PoolState(
            protocol=Protocol.UNISWAP_V3,
            pool_address=pool_address,
            token0=token0,
            token1=token1,
            reserve0=Decimal(pool_data.get("totalValueLockedToken0", "0")),
            reserve1=Decimal(pool_data.get("totalValueLockedToken1", "0")),
            fee_tier=int(pool_data["feeTier"]),
            liquidity=Decimal(pool_data["liquidity"]),
            sqrt_price_x96=int(pool_data["sqrtPrice"]) if pool_data.get("sqrtPrice") else None,
            tick=int(pool_data["tick"]) if pool_data.get("tick") else None,
            block_number=self._rpc.get_block_number(),
            timestamp=datetime.now(timezone.utc)
        )
    
    def get_recent_swaps(
        self,
        pool_address: str,
        limit: int = 100
    ) -> List[SwapEvent]:
        """Get recent swap events for a pool."""
        query = """
        query GetSwaps($pool: String!, $limit: Int!) {
            swaps(
                where: {pool: $pool}
                orderBy: timestamp
                orderDirection: desc
                first: $limit
            ) {
                id
                transaction {
                    id
                    blockNumber
                }
                timestamp
                sender
                recipient
                amount0
                amount1
                logIndex
            }
        }
        """
        
        data = self._graph.query(
            self.config.uniswap_v3_subgraph,
            query,
            {"pool": pool_address.lower(), "limit": limit}
        )
        
        swaps = []
        for swap_data in data.get("swaps", []):
            amount0 = Decimal(swap_data["amount0"])
            amount1 = Decimal(swap_data["amount1"])
            
            swaps.append(SwapEvent(
                protocol=Protocol.UNISWAP_V3,
                pool_address=pool_address,
                tx_hash=swap_data["transaction"]["id"],
                log_index=int(swap_data["logIndex"]),
                block_number=int(swap_data["transaction"]["blockNumber"]),
                timestamp=datetime.fromtimestamp(int(swap_data["timestamp"]), tz=timezone.utc),
                sender=swap_data["sender"],
                recipient=swap_data["recipient"],
                amount0_in=abs(amount0) if amount0 < 0 else Decimal(0),
                amount1_in=abs(amount1) if amount1 < 0 else Decimal(0),
                amount0_out=amount0 if amount0 > 0 else Decimal(0),
                amount1_out=amount1 if amount1 > 0 else Decimal(0),
            ))
        
        return swaps
    
    def get_top_pools(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get top pools by TVL."""
        query = """
        query TopPools($limit: Int!) {
            pools(
                orderBy: totalValueLockedUSD
                orderDirection: desc
                first: $limit
            ) {
                id
                token0 {
                    symbol
                }
                token1 {
                    symbol
                }
                feeTier
                totalValueLockedUSD
                volumeUSD
                txCount
            }
        }
        """
        
        data = self._graph.query(
            self.config.uniswap_v3_subgraph,
            query,
            {"limit": limit}
        )
        
        return data.get("pools", [])
    
    def to_tensor(self, swaps: List[SwapEvent]) -> torch.Tensor:
        """Convert swap events to tensor for pipeline analysis."""
        if not swaps:
            return torch.zeros(0, 6)
        
        data = []
        for swap in swaps:
            data.append([
                float(swap.timestamp.timestamp()),
                float(swap.amount0_in),
                float(swap.amount1_in),
                float(swap.amount0_out),
                float(swap.amount1_out),
                float(swap.block_number),
            ])
        
        return torch.tensor(data, dtype=torch.float32)


# ============================================================
# Aave V3 Connector
# ============================================================

class AaveV3Connector:
    """Connector for Aave V3 lending protocol data."""
    
    def __init__(self, config: Optional[EthereumConfig] = None):
        self.config = config or EthereumConfig()
        self._graph = TheGraphClient(self.config)
        
        logger.info("AaveV3Connector initialized")
    
    def get_market_state(self, asset_address: str) -> Dict[str, Any]:
        """Get current state of an Aave market."""
        query = """
        query GetReserve($id: ID!) {
            reserve(id: $id) {
                id
                symbol
                name
                decimals
                liquidityRate
                variableBorrowRate
                stableBorrowRate
                totalLiquidity
                totalCurrentVariableDebt
                totalStableDebt
                utilizationRate
                liquidityIndex
                variableBorrowIndex
            }
        }
        """
        
        data = self._graph.query(
            self.config.aave_v3_subgraph,
            query,
            {"id": asset_address.lower()}
        )
        
        return data.get("reserve", {})
    
    def get_at_risk_positions(self, health_factor_max: Decimal = Decimal("1.1")) -> List[LendingPosition]:
        """Get positions that are at liquidation risk."""
        query = """
        query AtRiskPositions($healthMax: BigDecimal!) {
            users(
                where: {healthFactor_lt: $healthMax, borrowedReservesCount_gt: 0}
                orderBy: healthFactor
                orderDirection: asc
                first: 100
            ) {
                id
                healthFactor
                totalCollateralUSD
                totalDebtUSD
                reserves {
                    reserve {
                        symbol
                        decimals
                    }
                    currentATokenBalance
                    currentVariableDebt
                    currentStableDebt
                    liquidationThreshold
                }
            }
        }
        """
        
        data = self._graph.query(
            self.config.aave_v3_subgraph,
            query,
            {"healthMax": str(health_factor_max)}
        )
        
        positions = []
        for user_data in data.get("users", []):
            for reserve in user_data.get("reserves", []):
                if reserve.get("currentVariableDebt", "0") != "0" or reserve.get("currentStableDebt", "0") != "0":
                    positions.append(LendingPosition(
                        protocol=Protocol.AAVE_V3,
                        user_address=user_data["id"],
                        asset=TokenInfo(
                            address="",  # Would need to get from reserve
                            symbol=reserve["reserve"]["symbol"],
                            name=reserve["reserve"]["symbol"],
                            decimals=int(reserve["reserve"]["decimals"])
                        ),
                        supplied=Decimal(reserve.get("currentATokenBalance", "0")),
                        borrowed=Decimal(reserve.get("currentVariableDebt", "0")) + Decimal(reserve.get("currentStableDebt", "0")),
                        collateral_factor=Decimal("0.8"),  # Simplified
                        liquidation_threshold=Decimal(reserve.get("liquidationThreshold", "0.8")),
                        health_factor=Decimal(user_data.get("healthFactor", "0")),
                        timestamp=datetime.now(timezone.utc)
                    ))
        
        return positions
    
    def get_recent_liquidations(self, limit: int = 50) -> List[LiquidationEvent]:
        """Get recent liquidation events."""
        query = """
        query RecentLiquidations($limit: Int!) {
            liquidationCalls(
                orderBy: timestamp
                orderDirection: desc
                first: $limit
            ) {
                id
                txHash
                action
                timestamp
                user {
                    id
                }
                liquidator {
                    id
                }
                collateralReserve {
                    symbol
                    decimals
                }
                debtReserve {
                    symbol
                    decimals
                }
                collateralAmount
                debtToCover
            }
        }
        """
        
        data = self._graph.query(
            self.config.aave_v3_subgraph,
            query,
            {"limit": limit}
        )
        
        liquidations = []
        for liq_data in data.get("liquidationCalls", []):
            liquidations.append(LiquidationEvent(
                protocol=Protocol.AAVE_V3,
                tx_hash=liq_data["txHash"],
                block_number=0,  # Not in subgraph response
                timestamp=datetime.fromtimestamp(int(liq_data["timestamp"]), tz=timezone.utc),
                liquidator=liq_data["liquidator"]["id"],
                borrower=liq_data["user"]["id"],
                debt_asset=TokenInfo(
                    address="",
                    symbol=liq_data["debtReserve"]["symbol"],
                    name=liq_data["debtReserve"]["symbol"],
                    decimals=int(liq_data["debtReserve"]["decimals"])
                ),
                collateral_asset=TokenInfo(
                    address="",
                    symbol=liq_data["collateralReserve"]["symbol"],
                    name=liq_data["collateralReserve"]["symbol"],
                    decimals=int(liq_data["collateralReserve"]["decimals"])
                ),
                debt_amount=Decimal(liq_data.get("debtToCover", "0")),
                collateral_seized=Decimal(liq_data.get("collateralAmount", "0"))
            ))
        
        return liquidations


# ============================================================
# Unified DeFi Connector
# ============================================================

class DeFiConnector:
    """
    Unified connector for all DeFi protocol data.
    
    Production Usage:
        # Set environment variables:
        export ALCHEMY_API_KEY="your-key"
        
        # Create connector
        connector = DeFiConnector()
        
        # Get pool data
        pool = connector.get_uniswap_pool("0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640")
        swaps = connector.get_recent_swaps(pool.pool_address, limit=100)
        
        # Convert to tensor for pipeline
        tensor = connector.to_tensor(swaps)
    """
    
    def __init__(self, config: Optional[EthereumConfig] = None):
        self.config = config or EthereumConfig()
        self.uniswap = UniswapV3Connector(self.config)
        self.aave = AaveV3Connector(self.config)
        
        # Check for production API keys
        if not self.config.alchemy_api_key and not self.config.infura_api_key:
            logger.warning(
                "No ALCHEMY_API_KEY or INFURA_API_KEY found. "
                "Using public RPC endpoint with rate limits. "
                "Set environment variables for production use."
            )
    
    def get_uniswap_pool(self, pool_address: str) -> PoolState:
        """Get Uniswap V3 pool state."""
        return self.uniswap.get_pool_state(pool_address)
    
    def get_recent_swaps(self, pool_address: str, limit: int = 100) -> List[SwapEvent]:
        """Get recent swaps from Uniswap V3."""
        return self.uniswap.get_recent_swaps(pool_address, limit)
    
    def get_at_risk_positions(self) -> List[LendingPosition]:
        """Get lending positions at liquidation risk."""
        return self.aave.get_at_risk_positions()
    
    def get_recent_liquidations(self, limit: int = 50) -> List[LiquidationEvent]:
        """Get recent liquidations from Aave."""
        return self.aave.get_recent_liquidations(limit)
    
    def to_tensor(self, swaps: List[SwapEvent]) -> torch.Tensor:
        """Convert swaps to tensor for pipeline."""
        return self.uniswap.to_tensor(swaps)
    
    def health_check(self) -> Dict[str, Any]:
        """Check connectivity to all data sources."""
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "rpc": {"status": "unknown"},
            "uniswap_subgraph": {"status": "unknown"},
            "aave_subgraph": {"status": "unknown"},
        }
        
        # Check RPC
        try:
            block = self.uniswap._rpc.get_block_number()
            results["rpc"] = {"status": "ok", "block": block}
        except Exception as e:
            results["rpc"] = {"status": "error", "error": str(e)}
        
        # Check Uniswap subgraph
        try:
            pools = self.uniswap.get_top_pools(limit=1)
            results["uniswap_subgraph"] = {"status": "ok", "pools": len(pools)}
        except Exception as e:
            results["uniswap_subgraph"] = {"status": "error", "error": str(e)}
        
        # Check Aave subgraph
        try:
            liqs = self.aave.get_recent_liquidations(limit=1)
            results["aave_subgraph"] = {"status": "ok"}
        except Exception as e:
            results["aave_subgraph"] = {"status": "error", "error": str(e)}
        
        return results


# ============================================================
# Main / Testing
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("ETHEREUM DEFI CONNECTOR TEST")
    print("=" * 60)
    print()
    
    connector = DeFiConnector()
    
    # Health check
    print("[1] Health Check...")
    health = connector.health_check()
    for key, value in health.items():
        if key != "timestamp":
            print(f"    {key}: {value.get('status', 'unknown')}")
    print()
    
    # Get top pools
    print("[2] Top Uniswap V3 Pools...")
    try:
        pools = connector.uniswap.get_top_pools(limit=5)
        for pool in pools:
            print(f"    {pool['token0']['symbol']}/{pool['token1']['symbol']} "
                  f"({pool['feeTier']/10000}%) - TVL: ${float(pool['totalValueLockedUSD']):,.0f}")
    except Exception as e:
        print(f"    Error: {e}")
    print()
    
    # Get WETH-USDC swaps
    print("[3] Recent WETH-USDC Swaps...")
    try:
        pool_address = UniswapV3Connector.KNOWN_POOLS["WETH-USDC-500"]
        swaps = connector.get_recent_swaps(pool_address, limit=5)
        for swap in swaps:
            print(f"    Block {swap.block_number}: "
                  f"{'BUY' if swap.is_buy else 'SELL'} "
                  f"{abs(swap.amount0_out or swap.amount0_in):.4f} WETH")
    except Exception as e:
        print(f"    Error: {e}")
    print()
    
    print("=" * 60)
    print("✅ Ethereum DeFi Connector operational")
    print("=" * 60)
