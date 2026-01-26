#!/usr/bin/env python3
"""
DeFi Data Ingester for Autonomous Discovery Engine

Converts smart contract data into tensor format for discovery.
"""

import torch
import json
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone


@dataclass
class ContractData:
    """Parsed smart contract data."""
    address: str
    name: str = ""
    functions: List[Dict] = field(default_factory=list)
    storage_slots: Dict[str, Any] = field(default_factory=dict)
    call_graph: List[Tuple[str, str]] = field(default_factory=list)
    token_flows: List[Dict] = field(default_factory=list)
    events: List[Dict] = field(default_factory=list)
    

@dataclass
class DeFiSnapshot:
    """Point-in-time DeFi state."""
    timestamp: datetime
    tvl: float
    token_balances: Dict[str, float]
    pool_reserves: Dict[str, Tuple[float, float]]
    price_ratios: Dict[str, float]


class DeFiIngester:
    """
    Ingest DeFi protocol data for discovery analysis.
    
    Converts:
        - Contract bytecode → state space tensor
        - Call graphs → adjacency matrix
        - Token flows → distribution tensor
        - Price series → time series tensor
    """
    
    def __init__(self, grid_bits: int = 12):
        self.grid_bits = grid_bits
        self.grid_size = 2 ** grid_bits
        
    def from_contract_abi(self, abi: List[Dict], address: str = "") -> ContractData:
        """Parse contract from ABI."""
        functions = []
        events = []
        
        for item in abi:
            if item.get("type") == "function":
                functions.append({
                    "name": item.get("name", ""),
                    "inputs": item.get("inputs", []),
                    "outputs": item.get("outputs", []),
                    "stateMutability": item.get("stateMutability", ""),
                })
            elif item.get("type") == "event":
                events.append({
                    "name": item.get("name", ""),
                    "inputs": item.get("inputs", []),
                })
        
        return ContractData(
            address=address,
            functions=functions,
            events=events,
        )
    
    def build_call_graph_tensor(self, contract: ContractData) -> torch.Tensor:
        """
        Build adjacency matrix from function call relationships.
        
        Returns: (N, N) tensor where N = number of functions
        """
        n_funcs = len(contract.functions)
        if n_funcs == 0:
            return torch.zeros(1, 1)
        
        # Initialize adjacency matrix
        adj = torch.zeros(n_funcs, n_funcs)
        
        # Build function name to index mapping
        func_idx = {f["name"]: i for i, f in enumerate(contract.functions)}
        
        # Add edges from explicit call graph
        for caller, callee in contract.call_graph:
            if caller in func_idx and callee in func_idx:
                adj[func_idx[caller], func_idx[callee]] = 1.0
        
        # Heuristic: external calls likely call internal helpers
        external_funcs = [i for i, f in enumerate(contract.functions) 
                        if f.get("stateMutability") in ["payable", "nonpayable"]]
        internal_funcs = [i for i, f in enumerate(contract.functions)
                        if f.get("stateMutability") in ["view", "pure"]]
        
        for ext in external_funcs:
            for internal in internal_funcs[:3]:  # Assume top 3 helpers
                adj[ext, internal] = 0.5  # Weaker connection
        
        return adj
    
    def build_token_flow_distribution(
        self, 
        flows: List[Dict],
        n_bins: int = None
    ) -> torch.Tensor:
        """
        Build distribution tensor from token transfer events.
        
        Args:
            flows: List of {from, to, amount, token} dicts
            n_bins: Number of histogram bins (default: grid_size)
            
        Returns: (n_bins,) distribution tensor
        """
        if n_bins is None:
            n_bins = min(self.grid_size, 1024)
        
        if not flows:
            return torch.ones(n_bins) / n_bins  # Uniform if no data
        
        # Extract amounts
        amounts = torch.tensor([float(f.get("amount", 0)) for f in flows])
        
        if amounts.max() == amounts.min():
            return torch.ones(n_bins) / n_bins
        
        # Normalize to [0, 1]
        amounts_norm = (amounts - amounts.min()) / (amounts.max() - amounts.min() + 1e-10)
        
        # Build histogram
        hist = torch.histc(amounts_norm, bins=n_bins, min=0, max=1)
        
        # Normalize to probability distribution
        hist = hist / (hist.sum() + 1e-10)
        
        return hist
    
    def build_price_series_tensor(
        self,
        prices: List[float],
        target_length: int = None
    ) -> torch.Tensor:
        """
        Build price series tensor with log returns.
        
        Args:
            prices: List of prices over time
            target_length: Resample to this length (power of 2)
            
        Returns: (target_length,) tensor of log returns
        """
        if target_length is None:
            target_length = min(self.grid_size, len(prices))
        
        prices = torch.tensor(prices, dtype=torch.float64)
        
        if len(prices) < 2:
            return torch.zeros(target_length)
        
        # Compute log returns
        log_returns = torch.log(prices[1:] / prices[:-1])
        
        # Handle infinities
        log_returns = torch.nan_to_num(log_returns, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Resample if needed
        if len(log_returns) != target_length:
            # Simple linear interpolation
            x_old = torch.linspace(0, 1, len(log_returns))
            x_new = torch.linspace(0, 1, target_length)
            log_returns = torch.nn.functional.interpolate(
                log_returns.unsqueeze(0).unsqueeze(0),
                size=target_length,
                mode='linear',
                align_corners=True
            ).squeeze()
        
        return log_returns
    
    def build_tvl_distribution(
        self,
        snapshots: List[DeFiSnapshot],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build TVL distribution over time.
        
        Returns:
            (historical_dist, current_dist) tensors for OT comparison
        """
        if len(snapshots) < 2:
            uniform = torch.ones(1024) / 1024
            return uniform, uniform
        
        tvl_values = torch.tensor([s.tvl for s in snapshots])
        
        # Split into historical and current
        split_idx = len(tvl_values) * 3 // 4
        historical = tvl_values[:split_idx]
        current = tvl_values[split_idx:]
        
        # Build distributions
        def to_dist(values, n_bins=1024):
            if len(values) == 0:
                return torch.ones(n_bins) / n_bins
            values_norm = (values - values.min()) / (values.max() - values.min() + 1e-10)
            hist = torch.histc(values_norm, bins=n_bins, min=0, max=1)
            return hist / (hist.sum() + 1e-10)
        
        return to_dist(historical), to_dist(current)
    
    def ingest_uniswap_v3_pool(
        self,
        pool_address: str,
        swap_events: List[Dict],
        liquidity_events: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        """
        Ingest Uniswap V3 pool data.
        
        Returns dict with tensors for discovery pipeline:
            - swap_amounts: distribution of swap sizes
            - liquidity_changes: distribution of liquidity events
            - tick_activity: which price ticks are most active
        """
        # Swap amount distribution
        swap_amounts = [abs(float(e.get("amount0", 0))) for e in swap_events]
        swap_dist = self.build_token_flow_distribution(
            [{"amount": a} for a in swap_amounts]
        )
        
        # Liquidity change distribution
        liq_amounts = [abs(float(e.get("liquidity", 0))) for e in liquidity_events]
        liq_dist = self.build_token_flow_distribution(
            [{"amount": a} for a in liq_amounts]
        )
        
        # Tick activity (which price ranges are active)
        ticks = [int(e.get("tick", 0)) for e in swap_events if "tick" in e]
        if ticks:
            tick_tensor = torch.tensor(ticks, dtype=torch.float32)
            tick_hist = torch.histc(tick_tensor, bins=256)
            tick_hist = tick_hist / (tick_hist.sum() + 1e-10)
        else:
            tick_hist = torch.ones(256) / 256
        
        return {
            "swap_distribution": swap_dist,
            "liquidity_distribution": liq_dist,
            "tick_activity": tick_hist,
            "pool_address": pool_address,
        }
    
    def ingest_lending_protocol(
        self,
        protocol_name: str,
        borrow_events: List[Dict],
        repay_events: List[Dict],
        liquidation_events: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        """
        Ingest lending protocol data (Aave, Compound style).
        
        Returns dict with tensors:
            - borrow_dist: distribution of borrow amounts
            - repay_dist: distribution of repay amounts  
            - liquidation_dist: distribution of liquidations
            - health_factor_series: health factors over time
        """
        borrow_dist = self.build_token_flow_distribution(borrow_events)
        repay_dist = self.build_token_flow_distribution(repay_events)
        liquidation_dist = self.build_token_flow_distribution(liquidation_events)
        
        # Extract health factors if available
        health_factors = [
            float(e.get("healthFactor", 1.0)) 
            for e in liquidation_events
        ]
        if health_factors:
            hf_tensor = torch.tensor(health_factors)
        else:
            hf_tensor = torch.ones(100)
        
        return {
            "borrow_distribution": borrow_dist,
            "repay_distribution": repay_dist,
            "liquidation_distribution": liquidation_dist,
            "health_factors": hf_tensor,
            "protocol": protocol_name,
        }
    
    def compute_anomaly_features(
        self,
        current_dist: torch.Tensor,
        historical_dist: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute quick anomaly features before full pipeline.
        
        Returns:
            - kl_divergence: KL(current || historical)
            - tv_distance: Total variation distance
            - max_ratio: Max probability ratio (concentration)
        """
        # Add small epsilon for numerical stability
        eps = 1e-10
        current = current_dist + eps
        historical = historical_dist + eps
        
        # Normalize
        current = current / current.sum()
        historical = historical / historical.sum()
        
        # KL divergence
        kl = float((current * (current.log() - historical.log())).sum())
        
        # Total variation
        tv = float(0.5 * (current - historical).abs().sum())
        
        # Max ratio (detect concentration)
        max_ratio = float((current / historical).max())
        
        return {
            "kl_divergence": kl,
            "tv_distance": tv,
            "max_ratio": max_ratio,
        }


def main():
    """Test the DeFi ingester."""
    ingester = DeFiIngester(grid_bits=10)
    
    # Simulate some DeFi data
    print("Testing DeFi Ingester...")
    
    # Test token flow distribution
    flows = [
        {"from": "0xA", "to": "0xB", "amount": 100},
        {"from": "0xB", "to": "0xC", "amount": 250},
        {"from": "0xC", "to": "0xD", "amount": 50},
        {"from": "0xA", "to": "0xD", "amount": 1000},
        {"from": "0xD", "to": "0xE", "amount": 500},
    ]
    
    flow_dist = ingester.build_token_flow_distribution(flows)
    print(f"✓ Token flow distribution: shape={flow_dist.shape}, sum={flow_dist.sum():.4f}")
    
    # Test price series
    prices = [100 + i * 0.5 + 2 * torch.randn(1).item() for i in range(100)]
    price_tensor = ingester.build_price_series_tensor(prices, target_length=64)
    print(f"✓ Price series: shape={price_tensor.shape}, mean_return={price_tensor.mean():.6f}")
    
    # Test call graph
    contract = ContractData(
        address="0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
        name="UniswapV2Router",
        functions=[
            {"name": "swapExactTokensForTokens", "stateMutability": "nonpayable"},
            {"name": "addLiquidity", "stateMutability": "nonpayable"},
            {"name": "removeLiquidity", "stateMutability": "nonpayable"},
            {"name": "getAmountOut", "stateMutability": "view"},
            {"name": "quote", "stateMutability": "pure"},
        ],
        call_graph=[
            ("swapExactTokensForTokens", "getAmountOut"),
            ("addLiquidity", "quote"),
        ]
    )
    
    call_graph = ingester.build_call_graph_tensor(contract)
    print(f"✓ Call graph: shape={call_graph.shape}, edges={call_graph.sum():.1f}")
    
    # Test anomaly features
    historical = torch.randn(256).abs()
    historical = historical / historical.sum()
    current = historical.clone()
    current[100:150] *= 10  # Inject anomaly
    current = current / current.sum()
    
    features = ingester.compute_anomaly_features(current, historical)
    print(f"✓ Anomaly features: KL={features['kl_divergence']:.4f}, TV={features['tv_distance']:.4f}")
    
    print("\n✅ DeFi Ingester ready for Phase 1")


if __name__ == "__main__":
    main()
