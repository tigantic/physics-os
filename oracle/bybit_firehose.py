"""
BYBIT FIRE HOSE - ENTIRE MARKET INGESTION
==========================================
Subscribes to ALL perpetual contracts on Bybit.
Hundreds of trading pairs, thousands of trades per second.

This is the full market microstructure feed.
"""

import torch
import triton
import triton.language as tl
import asyncio
import aiohttp
import json
import time
import signal
import sys
from collections import deque
from typing import Dict, Tuple, List
from datetime import datetime

try:
    import websockets
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets", "-q"])
    import websockets

torch.set_default_device("cuda")


# =============================================================================
# TRITON KERNEL - DYNAMIC ASSET COUNT
# =============================================================================

@triton.jit
def firehose_kernel(
    ticks_ptr, encoder_ptr, template_ptr, output_ptr,
    stride_tick_b, stride_tick_a,
    stride_enc_a, stride_enc_b,
    stride_tpl_b1, stride_tpl_a, stride_tpl_b2,
    stride_out_batch, stride_out_b1, stride_out_a, stride_out_b2,
    BATCH: tl.constexpr,
    ASSETS: tl.constexpr,
    BOND: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    elements_per_core = BOND * ASSETS * BOND
    total_elements = BATCH * elements_per_core
    mask = offs < total_elements
    
    core_offs = offs % elements_per_core
    batch_idx = offs // elements_per_core
    
    b2 = core_offs % BOND
    rem = core_offs // BOND
    a = rem % ASSETS
    b1 = rem // ASSETS
    
    tpl_off = b1 * stride_tpl_b1 + a * stride_tpl_a + b2 * stride_tpl_b2
    val = tl.load(template_ptr + tpl_off, mask=mask)
    
    tick_off = batch_idx * stride_tick_b + a * stride_tick_a
    tick_val = tl.load(ticks_ptr + tick_off, mask=mask)
    
    enc_off = a * stride_enc_a + b1 * stride_enc_b
    enc_val = tl.load(encoder_ptr + enc_off, mask=mask)
    
    x = tick_val * enc_val
    exp_2x = tl.exp(2.0 * x)
    signal = (exp_2x - 1.0) / (exp_2x + 1.0)
    
    is_diag = (b1 == b2)
    val = val + tl.where(is_diag, signal, 0.0)
    
    out_off = (batch_idx * stride_out_batch + 
               b1 * stride_out_b1 + 
               a * stride_out_a + 
               b2 * stride_out_b2)
    tl.store(output_ptr + out_off, val, mask=mask)


# =============================================================================
# FIRE HOSE SLICER - SCALES TO HUNDREDS OF ASSETS
# =============================================================================

class FireHoseSlicer:
    def __init__(self, batch_size: int, window_size: int, bond_dim: int, max_assets: int):
        self.batch_size = batch_size
        self.window_size = window_size
        self.bond_dim = bond_dim
        self.max_assets = max_assets
        
        self.cores = torch.zeros(window_size, bond_dim, max_assets, bond_dim, dtype=torch.float32)
        self.head = 0
        self.count = 0
        
        torch.manual_seed(42)
        self.encoder = torch.randn(max_assets, bond_dim, dtype=torch.float32) * 0.1
        self.template = torch.randn(bond_dim, max_assets, bond_dim, dtype=torch.float32) * 0.01
        self.output_buffer = torch.empty(batch_size, bond_dim, max_assets, bond_dim, dtype=torch.float32)
        
        total_elements = batch_size * bond_dim * max_assets * bond_dim
        self.block_size = 1024
        self.grid = (triton.cdiv(total_elements, self.block_size),)
        
        self.total_ticks = 0
        self.entropy_history = deque(maxlen=100)
        self._warmup()

    def _warmup(self):
        warmup_ticks = torch.rand(self.batch_size, self.max_assets, dtype=torch.float32)
        for _ in range(5):
            self._launch_kernel(warmup_ticks)
        torch.cuda.synchronize()

    def _launch_kernel(self, ticks: torch.Tensor):
        firehose_kernel[self.grid](
            ticks, self.encoder, self.template, self.output_buffer,
            ticks.stride(0), ticks.stride(1),
            self.encoder.stride(0), self.encoder.stride(1),
            self.template.stride(0), self.template.stride(1), self.template.stride(2),
            self.output_buffer.stride(0), self.output_buffer.stride(1),
            self.output_buffer.stride(2), self.output_buffer.stride(3),
            BATCH=self.batch_size, ASSETS=self.max_assets, BOND=self.bond_dim, BLOCK_SIZE=self.block_size
        )

    def ingest_batch(self, ticks: torch.Tensor) -> float:
        batch_size = ticks.shape[0]
        self._launch_kernel(ticks)
        
        n_to_copy = min(batch_size, self.window_size)
        indices = (torch.arange(n_to_copy, device=ticks.device) + self.head) % self.window_size
        self.cores.index_copy_(0, indices, self.output_buffer[:n_to_copy])
        
        self.head = (self.head + n_to_copy) % self.window_size
        self.count = min(self.count + n_to_copy, self.window_size)
        self.total_ticks += batch_size
        
        entropy = self._compute_entropy()
        self.entropy_history.append(entropy)
        return entropy

    def _compute_entropy(self) -> float:
        if self.count < 10:
            return 0.0
        mid_idx = (self.head - self.count // 2) % self.window_size
        core = self.cores[mid_idx]
        frob_sq = torch.sum(core * core).item()
        max_sq = torch.max(torch.abs(core)).item() ** 2
        if max_sq < 1e-10:
            return 0.0
        return float(torch.log2(torch.tensor(frob_sq / (max_sq + 1e-10) + 1.0)))

    def get_regime(self) -> Tuple[str, float]:
        if len(self.entropy_history) < 5:
            return ("UNKNOWN", 0.0)
        mean_ent = sum(list(self.entropy_history)[-20:]) / min(20, len(self.entropy_history))
        if mean_ent < 3.0:
            return ("STABLE", 0.8)
        elif mean_ent < 5.0:
            return ("TRENDING", 0.7)
        elif mean_ent < 7.0:
            return ("VOLATILE", 0.6)
        elif mean_ent < 9.0:
            return ("CHAOTIC", 0.5)
        else:
            return ("HYPERCHAOS", 0.3)


# =============================================================================
# BYBIT FIRE HOSE
# =============================================================================

class BybitFireHose:
    """Ingests the ENTIRE Bybit perpetuals market."""
    
    REST_URL = "https://api.bybit.com/v5/market/instruments-info"
    WS_URL = "wss://stream.bybit.com/v5/public/linear"
    
    def __init__(self, batch_size: int = 256, max_assets: int = 512):
        self.batch_size = batch_size
        self.max_assets = max_assets
        self.running = False
        
        self.symbols: List[str] = []
        self.symbol_idx: Dict[str, int] = {}
        self.prices: Dict[str, float] = {}
        self.price_ranges: Dict[str, Tuple[float, float]] = {}
        self.trade_counts: Dict[str, int] = {}
        
        self.slicer = None  # Initialized after we know asset count
        self.tick_buffer = None
        self.buffer_idx = 0
        
        self.total_trades = 0
        self.start_time = time.time()
        self.last_display = time.time()
        self.last_batch_time = time.time()

    async def _fetch_all_symbols(self) -> List[str]:
        """Fetch all USDT perpetual symbols from Bybit."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.REST_URL,
                params={"category": "linear", "limit": "1000"}
            ) as resp:
                data = await resp.json()
                
        symbols = []
        for item in data.get("result", {}).get("list", []):
            symbol = item.get("symbol", "")
            # Only USDT perpetuals, exclude weird pairs
            if symbol.endswith("USDT") and item.get("status") == "Trading":
                symbols.append(symbol)
        
        # Sort by expected volume (majors first)
        priority = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", 
                   "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT", "MATICUSDT"]
        
        def sort_key(s):
            try:
                return priority.index(s)
            except ValueError:
                return 1000
        
        symbols.sort(key=sort_key)
        return symbols[:self.max_assets]  # Cap at max_assets

    def _normalize_price(self, symbol: str, price: float) -> float:
        if symbol not in self.price_ranges:
            self.price_ranges[symbol] = (price * 0.999, price * 1.001)
        
        min_p, max_p = self.price_ranges[symbol]
        if price < min_p:
            min_p = price * 0.999
        if price > max_p:
            max_p = price * 1.001
        self.price_ranges[symbol] = (min_p, max_p)
        
        return (price - min_p) / (max_p - min_p) if max_p > min_p else 0.5

    def _process_trade(self, symbol: str, price: float):
        if symbol not in self.symbol_idx:
            return
        
        idx = self.symbol_idx[symbol]
        self.prices[symbol] = price
        self.trade_counts[symbol] = self.trade_counts.get(symbol, 0) + 1
        self.total_trades += 1
        
        norm_price = self._normalize_price(symbol, price)
        self.tick_buffer[self.buffer_idx, idx] = norm_price
        self.buffer_idx += 1
        
        if self.buffer_idx >= self.batch_size:
            self._flush_batch()

    def _flush_batch(self):
        if self.buffer_idx == 0:
            return
        
        now = time.time()
        gpu_ticks = self.tick_buffer[:self.buffer_idx].cuda()
        self.slicer.ingest_batch(gpu_ticks)
        
        self.last_batch_time = now
        self.buffer_idx = 0
        # Reset tick buffer
        self.tick_buffer.zero_()

    def _display_status(self):
        now = time.time()
        if now - self.last_display < 0.5:
            return
        self.last_display = now
        
        elapsed = now - self.start_time
        regime, _ = self.slicer.get_regime()
        entropy = self.slicer.entropy_history[-1] if self.slicer.entropy_history else 0.0
        tps = self.total_trades / elapsed if elapsed > 0 else 0
        
        regime_colors = {
            "STABLE": "\033[92m", "TRENDING": "\033[93m",
            "VOLATILE": "\033[91m", "CHAOTIC": "\033[95m", 
            "HYPERCHAOS": "\033[1;91m", "UNKNOWN": "\033[90m"
        }
        color = regime_colors.get(regime, "\033[0m")
        reset = "\033[0m"
        
        # Top 5 by volume
        top5 = sorted(self.trade_counts.items(), key=lambda x: -x[1])[:5]
        top5_str = " ".join([f"{s.replace('USDT','')}:{self.prices.get(s,0):,.0f}" 
                            if self.prices.get(s,0) >= 10 
                            else f"{s.replace('USDT','')}:{self.prices.get(s,0):.4f}"
                            for s, _ in top5])
        
        # Active symbols count
        active = sum(1 for c in self.trade_counts.values() if c > 0)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {top5_str} | "
              f"{color}{regime:10s}{reset} H={entropy:.1f} | "
              f"{active:>3}/{len(self.symbols)} syms | "
              f"{self.total_trades:>8,} trades {tps:>7,.0f}/s", flush=True)

    async def run(self):
        self.running = True
        signal.signal(signal.SIGINT, lambda s, f: setattr(self, 'running', False))
        
        print()
        print("=" * 85)
        print("  BYBIT FIRE HOSE - ENTIRE MARKET INGESTION")
        print("=" * 85)
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Max Assets: {self.max_assets}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Kernel: Triton Zero-Loop Fire Hose")
        print("=" * 85)
        print()
        
        # Fetch all symbols
        print("  Fetching all perpetual symbols...", flush=True)
        self.symbols = await self._fetch_all_symbols()
        print(f"  Found {len(self.symbols)} trading pairs", flush=True)
        
        # Build symbol index
        for i, sym in enumerate(self.symbols):
            self.symbol_idx[sym] = i
            self.prices[sym] = 0.0
        
        # Initialize slicer with actual asset count
        actual_assets = len(self.symbols)
        print(f"  Initializing GPU tensors for {actual_assets} assets...", flush=True)
        
        self.slicer = FireHoseSlicer(
            batch_size=self.batch_size,
            window_size=8192,
            bond_dim=16,  # Smaller bond dim for more assets
            max_assets=actual_assets
        )
        self.tick_buffer = torch.zeros(self.batch_size, actual_assets, dtype=torch.float32)
        
        print(f"  GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB", flush=True)
        print()
        
        # Subscribe to all symbols in batches (Bybit has subscription limits)
        # Max 10 args per subscribe message, use multiple WebSocket connections
        
        MAX_SUBS_PER_WS = 200  # Bybit limit
        ws_groups = [self.symbols[i:i+MAX_SUBS_PER_WS] 
                     for i in range(0, len(self.symbols), MAX_SUBS_PER_WS)]
        
        print(f"  Opening {len(ws_groups)} WebSocket connections...", flush=True)
        
        # Create tasks for each WebSocket
        tasks = []
        for i, group in enumerate(ws_groups):
            task = asyncio.create_task(self._stream_group(i, group))
            tasks.append(task)
        
        self.start_time = time.time()
        
        # Run until interrupted
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        
        # Final stats
        self._flush_batch()
        elapsed = time.time() - self.start_time
        
        print()
        print()
        print("─" * 85)
        print(f"  Duration:          {elapsed:.1f}s")
        print(f"  Symbols:           {len(self.symbols)}")
        print(f"  Active Symbols:    {sum(1 for c in self.trade_counts.values() if c > 0)}")
        print(f"  Total Trades:      {self.total_trades:,}")
        print(f"  GPU Ticks:         {self.slicer.total_ticks:,}")
        if elapsed > 0:
            print(f"  Throughput:        {self.total_trades/elapsed:,.0f} trades/sec")
        print(f"  Final Regime:      {self.slicer.get_regime()[0]}")
        print()
        
        # Top 10 by volume
        top10 = sorted(self.trade_counts.items(), key=lambda x: -x[1])[:10]
        print("  Top 10 by Volume:")
        for sym, count in top10:
            price = self.prices.get(sym, 0)
            print(f"    {sym:12s}: {count:>8,} trades @ ${price:>12,.4f}")
        print("─" * 85)

    async def _stream_group(self, group_id: int, symbols: List[str]):
        """Stream trades for a group of symbols."""
        subscribe_msg = {
            "op": "subscribe",
            "args": [f"publicTrade.{sym}" for sym in symbols]
        }
        
        retry_count = 0
        max_retries = 5
        
        while self.running and retry_count < max_retries:
            try:
                async with websockets.connect(
                    self.WS_URL,
                    ping_interval=20,
                    ping_timeout=10,
                    max_size=50_000_000
                ) as ws:
                    await ws.send(json.dumps(subscribe_msg))
                    
                    # Wait for confirmation
                    msg = await asyncio.wait_for(ws.recv(), timeout=10)
                    data = json.loads(msg)
                    if data.get("success"):
                        if group_id == 0:
                            print(f"  ✓ Group {group_id}: {len(symbols)} symbols subscribed", flush=True)
                    
                    async for message in ws:
                        if not self.running:
                            break
                        
                        try:
                            data = json.loads(message)
                            
                            if "topic" in data and data.get("type") == "snapshot":
                                topic = data["topic"]
                                symbol = topic.replace("publicTrade.", "")
                                
                                for trade in data.get("data", []):
                                    price = float(trade.get("p", 0))
                                    if price > 0:
                                        self._process_trade(symbol, price)
                                
                                self._display_status()
                        except:
                            pass
                            
            except Exception as e:
                retry_count += 1
                if self.running:
                    await asyncio.sleep(1)


async def main():
    firehose = BybitFireHose(batch_size=256, max_assets=512)
    await firehose.run()


if __name__ == "__main__":
    asyncio.run(main())
