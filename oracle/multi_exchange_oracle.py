"""
Multi-Exchange Oracle Engine
============================
Tries Binance US → Binance Testnet → Coinbase in sequence.

Binance has higher volume = more ticks = better regime detection.
"""

import torch
import triton
import triton.language as tl
import asyncio
import json
import time
import signal
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from datetime import datetime

try:
    import websockets
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets", "-q"])
    import websockets

torch.set_default_device("cuda")


# =============================================================================
# TRITON KERNEL
# =============================================================================

@triton.jit
def batched_ingest_kernel(
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
    """Batched ingest kernel."""
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
# ZERO-LOOP SLICER
# =============================================================================

class ZeroLoopSlicer:
    """GPU-accelerated slicer."""
    
    def __init__(self, batch_size: int, window_size: int, bond_dim: int, assets: int = 4):
        self.batch_size = batch_size
        self.window_size = window_size
        self.bond_dim = bond_dim
        self.assets = assets
        
        self.cores = torch.zeros(window_size, bond_dim, assets, bond_dim, dtype=torch.float32)
        self.head = 0
        self.count = 0
        
        torch.manual_seed(42)
        self.encoder = torch.randn(assets, bond_dim, dtype=torch.float32) * 0.1
        self.template = torch.randn(bond_dim, assets, bond_dim, dtype=torch.float32) * 0.01
        self.output_buffer = torch.empty(batch_size, bond_dim, assets, bond_dim, dtype=torch.float32)
        
        total_elements = batch_size * bond_dim * assets * bond_dim
        self.block_size = 1024
        self.grid = (triton.cdiv(total_elements, self.block_size),)
        
        self.total_ticks = 0
        self.total_batches = 0
        self.entropy_history = deque(maxlen=100)
        
        self._warmup()

    def _warmup(self):
        warmup_ticks = torch.rand(self.batch_size, self.assets, dtype=torch.float32)
        for _ in range(10):
            self._launch_kernel(warmup_ticks)
        torch.cuda.synchronize()

    def _launch_kernel(self, ticks: torch.Tensor):
        batched_ingest_kernel[self.grid](
            ticks, self.encoder, self.template, self.output_buffer,
            ticks.stride(0), ticks.stride(1),
            self.encoder.stride(0), self.encoder.stride(1),
            self.template.stride(0), self.template.stride(1), self.template.stride(2),
            self.output_buffer.stride(0), self.output_buffer.stride(1),
            self.output_buffer.stride(2), self.output_buffer.stride(3),
            BATCH=self.batch_size,
            ASSETS=self.assets,
            BOND=self.bond_dim,
            BLOCK_SIZE=self.block_size
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
        self.total_batches += 1
        
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
        spread = frob_sq / (max_sq + 1e-10)
        return float(torch.log2(torch.tensor(spread + 1.0)))

    def get_regime(self) -> Tuple[str, float]:
        if len(self.entropy_history) < 5:
            return ("UNKNOWN", 0.0)
        mean_ent = sum(list(self.entropy_history)[-20:]) / min(20, len(self.entropy_history))
        if mean_ent < 2.0:
            return ("STABLE", 0.8)
        elif mean_ent < 4.0:
            return ("TRENDING", 0.7)
        elif mean_ent < 6.0:
            return ("VOLATILE", 0.6)
        else:
            return ("CHAOTIC", 0.5)


# =============================================================================
# EXCHANGE CONFIGS
# =============================================================================

EXCHANGES = [
    {
        "name": "Binance US",
        "ws_url": "wss://stream.binance.us:9443/ws",
        "symbols": ["btcusd", "ethusd", "solusd", "dogeusd"],
        "stream_format": lambda s: f"{s}@trade",
        "parse": lambda d: (d.get("s", "").lower(), float(d.get("p", 0))) if "s" in d else (None, 0),
    },
    {
        "name": "Binance Testnet",
        "ws_url": "wss://testnet.binance.vision/ws",
        "symbols": ["btcusdt", "ethusdt", "solusdt", "dogeusdt"],
        "stream_format": lambda s: f"{s}@trade",
        "parse": lambda d: (d.get("s", "").lower(), float(d.get("p", 0))) if "s" in d else (None, 0),
    },
    {
        "name": "Coinbase",
        "ws_url": "wss://ws-feed.exchange.coinbase.com",
        "symbols": ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD"],
        "subscribe": lambda symbols: {"type": "subscribe", "product_ids": symbols, "channels": ["matches"]},
        "parse": lambda d: (d.get("product_id"), float(d.get("price", 0))) if d.get("type") == "match" else (None, 0),
    },
]


# =============================================================================
# MULTI-EXCHANGE ORACLE
# =============================================================================

class MultiExchangeOracle:
    """Tries multiple exchanges until one works."""
    
    DISPLAY_NAMES = ["BTC", "ETH", "SOL", "DOGE"]
    
    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size
        self.running = False
        self.exchange_name = ""
        
        self.slicer = ZeroLoopSlicer(
            batch_size=batch_size,
            window_size=4096,
            bond_dim=32,
            assets=4
        )
        
        self.tick_buffer = torch.zeros(batch_size, 4, dtype=torch.float32)
        self.buffer_idx = 0
        
        self.prices: Dict[str, float] = {}
        self.price_ranges: Dict[str, Tuple[float, float]] = {}
        self.symbol_idx: Dict[str, int] = {}
        
        self.trade_count = 0
        self.start_time = time.time()
        self.last_display = time.time()

    def _normalize_price(self, symbol: str, price: float) -> float:
        if symbol not in self.price_ranges:
            self.price_ranges[symbol] = (price * 0.999, price * 1.001)
        
        min_p, max_p = self.price_ranges[symbol]
        if price < min_p:
            min_p = price * 0.999
        if price > max_p:
            max_p = price * 1.001
        self.price_ranges[symbol] = (min_p, max_p)
        
        if max_p > min_p:
            return (price - min_p) / (max_p - min_p)
        return 0.5

    def _process_trade(self, symbol: str, price: float):
        if symbol not in self.symbol_idx:
            return
        
        idx = self.symbol_idx[symbol]
        self.prices[symbol] = price
        self.trade_count += 1
        
        norm_price = self._normalize_price(symbol, price)
        self.tick_buffer[self.buffer_idx, idx] = norm_price
        self.buffer_idx += 1
        
        if self.buffer_idx >= self.batch_size:
            self._flush_batch()

    def _flush_batch(self):
        if self.buffer_idx == 0:
            return
        gpu_ticks = self.tick_buffer[:self.buffer_idx].cuda()
        self.slicer.ingest_batch(gpu_ticks)
        self.buffer_idx = 0

    def _display_status(self):
        now = time.time()
        if now - self.last_display < 0.3:
            return
        self.last_display = now
        
        elapsed = now - self.start_time
        regime, _ = self.slicer.get_regime()
        entropy = self.slicer.entropy_history[-1] if self.slicer.entropy_history else 0.0
        tps = self.trade_count / elapsed if elapsed > 0 else 0
        
        regime_colors = {
            "STABLE": "\033[92m", "TRENDING": "\033[93m",
            "VOLATILE": "\033[91m", "CHAOTIC": "\033[95m", "UNKNOWN": "\033[90m"
        }
        color = regime_colors.get(regime, "\033[0m")
        reset = "\033[0m"
        
        prices_parts = []
        for i, (sym, _) in enumerate(self.symbol_idx.items()):
            p = self.prices.get(sym, 0)
            name = self.DISPLAY_NAMES[i] if i < len(self.DISPLAY_NAMES) else sym[:4]
            if p >= 1000:
                prices_parts.append(f"{name}:${p:,.0f}")
            elif p >= 1:
                prices_parts.append(f"{name}:${p:.2f}")
            else:
                prices_parts.append(f"{name}:${p:.4f}")
        
        print(f"\r[{datetime.now().strftime('%H:%M:%S')}] {' '.join(prices_parts)} | "
              f"{color}{regime:8s}{reset} H={entropy:.1f} | "
              f"{self.trade_count:>6,} trades {tps:>5,.0f}/s", end="", flush=True)

    async def _try_binance(self, config: dict) -> bool:
        """Try connecting to a Binance-style exchange."""
        try:
            streams = "/".join([config["stream_format"](s) for s in config["symbols"]])
            url = f"{config['ws_url']}/{streams}"
            
            print(f"  Trying {config['name']}... ", end="", flush=True)
            
            async with websockets.connect(url, ping_interval=20, ping_timeout=10, close_timeout=3, open_timeout=5) as ws:
                # Setup symbol mapping
                for i, s in enumerate(config["symbols"]):
                    self.symbol_idx[s] = i
                    self.prices[s] = 0.0
                
                self.exchange_name = config["name"]
                print(f"  ✓ Connected to {config['name']}!")
                print()
                
                async for message in ws:
                    if not self.running:
                        break
                    try:
                        data = json.loads(message)
                        symbol, price = config["parse"](data)
                        if symbol and price > 0:
                            self._process_trade(symbol, price)
                            self._display_status()
                    except:
                        pass
                return True
                
        except Exception as e:
            print(f"  ✗ {config['name']}: {e}")
            return False

    async def _try_coinbase(self, config: dict) -> bool:
        """Try connecting to Coinbase."""
        try:
            print(f"  Trying {config['name']}... ", end="", flush=True)
            
            async with websockets.connect(config["ws_url"], ping_interval=30, ping_timeout=10, max_size=50_000_000, open_timeout=5) as ws:
                # Subscribe
                await ws.send(json.dumps(config["subscribe"](config["symbols"])))
                
                # Setup symbol mapping
                for i, s in enumerate(config["symbols"]):
                    self.symbol_idx[s] = i
                    self.prices[s] = 0.0
                
                self.exchange_name = config["name"]
                print(f"  ✓ Connected to {config['name']}!")
                print()
                
                async for message in ws:
                    if not self.running:
                        break
                    try:
                        data = json.loads(message)
                        symbol, price = config["parse"](data)
                        if symbol and price > 0:
                            self._process_trade(symbol, price)
                            self._display_status()
                    except:
                        pass
                return True
                
        except Exception as e:
            print(f"  ✗ {config['name']}: {e}")
            return False

    async def run(self):
        """Try exchanges in order until one works."""
        self.running = True
        signal.signal(signal.SIGINT, lambda s, f: setattr(self, 'running', False))
        
        print("=" * 70)
        print("  MULTI-EXCHANGE ORACLE ENGINE")
        print("=" * 70)
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Kernel: Triton Zero-Loop (0.1µs/tick)")
        print("=" * 70)
        print()
        
        for config in EXCHANGES:
            if not self.running:
                break
            
            try:
                if "subscribe" in config:
                    success = await asyncio.wait_for(self._try_coinbase(config), timeout=10)
                else:
                    success = await asyncio.wait_for(self._try_binance(config), timeout=10)
                
                if success:
                    break
            except asyncio.TimeoutError:
                print(f"  ✗ {config['name']}: Connection timeout")
        
        # Final stats
        self._flush_batch()
        elapsed = time.time() - self.start_time
        
        print()
        print()
        print("─" * 70)
        print(f"  Exchange:          {self.exchange_name}")
        print(f"  Duration:          {elapsed:.1f}s")
        print(f"  Total Trades:      {self.trade_count:,}")
        print(f"  GPU Ticks:         {self.slicer.total_ticks:,}")
        if elapsed > 0:
            print(f"  Throughput:        {self.trade_count/elapsed:,.0f} trades/sec")
        print(f"  Final Regime:      {self.slicer.get_regime()[0]}")
        print("─" * 70)


async def main():
    oracle = MultiExchangeOracle(batch_size=64)
    await oracle.run()


if __name__ == "__main__":
    asyncio.run(main())
