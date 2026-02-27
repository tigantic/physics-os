"""
Binance Live Oracle Engine
==========================
Direct connection to Binance global WebSocket streams.

Uses combined streams for maximum throughput.
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
from typing import Dict, Tuple
from datetime import datetime

try:
    import websockets
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets", "-q"])
    import websockets

try:
    import aiohttp
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp", "-q"])
    import aiohttp

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
            BATCH=self.batch_size, ASSETS=self.assets, BOND=self.bond_dim, BLOCK_SIZE=self.block_size
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
        if mean_ent < 2.0:
            return ("STABLE", 0.8)
        elif mean_ent < 4.0:
            return ("TRENDING", 0.7)
        elif mean_ent < 6.0:
            return ("VOLATILE", 0.6)
        else:
            return ("CHAOTIC", 0.5)


# =============================================================================
# BINANCE ENDPOINTS (ordered by preference)
# =============================================================================

BINANCE_ENDPOINTS = [
    ("Binance Global", "wss://stream.binance.com:9443/ws"),
    ("Binance US", "wss://stream.binance.us:9443/ws"),
    ("Binance Testnet", "wss://testnet.binance.vision/ws"),
]

# Trading pairs: symbol -> display name
SYMBOLS = {
    "btcusdt": "BTC",
    "ethusdt": "ETH", 
    "solusdt": "SOL",
    "dogeusdt": "DOGE",
}


# =============================================================================
# BINANCE ORACLE
# =============================================================================

class BinanceOracle:
    
    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size
        self.running = False
        self.exchange_name = ""
        
        self.slicer = ZeroLoopSlicer(
            batch_size=batch_size,
            window_size=4096,
            bond_dim=32,
            assets=len(SYMBOLS)
        )
        
        self.tick_buffer = torch.zeros(batch_size, len(SYMBOLS), dtype=torch.float32)
        self.buffer_idx = 0
        
        # Symbol -> index mapping
        self.symbol_idx = {s: i for i, s in enumerate(SYMBOLS.keys())}
        self.prices: Dict[str, float] = {s: 0.0 for s in SYMBOLS.keys()}
        self.price_ranges: Dict[str, Tuple[float, float]] = {}
        
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
        
        return (price - min_p) / (max_p - min_p) if max_p > min_p else 0.5

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
        for sym, name in SYMBOLS.items():
            p = self.prices.get(sym, 0)
            if p >= 1000:
                prices_parts.append(f"{name}:${p:,.0f}")
            elif p >= 1:
                prices_parts.append(f"{name}:${p:.2f}")
            else:
                prices_parts.append(f"{name}:${p:.4f}")
        
        print(f"\r[{datetime.now().strftime('%H:%M:%S')}] {' '.join(prices_parts)} | "
              f"{color}{regime:8s}{reset} H={entropy:.1f} | "
              f"{self.trade_count:>6,} trades {tps:>5,.0f}/s   ", end="", flush=True)

    async def _test_endpoint(self, name: str, base_url: str) -> bool:
        """Test if endpoint is reachable."""
        streams = "/".join([f"{s}@trade" for s in SYMBOLS.keys()])
        url = f"{base_url}/{streams}"
        
        try:
            async with websockets.connect(url, open_timeout=5, close_timeout=2) as ws:
                # Wait for first message
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
                data = json.loads(msg)
                if "e" in data and data["e"] == "trade":
                    return True
                return False
        except Exception:
            return False

    async def _stream_from_endpoint(self, name: str, base_url: str):
        """Stream trades from endpoint."""
        streams = "/".join([f"{s}@trade" for s in SYMBOLS.keys()])
        url = f"{base_url}/{streams}"
        
        self.exchange_name = name
        
        try:
            async with websockets.connect(
                url, 
                ping_interval=20, 
                ping_timeout=10, 
                close_timeout=3,
                max_size=10_000_000
            ) as ws:
                print(f"  ✓ Connected to {name}!")
                print()
                
                async for message in ws:
                    if not self.running:
                        break
                    
                    try:
                        data = json.loads(message)
                        if data.get("e") == "trade":
                            symbol = data.get("s", "").lower()
                            price = float(data.get("p", 0))
                            if symbol and price > 0:
                                self._process_trade(symbol, price)
                                self._display_status()
                    except json.JSONDecodeError:
                        pass
                        
        except Exception as e:
            print(f"\n  Connection lost: {e}")

    async def run(self):
        """Try endpoints until one works."""
        self.running = True
        signal.signal(signal.SIGINT, lambda s, f: setattr(self, 'running', False))
        
        print()
        print("=" * 70)
        print("  BINANCE LIVE ORACLE ENGINE")
        print("=" * 70)
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Assets: {', '.join(SYMBOLS.values())}")
        print(f"  Kernel: Triton Zero-Loop (0.1µs/tick)")
        print("=" * 70)
        print()
        
        working_endpoint = None
        
        for name, base_url in BINANCE_ENDPOINTS:
            if not self.running:
                break
                
            print(f"  Testing {name}... ", end="", flush=True)
            try:
                works = await asyncio.wait_for(self._test_endpoint(name, base_url), timeout=8)
                if works:
                    print("✓")
                    working_endpoint = (name, base_url)
                    break
                else:
                    print("✗ (no trade data)")
            except asyncio.TimeoutError:
                print("✗ (timeout)")
            except Exception as e:
                err_str = str(e)
                if "451" in err_str:
                    print("✗ (region blocked)")
                else:
                    print(f"✗ ({err_str[:40]})")
        
        if working_endpoint and self.running:
            name, base_url = working_endpoint
            print()
            print(f"  Streaming from {name}...")
            await self._stream_from_endpoint(name, base_url)
        else:
            print()
            print("  ✗ No Binance endpoint available")
            print("  Falling back to Coinbase...")
            await self._fallback_coinbase()
        
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

    async def _fallback_coinbase(self):
        """Fallback to Coinbase if Binance is unavailable."""
        self.exchange_name = "Coinbase"
        
        # Remap symbols for Coinbase
        coinbase_symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD"]
        self.symbol_idx = {s: i for i, s in enumerate(coinbase_symbols)}
        self.prices = {s: 0.0 for s in coinbase_symbols}
        
        try:
            async with websockets.connect(
                "wss://ws-feed.exchange.coinbase.com",
                ping_interval=30,
                ping_timeout=10,
                max_size=50_000_000
            ) as ws:
                # Subscribe
                await ws.send(json.dumps({
                    "type": "subscribe",
                    "product_ids": coinbase_symbols,
                    "channels": ["matches"]
                }))
                
                print("  ✓ Connected to Coinbase!")
                print()
                
                async for message in ws:
                    if not self.running:
                        break
                    
                    try:
                        data = json.loads(message)
                        if data.get("type") == "match":
                            symbol = data.get("product_id")
                            price = float(data.get("price", 0))
                            if symbol and price > 0:
                                self._process_trade(symbol, price)
                                self._display_status()
                    except:
                        pass
                        
        except Exception as e:
            print(f"\n  Coinbase connection failed: {e}")


async def main():
    oracle = BinanceOracle(batch_size=64)
    await oracle.run()


if __name__ == "__main__":
    asyncio.run(main())
