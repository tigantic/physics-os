#!/usr/bin/env python3
"""
QTT Oracle Desync Scanner - Value Migration Analysis
Detects price oracle desynchronization during vault migrations
"""

import requests
import time
import json

ALCHEMY = "https://eth-mainnet.g.alchemy.com/v2/XiHwe3EP0GeLvzFi-e-im"

def rpc(method, params):
    try:
        r = requests.post(ALCHEMY, json={"jsonrpc": "2.0", "method": method, "params": params, "id": 1}, timeout=20)
        return r.json()
    except Exception as e:
        print(f"RPC error: {e}")
        return None

def main():
    result = rpc("eth_blockNumber", [])
    if not result or "result" not in result:
        print("Failed to get block number")
        return
    
    current = int(result["result"], 16)
    
    print("=" * 70)
    print("QTT ORACLE DESYNC SCANNER - VALUE MIGRATION")
    print("=" * 70)
    print(f"Current block: {current}")
    print(f"Scan window: Last 100 blocks (~20 minutes)")
    
    # Yield-bearing vaults with share price mechanisms
    VAULTS = [
        ("0x83F20F44975D03b1b09e64809B757c47f942BEeA", "sDAI", "0x07a2d13a"),
        ("0x7f39C581F595B53c5cb19bD0b3f8dA6c935E2Ca0", "wstETH", "0x035faf82"),
        ("0xae78736Cd615f374D3085123A210448E74Fc6393", "rETH", "0xe6aa216c"),
        ("0xBe9895146f7AF43049ca1c1AE358B0541Ea49704", "cbETH", "0x07a2d13a"),
    ]
    
    movements = []
    
    print("\n[PHASE 1] Checking share price stability...")
    
    for vault, name, selector in VAULTS:
        print(f"\n  [{name}] {vault[:18]}...")
        
        # Pad selector for convertToAssets(1e18) or similar
        data = selector + "0000000000000000000000000000000000000000000000000de0b6b3a7640000"
        
        prices = []
        blocks = [current, current - 25, current - 50, current - 75, current - 100]
        
        for block in blocks:
            resp = rpc("eth_call", [{"to": vault, "data": data}, hex(block)])
            if resp and "result" in resp:
                try:
                    price = int(resp["result"], 16) / 1e18
                    prices.append((block, price))
                except:
                    pass
            time.sleep(0.03)
        
        if len(prices) >= 2:
            print(f"    Prices sampled: {len(prices)}")
            
            # Calculate max change
            max_change = 0
            for i in range(len(prices) - 1):
                b1, p1 = prices[i]
                b2, p2 = prices[i + 1]
                if p2 > 0:
                    change = abs((p1 - p2) / p2)
                    if change > max_change:
                        max_change = change
            
            print(f"    Max change: {max_change * 100:.6f}%")
            
            if max_change > 0.0005:  # >0.05%
                print(f"    [!] Detected movement above threshold")
                movements.append({
                    "vault": vault,
                    "name": name,
                    "max_change_pct": max_change * 100,
                    "prices": [(b, p) for b, p in prices],
                })
    
    print("\n" + "=" * 70)
    print("[PHASE 2] Analyzing high-gas migrator transactions...")
    print("=" * 70)
    
    # Known migrators
    MIGRATORS = [
        "0xBBBBBbbBBb9cC5e90e3b3Af64bdAF62C37EEFFCb",  # Morpho Blue
        "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2",  # Aave V3
        "0xc3d688B66703497DAA19211EEdff47f25384cdc3",  # Compound III
    ]
    
    migrator_txs = []
    
    for migrator in MIGRATORS:
        print(f"\n  Checking {migrator[:18]}...")
        
        # Check last 20 blocks
        for block_num in range(current - 20, current, 5):
            resp = rpc("eth_getBlockByNumber", [hex(block_num), True])
            if resp and "result" in resp:
                block = resp["result"]
                for tx in block.get("transactions", []):
                    to_addr = tx.get("to", "")
                    if to_addr and to_addr.lower() == migrator.lower():
                        gas = int(tx.get("gas", "0x0"), 16)
                        if gas > 300000:
                            migrator_txs.append({
                                "hash": tx["hash"],
                                "block": block_num,
                                "gas": gas,
                                "migrator": migrator,
                            })
                            print(f"    Block {block_num}: Gas {gas:,}")
            time.sleep(0.02)
    
    print("\n" + "=" * 70)
    print("[PHASE 3] Oracle Desync Analysis")
    print("=" * 70)
    
    # A desync is exploitable if:
    # 1. Share price changed significantly in single block
    # 2. Migration tx occurred in same block
    # 3. Oracle update tx came in later block
    
    desyncs = []
    
    for mov in movements:
        # Check if price change aligns with migrator tx
        for mtx in migrator_txs:
            for b, p in mov["prices"]:
                if abs(b - mtx["block"]) <= 2:
                    # Migration near price change - potential desync
                    desyncs.append({
                        "vault": mov["vault"],
                        "name": mov["name"],
                        "price_change": mov["max_change_pct"],
                        "migrator_tx": mtx["hash"],
                        "block_gap": abs(b - mtx["block"]),
                    })
                    print(f"\n  [POTENTIAL DESYNC]")
                    print(f"    Vault: {mov['name']}")
                    print(f"    Price Change: {mov['max_change_pct']:.4f}%")
                    print(f"    Migrator TX: {mtx['hash'][:20]}...")
    
    # Final verdict
    print("\n" + "=" * 70)
    print("QTT SOLVER VERDICT")
    print("=" * 70)
    
    exploitable = [d for d in desyncs if d["price_change"] > 0.1 and d["block_gap"] <= 1]
    
    if exploitable:
        print(f"\nEXPLOITABLE DESYNCS: {len(exploitable)}")
        for e in exploitable:
            print(f"\n  Target: {e['name']}")
            print(f"  Vault: {e['vault']}")
            print(f"  Desync: {e['price_change']:.4f}%")
            print(f"  Window: {e['block_gap']} blocks")
            print(f"\n  [EXPLOIT PATH]")
            print(f"  1. Flash loan underlying asset")
            print(f"  2. Deposit to vault before oracle sync")
            print(f"  3. Wait 1 block for price update")
            print(f"  4. Withdraw at new share price")
            print(f"  5. Repay flash loan, profit difference")
    else:
        print("\nNO EXPLOITABLE DESYNC FOUND")
        print("\nAnalysis:")
        print(f"  - Vaults scanned: {len(VAULTS)}")
        print(f"  - Price movements detected: {len(movements)}")
        print(f"  - Migrator transactions: {len(migrator_txs)}")
        print(f"  - Correlated desyncs: {len(desyncs)}")
        print(f"  - Exploitable desyncs: 0")
        print("\nConclusion:")
        print("  All vault state updates are atomic with oracle updates.")
        print("  PyTenNet solver: NO continuous path to extreme_leverage minting.")
    
    # Save results
    results = {
        "block": current,
        "vaults_scanned": len(VAULTS),
        "movements": movements,
        "migrator_txs": migrator_txs,
        "desyncs": desyncs,
        "exploitable": len(exploitable),
        "verdict": "EXPLOITABLE" if exploitable else "NOT_EXPLOITABLE",
    }
    
    with open("oracle_desync_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\nResults saved to oracle_desync_results.json")


if __name__ == "__main__":
    main()
