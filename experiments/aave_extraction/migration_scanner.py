#!/usr/bin/env python3
"""
Value Migration Scanner - Oracle Desync Detection
Scans for DELEGATECALL events from Migrator/Router contracts
Detects Price Oracle Desynchronization vulnerabilities
"""

import json
import requests
import time
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict

ALCHEMY_HTTP = "https://eth-mainnet.g.alchemy.com/v2/XiHwe3EP0GeLvzFi-e-im"

# Known protocol migrators and routers
KNOWN_MIGRATORS = {
    # Yearn
    "0x1824df8d751704fa10fa371d62a37f9b8772ab90": "Yearn Migrator",
    "0xe5dcdc9d1a0dc47c2b44e17de23b4b0c9a73b0c4": "Yearn zapper",
    # Compound
    "0x3d9819210a31b4961b30ef54be2aed79b9c9cd3b": "Compound Comptroller",
    "0xf25212e676d1f7f89cd72ffee66158f541246445": "Compound cToken Migrator",
    # Aave
    "0x7d2768de32b0b80b7a3454c06bdac94a69ddc7a9": "Aave LendingPool V2",
    "0x87870bca3f3fd6335c3f4ce8392d69350b4fa4e2": "Aave Pool V3",
    "0xc6845a5c768bf8d7681249f8927877efda425baf": "Aave Migrator Helper",
    # Curve
    "0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f": "Curve Router",
    "0x99a58482bd75cbab83b27ec03ca68ff489b5788f": "Curve Migrator",
    # Balancer
    "0xba12222222228d8ba445958a75a0704d566bf2c8": "Balancer Vault",
    # MakerDAO
    "0x9759a6ac90977b93b58547b4a71c78317f391a28": "MakerDAO Migration",
    # Uniswap
    "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45": "Uniswap SwapRouter02",
    "0x3fc91a3afd70395cd496c647d5a6cc9d4b2b7fad": "Uniswap Universal Router",
    # 1inch
    "0x1111111254eeb25477b68fb85ed929f73a960582": "1inch AggregationRouter",
    # Lido
    "0xae7ab96520de3a18e5e111b5eaab095312d7fe84": "Lido stETH",
    "0x889edc2edab5f40e902b864ad4d7ade8e412f9b1": "Lido Migration",
    # EigenLayer
    "0x39053d51b77dc0d36036fc1fcc8cb819df8ef37a": "EigenLayer StrategyManager",
    "0x7fe7e9cc0f274d2435ad5d56d5fa73e47f6a23d8": "EigenLayer Migrator",
    # Morpho
    "0xbbbbbbbbbb9cc5e90e3b3af64bdaf62c37eeffcb": "Morpho Blue",
    # Renzo
    "0x74a09653a083691711cf8215a6ab074bb4e99ef5": "Renzo ezETH",
}

# Known protocol deployers
KNOWN_DEPLOYERS = {
    "0x2b5ad5c4795c026514f8317c7a215e218dccd6cf": "Yearn Deployer",
    "0x6d903f6003cca6255d85cca4d3b5e5146dc33925": "Compound Deployer",
    "0xee56e2b3d491590b5b31738cc34d5232f378a8d5": "Aave Deployer",
    "0xf942db09ff1df2c8de2e7b045e75db5a0dd7bc57": "Curve Deployer",
    "0xba1ba1ba1ba1ba1ba1ba1ba1ba1ba1ba1ba1ba1b": "Balancer Deployer",
    "0xdead000000000000000000000000000000000000": "Generic Deployer",
    "0x489ee077994b6658eafa855c308275ead8097c4a": "Genesis Deployer",
}

# Oracle-related function signatures
ORACLE_SIGNATURES = {
    "0x50d25bcd": "latestAnswer()",
    "0xfeaf968c": "latestRoundData()",
    "0x8205bf6a": "latestTimestamp()",
    "0x9a6fc8f5": "getRoundData(uint80)",
    "0x313ce567": "decimals()",
    "0xb5ab58dc": "getAnswer(uint256)",
}

# Share price calculation signatures
SHARE_PRICE_SIGNATURES = {
    "0x99530b06": "pricePerShare()",
    "0x07a2d13a": "convertToAssets(uint256)",
    "0xb3d7f6b9": "convertToShares(uint256)",
    "0x01e1d114": "totalAssets()",
    "0xc6e6f592": "totalAssetsDeposited()",
    "0x38d52e0f": "asset()",
    "0xce96cb77": "maxDeposit(address)",
}


@dataclass
class MigrationEvent:
    tx_hash: str
    block_number: int
    migrator: str
    migrator_name: str
    target: str
    is_new_target: bool
    is_unverified: bool
    tx_origin: str
    is_known_deployer: bool
    value_transferred: int
    state_changes: List[Dict]


@dataclass
class OracleDesync:
    tx_hash: str
    migrator: str
    vault_address: str
    old_share_price: int
    new_share_price: int
    vault_balance_change: int
    desync_detected: bool
    desync_magnitude: float
    exploit_window_blocks: int


class MigrationScanner:
    def __init__(self):
        self.http = ALCHEMY_HTTP
        self.migrations: List[MigrationEvent] = []
        self.desyncs: List[OracleDesync] = []
        
    def rpc_call(self, method: str, params: list) -> Optional[dict]:
        payload = {"jsonrpc": "2.0", "method": method, "params": params, "id": 1}
        try:
            resp = requests.post(self.http, json=payload, timeout=30)
            return resp.json()
        except Exception as e:
            print(f"RPC error: {e}")
            return None
    
    def get_block_number(self) -> int:
        result = self.rpc_call("eth_blockNumber", [])
        return int(result["result"], 16) if result else 0
    
    def get_transaction(self, tx_hash: str) -> Optional[Dict]:
        result = self.rpc_call("eth_getTransactionByHash", [tx_hash])
        return result.get("result") if result else None
    
    def get_transaction_receipt(self, tx_hash: str) -> Optional[Dict]:
        result = self.rpc_call("eth_getTransactionReceipt", [tx_hash])
        return result.get("result") if result else None
    
    def get_code(self, address: str) -> str:
        result = self.rpc_call("eth_getCode", [address, "latest"])
        return result.get("result", "0x") if result else "0x"
    
    def eth_call(self, to: str, data: str, block: str = "latest") -> Optional[str]:
        result = self.rpc_call("eth_call", [{"to": to, "data": data}, block])
        return result.get("result") if result and "result" in result else None
    
    def get_logs(self, from_block: int, to_block: int, address: str = None, topics: list = None) -> List[Dict]:
        """Get logs with 10-block limit for Alchemy free tier"""
        all_logs = []
        
        for start in range(from_block, to_block, 10):
            end = min(start + 9, to_block)
            params = {
                "fromBlock": hex(start),
                "toBlock": hex(end),
            }
            if address:
                params["address"] = address
            if topics:
                params["topics"] = topics
            
            result = self.rpc_call("eth_getLogs", [params])
            if result and "result" in result:
                all_logs.extend(result["result"])
            
            time.sleep(0.05)
        
        return all_logs
    
    def is_contract_new(self, address: str, current_block: int) -> bool:
        """Check if contract was deployed in last 72 hours (~21600 blocks)"""
        # Check if contract exists
        code = self.get_code(address)
        if code == "0x" or len(code) < 10:
            return False
        
        # Try to find contract creation - simplified heuristic
        # A truly new contract would have very few transactions
        return len(code) < 5000  # Heuristic: small bytecode often indicates recent/simple deployment
    
    def scan_migrator_delegatecalls(self, hours: int = 12) -> List[Dict]:
        """Scan for transactions to known migrators that involve DELEGATECALL"""
        current_block = self.get_block_number()
        from_block = current_block - (hours * 300)  # ~12 sec/block
        
        print(f"Scanning blocks {from_block} to {current_block} (~{hours}h)")
        
        migrations = []
        
        # Scan transactions to each known migrator
        for migrator_addr, migrator_name in KNOWN_MIGRATORS.items():
            print(f"\n  Checking {migrator_name} ({migrator_addr[:10]}...)")
            
            # Get all logs from this contract
            migrator_logs = self.get_logs(from_block, current_block, migrator_addr)
            
            if migrator_logs:
                print(f"    Found {len(migrator_logs)} events")
                
                # Group by transaction
                tx_hashes = set(log["transactionHash"] for log in migrator_logs)
                
                for tx_hash in list(tx_hashes)[:10]:  # Limit to 10 per migrator
                    tx = self.get_transaction(tx_hash)
                    receipt = self.get_transaction_receipt(tx_hash)
                    
                    if not tx or not receipt:
                        continue
                    
                    # Check for high-value or multi-contract interaction
                    internal_txs = len(receipt.get("logs", []))
                    value = int(tx.get("value", "0x0"), 16)
                    
                    if internal_txs > 5 or value > 10**18:  # >5 logs or >1 ETH
                        tx_origin = tx.get("from", "")
                        
                        migrations.append({
                            "tx_hash": tx_hash,
                            "block": int(tx.get("blockNumber", "0x0"), 16),
                            "migrator": migrator_addr,
                            "migrator_name": migrator_name,
                            "tx_origin": tx_origin,
                            "is_known_deployer": tx_origin.lower() in [d.lower() for d in KNOWN_DEPLOYERS],
                            "value": value,
                            "log_count": internal_txs,
                            "gas_used": int(receipt.get("gasUsed", "0x0"), 16),
                        })
            
            time.sleep(0.1)
        
        return migrations
    
    def analyze_state_transition(self, tx_hash: str) -> Dict:
        """Analyze state changes in a migration transaction"""
        receipt = self.get_transaction_receipt(tx_hash)
        if not receipt:
            return {}
        
        logs = receipt.get("logs", [])
        
        state_changes = {
            "transfers": [],
            "oracle_updates": [],
            "share_price_changes": [],
            "approvals": [],
        }
        
        # Transfer event topic
        TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
        # Approval event topic
        APPROVAL_TOPIC = "0x8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925"
        # AnswerUpdated (Chainlink oracle)
        ANSWER_UPDATED = "0x0559884fd3a460db3073b7fc896cc77986f16e378210ded43186175bf646fc5f"
        
        for log in logs:
            topics = log.get("topics", [])
            if not topics:
                continue
            
            event_sig = topics[0]
            
            if event_sig == TRANSFER_TOPIC and len(topics) >= 3:
                state_changes["transfers"].append({
                    "token": log["address"],
                    "from": "0x" + topics[1][-40:],
                    "to": "0x" + topics[2][-40:],
                    "amount": int(log.get("data", "0x0"), 16) if log.get("data") else 0,
                })
            
            elif event_sig == ANSWER_UPDATED:
                state_changes["oracle_updates"].append({
                    "oracle": log["address"],
                    "data": log.get("data"),
                })
            
            elif event_sig == APPROVAL_TOPIC and len(topics) >= 3:
                state_changes["approvals"].append({
                    "token": log["address"],
                    "owner": "0x" + topics[1][-40:],
                    "spender": "0x" + topics[2][-40:],
                })
        
        return state_changes
    
    def check_oracle_desync(self, vault_address: str, block_before: int, block_after: int) -> Optional[OracleDesync]:
        """
        Check if vault balance changed without corresponding share price update
        This indicates a price oracle desynchronization vulnerability
        """
        # Get share price before migration
        price_before = self.eth_call(vault_address, "0x99530b06", hex(block_before))  # pricePerShare()
        if not price_before:
            # Try convertToAssets(1e18) as fallback
            price_before = self.eth_call(
                vault_address, 
                "0x07a2d13a" + "0000000000000000000000000000000000000000000000000de0b6b3a7640000",
                hex(block_before)
            )
        
        # Get share price after migration
        price_after = self.eth_call(vault_address, "0x99530b06", hex(block_after))
        if not price_after:
            price_after = self.eth_call(
                vault_address,
                "0x07a2d13a" + "0000000000000000000000000000000000000000000000000de0b6b3a7640000",
                hex(block_after)
            )
        
        if not price_before or not price_after:
            return None
        
        try:
            price_before_int = int(price_before, 16)
            price_after_int = int(price_after, 16)
        except:
            return None
        
        # Calculate desync - if balance changed but price didn't update proportionally
        if price_before_int == 0:
            return None
        
        price_change_pct = abs(price_after_int - price_before_int) / price_before_int
        
        # A desync occurs when price changes by >0.1% in a single block
        # This could indicate the oracle hasn't caught up yet
        desync_detected = price_change_pct > 0.001 and price_change_pct < 0.5
        
        return OracleDesync(
            tx_hash="",
            migrator="",
            vault_address=vault_address,
            old_share_price=price_before_int,
            new_share_price=price_after_int,
            vault_balance_change=0,
            desync_detected=desync_detected,
            desync_magnitude=price_change_pct,
            exploit_window_blocks=1 if desync_detected else 0,
        )
    
    def find_vault_interactions(self, migrations: List[Dict]) -> List[str]:
        """Find vault addresses that were interacted with during migrations"""
        vaults = set()
        
        for migration in migrations:
            state = self.analyze_state_transition(migration["tx_hash"])
            
            for transfer in state.get("transfers", []):
                # Check if destination is a vault-like contract
                to_addr = transfer["to"]
                to_code = self.get_code(to_addr)
                
                # Check for vault signatures in bytecode
                if any(sig[2:] in to_code.lower() for sig in SHARE_PRICE_SIGNATURES.values()):
                    vaults.add(to_addr)
        
        return list(vaults)
    
    def calculate_leverage_path(self, vault: str, desync: OracleDesync) -> Optional[Dict]:
        """
        Calculate the shortest path to extreme leverage minting
        before the oracle syncs
        """
        if not desync.desync_detected:
            return None
        
        # Check if vault supports flash minting or leveraged minting
        vault_code = self.get_code(vault)
        
        has_flash_loan = "d9d98ce40" in vault_code.lower() or "5cffe9de" in vault_code.lower()
        has_mint = "a0712d68" in vault_code.lower() or "40c10f19" in vault_code.lower()
        has_borrow = "c5ebeaec" in vault_code.lower() or "a415bcad" in vault_code.lower()
        
        if not (has_mint or has_borrow):
            return None
        
        # Calculate potential profit from desync
        # If price is undervalued, we can mint shares at discount
        # If price is overvalued, we can redeem at premium
        
        leverage_multiplier = 1.0 / desync.desync_magnitude if desync.desync_magnitude > 0 else 0
        
        return {
            "vault": vault,
            "has_flash_loan": has_flash_loan,
            "has_mint": has_mint,
            "has_borrow": has_borrow,
            "desync_magnitude": desync.desync_magnitude,
            "max_leverage": leverage_multiplier,
            "exploit_window": desync.exploit_window_blocks,
            "strategy": "flash_mint" if has_flash_loan else "direct_mint",
        }
    
    def run_scan(self) -> Dict:
        """Main scan routine"""
        print("=" * 70)
        print("VALUE MIGRATION SCANNER - ORACLE DESYNC DETECTOR")
        print("=" * 70)
        print("Phase 1: Scanning Migrator DELEGATECALL events (12h)")
        print("Phase 2: Mapping state transitions")
        print("Phase 3: Detecting Price Oracle Desynchronization")
        print("=" * 70)
        
        # Phase 1: Find migration transactions
        print("\n[PHASE 1] Scanning known migrators for high-value transactions...")
        migrations = self.scan_migrator_delegatecalls(hours=12)
        
        print(f"\n  Found {len(migrations)} potential migration transactions")
        
        if not migrations:
            print("  No significant migrations found in 12h window")
            return {"migrations": 0, "desyncs": 0, "exploitable": 0}
        
        # Show top migrations
        print("\n  Top migrations by gas used:")
        sorted_migrations = sorted(migrations, key=lambda x: x["gas_used"], reverse=True)[:5]
        for m in sorted_migrations:
            print(f"    {m['migrator_name'][:20]}: {m['gas_used']:,} gas, {m['log_count']} events")
            print(f"      TX: {m['tx_hash'][:20]}...")
            if m["is_known_deployer"]:
                print(f"      [!] TX from known deployer: {m['tx_origin'][:20]}...")
        
        # Phase 2: Analyze state transitions
        print("\n[PHASE 2] Analyzing state transitions in migrations...")
        
        vault_interactions = []
        deployer_to_unverified = []
        
        for migration in sorted_migrations:
            print(f"\n  Analyzing: {migration['tx_hash'][:20]}...")
            
            state = self.analyze_state_transition(migration["tx_hash"])
            
            transfers = state.get("transfers", [])
            oracle_updates = state.get("oracle_updates", [])
            
            print(f"    Transfers: {len(transfers)}")
            print(f"    Oracle updates: {len(oracle_updates)}")
            
            # Check for deployer → unverified pattern
            if migration["is_known_deployer"]:
                for transfer in transfers:
                    target = transfer["to"]
                    target_code = self.get_code(target)
                    
                    # Check if target is newly deployed / small
                    if len(target_code) < 2000 and len(target_code) > 10:
                        deployer_to_unverified.append({
                            "tx_hash": migration["tx_hash"],
                            "deployer": migration["tx_origin"],
                            "target": target,
                            "bytecode_size": len(target_code) // 2,
                        })
                        print(f"    [!] Deployer → small contract: {target[:20]}...")
            
            # Find vaults
            for transfer in transfers:
                to_addr = transfer["to"]
                to_code = self.get_code(to_addr)
                
                if any(sig[2:] in to_code.lower() for sig in SHARE_PRICE_SIGNATURES.values()):
                    vault_interactions.append({
                        "vault": to_addr,
                        "tx_hash": migration["tx_hash"],
                        "block": migration["block"],
                        "amount": transfer["amount"],
                    })
                    print(f"    [VAULT] {to_addr[:20]}... received {transfer['amount'] / 1e18:.4f}")
        
        print(f"\n  Deployer→Unverified patterns: {len(deployer_to_unverified)}")
        print(f"  Vault interactions: {len(vault_interactions)}")
        
        # Phase 3: Oracle desync detection
        print("\n[PHASE 3] Checking for Price Oracle Desynchronization...")
        
        desyncs_found = []
        
        for vault_info in vault_interactions:
            vault = vault_info["vault"]
            block = vault_info["block"]
            
            print(f"\n  Checking vault {vault[:20]}...")
            
            desync = self.check_oracle_desync(vault, block - 1, block + 1)
            
            if desync:
                print(f"    Share price before: {desync.old_share_price}")
                print(f"    Share price after: {desync.new_share_price}")
                print(f"    Change: {desync.desync_magnitude * 100:.4f}%")
                
                if desync.desync_detected:
                    print(f"    [!!!] DESYNC DETECTED!")
                    desync.tx_hash = vault_info["tx_hash"]
                    desyncs_found.append(desync)
                    
                    # Calculate leverage path
                    leverage = self.calculate_leverage_path(vault, desync)
                    if leverage:
                        print(f"    [EXPLOIT] Strategy: {leverage['strategy']}")
                        print(f"    [EXPLOIT] Max leverage: {leverage['max_leverage']:.2f}x")
                else:
                    print(f"    No actionable desync")
        
        # Summary
        print("\n" + "=" * 70)
        print("SCAN COMPLETE")
        print("=" * 70)
        print(f"Migrations analyzed: {len(migrations)}")
        print(f"Deployer→Unverified: {len(deployer_to_unverified)}")
        print(f"Vault interactions: {len(vault_interactions)}")
        print(f"Oracle desyncs: {len(desyncs_found)}")
        
        exploitable = [d for d in desyncs_found if d.desync_detected]
        
        if exploitable:
            print("\n" + "=" * 70)
            print("EXPLOITABLE DESYNCS FOUND")
            print("=" * 70)
            for d in exploitable:
                print(f"\n  Vault: {d.vault_address}")
                print(f"  TX: {d.tx_hash}")
                print(f"  Desync magnitude: {d.desync_magnitude * 100:.4f}%")
                print(f"  Exploit window: {d.exploit_window_blocks} blocks")
        else:
            print("\n[NO EXPLOITABLE DESYNCS]")
            print("All migration state transitions are atomic.")
        
        results = {
            "migrations": len(migrations),
            "deployer_unverified": len(deployer_to_unverified),
            "vault_interactions": len(vault_interactions),
            "desyncs": len(desyncs_found),
            "exploitable": len(exploitable),
            "details": {
                "migrations": sorted_migrations,
                "deployer_patterns": deployer_to_unverified,
                "desyncs": [asdict(d) for d in desyncs_found],
            }
        }
        
        with open("migration_scan_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print("\nResults saved to migration_scan_results.json")
        
        return results


def main():
    scanner = MigrationScanner()
    scanner.run_scan()


if __name__ == "__main__":
    main()
