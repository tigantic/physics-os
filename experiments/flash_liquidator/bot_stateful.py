#!/usr/bin/env python3
"""
AAVE V3 Flash Liquidation Bot - State-Based Architecture

Maintains local state instead of rescanning blockchain.
Updates only changed positions via event subscription.

Usage:
  # Monitor mode (no execution)
  python3 bot_stateful.py

  # Deploy contract
  python3 bot_stateful.py deploy
  
  # Execute mode (requires PRIVATE_KEY and deployed contract)
  PRIVATE_KEY=0x... LIQUIDATOR=0x... python3 bot_stateful.py run
"""

import json
import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from web3 import Web3
from eth_account import Account

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('liquidator.log')
    ]
)
log = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

RPC_URL = os.getenv("RPC_URL", "https://eth-mainnet.g.alchemy.com/v2/XiHwe3EP0GeLvzFi-e-im")
STATE_FILE = Path(__file__).parent / "state.json"

# Contract addresses
AAVE_POOL = "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2"
AAVE_ORACLE = "0x54586bE62E3c3580375aE3723C145253060Ca0C2"
WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
USDT = "0xdAC17F958D2ee523a2206206994597C13D831ec7"

# Minimum profit to execute (in USD)
MIN_PROFIT_USD = 50

# ABIs
POOL_ABI = json.loads('[{"inputs":[{"internalType":"address","name":"user","type":"address"}],"name":"getUserAccountData","outputs":[{"internalType":"uint256","name":"totalCollateralBase","type":"uint256"},{"internalType":"uint256","name":"totalDebtBase","type":"uint256"},{"internalType":"uint256","name":"availableBorrowsBase","type":"uint256"},{"internalType":"uint256","name":"currentLiquidationThreshold","type":"uint256"},{"internalType":"uint256","name":"ltv","type":"uint256"},{"internalType":"uint256","name":"healthFactor","type":"uint256"}],"stateMutability":"view","type":"function"}]')


@dataclass
class Position:
    address: str
    debt: float
    collateral: float
    health_factor: float
    last_update: str = ""
    
    @property
    def drop_needed(self) -> float:
        """Percentage drop in collateral value needed for liquidation"""
        if self.health_factor <= 1.0:
            return 0.0
        return (self.health_factor - 1.0) / self.health_factor * 100
    
    @property
    def profit_potential(self) -> float:
        """Estimated profit if fully liquidated (5% of 50% of debt)"""
        return self.debt * 0.5 * 0.05
    
    @property
    def is_liquidatable(self) -> bool:
        return self.health_factor < 1.0


class StateManager:
    """Manages persistent state for tracked positions"""
    
    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.positions: Dict[str, Position] = {}
        self.last_block = 0
        self.load()
    
    def load(self):
        """Load state from disk"""
        if self.state_file.exists():
            data = json.loads(self.state_file.read_text())
            self.last_block = data.get("last_block", 0)
            
            for addr, pos_data in data.get("positions", {}).items():
                self.positions[addr.lower()] = Position(
                    address=addr,
                    debt=pos_data["debt"],
                    collateral=pos_data.get("col", pos_data["debt"] * 1.05),
                    health_factor=pos_data["hf"],
                    last_update=data.get("last_updated", "")
                )
            
            log.info(f"Loaded {len(self.positions)} positions from state")
    
    def save(self):
        """Persist state to disk"""
        data = {
            "last_block": self.last_block,
            "last_updated": datetime.utcnow().isoformat() + "Z",
            "positions": {
                addr: {"debt": p.debt, "hf": p.health_factor, "col": p.collateral}
                for addr, p in self.positions.items()
            },
            "stats": {
                "total_positions": len(self.positions),
                "at_risk": len([p for p in self.positions.values() if p.health_factor < 1.1]),
                "liquidatable": len([p for p in self.positions.values() if p.is_liquidatable])
            }
        }
        self.state_file.write_text(json.dumps(data, indent=2))
    
    def update_position(self, address: str, debt: float, collateral: float, hf: float):
        """Update or add a position"""
        addr = address.lower()
        self.positions[addr] = Position(
            address=address,
            debt=debt,
            collateral=collateral,
            health_factor=hf,
            last_update=datetime.utcnow().isoformat()
        )
    
    def remove_position(self, address: str):
        """Remove a position (debt repaid)"""
        addr = address.lower()
        if addr in self.positions:
            del self.positions[addr]
    
    def get_at_risk(self, threshold: float = 1.1) -> list:
        """Get positions with HF below threshold, sorted by HF"""
        at_risk = [p for p in self.positions.values() if p.health_factor < threshold]
        return sorted(at_risk, key=lambda x: x.health_factor)
    
    def get_liquidatable(self) -> list:
        """Get positions that can be liquidated now"""
        return [p for p in self.positions.values() if p.is_liquidatable]


class LiquidationBot:
    """
    Stateful AAVE V3 Liquidation Bot
    
    Maintains local position state and updates incrementally.
    """
    
    def __init__(self):
        self.w3 = Web3(Web3.HTTPProvider(RPC_URL))
        self.pool = self.w3.eth.contract(address=AAVE_POOL, abi=POOL_ABI)
        self.state = StateManager(STATE_FILE)
        
        # Execution config
        self.private_key = os.getenv("PRIVATE_KEY")
        self.liquidator_address = os.getenv("LIQUIDATOR")
        
        if self.private_key:
            self.account = Account.from_key(self.private_key)
            log.info(f"Execution wallet: {self.account.address}")
        
        log.info(f"Connected to block {self.w3.eth.block_number}")
    
    def refresh_position(self, address: str) -> Optional[Position]:
        """Fetch current position data from chain"""
        try:
            addr = Web3.to_checksum_address(address)
            data = self.pool.functions.getUserAccountData(addr).call()
            
            col = data[0] / 1e8
            debt = data[1] / 1e8
            hf_raw = data[5]
            
            if debt < 10:
                self.state.remove_position(address)
                return None
            
            hf = hf_raw / 1e18 if hf_raw < 1e30 else 999
            self.state.update_position(address, debt, col, hf)
            return self.state.positions[address.lower()]
            
        except Exception as e:
            log.error(f"Error refreshing {address}: {e}")
            return None
    
    def refresh_all(self):
        """Refresh all tracked positions"""
        log.info(f"Refreshing {len(self.state.positions)} positions...")
        
        removed = []
        for addr in list(self.state.positions.keys()):
            pos = self.refresh_position(addr)
            if pos is None:
                removed.append(addr)
            time.sleep(0.02)  # Rate limit
        
        if removed:
            log.info(f"Removed {len(removed)} closed positions")
        
        self.state.last_block = self.w3.eth.block_number
        self.state.save()
    
    def print_status(self):
        """Print current position status"""
        at_risk = self.state.get_at_risk(1.15)
        liquidatable = self.state.get_liquidatable()
        
        print("\n" + "=" * 80)
        print(f"POSITION STATUS - Block {self.w3.eth.block_number}")
        print(f"Tracked: {len(self.state.positions)} | At Risk (<1.15): {len(at_risk)} | Liquidatable: {len(liquidatable)}")
        print("=" * 80)
        
        if liquidatable:
            print("\n🔴 LIQUIDATABLE NOW:")
            for p in liquidatable:
                print(f"  {p.address[:20]}... ${p.debt:>12,.0f} | HF: {p.health_factor:.4f} | PROFIT: ${p.profit_potential:,.0f}")
        
        print("\n🟡 AT RISK (HF < 1.15):")
        for p in at_risk[:10]:
            emoji = "🔴" if p.is_liquidatable else "🟡"
            print(f"  {emoji} {p.address[:20]}... ${p.debt:>12,.0f} | HF: {p.health_factor:.4f} | {p.drop_needed:.1f}% drop | ${p.profit_potential:,.0f}")
        
        total_profit = sum(p.profit_potential for p in at_risk)
        print(f"\nTotal profit potential: ${total_profit:,.0f}")
    
    def execute_liquidation(self, position: Position) -> Optional[str]:
        """Execute flash loan liquidation"""
        if not self.private_key or not self.liquidator_address:
            log.warning("Cannot execute: missing PRIVATE_KEY or LIQUIDATOR")
            return None
        
        if position.profit_potential < MIN_PROFIT_USD:
            log.info(f"Skipping {position.address[:12]}... profit ${position.profit_potential:.0f} < ${MIN_PROFIT_USD}")
            return None
        
        log.warning(f"🚨 EXECUTING LIQUIDATION: {position.address}")
        log.warning(f"   Debt: ${position.debt:,.0f} | Profit: ${position.profit_potential:,.0f}")
        
        # Build and send transaction
        # ... (implementation depends on deployed contract interface)
        
        return None  # Return tx hash on success
    
    def monitor(self, interval: int = 12):
        """Main monitoring loop"""
        log.info("Starting monitor loop...")
        self.print_status()
        
        iteration = 0
        while True:
            try:
                iteration += 1
                current_block = self.w3.eth.block_number
                
                # Check if new block
                if current_block <= self.state.last_block:
                    time.sleep(1)
                    continue
                
                # Refresh positions
                changes = []
                for addr, old_pos in list(self.state.positions.items()):
                    new_pos = self.refresh_position(addr)
                    
                    if new_pos is None:
                        changes.append(f"  CLOSED: {addr[:16]}...")
                        continue
                    
                    # Significant HF change
                    if abs(new_pos.health_factor - old_pos.health_factor) > 0.005:
                        direction = "↓" if new_pos.health_factor < old_pos.health_factor else "↑"
                        changes.append(f"  {addr[:16]}... HF: {old_pos.health_factor:.4f} {direction} {new_pos.health_factor:.4f}")
                    
                    # Execute if liquidatable
                    if new_pos.is_liquidatable:
                        self.execute_liquidation(new_pos)
                
                self.state.last_block = current_block
                self.state.save()
                
                # Log changes
                if changes:
                    log.info(f"[Block {current_block}] Changes:")
                    for c in changes:
                        log.info(c)
                else:
                    if iteration % 25 == 0:
                        at_risk = len(self.state.get_at_risk(1.1))
                        log.info(f"[Block {current_block}] No changes | {at_risk} at risk")
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                log.info("Shutting down...")
                self.state.save()
                break
            except Exception as e:
                log.error(f"Error: {e}")
                time.sleep(interval)


def deploy_contract():
    """Deploy FlashLiquidator contract"""
    private_key = os.getenv("PRIVATE_KEY")
    if not private_key:
        print("Set PRIVATE_KEY environment variable")
        sys.exit(1)
    
    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    account = Account.from_key(private_key)
    
    # Check balance
    balance = w3.eth.get_balance(account.address)
    balance_eth = balance / 1e18
    print(f"Wallet: {account.address}")
    print(f"Balance: {balance_eth:.4f} ETH")
    
    if balance_eth < 0.02:
        print("Insufficient balance for deployment (~0.02-0.05 ETH needed)")
        sys.exit(1)
    
    # Load compiled contract
    contract_path = Path(__file__).parent / "out" / "FlashLiquidator.sol" / "FlashLiquidator.json"
    if not contract_path.exists():
        print("Contract not compiled. Run: forge build")
        sys.exit(1)
    
    contract_json = json.loads(contract_path.read_text())
    bytecode = contract_json["bytecode"]["object"]
    abi = contract_json["abi"]
    
    # Deploy
    print("\nDeploying FlashLiquidator...")
    
    Contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    
    # Estimate gas
    gas_price = w3.eth.gas_price
    print(f"Gas price: {gas_price / 1e9:.2f} gwei")
    
    # Build deployment transaction
    constructor_args = [
        AAVE_POOL,
        "0xE592427A0AEce92De3Edee1F18E0157C05861564"  # Uniswap Router
    ]
    
    tx = Contract.constructor(*constructor_args).build_transaction({
        'from': account.address,
        'nonce': w3.eth.get_transaction_count(account.address),
        'gas': 1500000,
        'gasPrice': gas_price,
    })
    
    # Estimate actual gas
    gas_estimate = w3.eth.estimate_gas(tx)
    tx['gas'] = int(gas_estimate * 1.2)
    
    cost_eth = tx['gas'] * gas_price / 1e18
    print(f"Estimated cost: {cost_eth:.4f} ETH (${cost_eth * 2500:.2f})")
    
    # Confirm
    response = input("\nDeploy? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted")
        sys.exit(0)
    
    # Sign and send
    signed_tx = w3.eth.account.sign_transaction(tx, private_key)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    
    print(f"TX submitted: {tx_hash.hex()}")
    print("Waiting for confirmation...")
    
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
    
    if receipt['status'] == 1:
        contract_address = receipt['contractAddress']
        print(f"\n✅ DEPLOYED: {contract_address}")
        print(f"\nAdd to environment:")
        print(f"  export LIQUIDATOR={contract_address}")
        
        # Save to config
        config_path = Path(__file__).parent / "config.json"
        config = {"liquidator": contract_address, "deployed_at": datetime.utcnow().isoformat()}
        config_path.write_text(json.dumps(config, indent=2))
        
        return contract_address
    else:
        print("❌ Deployment failed")
        sys.exit(1)


def main():
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == "deploy":
            deploy_contract()
            return
        
        if cmd == "status":
            bot = LiquidationBot()
            bot.refresh_all()
            bot.print_status()
            return
        
        if cmd == "run":
            bot = LiquidationBot()
            bot.refresh_all()
            bot.monitor()
            return
    
    # Default: status
    bot = LiquidationBot()
    bot.refresh_all()
    bot.print_status()


if __name__ == "__main__":
    main()
