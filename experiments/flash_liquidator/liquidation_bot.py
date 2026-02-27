#!/usr/bin/env python3
"""
AAVE V3 Flash Loan Liquidation Bot
Production-grade monitoring and execution system

Monitors at-risk positions and executes flash loan liquidations
when health factor drops below 1.0

Requirements:
- pip install web3 requests python-dotenv
- Deploy FlashLiquidator.sol contract
- Set environment variables:
  - ALCHEMY_URL: Your Alchemy endpoint
  - PRIVATE_KEY: Wallet private key (for signing)
  - LIQUIDATOR_CONTRACT: Deployed FlashLiquidator address
"""

import json
import time
import os
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# AAVE V3 Mainnet Addresses
AAVE_POOL = "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2"
AAVE_ORACLE = "0x54586bE62E3c3580375aE3723C145253060Ca0C2"
AAVE_DATA_PROVIDER = "0x7B4EB56E7CD4b454BA8ff71E4518426369a138a3"

# Uniswap V3
UNISWAP_ROUTER = "0xE592427A0AEce92De3Edee1F18E0157C05861564"
UNISWAP_QUOTER = "0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6"

# Common tokens
TOKENS = {
    "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
    "DAI": "0x6B175474E89094C44Da98b954EesddddddddbaF3",
    "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
    "wstETH": "0x7f39C581F595B53c5cb19bD0b3f8dA6c935E2Ca0",
    "cbETH": "0xBe9895146f7AF43049ca1c1AE358B0541Ea49704",
    "rETH": "0xae78736Cd615f374D3085123A210448E74Fc6393",
}

# ABIs
POOL_ABI = json.loads('''[
    {"inputs":[{"internalType":"address","name":"user","type":"address"}],"name":"getUserAccountData","outputs":[{"internalType":"uint256","name":"totalCollateralBase","type":"uint256"},{"internalType":"uint256","name":"totalDebtBase","type":"uint256"},{"internalType":"uint256","name":"availableBorrowsBase","type":"uint256"},{"internalType":"uint256","name":"currentLiquidationThreshold","type":"uint256"},{"internalType":"uint256","name":"ltv","type":"uint256"},{"internalType":"uint256","name":"healthFactor","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"internalType":"address","name":"collateralAsset","type":"address"},{"internalType":"address","name":"debtAsset","type":"address"},{"internalType":"address","name":"user","type":"address"},{"internalType":"uint256","name":"debtToCover","type":"uint256"},{"internalType":"bool","name":"receiveAToken","type":"bool"}],"name":"liquidationCall","outputs":[],"stateMutability":"nonpayable","type":"function"},
    {"inputs":[],"name":"getReservesList","outputs":[{"internalType":"address[]","name":"","type":"address[]"}],"stateMutability":"view","type":"function"}
]''')

DATA_PROVIDER_ABI = json.loads('''[
    {"inputs":[{"internalType":"address","name":"asset","type":"address"}],"name":"getReserveConfigurationData","outputs":[{"internalType":"uint256","name":"decimals","type":"uint256"},{"internalType":"uint256","name":"ltv","type":"uint256"},{"internalType":"uint256","name":"liquidationThreshold","type":"uint256"},{"internalType":"uint256","name":"liquidationBonus","type":"uint256"},{"internalType":"uint256","name":"reserveFactor","type":"uint256"},{"internalType":"bool","name":"usageAsCollateralEnabled","type":"bool"},{"internalType":"bool","name":"borrowingEnabled","type":"bool"},{"internalType":"bool","name":"stableBorrowRateEnabled","type":"bool"},{"internalType":"bool","name":"isActive","type":"bool"},{"internalType":"bool","name":"isFrozen","type":"bool"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"internalType":"address","name":"asset","type":"address"},{"internalType":"address","name":"user","type":"address"}],"name":"getUserReserveData","outputs":[{"internalType":"uint256","name":"currentATokenBalance","type":"uint256"},{"internalType":"uint256","name":"currentStableDebt","type":"uint256"},{"internalType":"uint256","name":"currentVariableDebt","type":"uint256"},{"internalType":"uint256","name":"principalStableDebt","type":"uint256"},{"internalType":"uint256","name":"scaledVariableDebt","type":"uint256"},{"internalType":"uint256","name":"stableBorrowRate","type":"uint256"},{"internalType":"uint256","name":"liquidityRate","type":"uint256"},{"internalType":"uint40","name":"stableRateLastUpdated","type":"uint40"},{"internalType":"bool","name":"usageAsCollateralEnabled","type":"bool"}],"stateMutability":"view","type":"function"}
]''')

ORACLE_ABI = json.loads('''[
    {"inputs":[{"internalType":"address","name":"asset","type":"address"}],"name":"getAssetPrice","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"}
]''')

LIQUIDATOR_ABI = json.loads('''[
    {"inputs":[{"internalType":"address","name":"collateralAsset","type":"address"},{"internalType":"address","name":"debtAsset","type":"address"},{"internalType":"address","name":"user","type":"address"},{"internalType":"uint256","name":"debtToCover","type":"uint256"},{"internalType":"uint24","name":"poolFee","type":"uint24"}],"name":"executeLiquidation","outputs":[],"stateMutability":"nonpayable","type":"function"},
    {"inputs":[],"name":"owner","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"}
]''')


@dataclass
class Position:
    """Represents a borrower's position on AAVE"""
    address: str
    collateral_usd: float
    debt_usd: float
    health_factor: float
    liquidation_threshold: float
    collateral_asset: Optional[str] = None
    debt_asset: Optional[str] = None
    collateral_amount: float = 0
    debt_amount: float = 0


@dataclass
class LiquidationOpportunity:
    """Represents a profitable liquidation opportunity"""
    position: Position
    debt_to_cover: int
    expected_collateral: int
    liquidation_bonus: float
    estimated_profit_usd: float
    gas_estimate: int
    net_profit_usd: float


class AAVELiquidationBot:
    """
    Production AAVE V3 Liquidation Bot
    
    Monitors positions and executes flash loan liquidations when profitable.
    """
    
    def __init__(
        self,
        rpc_url: str,
        private_key: Optional[str] = None,
        liquidator_address: Optional[str] = None,
        min_profit_usd: float = 50.0,
        gas_price_gwei: float = 30.0
    ):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        
        self.private_key = private_key
        self.liquidator_address = liquidator_address
        self.min_profit_usd = min_profit_usd
        self.gas_price_gwei = gas_price_gwei
        
        # Initialize contracts
        self.pool = self.w3.eth.contract(address=AAVE_POOL, abi=POOL_ABI)
        self.data_provider = self.w3.eth.contract(address=AAVE_DATA_PROVIDER, abi=DATA_PROVIDER_ABI)
        self.oracle = self.w3.eth.contract(address=AAVE_ORACLE, abi=ORACLE_ABI)
        
        if liquidator_address:
            self.liquidator = self.w3.eth.contract(
                address=Web3.to_checksum_address(liquidator_address),
                abi=LIQUIDATOR_ABI
            )
        
        # Track monitored positions
        self.watched_positions: Dict[str, Position] = {}
        self.liquidation_history: List[dict] = []
        
        # Token decimals cache
        self.decimals_cache: Dict[str, int] = {}
        
        logger.info(f"Bot initialized - Connected to block {self.w3.eth.block_number}")
    
    def get_account_data(self, user: str) -> Optional[Position]:
        """Get current position data for a user"""
        try:
            addr = Web3.to_checksum_address(user)
            data = self.pool.functions.getUserAccountData(addr).call()
            
            col = data[0] / 1e8  # USD with 8 decimals
            debt = data[1] / 1e8
            hf_raw = data[5]
            liq_threshold = data[3] / 100  # Percentage
            
            # Skip empty positions
            if debt < 10:
                return None
            
            # Health factor
            if hf_raw < 1e30:
                hf = hf_raw / 1e18
            else:
                return None  # Infinite HF = no debt
            
            return Position(
                address=addr,
                collateral_usd=col,
                debt_usd=debt,
                health_factor=hf,
                liquidation_threshold=liq_threshold
            )
            
        except Exception as e:
            logger.error(f"Error getting account data for {user}: {e}")
            return None
    
    def get_user_reserves(self, user: str) -> Tuple[Optional[str], Optional[str], float, float]:
        """
        Get user's collateral and debt assets
        Returns: (collateral_asset, debt_asset, collateral_amount, debt_amount)
        """
        try:
            addr = Web3.to_checksum_address(user)
            reserves = self.pool.functions.getReservesList().call()
            
            max_collateral = (None, 0)
            max_debt = (None, 0)
            
            for reserve in reserves:
                reserve_addr = Web3.to_checksum_address(reserve)
                user_data = self.data_provider.functions.getUserReserveData(reserve_addr, addr).call()
                
                atoken_balance = user_data[0]
                variable_debt = user_data[2]
                
                # Get price
                price = self.oracle.functions.getAssetPrice(reserve_addr).call() / 1e8
                
                # Track largest collateral
                if atoken_balance > 0:
                    value = atoken_balance * price
                    if value > max_collateral[1]:
                        max_collateral = (reserve_addr, atoken_balance)
                
                # Track largest debt
                if variable_debt > 0:
                    value = variable_debt * price
                    if value > max_debt[1]:
                        max_debt = (reserve_addr, variable_debt)
            
            return max_collateral[0], max_debt[0], max_collateral[1], max_debt[1]
            
        except Exception as e:
            logger.error(f"Error getting reserves for {user}: {e}")
            return None, None, 0, 0
    
    def calculate_liquidation_profit(self, position: Position) -> Optional[LiquidationOpportunity]:
        """
        Calculate potential profit from liquidating a position
        """
        if position.health_factor >= 1.0:
            return None
        
        try:
            # Get collateral and debt assets
            col_asset, debt_asset, col_amount, debt_amount = self.get_user_reserves(position.address)
            
            if not col_asset or not debt_asset:
                return None
            
            # Get liquidation bonus (typically 5-10%)
            config = self.data_provider.functions.getReserveConfigurationData(col_asset).call()
            liq_bonus = config[3] / 10000  # e.g., 10500 = 5% bonus
            
            # Max liquidation is 50% of debt or close factor
            close_factor = 0.5
            max_debt_to_cover = int(debt_amount * close_factor)
            
            # Expected collateral received (with bonus)
            col_price = self.oracle.functions.getAssetPrice(col_asset).call() / 1e8
            debt_price = self.oracle.functions.getAssetPrice(debt_asset).call() / 1e8
            
            debt_value_usd = max_debt_to_cover * debt_price / 1e18  # Adjust for decimals
            expected_col_value = debt_value_usd * liq_bonus
            
            # Gross profit from liquidation bonus
            gross_profit = expected_col_value - debt_value_usd
            
            # Estimate gas costs
            gas_estimate = 500000  # Conservative estimate
            gas_cost_eth = gas_estimate * self.gas_price_gwei * 1e-9
            eth_price = self.oracle.functions.getAssetPrice(TOKENS["WETH"]).call() / 1e8
            gas_cost_usd = gas_cost_eth * eth_price
            
            # Slippage buffer (1%)
            slippage = gross_profit * 0.01
            
            net_profit = gross_profit - gas_cost_usd - slippage
            
            return LiquidationOpportunity(
                position=position,
                debt_to_cover=max_debt_to_cover,
                expected_collateral=int(expected_col_value / col_price * 1e18),
                liquidation_bonus=liq_bonus,
                estimated_profit_usd=gross_profit,
                gas_estimate=gas_estimate,
                net_profit_usd=net_profit
            )
            
        except Exception as e:
            logger.error(f"Error calculating liquidation profit: {e}")
            return None
    
    def scan_for_borrowers(self, blocks_back: int = 1000) -> List[str]:
        """Scan recent blocks for addresses with active borrows"""
        borrowers = set()
        current = self.w3.eth.block_number
        
        BORROW_TOPIC = Web3.keccak(text='Borrow(address,address,address,uint256,uint8,uint256,uint16)').hex()
        
        logger.info(f"Scanning blocks {current - blocks_back} to {current}")
        
        chunk_size = 10
        for start in range(current - blocks_back, current, chunk_size):
            end = min(start + chunk_size - 1, current)
            try:
                logs = self.w3.eth.get_logs({
                    'address': AAVE_POOL,
                    'fromBlock': start,
                    'toBlock': end,
                    'topics': [BORROW_TOPIC]
                })
                for log in logs:
                    if len(log['topics']) > 2:
                        borrower = '0x' + log['topics'][2].hex()[-40:]
                        borrowers.add(borrower.lower())
            except Exception:
                pass
            time.sleep(0.02)
        
        return list(borrowers)
    
    def find_at_risk_positions(self, threshold: float = 1.1) -> List[Position]:
        """Find positions with health factor below threshold"""
        at_risk = []
        
        # Use pre-defined high-value targets if available
        targets = list(self.watched_positions.keys())
        
        # Also scan recent activity
        if len(targets) < 50:
            recent = self.scan_for_borrowers(blocks_back=500)
            targets.extend(recent)
        
        targets = list(set(targets))
        logger.info(f"Checking {len(targets)} addresses")
        
        for addr in targets:
            position = self.get_account_data(addr)
            if position and position.health_factor < threshold:
                at_risk.append(position)
                self.watched_positions[addr] = position
        
        # Sort by health factor
        at_risk.sort(key=lambda x: x.health_factor)
        return at_risk
    
    def execute_liquidation(
        self,
        opportunity: LiquidationOpportunity,
        use_flashbots: bool = False
    ) -> Optional[str]:
        """
        Execute flash loan liquidation
        Returns transaction hash if successful
        """
        if not self.private_key or not self.liquidator_address:
            logger.error("Private key or liquidator contract not configured")
            return None
        
        if opportunity.net_profit_usd < self.min_profit_usd:
            logger.info(f"Skipping - profit ${opportunity.net_profit_usd:.2f} below minimum ${self.min_profit_usd}")
            return None
        
        try:
            # Get user's assets
            col_asset, debt_asset, _, _ = self.get_user_reserves(opportunity.position.address)
            
            if not col_asset or not debt_asset:
                logger.error("Could not determine collateral/debt assets")
                return None
            
            # Build transaction
            account = self.w3.eth.account.from_key(self.private_key)
            nonce = self.w3.eth.get_transaction_count(account.address)
            
            # Use 3000 fee tier (0.3%) for most liquid pairs
            pool_fee = 3000
            
            tx = self.liquidator.functions.executeLiquidation(
                col_asset,
                debt_asset,
                opportunity.position.address,
                opportunity.debt_to_cover,
                pool_fee
            ).build_transaction({
                'from': account.address,
                'gas': opportunity.gas_estimate,
                'gasPrice': self.w3.to_wei(self.gas_price_gwei, 'gwei'),
                'nonce': nonce,
            })
            
            # Sign transaction
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)
            
            if use_flashbots:
                # Submit via Flashbots for MEV protection
                tx_hash = self._submit_flashbots(signed_tx)
            else:
                # Standard submission
                tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            
            logger.info(f"Liquidation TX submitted: {tx_hash.hex()}")
            
            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            if receipt['status'] == 1:
                logger.info(f"LIQUIDATION SUCCESS - TX: {tx_hash.hex()}")
                self.liquidation_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'target': opportunity.position.address,
                    'debt_covered': opportunity.debt_to_cover,
                    'estimated_profit': opportunity.estimated_profit_usd,
                    'tx_hash': tx_hash.hex(),
                    'status': 'success'
                })
                return tx_hash.hex()
            else:
                logger.error(f"Liquidation failed - TX reverted: {tx_hash.hex()}")
                return None
                
        except Exception as e:
            logger.error(f"Liquidation execution error: {e}")
            return None
    
    def _submit_flashbots(self, signed_tx) -> bytes:
        """Submit transaction via Flashbots relay"""
        flashbots_url = "https://relay.flashbots.net"
        
        bundle = [{
            "signed_transaction": signed_tx.raw_transaction.hex()
        }]
        
        target_block = self.w3.eth.block_number + 1
        
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_sendBundle",
            "params": [{
                "txs": [signed_tx.raw_transaction.hex()],
                "blockNumber": hex(target_block)
            }],
            "id": 1
        }
        
        response = requests.post(flashbots_url, json=payload)
        result = response.json()
        
        if 'error' in result:
            raise Exception(f"Flashbots error: {result['error']}")
        
        return signed_tx.hash
    
    def run_monitoring_loop(self, interval_seconds: int = 12):
        """
        Main monitoring loop - runs continuously
        """
        logger.info("=" * 60)
        logger.info("AAVE V3 LIQUIDATION BOT STARTED")
        logger.info(f"Minimum profit threshold: ${self.min_profit_usd}")
        logger.info(f"Gas price: {self.gas_price_gwei} gwei")
        logger.info("=" * 60)
        
        # Initial population of watched positions
        logger.info("Performing initial scan...")
        at_risk = self.find_at_risk_positions(threshold=1.5)
        logger.info(f"Found {len(at_risk)} positions with HF < 1.5")
        
        iteration = 0
        while True:
            try:
                iteration += 1
                current_block = self.w3.eth.block_number
                
                logger.info(f"\n[Block {current_block}] Iteration {iteration}")
                
                # Check watched positions
                liquidatable = []
                for addr, old_pos in list(self.watched_positions.items()):
                    new_pos = self.get_account_data(addr)
                    
                    if not new_pos:
                        del self.watched_positions[addr]
                        continue
                    
                    self.watched_positions[addr] = new_pos
                    
                    # Health factor changed significantly
                    if abs(new_pos.health_factor - old_pos.health_factor) > 0.01:
                        direction = "↓" if new_pos.health_factor < old_pos.health_factor else "↑"
                        logger.info(
                            f"  {addr[:12]}... HF: {old_pos.health_factor:.4f} {direction} {new_pos.health_factor:.4f}"
                        )
                    
                    if new_pos.health_factor < 1.0:
                        liquidatable.append(new_pos)
                
                # Execute liquidations
                for position in liquidatable:
                    logger.warning(f"🚨 LIQUIDATABLE: {position.address}")
                    logger.warning(f"   HF: {position.health_factor:.6f} | Debt: ${position.debt_usd:,.0f}")
                    
                    opportunity = self.calculate_liquidation_profit(position)
                    
                    if opportunity and opportunity.net_profit_usd > self.min_profit_usd:
                        logger.warning(f"   💰 Estimated profit: ${opportunity.net_profit_usd:,.2f}")
                        self.execute_liquidation(opportunity, use_flashbots=True)
                    else:
                        logger.info(f"   Profit below threshold, skipping")
                
                # Periodic re-scan for new positions (every 50 iterations)
                if iteration % 50 == 0:
                    logger.info("Performing periodic scan for new at-risk positions...")
                    at_risk = self.find_at_risk_positions(threshold=1.15)
                    logger.info(f"Now watching {len(self.watched_positions)} positions")
                
                time.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                logger.info("\nShutting down...")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval_seconds)
    
    def print_status(self):
        """Print current status of watched positions"""
        print("\n" + "=" * 80)
        print("CURRENT WATCHED POSITIONS")
        print("=" * 80)
        
        positions = sorted(
            self.watched_positions.values(),
            key=lambda x: x.health_factor
        )
        
        for pos in positions[:20]:
            status = "🔴 LIQ" if pos.health_factor < 1.0 else "🟡 RISK" if pos.health_factor < 1.1 else "🟢 WATCH"
            drop_needed = (pos.health_factor - 1.0) / pos.health_factor * 100 if pos.health_factor > 1 else 0
            max_profit = pos.debt_usd * 0.5 * 0.05
            
            print(f"{status} | {pos.address[:16]}...")
            print(f"     Debt: ${pos.debt_usd:>14,.0f} | HF: {pos.health_factor:.4f} | Drop: {drop_needed:.1f}%")
            print(f"     Max Profit: ${max_profit:>10,.0f}")


# =============================================================================
# HIGH-VALUE TARGETS (Pre-populated watch list)
# =============================================================================

HIGH_VALUE_TARGETS = [
    # $1.19B position - 3.9% from liquidation
    "0x9600A48ed0f931d0F168E2a9DED74D0568E90205",
    # $38M position - 3.3% from liquidation
    "0xFd595C310De371a6f5eB7E8686aa78a22BB20eC9",
    # $700K position - 7.8% from liquidation  
    "0xaB62b78c1715848Cf44D1Df0858D8286B1a7fEdA",
    # Additional targets from scan
    "0x172dd54776889669ae07d3e5F696e1fAb6a11989",
    "0x950556Fe435626604D98F6B58d07e67C5AEB4c5e",
    "0x44D322a634ca8f4E2ed0d25Ce79f56aD1D195c3b",
    "0xc916b2b96B5D633557de5C943E0DcBd0f2F7b3E4",
]


def main():
    """Main entry point"""
    # Load configuration from environment
    rpc_url = os.getenv("ALCHEMY_URL", "https://eth-mainnet.g.alchemy.com/v2/XiHwe3EP0GeLvzFi-e-im")
    private_key = os.getenv("PRIVATE_KEY")
    liquidator_address = os.getenv("LIQUIDATOR_CONTRACT")
    min_profit = float(os.getenv("MIN_PROFIT_USD", "50"))
    gas_price = float(os.getenv("GAS_PRICE_GWEI", "30"))
    
    # Initialize bot
    bot = AAVELiquidationBot(
        rpc_url=rpc_url,
        private_key=private_key,
        liquidator_address=liquidator_address,
        min_profit_usd=min_profit,
        gas_price_gwei=gas_price
    )
    
    # Pre-populate watch list with known high-value targets
    logger.info("Loading high-value targets...")
    for target in HIGH_VALUE_TARGETS:
        pos = bot.get_account_data(target)
        if pos:
            bot.watched_positions[target.lower()] = pos
            logger.info(f"  Added: {target[:16]}... | ${pos.debt_usd:,.0f} | HF: {pos.health_factor:.4f}")
    
    # Print initial status
    bot.print_status()
    
    # Check if we're in execution mode
    if private_key and liquidator_address:
        logger.info("\n✅ Execution mode enabled - will execute liquidations")
        bot.run_monitoring_loop(interval_seconds=12)
    else:
        logger.warning("\n⚠️  Monitor-only mode - no private key or contract configured")
        logger.warning("Set PRIVATE_KEY and LIQUIDATOR_CONTRACT to enable execution")
        
        # Run in monitor-only mode
        while True:
            try:
                at_risk = bot.find_at_risk_positions(threshold=1.1)
                
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Block {bot.w3.eth.block_number}")
                print(f"Positions at risk (HF < 1.1): {len(at_risk)}")
                
                for pos in at_risk[:5]:
                    status = "🔴 LIQUIDATABLE" if pos.health_factor < 1.0 else "🟡 AT RISK"
                    print(f"  {status} {pos.address[:16]}... ${pos.debt_usd:,.0f} HF:{pos.health_factor:.4f}")
                
                time.sleep(30)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                time.sleep(30)


if __name__ == "__main__":
    main()
