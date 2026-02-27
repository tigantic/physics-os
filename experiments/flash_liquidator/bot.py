#!/usr/bin/env python3
"""
AAVE V3 Flash Loan Liquidation Bot
Zero capital required - uses flash loans for all liquidation capital
"""

import json
import time
import os
from web3 import Web3
from eth_account import Account
from decimal import Decimal

# Configuration
ALCHEMY_URL = os.environ.get("ALCHEMY_URL", "https://eth-mainnet.g.alchemy.com/v2/XiHwe3EP0GeLvzFi-e-im")
PRIVATE_KEY = os.environ.get("PRIVATE_KEY", "")  # Set this!
LIQUIDATOR_CONTRACT = os.environ.get("LIQUIDATOR_CONTRACT", "")  # After deployment

# AAVE V3 Mainnet addresses
AAVE_POOL = "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2"
AAVE_DATA_PROVIDER = "0x7B4EB56E7CD4b454BA8ff71E4518426369a138a3"
AAVE_ORACLE = "0x54586bE62E3c3580375aE3723C145253060Ca0C2"

# Common tokens
TOKENS = {
    "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
    "DAI": "0x6B175474E89094C44Da98b954EessdFd36F6067",
    "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
}

# ABIs (minimal)
POOL_ABI = json.loads('''[
    {"inputs":[{"internalType":"address","name":"user","type":"address"}],"name":"getUserAccountData","outputs":[{"internalType":"uint256","name":"totalCollateralBase","type":"uint256"},{"internalType":"uint256","name":"totalDebtBase","type":"uint256"},{"internalType":"uint256","name":"availableBorrowsBase","type":"uint256"},{"internalType":"uint256","name":"currentLiquidationThreshold","type":"uint256"},{"internalType":"uint256","name":"ltv","type":"uint256"},{"internalType":"uint256","name":"healthFactor","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"internalType":"address","name":"collateralAsset","type":"address"},{"internalType":"address","name":"debtAsset","type":"address"},{"internalType":"address","name":"user","type":"address"},{"internalType":"uint256","name":"debtToCover","type":"uint256"},{"internalType":"bool","name":"receiveAToken","type":"bool"}],"name":"liquidationCall","outputs":[],"stateMutability":"nonpayable","type":"function"}
]''')

DATA_PROVIDER_ABI = json.loads('''[
    {"inputs":[{"internalType":"address","name":"user","type":"address"}],"name":"getUserReservesData","outputs":[{"components":[{"internalType":"address","name":"underlyingAsset","type":"address"},{"internalType":"uint256","name":"scaledATokenBalance","type":"uint256"},{"internalType":"bool","name":"usageAsCollateralEnabledOnUser","type":"bool"},{"internalType":"uint256","name":"stableBorrowRate","type":"uint256"},{"internalType":"uint256","name":"scaledVariableDebt","type":"uint256"},{"internalType":"uint256","name":"principalStableDebt","type":"uint256"},{"internalType":"uint256","name":"stableBorrowLastUpdateTimestamp","type":"uint256"}],"internalType":"struct IPoolDataProvider.UserReserveData[]","name":"","type":"tuple[]"},{"internalType":"uint8","name":"","type":"uint8"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"getAllReservesTokens","outputs":[{"components":[{"internalType":"string","name":"symbol","type":"string"},{"internalType":"address","name":"tokenAddress","type":"address"}],"internalType":"struct IPoolDataProvider.TokenData[]","name":"","type":"tuple[]"}],"stateMutability":"view","type":"function"}
]''')

LIQUIDATOR_ABI = json.loads('''[
    {"inputs":[{"internalType":"address","name":"debtAsset","type":"address"},{"internalType":"uint256","name":"debtAmount","type":"uint256"},{"internalType":"address","name":"collateralAsset","type":"address"},{"internalType":"address","name":"userToLiquidate","type":"address"},{"internalType":"uint24","name":"swapPoolFee","type":"uint24"}],"name":"executeLiquidation","outputs":[],"stateMutability":"nonpayable","type":"function"}
]''')

# Events to monitor for new borrowers
BORROW_EVENT_TOPIC = Web3.keccak(text="Borrow(address,address,address,uint256,uint8,uint256,uint16)")


class AaveLiquidationBot:
    def __init__(self):
        self.w3 = Web3(Web3.HTTPProvider(ALCHEMY_URL))
        if not self.w3.is_connected():
            raise ConnectionError("Failed to connect to Ethereum node")
        
        print(f"Connected to Ethereum. Block: {self.w3.eth.block_number}")
        
        self.pool = self.w3.eth.contract(address=AAVE_POOL, abi=POOL_ABI)
        self.data_provider = self.w3.eth.contract(address=AAVE_DATA_PROVIDER, abi=DATA_PROVIDER_ABI)
        
        # Tracked positions (address -> last health factor)
        self.positions = {}
        
        # Known borrowers from events
        self.borrowers = set()
        
    def get_health_factor(self, user: str) -> float:
        """Get user's current health factor"""
        try:
            data = self.pool.functions.getUserAccountData(user).call()
            health_factor = data[5] / 1e18  # healthFactor is at index 5
            return health_factor
        except Exception as e:
            print(f"Error getting health factor for {user}: {e}")
            return float('inf')
    
    def get_user_positions(self, user: str):
        """Get user's collateral and debt positions"""
        try:
            reserves, _ = self.data_provider.functions.getUserReservesData(user).call()
            collaterals = []
            debts = []
            
            for reserve in reserves:
                asset = reserve[0]
                collateral = reserve[1]  # scaledATokenBalance
                is_collateral = reserve[2]
                variable_debt = reserve[4]  # scaledVariableDebt
                stable_debt = reserve[5]  # principalStableDebt
                
                if collateral > 0 and is_collateral:
                    collaterals.append((asset, collateral))
                if variable_debt > 0 or stable_debt > 0:
                    debts.append((asset, variable_debt + stable_debt))
            
            return collaterals, debts
        except Exception as e:
            print(f"Error getting positions for {user}: {e}")
            return [], []
    
    def find_liquidatable_positions(self, addresses: list) -> list:
        """Find positions with health factor < 1"""
        liquidatable = []
        
        for addr in addresses:
            hf = self.get_health_factor(addr)
            if hf < 1.0:
                collaterals, debts = self.get_user_positions(addr)
                if collaterals and debts:
                    liquidatable.append({
                        'user': addr,
                        'health_factor': hf,
                        'collaterals': collaterals,
                        'debts': debts
                    })
        
        return liquidatable
    
    def scan_recent_borrowers(self, blocks_back: int = 1000):
        """Scan recent borrow events to find active borrowers"""
        print(f"Scanning last {blocks_back} blocks for borrowers...")
        
        current_block = self.w3.eth.block_number
        start_block = current_block - blocks_back
        
        # Scan in chunks to avoid rate limits
        chunk_size = 100
        borrowers = set()
        
        for start in range(start_block, current_block, chunk_size):
            end = min(start + chunk_size - 1, current_block)
            
            try:
                logs = self.w3.eth.get_logs({
                    'address': AAVE_POOL,
                    'fromBlock': start,
                    'toBlock': end,
                    'topics': [BORROW_EVENT_TOPIC.hex()]
                })
                
                for log in logs:
                    # Borrower is in topic[2] (the onBehalfOf address)
                    if len(log['topics']) > 2:
                        borrower = '0x' + log['topics'][2].hex()[-40:]
                        borrowers.add(Web3.to_checksum_address(borrower))
                
                time.sleep(0.05)  # Rate limit
                
            except Exception as e:
                print(f"Error scanning blocks {start}-{end}: {e}")
                time.sleep(0.5)
        
        print(f"Found {len(borrowers)} unique borrowers")
        return list(borrowers)
    
    def estimate_profit(self, position: dict) -> float:
        """Estimate profit from liquidating a position"""
        # Simplified estimation - in production would calculate exact values
        # Liquidation bonus is typically 5-15%
        # Flash loan fee is 0.05%
        # Gas cost ~$10-50
        
        # For now, just check if debt is significant enough
        # Full implementation would calculate exact collateral value vs debt
        return 0.0  # TODO: Implement proper profit calculation
    
    def execute_liquidation(self, position: dict):
        """Execute a liquidation via the deployed contract"""
        if not PRIVATE_KEY or not LIQUIDATOR_CONTRACT:
            print("Missing PRIVATE_KEY or LIQUIDATOR_CONTRACT")
            return False
        
        account = Account.from_key(PRIVATE_KEY)
        liquidator = self.w3.eth.contract(
            address=LIQUIDATOR_CONTRACT,
            abi=LIQUIDATOR_ABI
        )
        
        # Get the largest collateral and debt
        collaterals = position['collaterals']
        debts = position['debts']
        
        if not collaterals or not debts:
            return False
        
        # Take the first collateral and debt for simplicity
        collateral_asset, _ = collaterals[0]
        debt_asset, debt_amount = debts[0]
        
        # Build the transaction
        try:
            tx = liquidator.functions.executeLiquidation(
                debt_asset,
                debt_amount,
                collateral_asset,
                position['user'],
                3000  # 0.3% Uniswap pool fee
            ).build_transaction({
                'from': account.address,
                'gas': 500000,
                'maxFeePerGas': self.w3.eth.gas_price * 2,
                'maxPriorityFeePerGas': self.w3.to_wei(2, 'gwei'),
                'nonce': self.w3.eth.get_transaction_count(account.address),
            })
            
            signed = account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)
            
            print(f"Liquidation TX sent: {tx_hash.hex()}")
            
            # Wait for receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            if receipt['status'] == 1:
                print(f"Liquidation successful! Gas used: {receipt['gasUsed']}")
                return True
            else:
                print("Liquidation failed (reverted)")
                return False
                
        except Exception as e:
            print(f"Error executing liquidation: {e}")
            return False
    
    def run_monitor(self, interval: int = 12):
        """Main monitoring loop"""
        print("=" * 60)
        print("AAVE V3 FLASH LOAN LIQUIDATION BOT")
        print("=" * 60)
        
        # Initial scan for borrowers
        self.borrowers = set(self.scan_recent_borrowers(blocks_back=5000))
        
        print(f"\nMonitoring {len(self.borrowers)} borrowers")
        print(f"Checking every {interval} seconds (1 block)")
        print("-" * 60)
        
        while True:
            try:
                current_block = self.w3.eth.block_number
                
                # Check all known borrowers
                liquidatable = self.find_liquidatable_positions(list(self.borrowers))
                
                if liquidatable:
                    print(f"\n[Block {current_block}] Found {len(liquidatable)} liquidatable positions!")
                    
                    for position in liquidatable:
                        print(f"  User: {position['user']}")
                        print(f"  Health Factor: {position['health_factor']:.4f}")
                        print(f"  Collaterals: {len(position['collaterals'])}")
                        print(f"  Debts: {len(position['debts'])}")
                        
                        # In production: estimate profit and execute if profitable
                        # self.execute_liquidation(position)
                        
                else:
                    print(f"[Block {current_block}] No liquidatable positions", end='\r')
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                print("\nStopping monitor...")
                break
            except Exception as e:
                print(f"Error in monitor loop: {e}")
                time.sleep(5)


def main():
    """Entry point"""
    bot = AaveLiquidationBot()
    
    # Quick health check
    print("\nTesting with known addresses...")
    
    # Example: check if any positions are close to liquidation
    sample_addresses = [
        "0x0000000000000000000000000000000000000001",  # Replace with real addresses
    ]
    
    # Start the monitor
    bot.run_monitor(interval=12)


if __name__ == "__main__":
    main()
