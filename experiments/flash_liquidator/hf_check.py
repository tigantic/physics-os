#!/usr/bin/env python3
import json
from web3 import Web3

w3 = Web3(Web3.HTTPProvider('https://eth-mainnet.g.alchemy.com/v2/XiHwe3EP0GeLvzFi-e-im'))
POOL = '0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2'
POOL_ABI = json.loads('[{"inputs":[{"internalType":"address","name":"user","type":"address"}],"name":"getUserAccountData","outputs":[{"internalType":"uint256","name":"totalCollateralBase","type":"uint256"},{"internalType":"uint256","name":"totalDebtBase","type":"uint256"},{"internalType":"uint256","name":"availableBorrowsBase","type":"uint256"},{"internalType":"uint256","name":"currentLiquidationThreshold","type":"uint256"},{"internalType":"uint256","name":"ltv","type":"uint256"},{"internalType":"uint256","name":"healthFactor","type":"uint256"}],"stateMutability":"view","type":"function"}]')
pool = w3.eth.contract(address=POOL, abi=POOL_ABI)

# Known tight positions from scan
tight = [
    '0xc6dD9976066F3364b4D6A72cD4F1fA0468327Aa7',
    '0x97105cf9FE2a44299D1A4566a6E1BC9AEb05E3f7',
    '0x6e6abB59dc84D0914fC8Ca42C3CE31f3C2E0F3e2',
    '0x6EadDD09516cFd50bC6dDf62b39C6F4A8B2c1aDE',
    '0xdd0400a6BcEa1CE4e5db6f4C0be4c226193672F2',
    '0x172dd54776889669ae07d3e5F696e1fAb6a11989',
    '0xFd595C310De371a66E7701fb4e7b250c924C7907',
    '0x9600A48ed0f931d0c422D574e3275a90D8b22745',
    '0x058BbFF7bE9d3CB50cDb9E6F6929f023F2763B98',
    '0xB8a451107A9f87FDe481D4D686247D6e43Ed715e',
]

print('TIGHTEST POSITIONS - REAL TIME')
print('=' * 70)

results = []
for addr in tight:
    try:
        data = pool.functions.getUserAccountData(Web3.to_checksum_address(addr)).call()
        debt = data[1] / 1e8
        hf_raw = data[5]
        if debt > 1 and hf_raw < 1e30:
            hf = hf_raw / 1e18
            results.append((addr, debt, hf))
    except Exception as e:
        print(f'Error {addr[:12]}: {e}')

# Sort by HF
results.sort(key=lambda x: x[2])

for addr, debt, hf in results:
    drop = (hf - 1.0) / hf * 100 if hf > 1 else 0
    status = 'LIQ!' if hf < 1.0 else 'IMMIN' if hf < 1.02 else 'CLOSE' if hf < 1.05 else 'RISK'
    profit = debt * 0.5 * 0.05
    print(f'{status:>5} | HF:{hf:.6f} | {drop:>5.2f}% | ${debt:>14,.0f} | ${profit:>10,.0f}')
    print(f'      {addr}')
