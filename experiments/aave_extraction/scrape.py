from web3 import Web3
import json

# Your Alchemy Archive Node connection
RPC_URL = "https://eth-mainnet.g.alchemy.com/v2/XiHwe3EP0GeLvzFi-e-im"
w3 = Web3(Web3.HTTPProvider(RPC_URL))

if not w3.is_connected():
    print("Failed to connect to Alchemy. Check your URL.")
    exit()

# The specific Aave V1 and V2 Graveyard Contracts
targets = {
    "Aave_V1_PoolCore": "0x3dfd23A6c5E8BbcFc9581d2E864a68feb6a076d3",
    "Aave_V2_LendingPoolProxy": "0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9",
    "Aave_V2_PoolAddressesProviderRegistry": "0xbaA999AC55EAce41CcAE355c77809e68Bb345170"
}

state_dump = {}

for name, addr_hex in targets.items():
    print(f"Scraping {name}...")
    addr = w3.to_checksum_address(addr_hex)
    
    # Extract the raw EVM bytecode
    bytecode = w3.eth.get_code(addr).hex()
    
    state_dump[name] = {
        "address": addr,
        "bytecode": bytecode
    }

# Dump the bytecode to JSON for your QTT engine to ingest
with open("aave_graveyard_bytecode.json", "w") as f:
    json.dump(state_dump, f, indent=4)

print("Extraction complete. aave_graveyard_bytecode.json is ready.")
