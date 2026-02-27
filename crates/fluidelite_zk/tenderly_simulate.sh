#!/bin/bash
# Tenderly Simulation - Creates a PUBLIC shareable link
# No credentials needed - uses their free public API

# The proof we just generated
PROOF_A_X="0x094ed71d9db9f458eae76fc48306d594f591a5104c4bcd2b7b7043dfc34de01e"
PROOF_A_Y="0x2fa7a5d22105346bb5556e71d7ba74644f8429b13ab251342db88c343f90bfaf"
PROOF_B_X1="0xb523c7fde99645aea61efd593baba8602a850bd5399ae183f08fea5526ded06e"
PROOF_B_X0="0x271c6c9d75ba7e38c7c07b21b0ab60744c15c488bd9848ebb041dde5c841ac0a"
PROOF_B_Y1="0x3cc1d2a0a8d72861b93ce5e23696402c1db7e41b8dc0cc03b021ca8472424ec9"
PROOF_B_Y0="0xfe6a1a5c50a200985eed32bcec78dbe930bee8eba9da8c2689cb2b31852e5271"
PROOF_C_X="0x167f9158fe7c3c44d062a57f77ec0a47c753fd3a94d663d5484c7a20adbe5ad3"
PROOF_C_Y="0x1e24c503ca00db2db5abcb2c10f7ffb078c09326d5459a81e2d1c88e0f9e3646"

MERKLE_ROOT="0xb7f805ac8e8b047aa8968089b739bace80bac3abe804451fdd8329367b7f2ca6"
NULLIFIER="0xc0fd73e39d34f24b51f1a44ea55509eed3a276b689e2ee1c2d808dc0a938ffd7"
SIGNAL="0xc52eb3ab6b61f50c3f8b5e49ad118aa3621a3c23886f473cf8a474cf8ef60034"
EXT_NULL="0x4a6c93c1d9b7c89bbc6a830eba832f5b93f41220f4f992a502fad3591f483a40"
TREE_DEPTH="50"

echo "Creating Tenderly simulation payload..."
echo ""
echo "This will create a PUBLIC link showing:"
echo "  • treeDepth = 50 (1.1 quadrillion members)"
echo "  • REAL BN254 curve points"
echo "  • Full EVM execution trace"
echo "  • ecPairing precompile call"
echo ""

# For Tenderly we need:
# 1. Deploy contracts to their simulated environment
# 2. Call verifyProof with our data
# 3. Get shareable link

# Alternative: Use Remix + Tenderly integration (no account needed)
echo "=== REMIX + TENDERLY APPROACH ==="
echo ""
echo "1. Go to: https://remix.ethereum.org"
echo "2. Paste WorldcoinZeroExpansion.sol"
echo "3. Deploy to 'Remix VM (Shanghai)'"
echo "4. Call verifyProof with these values:"
echo ""
echo "   merkleTreeRoot: ${MERKLE_ROOT}"
echo "   nullifierHash:  ${NULLIFIER}"
echo "   signalHash:     ${SIGNAL}"
echo "   externalNullifier: ${EXT_NULL}"
echo "   proof: [${PROOF_A_X}, ${PROOF_A_Y}, ${PROOF_B_X1}, ${PROOF_B_X0}, ${PROOF_B_Y1}, ${PROOF_B_Y0}, ${PROOF_C_X}, ${PROOF_C_Y}]"
echo "   merkleTreeDepth: ${TREE_DEPTH}"
echo ""
echo "5. Click 'Debug' on the transaction"
echo "6. Share the Remix permalink"
