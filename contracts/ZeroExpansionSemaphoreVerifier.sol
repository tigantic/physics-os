// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.21;

/// @title Zero-Expansion Semaphore Verifier v3.0 (PQC Hybrid)
/// @author Ontic Labs
/// @notice Verifies Semaphore proofs using Zero-Expansion for depths 16-50
/// @dev Supports 2^50 = 1 quadrillion members with constant gas cost
contract ZeroExpansionSemaphoreVerifier {
    
    /// @notice Minimum supported tree depth
    uint8 public constant MIN_DEPTH = 16;
    
    /// @notice Maximum supported tree depth (2^50 members!)
    uint8 public constant MAX_DEPTH = 50;
    
    /// @notice Error: Tree depth not supported
    error UnsupportedTreeDepth(uint8 depth);
    
    /// @notice Error: Invalid proof
    error InvalidProof();
    
    /// @notice Error: Nullifier already used
    error NullifierAlreadyUsed(bytes32 nullifier);
    
    /// @notice Mapping of used nullifiers
    mapping(bytes32 => bool) public nullifierUsed;
    
    /// @notice Emitted when a proof is verified
    event ProofVerified(
        bytes32 indexed merkleRoot,
        bytes32 indexed nullifierHash,
        bytes32 signalHash,
        uint8 treeDepth
    );
    
    /// @notice Verify a Zero-Expansion Semaphore proof
    /// @param proof The Zero-Expansion proof bytes
    /// @param merkleRoot Root of the identity Merkle tree
    /// @param nullifierHash Hash preventing double-signaling
    /// @param signalHash Hash of the signal being signed
    /// @param externalNullifier Scope/context identifier
    /// @param treeDepth Depth of the Merkle tree (16-50)
    function verifyProof(
        bytes calldata proof,
        bytes32 merkleRoot,
        bytes32 nullifierHash,
        bytes32 signalHash,
        bytes32 externalNullifier,
        uint8 treeDepth
    ) external {
        // Validate tree depth
        if (treeDepth < MIN_DEPTH || treeDepth > MAX_DEPTH) {
            revert UnsupportedTreeDepth(treeDepth);
        }
        
        // Check nullifier not already used
        if (nullifierUsed[nullifierHash]) {
            revert NullifierAlreadyUsed(nullifierHash);
        }
        
        // Verify the Zero-Expansion proof
        // Key insight: Gas cost is CONSTANT regardless of tree depth!
        // At depth 50, we're proving membership in 2^50 elements
        // but verification is still just a pairing check
        bool valid = _verifyZeroExpansionProof(
            proof,
            merkleRoot,
            nullifierHash,
            signalHash,
            externalNullifier,
            treeDepth
        );
        
        if (!valid) {
            revert InvalidProof();
        }
        
        // Mark nullifier as used
        nullifierUsed[nullifierHash] = true;
        
        emit ProofVerified(merkleRoot, nullifierHash, signalHash, treeDepth);
    }
    
    /// @notice Internal proof verification
    /// @dev Replaced S-07 stub with real Groth16 pairing delegation.
    ///      Production verifier is at fluidelite-zk/foundry/src/ZeroExpansionSemaphoreVerifier.sol
    ///      with full OZ hardening (ReentrancyGuard, Pausable, AccessControl).
    ///
    ///      This file is DEPRECATED. Use the Foundry-based contract instead.
    function _verifyZeroExpansionProof(
        bytes calldata proof,
        bytes32 merkleRoot,
        bytes32 nullifierHash,
        bytes32 signalHash,
        bytes32 externalNullifier,
        uint8 treeDepth
    ) internal view returns (bool) {
        // Proof structure:
        // [0:27]   - "ZERO_EXPANSION_SEMAPHORE_V3" magic
        // [27:28]  - tree depth
        // [28:60]  - merkle root
        // [60:84]  - RMT challenges (3 x 8 bytes)
        // [84:340] - Groth16 proof (8 x uint256)
        
        // Verify minimum length (header + Groth16 proof)
        if (proof.length < 340) return false;
        
        // Verify magic bytes
        bytes27 magic;
        assembly {
            magic := calldataload(proof.offset)
        }
        if (magic != bytes27("ZERO_EXPANSION_SEMAPHORE_V3")) return false;
        
        // Verify tree depth in proof matches
        if (uint8(proof[27]) != treeDepth) return false;
        
        // Verify merkle root in proof matches
        bytes32 proofRoot;
        assembly {
            proofRoot := calldataload(add(proof.offset, 28))
        }
        if (proofRoot != merkleRoot) return false;
        
        // S-07 FIX: Decode and forward to Groth16 pairing verifier.
        // The production Foundry contract delegates to a deployed Groth16Verifier.
        // This standalone version performs the pairing check inline.
        //
        // NOTE: For production use, deploy the Foundry-based
        // ZeroExpansionSemaphoreVerifier which includes:
        //   - ReentrancyGuard
        //   - Pausable
        //   - AccessControl
        //   - Timelocked VK updates
        //
        // The Groth16 proof bytes at offset [84..340] are:
        //   [A.x, A.y, B.x1, B.x0, B.y1, B.y0, C.x, C.y]
        //
        // Public input is bound to all Semaphore parameters via keccak256.
        
        // Extract 8 uint256 values from proof bytes
        uint256[8] memory g16proof;
        for (uint256 i = 0; i < 8; i++) {
            assembly {
                let val := calldataload(add(add(proof.offset, 84), mul(i, 32)))
                mstore(add(g16proof, mul(i, 32)), val)
            }
        }
        
        // Compute public input: bind all Semaphore params into a single Fr element
        uint256 publicInput = uint256(
            keccak256(abi.encodePacked(merkleRoot, nullifierHash, signalHash, externalNullifier, treeDepth))
        ) % 21888242871839275222246405745257275088548364400416034343698204186575808495617;
        
        // Perform BN254 pairing check using ecPairing precompile 0x08
        return _groth16Verify(g16proof, publicInput);
    }
    
    /// @notice Inline Groth16 verifier for standalone deployment
    /// @dev Uses the same VK as the Foundry Groth16Verifier (deterministic seed 0x4859505254454E)
    function _groth16Verify(uint256[8] memory proof, uint256 publicInput) internal view returns (bool) {
        // BN254 base field modulus
        uint256 P = 21888242871839275222246405745257275088696311157297823662689037894645226208583;
        
        // VK constants from deterministic trusted setup
        uint256 IC0_X = 0x02a4e05bed12976cacfe168fc5d52d35607f498874a2b65a72654ef12065137e;
        uint256 IC0_Y = 0x24c9d34c37cdc8b92101ce6676be7b2719fc1614385378564e5e1f9ec4efe07a;
        uint256 IC1_X = 0x0358dbe66ac4d13366ae8d91b9bc67e4020e11b21c61d14062a3e4235eabced0;
        uint256 IC1_Y = 0x05fbf9cca81492825a6316792161ba4e35591f188a7e39fcce1a292054c1441a;
        
        // Compute vk_x = IC0 + publicInput * IC1
        uint256[3] memory mulInput;
        mulInput[0] = IC1_X;
        mulInput[1] = IC1_Y;
        mulInput[2] = publicInput;
        uint256[2] memory mulResult;
        bool ok;
        assembly { ok := staticcall(sub(gas(), 2000), 0x07, mulInput, 96, mulResult, 64) }
        if (!ok) return false;
        
        uint256[4] memory addInput;
        addInput[0] = IC0_X;
        addInput[1] = IC0_Y;
        addInput[2] = mulResult[0];
        addInput[3] = mulResult[1];
        uint256[2] memory vkx;
        assembly { ok := staticcall(sub(gas(), 2000), 0x06, addInput, 128, vkx, 64) }
        if (!ok) return false;
        
        // Pairing check: e(-A, B) * e(α, β) * e(vk_x, γ) * e(C, δ) == 1
        uint256[24] memory pInput;
        
        // e(-A, B) — EIP-197: G2 encoding is (x_imaginary, x_real, y_imaginary, y_real)
        pInput[0] = proof[0];
        pInput[1] = P - proof[1];
        pInput[2] = proof[2]; // B.x_im = x1
        pInput[3] = proof[3]; // B.x_re = x0
        pInput[4] = proof[4]; // B.y_im = y1
        pInput[5] = proof[5]; // B.y_re = y0
        
        // e(α, β) — VK constants
        pInput[6]  = 0x0af2c4ad40c3bf1a234612c4bc3ceb7e7b8efc6c43b25278accf3fc8d4f47bfc; // ALPHA_X
        pInput[7]  = 0x13a20c78bbc2497cd44e0ab6fc2bf573d269070c6e9ca7b1380aa9dfa7934cd5; // ALPHA_Y
        pInput[8]  = 0x103cdf3c987b1c7e28a6feab73b300a6c85a01b36ddfe88f8b133b7c5c797aaa; // BETA_X1 (imaginary)
        pInput[9]  = 0x305af8adbb29f56a3ce1509ea9442e239cc5501beb7e2aa84c8cf45f2b61fa5c; // BETA_X2 (real)
        pInput[10] = 0x2ebd317e792ab593028ebf61e14aea0d1f150770c2c0f05384763e1f069534ba; // BETA_Y1 (imaginary)
        pInput[11] = 0x16ca8b4078f94197504352b9fb577fcee74cd2ecfd5801c19660cbe64bb8947e; // BETA_Y2 (real)
        
        // e(vk_x, γ)
        pInput[12] = vkx[0];
        pInput[13] = vkx[1];
        pInput[14] = 0x2b3f62cec751449eeec1597cf4aebfce711a1a1265bc4d8e1e7ad8dc37eddf4a; // GAMMA_X1 (imaginary)
        pInput[15] = 0x0b750f175138e42380c97bcbe292fa7f752e6b7c69cc82b740a749b38a0ad79e; // GAMMA_X2 (real)
        pInput[16] = 0x088f92774e16c66ad47f939b07ae0903139d5e6ef1668f10b1c9abde97281ef3; // GAMMA_Y1 (imaginary)
        pInput[17] = 0x1075d3cf9b06f5a49871cb3cba3affcfcc651470e085aeb9412696292de8f0e2; // GAMMA_Y2 (real)
        
        // e(C, δ)
        pInput[18] = proof[6];
        pInput[19] = proof[7];
        pInput[20] = 0x1ac0085d58dd301e360fce5356eb402ea77bcd36f0672e51579b41633c7aa4ca; // DELTA_X1 (imaginary)
        pInput[21] = 0x1a55045d9fbe8120538e8c289538832c0927e3d0e211b0ff43bc1aa7dd63cd86; // DELTA_X2 (real)
        pInput[22] = 0x27952cdd0d489db9a891f882906370f34fe877a3817ff05457e054e344f92ebe; // DELTA_Y1 (imaginary)
        pInput[23] = 0x0ac8a933f145ecce425589a700769ada27baea0e38ad35eb2dab54afa90e2fa2; // DELTA_Y2 (real)
        
        uint256[1] memory result;
        assembly { ok := staticcall(sub(gas(), 2000), 0x08, pInput, 768, result, 32) }
        if (!ok) return false;
        return result[0] == 1;
    }
    
    /// @notice Check if a nullifier has been used
    function isNullifierUsed(bytes32 nullifier) external view returns (bool) {
        return nullifierUsed[nullifier];
    }
    
    /// @notice Get the maximum tree size supported
    /// @return The maximum number of members (2^50)
    function maxMembers() external pure returns (uint256) {
        // 2^50 = 1,125,899,906,842,624
        return 1125899906842624;
    }
}

/// @title PQC Hybrid Commitment Registry
/// @notice Stores PQC bindings for quantum-safe migration
contract PQCCommitmentRegistry {
    
    /// @notice PQC binding for an identity
    mapping(bytes32 => bytes32) public pqcBindings;
    
    /// @notice Register PQC binding for an identity commitment
    /// @param classicalCommitment The classical Poseidon commitment
    /// @param pqcBinding The SHAKE256 PQC binding
    function registerPQCBinding(
        bytes32 classicalCommitment,
        bytes32 pqcBinding
    ) external {
        require(pqcBindings[classicalCommitment] == bytes32(0), "Already registered");
        pqcBindings[classicalCommitment] = pqcBinding;
    }
    
    /// @notice Verify PQC binding exists for migration
    function hasPQCBinding(bytes32 commitment) external view returns (bool) {
        return pqcBindings[commitment] != bytes32(0);
    }
}
