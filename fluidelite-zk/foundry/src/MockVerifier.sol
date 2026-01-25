// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.21;

/// @title Mock Groth16 Verifier for Zero-Expansion Demo
/// @author HyperTensor Labs
/// @notice Validates proof format and field elements, always returns valid
/// @dev Used to demonstrate the integration works - production uses real pairing

contract MockGroth16Verifier {
    
    /// @notice BN254 scalar field modulus (Fr)
    uint256 internal constant FR_MODULUS = 21888242871839275222246405745257275088548364400416034343698204186575808495617;
    
    /// @notice BN254 base field modulus (Fq) 
    uint256 internal constant FQ_MODULUS = 21888242871839275222246405745257275088696311157297823662689037894645226208583;
    
    error InvalidProofElement(uint256 index, uint256 value);
    error InvalidPublicInput(uint256 index, uint256 value);
    
    event ProofVerified(
        uint256 indexed merkleRoot,
        uint256 indexed nullifierHash, 
        uint8 treeDepth
    );
    
    /// @notice Verify Groth16 proof format (mock - validates field elements)
    /// @param proof 8 uint256 values: [A.x, A.y, B.x1, B.x0, B.y1, B.y0, C.x, C.y]
    /// @param publicInputs 5 values: [merkleRoot, nullifierHash, signalHash, externalNullifier, treeDepth]
    /// @return success True if format is valid
    function verifyProof(
        uint256[8] calldata proof,
        uint256[5] calldata publicInputs
    ) external pure returns (bool success) {
        
        // Validate proof elements are valid Fq field elements (curve coordinates)
        for (uint256 i = 0; i < 8; i++) {
            if (proof[i] >= FQ_MODULUS) {
                return false;
            }
        }
        
        // Validate public inputs are valid Fr field elements (scalars)
        for (uint256 i = 0; i < 4; i++) {
            if (publicInputs[i] >= FR_MODULUS) {
                return false;
            }
        }
        
        // Tree depth must be reasonable (16-50 for Zero-Expansion)
        uint256 treeDepth = publicInputs[4];
        if (treeDepth < 16 || treeDepth > 50) {
            return false;
        }
        
        return true;
    }
    
    /// @notice Get verifier type for identification
    function verifierType() external pure returns (string memory) {
        return "ZERO_EXPANSION_MOCK_V1";
    }
}
