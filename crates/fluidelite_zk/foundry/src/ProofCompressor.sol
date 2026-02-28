// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title Proof Calldata Compressor — Gas-Optimized Proof Submission
/// @author Ontic Labs
/// @notice Provides compact encoding for BN254 proof elements to minimize
///         calldata gas costs (16 gas/non-zero byte, 4 gas/zero byte on L1).
/// @dev Techniques applied:
///   1. Point compression: G1 points use 33 bytes (x + sign bit) instead of 64
///   2. Packed header: domain + flags in single byte
///   3. Batch verification: multiple proofs in one transaction
contract ProofCompressor {

    // ═══════════════════════════════════════════════════════════════════════
    // CONSTANTS
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice BN254 base field modulus
    uint256 internal constant P = 21888242871839275222246405745257275088696311157297823662689037894645226208583;

    // ═══════════════════════════════════════════════════════════════════════
    // COMPRESSED PROOF FORMAT
    // ═══════════════════════════════════════════════════════════════════════
    //
    // Standard Groth16 proof (uncompressed): 8 × 32 = 256 bytes
    //   [A.x, A.y, B.x1, B.x0, B.y1, B.y0, C.x, C.y]
    //
    // Compressed format: 1 + 32 + 128 + 32 = 193 bytes (24.6% savings)
    //   [flags(1)] [A.x(32)] [B(128)] [C.x(32)]
    //   flags: bit 0 = A.y parity, bit 1 = C.y parity
    //   B is kept uncompressed (G2 point decompression is too expensive)
    //
    // For L2 (where calldata = execution gas), keep uncompressed.

    /// @notice Decompress a G1 point from x-coordinate + parity bit
    /// @param x The x-coordinate of the point
    /// @param yParity 0 for even y, 1 for odd y
    /// @return px The x-coordinate
    /// @return py The y-coordinate
    function decompressG1(uint256 x, uint8 yParity)
        public
        view
        returns (uint256 px, uint256 py)
    {
        // y² = x³ + 3 (BN254 curve equation: y² = x³ + b, where b = 3)
        uint256 x2 = mulmod(x, x, P);
        uint256 x3 = mulmod(x2, x, P);
        uint256 y2 = addmod(x3, 3, P);

        // Compute y = sqrt(y²) using Tonelli-Shanks via modexp precompile
        // For BN254: p ≡ 3 (mod 4), so sqrt(a) = a^((p+1)/4)
        uint256 exp = (P + 1) / 4;
        py = _modExp(y2, exp, P);

        // Verify the square root
        require(mulmod(py, py, P) == y2, "Invalid point: not on curve");

        // Apply parity
        if (py % 2 != yParity) {
            py = P - py;
        }

        px = x;
    }

    /// @notice Decode a compressed Groth16 proof
    /// @param compressed The compressed proof bytes (193 bytes for L1, 256 for L2)
    /// @return proof Standard 8-element proof array
    function decodeCompressedProof(bytes calldata compressed)
        external
        view
        returns (uint256[8] memory proof)
    {
        if (compressed.length == 256) {
            // Uncompressed format — just decode directly
            for (uint256 i = 0; i < 8; i++) {
                proof[i] = uint256(bytes32(compressed[i * 32:(i + 1) * 32]));
            }
            return proof;
        }

        require(compressed.length >= 193, "Proof too short");

        uint8 flags = uint8(compressed[0]);
        uint8 aYParity = flags & 1;
        uint8 cYParity = (flags >> 1) & 1;

        // A (compressed G1)
        uint256 ax = uint256(bytes32(compressed[1:33]));
        (proof[0], proof[1]) = decompressG1(ax, aYParity);

        // B (uncompressed G2 — 4 × 32 bytes)
        proof[2] = uint256(bytes32(compressed[33:65]));
        proof[3] = uint256(bytes32(compressed[65:97]));
        proof[4] = uint256(bytes32(compressed[97:129]));
        proof[5] = uint256(bytes32(compressed[129:161]));

        // C (compressed G1)
        uint256 cx = uint256(bytes32(compressed[161:193]));
        (proof[6], proof[7]) = decompressG1(cx, cYParity);
    }

    /// @notice Compress a standard Groth16 proof for L1 calldata savings
    /// @param proof Standard 8-element proof array
    /// @return compressed The compressed proof bytes (193 bytes)
    function compressProof(uint256[8] calldata proof)
        external
        pure
        returns (bytes memory compressed)
    {
        uint8 flags = 0;
        if (proof[1] % 2 == 1) flags |= 1;  // A.y parity
        if (proof[7] % 2 == 1) flags |= 2;  // C.y parity

        compressed = new bytes(193);
        compressed[0] = bytes1(flags);

        // A.x
        _writeUint256(compressed, 1, proof[0]);
        // B (full)
        _writeUint256(compressed, 33, proof[2]);
        _writeUint256(compressed, 65, proof[3]);
        _writeUint256(compressed, 97, proof[4]);
        _writeUint256(compressed, 129, proof[5]);
        // C.x
        _writeUint256(compressed, 161, proof[6]);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // BATCH VERIFICATION
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Verify multiple Groth16 proofs in a single transaction
    /// @dev Amortizes base transaction cost (21,000 gas) across N proofs.
    ///      Each proof is verified independently against its public input.
    /// @param verifier Address of the Groth16Verifier contract
    /// @param proofs Array of 8-element proof arrays
    /// @param publicInputs Array of public inputs (one per proof)
    /// @return results Array of verification results
    function batchVerify(
        address verifier,
        uint256[8][] calldata proofs,
        uint256[] calldata publicInputs
    ) external view returns (bool[] memory results) {
        require(proofs.length == publicInputs.length, "Length mismatch");
        require(proofs.length > 0, "Empty batch");

        results = new bool[](proofs.length);

        for (uint256 i = 0; i < proofs.length; i++) {
            (bool success, bytes memory data) = verifier.staticcall(
                abi.encodeWithSignature(
                    "verifyProof(uint256[8],uint256)",
                    proofs[i],
                    publicInputs[i]
                )
            );
            results[i] = success && abi.decode(data, (bool));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // INTERNAL HELPERS
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Modular exponentiation via precompile 0x05
    function _modExp(uint256 base, uint256 exp, uint256 mod)
        internal
        view
        returns (uint256 result)
    {
        bytes memory input = abi.encodePacked(
            uint256(32), uint256(32), uint256(32),
            base, exp, mod
        );

        bytes memory output = new bytes(32);
        bool success;
        assembly {
            success := staticcall(
                gas(),
                0x05,           // modexp precompile
                add(input, 32),
                mload(input),
                add(output, 32),
                32
            )
        }
        require(success, "modexp failed");
        result = uint256(bytes32(output));
    }

    /// @notice Write a uint256 into a bytes array at a given offset
    function _writeUint256(bytes memory buf, uint256 offset, uint256 value) internal pure {
        assembly {
            mstore(add(add(buf, 32), offset), value)
        }
    }
}
