// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {ReentrancyGuard} from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import {Pausable} from "@openzeppelin/contracts/utils/Pausable.sol";
import {AccessControl} from "@openzeppelin/contracts/access/AccessControl.sol";

import {Groth16Verifier} from "./Groth16Verifier.sol";

/// @title Zero-Expansion Semaphore Verifier v3.0 (Hardened)
/// @author Ontic Labs
/// @notice Verifies Groth16 proofs via BN254 pairing delegation.
///         Supports tree depths 16-50 with constant gas cost.
/// @dev S-07 FIX: `_verifyZeroExpansionProof()` now performs real Groth16
///      pairing verification instead of returning `true` unconditionally.
///
///      Security hardening (Task 2.9):
///        - ReentrancyGuard: prevents cross-function reentrancy
///        - Pausable: circuit-breaker for emergency stops
///        - AccessControl: role-based admin for pause/unpause + VK updates
contract ZeroExpansionSemaphoreVerifier is ReentrancyGuard, Pausable, AccessControl {

    // ═══════════════════════════════════════════════════════════════════════
    // ROLES
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Role allowed to pause/unpause the contract
    bytes32 public constant PAUSER_ROLE = keccak256("PAUSER_ROLE");

    /// @notice Role allowed to upgrade the verifier address
    bytes32 public constant UPGRADER_ROLE = keccak256("UPGRADER_ROLE");

    // ═══════════════════════════════════════════════════════════════════════
    // STATE
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Minimum supported tree depth
    uint8 public constant MIN_DEPTH = 16;

    /// @notice Maximum supported tree depth (2^50 members)
    uint8 public constant MAX_DEPTH = 50;

    /// @notice The Groth16 verifier contract with embedded VK
    Groth16Verifier public groth16Verifier;

    /// @notice Mapping of used nullifiers (prevents double-signaling)
    mapping(bytes32 => bool) public nullifierUsed;

    /// @notice Timestamp of the last verifier update (for timelock accounting)
    uint256 public lastVerifierUpdate;

    /// @notice Minimum delay between verifier updates (48 hours)
    uint256 public constant VERIFIER_UPDATE_DELAY = 48 hours;

    // ═══════════════════════════════════════════════════════════════════════
    // ERRORS
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Tree depth not in [MIN_DEPTH, MAX_DEPTH]
    error UnsupportedTreeDepth(uint8 depth);

    /// @notice Groth16 pairing check returned false
    error InvalidProof();

    /// @notice Nullifier was already consumed
    error NullifierAlreadyUsed(bytes32 nullifier);

    /// @notice Proof bytes shorter than required header
    error ProofTooShort(uint256 length);

    /// @notice Depth byte inside proof does not match `treeDepth` param
    error DepthMismatch(uint8 proofDepth, uint8 expectedDepth);

    /// @notice Merkle root inside proof does not match `merkleRoot` param
    error RootMismatch();

    /// @notice Verifier address is zero
    error ZeroAddress();

    /// @notice Verifier update attempted before timelock expired
    error TimelockNotExpired(uint256 earliest, uint256 current);

    // ═══════════════════════════════════════════════════════════════════════
    // EVENTS
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Emitted when a proof is verified and nullifier consumed
    event ProofVerified(
        bytes32 indexed merkleRoot,
        bytes32 indexed nullifierHash,
        bytes32 signalHash,
        uint8 treeDepth
    );

    /// @notice Emitted when the Groth16 verifier is updated
    event VerifierUpdated(address indexed oldVerifier, address indexed newVerifier);

    // ═══════════════════════════════════════════════════════════════════════
    // CONSTRUCTOR
    // ═══════════════════════════════════════════════════════════════════════

    /// @param _verifier Address of the deployed Groth16Verifier
    /// @param admin     Address to receive DEFAULT_ADMIN_ROLE, PAUSER_ROLE, UPGRADER_ROLE
    constructor(address _verifier, address admin) {
        if (_verifier == address(0)) revert ZeroAddress();
        if (admin == address(0)) revert ZeroAddress();

        groth16Verifier = Groth16Verifier(_verifier);
        lastVerifierUpdate = block.timestamp;

        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        _grantRole(PAUSER_ROLE, admin);
        _grantRole(UPGRADER_ROLE, admin);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ADMIN FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Pause proof verification (circuit-breaker)
    function pause() external onlyRole(PAUSER_ROLE) {
        _pause();
    }

    /// @notice Unpause proof verification
    function unpause() external onlyRole(PAUSER_ROLE) {
        _unpause();
    }

    /// @notice Update the Groth16 verifier address (timelocked)
    /// @param newVerifier Address of the new Groth16Verifier
    function updateVerifier(address newVerifier) external onlyRole(UPGRADER_ROLE) {
        if (newVerifier == address(0)) revert ZeroAddress();

        uint256 earliest = lastVerifierUpdate + VERIFIER_UPDATE_DELAY;
        if (block.timestamp < earliest) {
            revert TimelockNotExpired(earliest, block.timestamp);
        }

        address oldVerifier = address(groth16Verifier);
        groth16Verifier = Groth16Verifier(newVerifier);
        lastVerifierUpdate = block.timestamp;

        emit VerifierUpdated(oldVerifier, newVerifier);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PROOF VERIFICATION
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Verify a Zero-Expansion Semaphore proof
    /// @param proof The proof bytes (header + Groth16 A/B/C encoding)
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
    ) external nonReentrant whenNotPaused {
        // ── Validate tree depth ──
        if (treeDepth < MIN_DEPTH || treeDepth > MAX_DEPTH) {
            revert UnsupportedTreeDepth(treeDepth);
        }

        // ── Check nullifier not already used ──
        if (nullifierUsed[nullifierHash]) {
            revert NullifierAlreadyUsed(nullifierHash);
        }

        // ── Verify the Zero-Expansion proof (S-07 FIX: real verification) ──
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

        // ── Mark nullifier as used ──
        nullifierUsed[nullifierHash] = true;

        emit ProofVerified(merkleRoot, nullifierHash, signalHash, treeDepth);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // INTERNAL: PROOF VERIFICATION (S-07 FIX)
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Internal proof verification — delegates to Groth16 pairing check
    /// @dev Proof wire format:
    ///      [0:27]    "ZERO_EXPANSION_SEMAPHORE_V3" magic bytes
    ///      [27:28]   tree depth (uint8)
    ///      [28:60]   merkle root (bytes32)
    ///      [60:84]   RMT challenges (3 × 8 bytes)
    ///      [84:340]  Groth16 proof: 8 × uint256 = [A.x, A.y, B.x1, B.x0, B.y1, B.y0, C.x, C.y]
    ///
    ///      Total minimum length: 340 bytes (27 + 1 + 32 + 24 + 256)
    ///
    /// @dev The Groth16 public input is derived as:
    ///      publicInput = uint256(keccak256(merkleRoot, nullifierHash, signalHash, externalNullifier, treeDepth)) % Q
    ///      This binds ALL Semaphore parameters into the proof circuit.
    function _verifyZeroExpansionProof(
        bytes calldata proof,
        bytes32 merkleRoot,
        bytes32 nullifierHash,
        bytes32 signalHash,
        bytes32 externalNullifier,
        uint8 treeDepth
    ) internal view returns (bool) {
        // Minimum proof length: header(84) + Groth16 proof(256)
        if (proof.length < 340) revert ProofTooShort(proof.length);

        // ── Verify magic bytes ──
        // "ZERO_EXPANSION_SEMAPHORE_V3" = 27 bytes
        bytes27 magic;
        assembly {
            magic := calldataload(proof.offset)
        }
        if (magic != bytes27("ZERO_EXPANSION_SEMAPHORE_V3")) {
            return false;
        }

        // ── Verify depth ──
        uint8 proofDepth = uint8(proof[27]);
        if (proofDepth != treeDepth) {
            revert DepthMismatch(proofDepth, treeDepth);
        }

        // ── Verify merkle root ──
        bytes32 proofRoot;
        assembly {
            proofRoot := calldataload(add(proof.offset, 28))
        }
        if (proofRoot != merkleRoot) {
            revert RootMismatch();
        }

        // ── Decode Groth16 proof from bytes [84..340] ──
        uint256[8] memory groth16Proof;
        for (uint256 i = 0; i < 8; i++) {
            assembly {
                let val := calldataload(add(add(proof.offset, 84), mul(i, 32)))
                mstore(add(groth16Proof, mul(i, 32)), val)
            }
        }

        // ── Compute public input binding ──
        // Hash all Semaphore parameters into a single field element.
        // This ensures the Groth16 proof is bound to the specific
        // merkle root, nullifier, signal, and tree depth.
        uint256 publicInput = uint256(
            keccak256(
                abi.encodePacked(
                    merkleRoot,
                    nullifierHash,
                    signalHash,
                    externalNullifier,
                    treeDepth
                )
            )
        ) % 21888242871839275222246405745257275088548364400416034343698204186575808495617;

        // ── Delegate to Groth16 verifier (BN254 pairing check) ──
        return groth16Verifier.verifyProof(groth16Proof, publicInput);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // VIEW FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Check if a nullifier has been used
    function isNullifierUsed(bytes32 nullifier) external view returns (bool) {
        return nullifierUsed[nullifier];
    }

    /// @notice Get the maximum tree size supported
    /// @return The maximum number of members (2^50)
    function maxMembers() external pure returns (uint256) {
        return 1125899906842624; // 2^50
    }
}
