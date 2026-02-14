// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Test, console} from "forge-std/Test.sol";
import {Groth16Verifier} from "../src/Groth16Verifier.sol";
import {ZeroExpansionSemaphoreVerifier} from "../src/ZeroExpansionSemaphoreVerifier.sol";

/// @title ZeroExpansionSemaphoreVerifier Test Suite
/// @notice Tests security hardening, access control, pausability, reentrancy,
///         nullifier management, and proof delegation.
contract ZeroExpansionSemaphoreVerifierTest is Test {

    Groth16Verifier public groth16;
    ZeroExpansionSemaphoreVerifier public semaphore;

    address public admin = makeAddr("admin");
    address public user = makeAddr("user");
    address public attacker = makeAddr("attacker");

    // Semaphore test parameters
    bytes32 constant MERKLE_ROOT = bytes32(uint256(0xdeadbeef));
    bytes32 constant NULLIFIER = bytes32(uint256(0xcafebabe));
    bytes32 constant SIGNAL = bytes32(uint256(0x1234));
    bytes32 constant EXT_NULLIFIER = bytes32(uint256(0x5678));
    uint8 constant TREE_DEPTH = 20;

    // BN254 scalar field
    uint256 constant Q = 21888242871839275222246405745257275088548364400416034343698204186575808495617;

    // Deterministic test proof from generate-vk (secret=7, hash=49)
    uint256 constant PROOF_A_X = 0x245ea8304afe0d9c4647a0bc50952f7b62d4d469827bc5fab1a7fad31a0d889f;
    uint256 constant PROOF_A_Y = 0x1337a149ab5ab5814370c33cd14ce5d8f5cd1447bd203bd493313b60155e6d67;
    uint256 constant PROOF_B_X1 = 0x0be1b3fd7119a4be5fec155505faf449aff43e3f191cd3a6dd289312414ddd2e;
    uint256 constant PROOF_B_X0 = 0x2f9a374780a773ce910e926eca868aa8007bc2a0c9ad29158505a7fc4bfb5ea1;
    uint256 constant PROOF_B_Y1 = 0x16c4ffeb5b40992faf5afca84dda4a62e993a894451353749d64ccae73b3531d;
    uint256 constant PROOF_B_Y0 = 0x087e4c3d7af693005d980349e14d2172c5d243d245cd7a7d32791791b535caea;
    uint256 constant PROOF_C_X = 0x2bb74a8da88df1c3776550952a8c6462a99bc4fe41ed5f066eef0ca101eda202;
    uint256 constant PROOF_C_Y = 0x0fe2ff521b16e0dfa28d5702da4ba820123e5b4bb643df59ca7812bd62674ef2;

    function setUp() public {
        groth16 = new Groth16Verifier();
        vm.prank(admin);
        semaphore = new ZeroExpansionSemaphoreVerifier(address(groth16), admin);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // CONSTRUCTOR
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Constructor sets verifier and admin correctly
    function test_constructor_setsState() public view {
        assertEq(address(semaphore.groth16Verifier()), address(groth16));
        assertTrue(semaphore.hasRole(semaphore.DEFAULT_ADMIN_ROLE(), admin));
        assertTrue(semaphore.hasRole(semaphore.PAUSER_ROLE(), admin));
        assertTrue(semaphore.hasRole(semaphore.UPGRADER_ROLE(), admin));
    }

    /// @notice Constructor reverts on zero verifier address
    function test_constructor_zeroVerifier_reverts() public {
        vm.expectRevert(ZeroExpansionSemaphoreVerifier.ZeroAddress.selector);
        new ZeroExpansionSemaphoreVerifier(address(0), admin);
    }

    /// @notice Constructor reverts on zero admin address
    function test_constructor_zeroAdmin_reverts() public {
        vm.expectRevert(ZeroExpansionSemaphoreVerifier.ZeroAddress.selector);
        new ZeroExpansionSemaphoreVerifier(address(groth16), address(0));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TREE DEPTH VALIDATION
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Depth < MIN_DEPTH (16) reverts
    function test_depthBelowMin_reverts() public {
        bytes memory proof = _buildProofBytes(15);
        vm.prank(user);
        vm.expectRevert(abi.encodeWithSelector(
            ZeroExpansionSemaphoreVerifier.UnsupportedTreeDepth.selector, 15
        ));
        semaphore.verifyProof(proof, MERKLE_ROOT, NULLIFIER, SIGNAL, EXT_NULLIFIER, 15);
    }

    /// @notice Depth > MAX_DEPTH (50) reverts
    function test_depthAboveMax_reverts() public {
        bytes memory proof = _buildProofBytes(51);
        vm.prank(user);
        vm.expectRevert(abi.encodeWithSelector(
            ZeroExpansionSemaphoreVerifier.UnsupportedTreeDepth.selector, 51
        ));
        semaphore.verifyProof(proof, MERKLE_ROOT, NULLIFIER, SIGNAL, EXT_NULLIFIER, 51);
    }

    /// @notice Depth = 16 (MIN) is accepted (hits proof verification)
    function test_depthAtMin_accepted() public {
        bytes memory proof = _buildProofBytes(16);
        // Will revert with InvalidProof since proof won't pass pairing check,
        // but won't revert with UnsupportedTreeDepth
        vm.prank(user);
        vm.expectRevert(ZeroExpansionSemaphoreVerifier.InvalidProof.selector);
        semaphore.verifyProof(proof, MERKLE_ROOT, NULLIFIER, SIGNAL, EXT_NULLIFIER, 16);
    }

    /// @notice Depth = 50 (MAX) is accepted
    function test_depthAtMax_accepted() public {
        bytes memory proof = _buildProofBytes(50);
        vm.prank(user);
        vm.expectRevert(ZeroExpansionSemaphoreVerifier.InvalidProof.selector);
        semaphore.verifyProof(proof, MERKLE_ROOT, NULLIFIER, SIGNAL, EXT_NULLIFIER, 50);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PROOF HEADER VALIDATION
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Proof too short reverts
    function test_proofTooShort_reverts() public {
        bytes memory shortProof = new bytes(100);
        vm.prank(user);
        vm.expectRevert(abi.encodeWithSelector(
            ZeroExpansionSemaphoreVerifier.ProofTooShort.selector, 100
        ));
        semaphore.verifyProof(shortProof, MERKLE_ROOT, NULLIFIER, SIGNAL, EXT_NULLIFIER, TREE_DEPTH);
    }

    /// @notice Mismatched depth in proof header reverts
    function test_depthMismatch_reverts() public {
        // Build proof with depth 21 in header but pass depth 20
        bytes memory proof = _buildProofBytes(21);
        vm.prank(user);
        vm.expectRevert(abi.encodeWithSelector(
            ZeroExpansionSemaphoreVerifier.DepthMismatch.selector, 21, 20
        ));
        semaphore.verifyProof(proof, MERKLE_ROOT, NULLIFIER, SIGNAL, EXT_NULLIFIER, 20);
    }

    /// @notice Mismatched merkle root in proof header reverts
    function test_rootMismatch_reverts() public {
        // Build proof with correct depth but wrong root inside
        bytes memory proof = _buildProofBytesWithRoot(TREE_DEPTH, bytes32(uint256(0xbadbeef)));
        vm.prank(user);
        vm.expectRevert(ZeroExpansionSemaphoreVerifier.RootMismatch.selector);
        semaphore.verifyProof(proof, MERKLE_ROOT, NULLIFIER, SIGNAL, EXT_NULLIFIER, TREE_DEPTH);
    }

    /// @notice Wrong magic bytes causes InvalidProof (returns false, not revert)
    function test_wrongMagic_invalidProof() public {
        bytes memory proof = _buildProofBytes(TREE_DEPTH);
        // Corrupt first byte
        proof[0] = 0x00;
        vm.prank(user);
        vm.expectRevert(ZeroExpansionSemaphoreVerifier.InvalidProof.selector);
        semaphore.verifyProof(proof, MERKLE_ROOT, NULLIFIER, SIGNAL, EXT_NULLIFIER, TREE_DEPTH);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // NULLIFIER MANAGEMENT
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Nullifier not used by default
    function test_nullifier_notUsedInitially() public view {
        assertFalse(semaphore.isNullifierUsed(NULLIFIER));
    }

    /// @notice Double-use of nullifier reverts
    function test_doubleNullifier_reverts() public {
        // Directly set the nullifier as used via storage manipulation.
        // The `nullifierUsed` mapping is in the contract storage.
        // We use Forge's `vm.store` to mark it as used.
        // First, find the correct slot by computing mapping slot:
        // For OZ contracts, state layout depends on inherited storage.
        // Use a helper approach: call the view function to find the slot.
        bytes32 slot = keccak256(abi.encode(NULLIFIER, _findNullifierSlot()));
        vm.store(address(semaphore), slot, bytes32(uint256(1)));

        // Verify it's marked as used
        assertTrue(semaphore.isNullifierUsed(NULLIFIER));

        bytes memory proof = _buildProofBytes(TREE_DEPTH);
        vm.prank(user);
        vm.expectRevert(abi.encodeWithSelector(
            ZeroExpansionSemaphoreVerifier.NullifierAlreadyUsed.selector, NULLIFIER
        ));
        semaphore.verifyProof(proof, MERKLE_ROOT, NULLIFIER, SIGNAL, EXT_NULLIFIER, TREE_DEPTH);
    }

    /// @dev Find the storage slot for the nullifierUsed mapping by brute-force
    function _findNullifierSlot() internal returns (uint256) {
        bytes32 testNullifier = bytes32(uint256(0xdeadbeefdeadbeef));
        // Try each slot 0-20, write to it, check if the view function returns true
        for (uint256 slot = 0; slot <= 20; slot++) {
            bytes32 mappingSlot = keccak256(abi.encode(testNullifier, slot));
            vm.store(address(semaphore), mappingSlot, bytes32(uint256(1)));
            if (semaphore.isNullifierUsed(testNullifier)) {
                // Clean up
                vm.store(address(semaphore), mappingSlot, bytes32(uint256(0)));
                return slot;
            }
            // Clean up
            vm.store(address(semaphore), mappingSlot, bytes32(uint256(0)));
        }
        revert("nullifierUsed slot not found");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PAUSABLE
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Paused contract rejects proof verification
    function test_paused_revertsVerification() public {
        vm.prank(admin);
        semaphore.pause();

        bytes memory proof = _buildProofBytes(TREE_DEPTH);
        vm.prank(user);
        vm.expectRevert(abi.encodeWithSignature("EnforcedPause()"));
        semaphore.verifyProof(proof, MERKLE_ROOT, NULLIFIER, SIGNAL, EXT_NULLIFIER, TREE_DEPTH);
    }

    /// @notice Unpausing restores functionality
    function test_unpause_restoresVerification() public {
        vm.startPrank(admin);
        semaphore.pause();
        semaphore.unpause();
        vm.stopPrank();

        bytes memory proof = _buildProofBytes(TREE_DEPTH);
        // Should proceed past pause check and fail at proof verification
        vm.prank(user);
        vm.expectRevert(ZeroExpansionSemaphoreVerifier.InvalidProof.selector);
        semaphore.verifyProof(proof, MERKLE_ROOT, NULLIFIER, SIGNAL, EXT_NULLIFIER, TREE_DEPTH);
    }

    /// @notice Non-admin cannot pause
    function test_nonAdmin_cannotPause() public {
        vm.prank(attacker);
        vm.expectRevert();
        semaphore.pause();
    }

    /// @notice Non-admin cannot unpause
    function test_nonAdmin_cannotUnpause() public {
        vm.prank(admin);
        semaphore.pause();

        vm.prank(attacker);
        vm.expectRevert();
        semaphore.unpause();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ACCESS CONTROL
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Admin can grant PAUSER_ROLE to another address
    function test_admin_canGrantPauserRole() public {
        address newPauser = makeAddr("newPauser");
        bytes32 pauserRole = semaphore.PAUSER_ROLE();

        vm.prank(admin);
        semaphore.grantRole(pauserRole, newPauser);

        assertTrue(semaphore.hasRole(pauserRole, newPauser));

        // New pauser can pause
        vm.prank(newPauser);
        semaphore.pause();
        assertTrue(semaphore.paused());
    }

    /// @notice Non-admin cannot grant roles
    function test_nonAdmin_cannotGrantRole() public {
        bytes32 pauserRole = semaphore.PAUSER_ROLE();
        bytes32 adminRole = semaphore.DEFAULT_ADMIN_ROLE();

        vm.prank(attacker);
        vm.expectRevert(
            abi.encodeWithSignature(
                "AccessControlUnauthorizedAccount(address,bytes32)",
                attacker,
                adminRole
            )
        );
        semaphore.grantRole(pauserRole, attacker);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // VERIFIER UPDATE (TIMELOCKED)
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Verifier update before timelock expires reverts
    function test_updateVerifier_beforeTimelock_reverts() public {
        Groth16Verifier newVerifier = new Groth16Verifier();
        vm.prank(admin);
        vm.expectRevert(); // TimelockNotExpired
        semaphore.updateVerifier(address(newVerifier));
    }

    /// @notice Verifier update after timelock succeeds
    function test_updateVerifier_afterTimelock_succeeds() public {
        Groth16Verifier newVerifier = new Groth16Verifier();

        // Warp past the 48-hour timelock
        vm.warp(block.timestamp + 48 hours + 1);

        vm.prank(admin);
        semaphore.updateVerifier(address(newVerifier));

        assertEq(address(semaphore.groth16Verifier()), address(newVerifier));
    }

    /// @notice Verifier update to zero address reverts
    function test_updateVerifier_zeroAddress_reverts() public {
        vm.warp(block.timestamp + 48 hours + 1);
        vm.prank(admin);
        vm.expectRevert(ZeroExpansionSemaphoreVerifier.ZeroAddress.selector);
        semaphore.updateVerifier(address(0));
    }

    /// @notice Non-upgrader cannot update verifier
    function test_nonUpgrader_cannotUpdate() public {
        Groth16Verifier newVerifier = new Groth16Verifier();
        vm.warp(block.timestamp + 48 hours + 1);
        vm.prank(attacker);
        vm.expectRevert();
        semaphore.updateVerifier(address(newVerifier));
    }

    /// @notice Verifier update emits event
    function test_updateVerifier_emitsEvent() public {
        Groth16Verifier newVerifier = new Groth16Verifier();
        vm.warp(block.timestamp + 48 hours + 1);

        vm.prank(admin);
        vm.expectEmit(true, true, false, false);
        emit ZeroExpansionSemaphoreVerifier.VerifierUpdated(address(groth16), address(newVerifier));
        semaphore.updateVerifier(address(newVerifier));
    }

    /// @notice Consecutive verifier updates must wait between each
    function test_updateVerifier_consecutiveNeedsDelay() public {
        Groth16Verifier v2 = new Groth16Verifier();
        Groth16Verifier v3 = new Groth16Verifier();

        // First update after initial timelock
        vm.warp(block.timestamp + 48 hours + 1);
        vm.prank(admin);
        semaphore.updateVerifier(address(v2));

        // Second update immediately fails
        vm.prank(admin);
        vm.expectRevert(); // TimelockNotExpired
        semaphore.updateVerifier(address(v3));

        // After another 48 hours, succeeds
        vm.warp(block.timestamp + 48 hours + 1);
        vm.prank(admin);
        semaphore.updateVerifier(address(v3));
        assertEq(address(semaphore.groth16Verifier()), address(v3));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // VIEW FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice maxMembers returns 2^50
    function test_maxMembers() public view {
        assertEq(semaphore.maxMembers(), 1125899906842624);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // GAS BENCHMARKS
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Gas benchmark for full verifyProof flow (will fail at pairing)
    function test_gas_verifyProofFlow() public {
        bytes memory proof = _buildProofBytes(TREE_DEPTH);
        uint256 gasBefore = gasleft();
        vm.prank(user);
        try semaphore.verifyProof(proof, MERKLE_ROOT, NULLIFIER, SIGNAL, EXT_NULLIFIER, TREE_DEPTH) {
            // Won't reach here with fake proof
        } catch {
            // Expected
        }
        uint256 gasUsed = gasBefore - gasleft();
        console.log("Gas used for verifyProof (reject path):", gasUsed);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // HELPERS
    // ═══════════════════════════════════════════════════════════════════════

    /// @dev Build proof bytes with correct magic, depth, root, and Groth16 data
    function _buildProofBytes(uint8 depth) internal pure returns (bytes memory) {
        return _buildProofBytesWithRoot(depth, MERKLE_ROOT);
    }

    /// @dev Build proof bytes with specified root embedded
    function _buildProofBytesWithRoot(uint8 depth, bytes32 root) internal pure returns (bytes memory) {
        bytes memory proof = new bytes(340);

        // Magic: "ZERO_EXPANSION_SEMAPHORE_V3" (27 bytes)
        bytes memory magic = "ZERO_EXPANSION_SEMAPHORE_V3";
        for (uint256 i = 0; i < 27; i++) {
            proof[i] = magic[i];
        }

        // Depth at offset 27
        proof[27] = bytes1(depth);

        // Merkle root at offset 28 (32 bytes)
        assembly {
            mstore(add(add(proof, 32), 28), root)
        }

        // RMT challenges at offset 60 (24 bytes) — zero-filled for test
        // (The real system would compute these from QTT decomposition)

        // Groth16 proof at offset 84 (256 bytes = 8 × 32)
        uint256[8] memory g16 = [
            PROOF_A_X, PROOF_A_Y,
            PROOF_B_X1, PROOF_B_X0,
            PROOF_B_Y1, PROOF_B_Y0,
            PROOF_C_X, PROOF_C_Y
        ];
        for (uint256 i = 0; i < 8; i++) {
            assembly {
                mstore(add(add(proof, 32), add(84, mul(i, 32))), mload(add(g16, mul(i, 32))))
            }
        }

        return proof;
    }
}
