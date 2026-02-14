// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Test, console} from "forge-std/Test.sol";
import {VKGovernance} from "../src/VKGovernance.sol";

/// @title VKGovernance Tests
/// @notice Comprehensive tests for the timelock + multi-sig VK governance contract
contract VKGovernanceTest is Test {

    VKGovernance public governance;

    address public admin = makeAddr("admin");
    address public signer1 = makeAddr("signer1");
    address public signer2 = makeAddr("signer2");
    address public signer3 = makeAddr("signer3");
    address public attacker = makeAddr("attacker");

    // Dummy target contract
    address public target;
    DummyVerifier public dummyVerifier;

    function setUp() public {
        dummyVerifier = new DummyVerifier();
        target = address(dummyVerifier);

        address[] memory targets = new address[](1);
        targets[0] = target;

        vm.prank(admin);
        governance = new VKGovernance(
            admin,
            [signer1, signer2, signer3],
            targets
        );
    }

    // ─────────────────────────────────────────────────────
    // Constructor
    // ─────────────────────────────────────────────────────

    function test_constructor_sets_roles() public view {
        assertTrue(governance.hasRole(governance.DEFAULT_ADMIN_ROLE(), admin));
        assertTrue(governance.hasRole(governance.PROPOSER_ROLE(), admin));
        assertTrue(governance.hasRole(governance.EXECUTOR_ROLE(), admin));
        assertTrue(governance.hasRole(governance.GUARDIAN_ROLE(), admin));
        assertTrue(governance.hasRole(governance.SIGNER_ROLE(), signer1));
        assertTrue(governance.hasRole(governance.SIGNER_ROLE(), signer2));
        assertTrue(governance.hasRole(governance.SIGNER_ROLE(), signer3));
    }

    function test_constructor_registers_targets() public view {
        assertTrue(governance.allowedTargets(target));
    }

    function test_constructor_reverts_zero_admin() public {
        address[] memory targets = new address[](0);
        vm.expectRevert(VKGovernance.ZeroAddress.selector);
        new VKGovernance(address(0), [signer1, signer2, signer3], targets);
    }

    function test_constructor_reverts_zero_signer() public {
        address[] memory targets = new address[](0);
        vm.expectRevert(VKGovernance.ZeroAddress.selector);
        new VKGovernance(admin, [signer1, address(0), signer3], targets);
    }

    // ─────────────────────────────────────────────────────
    // Propose
    // ─────────────────────────────────────────────────────

    function test_propose_creates_proposal() public {
        bytes32 vkHash = keccak256("new_vk");
        bytes memory callData = abi.encodeWithSignature("setVk(bytes32)", vkHash);

        vm.prank(admin);
        uint256 id = governance.propose(target, callData, vkHash, "Update VK for thermal circuit");

        assertEq(id, 0);
        assertEq(governance.proposalCount(), 1);

        (
            address targetVerifier,
            ,
            bytes32 newVkHash,
            uint256 proposedAt,
            uint256 executableAfter,
            uint256 approvalCount,
            VKGovernance.ProposalState state,
        ) = governance.getProposal(0);

        assertEq(targetVerifier, target);
        assertEq(newVkHash, vkHash);
        assertEq(proposedAt, block.timestamp);
        assertEq(executableAfter, block.timestamp + 48 hours);
        assertEq(approvalCount, 0);
        assertEq(uint256(state), uint256(VKGovernance.ProposalState.Pending));
    }

    function test_propose_reverts_unallowed_target() public {
        vm.prank(admin);
        vm.expectRevert(abi.encodeWithSelector(VKGovernance.TargetNotAllowed.selector, attacker));
        governance.propose(attacker, "", keccak256("vk"), "bad target");
    }

    function test_propose_reverts_unauthorized() public {
        vm.prank(attacker);
        vm.expectRevert();
        governance.propose(target, "", keccak256("vk"), "no role");
    }

    // ─────────────────────────────────────────────────────
    // Approve
    // ─────────────────────────────────────────────────────

    function test_approve_increments_count() public {
        _createProposal();

        vm.prank(signer1);
        governance.approve(0);
        (, , , , , uint256 approvalCount, ,) = governance.getProposal(0);
        assertEq(approvalCount, 1);

        vm.prank(signer2);
        governance.approve(0);
        (, , , , , approvalCount, VKGovernance.ProposalState state,) = governance.getProposal(0);
        assertEq(approvalCount, 2);
        assertEq(uint256(state), uint256(VKGovernance.ProposalState.Approved));
    }

    function test_approve_reverts_duplicate() public {
        _createProposal();

        vm.prank(signer1);
        governance.approve(0);

        vm.prank(signer1);
        vm.expectRevert(abi.encodeWithSelector(VKGovernance.AlreadyApproved.selector, 0, signer1));
        governance.approve(0);
    }

    function test_approve_reverts_unauthorized() public {
        _createProposal();

        vm.prank(attacker);
        vm.expectRevert();
        governance.approve(0);
    }

    // ─────────────────────────────────────────────────────
    // Execute
    // ─────────────────────────────────────────────────────

    function test_execute_after_timelock() public {
        bytes32 vkHash = keccak256("new_vk");
        bytes memory callData = abi.encodeWithSignature("setVk(bytes32)", vkHash);

        vm.prank(admin);
        governance.propose(target, callData, vkHash, "VK update");

        vm.prank(signer1);
        governance.approve(0);
        vm.prank(signer2);
        governance.approve(0);

        // Fast-forward past timelock
        vm.warp(block.timestamp + 48 hours + 1);

        vm.prank(admin);
        governance.execute(0);

        (, , , , , , VKGovernance.ProposalState state,) = governance.getProposal(0);
        assertEq(uint256(state), uint256(VKGovernance.ProposalState.Executed));

        // Verify the target was called
        assertEq(dummyVerifier.vk(), vkHash);
    }

    function test_execute_reverts_before_timelock() public {
        _createAndApproveProposal();

        vm.prank(admin);
        vm.expectRevert();
        governance.execute(0);
    }

    function test_execute_reverts_expired() public {
        _createAndApproveProposal();

        // Fast-forward past timelock + MAX_DELAY
        vm.warp(block.timestamp + 48 hours + 14 days + 1);

        vm.prank(admin);
        vm.expectRevert(abi.encodeWithSelector(VKGovernance.ProposalExpired.selector, 0));
        governance.execute(0);
    }

    function test_execute_reverts_not_approved() public {
        _createProposal();

        // Only 1 approval (need 2)
        vm.prank(signer1);
        governance.approve(0);

        vm.warp(block.timestamp + 48 hours + 1);

        vm.prank(admin);
        vm.expectRevert();
        governance.execute(0);
    }

    // ─────────────────────────────────────────────────────
    // Cancel
    // ─────────────────────────────────────────────────────

    function test_cancel_pending() public {
        _createProposal();

        vm.prank(admin);
        governance.cancel(0);

        (, , , , , , VKGovernance.ProposalState state,) = governance.getProposal(0);
        assertEq(uint256(state), uint256(VKGovernance.ProposalState.Cancelled));
    }

    function test_cancel_approved() public {
        _createAndApproveProposal();

        vm.prank(admin);
        governance.cancel(0);

        (, , , , , , VKGovernance.ProposalState state,) = governance.getProposal(0);
        assertEq(uint256(state), uint256(VKGovernance.ProposalState.Cancelled));
    }

    // ─────────────────────────────────────────────────────
    // Pause
    // ─────────────────────────────────────────────────────

    function test_pause_blocks_propose() public {
        vm.prank(admin);
        governance.pause();

        vm.prank(admin);
        vm.expectRevert();
        governance.propose(target, "", keccak256("vk"), "blocked");
    }

    function test_unpause_restores_propose() public {
        vm.prank(admin);
        governance.pause();

        vm.prank(admin);
        governance.unpause();

        vm.prank(admin);
        governance.propose(target, "", keccak256("vk"), "unblocked");
        assertEq(governance.proposalCount(), 1);
    }

    // ─────────────────────────────────────────────────────
    // Target management
    // ─────────────────────────────────────────────────────

    function test_add_remove_target() public {
        address newTarget = makeAddr("newTarget");

        vm.prank(admin);
        governance.addTarget(newTarget);
        assertTrue(governance.allowedTargets(newTarget));

        vm.prank(admin);
        governance.removeTarget(newTarget);
        assertFalse(governance.allowedTargets(newTarget));
    }

    // ─────────────────────────────────────────────────────
    // View: getProposalState with expiry
    // ─────────────────────────────────────────────────────

    function test_getProposalState_returns_expired() public {
        _createAndApproveProposal();

        vm.warp(block.timestamp + 48 hours + 14 days + 1);
        assertEq(uint256(governance.getProposalState(0)), uint256(VKGovernance.ProposalState.Expired));
    }

    // ─────────────────────────────────────────────────────
    // Gas benchmarks
    // ─────────────────────────────────────────────────────

    function test_gas_propose() public {
        vm.prank(admin);
        uint256 gasBefore = gasleft();
        governance.propose(target, "", keccak256("vk"), "Gas benchmark");
        uint256 gasUsed = gasBefore - gasleft();
        console.log("Gas: propose() =", gasUsed);
    }

    function test_gas_approve() public {
        _createProposal();

        vm.prank(signer1);
        uint256 gasBefore = gasleft();
        governance.approve(0);
        uint256 gasUsed = gasBefore - gasleft();
        console.log("Gas: approve() =", gasUsed);
    }

    function test_gas_execute() public {
        bytes32 vkHash = keccak256("vk");
        bytes memory callData = abi.encodeWithSignature("setVk(bytes32)", vkHash);

        vm.prank(admin);
        governance.propose(target, callData, vkHash, "Gas benchmark");
        vm.prank(signer1);
        governance.approve(0);
        vm.prank(signer2);
        governance.approve(0);
        vm.warp(block.timestamp + 48 hours + 1);

        vm.prank(admin);
        uint256 gasBefore = gasleft();
        governance.execute(0);
        uint256 gasUsed = gasBefore - gasleft();
        console.log("Gas: execute() =", gasUsed);
    }

    // ─────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────

    function _createProposal() internal {
        vm.prank(admin);
        governance.propose(target, "", keccak256("vk"), "Test proposal");
    }

    function _createAndApproveProposal() internal {
        _createProposal();
        vm.prank(signer1);
        governance.approve(0);
        vm.prank(signer2);
        governance.approve(0);
    }
}

/// @notice Dummy verifier for testing governance execution
contract DummyVerifier {
    bytes32 public vk;

    function setVk(bytes32 _vk) external {
        vk = _vk;
    }
}
