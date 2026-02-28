// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {AccessControl} from "@openzeppelin/contracts/access/AccessControl.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import {Pausable} from "@openzeppelin/contracts/utils/Pausable.sol";

/// @title VK Update Governance — Timelock + Multi-Sig for Verification Key Rotation
/// @author Ontic Labs
/// @notice Queues VK updates with a 48-hour timelock and requires 2-of-3 multi-sig
///         approval before execution. Protects all FluidElite verifier contracts from
///         malicious VK swaps.
/// @dev Designed for `Groth16Verifier` and `FluidEliteHalo2Verifier` VK rotation.
///      Events emitted at every lifecycle stage for off-chain indexing.
contract VKGovernance is AccessControl, ReentrancyGuard, Pausable {

    // ═══════════════════════════════════════════════════════════════════════
    // ROLES
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Role that can propose VK updates
    bytes32 public constant PROPOSER_ROLE = keccak256("PROPOSER_ROLE");

    /// @notice Role that can approve (sign) queued proposals
    bytes32 public constant SIGNER_ROLE = keccak256("SIGNER_ROLE");

    /// @notice Role that can execute approved proposals after timelock
    bytes32 public constant EXECUTOR_ROLE = keccak256("EXECUTOR_ROLE");

    /// @notice Role that can cancel queued proposals and pause
    bytes32 public constant GUARDIAN_ROLE = keccak256("GUARDIAN_ROLE");

    // ═══════════════════════════════════════════════════════════════════════
    // CONSTANTS
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Minimum delay between proposal and execution (48 hours)
    uint256 public constant MIN_DELAY = 48 hours;

    /// @notice Maximum delay before a proposal expires (14 days)
    uint256 public constant MAX_DELAY = 14 days;

    /// @notice Required approvals for execution (2-of-3)
    uint256 public constant REQUIRED_APPROVALS = 2;

    // ═══════════════════════════════════════════════════════════════════════
    // TYPES
    // ═══════════════════════════════════════════════════════════════════════

    enum ProposalState {
        Pending,    // Proposed, awaiting approvals
        Approved,   // Has enough approvals, awaiting timelock
        Executed,   // Successfully executed
        Cancelled,  // Cancelled by guardian
        Expired     // Timelock + MAX_DELAY elapsed without execution
    }

    struct Proposal {
        /// @notice Contract whose VK will be updated
        address targetVerifier;
        /// @notice Calldata to execute on the target (e.g., updateVerifier or setVkHash)
        bytes callData;
        /// @notice SHA-256 of the new VK parameters for auditing
        bytes32 newVkHash;
        /// @notice Timestamp when proposal was created
        uint256 proposedAt;
        /// @notice Timestamp after which execution is allowed
        uint256 executableAfter;
        /// @notice Current approval count
        uint256 approvalCount;
        /// @notice Current state
        ProposalState state;
        /// @notice Description of the change
        string description;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // STATE
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Monotonically increasing proposal ID counter
    uint256 public proposalCount;

    /// @notice All proposals by ID
    mapping(uint256 => Proposal) public proposals;

    /// @notice Tracks which signers have approved each proposal
    mapping(uint256 => mapping(address => bool)) public hasApproved;

    /// @notice Registry of allowed target verifier contracts
    mapping(address => bool) public allowedTargets;

    // ═══════════════════════════════════════════════════════════════════════
    // ERRORS
    // ═══════════════════════════════════════════════════════════════════════

    error ProposalNotPending(uint256 proposalId, ProposalState state);
    error ProposalNotApproved(uint256 proposalId, ProposalState state);
    error TimelockNotElapsed(uint256 proposalId, uint256 executableAfter, uint256 current);
    error ProposalExpired(uint256 proposalId);
    error AlreadyApproved(uint256 proposalId, address signer);
    error TargetNotAllowed(address target);
    error ExecutionFailed(uint256 proposalId, bytes reason);
    error ZeroAddress();

    // ═══════════════════════════════════════════════════════════════════════
    // EVENTS
    // ═══════════════════════════════════════════════════════════════════════

    event ProposalCreated(
        uint256 indexed proposalId,
        address indexed targetVerifier,
        bytes32 newVkHash,
        uint256 executableAfter,
        string description
    );

    event ProposalApproved(
        uint256 indexed proposalId,
        address indexed signer,
        uint256 approvalCount
    );

    event ProposalExecuted(
        uint256 indexed proposalId,
        address indexed targetVerifier,
        bytes32 newVkHash
    );

    event ProposalCancelled(
        uint256 indexed proposalId,
        address indexed guardian
    );

    event TargetAdded(address indexed target);
    event TargetRemoved(address indexed target);

    // ═══════════════════════════════════════════════════════════════════════
    // CONSTRUCTOR
    // ═══════════════════════════════════════════════════════════════════════

    /// @param admin     Address receiving DEFAULT_ADMIN_ROLE
    /// @param signers   Array of 3 signer addresses (2-of-3 required for approval)
    /// @param targets   Initial allowed verifier contract addresses
    constructor(
        address admin,
        address[3] memory signers,
        address[] memory targets
    ) {
        if (admin == address(0)) revert ZeroAddress();

        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        _grantRole(PROPOSER_ROLE, admin);
        _grantRole(EXECUTOR_ROLE, admin);
        _grantRole(GUARDIAN_ROLE, admin);

        for (uint256 i = 0; i < 3; i++) {
            if (signers[i] == address(0)) revert ZeroAddress();
            _grantRole(SIGNER_ROLE, signers[i]);
        }

        for (uint256 i = 0; i < targets.length; i++) {
            if (targets[i] == address(0)) revert ZeroAddress();
            allowedTargets[targets[i]] = true;
            emit TargetAdded(targets[i]);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ADMIN: TARGET MANAGEMENT
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Register a verifier contract as a valid target
    function addTarget(address target) external onlyRole(DEFAULT_ADMIN_ROLE) {
        if (target == address(0)) revert ZeroAddress();
        allowedTargets[target] = true;
        emit TargetAdded(target);
    }

    /// @notice Remove a verifier contract from allowed targets
    function removeTarget(address target) external onlyRole(DEFAULT_ADMIN_ROLE) {
        allowedTargets[target] = false;
        emit TargetRemoved(target);
    }

    /// @notice Pause all governance operations (emergency)
    function pause() external onlyRole(GUARDIAN_ROLE) {
        _pause();
    }

    /// @notice Unpause governance operations
    function unpause() external onlyRole(DEFAULT_ADMIN_ROLE) {
        _unpause();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PROPOSAL LIFECYCLE
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Propose a VK update
    /// @param targetVerifier Address of the verifier contract to update
    /// @param callData       ABI-encoded function call to execute on target
    /// @param newVkHash      SHA-256 hash of the new verification key
    /// @param description    Human-readable description of the change
    /// @return proposalId    The new proposal's ID
    function propose(
        address targetVerifier,
        bytes calldata callData,
        bytes32 newVkHash,
        string calldata description
    ) external onlyRole(PROPOSER_ROLE) whenNotPaused returns (uint256 proposalId) {
        if (!allowedTargets[targetVerifier]) revert TargetNotAllowed(targetVerifier);

        proposalId = proposalCount++;
        uint256 executableAfter = block.timestamp + MIN_DELAY;

        proposals[proposalId] = Proposal({
            targetVerifier: targetVerifier,
            callData: callData,
            newVkHash: newVkHash,
            proposedAt: block.timestamp,
            executableAfter: executableAfter,
            approvalCount: 0,
            state: ProposalState.Pending,
            description: description
        });

        emit ProposalCreated(
            proposalId,
            targetVerifier,
            newVkHash,
            executableAfter,
            description
        );
    }

    /// @notice Approve a pending proposal (signer must have SIGNER_ROLE)
    /// @param proposalId The proposal to approve
    function approve(uint256 proposalId) external onlyRole(SIGNER_ROLE) whenNotPaused {
        Proposal storage p = proposals[proposalId];

        if (p.state != ProposalState.Pending) {
            revert ProposalNotPending(proposalId, p.state);
        }
        if (hasApproved[proposalId][msg.sender]) {
            revert AlreadyApproved(proposalId, msg.sender);
        }

        hasApproved[proposalId][msg.sender] = true;
        p.approvalCount++;

        emit ProposalApproved(proposalId, msg.sender, p.approvalCount);

        if (p.approvalCount >= REQUIRED_APPROVALS) {
            p.state = ProposalState.Approved;
        }
    }

    /// @notice Execute an approved proposal after timelock elapses
    /// @param proposalId The proposal to execute
    function execute(uint256 proposalId)
        external
        onlyRole(EXECUTOR_ROLE)
        whenNotPaused
        nonReentrant
    {
        Proposal storage p = proposals[proposalId];

        if (p.state != ProposalState.Approved) {
            revert ProposalNotApproved(proposalId, p.state);
        }
        if (block.timestamp < p.executableAfter) {
            revert TimelockNotElapsed(proposalId, p.executableAfter, block.timestamp);
        }
        if (block.timestamp > p.executableAfter + MAX_DELAY) {
            p.state = ProposalState.Expired;
            revert ProposalExpired(proposalId);
        }

        p.state = ProposalState.Executed;

        (bool success, bytes memory returnData) = p.targetVerifier.call(p.callData);
        if (!success) {
            revert ExecutionFailed(proposalId, returnData);
        }

        emit ProposalExecuted(proposalId, p.targetVerifier, p.newVkHash);
    }

    /// @notice Cancel a pending/approved proposal (guardian only)
    /// @param proposalId The proposal to cancel
    function cancel(uint256 proposalId) external onlyRole(GUARDIAN_ROLE) {
        Proposal storage p = proposals[proposalId];

        if (p.state != ProposalState.Pending && p.state != ProposalState.Approved) {
            revert ProposalNotPending(proposalId, p.state);
        }

        p.state = ProposalState.Cancelled;
        emit ProposalCancelled(proposalId, msg.sender);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // VIEW FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Get the effective state of a proposal (accounts for expiry)
    function getProposalState(uint256 proposalId) external view returns (ProposalState) {
        Proposal storage p = proposals[proposalId];

        if (p.state == ProposalState.Approved &&
            block.timestamp > p.executableAfter + MAX_DELAY) {
            return ProposalState.Expired;
        }

        return p.state;
    }

    /// @notice Get full proposal details
    function getProposal(uint256 proposalId) external view returns (
        address targetVerifier,
        bytes memory callData,
        bytes32 newVkHash,
        uint256 proposedAt,
        uint256 executableAfter,
        uint256 approvalCount,
        ProposalState state,
        string memory description
    ) {
        Proposal storage p = proposals[proposalId];
        return (
            p.targetVerifier,
            p.callData,
            p.newVkHash,
            p.proposedAt,
            p.executableAfter,
            p.approvalCount,
            p.state,
            p.description
        );
    }
}
