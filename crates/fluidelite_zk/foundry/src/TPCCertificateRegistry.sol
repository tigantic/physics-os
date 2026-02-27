// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {AccessControl} from "@openzeppelin/contracts/access/AccessControl.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/// @title TPC Certificate Registry — On-Chain Record of Trustless Physics Certificates
/// @author HyperTensor Labs
/// @notice Stores certificate hashes and metadata on-chain for public auditability.
///         Links TPC binary certificates to their on-chain verification proofs.
/// @dev Designed to work alongside `Groth16Verifier` and `FluidEliteHalo2Verifier`.
///      Each registered certificate contains a content hash, the verification tx,
///      and domain metadata (thermal, euler3d, ns_imex, fluidelite).
contract TPCCertificateRegistry is AccessControl, ReentrancyGuard {

    // ═══════════════════════════════════════════════════════════════════════
    // ROLES
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Role that can register new certificates (certificate authority)
    bytes32 public constant REGISTRAR_ROLE = keccak256("REGISTRAR_ROLE");

    /// @notice Role that can revoke certificates
    bytes32 public constant REVOKER_ROLE = keccak256("REVOKER_ROLE");

    // ═══════════════════════════════════════════════════════════════════════
    // TYPES
    // ═══════════════════════════════════════════════════════════════════════

    enum CertificateStatus {
        Valid,
        Revoked,
        Superseded
    }

    struct Certificate {
        /// @notice SHA-256 hash of the full TPC binary certificate
        bytes32 contentHash;
        /// @notice Ed25519 public key of the signer (32 bytes)
        bytes32 signerPubkey;
        /// @notice Physics domain (0=thermal, 1=euler3d, 2=ns_imex, 3=fluidelite)
        uint8 domain;
        /// @notice Block timestamp when registered
        uint256 registeredAt;
        /// @notice Address that submitted the registration
        address registeredBy;
        /// @notice On-chain proof verification transaction hash (if verified)
        bytes32 proofTxHash;
        /// @notice Certificate status
        CertificateStatus status;
        /// @notice UUID of the TPC certificate (16 bytes, stored as bytes16)
        bytes16 certificateId;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // STATE
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Total number of registered certificates
    uint256 public certificateCount;

    /// @notice Certificate by sequential index
    mapping(uint256 => Certificate) public certificates;

    /// @notice Lookup certificate index by content hash (unique)
    mapping(bytes32 => uint256) public hashToIndex;

    /// @notice Whether a content hash has been registered
    mapping(bytes32 => bool) public isRegistered;

    /// @notice Count of certificates per domain
    mapping(uint8 => uint256) public domainCount;

    /// @notice Allowed signer public keys (whitelisted CAs)
    mapping(bytes32 => bool) public allowedSigners;

    /// @notice PQC commitment hash linked to certificate (Dilithium2 binding)
    mapping(uint256 => bytes32) public pqcCommitments;

    // ═══════════════════════════════════════════════════════════════════════
    // ERRORS
    // ═══════════════════════════════════════════════════════════════════════

    error AlreadyRegistered(bytes32 contentHash);
    error CertificateNotFound(uint256 index);
    error InvalidDomain(uint8 domain);
    error SignerNotAllowed(bytes32 signerPubkey);
    error CertificateRevoked(uint256 index);
    error ZeroHash();

    // ═══════════════════════════════════════════════════════════════════════
    // EVENTS
    // ═══════════════════════════════════════════════════════════════════════

    event CertificateRegistered(
        uint256 indexed index,
        bytes32 indexed contentHash,
        bytes32 signerPubkey,
        uint8 domain,
        bytes16 certificateId
    );

    event CertificateRevoked(
        uint256 indexed index,
        bytes32 indexed contentHash,
        address revokedBy,
        string reason
    );

    event CertificateSuperseded(
        uint256 indexed oldIndex,
        uint256 indexed newIndex
    );

    event ProofLinked(
        uint256 indexed index,
        bytes32 proofTxHash
    );

    event SignerAdded(bytes32 indexed pubkey);
    event SignerRemoved(bytes32 indexed pubkey);
    event PQCCommitmentRegistered(uint256 indexed index, bytes32 commitmentHash);

    // ═══════════════════════════════════════════════════════════════════════
    // CONSTRUCTOR
    // ═══════════════════════════════════════════════════════════════════════

    /// @param admin      Address receiving DEFAULT_ADMIN_ROLE
    /// @param registrar  Address receiving REGISTRAR_ROLE (certificate authority)
    constructor(address admin, address registrar) {
        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        _grantRole(REGISTRAR_ROLE, registrar);
        _grantRole(REVOKER_ROLE, admin);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // SIGNER MANAGEMENT
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Add an Ed25519 public key to the allowed signers set
    function addSigner(bytes32 pubkey) external onlyRole(DEFAULT_ADMIN_ROLE) {
        if (pubkey == bytes32(0)) revert ZeroHash();
        allowedSigners[pubkey] = true;
        emit SignerAdded(pubkey);
    }

    /// @notice Remove an Ed25519 public key from allowed signers
    function removeSigner(bytes32 pubkey) external onlyRole(DEFAULT_ADMIN_ROLE) {
        allowedSigners[pubkey] = false;
        emit SignerRemoved(pubkey);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // CERTIFICATE REGISTRATION
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Register a new TPC certificate on-chain
    /// @param contentHash   SHA-256 of the complete TPC binary certificate
    /// @param signerPubkey  Ed25519 public key that signed the certificate
    /// @param domain        Physics domain (0-3)
    /// @param certificateId UUID from the TPC header (bytes16)
    /// @param proofTxHash   Transaction hash of the on-chain proof verification (0 if not yet verified)
    /// @return index        Sequential certificate index
    function register(
        bytes32 contentHash,
        bytes32 signerPubkey,
        uint8 domain,
        bytes16 certificateId,
        bytes32 proofTxHash
    ) external onlyRole(REGISTRAR_ROLE) nonReentrant returns (uint256 index) {
        if (contentHash == bytes32(0)) revert ZeroHash();
        if (isRegistered[contentHash]) revert AlreadyRegistered(contentHash);
        if (domain > 3) revert InvalidDomain(domain);
        if (!allowedSigners[signerPubkey]) revert SignerNotAllowed(signerPubkey);

        index = certificateCount++;

        certificates[index] = Certificate({
            contentHash: contentHash,
            signerPubkey: signerPubkey,
            domain: domain,
            registeredAt: block.timestamp,
            registeredBy: msg.sender,
            proofTxHash: proofTxHash,
            status: CertificateStatus.Valid,
            certificateId: certificateId
        });

        hashToIndex[contentHash] = index;
        isRegistered[contentHash] = true;
        domainCount[domain]++;

        emit CertificateRegistered(index, contentHash, signerPubkey, domain, certificateId);
    }

    /// @notice Link an on-chain proof verification to a certificate
    /// @param index      Certificate index
    /// @param proofTxHash Transaction hash of the proof verification
    function linkProof(
        uint256 index,
        bytes32 proofTxHash
    ) external onlyRole(REGISTRAR_ROLE) {
        if (index >= certificateCount) revert CertificateNotFound(index);

        Certificate storage cert = certificates[index];
        if (cert.status == CertificateStatus.Revoked) revert CertificateRevoked(index);

        cert.proofTxHash = proofTxHash;
        emit ProofLinked(index, proofTxHash);
    }

    /// @notice Revoke a certificate
    /// @param index  Certificate index
    /// @param reason Human-readable reason for revocation
    function revoke(
        uint256 index,
        string calldata reason
    ) external onlyRole(REVOKER_ROLE) {
        if (index >= certificateCount) revert CertificateNotFound(index);

        Certificate storage cert = certificates[index];
        cert.status = CertificateStatus.Revoked;

        emit CertificateRevoked(index, cert.contentHash, msg.sender, reason);
    }

    /// @notice Mark a certificate as superseded by a newer one
    /// @param oldIndex Index of the certificate being superseded
    /// @param newIndex Index of the replacement certificate
    function supersede(
        uint256 oldIndex,
        uint256 newIndex
    ) external onlyRole(REGISTRAR_ROLE) {
        if (oldIndex >= certificateCount) revert CertificateNotFound(oldIndex);
        if (newIndex >= certificateCount) revert CertificateNotFound(newIndex);

        certificates[oldIndex].status = CertificateStatus.Superseded;
        emit CertificateSuperseded(oldIndex, newIndex);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PQC BINDING (Task 4.9 — Dilithium2)
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Register a post-quantum commitment hash alongside a certificate
    /// @dev Stores SHA-256(Dilithium2_signature || Dilithium2_pubkey) for future
    ///      PQC verification. The full Dilithium2 signature is stored off-chain
    ///      in the TPC binary; only the binding hash lives on-chain.
    /// @param index          Certificate index
    /// @param commitmentHash SHA-256 of (Dilithium2 signature + public key)
    function registerPQCCommitment(
        uint256 index,
        bytes32 commitmentHash
    ) external onlyRole(REGISTRAR_ROLE) {
        if (index >= certificateCount) revert CertificateNotFound(index);
        if (commitmentHash == bytes32(0)) revert ZeroHash();

        pqcCommitments[index] = commitmentHash;
        emit PQCCommitmentRegistered(index, commitmentHash);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // VIEW FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Verify a certificate exists and is valid by content hash
    function verifyCertificate(bytes32 contentHash) external view returns (
        bool exists,
        CertificateStatus status,
        uint256 index
    ) {
        exists = isRegistered[contentHash];
        if (exists) {
            index = hashToIndex[contentHash];
            status = certificates[index].status;
        }
    }

    /// @notice Get full certificate details
    function getCertificate(uint256 index) external view returns (Certificate memory) {
        if (index >= certificateCount) revert CertificateNotFound(index);
        return certificates[index];
    }

    /// @notice Check if a PQC commitment exists for a certificate
    function hasPQCCommitment(uint256 index) external view returns (bool) {
        return pqcCommitments[index] != bytes32(0);
    }
}
