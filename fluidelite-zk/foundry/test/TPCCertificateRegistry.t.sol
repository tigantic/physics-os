// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Test, console} from "forge-std/Test.sol";
import {TPCCertificateRegistry} from "../src/TPCCertificateRegistry.sol";

/// @title TPCCertificateRegistry Tests
/// @notice Comprehensive tests for on-chain certificate registration, revocation,
///         PQC binding, and access control.
contract TPCCertificateRegistryTest is Test {

    TPCCertificateRegistry public registry;

    address public admin = makeAddr("admin");
    address public registrar = makeAddr("registrar");
    address public attacker = makeAddr("attacker");

    bytes32 public signerPubkey = keccak256("ed25519_pubkey");
    bytes32 public contentHash1 = keccak256("cert_content_1");
    bytes32 public contentHash2 = keccak256("cert_content_2");
    bytes16 public certId1 = bytes16(keccak256("uuid_1"));
    bytes16 public certId2 = bytes16(keccak256("uuid_2"));

    function setUp() public {
        vm.prank(admin);
        registry = new TPCCertificateRegistry(admin, registrar);

        // Whitelist the signer
        vm.prank(admin);
        registry.addSigner(signerPubkey);
    }

    // ─────────────────────────────────────────────────────
    // Constructor & Roles
    // ─────────────────────────────────────────────────────

    function test_constructor_sets_roles() public view {
        assertTrue(registry.hasRole(registry.DEFAULT_ADMIN_ROLE(), admin));
        assertTrue(registry.hasRole(registry.REGISTRAR_ROLE(), registrar));
        assertTrue(registry.hasRole(registry.REVOKER_ROLE(), admin));
    }

    // ─────────────────────────────────────────────────────
    // Signer Management
    // ─────────────────────────────────────────────────────

    function test_addSigner() public view {
        assertTrue(registry.allowedSigners(signerPubkey));
    }

    function test_removeSigner() public {
        vm.prank(admin);
        registry.removeSigner(signerPubkey);
        assertFalse(registry.allowedSigners(signerPubkey));
    }

    function test_addSigner_reverts_zero() public {
        vm.prank(admin);
        vm.expectRevert(TPCCertificateRegistry.ZeroHash.selector);
        registry.addSigner(bytes32(0));
    }

    function test_addSigner_reverts_unauthorized() public {
        vm.prank(attacker);
        vm.expectRevert();
        registry.addSigner(keccak256("rogue"));
    }

    // ─────────────────────────────────────────────────────
    // Registration
    // ─────────────────────────────────────────────────────

    function test_register_certificate() public {
        vm.prank(registrar);
        uint256 index = registry.register(
            contentHash1, signerPubkey, 0, certId1, bytes32(0)
        );

        assertEq(index, 0);
        assertEq(registry.certificateCount(), 1);
        assertTrue(registry.isRegistered(contentHash1));
        assertEq(registry.domainCount(0), 1);

        TPCCertificateRegistry.Certificate memory cert = registry.getCertificate(0);
        assertEq(cert.contentHash, contentHash1);
        assertEq(cert.signerPubkey, signerPubkey);
        assertEq(cert.domain, 0);
        assertEq(cert.registeredBy, registrar);
        assertEq(cert.certificateId, certId1);
        assertEq(uint256(cert.status), uint256(TPCCertificateRegistry.CertificateStatus.Valid));
    }

    function test_register_multiple_domains() public {
        vm.startPrank(registrar);
        registry.register(contentHash1, signerPubkey, 0, certId1, bytes32(0));
        registry.register(contentHash2, signerPubkey, 1, certId2, bytes32(0));
        vm.stopPrank();

        assertEq(registry.certificateCount(), 2);
        assertEq(registry.domainCount(0), 1); // thermal
        assertEq(registry.domainCount(1), 1); // euler3d
    }

    function test_register_reverts_duplicate() public {
        vm.prank(registrar);
        registry.register(contentHash1, signerPubkey, 0, certId1, bytes32(0));

        vm.prank(registrar);
        vm.expectRevert(
            abi.encodeWithSelector(TPCCertificateRegistry.AlreadyRegistered.selector, contentHash1)
        );
        registry.register(contentHash1, signerPubkey, 0, certId1, bytes32(0));
    }

    function test_register_reverts_invalid_domain() public {
        vm.prank(registrar);
        vm.expectRevert(abi.encodeWithSelector(TPCCertificateRegistry.InvalidDomain.selector, 4));
        registry.register(contentHash1, signerPubkey, 4, certId1, bytes32(0));
    }

    function test_register_reverts_unknown_signer() public {
        bytes32 unknownPubkey = keccak256("unknown");
        vm.prank(registrar);
        vm.expectRevert(
            abi.encodeWithSelector(TPCCertificateRegistry.SignerNotAllowed.selector, unknownPubkey)
        );
        registry.register(contentHash1, unknownPubkey, 0, certId1, bytes32(0));
    }

    function test_register_reverts_zero_hash() public {
        vm.prank(registrar);
        vm.expectRevert(TPCCertificateRegistry.ZeroHash.selector);
        registry.register(bytes32(0), signerPubkey, 0, certId1, bytes32(0));
    }

    function test_register_reverts_unauthorized() public {
        vm.prank(attacker);
        vm.expectRevert();
        registry.register(contentHash1, signerPubkey, 0, certId1, bytes32(0));
    }

    // ─────────────────────────────────────────────────────
    // Link Proof
    // ─────────────────────────────────────────────────────

    function test_linkProof() public {
        vm.prank(registrar);
        registry.register(contentHash1, signerPubkey, 0, certId1, bytes32(0));

        bytes32 proofTx = keccak256("proof_tx_hash");
        vm.prank(registrar);
        registry.linkProof(0, proofTx);

        TPCCertificateRegistry.Certificate memory cert = registry.getCertificate(0);
        assertEq(cert.proofTxHash, proofTx);
    }

    function test_linkProof_reverts_revoked() public {
        vm.prank(registrar);
        registry.register(contentHash1, signerPubkey, 0, certId1, bytes32(0));

        vm.prank(admin);
        registry.revoke(0, "compromised");

        vm.prank(registrar);
        vm.expectRevert(abi.encodeWithSelector(TPCCertificateRegistry.CertificateRevoked.selector, 0));
        registry.linkProof(0, keccak256("tx"));
    }

    // ─────────────────────────────────────────────────────
    // Revoke
    // ─────────────────────────────────────────────────────

    function test_revoke() public {
        vm.prank(registrar);
        registry.register(contentHash1, signerPubkey, 0, certId1, bytes32(0));

        vm.prank(admin);
        registry.revoke(0, "Key compromise");

        TPCCertificateRegistry.Certificate memory cert = registry.getCertificate(0);
        assertEq(uint256(cert.status), uint256(TPCCertificateRegistry.CertificateStatus.Revoked));
    }

    function test_revoke_reverts_nonexistent() public {
        vm.prank(admin);
        vm.expectRevert(abi.encodeWithSelector(TPCCertificateRegistry.CertificateNotFound.selector, 999));
        registry.revoke(999, "does not exist");
    }

    // ─────────────────────────────────────────────────────
    // Supersede
    // ─────────────────────────────────────────────────────

    function test_supersede() public {
        vm.startPrank(registrar);
        registry.register(contentHash1, signerPubkey, 0, certId1, bytes32(0));
        registry.register(contentHash2, signerPubkey, 0, certId2, bytes32(0));
        registry.supersede(0, 1);
        vm.stopPrank();

        TPCCertificateRegistry.Certificate memory cert = registry.getCertificate(0);
        assertEq(uint256(cert.status), uint256(TPCCertificateRegistry.CertificateStatus.Superseded));
    }

    // ─────────────────────────────────────────────────────
    // PQC Commitment (Dilithium2)
    // ─────────────────────────────────────────────────────

    function test_registerPQCCommitment() public {
        vm.prank(registrar);
        registry.register(contentHash1, signerPubkey, 0, certId1, bytes32(0));

        bytes32 pqcHash = keccak256("dilithium2_sig_and_pubkey");
        vm.prank(registrar);
        registry.registerPQCCommitment(0, pqcHash);

        assertEq(registry.pqcCommitments(0), pqcHash);
        assertTrue(registry.hasPQCCommitment(0));
    }

    function test_registerPQCCommitment_reverts_zero() public {
        vm.prank(registrar);
        registry.register(contentHash1, signerPubkey, 0, certId1, bytes32(0));

        vm.prank(registrar);
        vm.expectRevert(TPCCertificateRegistry.ZeroHash.selector);
        registry.registerPQCCommitment(0, bytes32(0));
    }

    function test_hasPQCCommitment_false_initially() public {
        vm.prank(registrar);
        registry.register(contentHash1, signerPubkey, 0, certId1, bytes32(0));

        assertFalse(registry.hasPQCCommitment(0));
    }

    // ─────────────────────────────────────────────────────
    // Verify Certificate
    // ─────────────────────────────────────────────────────

    function test_verifyCertificate_exists() public {
        vm.prank(registrar);
        registry.register(contentHash1, signerPubkey, 0, certId1, bytes32(0));

        (bool exists, TPCCertificateRegistry.CertificateStatus status, uint256 index) =
            registry.verifyCertificate(contentHash1);

        assertTrue(exists);
        assertEq(uint256(status), uint256(TPCCertificateRegistry.CertificateStatus.Valid));
        assertEq(index, 0);
    }

    function test_verifyCertificate_not_exists() public view {
        (bool exists, , ) = registry.verifyCertificate(keccak256("nonexistent"));
        assertFalse(exists);
    }

    function test_verifyCertificate_revoked() public {
        vm.prank(registrar);
        registry.register(contentHash1, signerPubkey, 0, certId1, bytes32(0));

        vm.prank(admin);
        registry.revoke(0, "test");

        (bool exists, TPCCertificateRegistry.CertificateStatus status, ) =
            registry.verifyCertificate(contentHash1);

        assertTrue(exists);
        assertEq(uint256(status), uint256(TPCCertificateRegistry.CertificateStatus.Revoked));
    }

    // ─────────────────────────────────────────────────────
    // Gas Benchmarks
    // ─────────────────────────────────────────────────────

    function test_gas_register() public {
        vm.prank(registrar);
        uint256 gasBefore = gasleft();
        registry.register(contentHash1, signerPubkey, 0, certId1, bytes32(0));
        uint256 gasUsed = gasBefore - gasleft();
        console.log("Gas: register() =", gasUsed);
    }

    function test_gas_verifyCertificate() public {
        vm.prank(registrar);
        registry.register(contentHash1, signerPubkey, 0, certId1, bytes32(0));

        uint256 gasBefore = gasleft();
        registry.verifyCertificate(contentHash1);
        uint256 gasUsed = gasBefore - gasleft();
        console.log("Gas: verifyCertificate() =", gasUsed);
    }

    function test_gas_registerPQCCommitment() public {
        vm.prank(registrar);
        registry.register(contentHash1, signerPubkey, 0, certId1, bytes32(0));

        vm.prank(registrar);
        uint256 gasBefore = gasleft();
        registry.registerPQCCommitment(0, keccak256("pqc"));
        uint256 gasUsed = gasBefore - gasleft();
        console.log("Gas: registerPQCCommitment() =", gasUsed);
    }
}
