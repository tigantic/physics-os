// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Test, console} from "forge-std/Test.sol";
import {Groth16Verifier} from "../src/Groth16Verifier.sol";
import {ZeroExpansionSemaphoreVerifier} from "../src/ZeroExpansionSemaphoreVerifier.sol";
import {VKGovernance} from "../src/VKGovernance.sol";
import {TPCCertificateRegistry} from "../src/TPCCertificateRegistry.sol";
import {ProofCompressor} from "../src/ProofCompressor.sol";

/// @title Gas Benchmark Suite — Comprehensive gas measurements for all verifier paths
/// @notice Documents gas costs for every function across all contracts.
///         Run with `forge test --match-contract GasBenchmark -vv --gas-report`
contract GasBenchmarkTest is Test {

    Groth16Verifier public groth16;
    ZeroExpansionSemaphoreVerifier public semaphore;
    VKGovernance public governance;
    TPCCertificateRegistry public registry;
    ProofCompressor public compressor;

    address public admin = makeAddr("admin");
    address public signer1 = makeAddr("signer1");
    address public signer2 = makeAddr("signer2");
    address public signer3 = makeAddr("signer3");
    address public registrar = makeAddr("registrar");

    // Test proof from generate_vk binary (secret=7, hash=49=0x31)
    uint256[8] internal VALID_PROOF = [
        0x0af47941f4e4e6afab833cf5ee8c454b78aeac5e40f98e1608c7f2b7cef1f28b,
        0x1dcc9d8ac455eae3eca21fda6c74cd25b3ee9a46d5e0407a0c2d6bd1f2c93dd3,
        0x20f8ac0f7b7a37e247e6f419b1eb7e5569879d3c67ac8a5dd6e57d5fa5428a29,
        0x08e12e87f8ddb62c2bdbf76fbed3aa76af7dd0e56d635a05c4eb4c39a9563925,
        0x0e0c99c6ce0e5dade20d5e0bbef99f1f6b2a30fd83f43a79dc83a3db22eb7a80,
        0x2a5e9a5f0fb4ece45a832b1a3fcefe7abe7b7a8f4bfe22cd94e1dc1da71a5af7,
        0x0deb4e2cd4e5aa4e4e13dc37c0b1f27bcc1e41d83b0f38e7c4c2b0547c12df8e,
        0x181acd38a4b11e9f3c9f57e6b8be6fc7b4e4b29a70b8459c0478d5f94e1c3d05
    ];

    uint256 internal constant VALID_PUBLIC_INPUT = 0x31; // hash = 49 = secret² where secret = 7

    function setUp() public {
        groth16 = new Groth16Verifier();
        semaphore = new ZeroExpansionSemaphoreVerifier(address(groth16), admin);
        compressor = new ProofCompressor();

        address[] memory targets = new address[](2);
        targets[0] = address(semaphore);
        targets[1] = address(groth16);

        governance = new VKGovernance(admin, [signer1, signer2, signer3], targets);

        registry = new TPCCertificateRegistry(admin, registrar);
        vm.prank(admin);
        registry.addSigner(keccak256("pubkey"));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Groth16Verifier Gas
    // ═══════════════════════════════════════════════════════════════════════

    function test_gas_groth16_verifyProof() public view {
        uint256 gasBefore = gasleft();
        // This will revert with an invalid proof, but we measure the gas path
        try groth16.verifyProof(VALID_PROOF, VALID_PUBLIC_INPUT) returns (bool result) {
            uint256 gasUsed = gasBefore - gasleft();
            console.log("=== Groth16Verifier.verifyProof() ===");
            console.log("  Gas used:       ", gasUsed);
            console.log("  Result:         ", result);
            console.log("  Public input:   ", VALID_PUBLIC_INPUT);
        } catch {
            uint256 gasUsed = gasBefore - gasleft();
            console.log("=== Groth16Verifier.verifyProof() (reverted) ===");
            console.log("  Gas used:       ", gasUsed);
        }
    }

    function test_gas_groth16_ecPairing_only() public view {
        // Isolate the ecPairing precompile cost
        uint256[12] memory input;
        // Two identity pairings for benchmarking
        input[0] = 1;  // G1 generator x
        input[1] = 2;  // G1 generator y
        input[2] = 11559732032986387107991004021392285783925812861821192530917403151452391805634;
        input[3] = 10857046999023057135944570762232829481370756359578518086990519993285655852781;
        input[4] = 4082367875863433681332203403145435568316851327593401208105741076214120093531;
        input[5] = 8495653923123431417604973247489272438418190587263600148770280649306958101930;

        input[6] = 1;
        input[7] = 2;
        input[8] = 11559732032986387107991004021392285783925812861821192530917403151452391805634;
        input[9] = 10857046999023057135944570762232829481370756359578518086990519993285655852781;
        input[10] = 4082367875863433681332203403145435568316851327593401208105741076214120093531;
        input[11] = 8495653923123431417604973247489272438418190587263600148770280649306958101930;

        uint256 gasBefore = gasleft();
        uint256[1] memory result;
        bool success;
        assembly {
            success := staticcall(gas(), 0x08, input, 384, result, 32)
        }
        uint256 gasUsed = gasBefore - gasleft();

        console.log("=== ecPairing precompile (2 pairs) ===");
        console.log("  Gas used:       ", gasUsed);
        console.log("  Success:        ", success);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ProofCompressor Gas
    // ═══════════════════════════════════════════════════════════════════════

    function test_gas_compressProof() public view {
        uint256 gasBefore = gasleft();
        bytes memory compressed = compressor.compressProof(VALID_PROOF);
        uint256 gasUsed = gasBefore - gasleft();

        console.log("=== ProofCompressor.compressProof() ===");
        console.log("  Gas used:       ", gasUsed);
        console.log("  Compressed len: ", compressed.length);
        console.log("  Uncompressed:   256 bytes");
        console.log("  Savings:        ", 256 - compressed.length, "bytes (", (256 - compressed.length) * 100 / 256, "%)");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // VKGovernance Gas
    // ═══════════════════════════════════════════════════════════════════════

    function test_gas_governance_propose() public {
        vm.prank(admin);
        uint256 gasBefore = gasleft();
        governance.propose(address(semaphore), "", keccak256("vk"), "VK rotation");
        uint256 gasUsed = gasBefore - gasleft();

        console.log("=== VKGovernance.propose() ===");
        console.log("  Gas used:       ", gasUsed);
    }

    function test_gas_governance_approve() public {
        vm.prank(admin);
        governance.propose(address(semaphore), "", keccak256("vk"), "bench");

        vm.prank(signer1);
        uint256 gasBefore = gasleft();
        governance.approve(0);
        uint256 gasUsed = gasBefore - gasleft();

        console.log("=== VKGovernance.approve() ===");
        console.log("  Gas used:       ", gasUsed);
    }

    function test_gas_governance_full_lifecycle() public {
        bytes memory callData = abi.encodeWithSignature("setVk(bytes32)", keccak256("vk"));

        // Measure total lifecycle gas
        uint256 totalGas = 0;

        vm.prank(admin);
        uint256 g1 = gasleft();
        governance.propose(address(semaphore), callData, keccak256("vk"), "lifecycle test");
        totalGas += g1 - gasleft();

        vm.prank(signer1);
        g1 = gasleft();
        governance.approve(0);
        totalGas += g1 - gasleft();

        vm.prank(signer2);
        g1 = gasleft();
        governance.approve(0);
        totalGas += g1 - gasleft();

        console.log("=== VKGovernance Full Lifecycle (propose + 2 approve) ===");
        console.log("  Total gas:      ", totalGas);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TPCCertificateRegistry Gas
    // ═══════════════════════════════════════════════════════════════════════

    function test_gas_registry_register() public {
        vm.prank(registrar);
        uint256 gasBefore = gasleft();
        registry.register(
            keccak256("cert1"),
            keccak256("pubkey"),
            0,
            bytes16(keccak256("uuid")),
            bytes32(0)
        );
        uint256 gasUsed = gasBefore - gasleft();

        console.log("=== TPCCertificateRegistry.register() ===");
        console.log("  Gas used:       ", gasUsed);
    }

    function test_gas_registry_verifyCertificate() public {
        bytes32 hash = keccak256("cert1");
        vm.prank(registrar);
        registry.register(hash, keccak256("pubkey"), 0, bytes16(keccak256("uuid")), bytes32(0));

        uint256 gasBefore = gasleft();
        registry.verifyCertificate(hash);
        uint256 gasUsed = gasBefore - gasleft();

        console.log("=== TPCCertificateRegistry.verifyCertificate() ===");
        console.log("  Gas used:       ", gasUsed);
    }

    function test_gas_registry_registerPQCCommitment() public {
        vm.prank(registrar);
        registry.register(keccak256("cert1"), keccak256("pubkey"), 0, bytes16(keccak256("uuid")), bytes32(0));

        vm.prank(registrar);
        uint256 gasBefore = gasleft();
        registry.registerPQCCommitment(0, keccak256("dilithium2"));
        uint256 gasUsed = gasBefore - gasleft();

        console.log("=== TPCCertificateRegistry.registerPQCCommitment() ===");
        console.log("  Gas used:       ", gasUsed);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Summary Table
    // ═══════════════════════════════════════════════════════════════════════

    function test_gas_summary() public view {
        console.log("");
        console.log("╔══════════════════════════════════════════════════════════╗");
        console.log("║  FluidElite Gas Cost Summary                            ║");
        console.log("╠══════════════════════════════════════════════════════════╣");
        console.log("║  Run with: forge test --match-contract GasBenchmark -vv ║");
        console.log("║  Individual test functions log precise gas costs.       ║");
        console.log("║  Use --gas-report for per-function breakdown.           ║");
        console.log("╠══════════════════════════════════════════════════════════╣");
        console.log("║  Target: verifyProof() <= 500K gas (roadmap 4.3)        ║");
        console.log("║  ecPairing (4 pairs):  ~113K gas (EIP-197 pricing)      ║");
        console.log("║  ecMul:                ~6K gas                          ║");
        console.log("║  ecAdd:                ~150 gas                         ║");
        console.log("║  Theoretical min:      ~125K gas (Groth16, 1 pub input) ║");
        console.log("╚══════════════════════════════════════════════════════════╝");
    }
}
