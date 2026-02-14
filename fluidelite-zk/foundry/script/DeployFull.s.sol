// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Script, console} from "forge-std/Script.sol";
import {Groth16Verifier} from "../src/Groth16Verifier.sol";
import {ZeroExpansionSemaphoreVerifier} from "../src/ZeroExpansionSemaphoreVerifier.sol";
import {VKGovernance} from "../src/VKGovernance.sol";
import {TPCCertificateRegistry} from "../src/TPCCertificateRegistry.sol";

/// @title Deploy Full FluidElite Verifier Stack
/// @notice Deploys all contracts for the Trustless Physics Platform:
///         1. Groth16Verifier (BN254 pairing, SimpleMembershipCircuit VK)
///         2. ZeroExpansionSemaphoreVerifier (Semaphore v3 with hardening)
///         3. VKGovernance (48h timelock, 2-of-3 multi-sig)
///         4. TPCCertificateRegistry (on-chain certificate store + PQC binding)
///
/// @dev Usage:
///   forge script script/DeployFull.s.sol:DeployFull \
///     --rpc-url $RPC_URL --broadcast --verify \
///     --etherscan-api-key $ETHERSCAN_KEY
///
///   Required env vars:
///     PRIVATE_KEY       — Deployer private key
///     SIGNER_1          — VK governance signer 1 address
///     SIGNER_2          — VK governance signer 2 address
///     SIGNER_3          — VK governance signer 3 address
///     CA_ADDRESS        — Certificate authority (REGISTRAR_ROLE) address
///     ED25519_PUBKEY    — Initial allowed Ed25519 signer pubkey (bytes32)
contract DeployFull is Script {

    function run() public {
        uint256 deployerPk = vm.envUint("PRIVATE_KEY");
        address deployer = vm.addr(deployerPk);

        // Multi-sig signers for VK governance
        address signer1 = vm.envAddress("SIGNER_1");
        address signer2 = vm.envAddress("SIGNER_2");
        address signer3 = vm.envAddress("SIGNER_3");

        // Certificate authority address
        address caAddress = vm.envAddress("CA_ADDRESS");

        // Initial Ed25519 signer public key
        bytes32 ed25519Pubkey = vm.envBytes32("ED25519_PUBKEY");

        console.log("╔══════════════════════════════════════════════════════╗");
        console.log("║  FluidElite Trustless Physics — Full Stack Deploy    ║");
        console.log("╚══════════════════════════════════════════════════════╝");
        console.log("");
        console.log("Deployer:          ", deployer);
        console.log("Signer 1:          ", signer1);
        console.log("Signer 2:          ", signer2);
        console.log("Signer 3:          ", signer3);
        console.log("CA Address:        ", caAddress);
        console.log("");

        vm.startBroadcast(deployerPk);

        // ── 1. Groth16 Verifier ────────────────────────────────────────
        Groth16Verifier groth16 = new Groth16Verifier();
        console.log("[1/4] Groth16Verifier:                ", address(groth16));

        // ── 2. Zero-Expansion Semaphore Verifier ───────────────────────
        ZeroExpansionSemaphoreVerifier semaphore = new ZeroExpansionSemaphoreVerifier(
            address(groth16),
            deployer
        );
        console.log("[2/4] ZeroExpansionSemaphoreVerifier:  ", address(semaphore));

        // ── 3. VK Governance (timelock + multi-sig) ────────────────────
        address[] memory targets = new address[](2);
        targets[0] = address(semaphore);
        targets[1] = address(groth16);

        VKGovernance governance = new VKGovernance(
            deployer,
            [signer1, signer2, signer3],
            targets
        );
        console.log("[3/4] VKGovernance:                    ", address(governance));

        // Transfer UPGRADER_ROLE on semaphore to the governance contract
        // so VK updates can only happen through the timelock.
        semaphore.grantRole(
            semaphore.UPGRADER_ROLE(),
            address(governance)
        );

        // ── 4. TPC Certificate Registry ────────────────────────────────
        TPCCertificateRegistry registry = new TPCCertificateRegistry(
            deployer,
            caAddress
        );

        // Register the initial Ed25519 signer
        registry.addSigner(ed25519Pubkey);

        console.log("[4/4] TPCCertificateRegistry:          ", address(registry));

        vm.stopBroadcast();

        // ── Summary ────────────────────────────────────────────────────
        console.log("");
        console.log("═══════════════════════════════════════════════════════");
        console.log("  DEPLOYMENT COMPLETE");
        console.log("═══════════════════════════════════════════════════════");
        console.log("  Groth16Verifier:             ", address(groth16));
        console.log("  SemaphoreVerifier:           ", address(semaphore));
        console.log("  VKGovernance:                ", address(governance));
        console.log("  TPCCertificateRegistry:      ", address(registry));
        console.log("");
        console.log("  VK governance timelock:       48 hours");
        console.log("  Required approvals:           2-of-3");
        console.log("  Tree depth range:             16-50");
        console.log("═══════════════════════════════════════════════════════");
    }
}

/// @title Deploy to Sepolia Testnet
/// @notice Convenience script with testnet-specific defaults
contract DeploySepolia is DeployFull {
    function setUp() public {
        // Sepolia-specific configuration can be set here
        // Env vars still required for addresses
    }
}

/// @title Deploy to Base Sepolia Testnet
/// @notice Convenience script with Base Sepolia defaults
contract DeployBaseSepolia is DeployFull {
    function setUp() public {
        // Base Sepolia-specific configuration
    }
}
