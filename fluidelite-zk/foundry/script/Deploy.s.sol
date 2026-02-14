// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Script, console} from "forge-std/Script.sol";
import {Groth16Verifier} from "../src/Groth16Verifier.sol";
import {ZeroExpansionSemaphoreVerifier} from "../src/ZeroExpansionSemaphoreVerifier.sol";

/// @title Deploy Zero-Expansion Verifier Stack
/// @notice Deploys Groth16Verifier and ZeroExpansionSemaphoreVerifier
contract DeployZeroExpansion is Script {

    function run() public {
        uint256 deployerPrivateKey = vm.envUint("PRIVATE_KEY");
        address deployer = vm.addr(deployerPrivateKey);

        vm.startBroadcast(deployerPrivateKey);

        // 1. Deploy the Groth16 Verifier (BN254 ecPairing precompile 0x08)
        Groth16Verifier groth16Verifier = new Groth16Verifier();
        console.log("Groth16Verifier deployed at:", address(groth16Verifier));

        // 2. Deploy the Semaphore Verifier with OZ hardening
        ZeroExpansionSemaphoreVerifier semaphore = new ZeroExpansionSemaphoreVerifier(
            address(groth16Verifier),
            deployer
        );
        console.log("ZeroExpansionSemaphoreVerifier deployed at:", address(semaphore));

        vm.stopBroadcast();

        console.log("");
        console.log("=== DEPLOYMENT COMPLETE ===");
        console.log("Admin:", deployer);
        console.log("Tree depth range: 16-50 (up to 2^50 members)");
    }
}
