// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.21;

import {Script, console} from "forge-std/Script.sol";
import {Groth16Verifier} from "../src/Groth16Verifier.sol";
import {WorldIDRouterZeroExpansion} from "../src/WorldcoinZeroExpansion.sol";

/// @title Deploy Zero-Expansion Worldcoin Stack
/// @notice Deploys Groth16Verifier and WorldIDRouterZeroExpansion
contract DeployZeroExpansion is Script {
    
    function run() public {
        uint256 deployerPrivateKey = vm.envUint("PRIVATE_KEY");
        
        vm.startBroadcast(deployerPrivateKey);
        
        // 1. Deploy the Groth16 Verifier (uses ecPairing precompile 0x08)
        Groth16Verifier groth16Verifier = new Groth16Verifier();
        console.log("Groth16Verifier deployed at:", address(groth16Verifier));
        
        // 2. Deploy the World ID Router (deploys Validator and WorldcoinVerifier internally)
        WorldIDRouterZeroExpansion router = new WorldIDRouterZeroExpansion(
            address(groth16Verifier)
        );
        console.log("WorldIDRouterZeroExpansion deployed at:", address(router));
        console.log("  - Validator:", address(router.validator()));
        console.log("  - Verifier:", address(router.verifier()));
        
        // 3. Set a test root for group 1
        bytes32 testRoot = keccak256("ZERO_EXPANSION_TEST_ROOT");
        router.updateRoot(1, uint256(testRoot));
        console.log("Test root set:", uint256(testRoot));
        
        vm.stopBroadcast();
        
        console.log("");
        console.log("=== DEPLOYMENT COMPLETE ===");
        console.log("Network: Sepolia");
        console.log("Tree Depth: 50 (1.1 quadrillion members)");
        console.log("");
        console.log("Verify on Etherscan:");
        console.log("  forge verify-contract", address(groth16Verifier), "src/Groth16Verifier.sol:Groth16Verifier --chain sepolia");
        console.log("  forge verify-contract", address(router), "src/WorldcoinZeroExpansion.sol:WorldIDRouterZeroExpansion --chain sepolia");
    }
}
