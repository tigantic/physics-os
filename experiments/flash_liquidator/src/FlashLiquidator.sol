// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {IPool} from "@aave/contracts/interfaces/IPool.sol";
import {IPoolAddressesProvider} from "@aave/contracts/interfaces/IPoolAddressesProvider.sol";
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {ISwapRouter} from "@uniswap/v3-periphery/contracts/interfaces/ISwapRouter.sol";

/**
 * @title FlashLiquidator
 * @notice Executes flash loan liquidations on AAVE V3
 * @dev Uses flash loans to liquidate undercollateralized positions with zero capital
 */
contract FlashLiquidator {
    IPool public immutable POOL;
    IPoolAddressesProvider public immutable ADDRESSES_PROVIDER;
    ISwapRouter public immutable SWAP_ROUTER;
    address public immutable OWNER;

    // AAVE V3 Mainnet addresses
    // Pool: 0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2
    // AddressesProvider: 0x2f39d218133AFaB8F2B819B1066c7E434Ad94E9e
    // Uniswap V3 Router: 0xE592427A0AEce92De3Edee1F18E0157C05861564

    constructor(
        address _addressesProvider,
        address _pool,
        address _swapRouter
    ) {
        ADDRESSES_PROVIDER = IPoolAddressesProvider(_addressesProvider);
        POOL = IPool(_pool);
        SWAP_ROUTER = ISwapRouter(_swapRouter);
        OWNER = msg.sender;
    }

    /**
     * @notice Execute a flash loan liquidation
     * @param debtAsset Token the borrower owes (e.g., USDC)
     * @param debtAmount How much they owe
     * @param collateralAsset Token they posted as collateral (e.g., WETH)
     * @param userToLiquidate The borrower's address
     * @param swapPoolFee Uniswap pool fee tier (500, 3000, 10000)
     */
    function executeLiquidation(
        address debtAsset,
        uint256 debtAmount,
        address collateralAsset,
        address userToLiquidate,
        uint24 swapPoolFee
    ) external {
        require(msg.sender == OWNER, "Not owner");

        // Encode params for the callback
        bytes memory params = abi.encode(
            collateralAsset,
            userToLiquidate,
            debtAsset,
            swapPoolFee
        );

        // Request flash loan of the debt asset
        POOL.flashLoanSimple(
            address(this), // receiver
            debtAsset,     // asset to borrow
            debtAmount,    // amount
            params,        // passed to callback
            0              // referral code
        );
    }

    /**
     * @notice AAVE callback after sending flash loan
     * @dev This is called by AAVE Pool after transferring flash loan funds
     */
    function executeOperation(
        address asset,
        uint256 amount,
        uint256 premium,
        address initiator,
        bytes calldata params
    ) external returns (bool) {
        require(msg.sender == address(POOL), "Caller not AAVE Pool");
        require(initiator == address(this), "Initiator not self");

        (
            address collateralAsset,
            address user,
            address debtAsset,
            uint24 swapFee
        ) = abi.decode(params, (address, address, address, uint24));

        // Approve AAVE to take the debt tokens for liquidation
        IERC20(asset).approve(address(POOL), amount);

        // Execute the liquidation - repays their debt, gives us collateral
        POOL.liquidationCall(
            collateralAsset,
            debtAsset,
            user,
            amount,  // debtToCover
            false    // receiveAToken = false (get underlying token)
        );

        // Get the collateral we received
        uint256 collateralBalance = IERC20(collateralAsset).balanceOf(address(this));
        require(collateralBalance > 0, "No collateral received");

        // Approve Uniswap to swap the collateral
        IERC20(collateralAsset).approve(address(SWAP_ROUTER), collateralBalance);

        // Calculate amount we need to repay (loan + fee)
        uint256 amountOwed = amount + premium;

        // Swap collateral back to debt asset
        ISwapRouter.ExactInputSingleParams memory swapParams = ISwapRouter.ExactInputSingleParams({
            tokenIn: collateralAsset,
            tokenOut: debtAsset,
            fee: swapFee,
            recipient: address(this),
            deadline: block.timestamp,
            amountIn: collateralBalance,
            amountOutMinimum: amountOwed, // Must cover loan + fee
            sqrtPriceLimitX96: 0
        });

        uint256 amountOut = SWAP_ROUTER.exactInputSingle(swapParams);

        // Approve AAVE to pull back the flash loan + fee
        IERC20(asset).approve(address(POOL), amountOwed);

        // Calculate and send profit to owner
        uint256 profit = amountOut - amountOwed;
        if (profit > 0) {
            IERC20(asset).transfer(OWNER, profit);
        }

        return true;
    }

    /**
     * @notice Emergency rescue function for stuck tokens
     */
    function rescue(address token) external {
        require(msg.sender == OWNER, "Not owner");
        uint256 balance = IERC20(token).balanceOf(address(this));
        if (balance > 0) {
            IERC20(token).transfer(OWNER, balance);
        }
    }

    /**
     * @notice Rescue native ETH
     */
    function rescueETH() external {
        require(msg.sender == OWNER, "Not owner");
        payable(OWNER).transfer(address(this).balance);
    }

    receive() external payable {}
}
