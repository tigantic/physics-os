#!/usr/bin/env pwsh
# Build script for TCI Core Rust extension
#
# Prerequisites:
#   - Rust toolchain (rustup install stable)
#   - maturin (pip install maturin)
#   - OpenBLAS (for ndarray-linalg)
#
# On Windows, you may need to install OpenBLAS or use Intel MKL instead.
# Alternative: Use ndarray-linalg with "intel-mkl" feature.

$ErrorActionPreference = "Stop"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  TCI Core Build Script" -ForegroundColor Cyan  
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check for Rust
Write-Host "Checking Rust installation..." -ForegroundColor Yellow
try {
    $rustVersion = rustc --version
    Write-Host "  Found: $rustVersion" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Rust not found. Install from https://rustup.rs/" -ForegroundColor Red
    exit 1
}

# Check for maturin
Write-Host "Checking maturin installation..." -ForegroundColor Yellow
try {
    $maturinVersion = maturin --version
    Write-Host "  Found: $maturinVersion" -ForegroundColor Green
} catch {
    Write-Host "  maturin not found, installing..." -ForegroundColor Yellow
    pip install maturin
}

# Navigate to tci_core directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$tciDir = Join-Path $scriptDir "tci_core"

if (-not (Test-Path $tciDir)) {
    Write-Host "ERROR: tci_core directory not found at $tciDir" -ForegroundColor Red
    exit 1
}

Push-Location $tciDir

try {
    Write-Host ""
    Write-Host "Building TCI Core..." -ForegroundColor Yellow
    Write-Host "  Working directory: $tciDir" -ForegroundColor Gray
    Write-Host ""
    
    # Build in development mode (faster, includes debug symbols)
    Write-Host "Running: maturin develop --release" -ForegroundColor Cyan
    maturin develop --release
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "================================================" -ForegroundColor Green
        Write-Host "  BUILD SUCCESSFUL!" -ForegroundColor Green
        Write-Host "================================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "To test the installation:" -ForegroundColor Yellow
        Write-Host '  python -c "from tci_core import TCISampler; print(\"TCI Core loaded!\")"'
        Write-Host ""
    } else {
        Write-Host ""
        Write-Host "BUILD FAILED" -ForegroundColor Red
        Write-Host ""
        Write-Host "Common issues:" -ForegroundColor Yellow
        Write-Host "  1. OpenBLAS not installed - install via vcpkg or use Intel MKL" -ForegroundColor Gray
        Write-Host "  2. Missing Visual Studio Build Tools - install from Microsoft" -ForegroundColor Gray
        Write-Host "  3. Python environment not activated" -ForegroundColor Gray
        exit 1
    }
} finally {
    Pop-Location
}
